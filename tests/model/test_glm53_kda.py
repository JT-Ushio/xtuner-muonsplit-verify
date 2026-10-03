"""GLM-5.3-Flash 的 Kimi Delta Attention，见 doc/xtuner_glm5p3flash_design.md F3。

需要 GPU：FLA 的 Triton kernel 没有 CPU 后端；SP 用例需要 2 卡。

TestKDAGate
    test_fused_kda_gate_matches_naive_reference           融合 gate 与朴素实现一致
TestKDAKernelDispatch
    test_grad_enabled_preserves_all_parameter_gradients  37/64/65 token 在 train/eval 下均有完整梯度
    test_no_grad_matches_hf_at_kernel_boundary           无梯度时 37/64/65 token 的输出与 HF 一致
    test_activation_checkpointing_matches_plain_training 重计算与普通训练的输出、输入及全部参数梯度一致
TestKDAModuleParity
    test_kda_module_matches_hf_single_document            单文档下与 HF 实现一致
    test_kda_module_packed_multi_document_matches_concatenated_single_document_forwards
                                                          packed 多文档等价于逐文档前向
TestKDASequenceParallel
    test_forward_for_sp_matches_non_sp                    2 卡 SP 与非 SP 结果一致
"""

from copy import deepcopy

import pytest
import torch
from torch.testing._internal.common_distributed import DistributedTestBase

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.utils.checkpointing import apply_activation_checkpointing
from xtuner.v1.module.attention.kda import KDAConfig, fused_kda_gate
from xtuner.v1.utils.test_utils import init_data_mesh


def _hf_glm53_kda(**overrides):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention

    kwargs = dict(
        hidden_size=64,
        linear_num_heads=4,
        linear_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        num_hidden_layers=1,
        layer_types=["linear_attention"],
    )
    kwargs.update(overrides)
    config = Glm5NextTextConfig(**kwargs)
    module = Glm5NextTextLinearAttention(config, layer_idx=0)
    # Standalone HF modules leave these torch.empty parameters to the pretrained-model initializer.
    with torch.no_grad():
        module.forget_gate.A_log.zero_()
        module.forget_gate.dt_bias.uniform_(-4.0, -1.0)
    return module, config


def _build_xtuner_kda(hidden_size=64, num_heads=4, head_dim=16, conv_kernel_size=4):
    cfg = KDAConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=conv_kernel_size,
        use_full_rank_gate=False,
        gate_lower_bound=-5.0,
        rms_norm_eps=1e-5,
    )
    return cfg.build(hidden_size=hidden_size, layer_idx=0)


def _copy_hf_weights_into_xtuner(hf_module, xtuner_module) -> None:
    """Bridge HF's fused ``conv1d``/``forget_gate`` layout into XTuner's published-checkpoint
    layout (separate q/k/v_conv1d, flat A_log/dt_bias), matching the mapping documented in
    doc/xtuner_glm5p3flash_design.md section 3.1."""
    with torch.no_grad():
        xtuner_module.q_proj.weight.copy_(hf_module.q_proj.weight)
        xtuner_module.k_proj.weight.copy_(hf_module.k_proj.weight)
        xtuner_module.v_proj.weight.copy_(hf_module.v_proj.weight)

        qkv_dim = hf_module.qkv_dim
        q_w, k_w, v_w = hf_module.conv1d.weight.split(qkv_dim, dim=0)
        xtuner_module.q_conv1d.weight.copy_(q_w)
        xtuner_module.k_conv1d.weight.copy_(k_w)
        xtuner_module.v_conv1d.weight.copy_(v_w)

        xtuner_module.f_a_proj.weight.copy_(hf_module.forget_gate.f_a_proj.weight)
        xtuner_module.f_b_proj.weight.copy_(hf_module.forget_gate.f_b_proj.weight)
        xtuner_module.dt_bias.copy_(hf_module.forget_gate.dt_bias)
        xtuner_module.A_log.copy_(hf_module.forget_gate.A_log)

        xtuner_module.b_proj.weight.copy_(hf_module.b_proj.weight)
        xtuner_module.g_a_proj.weight.copy_(hf_module.g_a_proj.weight)
        xtuner_module.g_b_proj.weight.copy_(hf_module.g_b_proj.weight)
        xtuner_module.o_norm.weight.copy_(hf_module.o_norm.weight)
        xtuner_module.o_proj.weight.copy_(hf_module.o_proj.weight)


class TestKDAGate:
    @pytest.mark.gpu
    def test_fused_kda_gate_matches_naive_reference(self):
        # 融合 gate kernel 必须与朴素公式一致。
        from fla.ops.kda.gate import naive_kda_lowerbound_gate

        torch.manual_seed(0)
        num_heads, head_dim = 4, 16
        g_raw = torch.randn(1, 8, num_heads, head_dim, device="cuda")
        a_log = torch.randn(num_heads, device="cuda")
        dt_bias = torch.randn(num_heads * head_dim, device="cuda")

        got = fused_kda_gate(g_raw, a_log, dt_bias=dt_bias, lower_bound=-5.0)
        expected = naive_kda_lowerbound_gate(g_raw, a_log, dt_bias=dt_bias, lower_bound=-5.0)
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


class TestKDAKernelDispatch:
    """验证 kernel 长度阈值两侧的完整梯度、推理精度与重计算一致性。"""

    @pytest.mark.gpu
    @pytest.mark.parametrize("seq_len", [37, 64, 65])
    @pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
    def test_grad_enabled_preserves_all_parameter_gradients(self, seq_len: int, training: bool) -> None:
        # eval 不等于 no_grad：阈值两侧都必须保留 q/k/v、卷积、forget gate、beta 等全部参数的梯度。
        torch.manual_seed(0)
        module = _build_xtuner_kda().cuda().train(training)
        hidden_states = torch.randn(1, seq_len, 64, device="cuda", requires_grad=True)
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, seq_len, dtype=torch.long),), device="cuda")

        output = module(hidden_states, seq_ctx)["projected_output"]
        assert torch.isfinite(output).all()
        (output * torch.randn_like(output)).sum().backward()

        missing = [name for name, param in module.named_parameters() if param.grad is None]
        assert not missing, f"Missing parameter gradients: {missing}"
        for name, param in module.named_parameters():
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient: {name}"
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()

    @pytest.mark.gpu
    @pytest.mark.parametrize("seq_len", [37, 64, 65])
    @pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
    def test_no_grad_matches_hf_at_kernel_boundary(self, seq_len: int, training: bool) -> None:
        # 通过完整模块输出核验推理行为，不依赖内部 kernel 的具体分派或调用次数。
        torch.manual_seed(0)
        hf_module, _ = _hf_glm53_kda()
        hf_module = hf_module.cuda().train(training)
        module = _build_xtuner_kda().cuda().train(training)
        _copy_hf_weights_into_xtuner(hf_module, module)
        hidden_states = torch.randn(1, seq_len, 64, device="cuda")
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, seq_len, dtype=torch.long),), device="cuda")

        with torch.no_grad():
            expected = hf_module(hidden_states)
            output = module(hidden_states, seq_ctx)["projected_output"]

        assert not output.requires_grad
        assert torch.isfinite(expected).all()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)

    @pytest.mark.gpu
    @pytest.mark.parametrize("seq_len", [37, 64, 65])
    def test_activation_checkpointing_matches_plain_training(self, seq_len: int) -> None:
        # 使用实际 reentrant 包装器：首次 no_grad 前向与有梯度重算必须采用一致的计算。
        torch.manual_seed(0)
        plain_module = _build_xtuner_kda().cuda().train()
        replay_module = deepcopy(plain_module)
        checkpointed_module = apply_activation_checkpointing(replay_module)
        hidden_states = torch.randn(1, seq_len, 64, device="cuda", requires_grad=True)
        checkpointed_hidden = hidden_states.detach().clone().requires_grad_()
        seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, seq_len, dtype=torch.long),), device="cuda")

        expected = plain_module(hidden_states, seq_ctx)["projected_output"]
        output = checkpointed_module(checkpointed_hidden, seq_ctx)["projected_output"]
        torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
        # 非线性目标使上游梯度依赖首次前向值，避免常量输出梯度掩盖错误重计算。
        expected.square().sum().backward()
        output.square().sum().backward()

        assert hidden_states.grad is not None
        assert checkpointed_hidden.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        assert torch.isfinite(checkpointed_hidden.grad).all()
        torch.testing.assert_close(checkpointed_hidden.grad, hidden_states.grad, atol=1e-6, rtol=1e-5)
        for name, param in plain_module.named_parameters():
            replay_param = replay_module.get_parameter(name)
            assert param.grad is not None, name
            assert replay_param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            assert torch.isfinite(replay_param.grad).all(), name
            torch.testing.assert_close(replay_param.grad, param.grad, atol=1e-6, rtol=1e-5, msg=name)


class TestKDAModuleParity:
    @pytest.mark.gpu
    def test_kda_module_matches_hf_single_document(self):
        # 单文档下整个 KDA 模块的输出与 HF 实现一致。
        torch.manual_seed(0)
        hf_module, _ = _hf_glm53_kda()
        hf_module = hf_module.cuda()
        xtuner_module = _build_xtuner_kda().cuda()
        _copy_hf_weights_into_xtuner(hf_module, xtuner_module)

        hidden_states = torch.randn(1, 37, 64, device="cuda")
        with torch.no_grad():
            hf_out = hf_module(hidden_states)  # returns a single tensor, not a tuple

            seq_ctx = SequenceContext.from_input_ids((torch.zeros(1, 37, dtype=torch.long),), device="cuda")
            xtuner_out = xtuner_module(hidden_states, seq_ctx)["projected_output"]

        torch.testing.assert_close(xtuner_out, hf_out, atol=2e-2, rtol=2e-2)

    @pytest.mark.gpu
    def test_kda_module_packed_multi_document_matches_concatenated_single_document_forwards(self):
        """Packed multi-document forward must equal the per-document forwards concatenated
        (design doc F3 test item 3: no cross-document leakage through the conv/recurrent state)."""
        # packed 多文档必须等价于逐文档单独前向再拼接，文档间不能串状态。
        torch.manual_seed(0)
        xtuner_module = _build_xtuner_kda().cuda()

        doc1 = torch.randn(1, 20, 64, device="cuda")
        doc2 = torch.randn(1, 33, 64, device="cuda")

        with torch.no_grad():
            seq_ctx1 = SequenceContext.from_input_ids((torch.zeros(1, 20, dtype=torch.long),), device="cuda")
            out1 = xtuner_module(doc1, seq_ctx1)["projected_output"]
            seq_ctx2 = SequenceContext.from_input_ids((torch.zeros(1, 33, dtype=torch.long),), device="cuda")
            out2 = xtuner_module(doc2, seq_ctx2)["projected_output"]

            packed_hidden = torch.cat([doc1, doc2], dim=1)
            packed_seq_ctx = SequenceContext.from_input_ids(
                (torch.zeros(1, 20, dtype=torch.long), torch.zeros(1, 33, dtype=torch.long)), device="cuda"
            )
            packed_out = xtuner_module(packed_hidden, packed_seq_ctx)["projected_output"]

        expected = torch.cat([out1, out2], dim=1)
        torch.testing.assert_close(packed_out, expected, atol=1e-4, rtol=1e-4)


class TestKDASequenceParallel(DistributedTestBase):
    @pytest.mark.gpu
    def test_forward_for_sp_matches_non_sp(self, device="cuda"):
        # 2 卡 Ulysses SP 的输出与非 SP 一致（head 切分不改变数学）。
        self.create_pg(device)
        torch.manual_seed(0)

        module = _build_xtuner_kda(hidden_size=64, num_heads=4, head_dim=16).to(device)
        for p in module.parameters():
            torch.distributed.broadcast(p.data, src=0)

        seq_len_per_rank = 40
        sp_size = self.world_size
        torch.manual_seed(1234)
        full_hidden = torch.randn(1, seq_len_per_rank * sp_size, 64, device=device)
        torch.distributed.broadcast(full_hidden, src=0)

        seq_ctx_non_sp = SequenceContext.from_input_ids(
            (torch.zeros(1, seq_len_per_rank * sp_size, dtype=torch.long),), device=device
        )
        with torch.no_grad():
            non_sp_out = module(full_hidden, seq_ctx_non_sp)["projected_output"]

        data_mesh = init_data_mesh(device, sp_size)
        sp_mesh = data_mesh["sp"]
        rank = sp_mesh.get_local_rank()
        local_hidden = full_hidden[:, rank * seq_len_per_rank : (rank + 1) * seq_len_per_rank, :].contiguous()

        seq_ctx_sp = SequenceContext.from_input_ids(
            (torch.zeros(1, seq_len_per_rank * sp_size, dtype=torch.long),), device=device
        )
        seq_ctx_sp = seq_ctx_sp.split(sequence_parallel_mesh=sp_mesh)

        with torch.no_grad():
            sp_out = module(local_hidden, seq_ctx_sp)["projected_output"]

        expected_local = non_sp_out[:, rank * seq_len_per_rank : (rank + 1) * seq_len_per_rank, :]
        torch.testing.assert_close(sp_out, expected_local, atol=2e-2, rtol=2e-2)

    @property
    def world_size(self) -> int:
        return 2
