# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn import functional as F

from xtuner.v1.model.moe.glm53.nope_dsa_mla import KPoolIndexer


class TestIndexerPrecision:
    def test_fp32_projection_norm_and_kernel_contract(self):
        torch.manual_seed(77)
        module = KPoolIndexer(
            hidden_size=31,
            q_lora_rank=17,
            index_head_dim=16,
            index_n_heads=4,
            index_topk=16,
            index_kpool=4,
            index_kpool_always_select_tail=True,
            indexer_backend="torch",
            alignment=1,
            compute_dtype="float32",
            projection_block_size=64,
        ).to(torch.bfloat16)
        hidden = torch.randn(1, 131, 31, dtype=torch.bfloat16)
        query = torch.randn(1, 131, 17, dtype=torch.bfloat16)
        captured = {}

        def capture(q, k, gates, weights, *args, **kwargs):
            captured.update(q=q, k=k, gates=gates, weights=weights)
            return torch.zeros(131, 1, 16, dtype=torch.int32)

        module._topk_indices_fn = capture
        module(hidden, query, None)
        assert captured["q"].dtype == captured["k"].dtype == torch.bfloat16
        assert captured["weights"].dtype == captured["gates"].dtype == torch.float32
        expected_key = F.layer_norm(
            F.linear(hidden.double(), module.wk.weight.double()),
            (16,),
            module.k_norm.weight.double(),
            module.k_norm.bias.double(),
            1e-6,
        ).to(torch.bfloat16)
        torch.testing.assert_close(captured["k"], expected_key, atol=0, rtol=0)
        expected_weights = F.linear(hidden.double(), module.weights_proj.weight.double())
        torch.testing.assert_close(captured["weights"].double(), expected_weights, atol=2e-6, rtol=2e-6)
        assert set(module.state_dict()) == {
            "wq_b.weight",
            "wk.weight",
            "k_norm.weight",
            "k_norm.bias",
            "weights_proj.weight",
            "index_kpool_compress_ape",
            "index_kpool_compress_gate",
        }
