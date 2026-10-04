"""SP gather 的归约精度：前向保持 BF16，反向先 FP32 求和再还原 dtype。

TestSequenceParallelPrecision
    test_fp32_reduce_scatter_matches_sum_oracle  四卡、非零维度和单卡退化路径
    test_compiled_gather_preserves_forward_and_gradient  编译前后均对齐独立 FP32 SUM
"""

import itertools
import unittest

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.ops.comm import gather_for_sequence_parallel


@pytest.mark.gpu
@unittest.skipIf(torch.cuda.device_count() < 4, "Requires four CUDA devices and NCCL")
class TestSequenceParallelPrecision(DeterministicDDPTestCase):
    """用真实 collective 和独立求和 oracle 检验低精度梯度归约。"""

    @property
    def world_size(self) -> int:
        return 4

    def test_fp32_reduce_scatter_matches_sum_oracle(self) -> None:
        """保留前向值和梯度 dtype，且不在 SUM 之外增加任何 SP 缩放。"""
        torch.cuda.set_device(self.rank)
        pg = self.create_pg("cuda")
        try:
            mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("sp",))
            for dtype in (torch.bfloat16, torch.float32):
                for dim in (0, 1, -1):
                    self._check_gather(mesh, dtype, dim, compiled=False)

            # 每个 rank 都属于一个 size=1 的 SP 子组；此时不能调用跨卡 SUM。
            singleton = init_device_mesh("cuda", (4, 1), mesh_dim_names=("dp", "single_sp"))["single_sp"]
            for sp_mesh in (None, singleton):
                local = torch.ones(2, 3, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                gathered = gather_for_sequence_parallel(local, 1, sp_mesh, reduce_dtype=torch.float32)
                assert gathered is local
                gathered.sum().backward()
                torch.testing.assert_close(local.grad, torch.ones_like(local), rtol=0, atol=0)
        finally:
            dist.destroy_process_group(pg)

    def test_compiled_gather_preserves_forward_and_gradient(self) -> None:
        """真实 Inductor fullgraph 前反向覆盖 dim=0 和非零维度。"""
        torch.cuda.set_device(self.rank)
        pg = self.create_pg("cuda")
        try:
            mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("sp",))
            for dim in (0, 1):
                self._check_gather(mesh, torch.bfloat16, dim, compiled=True)
        finally:
            dist.destroy_process_group(pg)

    def _check_gather(self, mesh: DeviceMesh, dtype: torch.dtype, dim: int, compiled: bool) -> None:
        # 穷举秩间大数抵消的排列；FP32 oracle 的精确和均为 2，避免依赖 NCCL 求和顺序。
        rank_contributions = torch.tensor(
            list(itertools.permutations((256.0, 1.0, -256.0, 1.0))), device="cuda", dtype=torch.float32
        )
        local = torch.arange(48, device="cuda", dtype=dtype).reshape(2, 24) + self.rank * 48
        coefficients = rank_contributions[:, self.rank].expand(8, 24).to(dtype)
        expected_forward = torch.arange(192, device="cuda", dtype=dtype).reshape(8, 24)
        expected_gradient = rank_contributions.sum(dim=1).expand(2, 24).to(dtype)
        if dim != 0:
            local = local.t().contiguous()
            coefficients = coefficients.t().contiguous()
            expected_forward = expected_forward.t().contiguous()
            expected_gradient = expected_gradient.t().contiguous()
        local.requires_grad_()

        def gather(input: torch.Tensor) -> torch.Tensor:
            return gather_for_sequence_parallel(input, dim, mesh, reduce_dtype=torch.float32)

        run = torch.compile(gather, fullgraph=True) if compiled else gather
        output = run(local)
        native = gather_for_sequence_parallel(local.detach(), dim, mesh)
        assert output.dtype == dtype
        torch.testing.assert_close(output, expected_forward, rtol=0, atol=0)
        torch.testing.assert_close(output, native, rtol=0, atol=0)
        output.backward(coefficients)
        assert local.grad is not None and local.grad.dtype == dtype
        torch.testing.assert_close(local.grad, expected_gradient, rtol=0, atol=0)
