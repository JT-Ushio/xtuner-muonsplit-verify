from typing import Protocol

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_gather_tensor,
    all_gather_tensor_autograd,
    reduce_scatter_tensor,
    wait_tensor,
)
from torch.distributed.device_mesh import DeviceMesh


def gather_for_sequence_parallel(
    input: torch.Tensor,
    dim: int,
    sp_mesh: DeviceMesh | None,
    reduce_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Gather sequence shards while reducing gradients back to their owners.

    Args:
        input (torch.Tensor): Local sequence shard.
        dim (int): Sequence dimension to gather.
        sp_mesh (DeviceMesh | None): Sequence-parallel mesh.
        reduce_dtype (torch.dtype | None): Optional backward reduction dtype. Forward
            communication retains the input dtype, and the reduced gradient is cast back
            to it. ``None`` preserves the native autograd collective.

    Returns:
        torch.Tensor: Gathered sequence in the input dtype.
    """
    if sp_mesh is None or sp_mesh.size() == 1:
        return input
    if reduce_dtype is not None:
        return _GatherWithReduceDtype.apply(input, dim, sp_mesh, reduce_dtype)
    return all_gather_tensor_autograd(input, gather_dim=dim, group=sp_mesh)


def split_for_sequence_parallel(input, dim: int, sp_mesh):
    """Splits the input tensor along a given dimension for sequence parallel.

    Args:
        input: The input tensor to be split.
        dim: The dimension along which the tensor should be split.
        sp_group: The sequence parallel process group.

    Returns:
        The split tensor corresponding to the current rank's chunk.
    """
    sp_group = sp_mesh.get_group()
    sp_size = sp_mesh.size()
    if sp_size == 1:
        return input

    rank = dist.get_rank(sp_group)
    dim_size = input.size(dim)
    assert dim_size % sp_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of sp size ({sp_size}), cannot split tensor evenly"
    )

    tensor_list = torch.split(input, dim_size // sp_size, dim=dim)
    output = tensor_list[rank].contiguous()

    return output


class _GatherContext(Protocol):
    dim: int
    sp_mesh: DeviceMesh
    reduce_dtype: torch.dtype
    input_dtype: torch.dtype


class _GatherWithReduceDtype(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _GatherContext,
        input: torch.Tensor,
        dim: int,
        sp_mesh: DeviceMesh,
        reduce_dtype: torch.dtype,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.sp_mesh = sp_mesh
        ctx.reduce_dtype = reduce_dtype
        ctx.input_dtype = input.dtype
        return wait_tensor(all_gather_tensor(input.contiguous(), gather_dim=dim, group=sp_mesh))

    @staticmethod
    def backward(ctx: _GatherContext, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        # Sum each rank's contribution before rounding back to the activation dtype.
        grad_input = reduce_scatter_tensor(
            grad_output.to(ctx.reduce_dtype).contiguous(), "sum", scatter_dim=ctx.dim, group=ctx.sp_mesh
        )
        return wait_tensor(grad_input).to(ctx.input_dtype), None, None, None
