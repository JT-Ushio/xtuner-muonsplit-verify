"""Tests for saved-tensor offload lifecycle management."""

from unittest import mock

import pytest
import torch

from xtuner.v1.utils.activation_offload import OffloadItem, OffloadManager, SingletonMeta, SwapTensor


@pytest.fixture
def manager():
    # OffloadManager is a process-level singleton; rebuild it per test so no
    # runtime state leaks across cases.
    SingletonMeta._instances.pop(OffloadManager, None)
    yield OffloadManager()
    SingletonMeta._instances.pop(OffloadManager, None)


class TestOffloadManager:
    def test_clear_step_releases_runtime_state_and_preserves_pin_cache(self, manager):
        manager.items["text_0_0"] = OffloadItem()
        manager.may_npu_tensors["text_0_1"] = OffloadItem()
        manager.items["other_0_0"] = OffloadItem()
        manager.pin_memory_cache["text_0_0"] = mock.sentinel.pinned_buffer

        manager.clear_step(group="text")

        assert "text_0_0" not in manager.items
        assert "text_0_1" not in manager.may_npu_tensors
        assert "other_0_0" in manager.items
        assert manager.pin_memory_cache["text_0_0"] is mock.sentinel.pinned_buffer

    def test_clear_step_is_noop_without_offload(self, manager):
        manager.clear_step()  # no entries; must not raise
        assert not manager.items


class TestSwapTensor:
    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
    @pytest.mark.parametrize("sliced", [False, True])
    def test_d2h_preserves_values_when_allocation_stream_reuses_storage(self, sliced: bool) -> None:
        allocation_stream = torch.cuda.Stream()
        copy_stream = torch.cuda.Stream()
        elements = 4 * 1024 * 1024
        with torch.cuda.stream(allocation_stream):
            source = torch.ones(elements, device="cuda", dtype=torch.float32)
            ready = torch.cuda.Event()
            ready.record()
        torch.cuda.current_stream().wait_event(ready)
        swap = SwapTensor(source[: elements // 2] if sliced else source, "d2h_lifetime")
        with torch.cuda.stream(copy_stream):
            # Keep D2H pending while the source allocation stream requests another block.
            torch.cuda._sleep(100_000_000)
        swap.launch_d2h(copy_stream)
        swap.wait_d2h_finished(copy_stream, flag=True)
        with torch.cuda.stream(allocation_stream):
            replacement = torch.empty(elements, device="cuda", dtype=torch.float32)
            replacement.fill_(2)
        torch.cuda.synchronize()
        torch.testing.assert_close(swap.tensor_cpu, torch.ones_like(swap.tensor_cpu), rtol=0, atol=0)
