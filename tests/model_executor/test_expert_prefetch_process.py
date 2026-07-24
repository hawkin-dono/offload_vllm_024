# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the process-mode expert prefetch worker.

Each test spawns a real worker process (a few seconds each: the child imports
torch and opens its own CUDA context), so assertions are grouped by scenario
rather than split per invariant.
"""

import math
import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import ExpertCacheOffloadConfig, OffloadConfig
from vllm.model_executor.layers.expert_prefetch import maybe_create_expert_cache
from vllm.model_executor.offloader import create_offloader, get_offloader, set_offloader

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="expert cache requires CUDA"
)

NUM_EXPERTS = 16
HIDDEN = 128
INTERMEDIATE = 256
NUM_LAYERS = 3


class UnquantizedFusedMoEMethod:
    """Named exactly like the real method: the quant-store guard checks the
    class *name*."""

    is_monolithic = False


class FakeRoutedExperts(nn.Module):
    """The subset of `RoutedExperts` the cache touches.

    Expert `e` of layer `l` is filled with `100 * l + e`, so a slot serving the
    wrong expert (or the wrong layer) is visible directly in the values.
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_name = f"model.layers.{layer_idx}.mlp.experts"
        self.global_num_experts = NUM_EXPERTS
        self.local_num_experts = NUM_EXPERTS
        self.top_k = 4
        self.expert_map = None
        self.moe_config = SimpleNamespace(has_bias=False)
        self.quant_method = UnquantizedFusedMoEMethod()

        fill = 100 * layer_idx + torch.arange(NUM_EXPERTS, dtype=torch.bfloat16)
        self.w13_weight = nn.Parameter(
            fill.view(-1, 1, 1)
            .expand(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN)
            .contiguous()
            .cuda(),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            fill.view(-1, 1, 1)
            .expand(NUM_EXPERTS, HIDDEN, INTERMEDIATE)
            .contiguous()
            .cuda(),
            requires_grad=False,
        )


def _fake_decoder_layer(layer_idx: int) -> nn.Module:
    experts = nn.Module()
    experts.routed_experts = FakeRoutedExperts(layer_idx)
    mlp = nn.Module()
    mlp.experts = experts
    layer = nn.Module()
    layer.mlp = mlp
    return layer


@pytest.fixture
def process_cache(request):
    quant_bits = getattr(request, "param", [])
    set_offloader(
        create_offloader(
            OffloadConfig(
                offload_backend="expert_cache",
                expert_cache=ExpertCacheOffloadConfig(
                    num_cache_slots=NUM_EXPERTS,
                    expert_quant_bits=quant_bits,
                    prefetch_worker_mode="process",
                ),
            )
        )
    )
    offloader = get_offloader()
    layers = offloader.wrap_modules(_fake_decoder_layer(i) for i in range(NUM_LAYERS))
    moes = [layer.mlp.experts.routed_experts for layer in layers]
    cache = maybe_create_expert_cache(moes)
    assert cache is not None
    offloader.post_init()
    assert cache._process_client is not None
    yield cache, moes
    cache.shutdown()
    if offloader.arena is not None:
        offloader.arena.cleanup()
    set_offloader(create_offloader(OffloadConfig()))


def _staged(cache, moe, expert_ids, num_chunks=2, num_bits=16, ref=None):
    """Run one prefetch to completion and return the buffer it landed in."""
    stream = torch.cuda.Stream()
    cache.prefetch(
        moe,
        torch.tensor(expert_ids, device="cuda"),
        stream,
        num_chunks=num_chunks,
        reference_ids=ref,
        num_bits=num_bits,
    )
    buf = cache.inactive
    buf.wait_until_ready()
    torch.cuda.synchronize()
    return buf


def test_process_prefetch_stages_correct_values(process_cache):
    """bf16 staging through the worker process: values, maps, staged_for and
    reference ids all land as the parent expects, across ping/pong flips."""
    cache, moes = process_cache
    ids = [7, 2, 11, 4, 9]
    ref = torch.tensor([7, 2, 5, 3], device="cuda")

    buf = _staged(cache, moes[1], ids, ref=ref)
    assert buf.staged_for == moes[1].layer_name
    assert buf.slot_to_expert[: len(ids)].tolist() == ids
    for name in ("w13_weight", "w2_weight"):
        got = [float(buf.params[name][s, 0, 0]) for s in range(len(ids))]
        assert got == [100 + e for e in ids]
    assert buf.reference_ids.tolist() == [7, 2, 5, 3]

    # Alternating buffers, changing layers and chunkings.
    for round_ in range(6):
        moe = moes[round_ % NUM_LAYERS]
        want = [(round_ + i) % NUM_EXPERTS for i in range(4)]
        buf = _staged(cache, moe, want, num_chunks=1 + round_ % 4)
        got = [float(buf.params["w13_weight"][s, 0, 0]) for s in range(4)]
        assert got == [100 * (round_ % NUM_LAYERS) + e for e in want]
        cache.flip()


@pytest.mark.parametrize("process_cache", [[8]], indirect=True)
def test_process_quant_prefetch_dequantizes(process_cache):
    """A quantized prefetch runs through the worker's own dequant ring."""
    cache, moes = process_cache
    ids = [7, 2, 11, 4, 9]
    buf = _staged(cache, moes[1], ids, num_bits=8)
    got = [float(buf.params["w13_weight"][s, 0, 0]) for s in range(len(ids))]
    assert all(math.isclose(g, 100 + e, rel_tol=0.05) for g, e in zip(got, ids)), got


def test_process_reset_and_telemetry(process_cache):
    """`reset` waits out an in-flight prefetch; copy times flow back."""
    cache, moes = process_cache
    ids = [1, 2, 3, 4]
    stream = torch.cuda.Stream()

    cache.prefetch(moes[0], torch.tensor(ids, device="cuda"), stream, num_chunks=2)
    cache.reset()
    assert cache.active_name == "ping"
    for buf in (cache.ping, cache.pong):
        assert buf.staged_for is None
        assert (buf.slot_to_expert < 0).all()

    cache.begin_forward(sampling=True)
    _staged(cache, moes[0], ids)
    stats = cache.drain_stats()
    cache.begin_forward(sampling=False)
    assert stats.t_e_ms.get(16, 0.0) > 0.0


def test_process_worker_death_degrades_to_misses(process_cache):
    """A killed worker must never hang the model: waits release, staged state
    invalidates, later prefetches become no-ops."""
    cache, moes = process_cache
    ids = [1, 2, 3, 4]
    stream = torch.cuda.Stream()

    os.kill(cache._process_client._proc.pid, 9)
    time.sleep(0.3)

    start = time.monotonic()
    cache.prefetch(moes[0], torch.tensor(ids, device="cuda"), stream)
    buf = cache.inactive
    buf.wait_until_ready()
    assert time.monotonic() - start < 20.0

    assert not cache._process_client.usable
    assert buf.staged_for is None
    assert (buf.expert_to_slot < 0).all()

    # Dead client: prefetch is a no-op that leaves the flags set.
    cache.prefetch(moes[1], torch.tensor(ids, device="cuda"), stream)
    assert cache.inactive.maps_ready.is_set()
    assert cache.inactive.copies_issued.is_set()


def test_process_shutdown_leaves_no_shm(process_cache):
    """Backing files are unlinked as soon as the worker attaches, so nothing
    survives this cache's lifetime -- checked against files whose name carries
    this process's pid, since other runs may share /dev/shm."""
    cache, moes = process_cache
    _staged(cache, moes[0], [1, 2, 3])
    cache.shutdown()
    mine = f"vllm-expert-arena-{os.getpid()}-"
    leftovers = [f for f in os.listdir("/dev/shm") if f.startswith(mine)]
    assert not leftovers
