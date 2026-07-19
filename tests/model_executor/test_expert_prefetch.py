# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for chunked prefetch and the measurements that drive its size."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import ExpertCacheOffloadConfig, OffloadConfig
from vllm.model_executor.layers.expert_prefetch import (
    EMPTY_SLOT,
    maybe_create_expert_cache,
)
from vllm.model_executor.offloader import create_offloader, get_offloader, set_offloader

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="expert cache requires CUDA"
)

NUM_EXPERTS = 16
TOP_K = 4
HIDDEN = 8
INTERMEDIATE = 12
NUM_LAYERS = 3


class FakeRoutedExperts(nn.Module):
    """The subset of `RoutedExperts` the cache touches.

    Expert `e` of layer `l` is filled with `100 * l + e`, so a slot serving the
    wrong expert is visible directly in the values.
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_name = f"model.layers.{layer_idx}.mlp.experts"
        self.global_num_experts = NUM_EXPERTS
        self.local_num_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.expert_map = None
        self.moe_config = SimpleNamespace(has_bias=False)
        self.quant_method = SimpleNamespace(is_monolithic=False)

        fill = 100 * layer_idx + torch.arange(NUM_EXPERTS, dtype=torch.float32)
        self.w13_weight = nn.Parameter(
            fill.view(-1, 1, 1).expand(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN).cuda(),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            fill.view(-1, 1, 1).expand(NUM_EXPERTS, HIDDEN, INTERMEDIATE).cuda(),
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
def cache_and_moes():
    set_offloader(
        create_offloader(
            OffloadConfig(
                offload_backend="expert_cache",
                expert_cache=ExpertCacheOffloadConfig(num_cache_slots=NUM_EXPERTS),
            )
        )
    )
    offloader = get_offloader()
    layers = offloader.wrap_modules(_fake_decoder_layer(i) for i in range(NUM_LAYERS))
    moes = [layer.mlp.experts.routed_experts for layer in layers]
    cache = maybe_create_expert_cache(moes)
    assert cache is not None
    offloader.post_init()
    yield cache, moes
    set_offloader(create_offloader(OffloadConfig()))


def _staged(cache, moe, expert_ids, num_chunks):
    """Run one prefetch to completion and return what landed in the buffer."""
    stream = torch.cuda.Stream()
    cache.prefetch(
        moe,
        torch.tensor(expert_ids, device="cuda"),
        stream,
        num_chunks=num_chunks,
    )
    buf = cache.inactive
    buf.wait_until_ready()
    torch.cuda.synchronize()
    return buf


@pytest.mark.parametrize("num_chunks", [1, 2, 4, 8])
def test_chunking_stages_exactly_what_one_chunk_would(cache_and_moes, num_chunks):
    """Splitting the transfer must not change which experts land where.

    The chunk boundaries are where a partially-applied prefetch would show up,
    so this is the load-bearing check on `_chunk_bounds`.
    """
    cache, moes = cache_and_moes
    ids = [7, 2, 11, 4, 9, 0, 15]

    buf = _staged(cache, moes[1], ids, num_chunks)
    assert buf.cached_expert_ids[: len(ids)].tolist() == ids
    assert (buf.cached_expert_ids[len(ids) :] == EMPTY_SLOT).all()
    # Layer 1's expert e is filled with 100 + e.
    for slot, expert in enumerate(ids):
        assert buf.params["w13_weight"][slot].unique().tolist() == [100 + expert]
        assert buf.params["w2_weight"][slot].unique().tolist() == [100 + expert]


def test_chunking_preserves_caller_ordering(cache_and_moes):
    """The tail is what gets dropped, so order must survive the split.

    `torch.unique` would sort these numerically; the prefetcher hands them over
    ranked by predictor score instead, and chunking must not undo that.
    """
    cache, moes = cache_and_moes
    ids = [15, 14, 13, 12, 3, 2, 1, 0]
    buf = _staged(cache, moes[0], ids, num_chunks=4)
    assert buf.cached_expert_ids[: len(ids)].tolist() == ids


def test_overflow_truncates_the_tail_and_is_reported(cache_and_moes):
    """An overflowing prefetch must tell the controller, or it will grow again."""
    cache, moes = cache_and_moes
    ids = list(range(NUM_EXPERTS + 5))

    assert cache.stats.truncated is False
    buf = _staged(cache, moes[0], ids, num_chunks=4)

    assert cache.stats.truncated is True
    assert cache.stats.staged == NUM_EXPERTS
    # The survivors are the head of what the caller passed, not the low ids.
    assert buf.cached_expert_ids.tolist() == ids[:NUM_EXPERTS]


def test_stats_count_only_prefetched_layers_and_reset(cache_and_moes):
    """A dropped buffer is all misses by construction and must not skew `acc`."""
    cache, moes = cache_and_moes
    topk_ids = torch.tensor([[0, 1, 2, 3]], device="cuda")

    # Staged for layer 0, but consumed by layer 1: the buffer is dropped.
    _staged(cache, moes[0], [0, 1, 2, 3], num_chunks=2)
    cache.resolve(moes[1], topk_ids)
    assert cache.drain_stats().layers == 0

    # Staged for the layer that actually consumes it.
    _staged(cache, moes[1], [0, 1, 2, 3], num_chunks=2)
    cache.flip()
    cache.resolve(moes[1], topk_ids)

    stats = cache.drain_stats()
    assert stats.layers == 1
    assert stats.needed == 4
    assert stats.hits == 4

    # Draining resets, so the next forward starts from zero.
    assert cache.drain_stats().needed == 0


def test_stats_see_a_partial_hit(cache_and_moes):
    cache, moes = cache_and_moes
    _staged(cache, moes[0], [0, 1], num_chunks=2)
    cache.flip()
    cache.resolve(moes[0], torch.tensor([[0, 1, 2, 3]], device="cuda"))

    stats = cache.drain_stats()
    assert (stats.hits, stats.needed) == (2, 4)


def test_calibration_measures_a_positive_copy_time(cache_and_moes):
    """The bubble constraint divides by this, so it must never be zero."""
    cache, _ = cache_and_moes
    t_e = cache.calibrate_copy_time(num_samples=4, warmup=1)
    assert t_e > 0.0
    # Calibration must not leave the buffer claiming to hold anything.
    assert (cache.ping.cached_expert_ids == EMPTY_SLOT).all()
    assert cache.ping.staged_for is None


def test_worker_exceptions_do_not_hang_the_consumer(cache_and_moes, caplog):
    """A failed prefetch must degrade to cache misses, not deadlock.

    `copies_issued` is what the consumer blocks on; if an exception skipped the
    `finally` that sets it, the next `resolve` would wait forever.
    """
    cache, moes = cache_and_moes
    stream = torch.cuda.Stream()
    broken = SimpleNamespace(layer_name=moes[0].layer_name)  # no weight params

    cache.prefetch(broken, torch.tensor([0, 1], device="cuda"), stream, num_chunks=2)
    cache.inactive.wait_until_ready()  # must return rather than hang
    assert "Expert prefetch failed" in caplog.text


def test_inflight_prefetch_is_skipped_not_queued(cache_and_moes):
    """Queueing behind a pacing worker would start the next prefetch too late."""
    cache, moes = cache_and_moes
    stream = torch.cuda.Stream()

    cache.prefetch(moes[0], torch.tensor([0, 1], device="cuda"), stream)
    inflight = cache._inflight
    cache.prefetch(moes[1], torch.tensor([2, 3], device="cuda"), stream)
    # The second call either found the first still running and returned, or the
    # first had already finished; either way it never queued a third future.
    assert cache._inflight is inflight or cache._inflight.done()
    cache.inactive.wait_until_ready()
