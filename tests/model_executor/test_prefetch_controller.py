# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the adaptive prefetch-size controller."""

import math

import pytest
import torch

from vllm.config.offload import ExpertCacheOffloadConfig
from vllm.model_executor.layers.expert_prefetch.expert_cache import (
    NATIVE_BITS,
    _chunk_bounds,
)
from vllm.model_executor.layers.expert_prefetch.expert_prefetcher import _rank_unique
from vllm.model_executor.layers.expert_prefetch.prefetch_controller import (
    PrefetchController,
)

TOP_K = 8
NUM_EXPERTS = 128


def _controller(**overrides) -> PrefetchController:
    cfg = ExpertCacheOffloadConfig(**overrides)
    return PrefetchController(
        top_k=TOP_K,
        num_experts=NUM_EXPERTS,
        num_slots=NUM_EXPERTS,
        cfg=cfg,
        # Copy times are now keyed by precision; with no resident quantized
        # widths there is only the native (bf16) candidate.
        t_e_ms={NATIVE_BITS: 0.1},
    )


def _feed(controller, num_tokens, hits, needed, staged, layers=48, **kwargs):
    """One interval's worth of observations, then a step."""
    controller.observe(
        num_tokens=num_tokens,
        hits=hits,
        needed=needed,
        staged=staged,
        layers=layers,
        truncated=kwargs.get("truncated", False),
    )
    if "t_comp" in kwargs:
        controller.observe_t_comp(num_tokens, kwargs["t_comp"])
    controller.step()


def test_poisson_target_matches_closed_form():
    """E_max and k_prefetch must agree with the Poisson derivation.

    K=8, acc=0.8937, C=99% gives lambda=0.8504, E_max~2.995, so 5 of the top-8
    predictions are correct at that confidence.
    """
    controller = _controller()
    acc = 0.8937
    lam = TOP_K * (1 - acc)
    e_max = lam + 2.326 * math.sqrt(lam)

    assert lam == pytest.approx(0.8504, abs=1e-4)
    assert e_max == pytest.approx(2.995, abs=1e-3)
    assert math.floor(TOP_K - e_max) == 5

    assert controller._poisson_target(acc) == pytest.approx(TOP_K - e_max, abs=1e-3)
    assert math.floor(controller._poisson_target(acc)) == 5


def test_poisson_target_is_absolute_not_a_nudge():
    """A perfect predictor must ask for all K, from wherever `p` happens to be.

    A multiplicative form (`p * target/hits`) returns `p` unchanged at acc=1 and
    so can never climb -- the reason this one is absolute.
    """
    controller = _controller()
    assert controller._poisson_target(1.0) == pytest.approx(TOP_K)
    # Hopeless accuracy asks for nothing; the bubble term decides from there.
    assert controller._poisson_target(0.1) == 0.0


def test_confidence_level_moves_the_target():
    """A stricter confidence bound must not stage *more* experts."""
    acc = 0.85
    targets = [
        _controller(prefetch_confidence=c)._poisson_target(acc)
        for c in (0.95, 0.99, 0.999)
    ]
    assert targets[0] > targets[1] > targets[2]


def test_bubble_term_converts_union_budget_to_per_token():
    """t_comp/t_e counts copies; p is per token, so the union ratio applies."""
    controller = _controller()
    # 2 ms of compute at 0.1 ms per expert = 20 experts fit under it.
    # At p=4 the batch staged 40 experts, so the union ratio is 10 experts per
    # unit of p, and the budget is worth p=2.
    assert controller._bubble_target(
        2.0, p_cur=4.0, staged_avg=40.0, t_e_ms=0.1
    ) == pytest.approx(2.0)
    # Batch of 1: union size equals p, so the budget passes through unscaled.
    assert controller._bubble_target(
        2.0, p_cur=4.0, staged_avg=4.0, t_e_ms=0.1
    ) == pytest.approx(20.0)


def test_bubble_target_capped_by_cache_size():
    """However much time there is, the cache cannot hold more than it holds."""
    controller = PrefetchController(
        top_k=TOP_K,
        num_experts=NUM_EXPERTS,
        num_slots=16,
        cfg=ExpertCacheOffloadConfig(),
        t_e_ms={NATIVE_BITS: 0.1},
    )
    # 100 ms would buy 1000 experts, but only 16 slots exist.
    assert (
        controller._bubble_target(100.0, p_cur=1.0, staged_avg=1.0, t_e_ms=0.1) == 16.0
    )


def test_buckets_separate_decode_from_prefill():
    """A prefill must not overwrite what decode learned, or vice versa."""
    controller = _controller()
    assert PrefetchController.bucket(1) == 1
    assert PrefetchController.bucket(4096) == 13
    assert PrefetchController.bucket(3) == PrefetchController.bucket(2)

    # Drive decode down (bad accuracy, no compute to hide behind).
    for _ in range(4):
        _feed(controller, num_tokens=1, hits=48 * 2, needed=48 * 8, staged=48 * 8)
    decode = controller.topk_for(1)

    # Prefill sees perfect accuracy and should stay high.
    for _ in range(4):
        _feed(
            controller, num_tokens=4096, hits=48 * 100, needed=48 * 100, staged=48 * 100
        )

    assert controller.topk_for(1) == decode
    assert controller.topk_for(4096) > controller.topk_for(1)


def test_pinned_topk_disables_adaptation():
    controller = _controller(prefetch_topk=3)
    assert controller.topk_for(1) == 3
    for _ in range(8):
        _feed(controller, num_tokens=1, hits=1, needed=100, staged=100)
    assert controller.topk_for(1) == 3
    assert controller.topk_for(4096) == 3


def test_truncation_freezes_upward_movement():
    """Growing a prefetch that already overflows makes the hit rate worse."""
    controller = _controller()
    start = controller.topk_for(64)
    # Perfect accuracy plus generous compute would otherwise push p up, but the
    # prefetch did not fit the cache.
    for _ in range(6):
        _feed(
            controller,
            num_tokens=64,
            hits=48 * 60,
            needed=48 * 60,
            staged=48 * 60,
            t_comp=50.0,
            truncated=True,
        )
    assert controller.topk_for(64) <= start


def test_step_ignores_intervals_with_too_few_samples():
    controller = _controller()
    before = controller.topk_for(1)
    _feed(controller, num_tokens=1, hits=1, needed=8, staged=8, layers=2)
    assert controller.topk_for(1) == before


def test_p_never_leaves_its_bounds():
    controller = _controller(prefetch_min_topk=2)
    # Pathologically bad accuracy for a long time must not go below the floor.
    for _ in range(50):
        _feed(controller, num_tokens=1, hits=48, needed=48 * 8, staged=48 * 8)
    assert 2 <= controller.topk_for(1) <= TOP_K

    # Perfect accuracy and unlimited compute must not exceed the model's top_k.
    for _ in range(50):
        _feed(
            controller,
            num_tokens=1,
            hits=48 * 8,
            needed=48 * 8,
            staged=48 * 8,
            t_comp=1000.0,
        )
    assert controller.topk_for(1) == TOP_K


def test_t_e_is_clamped_around_calibration():
    """A contended interval must not be able to erase the bubble budget."""
    controller = _controller()
    for _ in range(100):
        controller.observe_t_e({NATIVE_BITS: 1000.0})
    assert controller._t_e_ms[NATIVE_BITS] <= 0.1 * 4.0
    for _ in range(100):
        controller.observe_t_e({NATIVE_BITS: 1e-9})
    assert controller._t_e_ms[NATIVE_BITS] >= 0.1 * 0.5


def test_sampling_and_step_land_on_different_passes():
    """Timing events need a whole interval to complete before being read."""
    controller = _controller(prefetch_adapt_interval=4)
    sampled, stepped = [], []
    for i in range(8):
        if controller.due_to_sample():
            sampled.append(i)
        if controller.on_forward_end():
            stepped.append(i)
    assert sampled == [0, 4]
    assert stepped == [3, 7]
    assert not set(sampled) & set(stepped)


# ---------------------------------------------------------------------------
# Prefetch-set construction
# ---------------------------------------------------------------------------


def test_rank_unique_orders_by_score_not_by_expert_id():
    """Truncation drops the tail, so the tail must be the least likely experts.

    `torch.unique` sorts numerically, which would make an overflowing prefetch
    keep the lowest expert ids -- a systematically biased subset.
    """
    logits = torch.tensor([[0.0, 9.0, 1.0, 8.0]])
    assert _rank_unique(logits, top_k=4).tolist() == [1, 3, 2, 0]


def test_rank_unique_dedupes_across_tokens_keeping_best_score():
    logits = torch.tensor(
        [
            [0.0, 5.0, 1.0, 0.5],
            [9.0, 0.1, 0.2, 0.3],
        ]
    )
    ids = _rank_unique(logits, top_k=2)
    # Expert 0 scores 9.0 in the second token, expert 1 scores 5.0 in the first.
    assert ids.tolist()[:2] == [0, 1]
    assert len(set(ids.tolist())) == len(ids)


def test_rank_unique_is_a_subset_of_the_union_of_per_token_topk():
    """`_rank_unique` is fixed-shape: it returns exactly `top_k` experts, the
    best-scoring of the union of every token's top-`top_k` (no longer the whole
    union, whose length was data-dependent and cost a host sync)."""
    torch.manual_seed(0)
    logits = torch.randn(16, NUM_EXPERTS)
    union = set(torch.topk(logits, 3, dim=-1).indices.reshape(-1).tolist())
    ranked = _rank_unique(logits, top_k=3)
    assert ranked.numel() == 3
    assert len(set(ranked.tolist())) == 3
    assert set(ranked.tolist()) <= union
    # At batch size 1 the candidates are one token's top-k, which are already
    # distinct and descending -- so the result is exactly that top-k.
    single = _rank_unique(logits[:1], top_k=3)
    assert single.tolist() == torch.topk(logits[0], 3).indices.tolist()


@pytest.mark.parametrize(
    "n,num_chunks,expected",
    [
        (8, 4, [(0, 2), (2, 4), (4, 6), (6, 8)]),
        (10, 4, [(0, 3), (3, 6), (6, 9), (9, 10)]),
        (3, 4, [(0, 1), (1, 2), (2, 3)]),
        (5, 1, [(0, 5)]),
        (0, 4, []),
    ],
)
def test_chunk_bounds_cover_every_expert_exactly_once(n, num_chunks, expected):
    bounds = _chunk_bounds(n, num_chunks)
    assert bounds == expected
    covered = [i for lo, hi in bounds for i in range(lo, hi)]
    assert covered == list(range(n))
    assert all(lo < hi for lo, hi in bounds)
