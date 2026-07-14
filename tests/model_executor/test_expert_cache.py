# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the ping-pong expert cache."""

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
TOP_K = 4
HIDDEN = 8
INTERMEDIATE = 12
NUM_LAYERS = 3
NUM_SLOTS = 8


class FakeRoutedExperts(nn.Module):
    """The subset of `RoutedExperts` the cache actually touches.

    Expert `e` of layer `l` is filled with the constant `100 * l + e`, so a
    slot serving the wrong expert -- or the right expert from the wrong layer --
    is directly visible in the values.
    """

    def __init__(self, layer_idx: int, has_bias: bool = False):
        super().__init__()
        self.layer_name = f"model.layers.{layer_idx}.mlp.experts"
        self.global_num_experts = NUM_EXPERTS
        self.local_num_experts = NUM_EXPERTS
        self.top_k = TOP_K
        # Checked by ExpertCache._check_supported.
        self.expert_map = None
        self.moe_config = SimpleNamespace(has_bias=has_bias)
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
def cache_and_moes(request):
    num_slots = getattr(request, "param", NUM_SLOTS)
    set_offloader(
        create_offloader(
            OffloadConfig(
                offload_backend="expert_cache",
                expert_cache=ExpertCacheOffloadConfig(num_cache_slots=num_slots),
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


def _assert_serves_correct_weights(cache, moe, topk_ids):
    """The cache must be indistinguishable from reading the real weights."""
    buf, slot_ids = cache.resolve(moe, topk_ids)
    for name in ("w13_weight", "w2_weight"):
        got = buf.params[name][slot_ids]
        want = getattr(moe, name).cuda()[topk_ids]
        assert torch.equal(got, want), f"{moe.layer_name}: {name} mismatch"


def test_cache_is_allocated_on_gpu_from_cpu_weights(cache_and_moes):
    cache, moes = cache_and_moes
    assert cache.allocated
    assert cache.num_slots == NUM_SLOTS
    assert cache.param_names == ("w13_weight", "w2_weight")
    # Source weights offloaded and pinned; the cache mirrors them on the GPU.
    assert moes[0].w13_weight.device.type == "cpu"
    assert moes[0].w13_weight.is_pinned()
    assert cache.ping.params["w13_weight"].device.type == "cuda"
    assert cache.pong.params["w13_weight"].device.type == "cuda"


def test_fetch_on_demand_without_prediction(cache_and_moes):
    """With no prefetch at all, every expert is a miss. Slow, but correct --
    this is the baseline the predictor is measured against."""
    cache, moes = cache_and_moes
    cache.reset()
    torch.manual_seed(0)
    for moe in moes:
        topk_ids = torch.randint(0, NUM_EXPERTS, (2, TOP_K), device="cuda")
        _assert_serves_correct_weights(cache, moe, topk_ids)
        cache.flip()


def test_prefetch_then_consume(cache_and_moes):
    """The real access pattern: while running layer i, stage layer i+1."""
    cache, moes = cache_and_moes
    cache.reset()
    torch.manual_seed(0)
    stream = torch.cuda.Stream()
    topk = [
        torch.randint(0, NUM_EXPERTS, (2, TOP_K), device="cuda")
        for _ in range(NUM_LAYERS)
    ]

    cache.prefetch(moes[0], torch.unique(topk[0]), stream)
    cache.flip()
    for i, moe in enumerate(moes):
        _assert_serves_correct_weights(cache, moe, topk[i])
        if i + 1 < NUM_LAYERS:
            cache.prefetch(moes[i + 1], torch.unique(topk[i + 1]), stream)
        cache.flip()


def test_buffer_staged_for_another_layer_is_not_reused(cache_and_moes):
    """Buffers are recycled across layers, so an id table alone cannot be
    trusted: slot 3 may hold "expert 7" -- of the *previous* layer.

    Without the `staged_for` guard this silently serves the wrong layer's
    weights: no crash, just wrong numbers.
    """
    cache, moes = cache_and_moes
    cache.reset()
    stream = torch.cuda.Stream()
    expert_ids = torch.tensor([0, 1, 2, 3], device="cuda")

    # Stage these experts for layer 1, then read them as layer 2.
    cache.prefetch(moes[1], expert_ids, stream)
    cache.flip()
    _assert_serves_correct_weights(cache, moes[2], expert_ids.view(1, -1))


def test_stale_guard_is_load_bearing(cache_and_moes):
    """Negative control for the test above: defeating the guard must actually
    produce wrong weights, otherwise that test proves nothing."""
    cache, moes = cache_and_moes
    cache.reset()
    stream = torch.cuda.Stream()
    expert_ids = torch.tensor([0, 1, 2, 3], device="cuda")

    cache.prefetch(moes[1], expert_ids, stream)
    cache.flip()
    buf = cache.active
    buf.wait_until_ready()
    buf.staged_for = moes[2].layer_name  # lie about who it was staged for

    served, slot_ids = cache.resolve(moes[2], expert_ids.view(1, -1))
    got = served.params["w13_weight"][slot_ids]
    want = moes[2].w13_weight.cuda()[expert_ids.view(1, -1)]
    assert not torch.equal(got, want), "guard is vacuous: stale read went unnoticed"


def test_cache_too_small_raises(cache_and_moes):
    cache, moes = cache_and_moes
    cache.reset()
    # More distinct experts in one layer than the cache has slots.
    topk_ids = torch.arange(NUM_EXPERTS, device="cuda").view(1, -1)
    with pytest.raises(RuntimeError, match="Expert cache too small"):
        cache.resolve(moes[0], topk_ids)


def _allocate_with(num_slots=NUM_SLOTS, mutate=None):
    """Build + allocate a cache, optionally mutating the MoE layers first."""
    set_offloader(
        create_offloader(
            OffloadConfig(
                offload_backend="expert_cache",
                expert_cache=ExpertCacheOffloadConfig(num_cache_slots=num_slots),
            )
        )
    )
    try:
        offloader = get_offloader()
        layers = offloader.wrap_modules(_fake_decoder_layer(i) for i in range(2))
        moes = [layer.mlp.experts.routed_experts for layer in layers]
        if mutate is not None:
            for moe in moes:
                mutate(moe)
        maybe_create_expert_cache(moes)
        offloader.post_init()
    finally:
        set_offloader(create_offloader(OffloadConfig()))


def test_slots_below_top_k_rejected_at_allocation():
    """A cache that cannot even hold one token's experts is never workable."""
    with pytest.raises(ValueError, match="smaller than the layer's top_k"):
        _allocate_with(num_slots=TOP_K - 1)


def _set_expert_map(moe):
    moe.expert_map = torch.arange(NUM_EXPERTS, device="cuda")


def _set_has_bias(moe):
    moe.moe_config = SimpleNamespace(has_bias=True)


def _set_monolithic(moe):
    moe.quant_method = SimpleNamespace(is_monolithic=True)


@pytest.mark.parametrize(
    "mutate,match",
    [
        # Each of these would otherwise produce wrong numbers, not an error:
        # EP ids are global while the cache indexes local weights...
        (_set_expert_map, "expert parallelism"),
        # ...bias is indexed by expert id, not slot...
        (_set_has_bias, "expert bias"),
        # ...and a monolithic kernel never exposes topk_ids to remap.
        (_set_monolithic, "modular MoE path"),
    ],
)
def test_unsupported_configurations_rejected_at_load(mutate, match):
    with pytest.raises(NotImplementedError, match=match):
        _allocate_with(mutate=mutate)
