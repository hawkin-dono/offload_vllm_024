# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the expert cache against a real MoE layer.

The contract: an offloaded, cached MoE layer must be *bitwise identical* to the
stock fully-GPU-resident one. Caching changes where weights live and how they
are indexed, never what is computed -- so anything short of exact equality is a
bug, and there is no tolerance to tune.
"""

import pytest
import torch

from vllm.config import (
    ExpertCacheOffloadConfig,
    OffloadConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.expert_prefetch import maybe_create_expert_cache
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.offloader import create_offloader, get_offloader, set_offloader
from vllm.v1.worker.workspace import init_workspace_manager

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="expert cache requires CUDA"
)

NUM_EXPERTS = 16
TOP_K = 4
HIDDEN = 128
INTERMEDIATE = 256
NUM_LAYERS = 3


def _build_layers(vllm_config):
    torch.manual_seed(0)
    layers = []
    for i in range(NUM_LAYERS):
        layer = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            params_dtype=torch.bfloat16,
            prefix=f"model.layers.{i}.mlp.experts",
            renormalize=False,
            scoring_func="softmax",
        ).cuda()
        experts = layer.routed_experts
        # Distinct weights per expert *and* per layer, so serving the wrong slot
        # or the wrong layer's slot shows up as a numeric difference.
        experts.w13_weight.data.normal_(mean=0.02 * (i + 1), std=0.02)
        experts.w2_weight.data.normal_(mean=0.02 * (i + 1), std=0.02)
        layer._quant_method.process_weights_after_loading(experts)
        layers.append(layer)
    return layers


def _run_moe(layer, x, router_logits):
    quant_method = layer._quant_method
    topk_weights, topk_ids = layer.router.select_experts(
        hidden_states=x,
        router_logits=router_logits,
        topk_indices_dtype=quant_method.topk_indices_dtype,
    )
    out = quant_method.forward_native(
        layer=layer.routed_experts,
        x=x.clone(),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )
    return out, torch.unique(topk_ids)


def _routed_experts_of(layer, x, router_logits):
    _, topk_ids = layer.router.select_experts(
        hidden_states=x,
        router_logits=router_logits,
        topk_indices_dtype=layer._quant_method.topk_indices_dtype,
    )
    return torch.unique(topk_ids)


@pytest.fixture
def moe_layers(dist_init):
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        init_workspace_manager(torch.accelerator.current_device_index())
        yield _build_layers(vllm_config), vllm_config
    set_offloader(create_offloader(OffloadConfig()))


def _enable_cache(layers, num_cache_slots):
    set_offloader(
        create_offloader(
            OffloadConfig(
                offload_backend="expert_cache",
                expert_cache=ExpertCacheOffloadConfig(num_cache_slots=num_cache_slots),
            )
        )
    )
    offloader = get_offloader()
    offloader.wrap_modules(iter(layers))
    cache = maybe_create_expert_cache([layer.routed_experts for layer in layers])
    assert cache is not None
    offloader.post_init()
    return cache


@pytest.mark.parametrize(
    "num_tokens,num_cache_slots",
    [
        # The cache must hold a whole layer (a partial cache is rejected at
        # allocation: prefill routes to every expert). Fetch-on-demand and
        # eviction still get exercised through stale buffers, whose maps are
        # dropped so every routed expert misses.
        (6, NUM_EXPERTS),
        (2, NUM_EXPERTS),
    ],
)
@pytest.mark.parametrize("prefetch", [False, True])
def test_cached_moe_matches_stock_moe(
    moe_layers, num_tokens, num_cache_slots, prefetch
):
    layers, vllm_config = moe_layers
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        torch.manual_seed(1)
        x = 0.1 * torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
        router_logits = [
            torch.randn(num_tokens, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
            for _ in range(NUM_LAYERS)
        ]

        # Stock path: every expert resident on the GPU.
        baseline = [
            _run_moe(layers[i], x, router_logits[i])[0].clone()
            for i in range(NUM_LAYERS)
        ]
        assert layers[0].routed_experts.w13_weight.device.type == "cuda"

        # Cached path: expert weights now live in pinned CPU memory.
        cache = _enable_cache(layers, num_cache_slots)
        assert layers[0].routed_experts.w13_weight.device.type == "cpu"
        cache.reset()

        stream = torch.cuda.Stream()
        moes = [layer.routed_experts for layer in layers]
        if prefetch:
            # Perfect prediction: stage exactly what each layer will route to.
            cache.prefetch(
                moes[0], _routed_experts_of(layers[0], x, router_logits[0]), stream
            )
            cache.flip()

        for i, layer in enumerate(layers):
            if prefetch and i + 1 < NUM_LAYERS:
                cache.prefetch(
                    moes[i + 1],
                    _routed_experts_of(layers[i + 1], x, router_logits[i + 1]),
                    stream,
                )
            out, _ = _run_moe(layer, x, router_logits[i])
            assert torch.equal(out, baseline[i]), (
                f"layer {i} differs from the stock MoE "
                f"(slots={num_cache_slots}, prefetch={prefetch})"
            )


def test_cache_survives_repeated_forward_passes(moe_layers):
    """Buffers are recycled across passes; `reset` must leave no stale state."""
    layers, vllm_config = moe_layers
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        torch.manual_seed(2)
        x = 0.1 * torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda")
        router_logits = [
            torch.randn(4, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
            for _ in range(NUM_LAYERS)
        ]
        baseline = [
            _run_moe(layers[i], x, router_logits[i])[0].clone()
            for i in range(NUM_LAYERS)
        ]

        cache = _enable_cache(layers, NUM_EXPERTS)
        for step in range(3):
            cache.reset()
            for i, layer in enumerate(layers):
                out, _ = _run_moe(layer, x, router_logits[i])
                assert torch.equal(out, baseline[i]), f"pass {step}, layer {i} differs"
