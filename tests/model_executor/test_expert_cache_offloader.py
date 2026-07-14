# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the expert_cache offload backend."""

import pytest
import torch
import torch.nn as nn

from vllm.config import ExpertCacheOffloadConfig, OffloadConfig
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.model_executor.offloader import ExpertCacheOffloader, create_offloader
from vllm.model_executor.utils import replace_parameter

EXPERT_PARAMS = ("w13_weight", "w2_weight")


def _make_layer(device: torch.device) -> nn.Module:
    """A decoder layer shaped like the real MoE nesting:
    mlp.experts.routed_experts.{w13_weight,w2_weight,w13_weight_scale}
    """
    routed_experts = nn.Module()
    routed_experts.w13_weight = nn.Parameter(
        torch.randn(4, 8, 16, device=device), requires_grad=False
    )
    routed_experts.w2_weight = nn.Parameter(
        torch.randn(4, 16, 8, device=device), requires_grad=False
    )
    # A scale is *not* in the default offload set: it must stay resident.
    routed_experts.w13_weight_scale = nn.Parameter(
        torch.randn(4, device=device), requires_grad=False
    )

    experts = nn.Module()
    experts.routed_experts = routed_experts
    mlp = nn.Module()
    mlp.experts = experts

    layer = nn.Module()
    layer.mlp = mlp
    layer.self_attn = nn.Linear(16, 16).to(device)
    return layer


def _offloader() -> ExpertCacheOffloader:
    offloader = create_offloader(OffloadConfig(offload_backend="expert_cache"))
    assert isinstance(offloader, ExpertCacheOffloader)
    return offloader


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.layers.0.mlp.experts.routed_experts.w13_weight", True),
        ("model.layers.0.mlp.experts.routed_experts.w2_weight", True),
        # Segment matching must not let a scale ride along on a weight name.
        ("model.layers.0.mlp.experts.routed_experts.w13_weight_scale", False),
        # Pre-0.24 layout (no `routed_experts` level) must not match.
        ("model.layers.0.mlp.experts.w13_weight", False),
        ("model.layers.0.self_attn.qkv_proj.weight", False),
    ],
)
def test_param_segment_matching(name: str, expected: bool):
    assert _offloader()._should_offload(name) is expected


def test_auto_never_selects_expert_cache():
    # expert_cache only offloads routed experts, so it must be opt-in.
    offloader = create_offloader(OffloadConfig())
    assert not isinstance(offloader, ExpertCacheOffloader)


def test_empty_param_set_is_rejected():
    with pytest.raises(ValueError, match="expert_cache_params is empty"):
        create_offloader(
            OffloadConfig(
                offload_backend="expert_cache",
                expert_cache=ExpertCacheOffloadConfig(expert_cache_params=set()),
            )
        )


def test_matching_nothing_is_rejected():
    # Fail loudly rather than silently running with no offloading at all.
    offloader = create_offloader(
        OffloadConfig(
            offload_backend="expert_cache",
            expert_cache=ExpertCacheOffloadConfig(
                expert_cache_params={"experts.w13_weight"}  # pre-0.24 path
            ),
        )
    )
    layer = nn.Module()
    layer.lin = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="matched no parameters"):
        offloader.wrap_modules(layer for _ in range(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_only_expert_weights_are_offloaded():
    device = torch.device("cuda")
    offloader = _offloader()
    layers = offloader.wrap_modules(_make_layer(device) for _ in range(2))

    for layer in layers:
        routed = layer.mlp.experts.routed_experts
        for name in EXPERT_PARAMS:
            assert getattr(routed, name).device.type == "cpu"
        # Everything else stays on the accelerator.
        assert routed.w13_weight_scale.device.type == "cuda"
        assert layer.self_attn.weight.device.type == "cuda"

    assert offloader.offloaded_bytes > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_post_init_repins_after_weight_processing():
    """`post_init` must restore pinning that `process_weights_after_loading`
    drops.

    `device_loading_context` copies offloaded params to the GPU, lets the quant
    method re-register them (`replace_parameter` in `_setup_kernel`), then
    restores them with a plain `.to("cpu")` — which yields *unpinned* storage.
    Without the re-pin, every H2D expert copy would silently become synchronous.
    """
    device = torch.device("cuda")
    offloader = _offloader()
    layers = offloader.wrap_modules(_make_layer(device) for _ in range(2))
    if not offloader.pin_memory:
        pytest.skip("pinned memory unavailable")

    routed = layers[0].mlp.experts.routed_experts
    assert routed.w13_weight.is_pinned()

    # Simulate process_weights_after_loading.
    for layer in layers:
        experts = layer.mlp.experts.routed_experts
        with device_loading_context(experts, device):
            assert experts.w13_weight.device.type == "cuda"
            # _setup_kernel replaces the parameter object outright.
            replace_parameter(experts, "w13_weight", experts.w13_weight.data.clone())

    assert not routed.w13_weight.is_pinned(), "precondition: pinning was dropped"

    offloader.post_init()

    for layer in layers:
        experts = layer.mlp.experts.routed_experts
        for name in EXPERT_PARAMS:
            param = getattr(experts, name)
            assert param.device.type == "cpu"
            assert param.is_pinned()
