# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the quantized expert store and its dequant kernel."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.config import ExpertCacheOffloadConfig, OffloadConfig
from vllm.model_executor.layers.expert_prefetch import maybe_create_expert_cache
from vllm.model_executor.layers.expert_prefetch.expert_cache import (
    NATIVE_BITS,
)
from vllm.model_executor.layers.expert_prefetch.expert_quant import (
    QUANT_STORE_ATTR,
    SUPPORTED_EXPERT_QUANT_BITS,
    QuantBlobLayout,
    blob_aliases,
    dequant_experts_into_,
    quantize_experts,
)
from vllm.model_executor.layers.expert_prefetch.prefetch_controller import (
    PrefetchController,
)
from vllm.model_executor.offloader import create_offloader, get_offloader, set_offloader

NUM_EXPERTS = 6
ROWS = 5
COLS = 256
GROUP = 128

ALL_BITS = sorted(SUPPORTED_EXPERT_QUANT_BITS)

# Cache-integration sizes: both reduction dims must be divisible by GROUP.
CACHE_EXPERTS = 8
CACHE_TOP_K = 2
CACHE_HIDDEN = 256
CACHE_INTERMEDIATE = 128


def _reference_dequant(q, cols: int, num_bits: int) -> torch.Tensor:
    """Unpack and dequantize with plain torch, in the kernel's exact order."""
    packed = q.qweight
    per_byte = 8 // num_bits
    mask = (1 << num_bits) - 1
    lanes = [(packed >> (i * num_bits)) & mask for i in range(per_byte)]
    unpacked = torch.stack(lanes, dim=-1).reshape(*packed.shape[:-1], cols)
    codes = unpacked.to(torch.int32) - (1 << (num_bits - 1))

    num_groups = q.scale.shape[-1]
    scale = q.scale.to(torch.float32).repeat_interleave(cols // num_groups, dim=-1)
    return (codes.to(torch.float32) * scale).to(torch.bfloat16)


def _run_kernel(q, cols, slot_index, device, num_bits=4):
    packed = q.qweight.to(device)
    scale = q.scale.to(device)
    out = torch.zeros(
        (packed.shape[0], packed.shape[1], cols), dtype=torch.bfloat16, device=device
    )
    group_size = cols // q.scale.shape[-1]
    dequant_experts_into_(
        packed, scale, out, slot_index.to(device), group_size, num_bits
    )
    return out


@pytest.mark.parametrize("num_bits", ALL_BITS)
def test_packed_bit_order(num_bits):
    """Column c must land at bit offset (c % per_byte) * num_bits of byte
    c // per_byte.

    Pins the packed format so the packer and the kernel cannot drift apart.
    """
    per_byte = 8 // num_bits
    qmax = (1 << (num_bits - 1)) - 1
    zp = 1 << (num_bits - 1)

    # Scale is amax/qmax, so making the group max equal qmax makes each code
    # equal its own value and the expected bytes computable by hand.
    values = [float(v) for v in range(-qmax, qmax + 1)][:per_byte]
    values = (values + [float(qmax)] * per_byte)[:per_byte]
    src = torch.tensor(values, dtype=torch.bfloat16).view(1, 1, per_byte)

    q = quantize_experts(
        src, num_bits, group_size=-1, device=torch.device("cpu"), pin_memory=False
    )

    want = 0
    for lane, value in enumerate(values):
        want |= (int(value) + zp) << (lane * num_bits)
    assert q.qweight[0, 0, 0].item() == want


def test_blob_aliases_round_trip():
    """qweight and scale must be views of the blob, not copies of it.

    The blob is the only thing that crosses PCIe, so anything the packer writes
    through an alias has to be inside it -- otherwise the GPU would receive
    stale scales with no test failing.
    """
    layout = QuantBlobLayout(rows=3, cols=COLS, num_bits=4, num_groups=COLS // GROUP)
    blob = torch.zeros((2, layout.blob_bytes), dtype=torch.uint8)
    qweight, scale = blob_aliases(blob, layout)

    qweight[1, 2, 0] = 0xAB
    scale[1, 2, 0] = 1.5

    assert blob[1, 2 * layout.packed_cols].item() == 0xAB
    off = layout.scale_offset + 2 * layout.num_groups * 2
    assert blob[1, off : off + 2].view(torch.bfloat16).item() == 1.5
    # Scales must start past the packed weights, or the two would overlap.
    assert layout.scale_offset >= layout.qweight_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant kernel needs CUDA")
@pytest.mark.parametrize("group_size", [GROUP, -1])
@pytest.mark.parametrize("num_bits", ALL_BITS)
def test_kernel_matches_reference_dequant(group_size, num_bits):
    """The kernel must be bitwise identical to the torch reference.

    Same arithmetic in the same order, so any tolerance here means a bug.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    src = torch.randn(NUM_EXPERTS, ROWS, COLS, dtype=torch.bfloat16)

    q = quantize_experts(src, num_bits, group_size, device, pin_memory=False)
    out = _run_kernel(
        q, COLS, torch.arange(NUM_EXPERTS, dtype=torch.int32), device, num_bits
    )

    assert torch.equal(out.cpu(), _reference_dequant(q, COLS, num_bits))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant kernel needs CUDA")
def test_scattered_slots_match_dense_slots():
    """Slot indirection must work for arbitrary eviction victims.

    The fetch-on-demand path passes scattered victim slots; the prefetch path
    passes a dense range. Only this test exercises non-trivial indices.
    """
    device = torch.device("cuda")
    torch.manual_seed(1)
    src = torch.randn(NUM_EXPERTS, ROWS, COLS, dtype=torch.bfloat16)
    q = quantize_experts(src, 4, GROUP, device, pin_memory=False)

    dense = _run_kernel(q, COLS, torch.arange(NUM_EXPERTS, dtype=torch.int32), device)

    scattered = torch.tensor([4, 0, 5, 1], dtype=torch.int32)
    out = _run_kernel(q, COLS, scattered, device)

    for slot in scattered.tolist():
        assert torch.equal(out[slot], dense[slot]), f"slot {slot} differs"
    # Slots the kernel was not told to touch must be left alone.
    for slot in (2, 3):
        assert torch.all(out[slot] == 0), f"slot {slot} was written"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant kernel needs CUDA")
def test_rejects_strided_slot_vectors_and_output():
    """Triton hands the kernel a bare pointer, so a strided argument it has no
    stride for is addressed as if it were dense.

    `slot_index[:1].expand(n)` reads as [0, 0, 0] in torch but as [0, 1, 2] in
    the kernel -- the wrong experts, with no error and plausible output. These
    guards are the only thing standing between that and silently wrong logits,
    so they get a test of their own.
    """
    device = torch.device("cuda")
    src = torch.randn(NUM_EXPERTS, ROWS, COLS, dtype=torch.bfloat16)
    q = quantize_experts(src, 4, GROUP, device, pin_memory=False)
    packed, scale = q.qweight.to(device), q.scale.to(device)
    out = torch.zeros((NUM_EXPERTS, ROWS, COLS), dtype=torch.bfloat16, device=device)
    dense = torch.arange(3, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="contiguous"):
        dequant_experts_into_(packed, scale, out, dense[:1].expand(3), GROUP, 4)
    with pytest.raises(ValueError, match="contiguous"):
        dequant_experts_into_(
            packed, scale, out, dense, GROUP, 4, dst_slots=dense[:1].expand(3)
        )
    with pytest.raises(ValueError, match="out must be contiguous"):
        dequant_experts_into_(packed, scale, out[:, :, ::2], dense, GROUP, 4)

    # Slicing whole slots off the front stays contiguous, and the chunked fetch
    # path depends on that -- the guard must not reject it.
    dequant_experts_into_(packed, scale, out[:3], dense, GROUP, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant kernel needs CUDA")
def test_group_boundary_scales():
    """Groups with wildly different ranges catch scale-index off-by-ones.

    On random weights an off-by-one is invisible; here it is a 100x error.
    """
    device = torch.device("cuda")
    src = torch.empty(1, 1, 2 * GROUP, dtype=torch.bfloat16)
    src[..., :GROUP] = 1.0
    src[..., GROUP:] = 100.0

    q = quantize_experts(src, 4, GROUP, device, pin_memory=False)
    out = _run_kernel(q, 2 * GROUP, torch.zeros(1, dtype=torch.int32), device).cpu()

    assert torch.all(out[..., :GROUP] == 1.0)
    assert torch.all(out[..., GROUP:] == 100.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant kernel needs CUDA")
def test_dead_expert_row_does_not_produce_nan():
    """An all-zero row has amax 0; the scale floor must keep it finite."""
    device = torch.device("cuda")
    src = torch.randn(2, ROWS, COLS, dtype=torch.bfloat16)
    src[1, 2] = 0.0

    q = quantize_experts(src, 4, GROUP, device, pin_memory=False)
    out = _run_kernel(q, COLS, torch.arange(2, dtype=torch.int32), device)

    assert torch.isfinite(out).all()
    assert torch.all(out[1, 2] == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant kernel needs CUDA")
def test_quantization_error_matches_theory():
    """Pin the int4 error budget against the analytic value.

    For a Gaussian group of 128, E[amax] ~ 2.78 sigma, so the step is
    2.78/7 sigma and uniform-quantization RMSE is step/sqrt(12) ~ 0.115 sigma.
    Measured 0.114. This catches gross scheme errors (wrong denominator,
    unapplied scales) that the bitwise kernel test cannot see, since that one
    compares against the same packed values the packer produced.

    Note how large this is: int4 RTN costs ~11% relative error on the weights.
    int8 on the same data costs 0.6%. Do not enable int4 without an eval.
    """
    device = torch.device("cuda")
    torch.manual_seed(2)
    src = torch.randn(NUM_EXPERTS, ROWS, COLS, dtype=torch.bfloat16)

    q = quantize_experts(src, 4, GROUP, device, pin_memory=False)
    out = _run_kernel(q, COLS, torch.arange(NUM_EXPERTS, dtype=torch.int32), device)

    ref = src.to(device=device, dtype=torch.float32)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert 0.08 < rel < 0.15, f"int4 relative error {rel:.4f} is off theory (~0.115)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dequant kernel needs CUDA")
def test_smaller_groups_reduce_error():
    """Finer groups must be monotonically more accurate.

    A scale-indexing bug that ignored the group dim would flatten this curve.
    """
    device = torch.device("cuda")
    torch.manual_seed(3)
    src = torch.randn(2, ROWS, 1024, dtype=torch.bfloat16)
    ref = src.to(device=device, dtype=torch.float32)

    errors = []
    for group_size in (32, 128, -1):
        q = quantize_experts(src, 4, group_size, device, pin_memory=False)
        out = _run_kernel(q, 1024, torch.arange(2, dtype=torch.int32), device)
        errors.append(((out.float() - ref).norm() / ref.norm()).item())

    assert errors[0] < errors[1] < errors[2], f"error not monotonic in group: {errors}"


# ---------------------------------------------------------------------------
# Multi-precision config validation (CPU only)
# ---------------------------------------------------------------------------


def _offload_cfg(**expert_cache_kw) -> OffloadConfig:
    return OffloadConfig(
        offload_backend="expert_cache",
        expert_cache=ExpertCacheOffloadConfig(**expert_cache_kw),
    )


def test_config_dedups_and_orders_fidelity_descending():
    cfg = _offload_cfg(expert_quant_bits=[2, 8, 4, 4])
    assert cfg.expert_cache.expert_quant_bits == [8, 4, 2]


def test_config_rejects_unsupported_width():
    with pytest.raises(ValueError):
        _offload_cfg(expert_quant_bits=[3])


def test_config_group_size_must_divide_smallest_width():
    # min width 2 -> 4 values per byte; 130 is not a multiple of 4.
    with pytest.raises(ValueError):
        _offload_cfg(expert_quant_bits=[8, 2], expert_quant_group_size=130)


def test_config_group_size_per_row_allowed():
    cfg = _offload_cfg(expert_quant_bits=[2], expert_quant_group_size=-1)
    assert cfg.expert_cache.expert_quant_group_size == -1


def test_config_empty_disables_quant():
    assert _offload_cfg().expert_cache.expert_quant_bits == []


# ---------------------------------------------------------------------------
# Controller precision selection (CPU only)
# ---------------------------------------------------------------------------


def _controller(t_e, top_k=8, num_experts=64, num_slots=64, **cfg_kw):
    return PrefetchController(
        top_k=top_k,
        num_experts=num_experts,
        num_slots=num_slots,
        cfg=ExpertCacheOffloadConfig(**cfg_kw),
        t_e_ms=t_e,
    )


def test_select_precision_prefers_highest_fidelity_that_covers():
    c = _controller({16: 1.0, 8: 0.5, 4: 0.25, 2: 0.125})
    bits, p = c._select_precision(2.0, {16: 5.0, 8: 10.0, 4: 20.0, 2: 40.0})
    assert bits == NATIVE_BITS and p == 5.0


def test_select_precision_steps_down_when_bf16_insufficient():
    c = _controller({16: 1.0, 8: 0.5, 4: 0.25, 2: 0.125})
    # Poisson target 12: bf16 bubble 10 < 12, int8 bubble 20 >= 12.
    bits, p = c._select_precision(12.0, {16: 10.0, 8: 20.0, 4: 40.0, 2: 80.0})
    assert bits == 8 and p == 20.0


def test_select_precision_falls_back_to_int2_with_poisson():
    c = _controller({16: 1.0, 8: 0.5, 4: 0.25, 2: 0.125})
    # Even int2's bubble (80) is below the Poisson target (100).
    bits, p = c._select_precision(100.0, {16: 10.0, 8: 20.0, 4: 40.0, 2: 80.0})
    assert bits == 2 and p == 100.0


def test_select_precision_without_bubble_keeps_bf16():
    c = _controller({16: 1.0, 8: 0.5, 4: 0.25, 2: 0.125})
    bits, p = c._select_precision(3.0, {})
    assert bits == NATIVE_BITS and p == 3.0


def test_select_reduces_to_max_when_quant_disabled():
    c = _controller({16: 1.0})
    assert c._precisions == [NATIVE_BITS]
    # bubble < poisson -> poisson wins (max), still bf16.
    assert c._select_precision(7.0, {16: 3.0}) == (NATIVE_BITS, 7.0)
    # bubble >= poisson -> bubble wins (max), still bf16.
    assert c._select_precision(2.0, {16: 3.0}) == (NATIVE_BITS, 3.0)


def test_observe_t_e_updates_only_provided_width_and_clamps():
    c = _controller({16: 1.0, 2: 0.1})
    c.observe_t_e({2: 100.0})  # absurdly high -> clamped to 4x calibration
    assert c._t_e_ms[16] == 1.0  # untouched
    assert c._t_e_ms[2] == pytest.approx(0.95 * 0.1 + 0.05 * (0.1 * 4.0))


def test_select_pinned_returns_pinned_bits():
    c = _controller({16: 1.0, 4: 0.25}, prefetch_topk=5, prefetch_pin_bits=4)
    assert c.select(1) == (5, 4)


def test_pinned_bits_must_be_resident():
    with pytest.raises(ValueError):
        _controller({16: 1.0, 4: 0.25}, prefetch_topk=5, prefetch_pin_bits=8)


# ---------------------------------------------------------------------------
# Cache integration: multi-precision store, on-demand int2, prefetch per width
# ---------------------------------------------------------------------------


class UnquantizedFusedMoEMethod:
    """Named to match the guard, which checks the quant method by class name."""

    is_monolithic = False


class FakeRoutedExperts(nn.Module):
    """The subset of `RoutedExperts` the cache touches, with quant-exact weights.

    Expert `e` of layer `l` is filled uniformly with `qmax * 2 ** (8*l + e)` for
    the given `num_bits`. The group scale is `amax / qmax`, an exact power of
    two, so the code comes out exactly `qmax` and the value round-trips bitwise
    at that width -- a slot serving the wrong expert (or the right expert of the
    wrong layer) is then a bitwise difference with no tolerance to hide behind.
    """

    def __init__(self, layer_idx: int, hidden: int, num_bits: int):
        super().__init__()
        self.layer_name = f"model.layers.{layer_idx}.mlp.experts"
        self.global_num_experts = CACHE_EXPERTS
        self.local_num_experts = CACHE_EXPERTS
        self.top_k = CACHE_TOP_K
        self.expert_map = None
        self.moe_config = SimpleNamespace(has_bias=False)
        self.quant_method = UnquantizedFusedMoEMethod()

        experts = torch.arange(CACHE_EXPERTS, dtype=torch.float32)
        qmax = float((1 << (num_bits - 1)) - 1)
        fill = qmax * torch.pow(2.0, 8 * layer_idx + experts)
        self.w13_weight = nn.Parameter(
            fill.view(-1, 1, 1)
            .expand(CACHE_EXPERTS, 2 * CACHE_INTERMEDIATE, hidden)
            .to(torch.bfloat16)
            .cuda(),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            fill.view(-1, 1, 1)
            .expand(CACHE_EXPERTS, hidden, CACHE_INTERMEDIATE)
            .to(torch.bfloat16)
            .cuda(),
            requires_grad=False,
        )


def _fake_decoder_layer(layer_idx: int, hidden: int, num_bits: int) -> nn.Module:
    experts = nn.Module()
    experts.routed_experts = FakeRoutedExperts(layer_idx, hidden, num_bits)
    mlp = nn.Module()
    mlp.experts = experts
    layer = nn.Module()
    layer.mlp = mlp
    return layer


def _build_cache(quant_bits=(8, 4, 2), group_size=GROUP, num_layers=2, fill_bits=None):
    if fill_bits is None:
        fill_bits = min(quant_bits) if quant_bits else 4
    set_offloader(
        create_offloader(
            _offload_cfg(
                num_cache_slots=CACHE_EXPERTS,
                expert_quant_bits=list(quant_bits),
                expert_quant_group_size=group_size,
            )
        )
    )
    offloader = get_offloader()
    layers = offloader.wrap_modules(
        _fake_decoder_layer(i, CACHE_HIDDEN, fill_bits) for i in range(num_layers)
    )
    moes = [layer.mlp.experts.routed_experts for layer in layers]
    cache = maybe_create_expert_cache(moes)
    assert cache is not None
    offloader.post_init()
    return cache, moes, layers


@pytest.fixture(autouse=True)
def _restore_offloader():
    yield
    set_offloader(create_offloader(OffloadConfig()))


def _shrink_rings(cache, num_slots):
    device = cache.ping.device
    store = cache.owner.__dict__[QUANT_STORE_ATTR]
    shapes = {}
    for name in cache.param_names:
        layout = store[name][cache.quant_bits[0]].layout
        shapes[name] = (layout.rows, layout.cols, layout.num_groups)
    for staging in (cache.prefetch_staging, cache.ondemand_staging):
        staging.num_slots = num_slots
        staging.allocate(shapes, cache.quant_bits, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_multi_precision_store_built_per_width():
    cache, moes, layers = _build_cache(quant_bits=(8, 4, 2))

    assert QUANT_STORE_ATTR not in layers[0].__dict__
    store = moes[0].__dict__[QUANT_STORE_ATTR]
    assert set(store) == {"w13_weight", "w2_weight"}
    for name in store:
        assert set(store[name]) == {8, 4, 2}
        for q in store[name].values():
            assert q.qweight.is_pinned() and q.scale.is_pinned()
            assert q.qweight.dtype == torch.uint8

    w13 = store["w13_weight"]
    assert w13[8].qweight.shape == (CACHE_EXPERTS, 2 * CACHE_INTERMEDIATE, CACHE_HIDDEN)
    assert w13[4].qweight.shape[-1] == CACHE_HIDDEN // 2
    assert w13[2].qweight.shape[-1] == CACHE_HIDDEN // 4
    # The store is a mirror: the originals stay put on the host, still pinned.
    assert moes[0].w13_weight.device.type == "cpu"
    assert moes[0].w13_weight.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_store_absent_and_native_when_quant_off():
    cache, moes, _ = _build_cache(quant_bits=())
    assert QUANT_STORE_ATTR not in moes[0].__dict__
    assert cache.quant_bits == ()
    assert cache._ondemand_bits == NATIVE_BITS
    assert cache.prefetch_staging.blobs == {}
    assert cache.ondemand_staging.blobs == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_fetch_on_demand_uses_int2():
    cache, moes, _ = _build_cache(quant_bits=(8, 4, 2))  # fill_bits = 2
    assert cache._ondemand_bits == 2
    cache.reset()
    torch.manual_seed(0)

    for moe in moes:
        topk_ids = torch.randint(0, CACHE_EXPERTS, (3, CACHE_TOP_K), device="cuda")
        buf, slot_ids = cache.resolve(moe, topk_ids)
        for name in ("w13_weight", "w2_weight"):
            got = buf.params[name][slot_ids]
            want = getattr(moe, name).cuda()[topk_ids]
            assert torch.equal(got, want), f"{moe.layer_name}: {name} mismatch"
        cache.flip()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
@pytest.mark.parametrize("bits", [NATIVE_BITS, 8, 4, 2])
def test_prefetch_serves_dequantized_weights_per_precision(bits):
    quant = () if bits == NATIVE_BITS else (bits,)
    fill_bits = 4 if bits == NATIVE_BITS else bits
    cache, moes, _ = _build_cache(quant_bits=quant, fill_bits=fill_bits)
    cache.reset()
    stream = torch.cuda.Stream()
    torch.manual_seed(1)

    topk_ids = torch.randint(0, CACHE_EXPERTS, (3, CACHE_TOP_K), device="cuda")
    cache.prefetch(moes[0], torch.unique(topk_ids), stream, num_bits=bits)
    cache.flip()

    # Sampling records the hit accounting; without it `hit_rate` stays 0/0 and
    # cannot tell a landed prefetch from an on-demand rescue (both bit-exact).
    cache.begin_forward(sampling=True)
    buf, slot_ids = cache.resolve(moes[0], topk_ids)
    assert cache.hit_rate() == 1.0, "prefetched experts should all be hits"
    for name in ("w13_weight", "w2_weight"):
        got = buf.params[name][slot_ids]
        want = getattr(moes[0], name).cuda()[topk_ids]
        assert torch.equal(got, want), f"{name} mismatch after int{bits} prefetch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
@pytest.mark.parametrize("ring_slots", [1, 3])
def test_chunked_on_demand_int2_through_small_ring(ring_slots):
    cache, moes, _ = _build_cache(quant_bits=(8, 4, 2))  # on-demand int2
    _shrink_rings(cache, ring_slots)
    cache.reset()

    topk_ids = torch.arange(CACHE_EXPERTS, device="cuda").view(-1, 1)
    buf, slot_ids = cache.resolve(moes[0], topk_ids)
    for name in ("w13_weight", "w2_weight"):
        got = buf.params[name][slot_ids]
        want = getattr(moes[0], name).cuda()[topk_ids]
        assert torch.equal(got, want), f"{name} mismatch with ring={ring_slots}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_chunked_prefetch_int2_through_small_ring():
    cache, moes, _ = _build_cache(quant_bits=(2,))
    _shrink_rings(cache, 3)
    cache.reset()
    stream = torch.cuda.Stream()

    topk_ids = torch.arange(CACHE_EXPERTS, device="cuda").view(-1, 1)
    cache.prefetch(moes[0], torch.unique(topk_ids), stream, num_bits=2)
    cache.flip()

    cache.begin_forward(sampling=True)
    buf, slot_ids = cache.resolve(moes[0], topk_ids)
    assert cache.hit_rate() == 1.0
    for name in ("w13_weight", "w2_weight"):
        got = buf.params[name][slot_ids]
        want = getattr(moes[0], name).cuda()[topk_ids]
        assert torch.equal(got, want), f"{name} mismatch after chunked prefetch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_calibrate_copy_time_returns_all_precisions():
    cache, _, _ = _build_cache(quant_bits=(8, 4, 2))
    times = cache.calibrate_copy_time()
    assert set(times) == {NATIVE_BITS, 8, 4, 2}
    assert all(t > 0.0 for t in times.values())


