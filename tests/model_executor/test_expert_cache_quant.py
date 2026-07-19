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
    DEQUANT_RING_SLOTS,
)
from vllm.model_executor.layers.expert_prefetch.expert_quant import (
    QUANT_STORE_ATTR,
    SUPPORTED_EXPERT_QUANT_BITS,
    QuantBlobLayout,
    blob_aliases,
    dequant_experts_into_,
    quantize_experts,
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


class UnquantizedFusedMoEMethod:
    """Named to match the guard, which checks the quant method by class name."""

    is_monolithic = False


class Fp8MoEMethod:
    is_monolithic = False


class FakeRoutedExperts(nn.Module):
    """The subset of `RoutedExperts` the cache touches, with quant-exact weights.

    Expert `e` of layer `l` is filled with `qmax * 2 ** (8 * l + e)`. The group
    scale is `amax / qmax`, so it comes out an exact power of two and the code
    comes out exactly `qmax`: the value round-trips bitwise at *every* supported
    bit width, and a slot serving the wrong expert -- or the right expert of the
    wrong layer -- is a bitwise difference with no quantization tolerance to
    hide behind. Sizing the fill by `qmax` is what makes it width-independent;
    a fixed fill only round-trips at int4.
    """

    def __init__(self, layer_idx: int, hidden: int = CACHE_HIDDEN, num_bits: int = 4):
        super().__init__()
        self.layer_name = f"model.layers.{layer_idx}.mlp.experts"
        self.global_num_experts = CACHE_EXPERTS
        self.local_num_experts = CACHE_EXPERTS
        self.top_k = CACHE_TOP_K
        self.expert_map = None
        self.moe_config = SimpleNamespace(has_bias=False)
        # The quant guard checks this by class name.
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


def _fake_decoder_layer(
    layer_idx: int, hidden: int = CACHE_HIDDEN, num_bits: int = 4
) -> nn.Module:
    experts = nn.Module()
    experts.routed_experts = FakeRoutedExperts(layer_idx, hidden, num_bits)
    mlp = nn.Module()
    mlp.experts = experts
    layer = nn.Module()
    layer.mlp = mlp
    return layer


def _build_cache(group_size=GROUP, quant_bits=4, hidden=CACHE_HIDDEN, num_layers=2):
    set_offloader(
        create_offloader(
            OffloadConfig(
                offload_backend="expert_cache",
                expert_cache=ExpertCacheOffloadConfig(
                    num_cache_slots=CACHE_EXPERTS,
                    expert_quant_bits=quant_bits,
                    expert_quant_group_size=group_size,
                ),
            )
        )
    )
    offloader = get_offloader()
    layers = offloader.wrap_modules(
        _fake_decoder_layer(i, hidden, quant_bits or 4) for i in range(num_layers)
    )
    moes = [layer.mlp.experts.routed_experts for layer in layers]
    cache = maybe_create_expert_cache(moes)
    assert cache is not None
    offloader.post_init()
    return cache, moes, layers


@pytest.fixture
def quant_cache():
    cache, moes, layers = _build_cache()
    yield cache, moes, layers
    set_offloader(create_offloader(OffloadConfig()))


@pytest.fixture
def reset_offloader():
    yield
    set_offloader(create_offloader(OffloadConfig()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_store_is_pinned_and_lives_on_routed_experts(quant_cache):
    """The store must hang off RoutedExperts, not the decoder layer.

    `_offloaded` records (decoder_layer, dotted_name) while `fetch` looks the
    store up on the RoutedExperts by leaf name, so the submodule walk in
    `_quantize_offloaded` is what connects the two.
    """
    _, moes, layers = quant_cache

    assert QUANT_STORE_ATTR not in layers[0].__dict__
    store = moes[0].__dict__[QUANT_STORE_ATTR]
    assert set(store) == {"w13_weight", "w2_weight"}

    for quantized in store.values():
        assert quantized.qweight.is_pinned()
        assert quantized.scale.is_pinned()
        assert quantized.qweight.dtype == torch.uint8

    # Packed store is a quarter of the bf16 original, plus the scales.
    packed = store["w13_weight"].qweight
    assert packed.shape == (CACHE_EXPERTS, 2 * CACHE_INTERMEDIATE, CACHE_HIDDEN // 2)
    # The store is a mirror: the originals stay put, and stay pinned.
    assert moes[0].w13_weight.device.type == "cpu"
    assert moes[0].w13_weight.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_store_is_absent_when_quantization_is_off(reset_offloader):
    cache, moes, _ = _build_cache(quant_bits=0)
    assert QUANT_STORE_ATTR not in moes[0].__dict__
    assert cache.qspec is None
    assert cache.prefetch_staging.blobs == {}
    assert cache.ondemand_staging.blobs == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_fetch_on_demand_serves_dequantized_weights(quant_cache):
    """The miss path: scattered victim slots, dequant on the current stream."""
    cache, moes, _ = quant_cache
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
def test_prefetch_serves_dequantized_weights(quant_cache):
    """The prefetch path: dequant launched from the worker thread, side stream.

    Also the only coverage that the kernel is ordered before the prefetch event,
    since `resolve` waits on that event and then reads the slots.
    """
    cache, moes, _ = quant_cache
    cache.reset()
    stream = torch.cuda.Stream()
    torch.manual_seed(1)

    topk_ids = torch.randint(0, CACHE_EXPERTS, (3, CACHE_TOP_K), device="cuda")
    cache.prefetch(moes[0], torch.unique(topk_ids), stream)
    cache.flip()

    buf, slot_ids = cache.resolve(moes[0], topk_ids)
    assert cache.hit_rate() == 1.0, "prefetched experts should all be hits"
    for name in ("w13_weight", "w2_weight"):
        got = buf.params[name][slot_ids]
        want = getattr(moes[0], name).cuda()[topk_ids]
        assert torch.equal(got, want), f"{name} mismatch after prefetch"


def _shrink_rings(cache, num_slots):
    """Re-allocate both rings smaller than a layer, forcing chunked fetches.

    The default ring holds more experts than these fixtures have, so without
    this nothing ever exercises the loop -- and a ring that is reused across
    chunks is exactly where a stale slot or a mixed-up source/destination index
    would show up.
    """
    device = cache.ping.cached_expert_ids.device
    store = cache.owner.__dict__[QUANT_STORE_ATTR]
    layouts = {name: store[name].layout for name in cache.param_names}
    for staging in (cache.prefetch_staging, cache.ondemand_staging):
        staging.num_slots = num_slots
        staging.allocate(layouts, device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
@pytest.mark.parametrize("ring_slots", [1, 3])
def test_chunked_fetch_through_small_ring(reset_offloader, ring_slots):
    """A fetch longer than the ring must still serve every expert.

    Ring slots are recycled between chunks, so this is the case where the
    copies of chunk k+1 race the dequant of chunk k unless stream order holds
    them apart.
    """
    cache, moes, _ = _build_cache()
    _shrink_rings(cache, ring_slots)
    cache.reset()

    # Every expert of the layer at once, so the fetch spans several chunks.
    topk_ids = torch.arange(CACHE_EXPERTS, device="cuda").view(-1, 1)
    buf, slot_ids = cache.resolve(moes[0], topk_ids)

    for name in ("w13_weight", "w2_weight"):
        got = buf.params[name][slot_ids]
        want = getattr(moes[0], name).cuda()[topk_ids]
        assert torch.equal(got, want), f"{name} mismatch with ring={ring_slots}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_chunked_prefetch_through_small_ring(reset_offloader):
    """Same, on the side stream: the ring is the prefetch worker's own."""
    cache, moes, _ = _build_cache()
    _shrink_rings(cache, 3)
    cache.reset()
    stream = torch.cuda.Stream()

    topk_ids = torch.arange(CACHE_EXPERTS, device="cuda").view(-1, 1)
    cache.prefetch(moes[0], torch.unique(topk_ids), stream)
    cache.flip()

    buf, slot_ids = cache.resolve(moes[0], topk_ids)
    assert cache.hit_rate() == 1.0
    for name in ("w13_weight", "w2_weight"):
        got = buf.params[name][slot_ids]
        want = getattr(moes[0], name).cuda()[topk_ids]
        assert torch.equal(got, want), f"{name} mismatch after chunked prefetch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
@pytest.mark.parametrize("ring_slots", [3, DEQUANT_RING_SLOTS])
def test_fetch_into_scattered_victim_slots(reset_offloader, ring_slots):
    """Victims are the slots the layer does *not* need, so they come out
    scattered -- and a chunk's destinations are then unrelated to its ring
    slots.

    A fetch into a freshly reset buffer always evicts slots 0, 1, 2, ... in
    order, so destination and ring index coincide and every mapping bug is
    invisible. Priming the buffer first is what breaks that coincidence: here
    the second resolve keeps experts 1 and 3 in place and has to fill around
    them.
    """
    cache, moes, _ = _build_cache()
    _shrink_rings(cache, ring_slots)
    cache.reset()
    moe = moes[0]

    # Fill slots 0..3 with experts 0..3, then keep only 1 and 3.
    cache.resolve(moe, torch.arange(4, device="cuda").view(1, -1))
    kept_and_new = torch.tensor([[1, 3, 4, 5, 6, 7]], device="cuda")
    buf, slot_ids = cache.resolve(moe, kept_and_new)

    staged = buf.cached_expert_ids.tolist()
    victims = [staged.index(e) for e in (4, 5, 6, 7)]
    assert victims != sorted(range(len(victims))), (
        f"victims {victims} came out dense, so this test proves nothing"
    )

    for name in ("w13_weight", "w2_weight"):
        got = buf.params[name][slot_ids]
        want = getattr(moe, name).cuda()[kept_and_new]
        assert torch.equal(got, want), f"{name} mismatch at ring={ring_slots}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_rings_are_independent(reset_offloader):
    """The prefetch and on-demand rings must not be the same storage.

    They are written from different streams at overlapping times, so sharing
    one would corrupt whichever fetch lost the race -- silently, since both
    produce plausible weights.
    """
    cache, _, _ = _build_cache()
    for name in cache.param_names:
        prefetch_blob = cache.prefetch_staging.blobs[name]
        ondemand_blob = cache.ondemand_staging.blobs[name]
        assert prefetch_blob.data_ptr() != ondemand_blob.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_ring_is_sized_for_the_widest_bit_width(reset_offloader):
    """One ring serves every width, so int4 must still reserve int8 bytes.

    This is what lets a future per-expert precision choice cost no extra
    memory; sizing to the configured width instead would quietly break it.
    """
    cache, _, _ = _build_cache(quant_bits=4)
    staging = cache.prefetch_staging
    for name in cache.param_names:
        packed_int8, _ = staging.aliases(name, 8)
        packed_int4, _ = staging.aliases(name, 4)
        packed_int2, _ = staging.aliases(name, 2)
        # Same storage, addressed at three densities.
        assert packed_int8.data_ptr() == packed_int4.data_ptr()
        assert packed_int8.shape[-1] == 2 * packed_int4.shape[-1]
        assert packed_int4.shape[-1] == 2 * packed_int2.shape[-1]
        assert staging.blobs[name].shape[-1] >= packed_int8[0].numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
@pytest.mark.parametrize("num_bits", ALL_BITS)
def test_cache_round_trips_every_bit_width(reset_offloader, num_bits):
    """The fixture weights are exact at every width, so this stays bitwise.

    Expert fill is `qmax * (100 * layer + expert)` and the group scale is
    `amax / qmax`, which makes the scale a small integer and the code exactly
    +-qmax -- no quantization tolerance for a wrong slot to hide behind.
    """
    cache, moes, _ = _build_cache(quant_bits=num_bits)
    cache.reset()

    topk_ids = torch.arange(CACHE_TOP_K, device="cuda").view(1, -1)
    buf, slot_ids = cache.resolve(moes[0], topk_ids)
    for name in ("w13_weight", "w2_weight"):
        got = buf.params[name][slot_ids]
        want = getattr(moes[0], name).cuda()[topk_ids]
        assert torch.equal(got, want), f"int{num_bits}: {name} mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_per_row_group_size_is_accepted(reset_offloader):
    """-1 is the escape hatch when TP leaves the reduction dim indivisible."""
    cache, moes, _ = _build_cache(group_size=-1)
    _, scale = cache.prefetch_staging.aliases("w13_weight", 4)
    assert scale.shape[-1] == 1
    cache.reset()
    topk_ids = torch.arange(CACHE_TOP_K, device="cuda").view(1, -1)
    buf, slot_ids = cache.resolve(moes[0], topk_ids)
    assert torch.equal(
        buf.params["w2_weight"][slot_ids], moes[0].w2_weight.cuda()[topk_ids]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_rejects_indivisible_reduction_dim(reset_offloader):
    """hidden=192 is even but not a multiple of 128, as TP sharding can produce."""
    with pytest.raises(NotImplementedError, match="expert_quant_group_size"):
        _build_cache(hidden=192)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
def test_rejects_already_quantized_layer(reset_offloader):
    set_offloader(
        create_offloader(
            OffloadConfig(
                offload_backend="expert_cache",
                expert_cache=ExpertCacheOffloadConfig(expert_quant_bits=4),
            )
        )
    )
    offloader = get_offloader()
    layers = offloader.wrap_modules(iter([_fake_decoder_layer(0)]))
    moe = layers[0].mlp.experts.routed_experts
    moe.quant_method = Fp8MoEMethod()
    cache = maybe_create_expert_cache([moe])
    assert cache is not None
    with pytest.raises(NotImplementedError, match="unquantized expert weights"):
        offloader.post_init()


# --- End-to-end against a real MoE layer ------------------------------------
#
# Unlike the unquantized cache, which is contractually bitwise identical to the
# stock layer, int4 staging changes the numbers. So the contract here is a
# bounded divergence, and the bound is what a wiring bug has to stay under.

MOE_HIDDEN = 256
MOE_INTERMEDIATE = 384
MOE_LAYERS = 2


def _build_moe_layers():
    from vllm.model_executor.layers.fused_moe import FusedMoE

    torch.manual_seed(0)
    layers = []
    for i in range(MOE_LAYERS):
        layer = FusedMoE(
            num_experts=CACHE_EXPERTS,
            top_k=CACHE_TOP_K,
            hidden_size=MOE_HIDDEN,
            intermediate_size=MOE_INTERMEDIATE,
            params_dtype=torch.bfloat16,
            prefix=f"model.layers.{i}.mlp.experts",
            renormalize=False,
            scoring_func="softmax",
        ).cuda()
        experts = layer.routed_experts
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
    return quant_method.forward_native(
        layer=layer.routed_experts,
        x=x.clone(),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts=None,
        shared_experts_input=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="expert cache requires CUDA")
@pytest.mark.parametrize("prefetch", [False, True])
def test_cached_quant_moe_close_to_stock_moe(dist_init, prefetch):
    """A real MoE layer staged as int4 must track the stock one closely.

    Parametrized over prefetch so both dequant launch paths are covered: the
    worker thread on a side stream, and fetch-on-demand on the current stream.
    Both currently produce identical output, which is the point.

    Measured relative L2 is 0.7-1.9% -- an order of magnitude below the ~11%
    error on the weights themselves, because the errors partly cancel across
    the K-reduction. The 5% bound leaves room for that to vary with the seed
    while still catching a scale-indexing or slot-remapping bug, which lands
    far outside it.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.v1.worker.workspace import init_workspace_manager

    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = dict()
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        init_workspace_manager(torch.accelerator.current_device_index())
        layers = _build_moe_layers()

        torch.manual_seed(1)
        x = 0.1 * torch.randn(6, MOE_HIDDEN, dtype=torch.bfloat16, device="cuda")
        router_logits = [
            torch.randn(6, CACHE_EXPERTS, dtype=torch.bfloat16, device="cuda")
            for _ in range(MOE_LAYERS)
        ]

        baseline = [
            _run_moe(layers[i], x, router_logits[i]).clone() for i in range(MOE_LAYERS)
        ]

        set_offloader(
            create_offloader(
                OffloadConfig(
                    offload_backend="expert_cache",
                    expert_cache=ExpertCacheOffloadConfig(
                        num_cache_slots=CACHE_EXPERTS,
                        expert_quant_bits=4,
                        expert_quant_group_size=GROUP,
                    ),
                )
            )
        )
        offloader = get_offloader()
        offloader.wrap_modules(iter(layers))
        moes = [layer.routed_experts for layer in layers]
        cache = maybe_create_expert_cache(moes)
        assert cache is not None
        offloader.post_init()
        cache.reset()

        # Multiple groups per row, or the group indexing is never exercised.
        _, w13_scale = cache.prefetch_staging.aliases("w13_weight", 4)
        _, w2_scale = cache.prefetch_staging.aliases("w2_weight", 4)
        assert w13_scale.shape[-1] == MOE_HIDDEN // GROUP
        assert w2_scale.shape[-1] == MOE_INTERMEDIATE // GROUP

        stream = torch.cuda.Stream()
        for i, layer in enumerate(layers):
            if prefetch:
                _, topk_ids = layer.router.select_experts(
                    hidden_states=x,
                    router_logits=router_logits[i],
                    topk_indices_dtype=layer._quant_method.topk_indices_dtype,
                )
                cache.prefetch(moes[i], torch.unique(topk_ids), stream)
                cache.flip()

            out = _run_moe(layer, x, router_logits[i])
            ref = baseline[i]

            # Elementwise allclose would flake: these are near-cancelling sums
            # in bf16. Relative L2 and cosine are the stable statistics.
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            cos = torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref.float().flatten(), dim=0
            ).item()
            assert rel < 0.05, f"layer {i}: relative L2 {rel:.4f} too high"
            assert cos > 0.995, f"layer {i}: cosine {cos:.5f} too low"
            # ...but it must not be *identical*, or int4 staging is not on.
            assert not torch.equal(out, ref), f"layer {i}: int4 path had no effect"

    set_offloader(create_offloader(OffloadConfig()))
