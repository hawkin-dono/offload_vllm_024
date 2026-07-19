# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-quantized CPU-side store for routed MoE expert weights.

The expert cache is bound by PCIe: every staged expert is a large bf16 block,
and every prefetch miss is a synchronous copy on the critical path. This module
builds a second, symmetric round-to-nearest int2/int4/int8 copy of the expert
weights in pinned host memory, so the cache can stage *that* over the bus and
unpack it on the GPU into the bf16 slots the MoE kernel already reads.

Layout mirrors the bf16 originals -- expert-major, contiguous per expert, so
`blob[expert_id]` stays a view into pinned storage and H2D copies stay async
(see `ExpertBuffer.fetch`). Scales are grouped along the reduction (last) dim.

One expert is a *single* blob of bytes: packed weights, padding to
`_BLOB_ALIGN`, then scales. `qweight` and `scale` are strided aliases of that
blob rather than tensors of their own, so staging an expert is one H2D copy
instead of two. The scales are a fixed ~74 KiB per expert whatever the bit
width, so as the weights shrink that second small copy costs relatively more:
measured on an A100-PCIE (Gen4 x16, 32 experts of a Qwen3-30B-A3B layer),
folding it into the blob is worth ~2.5% at int8, ~5% at int4 and ~7% at int2.

This is a model-quality change, not just a perf one: RTN is uncalibrated, so
evaluate before enabling it -- especially at int2, where a group is quantized
to four levels.
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton

__all__ = [
    "QUANT_STORE_ATTR",
    "SUPPORTED_EXPERT_QUANT_BITS",
    "ExpertQuantSpec",
    "QuantBlobLayout",
    "QuantizedExpertWeight",
    "blob_aliases",
    "dequant_experts_into_",
    "quantize_experts",
]

SUPPORTED_EXPERT_QUANT_BITS = frozenset({2, 4, 8})
MAX_EXPERT_QUANT_BITS = max(SUPPORTED_EXPERT_QUANT_BITS)

# Attribute on the RoutedExperts module holding {param_name: QuantizedExpertWeight}.
# Set through __dict__ so the store stays out of state_dict().
QUANT_STORE_ATTR = "expert_quant_store"

# Scales start on a 16-byte boundary inside the blob. Two are load-bearing: the
# offset must be even for the bfloat16 alias to be expressible as a stride, and
# 16 keeps the scale rows at the alignment the copy engines like.
_BLOB_ALIGN = 16

# Working set of the GPU quantization chunk, in bytes.
_CHUNK_BYTES = 2 * 1024**3
# fp32 upcast plus intermediates, per source element.
_CHUNK_BYTES_PER_ELEM = 6


@dataclass(frozen=True)
class ExpertQuantSpec:
    """How the CPU-side expert store is quantized."""

    num_bits: int
    group_size: int  # -1 means one scale per output row

    @property
    def per_byte(self) -> int:
        """How many quantized values share a byte."""
        return 8 // self.num_bits

    @property
    def qmax(self) -> int:
        """Largest representable code. Symmetric, so dividing a group's amax by
        this puts the largest magnitude exactly on `qmax` and never clips."""
        return (1 << (self.num_bits - 1)) - 1

    @property
    def zero_point(self) -> int:
        """Added before packing so codes are unsigned and packing is a plain OR."""
        return 1 << (self.num_bits - 1)


@dataclass(frozen=True)
class QuantBlobLayout:
    """Byte layout of one expert inside its blob.

    Sizes are per expert; the store and the GPU staging ring both lay their
    experts out this way, which is what lets a whole expert cross PCIe as one
    contiguous copy.
    """

    rows: int
    cols: int
    num_bits: int
    num_groups: int

    @property
    def packed_cols(self) -> int:
        return self.cols * self.num_bits // 8

    @property
    def qweight_bytes(self) -> int:
        return self.rows * self.packed_cols

    @property
    def scale_offset(self) -> int:
        return self.qweight_bytes + (-self.qweight_bytes % _BLOB_ALIGN)

    @property
    def scale_bytes(self) -> int:
        return self.rows * self.num_groups * 2

    @property
    def blob_bytes(self) -> int:
        return self.scale_offset + self.scale_bytes


def blob_aliases(
    blob: torch.Tensor,
    layout: QuantBlobLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    """View `blob` (num_slots, blob_bytes) uint8 as (qweight, scale).

    Both alias the same storage, so writing either writes the blob -- which is
    the point: the blob is what crosses the bus, and these are how the packer
    and the dequant kernel address it. The scale alias needs a dtype the byte
    tensor cannot express, so it is built with `set_` rather than `view`.

    `blob` must own its storage from offset 0, since the aliases index that
    storage directly.
    """
    if blob.storage_offset() != 0:
        raise ValueError("blob_aliases requires a blob at storage offset 0.")
    if blob.ndim != 2 or blob.dtype != torch.uint8:
        raise ValueError(
            f"blob must be (num_slots, blob_bytes) uint8, got {tuple(blob.shape)} "
            f"{blob.dtype}."
        )
    if blob.shape[1] < layout.blob_bytes:
        raise ValueError(
            f"blob rows are {blob.shape[1]} bytes but the layout needs "
            f"{layout.blob_bytes}."
        )

    storage = blob.untyped_storage()
    slots, row_bytes = blob.shape
    qweight = torch.empty(0, dtype=torch.uint8, device=blob.device).set_(
        storage,
        0,
        (slots, layout.rows, layout.packed_cols),
        (row_bytes, layout.packed_cols, 1),
    )
    scale = torch.empty(0, dtype=torch.bfloat16, device=blob.device).set_(
        storage,
        layout.scale_offset // 2,
        (slots, layout.rows, layout.num_groups),
        (row_bytes // 2, layout.num_groups, 1),
    )
    return qweight, scale


@dataclass
class QuantizedExpertWeight:
    """Pinned quantized mirror of one expert weight tensor.

    Attributes:
        blob: (num_experts, blob_bytes) uint8 -- the thing that crosses PCIe.
        layout: how one expert is laid out inside a blob row.
        qweight: (num_experts, rows, packed_cols) uint8 alias of `blob`.
        scale: (num_experts, rows, num_groups) bfloat16 alias of `blob`.
    """

    blob: torch.Tensor
    layout: QuantBlobLayout
    qweight: torch.Tensor
    scale: torch.Tensor

    def nbytes(self) -> int:
        return self.blob.numel()


def validate_quantizable(
    src: torch.Tensor, group_size: int, name: str, num_bits: int = 4
) -> None:
    """Reject shapes the packer would otherwise get silently wrong.

    Lives here rather than only in the cache's load-time guard because the
    quantization pass runs first: `cols // group_size` would floor to one group
    for an indivisible dim and fail later with an opaque shape mismatch.

    Raises:
        NotImplementedError: if `src` cannot be packed at this bit width and
            group size.
    """
    if src.ndim != 3:
        raise NotImplementedError(
            "expert_quant_bits expects (num_experts, rows, cols) expert "
            f"weights; {name} has shape {tuple(src.shape)}."
        )
    per_byte = 8 // num_bits
    cols = src.shape[-1]
    if cols % per_byte:
        raise NotImplementedError(
            f"expert_quant_bits={num_bits} packs {per_byte} values per byte, so "
            f"the reduction dim of {name} ({cols}) must be divisible by "
            f"{per_byte}."
        )
    if group_size > 0 and cols % group_size:
        raise NotImplementedError(
            f"expert_quant_group_size ({group_size}) does not divide the "
            f"reduction dim of {name} ({cols}). Note this dim is sharded by "
            "tensor parallelism. Pass --expert-quant-group-size=-1 for one "
            "scale per row."
        )


def quantize_experts(
    src: torch.Tensor,
    num_bits: int,
    group_size: int,
    device: torch.device,
    pin_memory: bool,
    name: str = "expert weight",
) -> QuantizedExpertWeight:
    """Quantize `src` (num_experts, rows, cols) to a pinned blob store.

    Runs on `device` in chunks of experts: a pure-CPU pass over a large expert
    set takes tens of minutes, and this runs before memory profiling, when the
    GPU is at its most available.

    Args:
        src: bf16/fp16 expert weights, expert-major.
        num_bits: 2, 4 or 8.
        group_size: scale group along the last (reduction) dim; -1 for per-row.
        device: accelerator to run the quantization math on.
        pin_memory: whether to pin the resulting store.
        name: parameter name, used only in error messages.

    Returns:
        The blob and its `qweight`/`scale` aliases, all in host memory.

    Raises:
        NotImplementedError: if `src` cannot be packed this way.
        ValueError: if `num_bits` is unsupported.
    """
    if num_bits not in SUPPORTED_EXPERT_QUANT_BITS:
        raise ValueError(
            f"num_bits={num_bits} is not supported; expected one of "
            f"{sorted(SUPPORTED_EXPERT_QUANT_BITS)}."
        )
    validate_quantizable(src, group_size, name, num_bits)
    spec = ExpertQuantSpec(num_bits=num_bits, group_size=group_size)
    num_experts, rows, cols = src.shape
    group = cols if group_size <= 0 else group_size
    num_groups = cols // group
    layout = QuantBlobLayout(
        rows=rows, cols=cols, num_bits=num_bits, num_groups=num_groups
    )

    blob = torch.empty(
        (num_experts, layout.blob_bytes), dtype=torch.uint8, pin_memory=pin_memory
    )
    qweight, scale = blob_aliases(blob, layout)

    per_expert = max(rows * cols * _CHUNK_BYTES_PER_ELEM, 1)
    chunk = max(1, _CHUNK_BYTES // per_expert)
    # A dead expert row (amax == 0) would otherwise divide by zero and poison
    # the whole slot with NaN. q is 0 there anyway, so the output stays 0.
    tiny = torch.finfo(torch.bfloat16).tiny

    for start in range(0, num_experts, chunk):
        stop = min(start + chunk, num_experts)
        w = src[start:stop].to(device=device, dtype=torch.float32)
        w = w.view(-1, rows, num_groups, group)

        s = (w.abs().amax(-1) / spec.qmax).clamp_min(tiny).to(torch.bfloat16)
        # Divide by the bf16-rounded scale, which is the value the dequant
        # kernel loads. Measured effect at int4 is nil (the perturbation is far
        # below the rounding grid), but it keeps quantize and dequantize
        # describing the same function, which matters as bit widths grow -- at
        # int2 a group has four levels and the rounding grid is everything.
        q = (w / s.to(torch.float32).unsqueeze(-1)).round_()
        q = q.clamp_(-spec.qmax - 1, spec.qmax).to(torch.int8)
        q = (q + spec.zero_point).to(torch.uint8).view(-1, rows, cols)

        # Value at column c goes to byte c // per_byte, at bit offset
        # (c % per_byte) * num_bits -- so the lowest columns take the lowest
        # bits. `test_packed_bit_order` pins this against the kernel.
        lanes = q.view(-1, rows, layout.packed_cols, spec.per_byte)
        packed = lanes[..., 0]
        for lane in range(1, spec.per_byte):
            packed = packed | (lanes[..., lane] << (lane * num_bits))

        qweight[start:stop].copy_(packed)
        scale[start:stop].copy_(s.view(-1, rows, num_groups))

    return QuantizedExpertWeight(blob=blob, layout=layout, qweight=qweight, scale=scale)


@triton.jit
def _dequant_experts_kernel(
    src_slot_ptr,
    dst_slot_ptr,
    packed_ptr,
    scale_ptr,
    out_ptr,
    rows,
    cols,
    packed_slot_stride,
    packed_row_stride,
    scale_slot_stride,
    scale_row_stride,
    GROUP_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    PER_BYTE: tl.constexpr,
    MASK_BITS: tl.constexpr,
    ZERO_POINT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Unpack experts from `src_slot_ptr` of the staging ring into the
    `dst_slot_ptr` slots of `out_ptr`.

    Source and destination slots are separate because the ring is small and
    recycled: ring slot 3 can be destined for cache slot 57. The strides are
    passed rather than derived, since a ring row is one padded blob (packed
    weights, then scales) and so is wider than `rows * packed_row_stride`.

    Slot and row are folded into program_id(0): grid dims y and z cap at 65535,
    and `rows` (2*intermediate, or hidden) can exceed that.
    """
    pid = tl.program_id(0)
    i = pid // rows
    row = pid - i * rows
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < cols

    src_slot = tl.load(src_slot_ptr + i).to(tl.int64)
    dst_slot = tl.load(dst_slot_ptr + i).to(tl.int64)

    # PER_BYTE values per byte, lowest column in the lowest bits. Widen to int32
    # before shifting: uint8 arithmetic in Triton has been version-dependent.
    b = tl.load(
        packed_ptr
        + src_slot * packed_slot_stride
        + row * packed_row_stride
        + (col // PER_BYTE),
        mask=mask,
        other=0,
    ).to(tl.int32)
    q = ((b >> ((col % PER_BYTE) * BITS)) & MASK_BITS) - ZERO_POINT

    if GROUP_SIZE == 0:  # one scale per row
        s = tl.load(scale_ptr + src_slot * scale_slot_stride + row)
    else:
        s = tl.load(
            scale_ptr
            + src_slot * scale_slot_stride
            + row * scale_row_stride
            + (col // GROUP_SIZE),
            mask=mask,
            other=0.0,
        )

    val = q.to(tl.float32) * s.to(tl.float32)
    tl.store(
        out_ptr + (dst_slot * rows + row) * cols + col,
        val.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def dequant_experts_into_(
    packed: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    src_slots: torch.Tensor,
    group_size: int,
    num_bits: int = 4,
    dst_slots: torch.Tensor | None = None,
) -> None:
    """Dequantize `src_slots` of `packed`/`scale` into `dst_slots` of `out`.

    `packed` and `scale` are the aliases of a staging ring (see `blob_aliases`);
    `out` is the plain bf16 cache buffer the MoE kernel reads. Neither slot list
    need be contiguous — the fetch-on-demand path passes arbitrary eviction
    victims while the prefetch path passes a dense range. `dst_slots` defaults
    to `src_slots`, which is the in-place case the tests use.

    Launches on the caller's current stream, so it orders after the H2D copies
    that filled the ring without an explicit event.
    """
    n = src_slots.numel()
    if n == 0:
        return
    if dst_slots is None:
        dst_slots = src_slots
    elif dst_slots.numel() != n:
        raise ValueError(
            f"src_slots ({n}) and dst_slots ({dst_slots.numel()}) must name the "
            "same number of experts."
        )
    # Triton passes a tensor as a bare pointer, so anything the kernel does not
    # receive an explicit stride for is addressed as if it were dense. That
    # holds for the two slot vectors and for `out`; a strided one (a
    # `.expand()`, say) would be read or written at the wrong addresses and
    # quietly dequantize -- or overwrite -- the wrong experts. `packed` and
    # `scale` are exempt: their strides *are* passed below, which is what lets
    # them be padded aliases of a staging ring.
    if not src_slots.is_contiguous() or not dst_slots.is_contiguous():
        raise ValueError("src_slots and dst_slots must be contiguous.")
    if not out.is_contiguous():
        raise ValueError(
            "out must be contiguous; the kernel computes its offsets as "
            "(slot * rows + row) * cols + col."
        )

    num_slots, cols = out.shape[0], out.shape[-1]
    rows = out.numel() // (num_slots * cols)
    block = min(triton.next_power_of_2(cols), 512)
    # Fixed launch config on purpose: triton.autotune would run a benchmark
    # sweep (with a synchronize) inside the prefetch worker on a key miss.
    _dequant_experts_kernel[(n * rows, triton.cdiv(cols, block))](
        src_slots,
        dst_slots,
        packed,
        scale,
        out,
        rows,
        cols,
        packed.stride(0),
        packed.stride(1),
        scale.stride(0),
        scale.stride(1) if group_size > 0 else 1,
        GROUP_SIZE=group_size if group_size > 0 else 0,
        BITS=num_bits,
        PER_BYTE=8 // num_bits,
        MASK_BITS=(1 << num_bits) - 1,
        ZERO_POINT=1 << (num_bits - 1),
        BLOCK=block,
        num_warps=4,
    )
