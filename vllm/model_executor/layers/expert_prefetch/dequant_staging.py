# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU dequant ring: packed experts land here before being unpacked to bf16."""

import torch

from vllm.model_executor.layers.expert_prefetch.constants import DEQUANT_RING_SLOTS
from vllm.model_executor.layers.expert_prefetch.expert_quant import (
    MAX_EXPERT_QUANT_BITS,
    QuantBlobLayout,
    blob_aliases,
)


class DequantStaging:
    """Small GPU ring the packed experts land in before being unpacked.

    Quantization only ever shrinks what crosses PCIe; the MoE kernel still reads
    bf16, so every staged expert has to be dequantized into an `ExpertBuffer`
    slot anyway. That makes the packed bytes *transient* -- alive only between
    an expert's H2D copy and the dequant kernel that reads it -- so they do not
    need a slot per cache slot, and they do not need a buffer per bit width.
    One ring of `num_slots` experts, sized in bytes at the widest supported
    width, serves every resident width at once: at 128 cache slots that is
    ~150 MiB instead of the ~2 GiB a full-size buffer per width would cost.

    A fetch walks its experts a ring-full at a time. Copies and dequant are
    issued on the *same* stream, so the copies of chunk k+1 are ordered behind
    the dequant of chunk k that reads those ring slots -- the write-after-read
    hazard is handled by stream order alone, with no events to get wrong.

    One ring cannot be shared by the whole cache, though: `ExpertCache.prefetch`
    runs on a side stream while `ExpertCache.resolve` fetches misses on the
    compute stream, and the two overlap in time. Each path owns one.
    """

    def __init__(self, name: str, num_slots: int = DEQUANT_RING_SLOTS):
        self.name = name
        self.num_slots = num_slots
        # Packed blobs, keyed by param name: (num_slots, blob_bytes) uint8,
        # sized at the widest bit width.
        self.blobs: dict[str, torch.Tensor] = {}
        # {name: {num_bits: landing}} -- the blob rows trimmed to what a store
        # of that width occupies, so an H2D copy is one contiguous row-sized
        # transfer. A narrower store leaves the tail of each ring row untouched.
        self.landing: dict[str, dict[int, torch.Tensor]] = {}
        # {name: {num_bits: (qweight, scale)}} -- strided aliases of the blob,
        # built once here so a fetch never re-derives them.
        self._aliases: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        # Ring slot indices, sliced per chunk; kept on device so a fetch does
        # not have to build one.
        self.slot_index: torch.Tensor = torch.empty(0, dtype=torch.int32)
        # Staging pool for per-chunk destination indices (`dst_index`): a few
        # pinned rows and their device mirror, rotated per call.
        self._dst_host: torch.Tensor | None = None
        self._dst_dev: torch.Tensor | None = None
        self._dst_events: list[torch.cuda.Event] = []
        self._dst_cursor = 0

    def allocate(
        self,
        shapes: dict[str, tuple[int, int, int]],
        bits: tuple[int, ...],
        device: torch.device,
    ) -> None:
        """Size the ring from `shapes` ({name: (rows, cols, num_groups)}).

        Rows/cols/num_groups are the same across widths (they depend on the
        weight shape and group size, not the bit width), so the ring is built
        once at `MAX_EXPERT_QUANT_BITS` and every resident width's landing and
        aliases address the same storage.
        """
        for name, (rows, cols, num_groups) in shapes.items():
            widest = QuantBlobLayout(
                rows=rows,
                cols=cols,
                num_bits=MAX_EXPERT_QUANT_BITS,
                num_groups=num_groups,
            )
            blob = torch.zeros(
                (self.num_slots, widest.blob_bytes), dtype=torch.uint8, device=device
            )
            self.blobs[name] = blob
            self.landing[name] = {}
            self._aliases[name] = {}
            for num_bits in bits:
                layout = QuantBlobLayout(
                    rows=rows, cols=cols, num_bits=num_bits, num_groups=num_groups
                )
                self.landing[name][num_bits] = blob[:, : layout.blob_bytes]
                self._aliases[name][num_bits] = blob_aliases(blob, layout)
        self.slot_index = torch.arange(self.num_slots, dtype=torch.int32, device=device)
        pool = 4
        self._dst_host = torch.empty(
            (pool, self.num_slots), dtype=torch.int32, pin_memory=True
        )
        self._dst_dev = torch.empty(
            (pool, self.num_slots), dtype=torch.int32, device=device
        )
        self._dst_events = [torch.cuda.Event(blocking=True) for _ in range(pool)]
        self._dst_cursor = 0

    def dst_index(self, slots: list[int]) -> torch.Tensor:
        """Device int32 view of `slots`, staged without a stream sync.

        Replaces `torch.tensor(slots, device=...)`, whose pageable H2D
        synchronizes the current stream -- on the on-demand path that is the
        compute stream, so every cache miss stalled the host until compute
        drained. Rows rotate through a small pinned pool; a row is rewritten
        only after the copy that last read it completed (the event sync, which
        is all but always already done: it was recorded a few chunks of copies
        and dequant kernels ago).
        """
        assert self._dst_host is not None and self._dst_dev is not None
        row = self._dst_cursor % len(self._dst_events)
        self._dst_cursor += 1
        event = self._dst_events[row]
        event.synchronize()
        n = len(slots)
        host = self._dst_host[row, :n]
        host.numpy()[:] = slots
        dev = self._dst_dev[row, :n]
        dev.copy_(host, non_blocking=True)
        event.record()
        return dev

    def aliases(self, name: str, num_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The (qweight, scale) views of `name`'s ring at `num_bits`."""
        return self._aliases[name][num_bits]

    def landing_for(self, name: str, num_bits: int) -> torch.Tensor:
        """The landing region of `name`'s ring at `num_bits`."""
        return self.landing[name][num_bits]

    def nbytes(self) -> int:
        return sum(blob.numel() for blob in self.blobs.values())
