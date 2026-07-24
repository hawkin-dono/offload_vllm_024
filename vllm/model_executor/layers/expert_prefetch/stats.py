# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-forward measurements the adaptive prefetch controller solves against."""

from dataclasses import dataclass, field

import torch


@dataclass
class CacheStats:
    """One forward pass of measurements, for `PrefetchController`.

    All host-side ints: the resolve decision runs on the CPU, so the hit/needed
    tallies are already on the host and never touch the device.
    """

    # Summed over the layers that ran against a prefetched buffer.
    hits: int = 0
    needed: int = 0
    # Hits the predictor's full top-`K` picks *would* have got. The accuracy the
    # Poisson model wants, uncontaminated by how much we chose to stage.
    reference_hits: int = 0
    staged: int = 0
    layers: int = 0
    truncated: bool = False
    # Unread (start, end) event pairs; drained lazily, a forward pass later.
    t_comp_events: list["_TCompSample"] = field(default_factory=list)


@dataclass
class ForwardStats:
    """What one forward pass measured, read off the device exactly once."""

    hits: int
    reference_hits: int
    needed: int
    staged: int
    layers: int
    truncated: bool
    t_comp_ms: list[float]
    # Per-expert copy time in ms, keyed by the precision (num_bits) that staged.
    # NATIVE_BITS is the bf16 copy; only widths a forward actually ran appear.
    t_e_ms: dict[int, float]


@dataclass
class _TCompSample:
    """Events bracketing one layer's compute, H2D excluded.

    `t_comp = elapsed(prev_ready, begin)` -- the compute-stream span from the
    previous layer's resolve finishing (`prev_ready`, after its wait and any
    on-demand fetches) to this layer's resolve starting (`begin`, before its
    wait). That interval is exactly one layer's compute -- the previous layer's
    MoE kernel plus this layer's attention and gating -- with both layers' H2D
    stalls left out, which is the bubble a single prefetch can hide behind.

    Anchored at the previous *resolve*, not at when the prefetch was issued: a
    prefetch driven by an `attn_input` predictor is issued a whole attention
    block before that resolve, so measuring from the issue point would fold the
    previous layer's attention into the window and roughly double `t_comp`.
    """

    begin: torch.cuda.Event
    prev_ready: torch.cuda.Event

    def ready(self) -> bool:
        return self.begin.query() and self.prev_ready.query()

    def elapsed_ms(self) -> float:
        return self.prev_ready.elapsed_time(self.begin)
