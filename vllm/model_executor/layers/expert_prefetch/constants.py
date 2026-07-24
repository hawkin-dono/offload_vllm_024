# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants for the ping-pong expert cache and its prefetch path."""

EMPTY_SLOT = -1

# Bit width standing for the native (bf16) originals. Always a prefetch
# candidate, costs no extra memory, and takes the plain per-slot copy path.
NATIVE_BITS = 16

# Experts a dequant ring holds before recycling. Sized in bytes at the widest
# supported width so one ring serves int8/int4/int2 alike.
DEQUANT_RING_SLOTS = 16
