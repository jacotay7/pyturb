"""Shared row-ring-buffer bookkeeping for non-periodic extruders."""

from __future__ import annotations

from math import floor
from typing import Any, Tuple


def compact_row_ring(
    buffer: Any,
    base: int,
    fill: int,
    stencil_rows: int,
    first_readable_row: float,
) -> Tuple[int, int]:
    """Recycle rows while preserving the virtual-row recurrence invariant.

    ``buffer[i]`` represents virtual row ``base + i``. The readable window can
    be far ahead of rows already extruded, so compaction is bounded by both its
    first row and the final ``stencil_rows`` needed to synthesize the next row.
    The function works with NumPy and CuPy arrays and returns ``(base, fill)``.
    """
    target_bound = int(floor(first_readable_row)) - 1
    stencil_bound = base + fill - stencil_rows
    keep_from = min(target_bound, stencil_bound) - base
    keep_from = max(1, keep_from)
    keep = fill - keep_from
    buffer[:keep] = buffer[keep_from:fill].copy()
    return base + keep_from, keep
