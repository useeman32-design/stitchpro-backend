"""
Running stitch generator (brief section 9).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md). Concept only: a corner must get an
explicit stitch placed exactly at the vertex, not smoothed over.
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def generate_running_stitch(
    points: list[Point],
    stitch_length_mm: float = 3.0,
) -> list[Point]:
    """
    Convert a polyline (already-flattened -- see geometry/bezier.py to flatten
    curves first) into running-stitch coordinates.

    Guarantees:
    - The first and last input points are always present in the output
      (explicit start/end points, per brief section 9).
    - Every intermediate vertex of the input polyline is always present in
      the output, regardless of stitch length -- this is what makes corners
      and sharp turns stitch correctly instead of being cut short.
    - Each segment between two vertices is subdivided using `ceil(seg_len /
      stitch_length_mm)` evenly-spaced steps, so no individual stitch ever
      exceeds `stitch_length_mm` (important: DST hard-caps stitch length at
      12.1mm) while staying as close to the target length as possible and
      never leaving one very short leftover stitch at a segment's end.
    """
    if stitch_length_mm <= 0:
        raise ValueError("stitch_length_mm must be > 0")
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [points[0]]

    stitches: list[Point] = [points[0]]
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len < 1e-9:
            continue  # duplicate point, skip
        n_steps = max(1, math.ceil(seg_len / stitch_length_mm))
        for k in range(1, n_steps + 1):
            t = k / n_steps
            stitches.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    return stitches
