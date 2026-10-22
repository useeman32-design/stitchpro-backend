"""
Cross-stitch generator (follow-up feature: "we should also have cross
stitches if the design requires it" -- a selectable stitch style alongside
tatami FILL for regions where a cross-stitch texture is the better/intended
look, e.g. a deliberately pixelated or sampler-style design).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md). Conceptually this mirrors real
hand/machine cross-stitch: a grid of evenly spaced cells, each one stitched
as two crossing diagonals (bottom-left -> top-right, then top-left ->
bottom-right), clipped to whatever polygon region it's filling -- the same
rings[0]=outer/rings[1:]=holes contract digitizer/stitches/fill/generator.py
already uses, so it slots into the exact same place in the pipeline
(dispatch.py, routing, DST/points export) with no special-casing needed
anywhere else.
"""

from __future__ import annotations

import math

from shapely.affinity import rotate
from shapely.geometry import Point as ShapelyPoint, Polygon

Point = tuple[float, float]
Stitch = tuple[float, float, bool]  # (x, y, is_jump)


def generate_cross_stitch(
    rings: list[list[Point]],
    cell_mm: float = 2.0,
    angle_deg: float = 0.0,
    pull_compensation_mm: float = 0.0,
) -> list[Stitch]:
    """
    Converts a polygon-with-holes into a grid of cross-stitches.

    `rings[0]` is the outer boundary; `rings[1:]` (zero or more) are holes,
    identical contract to `generate_fill_stitch`. A grid cell is included
    only if its *center* point lies inside the polygon (standard "pixel
    art" cross-stitch convention: a cell counts if its representative point
    is inside the shape, which naturally and cheaply handles concave
    boundaries and holes without any special casing).

    Guarantees:
    - Grid cells are `cell_mm` square, aligned to `angle_deg` (0 = axis-
      aligned), computed the same way `generate_fill_stitch` handles
      `angle_deg`: rotate the polygon into a working frame, build an
      axis-aligned grid there, then rotate the resulting stitches back.
    - Cells are visited in boustrophedon (snake) row order to keep travel
      between consecutive cells short, same principle as the fill
      generator's row ordering.
    - Each included cell produces exactly two diagonal stitches (four
      points) with a needle jump between them (the thread must lift to move
      from the end of one diagonal to the start of the crossing one) --
      `is_jump=True` marks that transition and the very first point of each
      new disconnected run, matching the fill generator's `(x, y, is_jump)`
      contract.
    - `pull_compensation_mm` expands the outer boundary and shrinks each
      hole, identical semantics to `generate_fill_stitch`.
    """
    if cell_mm <= 0:
        raise ValueError("cell_mm must be > 0")
    if not rings:
        raise ValueError("rings must contain at least an outer boundary")
    for ring in rings:
        if len(ring) < 3:
            raise ValueError("every ring must have at least 3 points")

    polygon = Polygon(shell=rings[0], holes=rings[1:])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area < 1e-9:
        raise ValueError("polygon is degenerate (zero area, or holes fully cover the outer boundary)")

    if pull_compensation_mm:
        polygon = polygon.buffer(pull_compensation_mm, join_style=2)
        if polygon.is_empty or polygon.area < 1e-9:
            raise ValueError("pull_compensation_mm made the polygon degenerate (holes likely closed up)")

    pivot = polygon.centroid
    working = rotate(polygon, -angle_deg, origin=pivot, use_radians=False)

    min_x, min_y, max_x, max_y = working.bounds
    if max_x <= min_x or max_y <= min_y:
        return []

    n_cols = max(1, math.ceil((max_x - min_x) / cell_mm))
    n_rows = max(1, math.ceil((max_y - min_y) / cell_mm))

    rotated_stitches: list[Stitch] = []
    for row in range(n_rows):
        cy = min_y + cell_mm / 2 + row * cell_mm
        if cy > max_y:
            break
        cols = range(n_cols - 1, -1, -1) if row % 2 == 1 else range(n_cols)
        for col in cols:
            cx = min_x + cell_mm / 2 + col * cell_mm
            if cx > max_x:
                continue
            if not working.contains(ShapelyPoint(cx, cy)):
                continue
            half = cell_mm / 2
            bl = (cx - half, cy - half)
            tr = (cx + half, cy + half)
            tl = (cx - half, cy + half)
            br = (cx + half, cy - half)
            rotated_stitches.append((bl[0], bl[1], True))
            rotated_stitches.append((tr[0], tr[1], False))
            rotated_stitches.append((tl[0], tl[1], True))
            rotated_stitches.append((br[0], br[1], False))

    if not rotated_stitches:
        return []

    if angle_deg == 0:
        return rotated_stitches

    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    px, py = pivot.x, pivot.y
    out: list[Stitch] = []
    for x, y, is_jump in rotated_stitches:
        dx, dy = x - px, y - py
        rx = dx * cos_a - dy * sin_a + px
        ry = dx * sin_a + dy * cos_a + py
        out.append((rx, ry, is_jump))
    return out
