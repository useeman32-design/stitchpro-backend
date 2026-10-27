"""
Motif stitch generator (new stitch type follow-up: Bean/Motif/3D Puff/Applique).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md).

A "motif" (or decorative/ornamental) stitch tiles a small repeating shape
along a path -- the embroidery equivalent of a sewing machine's built-in
decorative stitch patterns (a chain of little diamonds, leaves, scallops,
or arrows running along a line), connected end-to-end by the stitching
thread itself. Real digitizers reach for it on long, thin, decorative
border/stem regions where a plain running or bean stitch would look too
plain -- see `digitizer.pipeline`'s MOTIF eligibility heuristic for exactly
which auto-detected regions get this treatment (long + thin bands) versus
BEAN (short + thin) or SATIN (ordinary stroke width).

Each repeated motif is a small closed shape defined in a normalized "unit
cell" (x spans roughly [-0.5, 0.5] along the local path direction, y spans
roughly [-0.5, 0.5] perpendicular to it), scaled to this call's requested
size, rotated to the path's local tangent at that point, translated to the
tick position, and stitched as its own tiny closed running-stitch loop.
Consecutive motifs are joined by a resampled running-stitch segment (never
a single long stitch, regardless of `spacing_mm`, so an oversized spacing
setting can never produce a stitch longer than DST's real 12.1mm hard cap)
rather than a jump, so the finished line reads as one continuous
ornamented thread the way a real motif/decorative border does.
"""

from __future__ import annotations

import math

from digitizer.stitches.running.generator import generate_running_stitch

Point = tuple[float, float]
Stitch = tuple[float, float, bool]  # (x, y, is_jump)

# Built-in repeating shapes, each a small closed polyline in unit-cell
# coordinates (see module docstring). Every pattern's last point equals its
# first so each repeat is a clean closed loop.
MOTIF_PATTERNS: dict[str, list[Point]] = {
    "diamond": [(0.0, -0.5), (0.35, 0.0), (0.0, 0.5), (-0.35, 0.0), (0.0, -0.5)],
    "leaf": [
        (-0.45, 0.0),
        (-0.2, -0.35),
        (0.15, -0.3),
        (0.45, 0.0),
        (0.15, 0.3),
        (-0.2, 0.35),
        (-0.45, 0.0),
    ],
    "scallop": [
        (-0.5, 0.0),
        (-0.3, -0.35),
        (0.0, -0.45),
        (0.3, -0.35),
        (0.5, 0.0),
        (-0.5, 0.0),
    ],
    "arrow": [
        (-0.5, 0.2),
        (0.1, 0.2),
        (0.1, 0.45),
        (0.5, 0.0),
        (0.1, -0.45),
        (0.1, -0.2),
        (-0.5, -0.2),
        (-0.5, 0.2),
    ],
}

DEFAULT_MOTIF_PATTERN = "diamond"


def _resample_with_tangent(points: list[Point], step_mm: float) -> list[tuple[Point, float]]:
    """Walks `points` (an open polyline) at ~`step_mm` arc-length intervals,
    returning `(position, tangent_angle_rad)` at each tick -- the tangent is
    the direction of the polyline segment the tick falls on, used to
    orient each motif repeat to the path's local direction.
    """
    if len(points) == 0:
        return []
    if len(points) == 1:
        return [(points[0], 0.0)]

    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = cumulative[-1]
    if total < 1e-9:
        return [(points[0], 0.0)]

    n = max(1, round(total / step_mm))
    out: list[tuple[Point, float]] = []
    seg_i = 0
    for k in range(n + 1):
        s = min(total, (k / n) * total)
        while seg_i < len(cumulative) - 2 and cumulative[seg_i + 1] < s:
            seg_i += 1
        a, b = points[seg_i], points[seg_i + 1]
        seg_len = cumulative[seg_i + 1] - cumulative[seg_i]
        t = 0.0 if seg_len < 1e-9 else (s - cumulative[seg_i]) / seg_len
        px = a[0] + (b[0] - a[0]) * t
        py = a[1] + (b[1] - a[1]) * t
        angle = math.atan2(b[1] - a[1], b[0] - a[0])
        out.append(((px, py), angle))
    return out


def generate_motif_stitch(
    geometry: list[Point],
    pattern: str = DEFAULT_MOTIF_PATTERN,
    spacing_mm: float = 4.0,
    scale: float = 1.0,
    stitch_length_mm: float = 3.0,
) -> list[Stitch]:
    """Convert `geometry` (an open polyline path to decorate) into motif
    stitch coordinates: a small `pattern` shape repeated every `spacing_mm`
    along the path, each sized `spacing_mm * scale * 0.9` (the 0.9 leaves a
    small visible gap between consecutive repeats rather than letting them
    touch/overlap), oriented to the path's local tangent, and connected to
    its neighbors by resampled running-stitch segments.

    Falls back to `MOTIF_PATTERNS[DEFAULT_MOTIF_PATTERN]` for an unknown
    `pattern` name rather than raising, so a stale/typo'd setting degrades
    gracefully instead of breaking a whole design's generation.
    """
    if spacing_mm <= 0:
        raise ValueError("spacing_mm must be > 0")
    if len(geometry) < 2:
        return [(x, y, i == 0) for i, (x, y) in enumerate(geometry)]

    unit_cell = MOTIF_PATTERNS.get(pattern, MOTIF_PATTERNS[DEFAULT_MOTIF_PATTERN])
    ticks = _resample_with_tangent(geometry, spacing_mm)
    size_mm = spacing_mm * max(0.1, scale) * 0.9

    out: list[Stitch] = []
    prev_end: Point | None = None
    first = True
    for (cx, cy), angle in ticks:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        shape_points: list[Point] = []
        for ux, uy in unit_cell:
            sx, sy = ux * size_mm, uy * size_mm
            rx = sx * cos_a - sy * sin_a
            ry = sx * sin_a + sy * cos_a
            shape_points.append((cx + rx, cy + ry))

        if prev_end is not None:
            connector = generate_running_stitch([prev_end, shape_points[0]], stitch_length_mm)
            # connector[0] == prev_end, already stitched as the previous
            # motif's last point -- skip it to avoid a duplicate stitch at
            # the same coordinate.
            for x, y in connector[1:]:
                out.append((x, y, False))

        shape_stitches = generate_running_stitch(shape_points, stitch_length_mm)
        for i, (x, y) in enumerate(shape_stitches):
            is_jump = first and i == 0
            out.append((x, y, is_jump))
            first = False
        prev_end = shape_stitches[-1] if shape_stitches else prev_end

    return out
