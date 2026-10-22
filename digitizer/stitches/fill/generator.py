"""
Fill (tatami) stitch generator (brief section 11).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md). The conceptual model (parallel
rows at a chosen angle computed on the *unrotated* polygon then rotated into
place, rows staggered in phase to avoid a visible straight seam) is
documented, concepts-only, in research/inkstitch-notes.md; the geometry/
stitch math below is original. Row/polygon boolean work (buffering for pull
compensation, line-polygon intersection for the actual scanline) uses
Shapely (BSD-3+LGPL-dynamic, see docs/THIRD_PARTY_LICENSES.md) purely as a
geometry primitive -- same category of use as pyembroidery for formats.

Unlike running stitch and satin (each always one single connected stitch
run), a fill region can be concave, which means a single scan row can cross
the polygon boundary more than twice -- producing more than one disconnected
span on that row. The needle must *jump* (not stitch) between such spans, so
this generator returns `(x, y, is_jump)` triples instead of bare points:
`is_jump=True` means "lift the needle and jump here, starting a new
disconnected run"; `is_jump=False` means "stitch here, connected to the
previous point."
"""

from __future__ import annotations

import math

from shapely.affinity import rotate
from shapely.geometry import Polygon

Point = tuple[float, float]
Stitch = tuple[float, float, bool]  # (x, y, is_jump)


def _extract_spans(intersection) -> list[tuple[float, float]]:
    """Normalize a Shapely polygon/line intersection result (which can be
    empty, a single LineString, a MultiLineString, or a GeometryCollection
    mixing LineStrings with zero-width Points where the scan line only
    grazes a vertex) into a sorted, de-overlapped list of (min_x, max_x)
    spans along that row."""
    raw: list[tuple[float, float]] = []
    geoms = list(intersection.geoms) if hasattr(intersection, "geoms") else [intersection]
    for g in geoms:
        if g.is_empty:
            continue
        if g.geom_type == "LineString":
            xs = [c[0] for c in g.coords]
            if xs:
                raw.append((min(xs), max(xs)))
        elif g.geom_type == "MultiLineString":
            for line in g.geoms:
                xs = [c[0] for c in line.coords]
                if xs:
                    raw.append((min(xs), max(xs)))
        # Point/MultiPoint (tangent touches) and anything else carry no
        # stitchable width -- ignored.

    raw.sort()
    merged: list[tuple[float, float]] = []
    for span in raw:
        if merged and span[0] <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged.append(span)
    return [(a, b) for a, b in merged if b - a > 1e-6]


def _subdivide_span(
    x0: float, x1: float, y: float, stitch_length_mm: float, phase_mm: float, reverse: bool
) -> list[Point]:
    """Stitch one row span from x0 to x1 (x0 < x1), stepping by at most
    `stitch_length_mm`. Both span endpoints are always included (a span
    boundary is a real polygon-edge crossing and must not be smoothed over,
    same corner-preservation principle as running stitch's vertices).
    `reverse=True` walks the span from x1 back to x0, for boustrophedon
    (snake) row ordering.

    `phase_mm` staggers alternating rows (to avoid a visible straight seam
    running across them) by starting the stitch grid `phase_mm` in from x0
    instead of exactly at x0, then stepping by `stitch_length_mm` as usual.
    Note this is NOT implemented as "shift every interior point over by
    phase_mm": doing that while keeping both original endpoints fixed can
    push the gap at one end past `stitch_length_mm` (whichever end the
    shift points away from). Building a phase-shifted grid from scratch
    keeps every internal gap at exactly `stitch_length_mm` and only the
    first (`phase_mm`) and last (whatever remains) gaps different -- and
    the leftover final gap is explicitly split if it would otherwise exceed
    `stitch_length_mm`.
    """
    span_len = x1 - x0
    if span_len < 1e-9:
        return [(x0, y)]

    if not phase_mm:
        n_steps = max(1, math.ceil(span_len / stitch_length_mm))
        pts = [(x0 + span_len * (k / n_steps), y) for k in range(n_steps + 1)]
        return list(reversed(pts)) if reverse else pts

    phase_mm = min(phase_mm, stitch_length_mm, span_len)
    positions = [x0]
    cur = x0 + phase_mm
    while cur < x1 - 1e-9:
        positions.append(cur)
        cur += stitch_length_mm
    if x1 - positions[-1] > 1e-9:
        positions.append(x1)

    # Fix up the final gap if the leftover distance exceeds the bound.
    if len(positions) >= 2 and positions[-1] - positions[-2] > stitch_length_mm + 1e-9:
        start, end = positions[-2], positions[-1]
        extra_steps = max(1, math.ceil((end - start) / stitch_length_mm))
        positions = positions[:-2] + [start + (end - start) * (k / extra_steps) for k in range(extra_steps + 1)]

    pts = [(p, y) for p in positions]
    return list(reversed(pts)) if reverse else pts


def generate_fill_stitch(
    polygon_points: list[Point],
    stitch_length_mm: float = 3.0,
    density_mm: float = 0.4,
    angle_deg: float = 0.0,
    pull_compensation_mm: float = 0.0,
) -> list[Stitch]:
    """
    Convert a simple (non-self-intersecting) polygon into tatami fill
    stitches.

    Guarantees:
    - Rows run parallel to `angle_deg` (0 = horizontal), spaced `density_mm`
      apart -- computed by rotating the polygon by `-angle_deg` so rows are
      horizontal in that working frame (a plain scanline), then rotating
      every resulting stitch back by `+angle_deg` at the end. Much simpler
      and more robust than trying to scan along an arbitrary angle directly.
    - Each row is split into one or more spans by intersecting it with the
      polygon -- a concave polygon can produce multiple disjoint spans per
      row, each becoming its own disconnected stitch run (`is_jump=True` at
      its first point).
    - No single stitch within a span exceeds `stitch_length_mm`; both
      endpoints of every span are always exact stitch points.
    - Alternating rows are stitched in opposite directions (boustrophedon /
      "snake" ordering) to keep the needle's travel between rows short, and
      alternating rows' interior stitches are phase-shifted by half a
      stitch length to avoid a visible straight "valley" seam running
      across the rows.
    - `pull_compensation_mm` expands the fill area outward (a uniform
      polygon buffer) before scanning, so the finished (pulled-in)
      embroidery comes out closer to the intended size.
    """
    if stitch_length_mm <= 0:
        raise ValueError("stitch_length_mm must be > 0")
    if density_mm <= 0:
        raise ValueError("density_mm must be > 0")
    if len(polygon_points) < 3:
        raise ValueError("polygon must have at least 3 points")

    polygon = Polygon(polygon_points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)  # standard Shapely fix for minor ring issues
    if polygon.is_empty or polygon.area < 1e-9:
        raise ValueError("polygon is degenerate (zero area)")

    if pull_compensation_mm:
        polygon = polygon.buffer(pull_compensation_mm, join_style=2)

    pivot = polygon.centroid
    working = rotate(polygon, -angle_deg, origin=pivot, use_radians=False)

    min_x, min_y, max_x, max_y = working.bounds
    if max_y <= min_y:
        return []

    # Center the rows within the shape's height rather than anchoring the
    # first row to the very bottom edge (standard tatami behavior).
    n_rows = max(1, math.floor((max_y - min_y) / density_mm) + 1)

    rotated_stitches: list[Stitch] = []
    for row_index in range(n_rows):
        y = min_y + density_mm / 2 + row_index * density_mm
        if y > max_y:
            break
        line_coords = [(min_x - 1.0, y), (max_x + 1.0, y)]
        from shapely.geometry import LineString

        spans = _extract_spans(working.intersection(LineString(line_coords)))
        if not spans:
            continue

        reverse_row = row_index % 2 == 1
        phase = stitch_length_mm / 2 if reverse_row else 0.0
        ordered_spans = list(reversed(spans)) if reverse_row else spans

        for span_start, span_end in ordered_spans:
            run = _subdivide_span(span_start, span_end, y, stitch_length_mm, phase, reverse_row)
            for i, (x, y2) in enumerate(run):
                rotated_stitches.append((x, y2, i == 0))

    # Rotate every stitch back from the working frame to the original space.
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
