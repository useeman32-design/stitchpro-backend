"""
Edge-walk satin border generator (fill/cross-stitch edge-quality follow-up).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md).

Why this exists
----------------
A tatami fill (digitizer/stitches/fill/generator.py) stitches a region by
scanning horizontal rows and stopping wherever each row crosses the region's
boundary. That boundary crossing happens at whatever x the scanline math
lands on -- it is never snapped to the actual smooth polygon edge -- so the
region's outer silhouette ends up looking like a staircase of row-ends
instead of one clean line, no matter how dense the fill rows are. This is
the single biggest visual tell of amateur/auto-digitized work vs. a
professional file: real digitizers always finish a filled region's outline
with a thin satin column (an "edge walk" or "bean" border) traced directly
along the smooth vector boundary, on top of the fill, which is what actually
reads as a crisp clean edge.

This module converts a closed boundary ring (the same ring data FILL/
CROSS_STITCH already carry as `EmbroideryObject.geometry`) into satin rung
geometry straddling that ring, ready to hand to
`digitizer.stitches.satin.generator.generate_satin_stitch`. The generated
column is centered ON the boundary line (half its width inside the fill,
half outside) rather than fully inside or outside it, which is what
actually hides the fill's jagged row-ends under the border stitch instead of
just drawing a second, separate line next to them.
"""

from __future__ import annotations

import math

from digitizer.stitches.satin.generator import generate_satin_stitch

Point = tuple[float, float]


def _ring_perimeter_and_cumulative(ring: list[Point]) -> tuple[list[Point], list[float], float]:
    """Treats `ring` as closed (wrapping from the last point back to the
    first even if that edge isn't already represented as a literal
    duplicate point) and returns (closed_points, cumulative_dist_at_each_point,
    total_perimeter)."""
    pts = list(ring)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]  # de-duplicate an explicit closing point; we re-close by wrapping below
    n = len(pts)
    cumulative = [0.0] * n
    total = 0.0
    for i in range(n):
        cumulative[i] = total
        nxt = pts[(i + 1) % n]
        total += math.hypot(nxt[0] - pts[i][0], nxt[1] - pts[i][1])
    return pts, cumulative, total


def _point_at(pts: list[Point], cumulative: list[float], total: float, s: float) -> Point:
    """Point on the closed polyline `pts` at arc-length `s` (wrapped into
    [0, total))."""
    n = len(pts)
    s = s % total
    # cumulative is sorted ascending; find the last index whose cumulative
    # distance is <= s (linear scan is fine -- rings are at most a few
    # hundred points for a typical digitized region).
    i = 0
    for j in range(n):
        if cumulative[j] <= s:
            i = j
        else:
            break
    a = pts[i]
    b = pts[(i + 1) % n]
    seg_start = cumulative[i]
    seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
    if seg_len < 1e-9:
        return a
    t = (s - seg_start) / seg_len
    t = max(0.0, min(1.0, t))
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def ring_to_satin_rungs(
    ring: list[Point],
    width_mm: float,
    step_mm: float = 1.5,
) -> list[Point]:
    """
    Resample closed boundary `ring` at ~`step_mm` arc-length intervals and
    build rail-A/rail-B rung pairs straddling it by `width_mm` total (half on
    each side of the original boundary line), in the flat
    `[A0, B0, A1, B1, ...]` shape `generate_satin_stitch` expects.

    Returns `[]` if the ring is degenerate (fewer than 3 points, or zero
    perimeter) -- callers should skip border generation for that ring
    entirely in that case, same as they already must for a degenerate fill
    polygon.

    The column is closed into a loop: the first rung is repeated at the end
    so the satin stitch visually wraps all the way around the shape instead
    of leaving a gap where it started.
    """
    if width_mm <= 0:
        raise ValueError("width_mm must be > 0")
    if step_mm <= 0:
        raise ValueError("step_mm must be > 0")
    if len(ring) < 3:
        return []

    pts, cumulative, total = _ring_perimeter_and_cumulative(ring)
    if total < 1e-6:
        return []

    n_steps = max(3, math.ceil(total / step_mm))
    half_w = width_mm / 2.0

    rungs: list[Point] = []
    first_pair: tuple[Point, Point] | None = None
    for k in range(n_steps):
        s = (k / n_steps) * total
        # Central difference over a small arc-length window gives a stable
        # tangent even where the resampled step lands exactly on a vertex.
        eps = min(step_mm * 0.25, total / 4)
        p_back = _point_at(pts, cumulative, total, s - eps)
        p_fwd = _point_at(pts, cumulative, total, s + eps)
        tx, ty = p_fwd[0] - p_back[0], p_fwd[1] - p_back[1]
        tlen = math.hypot(tx, ty)
        if tlen < 1e-9:
            # Degenerate tangent (e.g. a ring with a zero-length segment) --
            # fall back to the previous rung's normal direction, or skip.
            if not rungs:
                continue
            nx, ny = (rungs[-2][0] - rungs[-1][0]), (rungs[-2][1] - rungs[-1][1])
            nlen = math.hypot(nx, ny)
            if nlen < 1e-9:
                continue
            nx, ny = nx / nlen, ny / nlen
        else:
            # Perpendicular to the tangent -- straddles the boundary evenly.
            nx, ny = -ty / tlen, tx / tlen

        p = _point_at(pts, cumulative, total, s)
        a = (p[0] + nx * half_w, p[1] + ny * half_w)
        b = (p[0] - nx * half_w, p[1] - ny * half_w)
        rungs.append(a)
        rungs.append(b)
        if first_pair is None:
            first_pair = (a, b)

    if first_pair is None or len(rungs) < 4:
        return []

    rungs.append(first_pair[0])
    rungs.append(first_pair[1])
    return rungs


def rings_to_border_satin_stitches(
    rings: list[list[Point]],
    border_mm: float,
    step_mm: float = 1.5,
    density_mm: float = 0.4,
) -> list[tuple[float, float, bool]]:
    """Independent closed satin loop traced around every ring in `rings`
    (index 0 = outer boundary, 1: = holes), `border_mm` wide, as `(x, y,
    is_jump)` triples -- one disconnected loop per ring, so holes get their
    own clean inner edge instead of being joined to the outer border by a
    visible long jump (see `ring_to_satin_rungs`'s docstring for why).

    Factored out of dispatch.py's original FILL/CROSS_STITCH-only
    `_border_stitches` helper so the exact same "trace a clean satin line
    along this shape's real vector boundary" technique can be reused by
    PUFF_3D's foam edge cap and APPLIQUE's raw-edge-covering border
    (stitches/puff/generator.py, stitches/applique/generator.py) without
    duplicating this loop in three places. Returns `[]` if `border_mm <= 0`
    (0 = no border, same convention `StitchSettings.border_width_mm`
    already uses for FILL/CROSS_STITCH).
    """
    if border_mm <= 0:
        return []
    out: list[tuple[float, float, bool]] = []
    for ring in rings:
        rungs = ring_to_satin_rungs(ring, border_mm, step_mm=step_mm)
        if len(rungs) < 4:
            continue
        points = generate_satin_stitch(rungs, density_mm=density_mm)
        out.extend((x, y, i == 0) for i, (x, y) in enumerate(points))
    return out
