"""
Arc-length resampling helpers (brief section 9-10 support).

Independent implementation, standard arc-length parameterization -- not
derived from any third-party project.

Why this exists: a satin column is two independently-traced rail curves.
Real vector art/contour tracing essentially never produces two rails with
matching point density or spacing -- one rail might have 5 points, the
other 50, and their points don't correspond index-for-index. Pairing
`rail_a[i]` with `rail_b[i]` directly (as `generate_satin_stitch` requires
for its rung-pair geometry input) assumes they do correspond, and a naive
pairing on mismatched rails distorts the column's apparent width unevenly
along its length -- this is a large part of what makes a satin curve look
"unprofessional" instead of smoothly following the intended shape.
`build_rungs_from_rails` fixes this by resampling both rails at the same
set of fractional arc-length positions, regardless of how differently
they were originally traced.
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _cumulative_lengths(points: list[Point]) -> list[float]:
    lengths = [0.0]
    for i in range(1, len(points)):
        lengths.append(lengths[-1] + _dist(points[i - 1], points[i]))
    return lengths


def point_at_arc_fraction(points: list[Point], fraction: float) -> Point:
    """The point lying `fraction` (0..1) of the way along `points`' total
    path length, linearly interpolating within whichever segment that
    falls in. `fraction` is clamped to [0, 1]; a single-point input always
    returns that point."""
    if len(points) == 1:
        return points[0]
    fraction = min(max(fraction, 0.0), 1.0)
    lengths = _cumulative_lengths(points)
    total = lengths[-1]
    if total < 1e-9:
        return points[0]
    target = fraction * total
    for i in range(1, len(lengths)):
        if lengths[i] >= target - 1e-9:
            seg_len = lengths[i] - lengths[i - 1]
            t = 0.0 if seg_len < 1e-9 else (target - lengths[i - 1]) / seg_len
            a, b = points[i - 1], points[i]
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    return points[-1]


def build_rungs_from_rails(rail_a: list[Point], rail_b: list[Point], n_rungs: int = 20) -> list[Point]:
    """
    Resample two independently-traced rail curves (each just a polyline,
    arbitrary point density/spacing) into `n_rungs` evenly-arc-length-spaced
    rung pairs, in the alternating [A0, B0, A1, B1, ...] shape
    `generate_satin_stitch` expects as its `geometry` input.

    Both rails are walked by fraction of their own total length (0.0 at the
    start, 1.0 at the end), not by point index -- so a curve traced with 6
    points and one traced with 60 still produce rungs that correspond to
    the *same relative position* along each curve, keeping the column's
    width consistent and its direction correctly perpendicular-ish to both
    rails along its whole length.
    """
    if n_rungs < 2:
        raise ValueError("n_rungs must be >= 2")
    if not rail_a or not rail_b:
        raise ValueError("rail_a and rail_b must each have at least 1 point")

    rungs: list[Point] = []
    for k in range(n_rungs):
        t = k / (n_rungs - 1)
        rungs.append(point_at_arc_fraction(rail_a, t))
        rungs.append(point_at_arc_fraction(rail_b, t))
    return rungs
