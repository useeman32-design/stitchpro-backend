"""
Satin column stitch generator (brief section 10).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md). The conceptual model (two
boundary "rails" plus perpendicular "rungs", density expressed as spacing
between crossing stitches rather than a stitch count, width derived from the
live perpendicular distance between rails rather than a fixed value) is
documented, concepts-only, in research/inkstitch-notes.md; the data shape
below and all of the stitch math are original.

Input shape
-----------
A satin column's `EmbroideryObject.geometry` is a flat list of points in
rail-A / rail-B alternating pairs:

    [A0, B0, A1, B1, A2, B2, ..., An, Bn]

geometry[0::2] is rail A, geometry[1::2] is rail B. Each (A_i, B_i) pair is
one "rung" -- a straight cross-line marking corresponding points on the two
rails. Consecutive rungs bound one section of the column; letting each
rung's two endpoints move independently (instead of requiring the two rails
to simply be parallel/equal-length curves) is what lets one column widen,
narrow, or curve along its length. A straight satin rectangle is just two
rungs: the two short ends of the rectangle.
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def _lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _expand_rung(a: Point, b: Point, amount: float) -> tuple[Point, Point]:
    """Push `a` and `b` apart along their connecting line by `amount` each,
    i.e. the rung (and so the column's width at that point) grows by
    `2 * amount` total. Used for pull compensation: fabric pulls inward as
    it's stitched, so the column is drawn slightly wider than the target
    finished size to compensate. A degenerate (coincident) rung is returned
    unchanged -- there's no direction to expand along."""
    if amount == 0:
        return a, b
    length = _dist(a, b)
    if length < 1e-9:
        return a, b
    ux = (b[0] - a[0]) / length
    uy = (b[1] - a[1]) / length
    return (a[0] - ux * amount, a[1] - uy * amount), (b[0] + ux * amount, b[1] + uy * amount)


def generate_satin_stitch(
    geometry: list[Point],
    density_mm: float = 0.4,
    pull_compensation_mm: float = 0.0,
) -> list[Point]:
    """
    Convert a satin column's rung-pair geometry (see module docstring) into
    zig-zag satin stitch coordinates.

    Guarantees:
    - Every input rung's two endpoints appear in the output (after pull
      compensation is applied, if any) -- a rung marks an intentional width
      change or curve point in the column and must not be smoothed over,
      the same corner-preservation guarantee `generate_running_stitch` makes
      for polyline vertices.
    - Between two consecutive rungs, crossing stitches are added so that no
      single stitch travels further than `density_mm` along either rail --
      density is measured as "distance between stitches along the rail
      direction", matching how density is actually specified on a real
      embroidery machine, not as a fixed stitch count. Because the two
      rails can have different lengths between the same two rungs (a
      tapering or curving column), the longer rail's length drives the
      step count so neither rail ever ends up under-stitched/too sparse.
    - Output alternates rail A / rail B at every step (A0, B0, A1, B1, ...)
      -- this *is* the satin stitch: the needle crosses from one rail to the
      other and back on every stitch.
    - `pull_compensation_mm` expands every rung outward (along the A-B line)
      before stitch generation, by that many mm on each side, so the
      finished (pulled-in) embroidery comes out closer to the intended size.
    """
    if density_mm <= 0:
        raise ValueError("density_mm must be > 0")
    if len(geometry) % 2 != 0:
        raise ValueError("geometry must have an even number of points (rail-A/rail-B pairs)")
    if len(geometry) < 4:
        raise ValueError("geometry must contain at least 2 rungs (4 points)")

    rail_a = geometry[0::2]
    rail_b = geometry[1::2]
    n_rungs = len(rail_a)

    # Apply pull compensation once per rung (each rung is shared by the
    # section before and after it, so this must happen before building
    # sections, not per-section, to avoid compensating the same rung twice).
    if pull_compensation_mm:
        expanded = [_expand_rung(rail_a[i], rail_b[i], pull_compensation_mm) for i in range(n_rungs)]
        rail_a = [a for a, _ in expanded]
        rail_b = [b for _, b in expanded]

    stitches: list[Point] = []
    for i in range(n_rungs - 1):
        a_start, b_start = rail_a[i], rail_b[i]
        a_end, b_end = rail_a[i + 1], rail_b[i + 1]

        travel = max(_dist(a_start, a_end), _dist(b_start, b_end))
        if travel < 1e-9:
            continue  # duplicate rung, skip (matches generate_running_stitch's
            # handling of duplicate points)

        steps = max(1, math.ceil(travel / density_mm))
        start_k = 0 if i == 0 else 1  # skip k=0 after the first section: it's
        # the same point as the previous section's k=steps, already emitted
        for k in range(start_k, steps + 1):
            t = k / steps
            stitches.append(_lerp(a_start, a_end, t))
            stitches.append(_lerp(b_start, b_end, t))

    return stitches
