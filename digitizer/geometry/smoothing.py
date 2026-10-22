"""
Corner-preserving curve smoothing for raster-derived polygon outlines
(follow-up to Milestone 5's "improve curves" feedback).

Why this is needed
--------------------
digitizer/segmentation/polygons.py builds a region's boundary by unioning
1x1 pixel squares and then running Shapely's Douglas-Peucker `simplify()`
on the result. That removes the unit-square *staircase*, but what's left is
still a plain polyline: a circular logo badge, a rounded letterform, or any
other originally-curved edge comes out as a many-sided (but still sharp-
cornered) polygon approximation, which reads as faceted/angular rather than
smoothly curved once stitched.

Why plain Chaikin smoothing isn't enough on its own
------------------------------------------------------
Chaikin's corner-cutting algorithm (repeatedly replacing each vertex with
two points 1/4 and 3/4 along its adjacent edges) is the standard way to turn
a polyline into a smooth curve -- but applying it uniformly to *every*
vertex would also round off genuinely sharp corners that are part of the
actual design (a rectangular tag's corners, a letter's serif, an arrow's
point), which is exactly the kind of over-smoothing a professional
digitizer avoids.

The fix: smooth *adaptively*, per vertex. For every vertex, the turn angle
between its incoming and outgoing edges is measured. A vertex that turns
only gently (close to a straight line) is treated as part of an
approximated curve -- exactly what a staircase-then-simplified circular arc
looks like close-up -- and gets corner-cut. A vertex that turns sharply
(close to a right angle or tighter) is treated as an intentional design
corner and is left completely untouched. Iterating this a couple of times
produces outlines that read as properly curved where the source was curved,
while keeping straight-edged shapes straight-edged.

Independent implementation (standard Chaikin corner-cutting plus an angle-
based gate on top) -- not derived from any third-party project.
"""

from __future__ import annotations

import math

Point = tuple[float, float]

DEFAULT_SHARP_ANGLE_DEG = 35.0


def _turn_angle_deg(prev: Point, cur: Point, nxt: Point) -> float:
    """Returns how sharply the path turns at `cur`, in degrees: 0 means
    dead straight through, 180 means a full reversal. Degenerate
    (zero-length) edges are treated as perfectly straight (angle 0) so they
    never block smoothing."""
    v1x, v1y = cur[0] - prev[0], cur[1] - prev[1]
    v2x, v2y = nxt[0] - cur[0], nxt[1] - cur[1]
    len1 = math.hypot(v1x, v1y)
    len2 = math.hypot(v2x, v2y)
    if len1 < 1e-9 or len2 < 1e-9:
        return 0.0
    cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (len1 * len2)))
    return math.degrees(math.acos(cos_theta))


def smooth_ring(
    ring: list[Point],
    iterations: int = 2,
    sharp_angle_deg: float = DEFAULT_SHARP_ANGLE_DEG,
) -> list[Point]:
    """Smooths a closed ring (first point == last point, Shapely's own
    convention) with adaptive, corner-preserving Chaikin cutting.

    Vertices that turn more sharply than `sharp_angle_deg` are kept exactly
    as-is; gentler vertices are corner-cut toward a smooth curve. Too few
    points to have a meaningful interior (a triangle or smaller) are
    returned unchanged -- there's nothing to smooth without risking
    collapsing the shape.
    """
    if len(ring) < 5:  # 4 unique points + closing repeat is the smallest
        return ring  # ring smoothing can meaningfully act on
    if ring[0] != ring[-1]:
        # Not actually closed (shouldn't happen for Shapely-sourced rings,
        # but stay defensive rather than silently producing a bad shape).
        return ring

    points = ring[:-1]  # drop the repeated closing point for the working set
    n = len(points)

    for _ in range(max(0, iterations)):
        if n < 4:
            break
        new_points: list[Point] = []
        for i in range(n):
            prev = points[(i - 1) % n]
            cur = points[i]
            nxt = points[(i + 1) % n]
            angle = _turn_angle_deg(prev, cur, nxt)
            if angle >= sharp_angle_deg:
                # Sharp, intentional corner -- preserve it exactly.
                new_points.append(cur)
                continue
            # Gentle turn -- almost certainly part of an approximated
            # curve. Cut the corner: replace `cur` with two points a
            # quarter and three-quarters of the way along its adjacent
            # edges (standard Chaikin ratios).
            q1 = (prev[0] + 0.75 * (cur[0] - prev[0]), prev[1] + 0.75 * (cur[1] - prev[1]))
            q2 = (cur[0] + 0.25 * (nxt[0] - cur[0]), cur[1] + 0.25 * (nxt[1] - cur[1]))
            new_points.append(q1)
            new_points.append(q2)
        points = new_points
        n = len(points)

    points.append(points[0])
    return points


def smooth_ring_group(
    rings: list[list[Point]],
    iterations: int = 2,
    sharp_angle_deg: float = DEFAULT_SHARP_ANGLE_DEG,
) -> list[list[Point]]:
    """Applies `smooth_ring` to every ring in a ring-group (outer boundary
    plus holes -- see digitizer/segmentation/polygons.py's `RingGroup`)."""
    return [smooth_ring(ring, iterations, sharp_angle_deg) for ring in rings]
