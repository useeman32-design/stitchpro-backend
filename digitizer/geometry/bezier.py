"""
Small geometry helper: flatten a cubic bezier curve into a polyline so the
running/satin/fill stitch generators (which all operate on straight-line
polylines) can handle curved input paths from SVG/vector artwork.

Independent implementation, standard De Casteljau subdivision -- not derived
from any third-party project.
"""

from __future__ import annotations

Point = tuple[float, float]


def flatten_cubic_bezier(
    p0: Point, p1: Point, p2: Point, p3: Point, segments: int = 16
) -> list[Point]:
    """Sample a cubic bezier (p0=start, p1/p2=control points, p3=end) into
    `segments` straight-line pieces. Returns `segments + 1` points including
    both endpoints."""
    if segments < 1:
        raise ValueError("segments must be >= 1")
    pts: list[Point] = []
    for i in range(segments + 1):
        t = i / segments
        mt = 1 - t
        x = (
            mt**3 * p0[0]
            + 3 * mt**2 * t * p1[0]
            + 3 * mt * t**2 * p2[0]
            + t**3 * p3[0]
        )
        y = (
            mt**3 * p0[1]
            + 3 * mt**2 * t * p1[1]
            + 3 * mt * t**2 * p2[1]
            + t**3 * p3[1]
        )
        pts.append((x, y))
    return pts
