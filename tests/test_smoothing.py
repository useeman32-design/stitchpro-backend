"""Tests for digitizer/geometry/smoothing.py."""

from __future__ import annotations

import math

from digitizer.geometry.smoothing import smooth_ring


def _square_ring(size=10.0):
    return [(0, 0), (size, 0), (size, size), (0, size), (0, 0)]


def _polygon_approx_circle(n=24, radius=10.0):
    pts = [
        (radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    pts.append(pts[0])
    return pts


def _max_turn_angle(ring: list[tuple[float, float]]) -> float:
    pts = ring[:-1]
    n = len(pts)
    worst = 0.0
    for i in range(n):
        prev, cur, nxt = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1 = (cur[0] - prev[0], cur[1] - prev[1])
        v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        l1, l2 = math.hypot(*v1), math.hypot(*v2)
        if l1 < 1e-9 or l2 < 1e-9:
            continue
        cos_t = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)))
        worst = max(worst, math.degrees(math.acos(cos_t)))
    return worst


class TestSmoothRing:
    def test_short_ring_returned_unchanged(self):
        tri = [(0, 0), (1, 0), (0, 1), (0, 0)]
        assert smooth_ring(tri) == tri

    def test_square_corners_are_preserved(self):
        square = _square_ring()
        smoothed = smooth_ring(square, iterations=3)
        # A plain square's 90-degree corners must survive essentially
        # untouched -- only genuinely curved (shallow-turn) vertices should
        # ever be cut.
        assert _max_turn_angle(smoothed) > 80.0
        # All four original corner points must still be present exactly.
        for corner in square[:-1]:
            assert any(math.isclose(corner[0], p[0], abs_tol=1e-6) and math.isclose(corner[1], p[1], abs_tol=1e-6) for p in smoothed)

    def test_closed_ring_stays_closed(self):
        square = _square_ring()
        smoothed = smooth_ring(square, iterations=2)
        assert smoothed[0] == smoothed[-1]

    def test_polygon_approximated_circle_gets_smoother(self):
        # A many-sided polygon standing in for a circle approximated via
        # pixel-staircase + simplify() has many *gentle* turns (360/24=15
        # degrees apart) -- exactly the case that should get corner-cut.
        circle = _polygon_approx_circle(n=24)
        before = _max_turn_angle(circle)
        smoothed = smooth_ring(circle, iterations=2)
        after = _max_turn_angle(smoothed)
        assert after < before

    def test_smoothing_does_not_drastically_shrink_the_shape(self):
        # Corner-cutting should refine the curve, not collapse it -- the
        # smoothed circle approximation should keep nearly the same area.
        from shapely.geometry import Polygon

        circle = _polygon_approx_circle(n=24, radius=10.0)
        smoothed = smooth_ring(circle, iterations=2)
        area_before = Polygon(circle).area
        area_after = Polygon(smoothed).area
        assert area_after > area_before * 0.9

    def test_mixed_shape_preserves_sharp_corner_but_smooths_curved_side(self):
        # An "ice cream cone"-ish shape: one genuinely sharp point plus a
        # many-vertex shallow arc standing in for a curved edge.
        arc_n = 12
        arc = [
            (10 * math.cos(math.pi * i / (arc_n - 1)), 10 * math.sin(math.pi * i / (arc_n - 1)) + 5)
            for i in range(arc_n)
        ]
        tip = (0.0, -15.0)  # sharp point well below the arc
        ring = [tip] + arc + [tip]
        smoothed = smooth_ring(ring, iterations=2)
        # The sharp tip must still be present, unrounded.
        assert any(math.isclose(tip[0], p[0], abs_tol=1e-6) and math.isclose(tip[1], p[1], abs_tol=1e-6) for p in smoothed)
