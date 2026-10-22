"""
Quality-improvement tests: arc-length rail resampling for curved satin
columns (digitizer/geometry/resample.py).

Without this, pairing two independently-traced rail curves index-for-index
(rail_a[i] with rail_b[i]) assumes they have matching point density, which
real traced curves essentially never do -- this is a big part of what made
curved satin columns look "unprofessional" (distorted width along the
curve) before this fix.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digitizer.geometry.resample import build_rungs_from_rails, point_at_arc_fraction
from digitizer.stitches.satin.generator import generate_satin_stitch


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def test_point_at_arc_fraction_endpoints():
    line = [(0, 0), (10, 0), (20, 0)]
    assert point_at_arc_fraction(line, 0.0) == (0, 0)
    assert point_at_arc_fraction(line, 1.0) == (20, 0)


def test_point_at_arc_fraction_midpoint_on_equal_segments():
    # total length 20 (two 10mm segments); the 50% mark (10mm in) falls
    # exactly on the shared vertex.
    line = [(0, 0), (10, 0), (20, 0)]
    mid = point_at_arc_fraction(line, 0.5)
    assert abs(mid[0] - 10) < 1e-9
    assert abs(mid[1] - 0) < 1e-9


def test_point_at_arc_fraction_lands_inside_a_segment():
    # total length 30 (segments of 10 and 20); the 50% mark (15mm in) falls
    # inside the second segment (10 -> 30), 5mm into it.
    line = [(0, 0), (10, 0), (30, 0)]
    mid = point_at_arc_fraction(line, 0.5)
    assert abs(mid[0] - 15) < 1e-9
    assert abs(mid[1] - 0) < 1e-9


def test_build_rungs_from_rails_matches_endpoints():
    rail_a = [(0, 0), (20, 0)]  # sparse: 2 points
    rail_b = [(0, 10), (5, 10), (10, 10), (15, 10), (20, 10)]  # dense: 5 points
    rungs = build_rungs_from_rails(rail_a, rail_b, n_rungs=5)
    assert len(rungs) == 10  # 5 rungs x 2 points each

    assert rungs[0] == (0, 0)
    assert rungs[1] == (0, 10)
    assert rungs[-2] == (20, 0)
    assert rungs[-1] == (20, 10)


def test_build_rungs_from_rails_proportional_not_index_based():
    # rail_a is traced with far fewer points than rail_b, but both rails
    # are the same straight 20mm line -- resampling at matching arc-length
    # fractions must still produce rungs with a consistent (here: zero)
    # perpendicular offset the whole way along, proving it's not just
    # pairing by raw index (which would badly mismatch point counts).
    rail_a = [(0, 0), (20, 0)]
    rail_b = [(0, 5), (2, 5), (6, 5), (14, 5), (18, 5), (20, 5)]  # uneven spacing, same line
    rungs = build_rungs_from_rails(rail_a, rail_b, n_rungs=9)
    rail_a_pts = rungs[0::2]
    rail_b_pts = rungs[1::2]
    for a, b in zip(rail_a_pts, rail_b_pts):
        assert abs(a[0] - b[0]) < 1e-6, "same arc-length fraction should land at the same x on both rails"
        assert abs(b[1] - 5) < 1e-9


def test_build_rungs_from_rails_feeds_generate_satin_stitch():
    # A gently curving column: rail_a and rail_b each traced with a
    # different number of points, fed through resampling into a working
    # satin column.
    rail_a = [(0, 0), (20, -3), (40, 0)]
    rail_b = [(0, 6), (10, 4), (20, 3), (30, 4), (40, 6)]
    rungs = build_rungs_from_rails(rail_a, rail_b, n_rungs=12)
    stitches = generate_satin_stitch(rungs, density_mm=2.0)
    assert stitches
    # alternating rails -- even indices near rail_a's y range, odd near rail_b's
    for i, (x, y) in enumerate(stitches):
        if i % 2 == 0:
            assert -3.5 <= y <= 0.5
        else:
            assert 2.5 <= y <= 6.5


def test_too_few_rungs_raises():
    try:
        build_rungs_from_rails([(0, 0)], [(0, 1)], n_rungs=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_empty_rail_raises():
    try:
        build_rungs_from_rails([], [(0, 1)], n_rungs=5)
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
