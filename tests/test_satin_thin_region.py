"""
Automatic SATIN classification for thin/elongated regions -- see
digitizer/stitches/satin/thin_region.py's module docstring for the problem
this solves and the deliberately conservative scope (straight/hole-free
thin shapes only; curved shapes and anything with a hole must safely fall
back to FILL, never produce a broken/ambiguous satin column).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digitizer.stitches.satin.generator import generate_satin_stitch
from digitizer.stitches.satin.thin_region import (
    generate_satin_rungs_for_thin_region,
    is_satin_eligible,
)


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _straight_bar(length=20.0, width=2.0):
    """A thin, straight, axis-aligned rectangle -- the simplest possible
    satin candidate (a letter "I"'s stem, a ruled line, a badge border
    bar)."""
    return [(0, 0), (length, 0), (length, width), (0, width)]


def _diagonal_bar(length=20.0, width=2.0):
    """Same shape as `_straight_bar` but rotated 37 degrees, to confirm
    classification isn't accidentally axis-dependent."""
    angle = math.radians(37)
    ca, sa = math.cos(angle), math.sin(angle)

    def rot(p):
        x, y = p
        return (x * ca - y * sa, x * sa + y * ca)

    return [rot(p) for p in _straight_bar(length, width)]


def _blobby_square(size=10.0):
    """Not elongated -- should never be satin-classified."""
    return [(0, 0), (size, 0), (size, size), (0, size)]


def _ring_with_hole():
    """An annulus-like square-with-a-square-hole -- has a hole, so must be
    rejected outright regardless of its outer shape's proportions."""
    outer = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(5, 5), (5, 15), (15, 15), (15, 5)]
    return [outer, hole]


def _lozenge(length=20.0, half_width=2.0):
    """A thin diamond/lozenge that tapers to an actual point at both ends
    of its long axis -- exactly the shape a jagged, despeckled glyph-stroke
    fragment from a real photo-pipeline digitize often collapses to (see
    the regression this guards: a scan line landing exactly on or past a
    tapering tip legitimately finds zero crossings there, and that must
    skip just that one rung rather than reject the whole region)."""
    mid = length / 2
    return [(0, 0), (mid, half_width), (length, 0), (mid, -half_width)]


def _horseshoe():
    """A tightly curved "C"-like shape: a thick ring with a wedge cut out of
    one side. A straight perpendicular scan along any single global axis
    will cross this shape's boundary more than twice at many positions
    (both arms of the "C"), so this must trigger the conservative bail-out
    even though it has no hole and could pass the cheap elongation
    pre-filter."""
    pts = []
    outer_r, inner_r = 10.0, 6.0
    n = 48
    # Sweep 300 degrees around a ring, leaving a 60-degree gap -- classic
    # "C" shape -- outer arc then back along the inner arc.
    start_deg, end_deg = 30, 330
    for i in range(n + 1):
        t = start_deg + (end_deg - start_deg) * i / n
        rad = math.radians(t)
        pts.append((outer_r * math.cos(rad), outer_r * math.sin(rad)))
    for i in range(n + 1):
        t = end_deg - (end_deg - start_deg) * i / n
        rad = math.radians(t)
        pts.append((inner_r * math.cos(rad), inner_r * math.sin(rad)))
    return pts


class TestIsSatinEligible:
    def test_straight_thin_bar_is_eligible(self):
        assert is_satin_eligible([_straight_bar()]) is True

    def test_rotated_thin_bar_is_still_eligible(self):
        assert is_satin_eligible([_diagonal_bar()]) is True

    def test_blobby_square_is_not_elongated_enough(self):
        assert is_satin_eligible([_blobby_square()]) is False

    def test_region_with_a_hole_is_never_eligible(self):
        # Even though the outer ring alone would read as moderately
        # elongated/thin-walled, a hole means this region isn't the simple
        # polygon the scan-based generator requires (see module docstring).
        assert is_satin_eligible(_ring_with_hole()) is False

    def test_bar_wider_than_max_satin_width_is_not_eligible(self):
        wide_bar = _straight_bar(length=20.0, width=9.0)
        assert is_satin_eligible([wide_bar], max_satin_width_mm=7.0) is False

    def test_degenerate_ring_is_not_eligible(self):
        assert is_satin_eligible([[]]) is False
        assert is_satin_eligible([[(0, 0), (1, 1)]]) is False


class TestGenerateSatinRungsForThinRegion:
    def test_straight_bar_produces_rungs_spanning_its_width(self):
        width = 2.0
        rungs = generate_satin_rungs_for_thin_region(_straight_bar(length=20.0, width=width))
        assert rungs is not None
        assert len(rungs) >= 4
        assert len(rungs) % 2 == 0
        rail_a, rail_b = rungs[0::2], rungs[1::2]
        for a, b in zip(rail_a, rail_b):
            assert math.isclose(dist(a, b), width, abs_tol=0.2)

    def test_rotated_bar_still_produces_valid_rungs(self):
        rungs = generate_satin_rungs_for_thin_region(_diagonal_bar(length=20.0, width=2.0))
        assert rungs is not None
        assert len(rungs) >= 4

    def test_rungs_feed_cleanly_into_generate_satin_stitch(self):
        # End-to-end sanity: the rung geometry this module produces must be
        # directly usable by the existing satin stitch generator, exactly
        # like the hand-built border rungs already are.
        rungs = generate_satin_rungs_for_thin_region(_straight_bar(length=20.0, width=2.0))
        assert rungs is not None
        stitches = generate_satin_stitch(rungs, density_mm=0.4)
        assert len(stitches) > 10
        # Every stitch should stay within (a small margin around) the bar's
        # bounding box -- no wild excursions from a malformed rung.
        for x, y in stitches:
            assert -0.5 <= x <= 20.5
            assert -0.5 <= y <= 2.5

    def test_horseshoe_shape_bails_out_to_none(self):
        # The core safety guarantee: a shape where a straight scan line can
        # cross the boundary more than twice must never be forced into a
        # satin column -- it must come back None so the caller falls back
        # to FILL, not a broken/self-crossing stitch path.
        rungs = generate_satin_rungs_for_thin_region(_horseshoe())
        assert rungs is None

    def test_degenerate_ring_returns_none(self):
        assert generate_satin_rungs_for_thin_region([]) is None
        assert generate_satin_rungs_for_thin_region([(0, 0), (1, 1)]) is None

    def test_small_square_too_short_along_its_axis_returns_none(self):
        # A tiny square has no meaningful "long axis" at all relative to a
        # typical step size -- should bail rather than produce a
        # degenerate single-rung "column".
        tiny = [(0, 0), (0.3, 0), (0.3, 0.3), (0, 0.3)]
        assert generate_satin_rungs_for_thin_region(tiny, step_mm=1.0) is None

    def test_tapering_lozenge_produces_valid_rungs_instead_of_bailing(self):
        # Regression test for a real production bug: a scan line landing
        # exactly on (or past) a shape's tapering tip finds zero clean
        # crossings there, which used to bail the *entire* region out to
        # FILL even though every other scan line along its length was
        # perfectly clean. Benchmarking against real photo-pipeline output
        # showed this made the SATIN classifier fire on 0 of 119
        # geometrically-eligible regions in a real text/line-art asset --
        # it must now skip just the tapering end rungs and still produce a
        # valid column.
        rungs = generate_satin_rungs_for_thin_region(_lozenge(length=20.0, half_width=2.0))
        assert rungs is not None
        assert len(rungs) >= 4
        assert len(rungs) % 2 == 0
        stitches = generate_satin_stitch(rungs, density_mm=0.4)
        assert len(stitches) > 10
        for x, y in stitches:
            assert -0.5 <= x <= 20.5
            assert -2.5 <= y <= 2.5

    def test_bent_bracket_shape_safely_bails_to_none(self):
        # A sharply bent (L-shaped) thin bar has no single global axis that
        # cleanly covers both arms -- whatever the exact reason (not
        # eligible at all, or an ambiguous/gapped scan partway along), the
        # safety guarantee holds: it must never come back as a broken or
        # self-crossing satin column.
        bent = [(0, 0), (30, 0), (30, 2), (2, 2), (2, 30), (0, 30)]
        assert generate_satin_rungs_for_thin_region(bent) is None
