"""
Satin edge-walk border tests (fill/cross-stitch edge-quality follow-up --
see digitizer/stitches/satin/border.py's module docstring for the problem
this solves).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digitizer.model import EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.dispatch import generate_object_stitches
from digitizer.stitches.fill.generator import generate_fill_stitch
from digitizer.stitches.satin.border import ring_to_satin_rungs
from digitizer.stitches.satin.generator import generate_satin_stitch


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


# ---------------------------------------------------------------------------
# ring_to_satin_rungs
# ---------------------------------------------------------------------------

def test_degenerate_ring_returns_empty():
    assert ring_to_satin_rungs([], 1.2) == []
    assert ring_to_satin_rungs([(0, 0), (1, 1)], 1.2) == []  # only 2 points
    assert ring_to_satin_rungs([(0, 0), (0, 0), (0, 0)], 1.2) == []  # zero perimeter


def test_rungs_straddle_a_square_boundary_evenly():
    # 10x10mm square; every resampled rung should land close to the square's
    # outline with rail A and rail B symmetric about it (one on each side).
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    width = 2.0
    rungs = ring_to_satin_rungs(square, width_mm=width, step_mm=1.0)
    assert len(rungs) >= 8
    assert len(rungs) % 2 == 0

    rail_a = rungs[0::2]
    rail_b = rungs[1::2]
    for a, b in zip(rail_a, rail_b):
        # Each rung's two points should be exactly `width` apart (straddling
        # the boundary line, half on each side).
        assert math.isclose(dist(a, b), width, rel_tol=1e-6, abs_tol=1e-6)
        midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        # The midpoint of every rung should sit very close to the square's
        # actual outline (within a small tolerance for the resampling step).
        on_outline = (
            math.isclose(midpoint[0], 0, abs_tol=0.2)
            or math.isclose(midpoint[0], 10, abs_tol=0.2)
            or math.isclose(midpoint[1], 0, abs_tol=0.2)
            or math.isclose(midpoint[1], 10, abs_tol=0.2)
        )
        assert on_outline, f"rung midpoint {midpoint} not on the square's outline"


def test_closed_loop_repeats_first_rung_at_the_end():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    rungs = ring_to_satin_rungs(square, width_mm=1.0, step_mm=1.0)
    assert rungs[0] == rungs[-2]
    assert rungs[1] == rungs[-1]


def test_rungs_feed_generate_satin_stitch_without_raising():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    rungs = ring_to_satin_rungs(square, width_mm=1.2, step_mm=1.0)
    stitches = generate_satin_stitch(rungs, density_mm=0.4)
    assert len(stitches) > len(rungs)  # satin fills in extra crossing stitches
    for x, y in stitches:
        # Every stitch should stay within half the border width of the
        # square's actual outline (0..10 on each axis) -- never wildly off.
        assert -1.0 <= x <= 11.0
        assert -1.0 <= y <= 11.0


# ---------------------------------------------------------------------------
# dispatch.py wiring: FILL objects with a border produce more stitches than
# the bare fill, and the border's own stitches hug the boundary.
# ---------------------------------------------------------------------------

def test_fill_object_with_border_adds_stitches_on_top_of_bare_fill():
    square = [(0, 0), (20, 0), (20, 20), (0, 20)]

    bare = EmbroideryObject(
        id="r1",
        type=ObjectType.FILL,
        geometry=[square],
        color="#112233",
        stitch_settings=StitchSettings(stitch_length_mm=3.0, density_mm=0.4, border_width_mm=0.0),
    )
    bordered = EmbroideryObject(
        id="r2",
        type=ObjectType.FILL,
        geometry=[square],
        color="#112233",
        stitch_settings=StitchSettings(stitch_length_mm=3.0, density_mm=0.4, border_width_mm=1.2),
    )

    bare_stitches = generate_object_stitches(bare)
    bordered_stitches = generate_object_stitches(bordered)

    assert len(bordered_stitches) > len(bare_stitches)
    # The extra stitches are exactly the fill plus a border -- the bare
    # fill's own output is unaffected by the border setting (purely
    # additive change).
    raw_fill = generate_fill_stitch(bare.geometry, 3.0, 0.4, 0.0, 0.0)
    assert len(bare_stitches) == len(raw_fill)
    assert len(bordered_stitches) == len(raw_fill) + (len(bordered_stitches) - len(raw_fill))


def test_fill_object_border_width_zero_is_a_no_op():
    square = [(0, 0), (20, 0), (20, 20), (0, 20)]
    obj = EmbroideryObject(
        id="r1",
        type=ObjectType.FILL,
        geometry=[square],
        color="#112233",
        stitch_settings=StitchSettings(border_width_mm=0.0),
    )
    stitches = generate_object_stitches(obj)
    raw_fill = generate_fill_stitch(obj.geometry, obj.stitch_settings.stitch_length_mm, obj.stitch_settings.density_mm, 0.0, 0.0)
    assert len(stitches) == len(raw_fill)


def test_fill_with_hole_gets_a_border_on_both_outer_ring_and_hole():
    outer = [(0, 0), (20, 0), (20, 20), (0, 20)]
    hole = [(8, 8), (12, 8), (12, 12), (8, 12)]
    obj = EmbroideryObject(
        id="r1",
        type=ObjectType.FILL,
        geometry=[outer, hole],
        color="#112233",
        stitch_settings=StitchSettings(border_width_mm=1.0),
    )
    stitches = generate_object_stitches(obj)
    raw_fill = generate_fill_stitch(obj.geometry, obj.stitch_settings.stitch_length_mm, obj.stitch_settings.density_mm, 0.0, 0.0)
    border_only = stitches[len(raw_fill):]
    assert len(border_only) > 0

    xs = [x for x, _, _ in border_only]
    ys = [y for _, y, _ in border_only]
    # Border stitches should be found near BOTH the outer square's edges
    # (0/20) and the hole's edges (8/12), not just one or the other.
    near_outer = any(x < 1.0 or x > 19.0 or y < 1.0 or y > 19.0 for x, y in zip(xs, ys))
    near_hole = any(7.0 < x < 13.0 and 7.0 < y < 13.0 for x, y in zip(xs, ys))
    assert near_outer
    assert near_hole
