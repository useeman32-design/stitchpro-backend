"""
Quality-improvement tests: fill with holes (letterform/donut shapes).

Milestone 3's fill generator originally only accepted a single boundary
ring; real letterforms and logos need enclosed holes excluded from
stitching (an "O", "A", "B", "D", "P", "Q", "R", a donut logo, ...), so
`generate_fill_stitch` now takes a list of rings: rings[0] is the outer
boundary, rings[1:] are holes.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.fill.generator import generate_fill_stitch
from formats.dst.writer import design_to_dst_bytes


def test_single_hole_excluded_from_stitches():
    outer = [(0, 0), (30, 0), (30, 30), (0, 30)]
    hole = [(8, 8), (22, 8), (22, 22), (8, 22)]
    stitches = generate_fill_stitch([outer, hole], stitch_length_mm=2.0, density_mm=1.0)
    assert stitches

    inside_hole = [(x, y) for x, y, _ in stitches if 8.5 < x < 21.5 and 8.5 < y < 21.5]
    assert not inside_hole, f"found {len(inside_hole)} stitches inside the hole"


def test_hole_produces_more_jumps_than_solid_shape():
    outer = [(0, 0), (30, 0), (30, 30), (0, 30)]
    hole = [(8, 8), (22, 8), (22, 22), (8, 22)]
    solid = generate_fill_stitch([outer], stitch_length_mm=2.0, density_mm=1.0)
    with_hole = generate_fill_stitch([outer, hole], stitch_length_mm=2.0, density_mm=1.0)

    solid_jumps = sum(1 for _, _, j in solid if j)
    hole_jumps = sum(1 for _, _, j in with_hole if j)
    # every row crossing the hole now needs 2 spans (and so 2 jumps)
    # instead of 1 -- there must be more jumps with the hole present
    assert hole_jumps > solid_jumps


def test_two_separate_holes():
    outer = [(0, 0), (40, 0), (40, 20), (0, 20)]
    hole_a = [(5, 5), (12, 5), (12, 15), (5, 15)]
    hole_b = [(28, 5), (35, 5), (35, 15), (28, 15)]
    stitches = generate_fill_stitch([outer, hole_a, hole_b], stitch_length_mm=2.0, density_mm=1.0)
    assert stitches

    for hx0, hy0, hx1, hy1 in [(5.5, 5.5, 11.5, 14.5), (28.5, 5.5, 34.5, 14.5)]:
        inside = [(x, y) for x, y, _ in stitches if hx0 < x < hx1 and hy0 < y < hy1]
        assert not inside, "stitch found inside a hole"


def test_hole_fully_covering_outer_raises():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(-5, -5), (15, -5), (15, 15), (-5, 15)]  # bigger than the outer ring
    try:
        generate_fill_stitch([outer, hole], stitch_length_mm=2.0, density_mm=1.0)
        assert False, "expected ValueError for a hole that covers the whole shape"
    except ValueError:
        pass


def test_ring_with_too_few_points_raises():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    bad_hole = [(1, 1), (2, 2)]  # only 2 points
    try:
        generate_fill_stitch([outer, bad_hole], stitch_length_mm=2.0, density_mm=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_empty_rings_list_raises():
    try:
        generate_fill_stitch([], stitch_length_mm=2.0, density_mm=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_letterform_o_to_dst_roundtrip():
    outer = [(5, 5), (35, 5), (35, 35), (5, 35)]
    hole = [(13, 13), (27, 13), (27, 27), (13, 27)]
    design = EmbroideryDesign(
        width_mm=50,
        height_mm=50,
        objects=[
            EmbroideryObject(
                id="letter-o",
                type=ObjectType.FILL,
                geometry=[outer, hole],
                color="#1A1A1A",
                stitch_settings=StitchSettings(stitch_length_mm=2.0, density_mm=1.0),
                sequence=0,
            )
        ],
        threads=["#1A1A1A"],
    )
    dst_bytes = design_to_dst_bytes(design)
    pattern = read_embroidery(io.BytesIO(dst_bytes))
    assert len(pattern.stitches) > 0
    min_x, min_y, max_x, max_y = pattern.bounds()
    assert 280 <= (max_x - min_x) <= 320
    assert 280 <= (max_y - min_y) <= 320


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
