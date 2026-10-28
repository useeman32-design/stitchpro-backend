"""
Dedicated embroidery lettering engine tests (digitizer/lettering/*) --
mirrors the same methodology as tests/test_satin_stitch.py: check our own
stitch-plan output AND round-trip through pyembroidery (write -> read back)
to catch format-level bugs, not just geometry bugs.
"""

import math
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery

from digitizer.lettering.alphabet_block import GLYPHS, glyph_for, is_fully_supported
from digitizer.lettering.engine import build_text_design
from digitizer.lettering.skeleton import arc_points, line_points, polyline_points, stroke_to_satin_geometry
from digitizer.model import ObjectType
from formats.dst.writer import design_to_dst_bytes


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


# ---------------------------------------------------------------------------
# skeleton.py primitives
# ---------------------------------------------------------------------------


def test_line_points_includes_both_endpoints():
    pts = line_points((0, 0), (10, 0), step=3)
    assert pts[0] == (0, 0)
    assert pts[-1] == (10, 0)
    # every consecutive gap should be <= step (generator never produces a
    # single jump bigger than requested)
    for a, b in zip(pts, pts[1:]):
        assert dist(a, b) <= 3 + 1e-9


def test_polyline_points_preserves_every_vertex_exactly():
    vertices = [(0, 10), (3.5, 0), (7, 10)]  # 'A' apex shape
    pts = polyline_points(vertices, step=0.5)
    for v in vertices:
        assert v in pts, f"vertex {v} was not preserved exactly"


def test_arc_points_full_circle_closes():
    pts = arc_points(0, 0, 5, 5, 0, 360, step_deg=10)
    assert dist(pts[0], pts[-1]) < 1e-6  # a 0..360 sweep must close the loop


def test_arc_points_quarter_circle_endpoints():
    # theta=0 -> (5,0); theta=90 -> (0,5) under x=cx+r*cos, y=cy+r*sin
    pts = arc_points(0, 0, 5, 5, 0, 90, step_deg=5)
    assert dist(pts[0], (5, 0)) < 1e-9
    assert dist(pts[-1], (0, 5)) < 1e-9


def test_stroke_to_satin_geometry_straight_line_constant_width():
    skeleton = line_points((0, 0), (20, 0), step=2)
    geometry = stroke_to_satin_geometry(skeleton, width_units=4)
    assert len(geometry) == 2 * len(skeleton)
    rail_a = geometry[0::2]
    rail_b = geometry[1::2]
    # A straight horizontal stroke's rails are two parallel horizontal
    # lines, offset +/-2 from the skeleton (width 4 => half-width 2).
    for (ax, ay), (bx, by) in zip(rail_a, rail_b):
        assert abs(abs(ay - by) - 4) < 1e-9
        assert ay != by  # rails are genuinely offset, not collapsed


def test_stroke_to_satin_geometry_rejects_too_short_skeleton():
    try:
        stroke_to_satin_geometry([(0, 0)], width_units=2)
        assert False, "expected ValueError for a single-point skeleton"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# alphabet_block.py glyph data
# ---------------------------------------------------------------------------


def test_every_uppercase_letter_and_digit_has_a_glyph():
    for ch in string.ascii_uppercase + string.digits + " ":
        assert glyph_for(ch) is not None, f"missing glyph for {ch!r}"


def test_glyph_lookup_is_case_insensitive():
    assert glyph_for("a") == glyph_for("A")


def test_is_fully_supported():
    assert is_fully_supported("HELLO WORLD 123")
    assert not is_fully_supported("HELLO™")  # trademark symbol has no glyph


def test_every_glyph_stroke_has_at_least_two_points_worth_of_geometry():
    # Every stroke must be convertible to a valid skeleton (>=2 points)
    # since build_text_design feeds each one straight into
    # stroke_to_satin_geometry, which requires that.
    from digitizer.lettering.engine import _stroke_skeleton

    for ch, (_advance, strokes) in GLYPHS.items():
        for stroke in strokes:
            pts = _stroke_skeleton(stroke, step_units=0.5)
            assert len(pts) >= 2, f"glyph {ch!r} stroke {stroke!r} produced too few points"


# ---------------------------------------------------------------------------
# engine.py: full text -> design -> real DST
# ---------------------------------------------------------------------------


def test_build_text_design_basic_shape():
    design = build_text_design("HI", size_mm=10, color="#112233")
    assert design.width_mm > 0
    assert design.height_mm > 0
    assert design.threads == ["#112233"]
    assert all(o.type == ObjectType.SATIN for o in design.objects)
    # "H" has 3 strokes (2 verticals + crossbar), "I" has 1 -> 4 objects total
    assert len(design.objects) == 4


def test_build_text_design_empty_string_rejected():
    try:
        build_text_design("", size_mm=10)
        assert False, "expected ValueError for empty text"
    except ValueError:
        pass


def test_build_text_design_unsupported_character_rejected():
    try:
        build_text_design("HI™", size_mm=10)
        assert False, "expected ValueError for an unsupported character"
    except ValueError as e:
        assert "™" in str(e)


def test_build_text_design_unsupported_layout_rejected():
    try:
        build_text_design("HI", size_mm=10, layout="arc_up")
        assert False, "expected ValueError for an unsupported layout"
    except ValueError:
        pass


def test_build_text_design_scales_with_size_mm():
    small = build_text_design("HI", size_mm=10)
    big = build_text_design("HI", size_mm=20)
    assert big.height_mm > small.height_mm * 1.8  # roughly double, some margin


def test_build_text_design_longer_text_is_wider():
    short = build_text_design("HI", size_mm=10)
    long_ = build_text_design("HIHIHI", size_mm=10)
    assert long_.width_mm > short.width_mm * 2


def test_build_text_design_real_dst_roundtrip():
    design = build_text_design("STITCHPRO 2026", size_mm=12, color="#5B4FE8")
    dst_bytes = design_to_dst_bytes(design)
    assert len(dst_bytes) > 100

    pattern = read_embroidery(io_bytes(dst_bytes))
    assert len(pattern.stitches) > 500  # a real multi-letter satin design is never tiny

    # Every stitch coordinate must be finite and within a sane bound of the
    # design's own declared size (0.1mm units, some margin for satin width).
    max_extent_tenth_mm = (max(design.width_mm, design.height_mm) + 10) * 10
    for x, y, _cmd in pattern.stitches:
        assert math.isfinite(x) and math.isfinite(y)
        assert abs(x) < max_extent_tenth_mm * 5
        assert abs(y) < max_extent_tenth_mm * 5


def io_bytes(data: bytes):
    import io

    return io.BytesIO(data)


def test_build_text_design_each_letter_is_a_separate_object_for_routing():
    # Letters should route/jump independently (not be fused into one
    # mega-object), so the machine can trim threads between them if needed
    # and so per-letter inspection/debugging stays possible.
    design = build_text_design("ABC", size_mm=10)
    # 'A' 2 strokes, 'B' 3 strokes, 'C' 1 stroke => 6 objects
    assert len(design.objects) == 6
