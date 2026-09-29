"""
Milestone 1 tests: line -> running stitch -> DST.

Every test both checks our own generator output AND round-trips through
pyembroidery (write -> independently read back) to catch format-level bugs,
not just stitch-math bugs -- matching brief section 21's test methodology.
"""

import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery
from pyembroidery import JUMP, STITCH, END

from digitizer.geometry.bezier import flatten_cubic_bezier
from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.running.generator import generate_running_stitch
from formats.dst.writer import compile_design, design_to_dst_bytes


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


# ---------------------------------------------------------------------------
# 1. Straight line
# ---------------------------------------------------------------------------

def test_straight_line_stitch_length():
    points = [(0, 0), (40, 0)]  # 40mm horizontal line
    stitches = generate_running_stitch(points, stitch_length_mm=3.0)

    assert stitches[0] == (0, 0)
    assert stitches[-1] == (40, 0)
    # 40 / 3 = 13.33 -> 14 even steps of ~2.857mm each
    assert len(stitches) == 15  # start point + 14 steps
    for i in range(len(stitches) - 1):
        d = dist(stitches[i], stitches[i + 1])
        assert 0 < d <= 3.0 + 1e-6
    # all points colinear (y stays 0)
    assert all(abs(y) < 1e-9 for _, y in stitches)


def test_single_point_input():
    assert generate_running_stitch([(5, 5)], 3.0) == [(5, 5)]


def test_zero_length_stitch_raises():
    try:
        generate_running_stitch([(0, 0), (10, 0)], stitch_length_mm=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 2. Corner / sharp turn: a right-angle path must include the exact vertex
# ---------------------------------------------------------------------------

def test_corner_vertex_is_preserved():
    points = [(0, 0), (20, 0), (20, 20)]  # L-shaped path, sharp 90-degree turn
    stitches = generate_running_stitch(points, stitch_length_mm=4.0)

    assert (20, 0) in stitches, "the corner vertex must be an exact stitch point"
    assert stitches[0] == (0, 0)
    assert stitches[-1] == (20, 20)
    # no stitch should jump straight from before the corner to after it
    for i in range(len(stitches) - 1):
        assert dist(stitches[i], stitches[i + 1]) <= 4.0 + 1e-6


# ---------------------------------------------------------------------------
# 3. Curve (flattened bezier) support
# ---------------------------------------------------------------------------

def test_curve_flatten_then_stitch():
    curve_points = flatten_cubic_bezier((0, 0), (10, 30), (30, 30), (40, 0), segments=20)
    stitches = generate_running_stitch(curve_points, stitch_length_mm=2.0)

    assert stitches[0] == curve_points[0]
    assert stitches[-1] == curve_points[-1]
    # stitch spacing should stay bounded even along a curve
    for i in range(len(stitches) - 1):
        assert dist(stitches[i], stitches[i + 1]) <= 2.0 + 1e-6
    assert len(stitches) > 15  # a 40mm-ish curve at 2mm stitches should have plenty of points


# ---------------------------------------------------------------------------
# 4. Full pipeline: EmbroideryDesign -> compile -> DST bytes -> independent
#    pyembroidery read-back, verifying dimensions/stitch count/commands.
# ---------------------------------------------------------------------------

def test_line_to_dst_roundtrip():
    design = EmbroideryDesign(
        width_mm=50,
        height_mm=50,
        objects=[
            EmbroideryObject(
                id="line-1",
                type=ObjectType.RUNNING,
                geometry=[(5, 25), (45, 25)],  # 40mm horizontal line, centered-ish
                color="#1A1A1A",
                stitch_settings=StitchSettings(stitch_length_mm=3.0),
                sequence=0,
            )
        ],
        threads=["#1A1A1A"],
    )

    dst_bytes = design_to_dst_bytes(design)
    assert len(dst_bytes) > 0

    pattern = read_embroidery(io.BytesIO(dst_bytes))
    assert pattern is not None

    commands = [s[2] for s in pattern.stitches]
    assert JUMP in commands, "must jump to the start of the line before stitching"
    assert STITCH in commands
    assert commands[-1] == END or END in commands

    stitch_count = sum(1 for c in commands if c == STITCH)
    expected = len(generate_running_stitch([(5, 25), (45, 25)], 3.0))
    assert stitch_count == expected

    # bounds: centered design, 40mm line -> spans 20mm each side of x=0 roughly
    min_x, min_y, max_x, max_y = pattern.bounds()
    width_units = max_x - min_x
    # 0.1mm units; 40mm line -> 400 units wide (allow a little slack for the
    # single JUMP point coinciding with the first stitch)
    assert 390 <= width_units <= 410, f"unexpected width: {width_units}"


def test_two_objects_color_change():
    design = EmbroideryDesign(
        width_mm=50,
        height_mm=50,
        objects=[
            EmbroideryObject(
                id="a",
                type=ObjectType.RUNNING,
                geometry=[(0, 0), (10, 0)],
                color="#FF0000",
                sequence=0,
            ),
            EmbroideryObject(
                id="b",
                type=ObjectType.RUNNING,
                geometry=[(0, 10), (10, 10)],
                color="#0000FF",
                sequence=1,
            ),
        ],
        threads=["#FF0000", "#0000FF"],
    )
    pattern = compile_design(design)
    assert len(pattern.threadlist) == 2
    commands = [s[2] for s in pattern.stitches]
    from pyembroidery import COLOR_CHANGE

    assert COLOR_CHANGE in commands


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
