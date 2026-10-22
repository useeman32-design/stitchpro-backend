"""
Milestone 2 tests: satin column -> satin stitch -> DST.

Every test both checks our own generator output AND (for the full-pipeline
test) round-trips through pyembroidery (write -> independently read back) to
catch format-level bugs, not just stitch-math bugs -- matching brief section
21's test methodology, same as tests/test_running_stitch.py.
"""

import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery
from pyembroidery import END, JUMP, STITCH

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.satin.generator import generate_satin_stitch
from formats.dst.writer import compile_design, design_to_dst_bytes


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


# ---------------------------------------------------------------------------
# 1. Straight parallel-rail rectangle (simplest possible column: 2 rungs)
# ---------------------------------------------------------------------------

def test_straight_rectangle_endpoints_preserved():
    # A 20mm-long, 4mm-wide column: rail A along y=0, rail B along y=4.
    geometry = [(0, 0), (0, 4), (20, 0), (20, 4)]
    stitches = generate_satin_stitch(geometry, density_mm=2.0)

    assert stitches[0] == (0, 0)
    assert stitches[1] == (0, 4)
    assert stitches[-2] == (20, 0)
    assert stitches[-1] == (20, 4)


def test_straight_rectangle_alternates_rails():
    geometry = [(0, 0), (0, 4), (20, 0), (20, 4)]
    stitches = generate_satin_stitch(geometry, density_mm=2.0)

    # every even index should be on rail A (y=0), every odd index on rail B (y=4)
    for i, (x, y) in enumerate(stitches):
        if i % 2 == 0:
            assert abs(y - 0) < 1e-9
        else:
            assert abs(y - 4) < 1e-9


def test_straight_rectangle_density_bounds_rail_step():
    geometry = [(0, 0), (0, 4), (20, 0), (20, 4)]
    density = 2.0
    stitches = generate_satin_stitch(geometry, density_mm=density)

    rail_a = stitches[0::2]
    for i in range(len(rail_a) - 1):
        d = dist(rail_a[i], rail_a[i + 1])
        assert 0 < d <= density + 1e-6
    # 20mm / 2mm = 10 exact steps -> 11 rail-A points
    assert len(rail_a) == 11


def test_single_point_input_raises_odd_length():
    try:
        generate_satin_stitch([(0, 0), (0, 4), (20, 0)], density_mm=2.0)
        assert False, "expected ValueError for odd-length geometry"
    except ValueError:
        pass


def test_too_few_points_raises():
    try:
        generate_satin_stitch([(0, 0), (0, 4)], density_mm=2.0)
        assert False, "expected ValueError: need at least 2 rungs"
    except ValueError:
        pass


def test_zero_density_raises():
    try:
        generate_satin_stitch([(0, 0), (0, 4), (20, 0), (20, 4)], density_mm=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 2. Tapering column: rungs of different widths, rails of different lengths
#    between two rungs -- both rails must still be densely/evenly stitched.
# ---------------------------------------------------------------------------

def test_tapering_column_preserves_rung_vertices():
    # Column widens from 2mm to 8mm over its 30mm length -- a single
    # interior rung at x=15 marks the widening point.
    geometry = [
        (0, -1), (0, 1),      # rung 0: 2mm wide
        (15, -4), (15, 4),    # rung 1 (interior): 8mm wide
        (30, -1), (30, 1),    # rung 2: back to 2mm wide
    ]
    stitches = generate_satin_stitch(geometry, density_mm=3.0)

    assert (0, -1) in stitches and (0, 1) in stitches
    assert (15, -4) in stitches and (15, 4) in stitches
    assert (30, -1) in stitches and (30, 1) in stitches
    # interior rung must not be duplicated (shared by both sections)
    assert stitches.count((15, -4)) == 1
    assert stitches.count((15, 4)) == 1


def test_uneven_rail_lengths_between_rungs_both_bounded():
    # Rail A travels a straight 30mm; rail B curves out and travels further
    # between the same two rungs (simulated here as just a longer straight
    # distance) -- the longer rail's length must drive the step count so
    # neither rail gets an overlong individual stitch.
    geometry = [(0, 0), (0, 0), (30, 0), (10, 40)]
    density = 4.0
    stitches = generate_satin_stitch(geometry, density_mm=density)

    rail_a = stitches[0::2]
    rail_b = stitches[1::2]
    for rail in (rail_a, rail_b):
        for i in range(len(rail) - 1):
            assert dist(rail[i], rail[i + 1]) <= density + 1e-6


# ---------------------------------------------------------------------------
# 3. Pull compensation: rungs should expand outward by the given amount.
# ---------------------------------------------------------------------------

def test_pull_compensation_expands_rungs():
    geometry = [(0, -2), (0, 2), (20, -2), (20, 2)]
    comp = 0.5
    stitches = generate_satin_stitch(geometry, density_mm=2.0, pull_compensation_mm=comp)

    # first rung's rail-A/rail-B points should have moved further apart by
    # 2 * comp total (comp on each side)
    first_a, first_b = stitches[0], stitches[1]
    assert abs(first_a[1] - (-2 - comp)) < 1e-9
    assert abs(first_b[1] - (2 + comp)) < 1e-9


def test_zero_pull_compensation_is_noop():
    geometry = [(0, -2), (0, 2), (20, -2), (20, 2)]
    stitches = generate_satin_stitch(geometry, density_mm=2.0, pull_compensation_mm=0.0)
    assert stitches[0] == (0, -2)
    assert stitches[1] == (0, 2)


# ---------------------------------------------------------------------------
# 4. Full pipeline: EmbroideryDesign -> compile -> DST bytes -> independent
#    pyembroidery read-back, verifying dimensions/stitch count/commands.
# ---------------------------------------------------------------------------

def test_satin_to_dst_roundtrip():
    geometry = [(5, 23), (5, 27), (45, 23), (45, 27)]  # 40mm x 4mm satin bar
    design = EmbroideryDesign(
        width_mm=50,
        height_mm=50,
        objects=[
            EmbroideryObject(
                id="satin-1",
                type=ObjectType.SATIN,
                geometry=geometry,
                color="#1A1A1A",
                stitch_settings=StitchSettings(density_mm=2.0),
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
    assert JUMP in commands
    assert STITCH in commands
    assert commands[-1] == END or END in commands

    stitch_count = sum(1 for c in commands if c == STITCH)
    expected = len(generate_satin_stitch(geometry, density_mm=2.0))
    assert stitch_count == expected

    min_x, min_y, max_x, max_y = pattern.bounds()
    width_units = max_x - min_x
    height_units = max_y - min_y
    # 40mm long, 4mm wide -> ~400 x ~40 units (0.1mm)
    assert 390 <= width_units <= 410, f"unexpected width: {width_units}"
    assert 30 <= height_units <= 50, f"unexpected height: {height_units}"


def test_satin_and_running_objects_together_color_change():
    design = EmbroideryDesign(
        width_mm=50,
        height_mm=50,
        objects=[
            EmbroideryObject(
                id="outline",
                type=ObjectType.RUNNING,
                geometry=[(0, 0), (10, 0)],
                color="#FF0000",
                sequence=0,
            ),
            EmbroideryObject(
                id="fill-bar",
                type=ObjectType.SATIN,
                geometry=[(0, 5), (0, 8), (10, 5), (10, 8)],
                color="#0000FF",
                stitch_settings=StitchSettings(density_mm=2.0),
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
