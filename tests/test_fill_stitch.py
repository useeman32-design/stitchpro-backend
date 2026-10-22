"""
Milestone 3 tests: polygon -> fill (tatami) stitch -> DST.

Every test both checks our own generator output AND (for the full-pipeline
test) round-trips through pyembroidery (write -> independently read back) to
catch format-level bugs, not just stitch-math bugs -- matching brief section
21's test methodology, same as tests/test_running_stitch.py and
tests/test_satin_stitch.py.
"""

import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery
from pyembroidery import END, JUMP, STITCH

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.fill.generator import generate_fill_stitch
from formats.dst.writer import compile_design, design_to_dst_bytes


def dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


# ---------------------------------------------------------------------------
# 1. Simple convex rectangle -- one span per row, roughly height/density rows
# ---------------------------------------------------------------------------

def test_rectangle_row_count():
    # 20mm wide, 10mm tall rectangle, rows every 2mm -> ~5 rows
    rect = [(0, 0), (20, 0), (20, 10), (0, 10)]
    stitches = generate_fill_stitch([rect], stitch_length_mm=3.0, density_mm=2.0)
    assert stitches, "expected at least some stitches"

    jump_indices = [i for i, (_, _, is_jump) in enumerate(stitches) if is_jump]
    # one jump per row (single span each, convex rectangle)
    assert 4 <= len(jump_indices) <= 6


def test_rectangle_spans_cover_full_width():
    rect = [(0, 0), (20, 0), (20, 10), (0, 10)]
    stitches = generate_fill_stitch([rect], stitch_length_mm=2.0, density_mm=3.0)
    xs = [x for x, _, _ in stitches]
    assert min(xs) >= -1e-6
    assert max(xs) <= 20 + 1e-6
    # should get reasonably close to both edges
    assert min(xs) < 1.0
    assert max(xs) > 19.0


def test_rectangle_stitch_length_bounded_within_runs():
    rect = [(0, 0), (20, 0), (20, 10), (0, 10)]
    stitch_length = 2.5
    stitches = generate_fill_stitch([rect], stitch_length_mm=stitch_length, density_mm=3.0)

    for i in range(1, len(stitches)):
        x0, y0, is_jump = stitches[i]
        if is_jump:
            continue  # jumps aren't subject to the stitch-length bound
        x1, y1, _ = stitches[i - 1]
        assert dist((x0, y0), (x1, y1)) <= stitch_length + 1e-6


def test_rows_alternate_direction():
    rect = [(0, 0), (20, 0), (20, 10), (0, 10)]
    stitches = generate_fill_stitch([rect], stitch_length_mm=3.0, density_mm=2.0)

    jump_positions = [(x, y) for x, y, is_jump in stitches if is_jump]
    assert len(jump_positions) >= 2
    # consecutive row starts should alternate which side of the rectangle
    # they're on (boustrophedon / snake ordering), not always start at x=0
    xs_of_starts = [x for x, _ in jump_positions]
    assert min(xs_of_starts) < 1.0
    assert max(xs_of_starts) > 19.0


# ---------------------------------------------------------------------------
# 2. Concave polygon (U-shape / notch) -- some rows must split into 2 spans
# ---------------------------------------------------------------------------

def test_concave_u_shape_produces_multiple_spans_on_notched_rows():
    # A 30x20 U-shape: full rectangle with a 10-wide, 12-tall notch cut out
    # of the top-middle, so rows near the top are split into two spans.
    u_shape = [
        (0, 0), (30, 0), (30, 20), (20, 20), (20, 8), (10, 8), (10, 20), (0, 20),
    ]
    stitches = generate_fill_stitch([u_shape], stitch_length_mm=3.0, density_mm=2.0)
    assert stitches

    # Rows near y=18 (inside the notch region) should have 2 spans -> 2 jumps
    # within that row's band; rows near y=2 (below the notch) should have 1.
    jumps_by_approx_row: dict[int, int] = {}
    for x, y, is_jump in stitches:
        if not is_jump:
            continue
        row_key = round(y)
        jumps_by_approx_row[row_key] = jumps_by_approx_row.get(row_key, 0) + 1

    near_top_rows = [v for k, v in jumps_by_approx_row.items() if 10 <= k <= 19]
    near_bottom_rows = [v for k, v in jumps_by_approx_row.items() if k <= 5]
    assert any(v == 2 for v in near_top_rows), "expected a 2-span (notched) row near the top"
    assert any(v == 1 for v in near_bottom_rows), "expected a 1-span row below the notch"


def test_concave_shape_spans_stay_within_polygon_bounds():
    u_shape = [
        (0, 0), (30, 0), (30, 20), (20, 20), (20, 8), (10, 8), (10, 20), (0, 20),
    ]
    stitches = generate_fill_stitch([u_shape], stitch_length_mm=3.0, density_mm=2.0)
    for x, y, _ in stitches:
        assert -0.5 <= x <= 30.5
        assert -0.5 <= y <= 20.5
        # nothing should land deep inside the notch (x in [10,20], y > 9)
        assert not (10.5 < x < 19.5 and y > 9), f"stitch at ({x},{y}) falls inside the notch"


# ---------------------------------------------------------------------------
# 3. Angle: rows should run parallel to angle_deg
# ---------------------------------------------------------------------------

def test_angle_90_rotates_row_direction():
    # Wide-short rectangle: at angle 0, rows run horizontally (few long
    # rows); at angle 90, rows run vertically (many short rows) because the
    # row-spacing axis and the row-length axis swap.
    wide_rect = [(0, 0), (40, 0), (40, 10), (0, 10)]
    density = 2.0

    stitches_0 = generate_fill_stitch([wide_rect], stitch_length_mm=3.0, density_mm=density, angle_deg=0)
    stitches_90 = generate_fill_stitch([wide_rect], stitch_length_mm=3.0, density_mm=density, angle_deg=90)

    runs_0 = sum(1 for _, _, is_jump in stitches_0 if is_jump)
    runs_90 = sum(1 for _, _, is_jump in stitches_90 if is_jump)

    # at angle 0: ~10mm tall / 2mm density -> ~5 rows
    # at angle 90: ~40mm wide / 2mm density -> ~20 rows
    assert runs_90 > runs_0 * 2


def test_angle_stitches_stay_within_rotated_bounds():
    # A square is rotation-invariant in bounds, so angle shouldn't push any
    # stitch outside the original square regardless of angle.
    square = [(0, 0), (20, 0), (20, 20), (0, 20)]
    stitches = generate_fill_stitch([square], stitch_length_mm=3.0, density_mm=2.0, angle_deg=35)
    for x, y, _ in stitches:
        assert -1.0 <= x <= 21.0
        assert -1.0 <= y <= 21.0


# ---------------------------------------------------------------------------
# 4. Pull compensation and validation
# ---------------------------------------------------------------------------

def test_pull_compensation_expands_bounds():
    rect = [(0, 0), (20, 0), (20, 10), (0, 10)]
    plain = generate_fill_stitch([rect], stitch_length_mm=3.0, density_mm=2.0)
    expanded = generate_fill_stitch([rect], stitch_length_mm=3.0, density_mm=2.0, pull_compensation_mm=1.0)

    plain_xs = [x for x, _, _ in plain]
    expanded_xs = [x for x, _, _ in expanded]
    assert min(expanded_xs) < min(plain_xs)
    assert max(expanded_xs) > max(plain_xs)


def test_too_few_points_raises():
    try:
        generate_fill_stitch([[(0, 0), (10, 0)]], stitch_length_mm=2.0, density_mm=2.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_zero_density_raises():
    try:
        generate_fill_stitch([[(0, 0), (10, 0), (10, 10)]], stitch_length_mm=2.0, density_mm=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_zero_stitch_length_raises():
    try:
        generate_fill_stitch([[(0, 0), (10, 0), (10, 10)]], stitch_length_mm=0, density_mm=2.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_degenerate_zero_area_polygon_raises():
    try:
        generate_fill_stitch([[(0, 0), (10, 0), (20, 0)]], stitch_length_mm=2.0, density_mm=2.0)
        assert False, "expected ValueError for a zero-area (collinear) polygon"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 5. Full pipeline: EmbroideryDesign -> compile -> DST bytes -> independent
#    pyembroidery read-back.
# ---------------------------------------------------------------------------

def test_fill_to_dst_roundtrip():
    rect = [(5, 5), (35, 5), (35, 25), (5, 25)]  # 30mm x 20mm filled rectangle
    design = EmbroideryDesign(
        width_mm=50,
        height_mm=50,
        objects=[
            EmbroideryObject(
                id="fill-1",
                type=ObjectType.FILL,
                geometry=[rect],
                color="#2E8B57",
                stitch_settings=StitchSettings(stitch_length_mm=3.0, density_mm=2.0),
                sequence=0,
            )
        ],
        threads=["#2E8B57"],
    )

    dst_bytes = design_to_dst_bytes(design)
    assert len(dst_bytes) > 0

    pattern = read_embroidery(io.BytesIO(dst_bytes))
    assert pattern is not None

    commands = [s[2] for s in pattern.stitches]
    assert STITCH in commands
    assert commands[-1] == END or END in commands
    # a multi-row fill must have more than one JUMP (one per row at least)
    jump_count = sum(1 for c in commands if c == JUMP)
    assert jump_count >= 4

    min_x, min_y, max_x, max_y = pattern.bounds()
    width_units = max_x - min_x
    height_units = max_y - min_y
    assert 280 <= width_units <= 320, f"unexpected width: {width_units}"
    assert 180 <= height_units <= 220, f"unexpected height: {height_units}"


def test_all_three_stitch_types_together():
    design = EmbroideryDesign(
        width_mm=60,
        height_mm=40,
        objects=[
            EmbroideryObject(
                id="outline",
                type=ObjectType.RUNNING,
                geometry=[(0, 0), (20, 0)],
                color="#FF0000",
                sequence=0,
            ),
            EmbroideryObject(
                id="band",
                type=ObjectType.SATIN,
                geometry=[(0, 5), (0, 8), (20, 5), (20, 8)],
                color="#0000FF",
                stitch_settings=StitchSettings(density_mm=2.0),
                sequence=1,
            ),
            EmbroideryObject(
                id="body",
                type=ObjectType.FILL,
                geometry=[[(0, 12), (20, 12), (20, 22), (0, 22)]],
                color="#2E8B57",
                stitch_settings=StitchSettings(stitch_length_mm=3.0, density_mm=2.0),
                sequence=2,
            ),
        ],
        threads=["#FF0000", "#0000FF", "#2E8B57"],
    )
    pattern = compile_design(design)
    assert len(pattern.threadlist) == 3
    commands = [s[2] for s in pattern.stitches]
    from pyembroidery import COLOR_CHANGE

    assert commands.count(COLOR_CHANGE) == 2


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
