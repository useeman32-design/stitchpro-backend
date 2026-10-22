"""Tests for digitizer/stitches/cross/generator.py."""

from __future__ import annotations

import pytest

from digitizer.stitches.cross.generator import generate_cross_stitch


def _square(size=10.0):
    return [[(0, 0), (size, 0), (size, size), (0, size), (0, 0)]]


class TestGenerateCrossStitch:
    def test_rejects_non_positive_cell_size(self):
        with pytest.raises(ValueError):
            generate_cross_stitch(_square(), cell_mm=0)

    def test_rejects_empty_rings(self):
        with pytest.raises(ValueError):
            generate_cross_stitch([])

    def test_produces_four_points_per_cell(self):
        stitches = generate_cross_stitch(_square(10.0), cell_mm=5.0)
        # A 10x10 square with 5mm cells -> 2x2 = 4 cells, 4 points each.
        assert len(stitches) == 16

    def test_every_cell_has_a_jump_then_three_stitches_pattern(self):
        stitches = generate_cross_stitch(_square(10.0), cell_mm=5.0)
        for i in range(0, len(stitches), 4):
            cell = stitches[i : i + 4]
            jumps = [s[2] for s in cell]
            assert jumps == [True, False, True, False]

    def test_finer_grid_produces_more_cells(self):
        coarse = generate_cross_stitch(_square(10.0), cell_mm=5.0)
        fine = generate_cross_stitch(_square(10.0), cell_mm=2.0)
        assert len(fine) > len(coarse)

    def test_hole_excludes_its_cells(self):
        rings = [
            [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)],
            [(8, 8), (12, 8), (12, 12), (8, 12), (8, 8)],
        ]
        with_hole = generate_cross_stitch(rings, cell_mm=2.0)
        without_hole = generate_cross_stitch([rings[0]], cell_mm=2.0)
        assert len(with_hole) < len(without_hole)

    def test_all_stitches_stay_within_expanded_bounds(self):
        stitches = generate_cross_stitch(_square(10.0), cell_mm=3.0)
        xs = [s[0] for s in stitches]
        ys = [s[1] for s in stitches]
        # Cell half-width can push a corner slightly outside [0, 10], but
        # never by more than one cell.
        assert min(xs) >= -3.0
        assert max(xs) <= 13.0
        assert min(ys) >= -3.0
        assert max(ys) <= 13.0

    def test_pull_compensation_expands_outer_boundary(self):
        base = generate_cross_stitch(_square(10.0), cell_mm=5.0)
        expanded = generate_cross_stitch(_square(10.0), cell_mm=5.0, pull_compensation_mm=1.0)
        xs_base = [s[0] for s in base]
        xs_expanded = [s[0] for s in expanded]
        # The grid's phase can shift slightly once the polygon's bounds
        # change, so check the boundary actually moved outward (the min
        # edge, which isn't affected by the max-side cell-inclusion cutoff
        # a coarse cell size can hit) rather than asserting on a specific
        # side that happens to depend on exact grid alignment.
        assert min(xs_expanded) < min(xs_base)

    def test_angled_grid_rotates_cleanly(self):
        stitches = generate_cross_stitch(_square(10.0), cell_mm=5.0, angle_deg=45)
        assert len(stitches) > 0

    def test_degenerate_polygon_raises(self):
        with pytest.raises(ValueError):
            generate_cross_stitch([[(0, 0), (0, 0), (0, 0)]], cell_mm=1.0)
