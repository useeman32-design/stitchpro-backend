"""Tests for digitizer/segmentation/{color_quantize,polygons}.py."""

from __future__ import annotations

from PIL import Image

from digitizer.segmentation.color_quantize import BACKGROUND_LABEL, build_label_grid
from digitizer.segmentation.polygons import label_grid_to_ring_groups


def _solid_image(w, h, rgb):
    return Image.new("RGB", (w, h), rgb)


class TestBuildLabelGrid:
    def test_solid_background_only_image_has_no_foreground_labels(self):
        img = _solid_image(20, 20, (255, 255, 255))
        palette, grid = build_label_grid(img, max_colors=3)
        flat = {v for row in grid for v in row}
        assert flat == {BACKGROUND_LABEL}

    def test_centered_shape_is_foreground_and_border_is_background(self):
        img = Image.new("RGB", (30, 30), (255, 255, 255))
        for y in range(10, 20):
            for x in range(10, 20):
                img.putpixel((x, y), (10, 10, 10))
        palette, grid = build_label_grid(img, max_colors=4)
        assert grid[0][0] == BACKGROUND_LABEL
        assert grid[29][29] == BACKGROUND_LABEL
        assert grid[15][15] != BACKGROUND_LABEL

    def test_transparent_pixels_are_background_regardless_of_color(self):
        img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
        for x in range(10):
            img.putpixel((x, 5), (10, 200, 10, 0))  # fully transparent, non-bg color
        palette, grid = build_label_grid(img, max_colors=3)
        assert all(grid[5][x] == BACKGROUND_LABEL for x in range(10))

    def test_background_detection_uses_whole_border_not_just_corners(self):
        # A stripe touching only the top edge (not the corners) must not
        # confuse background detection away from the white that dominates
        # the rest of the border.
        img = Image.new("RGB", (40, 40), (255, 255, 255))
        for x in range(10, 30):
            img.putpixel((x, 0), (200, 30, 30))
        palette, grid = build_label_grid(img, max_colors=3)
        # bottom-left corner, far from the stripe, must read as background
        assert grid[39][0] == BACKGROUND_LABEL


class TestLabelGridToRingGroups:
    def _square_grid(self, w, h, label, x0, y0, x1, y1):
        grid = [[BACKGROUND_LABEL] * w for _ in range(h)]
        for y in range(y0, y1):
            for x in range(x0, x1):
                grid[y][x] = label
        return grid

    def test_single_blob_produces_one_ring_group(self):
        grid = self._square_grid(20, 20, 0, 5, 5, 15, 15)
        groups = label_grid_to_ring_groups(grid, 0, 1.0, 1.0)
        assert len(groups) == 1
        assert len(groups[0]) == 1  # no holes

    def test_disconnected_blobs_produce_separate_ring_groups(self):
        grid = [[BACKGROUND_LABEL] * 20 for _ in range(20)]
        for y in range(2, 6):
            for x in range(2, 6):
                grid[y][x] = 0
        for y in range(12, 16):
            for x in range(12, 16):
                grid[y][x] = 0
        groups = label_grid_to_ring_groups(grid, 0, 1.0, 1.0)
        assert len(groups) == 2

    def test_enclosed_background_becomes_a_hole(self):
        grid = [[BACKGROUND_LABEL] * 20 for _ in range(20)]
        for y in range(2, 18):
            for x in range(2, 18):
                grid[y][x] = 0
        for y in range(8, 12):
            for x in range(8, 12):
                grid[y][x] = BACKGROUND_LABEL
        groups = label_grid_to_ring_groups(grid, 0, 1.0, 1.0)
        assert len(groups) == 1
        assert len(groups[0]) == 2  # outer + one hole

    def test_tiny_speck_below_min_area_is_discarded(self):
        grid = [[BACKGROUND_LABEL] * 20 for _ in range(20)]
        grid[5][5] = 0  # single pixel speck
        groups = label_grid_to_ring_groups(grid, 0, 1.0, 1.0, min_area_px=4.0)
        assert groups == []

    def test_no_pixels_of_label_returns_empty(self):
        grid = [[BACKGROUND_LABEL] * 10 for _ in range(10)]
        assert label_grid_to_ring_groups(grid, 0, 1.0, 1.0) == []

    def test_coordinates_scaled_to_mm(self):
        grid = self._square_grid(20, 20, 0, 0, 0, 10, 10)
        groups = label_grid_to_ring_groups(grid, 0, 2.0, 3.0, simplify_tolerance_mm=0.01)
        outer = groups[0][0]
        xs = [p[0] for p in outer]
        ys = [p[1] for p in outer]
        assert max(xs) <= 20.0 + 1e-6  # 10px * 2.0mm/px
        assert max(ys) <= 30.0 + 1e-6  # 10px * 3.0mm/px
