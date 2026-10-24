"""Tests for digitizer/segmentation/{color_quantize,polygons}.py."""

from __future__ import annotations

from PIL import Image, ImageDraw

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

    def test_two_distinct_colors_survive_anti_aliased_downscale(self):
        # Regression test: a large image with two clearly distinct
        # foreground colors, downscaled with LANCZOS (the same resample
        # pipeline.py's load_and_downscale uses), produces a halo of
        # subtly-different near-white anti-aliased edge pixels. With a
        # small max_colors budget, MEDIANCUT used to waste palette slots
        # splitting that halo and collapse both real foreground colors into
        # a single muddy average bucket. Both colors must survive.
        big = Image.new("RGB", (300, 300), (255, 255, 255))
        navy = (58, 49, 120)
        gold = (212, 170, 80)
        for y in range(40, 140):
            for x in range(40, 260):
                big.putpixel((x, y), navy)
        for y in range(160, 260):
            for x in range(40, 260):
                big.putpixel((x, y), gold)
        small = big.resize((60, 60), Image.LANCZOS)

        palette, grid = build_label_grid(small, max_colors=4)
        used_labels = {v for row in grid for v in row if v != BACKGROUND_LABEL}
        used_colors = {palette[i] for i in used_labels}

        def _closest(target):
            return min(
                used_colors,
                key=lambda h: sum(
                    (a - b) ** 2
                    for a, b in zip(
                        (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)), target
                    )
                ),
            )

        assert len(used_colors) >= 2
        navy_match = _closest(navy)
        gold_match = _closest(gold)
        assert navy_match != gold_match

    def test_exact_duplicate_quantize_colors_are_merged_into_one_label(self):
        # Directly exercises _merge_near_duplicate_colors's actual target:
        # two near-identical (not just "somewhat close") colors that
        # MEDIANCUT can allocate separate palette slots for even though
        # they're visually the same thread color. The broader "one solid
        # shape doesn't fragment into multiple *geometric* objects" claim
        # (which also depends on a genuinely-different blend-ring color
        # being filtered out as a thin sliver, a job for
        # label_grid_to_ring_groups's `min_bbox_mm`, not this function) is
        # covered end-to-end in test_pipeline.py.
        img = Image.new("RGB", (20, 20), (255, 255, 255))
        for y in range(5, 15):
            for x in range(5, 10):
                img.putpixel((x, y), (20, 60, 160))
            for x in range(10, 15):
                img.putpixel((x, y), (21, 61, 161))  # 1 unit off, same thread in practice
        palette, grid = build_label_grid(img, max_colors=4)
        used_labels = {v for row in grid for v in row if v != BACKGROUND_LABEL}
        assert len(used_labels) == 1

    def test_heavily_anti_aliased_thin_stroke_stays_one_color(self):
        # Regression test for a real, observed failure: a thin shape (a
        # font glyph's stroke, a logo's outline) is mostly *edge* relative
        # to its own area, so downscaling it can produce a hundred-plus
        # distinct intermediate gray shades between "full ink" and "full
        # background" even though only one real design color is involved
        # -- unlike test_two_distinct_colors_survive_anti_aliased_downscale
        # above, where the halo is a thin minority around large flat
        # interior blocks. Before the blend-line merge pass, a small
        # `max_colors` budget (as low as the default 6) forced MEDIANCUT to
        # spend its whole budget on different points along that one
        # gradient, fragmenting one solid-black stroke into several
        # unmerged near-black/gray "thread colors" -- exactly the "fill
        # isn't solid, looks like scattered lines" symptom reported against
        # real text/logo designs.
        big = Image.new("RGB", (400, 400), (255, 255, 255))
        draw = ImageDraw.Draw(big)
        draw.line([(20, 20), (380, 380)], fill=(20, 20, 20), width=14)
        small = big.resize((100, 100), Image.LANCZOS)

        fg_pixel_colors = {
            small.getpixel((x, y))
            for y in range(small.height)
            for x in range(small.width)
            if small.getpixel((x, y)) != (255, 255, 255)
        }
        # Sanity check the test fixture actually exercises heavy
        # anti-aliasing (many distinct intermediate shades relative to a
        # thin stroke's small pixel count), not just a couple of edge
        # pixels -- otherwise this wouldn't be reproducing the real-world
        # failure at all.
        assert len(fg_pixel_colors) > 30

        palette, grid = build_label_grid(small, max_colors=6)
        used_labels = {v for row in grid for v in row if v != BACKGROUND_LABEL}
        assert len(used_labels) == 1


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

    def test_thin_long_sliver_below_min_bbox_mm_is_discarded(self):
        # Regression test: a one-pixel-wide but many-pixels-long sliver
        # (the shape left behind by an anti-aliased-edge pixel band that
        # ended up in its own color bucket -- see color_quantize.py's
        # _merge_near_duplicate_colors module docstring) has plenty of
        # pixel-space *area* to survive `min_area_px`, but is still just a
        # hairline streak that was never a real, stitchable design feature.
        # At 0.1mm/px, a 1x20px sliver is 0.1mm x 2.0mm -- its shorter side
        # must trip `min_bbox_mm`'s default (0.6mm).
        grid = [[BACKGROUND_LABEL] * 25 for _ in range(25)]
        for y in range(2, 22):
            grid[y][10] = 0  # 1px wide, 20px long vertical sliver
        groups = label_grid_to_ring_groups(grid, 0, 0.1, 0.1, min_area_px=0.5)
        assert groups == []

    def test_small_round_dot_above_min_bbox_mm_survives(self):
        # A genuinely small but roughly square/round design detail (e.g. a
        # letter's dot) should NOT be swept up by the same filter -- only
        # long, hair-thin slivers are the target.
        grid = self._square_grid(20, 20, 0, 8, 8, 12, 12)  # 4x4px square
        groups = label_grid_to_ring_groups(grid, 0, 1.0, 1.0)  # 4mm x 4mm
        assert len(groups) == 1

    def test_coordinates_scaled_to_mm(self):
        grid = self._square_grid(20, 20, 0, 0, 0, 10, 10)
        groups = label_grid_to_ring_groups(grid, 0, 2.0, 3.0, simplify_tolerance_mm=0.01)
        outer = groups[0][0]
        xs = [p[0] for p in outer]
        ys = [p[1] for p in outer]
        assert max(xs) <= 20.0 + 1e-6  # 10px * 2.0mm/px
        assert max(ys) <= 30.0 + 1e-6  # 10px * 3.0mm/px
