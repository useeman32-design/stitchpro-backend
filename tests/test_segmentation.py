"""Tests for digitizer/segmentation/{color_quantize,polygons}.py."""

from __future__ import annotations

from PIL import Image, ImageDraw

from digitizer.segmentation.color_quantize import (
    BACKGROUND_LABEL,
    _merge_near_duplicate_colors,
    build_label_grid,
    despeckle_label_grid,
)
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

    def test_adjacent_design_colors_dont_leak_a_spurious_blend_color(self):
        # Regression test for a real, observed failure distinct from the
        # two tests above: those both involve exactly one real foreground
        # color blending into the *background*, which the blend-line merge
        # against `background_refs` already handled. This covers two
        # *different* real design colors that touch each other directly (a
        # red block overlapping a blue ellipse) -- downscaling/compression
        # produces a handful of blend pixels along that internal boundary
        # (e.g. a muddy red-purple that is neither red nor blue), and
        # without a merge rule that also checks segments between pairs of
        # *foreground* palette colors (not just against the background),
        # that blend survived as its own bogus "extra thread color" --
        # exactly the "color detection shows noisy extra colors beyond the
        # real count" complaint for any multi-color design. A generous
        # `max_colors` budget must not surface more than the real 2 colors
        # here, and must not grow as the budget grows.
        big = Image.new("RGB", (400, 400), (255, 255, 255))
        draw = ImageDraw.Draw(big)
        draw.rectangle([20, 20, 250, 250], fill=(230, 30, 40))
        draw.ellipse([150, 150, 380, 380], fill=(20, 90, 180))
        small = big.resize((100, 100), Image.LANCZOS)

        for max_colors in (2, 4, 6, 8):
            palette, grid = build_label_grid(small, max_colors=max_colors)
            used_labels = {v for row in grid for v in row if v != BACKGROUND_LABEL}
            assert len(used_labels) == min(2, max_colors), (
                f"max_colors={max_colors} produced {len(used_labels)} colors: {palette}"
            )


class TestMergeNearDuplicateColorsBackgroundRefCollapse:
    # Regression test for a real, observed failure distinct from the
    # anti-aliasing cases above: `image_to_vector_regions`'s
    # `bg_mask_override` path derives its background reference colors via
    # `reference_colors_from_mask`, which samples up to 16 shades across
    # the whole masked-background region (to handle gradients/shadows/
    # texture) rather than the one-or-few "pure" seed color(s) the
    # internal `remove_background=True` route tracks. On a background with
    # any gradient or shadow, one of those 16 sampled shades can
    # coincidentally land exactly on (or within `blend_tolerance` of) a
    # real foreground palette color. The blend-line check's "distance from
    # point `p` to the segment between `a` and `bg`" then degenerates: if
    # `bg` itself already equals `p`, the segment trivially terminates at
    # `p`, so the distance is 0 *regardless of which `a` is being tested*
    # -- not because `p` is really an anti-aliased blend between `a` and
    # the background, but merely because `p` happens to coincide with one
    # of the (many) sampled background shades. Left unguarded, that one
    # palette entry free-passes a merge with literally every other palette
    # entry in turn, collapsing the entire palette down to a single color
    # -- the exact "vectorize gives 1 color, image_to_design gives 7-8"
    # symptom observed on `manual-canvas.jpg` and `cap-logo.jpg`.
    def test_background_ref_coinciding_with_one_palette_color_does_not_collapse_the_rest(self):
        # Three clearly-separate "design" colors, none close to each
        # other. `(50, 50, 50)` stands in for the pathological background
        # reference that happens to exactly equal one of the palette
        # entries below (simulating `reference_colors_from_mask` sampling
        # a background gradient shade that collides with a real
        # foreground color) -- while `(255, 255, 255)` is a normal, non-
        # colliding background reference.
        palette = [(50, 50, 50), (10, 200, 10), (200, 10, 200)]
        merged, remap = _merge_near_duplicate_colors(
            palette,
            merge_tolerance=5.0,
            background_refs=((255, 255, 255), (50, 50, 50)),
            blend_tolerance=8.0,
        )
        # All three real design colors must survive as distinct entries --
        # the degenerate background-ref match on entry 0 must not drag the
        # unrelated entries 1 and 2 into one group with it.
        assert len(merged) == 3
        assert remap[0] != remap[1]
        assert remap[0] != remap[2]
        assert remap[1] != remap[2]

    def test_vectorize_bg_mask_override_path_matches_remove_background_path(self):
        # End-to-end version of the same bug via the public
        # `build_label_grid` entry point: a background with a subtle
        # gradient (so `reference_colors_from_mask`'s broader sampling
        # differs from the internal route's own reference) must still
        # produce the same number of foreground colors whichever way the
        # background mask was supplied.
        from digitizer.segmentation.background_removal import compute_background_mask

        img = Image.new("RGB", (60, 60), (240, 240, 240))
        for y in range(60):
            for x in range(60):
                if (x + y) % 7 == 0:
                    img.putpixel((x, y), (224, 224, 224))  # subtle bg gradient/texture
        draw = ImageDraw.Draw(img)
        draw.rectangle([5, 5, 25, 25], fill=(20, 90, 180))
        draw.rectangle([35, 35, 55, 55], fill=(200, 40, 40))

        rgba = img.convert("RGBA")
        mask = compute_background_mask(rgba)

        palette_internal, grid_internal = build_label_grid(
            img, max_colors=6, remove_background=True, background_blend_tolerance=0.0
        )
        palette_override, grid_override = build_label_grid(
            img, max_colors=6, bg_mask_override=mask, background_blend_tolerance=0.0
        )
        labels_internal = {v for row in grid_internal for v in row if v != BACKGROUND_LABEL}
        labels_override = {v for row in grid_override for v in row if v != BACKGROUND_LABEL}
        assert len(labels_internal) == len(labels_override) == 2


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


class TestDespeckleLabelGrid:
    def test_isolated_single_pixel_speckle_absorbed_into_surrounding_label(self):
        grid = [[0] * 10 for _ in range(10)]
        grid[5][5] = 1  # a single rogue pixel of a different label
        out = despeckle_label_grid(grid, min_component_px=4)
        assert out[5][5] == 0

    def test_large_region_is_left_untouched(self):
        grid = [[0] * 10 for _ in range(10)]
        for y in range(2, 8):
            for x in range(2, 8):
                grid[y][x] = 1
        out = despeckle_label_grid(grid, min_component_px=4)
        for y in range(2, 8):
            for x in range(2, 8):
                assert out[y][x] == 1

    def test_small_component_absorbed_into_majority_neighbor(self):
        # A 2x2 speckle of label 2 sitting entirely inside a sea of label 1
        # (itself inside label 0) should be absorbed into 1, the label
        # actually touching all of its border.
        grid = [[0] * 10 for _ in range(10)]
        for y in range(1, 9):
            for x in range(1, 9):
                grid[y][x] = 1
        grid[4][4] = 2
        grid[4][5] = 2
        grid[5][4] = 2
        grid[5][5] = 2
        out = despeckle_label_grid(grid, min_component_px=5)
        assert out[4][4] == 1 and out[5][5] == 1

    def test_empty_grid_does_not_crash(self):
        assert despeckle_label_grid([]) == []

    def test_background_label_participates_like_any_other_label(self):
        grid = [[BACKGROUND_LABEL] * 10 for _ in range(10)]
        for y in range(2, 8):
            for x in range(2, 8):
                grid[y][x] = 0
        grid[4][4] = BACKGROUND_LABEL  # tiny bg speck inside the foreground
        out = despeckle_label_grid(grid, min_component_px=4)
        assert out[4][4] == 0
