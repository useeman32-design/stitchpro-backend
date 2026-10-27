"""Tests for digitizer/pipeline.py -- the Milestone 5 image -> design path."""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from digitizer.model import ObjectType
from digitizer.pipeline import image_to_design
from formats.dst.writer import compile_design
from formats.writer import design_to_format_bytes
from pyembroidery import read_dst


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _letter_o_png() -> bytes:
    img = Image.new("RGB", (120, 120), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 110, 110], fill=(20, 20, 20))
    d.ellipse([35, 35, 85, 85], fill=(255, 255, 255))
    return _png_bytes(img)


def _two_color_png() -> bytes:
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 50, 50], fill=(200, 30, 30))
    d.rectangle([60, 60, 90, 90], fill=(30, 30, 200))
    return _png_bytes(img)


class TestImageToDesign:
    def test_letter_o_produces_one_fill_object_with_a_hole(self):
        design = image_to_design(_letter_o_png(), width_mm=40, height_mm=40, max_colors=3)
        assert len(design.objects) == 1
        obj = design.objects[0]
        assert obj.type == ObjectType.FILL
        assert len(obj.geometry) == 2  # outer + hole

    def test_two_separate_colors_produce_two_objects_and_threads(self):
        design = image_to_design(_two_color_png(), width_mm=50, height_mm=50, max_colors=4)
        assert len(design.objects) == 2
        assert len(design.threads) == 2
        assert {o.color for o in design.objects} == set(design.threads)

    def test_design_dimensions_match_requested_size(self):
        design = image_to_design(_two_color_png(), width_mm=73, height_mm=41, max_colors=4)
        assert design.width_mm == 73
        assert design.height_mm == 41

    def test_blank_image_raises_value_error(self):
        img = Image.new("RGB", (50, 50), (255, 255, 255))
        with pytest.raises(ValueError):
            image_to_design(_png_bytes(img), width_mm=30, height_mm=30)

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            image_to_design(_two_color_png(), width_mm=0, height_mm=30)

    def test_bad_image_bytes_raise_value_error(self):
        with pytest.raises(ValueError):
            image_to_design(b"not an image", width_mm=30, height_mm=30)

    def test_max_colors_out_of_range_raises(self):
        with pytest.raises(ValueError):
            image_to_design(_two_color_png(), width_mm=30, height_mm=30, max_colors=0)

    def test_round_trips_through_dst(self):
        design = image_to_design(_letter_o_png(), width_mm=40, height_mm=40, max_colors=3)
        pattern = compile_design(design)
        assert len(pattern.stitches) > 0

    def test_round_trips_through_every_supported_format(self):
        design = image_to_design(_two_color_png(), width_mm=40, height_mm=40, max_colors=4)
        for fmt in ("dst", "pes", "jef", "vp3", "exp"):
            data = design_to_format_bytes(design, fmt)
            assert len(data) > 0

    def test_dst_bytes_are_well_formed(self):
        design = image_to_design(_letter_o_png(), width_mm=40, height_mm=40, max_colors=3)
        data = design_to_format_bytes(design, "dst")
        pattern = read_dst(io.BytesIO(data))
        assert len(pattern.stitches) > 0

    def test_cross_stitch_style_produces_cross_stitch_objects(self):
        design = image_to_design(
            _two_color_png(), width_mm=50, height_mm=50, max_colors=4, stitch_style="cross"
        )
        assert len(design.objects) == 2
        assert all(o.type == ObjectType.CROSS_STITCH for o in design.objects)
        pattern = compile_design(design)
        assert len(pattern.stitches) > 0

    def test_unknown_stitch_style_raises(self):
        with pytest.raises(ValueError):
            image_to_design(_two_color_png(), width_mm=30, height_mm=30, stitch_style="zigzag")

    def test_density_none_picks_a_style_appropriate_default(self):
        fill_design = image_to_design(_two_color_png(), width_mm=30, height_mm=30, stitch_style="fill")
        cross_design = image_to_design(_two_color_png(), width_mm=30, height_mm=30, stitch_style="cross")
        assert fill_design.objects[0].stitch_settings.density_mm == pytest.approx(0.4)
        assert cross_design.objects[0].stitch_settings.density_mm == pytest.approx(1.8)

    def test_explicit_density_overrides_the_style_default(self):
        design = image_to_design(
            _two_color_png(), width_mm=30, height_mm=30, stitch_style="cross", density_mm=3.0
        )
        assert design.objects[0].stitch_settings.density_mm == pytest.approx(3.0)

    def test_downscaled_solid_circle_does_not_fragment_into_multiple_objects(self):
        # End-to-end regression test for the combination of fixes that
        # together solve the user-reported "fill isn't solid" symptom:
        # LANCZOS downscaling (pipeline.py's load_and_downscale, triggered
        # here by using a large source image against the default max_dim)
        # creates a thin anti-aliased blend ring around any solid-color
        # shape's edge. Without _merge_near_duplicate_colors (color_
        # quantize.py) and the min_bbox_mm thin-sliver filter (polygons.py)
        # working together, that ring could show up as its own small
        # "object" or two near-identical "colors" splitting what should be
        # one clean fill region -- exactly what a real user's logo photo
        # hits. A single solid-colored circle must always come back as
        # exactly one FILL object.
        img = Image.new("RGB", (300, 300), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.ellipse([50, 50, 250, 250], fill=(20, 60, 160))
        design = image_to_design(_png_bytes(img), width_mm=50, height_mm=50, max_colors=4)
        assert len(design.objects) == 1

    def test_smooth_curves_false_skips_smoothing(self):
        # Both should still produce valid, stitchable designs either way --
        # this just confirms the flag is actually wired through, not that
        # one specific smoothing outcome is "correct" (that's smoothing.py's
        # own test module's job).
        smoothed = image_to_design(_letter_o_png(), width_mm=40, height_mm=40, smooth_curves=True)
        unsmoothed = image_to_design(_letter_o_png(), width_mm=40, height_mm=40, smooth_curves=False)
        assert len(smoothed.objects) == 1
        assert len(unsmoothed.objects) == 1


class TestAutoVectorize:
    """Tests for the `auto_vectorize` classify-then-artworkify step (see
    pipeline.py's module docstring) and the `artwork_kind`/
    `artwork_confidence` metadata it records."""

    def test_default_on_clean_flat_art_records_flat_kind(self):
        # Every existing synthetic test image in this module is flat,
        # solid-color art -- auto_vectorize defaulting to True must not
        # change their behavior at all (classify_artwork should correctly
        # recognize them as 'flat' and skip artworkify).
        design = image_to_design(_two_color_png(), width_mm=40, height_mm=40)
        assert design.metadata["artwork_kind"] == "flat"
        assert len(design.threads) == 2

    def test_auto_vectorize_false_skips_classification(self):
        design = image_to_design(_two_color_png(), width_mm=40, height_mm=40, auto_vectorize=False)
        assert design.metadata["artwork_kind"] == "flat (forced)"
        assert design.metadata["artwork_confidence"] == 1.0
        assert len(design.threads) == 2

    def test_photo_like_gradient_background_no_longer_collapses_to_one_color(self):
        # Regression test for the color-collapse bug this segment's work
        # fixes: a design sitting on a smoothly-shaded (photographic-style)
        # background used to combinatorially false-merge every foreground
        # color down to one via color_quantize.py's blend-line heuristic
        # (see pipeline.py/color_quantize.py docstrings). Two clearly
        # distinct, well-separated foreground colors over such a background
        # must still come back as two distinct threads once auto_vectorize
        # classifies this as a photo and routes it through artworkify with
        # a tightened background_blend_tolerance.
        size = 200
        img = Image.new("RGB", (size, size), (0, 0, 0))
        px = img.load()
        for y in range(size):
            for x in range(size):
                # Smooth two-axis gradient background with a touch of
                # per-pixel noise -- enough to read as "photographic" to
                # classify_artwork without needing a real JPEG.
                shade = int(40 + 180 * (x / size) * (y / size) + ((x * 7 + y * 13) % 37))
                px[x, y] = (shade, shade, min(255, shade + 40))
        d = ImageDraw.Draw(img)
        d.rectangle([20, 20, 80, 180], fill=(220, 30, 30))
        d.rectangle([120, 20, 180, 180], fill=(30, 30, 220))
        design = image_to_design(_png_bytes(img), width_mm=50, height_mm=50, max_colors=6)
        assert design.metadata["artwork_kind"] == "photo"
        assert len(design.threads) >= 2

    def test_metadata_always_present(self):
        design = image_to_design(_letter_o_png(), width_mm=40, height_mm=40)
        assert "artwork_kind" in design.metadata
        assert "artwork_confidence" in design.metadata
