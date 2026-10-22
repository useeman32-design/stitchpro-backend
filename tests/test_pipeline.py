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
