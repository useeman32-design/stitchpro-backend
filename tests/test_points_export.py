"""Tests for digitizer/points_export.py."""

from __future__ import annotations

import pytest

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.points_export import design_to_points


def _rect_design(color="#112233"):
    rect = [(5, 5), (25, 5), (25, 25), (5, 25)]
    return EmbroideryDesign(
        width_mm=40,
        height_mm=40,
        objects=[
            EmbroideryObject(
                id="r1",
                type=ObjectType.RUNNING,
                geometry=rect,
                color=color,
                stitch_settings=StitchSettings(stitch_length_mm=3.0),
                sequence=0,
            )
        ],
        threads=[color],
    )


class TestDesignToPoints:
    def test_basic_shape_returns_expected_keys(self):
        result = design_to_points(_rect_design())
        assert set(result.keys()) == {"width_mm", "height_mm", "threads", "points"}
        assert result["width_mm"] == 40
        assert result["height_mm"] == 40
        assert result["threads"] == ["#112233"]
        assert len(result["points"]) > 0

    def test_points_have_expected_shape(self):
        result = design_to_points(_rect_design())
        for p in result["points"]:
            assert set(p.keys()) == {"x", "y", "command"}
            assert p["command"] in ("jump", "stitch", "colorChange")
            assert isinstance(p["x"], int)
            assert isinstance(p["y"], int)

    def test_first_point_is_a_jump(self):
        result = design_to_points(_rect_design())
        assert result["points"][0]["command"] == "jump"

    def test_two_colors_produce_one_color_change_and_two_threads(self):
        rect_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
        rect_b = [(20, 20), (30, 20), (30, 30), (20, 30)]
        design = EmbroideryDesign(
            width_mm=50,
            height_mm=50,
            objects=[
                EmbroideryObject(
                    id="a", type=ObjectType.RUNNING, geometry=rect_a, color="#FF0000",
                    stitch_settings=StitchSettings(), sequence=0,
                ),
                EmbroideryObject(
                    id="b", type=ObjectType.RUNNING, geometry=rect_b, color="#0000FF",
                    stitch_settings=StitchSettings(), sequence=1,
                ),
            ],
            threads=["#FF0000", "#0000FF"],
        )
        result = design_to_points(design)
        assert result["threads"] == ["#FF0000", "#0000FF"]
        color_changes = [p for p in result["points"] if p["command"] == "colorChange"]
        assert len(color_changes) == 1

    def test_coordinates_centered_on_origin(self):
        # A design exactly filling its declared width/height should have
        # roughly symmetric point coordinates around 0 (0.1mm units).
        result = design_to_points(_rect_design())
        xs = [p["x"] for p in result["points"]]
        ys = [p["y"] for p in result["points"]]
        assert min(xs) < 0 < max(xs) or min(xs) <= 0
        assert min(ys) < 0 < max(ys) or min(ys) <= 0

    def test_empty_design_raises(self):
        design = EmbroideryDesign(width_mm=10, height_mm=10, objects=[], threads=[])
        with pytest.raises(ValueError):
            design_to_points(design)
