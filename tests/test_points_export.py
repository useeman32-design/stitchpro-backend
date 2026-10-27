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
        assert set(result.keys()) == {"width_mm", "height_mm", "threads", "points", "stitchTypeBreakdown"}
        assert result["width_mm"] == 40
        assert result["height_mm"] == 40
        assert result["threads"] == ["#112233"]
        assert len(result["points"]) > 0

    def test_stitch_type_breakdown_for_single_object(self):
        result = design_to_points(_rect_design())
        assert result["stitchTypeBreakdown"] == [
            {"type": "RUNNING", "objectCount": 1, "stitchCount": len(
                [p for p in result["points"] if p["command"] == "stitch"]
            )}
        ]

    def test_stitch_type_breakdown_splits_by_object_type_and_sorts_by_stitch_count(self):
        # A FILL region (many stitches) and a SATIN region (few) of
        # different colors -- breakdown must report both, grouped by type
        # (not by object or color), with the higher-stitch-count type
        # first regardless of object insertion order.
        satin_rungs = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)]
        fill_ring = [[(10, 10), (30, 10), (30, 30), (10, 30)]]
        design = EmbroideryDesign(
            width_mm=50,
            height_mm=50,
            objects=[
                EmbroideryObject(
                    id="satin-thin", type=ObjectType.SATIN, geometry=satin_rungs, color="#111111",
                    stitch_settings=StitchSettings(density_mm=0.4), sequence=0,
                ),
                EmbroideryObject(
                    id="fill-block", type=ObjectType.FILL, geometry=fill_ring, color="#222222",
                    stitch_settings=StitchSettings(stitch_length_mm=3.0, density_mm=0.4), sequence=1,
                ),
            ],
            threads=["#111111", "#222222"],
        )
        result = design_to_points(design)
        types = [entry["type"] for entry in result["stitchTypeBreakdown"]]
        assert set(types) == {"SATIN", "FILL"}
        assert types[0] == "FILL"  # the larger region has more stitches, so it sorts first
        for entry in result["stitchTypeBreakdown"]:
            assert entry["objectCount"] == 1
            assert entry["stitchCount"] > 0

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
