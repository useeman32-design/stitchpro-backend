"""Tests for the editable-vector pipeline (digitizer/pipeline.py's
image_to_vector_regions + design_from_region_records, and api/app.py's
/api/design/vectorize + /api/design/regions-to-points + /api/design/
regions-to-file routes).

Brief: "whichever background-removal choice the user makes must be
converted into a real, editable vector, not just handed to the digitizer
as a raster" plus "an adjustable-size brush to manually restore or erase
parts of the background-removal mask" plus "select/delete/redraw vector
regions after generation".
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image, ImageDraw

from api.app import app
from digitizer.pipeline import design_from_region_records, image_to_design, image_to_vector_regions
from digitizer.segmentation.background_removal import apply_mask_strokes


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _two_color_png() -> bytes:
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 50, 50], fill=(200, 30, 30))
    d.rectangle([60, 60, 90, 90], fill=(30, 30, 200))
    return _png_bytes(img)


def _solid_box_png() -> bytes:
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 80, 80], fill=(200, 30, 30))
    return _png_bytes(img)


def _client():
    app.config["TESTING"] = True
    return app.test_client()


class TestImageToVectorRegions:
    def test_returns_one_region_per_color(self):
        result = image_to_vector_regions(_two_color_png(), width_mm=50, height_mm=50, max_colors=4)
        assert len(result["regions"]) == 2
        colors = {r["color_hex"] for r in result["regions"]}
        assert len(colors) == 2

    def test_region_ids_match_image_to_design_object_ids(self):
        result = image_to_vector_regions(_two_color_png(), width_mm=50, height_mm=50, max_colors=4)
        design = image_to_design(_two_color_png(), width_mm=50, height_mm=50, max_colors=4)
        assert {r["id"] for r in result["regions"]} == {o.id for o in design.objects}

    def test_rings_are_plain_mm_coordinate_lists(self):
        result = image_to_vector_regions(_two_color_png(), width_mm=50, height_mm=50, max_colors=4)
        region = result["regions"][0]
        assert isinstance(region["rings"], list)
        ring = region["rings"][0]
        assert len(ring) >= 3
        x, y = ring[0]
        assert isinstance(x, float) and isinstance(y, float)

    def test_width_height_mm_echoed_back(self):
        result = image_to_vector_regions(_two_color_png(), width_mm=73, height_mm=41, max_colors=4)
        assert result["width_mm"] == 73
        assert result["height_mm"] == 41

    def test_blank_image_raises_value_error(self):
        img = Image.new("RGB", (50, 50), (255, 255, 255))
        with pytest.raises(ValueError):
            image_to_vector_regions(_png_bytes(img), width_mm=40, height_mm=40)

    def test_remove_background_false_keeps_background_as_a_region(self):
        # A fully-opaque image with remove_background=False treats the
        # white backdrop as foreground too (same contract as image_to_design).
        result = image_to_vector_regions(
            _solid_box_png(), width_mm=50, height_mm=50, max_colors=4, remove_background=False
        )
        colors = {r["color_hex"] for r in result["regions"]}
        assert "#ffffff" in colors

    def test_erase_brush_stroke_shrinks_a_region(self):
        baseline = image_to_vector_regions(_solid_box_png(), width_mm=50, height_mm=50, max_colors=4)
        baseline_area = baseline["regions"][0]["area_mm2"]

        strokes = [{"mode": "erase", "radius": 0.15, "points": [[0.2, 0.5], [0.5, 0.5]]}]
        corrected = image_to_vector_regions(
            _solid_box_png(), width_mm=50, height_mm=50, max_colors=4, bg_strokes=strokes
        )
        corrected_area = corrected["regions"][0]["area_mm2"]
        assert corrected_area < baseline_area

    def test_restore_brush_stroke_can_recover_erased_background(self):
        # Erase the whole foreground box with erase strokes, then restore
        # a chunk of it back -- net result should still have a foreground
        # region covering roughly the restored area, not zero regions.
        erase_everything = [{"mode": "erase", "radius": 0.5, "points": [[0.5, 0.5]]}]
        with pytest.raises(ValueError):
            image_to_vector_regions(
                _solid_box_png(), width_mm=50, height_mm=50, max_colors=4, bg_strokes=erase_everything
            )

        restore_some = erase_everything + [{"mode": "restore", "radius": 0.1, "points": [[0.5, 0.5]]}]
        result = image_to_vector_regions(
            _solid_box_png(), width_mm=50, height_mm=50, max_colors=4, bg_strokes=restore_some
        )
        assert len(result["regions"]) >= 1


class TestApplyMaskStrokes:
    def test_no_strokes_returns_same_mask_values(self):
        mask = [[False, False], [False, False]]
        out = apply_mask_strokes(mask, 2, 2, [])
        assert out == mask

    def test_erase_stroke_sets_pixels_true(self):
        mask = [[False] * 10 for _ in range(10)]
        out = apply_mask_strokes(mask, 10, 10, [{"mode": "erase", "radius": 0.3, "points": [[0.5, 0.5]]}])
        assert out[5][5] is True
        assert out[0][0] is False  # far corner untouched

    def test_restore_stroke_sets_pixels_false(self):
        mask = [[True] * 10 for _ in range(10)]
        out = apply_mask_strokes(mask, 10, 10, [{"mode": "restore", "radius": 0.3, "points": [[0.5, 0.5]]}])
        assert out[5][5] is False
        assert out[0][0] is True

    def test_does_not_mutate_input_mask(self):
        mask = [[False] * 10 for _ in range(10)]
        apply_mask_strokes(mask, 10, 10, [{"mode": "erase", "radius": 0.3, "points": [[0.5, 0.5]]}])
        assert mask[5][5] is False

    def test_later_stroke_wins_over_earlier_one(self):
        mask = [[False] * 10 for _ in range(10)]
        strokes = [
            {"mode": "erase", "radius": 0.3, "points": [[0.5, 0.5]]},
            {"mode": "restore", "radius": 0.3, "points": [[0.5, 0.5]]},
        ]
        out = apply_mask_strokes(mask, 10, 10, strokes)
        assert out[5][5] is False


class TestDesignFromRegionRecords:
    def test_round_trips_through_vectorize(self):
        vec = image_to_vector_regions(_two_color_png(), width_mm=50, height_mm=50, max_colors=4)
        design = design_from_region_records(vec["regions"], width_mm=50, height_mm=50)
        assert len(design.objects) == 2
        assert set(design.threads) == {r["color_hex"] for r in vec["regions"]}

    def test_deleting_a_region_removes_its_object(self):
        vec = image_to_vector_regions(_two_color_png(), width_mm=50, height_mm=50, max_colors=4)
        kept = vec["regions"][:1]
        design = design_from_region_records(kept, width_mm=50, height_mm=50)
        assert len(design.objects) == 1
        assert design.objects[0].color == kept[0]["color_hex"]

    def test_empty_regions_raises_value_error(self):
        with pytest.raises(ValueError):
            design_from_region_records([], width_mm=50, height_mm=50)

    def test_degenerate_region_raises_value_error(self):
        with pytest.raises(ValueError):
            design_from_region_records(
                [{"color_hex": "#ff0000", "rings": [[[0, 0], [1, 1]]]}],  # only 2 points, not a polygon
                width_mm=50,
                height_mm=50,
            )

    def test_invalid_stitch_style_raises_value_error(self):
        vec = image_to_vector_regions(_two_color_png(), width_mm=50, height_mm=50, max_colors=4)
        with pytest.raises(ValueError):
            design_from_region_records(vec["regions"], width_mm=50, height_mm=50, stitch_style="not-a-style")

    def test_produces_a_valid_exportable_dst(self):
        from formats.writer import design_to_format_bytes
        from pyembroidery import read_dst

        vec = image_to_vector_regions(_two_color_png(), width_mm=50, height_mm=50, max_colors=4)
        design = design_from_region_records(vec["regions"], width_mm=50, height_mm=50)
        dst_bytes = design_to_format_bytes(design, "dst")
        pattern = read_dst(io.BytesIO(dst_bytes))
        assert len(pattern.stitches) > 0


class TestVectorizeEndpoint:
    def test_happy_path(self):
        client = _client()
        resp = client.post(
            "/api/design/vectorize",
            data={
                "image": (io.BytesIO(_two_color_png()), "logo.png"),
                "width_mm": "50",
                "height_mm": "50",
                "max_colors": "4",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert len(body["regions"]) == 2
        assert body["widthMm"] == 50
        for r in body["regions"]:
            assert set(r.keys()) == {"id", "colorHex", "areaMm2", "rings"}

    def test_missing_image_is_400(self):
        client = _client()
        resp = client.post(
            "/api/design/vectorize", data={"width_mm": "50", "height_mm": "50"}, content_type="multipart/form-data"
        )
        assert resp.status_code == 400

    def test_bg_strokes_accepted_and_change_output(self):
        client = _client()
        baseline = client.post(
            "/api/design/vectorize",
            data={"image": (io.BytesIO(_solid_box_png()), "logo.png"), "width_mm": "50", "height_mm": "50"},
            content_type="multipart/form-data",
        ).get_json()

        strokes = json.dumps([{"mode": "erase", "radius": 0.15, "points": [[0.2, 0.5], [0.5, 0.5]]}])
        corrected = client.post(
            "/api/design/vectorize",
            data={
                "image": (io.BytesIO(_solid_box_png()), "logo.png"),
                "width_mm": "50",
                "height_mm": "50",
                "bg_strokes": strokes,
            },
            content_type="multipart/form-data",
        ).get_json()
        assert corrected["regions"][0]["areaMm2"] < baseline["regions"][0]["areaMm2"]

    def test_invalid_bg_strokes_json_is_400(self):
        client = _client()
        resp = client.post(
            "/api/design/vectorize",
            data={
                "image": (io.BytesIO(_two_color_png()), "logo.png"),
                "width_mm": "50",
                "height_mm": "50",
                "bg_strokes": "{not valid json",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400


class TestRegionsToPointsEndpoint:
    def test_happy_path(self):
        client = _client()
        vec = client.post(
            "/api/design/vectorize",
            data={"image": (io.BytesIO(_two_color_png()), "logo.png"), "width_mm": "50", "height_mm": "50"},
            content_type="multipart/form-data",
        ).get_json()

        resp = client.post(
            "/api/design/regions-to-points",
            json={"widthMm": 50, "heightMm": 50, "regions": vec["regions"]},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert len(body["points"]) > 0
        assert len(body["threads"]) == 2

    def test_deleted_region_reduces_stitch_count(self):
        client = _client()
        vec = client.post(
            "/api/design/vectorize",
            data={"image": (io.BytesIO(_two_color_png()), "logo.png"), "width_mm": "50", "height_mm": "50"},
            content_type="multipart/form-data",
        ).get_json()

        full = client.post(
            "/api/design/regions-to-points", json={"widthMm": 50, "heightMm": 50, "regions": vec["regions"]}
        ).get_json()
        partial = client.post(
            "/api/design/regions-to-points", json={"widthMm": 50, "heightMm": 50, "regions": vec["regions"][:1]}
        ).get_json()
        assert len(partial["points"]) < len(full["points"])
        assert len(partial["threads"]) == 1

    def test_missing_regions_is_400(self):
        client = _client()
        resp = client.post("/api/design/regions-to-points", json={"widthMm": 50, "heightMm": 50})
        assert resp.status_code == 400

    def test_non_json_body_is_400(self):
        client = _client()
        resp = client.post("/api/design/regions-to-points", data="not json")
        assert resp.status_code == 400


class TestRegionsToFileEndpoint:
    def test_happy_path_downloads_a_dst(self):
        client = _client()
        vec = client.post(
            "/api/design/vectorize",
            data={"image": (io.BytesIO(_two_color_png()), "logo.png"), "width_mm": "50", "height_mm": "50"},
            content_type="multipart/form-data",
        ).get_json()

        resp = client.post(
            "/api/design/regions-to-file",
            json={"widthMm": 50, "heightMm": 50, "regions": vec["regions"], "format": "dst", "name": "my-design"},
        )
        assert resp.status_code == 200
        assert "my-design.dst" in resp.headers.get("Content-Disposition", "")
        assert len(resp.data) > 0

    def test_unsupported_format_is_400(self):
        client = _client()
        resp = client.post(
            "/api/design/regions-to-file",
            json={"widthMm": 50, "heightMm": 50, "regions": [{"colorHex": "#f00", "rings": [[[0, 0]]]}], "format": "xyz"},
        )
        assert resp.status_code == 400
