"""Tests for api/app.py's HTTP surface, using Flask's own test client (no
real network/process involved). Previously untested directly -- the other
98+ tests all exercise the digitizer/formats modules underneath it -- added
alongside the stitch_style/density_mm/angle_deg request-parsing changes
since that parsing logic (`_digitize_image_from_request`) didn't have any
dedicated coverage before."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from api.app import app


def _png_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([20, 20, 80, 80], fill=(30, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _client():
    app.config["TESTING"] = True
    return app.test_client()


class TestHealthAndFormats:
    def test_health(self):
        resp = _client().get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_formats(self):
        resp = _client().get("/api/formats")
        assert resp.status_code == 200
        body = resp.get_json()
        assert "dst" in body["formats"]


class TestImageToPoints:
    def test_default_stitch_style_is_fill(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_png_bytes()), "logo.png"),
            "width_mm": "40",
            "height_mm": "40",
        }
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert len(body["points"]) > 0
        assert len(body["threads"]) >= 1

    def test_cross_stitch_style_accepted(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_png_bytes()), "logo.png"),
            "width_mm": "40",
            "height_mm": "40",
            "stitch_style": "cross",
        }
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert len(resp.get_json()["points"]) > 0

    def test_response_includes_artwork_kind_and_confidence(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_png_bytes()), "logo.png"),
            "width_mm": "40",
            "height_mm": "40",
        }
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["artworkKind"] == "flat"
        assert 0.0 <= body["artworkConfidence"] <= 1.0

    def test_auto_vectorize_false_is_accepted_and_forces_flat(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_png_bytes()), "logo.png"),
            "width_mm": "40",
            "height_mm": "40",
            "auto_vectorize": "false",
        }
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.get_json()["artworkKind"] == "flat (forced)"

    def test_invalid_stitch_style_rejected(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_png_bytes()), "logo.png"),
            "width_mm": "40",
            "height_mm": "40",
            "stitch_style": "herringbone",
        }
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "stitch_style" in resp.get_json()["error"]

    def test_missing_image_rejected(self):
        client = _client()
        resp = client.post(
            "/api/design/image-to-points",
            data={"width_mm": "40", "height_mm": "40"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_missing_dimensions_rejected(self):
        client = _client()
        data = {"image": (io.BytesIO(_png_bytes()), "logo.png")}
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_angle_deg_accepted(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_png_bytes()), "logo.png"),
            "width_mm": "40",
            "height_mm": "40",
            "angle_deg": "30",
        }
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200

    def test_smooth_curves_false_accepted(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_png_bytes()), "logo.png"),
            "width_mm": "40",
            "height_mm": "40",
            "smooth_curves": "false",
        }
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200


class TestImageToFile:
    def test_cross_stitch_download(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_png_bytes()), "logo.png"),
            "width_mm": "40",
            "height_mm": "40",
            "stitch_style": "cross",
            "format": "dst",
        }
        resp = client.post("/api/design/image-to-file", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert len(resp.data) > 0
