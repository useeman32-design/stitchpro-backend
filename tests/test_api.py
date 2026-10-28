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

    def test_force_photo_field_overrides_classification(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_png_bytes()), "logo.png"),
            "width_mm": "40",
            "height_mm": "40",
            "force_photo": "true",
        }
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.get_json()["artworkKind"] == "photo (forced)"

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


def _puff_candidate_png_bytes() -> bytes:
    """A bold, blocky, moderate-size, non-dominant rounded shape -- real
    `image_to_design` heuristics (digitizer/stitches/puff/eligibility.py)
    classify this as PUFF_3D, same shape used by
    tests/test_new_stitch_types_pipeline.py's own PUFF scenario."""
    img = Image.new("RGB", (210, 70), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([5, 10, 60, 60], radius=8, fill=(10, 10, 10))
    d.rounded_rectangle([75, 10, 130, 60], radius=8, fill=(30, 30, 200))
    d.rounded_rectangle([145, 10, 200, 60], radius=8, fill=(30, 160, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestNewStitchTypesRealDownloadFlow:
    """New stitch type follow-up (Bean/Motif/3D Puff/Applique): the exact
    real-world flow stitch-pro's frontend uses (confirmed in
    src/utils/downloadStitchFile.ts / backendClient.ts) -- fetch a server-
    digitized design's point stream from /image-to-points, then feed those
    *exact* points into /points-to-file to get the real downloadable file.
    A PUFF_3D/APPLIQUE region's "stop" command must survive this whole
    round trip, or a real design containing one would fail to download.
    """

    def test_puff_design_points_stream_includes_a_real_stop_command(self):
        client = _client()
        data = {
            "image": (io.BytesIO(_puff_candidate_png_bytes()), "badge.png"),
            "width_mm": "60",
            "height_mm": "20",
        }
        resp = client.post("/api/design/image-to-points", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        commands = {p["command"] for p in body["points"]}
        assert "stop" in commands
        breakdown_types = {entry["type"] for entry in body["stitchTypeBreakdown"]}
        assert "PUFF_3D" in breakdown_types

    def test_puff_design_points_stream_downloads_as_a_real_dst_file(self):
        client = _client()
        points_data = {
            "image": (io.BytesIO(_puff_candidate_png_bytes()), "badge.png"),
            "width_mm": "60",
            "height_mm": "20",
        }
        points_resp = client.post(
            "/api/design/image-to-points", data=points_data, content_type="multipart/form-data"
        )
        assert points_resp.status_code == 200
        body = points_resp.get_json()
        assert "stop" in {p["command"] for p in body["points"]}

        download_resp = client.post(
            "/api/design/points-to-file",
            json={"points": body["points"], "colors": body["threads"], "format": "dst", "name": "badge"},
        )
        assert download_resp.status_code == 200
        assert len(download_resp.data) > 0

        from pyembroidery import read_dst

        pattern = read_dst(io.BytesIO(download_resp.data))
        assert len(pattern.stitches) > 0
