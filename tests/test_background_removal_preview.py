"""Tests for the background-removal *preview* feature: the standalone
`digitizer.pipeline.remove_background_preview` function, the
`image_to_design(..., remove_background=...)` passthrough it's meant to stay
consistent with, and the `/api/design/remove-background` HTTP endpoint.

See digitizer/pipeline.py's `remove_background_preview` docstring for why
this exists as its own cheap, high-resolution, stitch-generation-free step:
a client is expected to show its output to the user and require an explicit
choice (proceed with the background removed, or keep the original) before
ever calling the real digitize/export endpoints -- never silently apply
background removal without the user seeing it first.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from api.app import app
from digitizer.pipeline import image_to_design, remove_background_preview


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _patch_on_flat_background() -> bytes:
    """A uniform off-white "background" with a solid, distinctly-colored
    square patch inset from every edge -- the simplest possible case for
    both the preview function and the remove_background passthrough."""
    img = Image.new("RGB", (100, 100), (250, 250, 248))
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, 70, 70], fill=(20, 110, 200))
    return _png_bytes(img)


def _client():
    app.config["TESTING"] = True
    return app.test_client()


class TestRemoveBackgroundPreviewFunction:
    def test_background_pixels_become_transparent(self):
        png_bytes = remove_background_preview(_patch_on_flat_background())
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        assert img.getpixel((2, 2))[3] == 0  # corner -- clearly background

    def test_foreground_patch_stays_opaque_with_original_color(self):
        png_bytes = remove_background_preview(_patch_on_flat_background())
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        r, g, b, a = img.getpixel((50, 50))  # center of the blue patch
        assert a == 255
        assert (r, g, b) == (20, 110, 200)

    def test_result_is_a_real_decodable_png(self):
        png_bytes = remove_background_preview(_patch_on_flat_background())
        img = Image.open(io.BytesIO(png_bytes))
        assert img.format == "PNG"
        assert img.mode in ("RGBA", "LA")

    def test_invalid_image_bytes_raise_value_error(self):
        try:
            remove_background_preview(b"not an image")
        except ValueError:
            return
        raise AssertionError("expected ValueError for undecodable image bytes")

    def test_higher_resolution_than_the_digitizing_pipeline_default(self):
        # A deliberately large flat image -- the digitizing pipeline would
        # downscale this to 150px; the preview should keep far more detail
        # since it's only ever shown to a human, never turned into stitches.
        img = Image.new("RGB", (1000, 400), (250, 250, 248))
        ImageDraw.Draw(img).rectangle([400, 100, 600, 300], fill=(20, 110, 200))
        png_bytes = remove_background_preview(_png_bytes(img))
        out = Image.open(io.BytesIO(png_bytes))
        assert max(out.size) > 150


class TestRemoveBackgroundFlagOnImageToDesign:
    def test_default_removes_background_as_before(self):
        design = image_to_design(_patch_on_flat_background(), width_mm=40, height_mm=40)
        # Only the patch's blue survives as a thread color -- the flat
        # off-white background never becomes an object.
        assert len(design.threads) == 1

    def test_remove_background_false_keeps_the_background_as_real_foreground(self):
        design = image_to_design(
            _patch_on_flat_background(), width_mm=40, height_mm=40, remove_background=False
        )
        # With background removal disabled, the off-white backdrop is a
        # real foreground color too -- so there are now at least two
        # thread colors (backdrop + patch) instead of just one.
        assert len(design.threads) >= 2


class TestRemoveBackgroundEndpoint:
    def test_returns_a_png_with_transparent_background(self):
        client = _client()
        resp = client.post(
            "/api/design/remove-background",
            data={"image": (io.BytesIO(_patch_on_flat_background()), "photo.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        assert resp.mimetype == "image/png"
        img = Image.open(io.BytesIO(resp.data)).convert("RGBA")
        assert img.getpixel((2, 2))[3] == 0
        assert img.getpixel((50, 50))[3] == 255

    def test_missing_image_field_is_a_400(self):
        client = _client()
        resp = client.post("/api/design/remove-background", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_max_dim_out_of_range_is_a_400(self):
        client = _client()
        resp = client.post(
            "/api/design/remove-background",
            data={
                "image": (io.BytesIO(_patch_on_flat_background()), "photo.png"),
                "max_dim": "99999",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_unreadable_image_bytes_is_a_400(self):
        client = _client()
        resp = client.post(
            "/api/design/remove-background",
            data={"image": (io.BytesIO(b"not an image"), "bad.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()


class TestRemoveBackgroundFlagOnImageToPointsEndpoint:
    def test_remove_background_false_passes_through_and_keeps_backdrop_color(self):
        client = _client()
        resp = client.post(
            "/api/design/image-to-points",
            data={
                "image": (io.BytesIO(_patch_on_flat_background()), "photo.png"),
                "width_mm": "40",
                "height_mm": "40",
                "remove_background": "false",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert len(body["threads"]) >= 2

    def test_remove_background_true_is_still_the_default(self):
        client = _client()
        resp = client.post(
            "/api/design/image-to-points",
            data={
                "image": (io.BytesIO(_patch_on_flat_background()), "photo.png"),
                "width_mm": "40",
                "height_mm": "40",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert len(body["threads"]) == 1
