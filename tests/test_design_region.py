"""Tests for digitizer.segmentation.design_region.detect_design_region and
the /api/design/detect-region HTTP endpoint that exposes it.

See digitizer/segmentation/design_region.py's module docstring for the
full rationale: this is the "a photo of someone wearing the garment
should only extract the printed/stitched design, not the whole person +
plain fabric" step, distinct from background_removal.py's border-
connected flood fill.

These synthetic cases mirror the ones used ad hoc while building the
detector (see session notes): a busy patch on an otherwise flat garment,
a fully flat/design-free garment, a flat garment with only a lighting
gradient, and a flat logo that already fills the whole frame (acknowledged
out-of-scope case, handled at the endpoint layer via classify_artwork
gating rather than inside detect_design_region itself).
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageDraw

from api.app import app
from digitizer.segmentation.design_region import detect_design_region


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_MARGIN = 12  # thin, distinctly-colored border so compute_background_mask's
# border-connected flood fill has something to classify as background
# before the (uniformly-colored) "shirt" starts -- without this the whole
# frame is one single color/blob and nothing is foreground at all, which
# isn't representative of a real photo (there's always at least a sliver
# of differently-colored backdrop/skin visible somewhere near the edges).


def _busy_patch_on_flat_shirt() -> Image.Image:
    """A pale backdrop margin around a flat navy "shirt" with a noisy,
    high-detail patch inset in its middle -- the design-is-there case."""
    rng = random.Random(42)
    img = Image.new("RGB", (300, 300), (235, 235, 230))
    d = ImageDraw.Draw(img)
    d.rectangle([_MARGIN, _MARGIN, 299 - _MARGIN, 299 - _MARGIN], fill=(25, 35, 70))
    px = img.load()
    for y in range(90, 174):
        for x in range(81, 219):
            px[x, y] = (
                rng.randint(120, 255),
                rng.randint(0, 120),
                rng.randint(0, 120),
            )
    return img


def _flat_shirt_no_design() -> Image.Image:
    """Pale backdrop margin around a perfectly uniform navy "shirt" -- the
    no-design case; should NOT manufacture a fake sub-region."""
    img = Image.new("RGB", (300, 300), (235, 235, 230))
    d = ImageDraw.Draw(img)
    d.rectangle([_MARGIN, _MARGIN, 299 - _MARGIN, 299 - _MARGIN], fill=(25, 35, 70))
    return img


def _flat_shirt_lighting_gradient() -> Image.Image:
    """Pale backdrop margin around a smooth lighting gradient across an
    otherwise flat garment -- a real-world near-miss that must still
    resolve to "no design found", since a gradient isn't printed/stitched
    design detail."""
    img = Image.new("RGB", (300, 300), (235, 235, 230))
    px = img.load()
    for y in range(_MARGIN, 300 - _MARGIN):
        for x in range(_MARGIN, 300 - _MARGIN):
            shade = int(20 + 40 * (x / 299.0))
            px[x, y] = (shade, shade + 10, shade + 45)
    return img


def _flat_logo_fills_frame() -> Image.Image:
    """A flat-art logo (red circle + blue square, no garment/background at
    all) already filling the whole frame -- the acknowledged limitation
    case: detect_design_region itself may still find an internal color
    boundary here, which is why production gates this module behind
    classify_artwork and never calls it for 'flat'-classified uploads."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 100, 100], fill=(220, 30, 30))
    d.rectangle([100, 100, 190, 190], fill=(30, 60, 220))
    return img


def _client():
    app.config["TESTING"] = True
    return app.test_client()


class TestDetectDesignRegion:
    def test_finds_tight_box_around_busy_patch(self):
        result = detect_design_region(_busy_patch_on_flat_shirt())
        assert result["foreground_detected"] is True
        # The real patch spans roughly x:[0.27, 0.73], y:[0.30, 0.58] of
        # the frame -- box should bracket it reasonably tightly, not the
        # whole frame.
        assert 0.15 < result["x0"] < 0.35
        assert 0.65 < result["x1"] < 0.85
        assert 0.20 < result["y0"] < 0.40
        assert 0.48 < result["y1"] < 0.72
        assert result["confidence"] >= 0.5

    def test_flat_no_design_shirt_falls_back_to_full_box_low_confidence(self):
        result = detect_design_region(_flat_shirt_no_design())
        assert result["foreground_detected"] is True
        assert result["x0"] < 0.05
        assert result["y0"] < 0.05
        assert result["x1"] > 0.95
        assert result["y1"] > 0.95
        assert result["confidence"] <= 0.3

    def test_lighting_gradient_does_not_look_like_a_design(self):
        result = detect_design_region(_flat_shirt_lighting_gradient())
        assert result["foreground_detected"] is True
        assert result["x0"] < 0.05
        assert result["x1"] > 0.95
        assert result["confidence"] <= 0.3

    def test_never_raises_on_blank_image(self):
        result = detect_design_region(Image.new("RGB", (50, 50), (128, 128, 128)))
        assert result["foreground_detected"] is False
        assert result["confidence"] == 0.0
        assert result == {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0, "confidence": 0.0, "foreground_detected": False}

    def test_flat_logo_filling_frame_is_a_known_limitation_not_a_crash(self):
        # Documented as out of scope to "fix" inside this function itself
        # (see module docstring) -- production instead gates this whole
        # module behind classify_artwork at the endpoint layer (see
        # TestDetectRegionEndpoint.test_flat_art_is_skipped below). This
        # test just pins down that calling the function directly on flat
        # art still behaves sanely (returns a valid, in-bounds box) rather
        # than erroring, even though the box itself may not be meaningful
        # for this input type.
        result = detect_design_region(_flat_logo_fills_frame())
        assert result["foreground_detected"] is True
        assert 0.0 <= result["x0"] < result["x1"] <= 1.0
        assert 0.0 <= result["y0"] < result["y1"] <= 1.0


class TestDetectRegionEndpoint:
    def test_photo_like_image_returns_real_box(self):
        client = _client()
        resp = client.post(
            "/api/design/detect-region",
            data={
                "image": (io.BytesIO(_png_bytes(_busy_patch_on_flat_shirt())), "shirt.png"),
                "force_photo": "true",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["skip"] is False
        assert body["artworkKind"] == "photo (forced)"
        assert 0.0 <= body["x0"] < body["x1"] <= 1.0
        assert 0.0 <= body["y0"] < body["y1"] <= 1.0
        assert 0.0 <= body["confidence"] <= 1.0

    def test_flat_art_is_skipped(self):
        client = _client()
        resp = client.post(
            "/api/design/detect-region",
            data={"image": (io.BytesIO(_png_bytes(_flat_logo_fills_frame())), "logo.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["artworkKind"] == "flat"
        assert body["skip"] is True
        assert body["x0"] == 0.0 and body["y0"] == 0.0
        assert body["x1"] == 1.0 and body["y1"] == 1.0

    def test_missing_image_is_a_400(self):
        client = _client()
        resp = client.post("/api/design/detect-region", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_invalid_max_dim_is_a_400(self):
        client = _client()
        resp = client.post(
            "/api/design/detect-region",
            data={
                "image": (io.BytesIO(_png_bytes(_flat_shirt_no_design())), "shirt.png"),
                "max_dim": "99999999",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()
