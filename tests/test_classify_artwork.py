"""Tests for digitizer/classifier/classify_artwork.py."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from digitizer.classifier.classify_artwork import classify_artwork


def _flat_logo() -> Image.Image:
    """A clean, flat-color synthetic logo: solid fills, crisp edges, no
    gradients/noise -- the kind of thing a vector export (Illustrator,
    Canva, Figma) produces."""
    img = Image.new("RGB", (300, 300), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([40, 40, 260, 260], fill=(20, 60, 160))
    d.polygon([(150, 80), (210, 220), (90, 220)], fill=(240, 200, 30))
    return img


def _photo_like() -> Image.Image:
    """A smooth-gradient-plus-noise synthetic photo: continuous lighting
    variation and per-pixel sensor-noise-like texture, nothing a flat
    vector export would ever produce."""
    size = 300
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)
    base = np.stack(
        [
            60 + 140 * xx + 60 * np.sin(yy * 14) + 40 * np.sin(xx * 23),
            50 + 150 * yy + 50 * np.cos(xx * 11) + 35 * np.cos(yy * 19),
            80 + 120 * (xx * yy) + 45 * np.sin((xx + yy) * 17),
        ],
        axis=-1,
    )
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 22, base.shape)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


class TestClassifyArtwork:
    def test_flat_synthetic_logo_is_classified_flat(self):
        result = classify_artwork(_flat_logo())
        assert result["kind"] == "flat"
        assert result["confidence"] > 0.5

    def test_noisy_gradient_photo_is_classified_photo(self):
        result = classify_artwork(_photo_like())
        assert result["kind"] == "photo"

    def test_returns_metrics_dict(self):
        result = classify_artwork(_flat_logo())
        assert set(result["metrics"]) == {"unique_color_ratio", "mean_gradient", "soft_edge_fraction"}

    def test_tiny_image_does_not_crash(self):
        img = Image.new("RGB", (1, 1), (10, 10, 10))
        result = classify_artwork(img)
        assert result["kind"] in ("flat", "photo")

    def test_confidence_is_in_unit_range(self):
        for img in (_flat_logo(), _photo_like()):
            result = classify_artwork(img)
            assert 0.0 <= result["confidence"] <= 1.0
