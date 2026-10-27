"""Tests for digitizer/segmentation/artworkify.py."""

from __future__ import annotations

import numpy as np
from PIL import Image

from digitizer.segmentation.artworkify import artworkify


def _smooth_gradient_rgba(alpha_stripe: bool = False) -> Image.Image:
    size = 120
    x = np.linspace(0, 255, size)
    y = np.linspace(0, 255, size)
    xx, yy = np.meshgrid(x, y)
    r = xx.astype(np.uint8)
    g = yy.astype(np.uint8)
    b = ((xx + yy) / 2).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    if alpha_stripe:
        alpha = np.full((size, size), 255, dtype=np.uint8)
        alpha[:10, :] = 0
        rgba = np.dstack([rgb, alpha])
        return Image.fromarray(rgba, "RGBA")
    return Image.fromarray(rgb, "RGB")


class TestArtworkify:
    def test_output_same_size_as_input(self):
        img = _smooth_gradient_rgba()
        out = artworkify(img)
        assert out.size == img.size

    def test_reduces_unique_color_count(self):
        img = _smooth_gradient_rgba()
        out = artworkify(img).convert("RGB")
        before = len(set(img.getdata()))
        after = len(set(out.getdata()))
        # Posterizing a smooth gradient must collapse it toward many fewer
        # distinct colors -- this is the entire point of the preprocessing
        # step (see module docstring: breaking the "everything blends into
        # everything" false-merge failure mode).
        assert after < before / 2

    def test_preserves_alpha_channel(self):
        img = _smooth_gradient_rgba(alpha_stripe=True)
        out = artworkify(img)
        assert out.mode == "RGBA"
        out_alpha = out.split()[3]
        in_alpha = img.split()[3]
        assert list(out_alpha.getdata()) == list(in_alpha.getdata())

    def test_rgb_input_without_alpha_stays_opaque_rgb(self):
        img = _smooth_gradient_rgba()
        out = artworkify(img)
        assert out.mode == "RGB"

    def test_is_roughly_stable_on_already_flat_input(self):
        # Posterizing already-flat colors should just snap them to the
        # nearest step -- a large solid block should stay a single color,
        # not fragment into several.
        img = Image.new("RGB", (60, 60), (120, 80, 200))
        out = artworkify(img)
        assert len(set(out.getdata())) == 1
