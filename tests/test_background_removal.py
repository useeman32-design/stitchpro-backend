"""Tests for digitizer/segmentation/background_removal.py.

The key scenario this module exists for -- and that the *old* single
global-color-distance approach (see color_quantize.py's git history)
could not handle -- is a real photo of a design on a real garment: the
"background" (the shirt/material) has lighting falloff or shading across
the frame, not one flat color, while the design itself is a small,
locally-distinct region fully inset from the frame edges.
"""

from __future__ import annotations

from PIL import Image

from digitizer.segmentation.background_removal import compute_background_mask


def _gradient_with_centered_patch(size=100, patch=(35, 35, 65, 65), patch_rgb=(20, 140, 60)):
    """A background that ramps from dark gray to near-white left-to-right
    (simulating uneven studio/garment lighting -- each step is small, but
    the total range from one edge to the other is large) with a solid,
    distinctly-colored patch in the middle standing in for a real design."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            v = int(60 + (195 * x / (size - 1)))  # 60 -> 255 left to right
            px[x, y] = (v, v, v)
    x0, y0, x1, y1 = patch
    for y in range(y0, y1):
        for x in range(x0, x1):
            px[x, y] = patch_rgb
    return img


class TestComputeBackgroundMask:
    def test_wide_gradient_background_is_entirely_removed(self):
        img = _gradient_with_centered_patch()
        mask = compute_background_mask(img)
        # Every border pixel must be classified as background despite the
        # left edge (~60,60,60) and right edge (~255,255,255) of the same
        # image being far more than any single fixed tolerance apart --
        # this is exactly the case a single global-color distance test
        # cannot handle, but a local-step flood fill can.
        size = img.size[0]
        for x in range(size):
            assert mask[0][x] is True
            assert mask[size - 1][x] is True
        for y in range(size):
            assert mask[y][0] is True
            assert mask[y][size - 1] is True

    def test_centered_patch_survives_as_foreground_despite_gradient(self):
        img = _gradient_with_centered_patch()
        mask = compute_background_mask(img)
        assert mask[50][50] is False  # center of the green patch

    def test_transparent_pixels_always_background(self):
        img = Image.new("RGBA", (20, 20), (10, 200, 10, 255))
        for x in range(20):
            for y in range(20):
                img.putpixel((x, y), (10, 200, 10, 0))
        mask = compute_background_mask(img)
        assert all(all(row) for row in mask)

    def test_enclosed_region_matching_background_color_is_a_hole(self):
        # A ring shape (foreground) with a background-colored counter in
        # the middle, same idea as the letter "O" -- the counter does not
        # touch the image border, but must still be classified background
        # since it matches the surrounding material's own color.
        img = Image.new("RGB", (60, 60), (250, 250, 250))
        px = img.load()
        for y in range(10, 50):
            for x in range(10, 50):
                px[x, y] = (30, 30, 30)
        for y in range(22, 38):
            for x in range(22, 38):
                px[x, y] = (250, 250, 250)
        mask = compute_background_mask(img)
        assert mask[30][30] is True  # the enclosed counter
        assert mask[15][15] is False  # the dark ring itself

    def test_does_not_tunnel_through_an_anti_aliased_edge_into_a_solid_blob(self):
        # Regression test: a real anti-aliased boundary between a grayish
        # background and a solidly-colored foreground blob blends over a
        # few pixels. Each individual blend step can be small enough to
        # pass `step_tolerance`, which let an earlier version of this
        # module's flood fill "tunnel" straight through the blend and
        # swallow the blob's solid interior as background too -- caught via
        # a real end-to-end pipeline test with a photographed-logo-style
        # image (see continue.md / session notes for the concrete
        # reproduction). This directly exercises that same failure shape at
        # the unit level: a gray background blending, over several pixels,
        # into a distinctly blue circular blob.
        size = 60
        img = Image.new("RGB", (size, size), (180, 180, 190))
        px = img.load()
        cx, cy, r = 30, 30, 18
        blob_rgb = (20, 60, 160)
        bg_rgb = (180, 180, 190)
        for y in range(size):
            for x in range(size):
                d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                if d <= r - 3:
                    px[x, y] = blob_rgb
                elif d <= r + 3:
                    # Blend band: smoothly interpolate background -> blob
                    # color over a 6px-wide ring, small per-pixel steps.
                    t = (d - (r - 3)) / 6.0
                    px[x, y] = tuple(
                        int(blob_rgb[i] * (1 - t) + bg_rgb[i] * t) for i in range(3)
                    )
        mask = compute_background_mask(img)
        assert mask[cy][cx] is False  # blob center must stay foreground
        assert mask[0][0] is True  # corner background still correctly removed

    def test_distinctly_colored_enclosed_shape_is_not_swallowed_as_a_hole(self):
        # An enclosed shape whose color is nothing like the background must
        # NOT be folded into the hole pass -- only genuine background-
        # colored enclosed regions should be treated as holes.
        img = Image.new("RGB", (60, 60), (250, 250, 250))
        px = img.load()
        for y in range(10, 50):
            for x in range(10, 50):
                px[x, y] = (30, 30, 30)
        for y in range(22, 38):
            for x in range(22, 38):
                px[x, y] = (210, 40, 40)  # distinct red, not background-ish
        mask = compute_background_mask(img)
        assert mask[30][30] is False
