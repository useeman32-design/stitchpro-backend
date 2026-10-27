"""
Photo -> flat-artwork preprocessing (the other half of the flat-vs-photo
split started in digitizer/classifier/classify_artwork.py).

Why this exists: `color_quantize.py`'s anti-aliasing-aware merge logic
(`_merge_near_duplicate_colors`) assumes any two palette colors that lie
near a straight line between a design color and a background reference are
the *same* design color blending into its backdrop -- true, and necessary,
for a flat logo's genuinely anti-aliased edges. But a real photograph of a
design (lighting falloff across a garment, fabric sheen, a gradient
shadow) produces *continuous* smooth color transitions almost everywhere,
which satisfies that same "lies on a blend line" test between pairs of
colors that are actually unrelated, genuinely distinct design regions --
confirmed empirically on this project's own photographed sample assets,
where a design with four or five real colors (gold thread, navy thread,
cream backing, etc.) was collapsing to a single washed-out average color
regardless of working resolution, because literally every pair of its
"colors" looked like a plausible photographic blend of the other.

The fix is not a resolution change (that was tried and measured -- see
session notes -- raising `max_dim` alone does not help here, since the
underlying color *relationships* are the problem, not pixel count) but to
flatten the image's own color statistics *before* quantization even sees
it, so the existing, already-correct flat-art pipeline receives something
that looks like flat art:

1. **Edge-preserving denoise** (`ImageFilter.MedianFilter`) -- removes
   fine photographic/fabric-weave/JPEG-block noise without smearing real
   region boundaries the way a Gaussian blur would (a median filter
   replaces a pixel with its neighborhood's median, which is robust to the
   kind of small, high-frequency noise a woven fabric's lighting speckle
   produces, while a true, larger-scale edge survives because the
   neighborhood on each side of it is internally consistent).
2. **Posterize** (`ImageOps.posterize`) -- the actual fix for the blend-
   line false-positive problem above: collapses each channel's continuous
   0-255 range to a handful of discrete steps, turning smooth photographic
   gradients into flat plateaus with hard cliffs between them. Two
   genuinely different design colors under continuous lighting no longer
   have a smooth path of intermediate shades connecting them for the
   blend-line heuristic to walk -- they're now simply two different flat
   colors, exactly what the downstream pipeline already handles well. A
   real anti-aliased edge between two posterized bands is still only 1-2
   intermediate shades wide (a true blend, correctly still merged).
3. **Mild local-contrast stretch** -- real product photography is rarely
   shot at full 0-255 contrast (studio lighting softens blacks, a glossy
   material's highlight never quite clips to pure white), which otherwise
   leaves posterize with less effective range to work with than it looks
   like it has. A conservative per-channel percentile stretch (2nd/98th
   percentile, not min/max, so a handful of outlier noise pixels can't
   compress everything else into one posterize band) widens the usable
   range first.

Deliberately conservative and Pillow/numpy-only (no OpenCV/scikit-image --
see color_quantize.py's module docstring for why that dependency stance
exists). This never runs for images `classify_artwork` calls `'flat'` --
posterizing a PNG that's already flat color blocks would only destroy
genuine fine anti-aliasing with no benefit.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# Posterize bit depth: 4 bits/channel = 16 levels/channel, 4096 possible
# colors -- enough headroom for a design to have several genuinely
# distinct flat colors while still being aggressive enough to break
# photographic gradients into separable steps. Tested against this
# project's own photographed sample assets (see session notes): 5 bits
# (32 levels) was not aggressive enough to stop the blend-line merge from
# still collapsing distinct colors; 3 bits (8 levels) started visibly
# banding genuinely smooth, intentional color transitions (e.g. a
# realistic fur/gradient fill some logos deliberately use). 4 is the
# measured sweet spot.
POSTERIZE_BITS = 4

# Median filter kernel size (odd, pixels) for the pre-posterize denoise
# pass, applied at the image's *own* resolution. Raised from an initial 3
# to 7 after measuring on this project's own photographed sample assets
# (see session notes): 3 was large enough to kill single-pixel JPEG noise
# but left individual embroidery *thread strands* -- each with their own
# highlight/shadow a close-up product photo resolves clearly -- as their
# own tiny, disconnected posterized islands, very nearly doubling the
# final stitch object count for no accuracy benefit (805 vs. 424 objects
# on the same test image, with no loss in overall shape/color
# recognizability). 7 is still well short of blurring across an actual
# region boundary (a design's real features are many pixels across at
# this pipeline's working resolution), it just stops resolving
# individual-thread-level texture that was never going to be reproduced
# as individual stitches anyway.
_DENOISE_KERNEL = 7

_CONTRAST_LOW_PERCENTILE = 2.0
_CONTRAST_HIGH_PERCENTILE = 98.0


def _stretch_contrast(rgb: Image.Image) -> Image.Image:
    arr = np.asarray(rgb, dtype=np.float32)
    out = np.empty_like(arr)
    for c in range(3):
        channel = arr[..., c]
        lo, hi = np.percentile(channel, [_CONTRAST_LOW_PERCENTILE, _CONTRAST_HIGH_PERCENTILE])
        if hi - lo < 1e-3:
            out[..., c] = channel
            continue
        stretched = (channel - lo) * (255.0 / (hi - lo))
        out[..., c] = np.clip(stretched, 0.0, 255.0)
    return Image.fromarray(out.astype(np.uint8), "RGB")


def artworkify(img: Image.Image) -> Image.Image:
    """Converts a photographic `img` into a flatter, posterized
    approximation suitable for the existing flat-art quantize/background-
    removal/polygon pipeline. Preserves the image's mode-relevant alpha
    channel (if any) untouched -- only RGB content is stylized, since alpha
    already cleanly marks transparency with no photographic-lighting
    ambiguity to resolve.

    Idempotent-ish on genuinely flat input (posterizing already-flat colors
    just snaps them to the nearest step, a no-op in practice for colors
    that were already flat), but should only be called for images
    `classify_artwork` flagged as `'photo'` -- see module docstring.
    """
    has_alpha = img.mode in ("RGBA", "LA") or ("transparency" in img.info)
    rgba = img.convert("RGBA") if has_alpha else None
    rgb = img.convert("RGB")

    denoised = rgb.filter(ImageFilter.MedianFilter(size=_DENOISE_KERNEL))
    stretched = _stretch_contrast(denoised)
    posterized = ImageOps.posterize(stretched, POSTERIZE_BITS)

    if rgba is not None:
        alpha = rgba.split()[3]
        result = posterized.convert("RGBA")
        result.putalpha(alpha)
        return result
    return posterized
