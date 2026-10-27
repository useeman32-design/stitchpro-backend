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
# photographic gradients into separable steps. Benchmarked end-to-end
# (the real `image_to_design` pipeline, counting actual surviving
# `design.threads` -- not a palette-size proxy, which was tried first and
# gave misleadingly optimistic numbers, see session notes) across this
# project's full 8-asset photographed demo library: 4 is the largest
# value confirmed to never cost a single real color on any of the 8
# assets. 3 bits (8 levels) looked like a big win on a raw-palette-size
# proxy metric, but actually *lost* a real color outright on two separate
# assets once measured against real surviving threads -- a 3-bit test
# that looked like a pure improvement was actually trading accuracy for a
# smaller stitch-object count, which this project's standing
# accuracy-first priority rules out. 2 bits (4 levels) is more obviously
# too aggressive (drops monogram-m.jpg to 4 colors and cap-logo.jpg to a
# single color even on the flawed proxy metric). 4 is the correct,
# measured, accuracy-preserving choice; the remaining texture-driven
# object-count fragmentation on the busiest photos is handled separately,
# after quantization, by `color_quantize.despeckle_label_grid`.
POSTERIZE_BITS = 4

# Median filter kernel size (odd, pixels) for the pre-posterize denoise
# pass, applied at the image's *own* resolution -- deliberately small so
# it only removes sub-region noise, never blurs across an actual design
# boundary. A larger kernel (5-9) was tried to cut down on small
# disconnected texture-level fragments surviving posterization (e.g. a
# close-up lion-patch photo's individual fur highlights/shadows each
# becoming their own tiny region), and it does help that one case --
# but benchmarked across this project's full 8-asset demo library (see
# session notes), a kernel above 3 also blurs away several *genuinely*
# distinct, subtle foreground colors on other designs (monogram-m.jpg and
# cap-logo.jpg both lost real colors, down to as few as 1, at kernel=7) --
# i.e. it trades away the exact color-fidelity fix this module exists for.
# 3 is the largest kernel that was confirmed to never cost real colors on
# any of the 8 assets. The residual micro-texture fragmentation this
# leaves behind is a separate, lower-priority stitch-efficiency polish
# item (more small FILL objects/trims than ideal), not a correctness bug.
_DENOISE_KERNEL = 3

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
