"""
Flat-art vs. photo classification (closes the gap flagged in
docs/embroidery-engine-research.md's vector-pipeline research: "a
photo-vs-flat-art classification step... determines whether to trace
directly or run a photo-specific segmentation pipeline").

Why this matters: a clean logo/clipart export (flat color fills, crisp
edges, no lighting) and a photograph of a design already stitched onto a
real garment (gradients, fabric sheen, photographic noise, directional
lighting) need fundamentally different treatment before either can be
traced into accurate vector regions. Feeding a photo through the same
quantize-and-flood-fill path a flat PNG wants reliably mis-reads
photographic shading as extra "colors" and can confuse a design's own dark
details for background (two materials that are genuinely different but
happen to catch similar light), which is exactly the "it didn't understand
the image" failure this module exists to catch *before* that happens --
see `artworkify.py` for the photo-specific preprocessing this feeds into.

Approach: three cheap, resolution-independent statistics computed on a
small downsampled copy (numpy, already an effective dependency via
shapely>=2 -- no new heavy dependency):

1. `unique_color_ratio` -- how many distinct (coarsely-bucketed) colors
   appear relative to the pixel count. Flat art is made of a handful of
   large, uniform-color regions, so this stays low even at a few hundred
   pixels across. A photo's lighting gradients, fabric texture, and JPEG
   noise mean nearly every pixel can be its own shade, so this is reliably
   much higher.
2. `mean_gradient` -- average local gradient magnitude (numpy finite-
   difference, no OpenCV/scipy convolution needed). Flat art's edges are
   sparse (a handful of sharp outlines on an otherwise zero-gradient
   field); a photo has gradient energy almost everywhere from lighting and
   texture.
3. `soft_edge_fraction` -- the fraction of pixels with *moderate* gradient
   magnitude (neither ~flat nor a hard edge). This is the single best
   discriminator for "gradient/lighting," which is what actually breaks
   the flood-fill background remover and the anti-aliasing color merge:
   flat art's gradient histogram is bimodal (mostly ~0, a thin spike of
   large values at outlines), while a photo's is smeared across the middle
   range from continuous shading.

These three combine into a simple majority vote rather than one fragile
fixed threshold on any single metric -- calibrated against this project's
own sample library (a genuine flat vector wordmark/crest vs. a genuine
photographed patch) and kept deliberately conservative: when in doubt,
callers should treat `'flat'` as the safe default (it's today's existing
behavior), so the artworkify preprocessing only ever engages when the
photo signal is actually strong.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

ArtworkKind = str  # 'flat' | 'photo'

# Classification sample is deliberately small and independent of the
# pipeline's own working resolution (digitizer/pipeline.py's DEFAULT_MAX_DIM)
# -- these statistics are about the image's *character* (flat vs.
# photographic), not its content detail, so a coarse downsample is both
# sufficient and keeps this classification step cheap regardless of how
# large the caller's original upload was.
SAMPLE_MAX_DIM = 200

# Color-bucket width (per channel) for the unique-color-ratio statistic --
# coarse enough to not count ordinary JPEG compression ringing around a
# flat-art edge as "more colors", fine enough to still separate genuinely
# different flat-art fill colors from each other.
_BUCKET_WIDTH = 10


def _resize_for_sampling(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, SAMPLE_MAX_DIM / max(w, h))
    if scale < 1.0:
        rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    return rgb


def classify_artwork(img: Image.Image) -> dict:
    """Classifies `img` as `'flat'` (clean vector-like artwork: a logo,
    clipart, line art, flat-color graphic -- the existing quantize+
    flood-fill pipeline already handles these well) or `'photo'` (a
    photographed or photorealistically-rendered scene with gradients,
    lighting, and texture -- needs the `artworkify` preprocessing pass
    first).

    Returns `{"kind": "flat" | "photo", "confidence": 0..1, "metrics": {...}}`.
    `metrics` is included for logging/debugging and the API response, not
    meant to be re-derived by callers.
    """
    sample = _resize_for_sampling(img)
    arr = np.asarray(sample, dtype=np.float32)
    h, w, _ = arr.shape
    if h < 2 or w < 2:
        return {"kind": "flat", "confidence": 1.0, "metrics": {}}

    # 1. Unique-color ratio on coarsely-bucketed colors.
    buckets = np.floor(arr / _BUCKET_WIDTH).astype(np.int32)
    codes = (buckets[..., 0] * 4096 + buckets[..., 1] * 64 + buckets[..., 2]).reshape(-1)
    unique_color_ratio = float(np.unique(codes).size) / float(h * w)

    # 2/3. Gradient magnitude stats via simple finite differences (no
    # convolution/OpenCV dependency needed for a plain first-difference).
    gray = arr.mean(axis=2)
    gx = np.diff(gray, axis=1)[: h - 1, :]
    gy = np.diff(gray, axis=0)[:, : w - 1]
    grad_mag = np.sqrt(gx * gx + gy * gy)
    mean_gradient = float(grad_mag.mean())
    soft_edge_fraction = float(np.mean((grad_mag > 2.0) & (grad_mag < 18.0)))

    metrics = {
        "unique_color_ratio": unique_color_ratio,
        "mean_gradient": mean_gradient,
        "soft_edge_fraction": soft_edge_fraction,
    }

    # Majority vote across three independent, individually-noisy signals,
    # each with its own calibrated threshold -- see module docstring for
    # why no single one is trusted alone.
    votes = 0
    if unique_color_ratio > 0.12:
        votes += 1
    if mean_gradient > 7.0:
        votes += 1
    if soft_edge_fraction > 0.14:
        votes += 1

    kind: ArtworkKind = "photo" if votes >= 2 else "flat"
    confidence = (votes / 3.0) if kind == "photo" else ((3 - votes) / 3.0)
    return {"kind": kind, "confidence": confidence, "metrics": metrics}
