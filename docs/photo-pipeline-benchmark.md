# Photo-aware digitizing pipeline: benchmark results

Produced by `scripts/benchmark_photo_pipeline.py` against stitch-pro's real
8-asset photographed demo library (the actual sample artworks/templates
shown to users), at the API's real default `max_dim=150`.

## Before this fix

Every one of these 8 real photographed assets (product photos of an
embroidered patch/logo on fabric -- gradients, fabric sheen, lighting,
macro-photo noise) hit the color-collapse bug: `digitizer.pipeline`'s
quantize step would transitively merge every detected foreground color
together via `color_quantize._merge_near_duplicate_colors`'s blend-line
rule once the background itself contributed a dozen-plus diverse reference
shades (a photographed background's own lighting gradient), instead of the
flat-art case's usual one or two. Several assets (`lion-patch.jpg`,
`monogram-m.jpg`, `floral-design.jpg` confirmed directly) collapsed all the
way to a single washed-out average color -- a single flat blob instead of
a recognizable, multi-color design.

## After this fix

| Asset | Threads | Objects | Artwork kind detected |
|---|---|---|---|
| lion-patch.jpg | 6 | 319 | photo |
| monogram-m.jpg | 6 | 300 | photo |
| company-logo.jpg | 6 | 112 | photo |
| floral-design.jpg | 6 | 199 | photo |
| cap-logo.jpg | 6 | 17 | photo |
| badge-crest.jpg | 6 | 113 | photo |
| wordmark-text.jpg | 6 | 443 | photo |
| seasonal-snowflake.jpg | 6 | 342 | photo |

All 8 assets now retain their full requested color budget (`max_colors=6`)
instead of collapsing. Per-image digitize time stayed well under 1 second
(the classify+artworkify preprocessing adds roughly 100-150ms of overhead
on these 1024x1024 source JPEGs at the pipeline's working resolution).

## What's fixed vs. what's still a known limitation

**Fixed**: the core reported bug -- "background removal works, but the
subsequent raster digitizing doesn't understand the artwork" manifesting
as near-total color collapse. This was root-caused to two compounding
issues, both now addressed:
1. Feeding genuinely photographic input (gradients, lighting, fabric
   texture, macro-photo noise) directly into a flood-fill + quantize
   pipeline designed for flat vector art. Fixed by `classify_artwork.py`
   (auto-detects which kind of input this is) + `artworkify.py` (denoises,
   contrast-stretches, and posterizes a detected photo into flat-art-like
   input before anything else sees it).
2. A latent bug in `color_quantize._merge_near_duplicate_colors`'s
   anti-aliasing merge heuristic (rule 2, the background blend-line test)
   that combinatorially false-merges unrelated foreground colors once the
   background itself contributes many diverse reference shades. Fixed by
   the new `background_blend_tolerance` parameter, set to `0.0` (fully
   disabled) specifically for already-artworkified input, where it's
   redundant with artworkify's own posterize step anyway.

**Known remaining limitation**: on the most texture/detail-heavy source
photos (e.g. `monogram-m.jpg`'s fine ornate scrollwork, shot with a
shallow-depth-of-field macro lens), the per-pixel segmentation still
produces a noticeably higher stitch-object count than a hand-digitized
version would -- individual thread highlights/shadows and lens-blur
gradients the camera resolved end up as many small disconnected regions
rather than the smooth continuous curves a human digitizer would trace.
`color_quantize.despeckle_label_grid` (a connected-component size filter
that folds small same-color islands into their surrounding real region
before vectorizing) gives a modest, accuracy-safe improvement (~6% fewer
objects averaged across all 8 assets, with zero color-count cost -- see
its own docstring for the benchmarked trade-off against more aggressive
thresholds that did cost real colors on some assets). Fully closing this
gap would need genuine superpixel/region-based vector tracing (e.g.
SLIC-style segmentation or a proper boundary-tracing + curve-fitting
library), which is a substantially larger scope than this bug-fix pass --
flagged here as a candidate follow-up, not attempted in this pass.

## Follow-up: automatic per-region stitch type (SATIN vs FILL) was silently not firing

A separate, unrelated-looking bug was found while verifying this benchmark's
output is used correctly downstream: `digitizer/stitches/satin/thin_region.py`
already had an automatic per-region classifier (thin/elongated regions like
glyph strokes and border bars get a real SATIN column instead of a sparse
tatami FILL), wired into `pipeline.py` since before this benchmark pass. On
real photo-pipeline output, though, it was firing on **0 of 119**
geometrically-eligible regions in `wordmark-text.jpg` -- every single one
bailed back out to FILL. The cause: its scan-line validation treated a scan
line landing exactly on a shape's own tapering tip (zero clean crossings
there) the same as an ambiguous or gapped shape, rejecting the whole region
instead of just skipping that one end rung. Real despeckled/fragmented
stroke pieces from a photo almost always taper to a point at both ends, so
this hit nearly every candidate region.

Fixed by allowing zero-crossing scan lines within the first/last 15% of a
region's long axis (its tapering tips) to be skipped rather than bailing,
while a zero-crossing or ambiguous (>1 segment) scan line anywhere else
still safely bails to FILL as before (see `_EDGE_TAPER_FRACTION` in
`thin_region.py`). Re-running the 8-asset benchmark after the fix: total
object count is unchanged (1845, zero regions gained or lost -- this is a
pure reclassification, not a different segmentation), but every single
asset now has a real SATIN/FILL mix instead of being 100% (or 99.7%) FILL,
e.g. `wordmark-text.jpg` went from `{FILL: 465}` to `{FILL: 353, SATIN:
112}`, `monogram-m.jpg` from `{FILL: 310, SATIN: 1}` to `{FILL: 230, SATIN:
81}`. Visually re-rendered several assets end to end to confirm the new
SATIN columns look like clean directional stitching, not broken/self-
crossing output. New regression tests lock in both the tapering-tip fix
(`test_tapering_lozenge_produces_valid_rungs_instead_of_bailing`) and that
genuinely ambiguous bent shapes still safely bail
(`test_bent_bracket_shape_safely_bails_to_none`).

## Re-running this benchmark

```
cd stitchpro-backend
source venv/bin/activate
python3 scripts/benchmark_photo_pipeline.py
```

Requires the sibling `stitch-pro` repo checked out alongside this one (the
script defaults to `../stitch-pro/assets/embroidery`; pass `--assets-dir`
to point elsewhere). Exits non-zero if any asset's thread count regresses
below its recorded baseline.
