"""
Design-region detection within a real garment photo (brief: "when creating
with [a] real world photo, example clothes on someone, it's supposed to
detect it and extract only the design... but now it's just removing the
background normally and trying to convert to vector which will give very
bad result").

Why this is a separate step from `background_removal.py`
----------------------------------------------------------
`background_removal.compute_background_mask` already does real work here,
but it solves a narrower problem than its docstring's own framing
("extract only the design, never the fabric/material behind it") fully
covers: it is a *border-connected* flood fill, so it only ever removes
surrounding material that touches the image frame -- a studio backdrop, a
flat-lay's table, the wall behind a hung garment. It explicitly assumes
"the design is fully inset from the frame edges, with surrounding material
visible all the way around it" (see that module's docstring, pass 1).

A photo of someone *wearing* the garment breaks that assumption outright:
the shirt's own plain fabric extends well past where the design/logo
actually is, often filling most or all of the frame, with skin/face/hair/
arms also in shot. None of that is "surrounding material" by the border-
flood-fill definition -- it's all one single border-connected blob of
foreground, design and blank fabric and skin together. Running the existing
pipeline directly on a photo like that was always going to try to
vectorize the *entire person+shirt silhouette* as "the design", which
is exactly the bad-vector-result failure this module exists to prevent.

What this module actually does
-------------------------------
Given a photo (already optionally background-removed via the module
above, though this also works fine on the original), find the sub-region
that actually looks like a printed/stitched design -- i.e. a patch with
noticeably more local color/edge detail than the large flat areas
(skin, plain fabric) around it -- and return its bounding box.

Deliberately a *suggestion*, not a silent auto-crop: per this project's
standing rule that nothing gets applied to a user's design without an
explicit on-screen confirmation (see background_removal.py's own preview-
endpoint precedent), the caller is expected to show this box to the user
as an adjustable crop overlay they can confirm or resize before anything
is actually cropped and digitized -- never crop automatically behind the
scenes. `confidence` exists specifically so a caller can flag "this guess
might not be right, please check it" rather than presenting a possibly-
wrong box with unearned certainty.

Algorithm (numpy + PIL only, no OpenCV/scipy, matching this project's
existing dependency stance -- see classify_artwork.py's own docstring for
the same reasoning)
-----------------------------------------------------------------------
1. Downsample to a small working grid (independent of the caller's own
   working resolution, like classify_artwork.SAMPLE_MAX_DIM) -- this is
   about finding a coarse region, not tracing fine detail.
2. Reuse `compute_background_mask` to get *a* foreground mask -- this
   still correctly strips any actual border-connected backdrop before
   anything else runs, it just isn't assumed to also exclude the plain
   garment fabric / skin, which is this module's actual job.
3. Compute a per-pixel local-detail (color variability) map via a fast
   box-filtered local standard deviation (summed-area-table trick --
   O(1) per pixel after one cumulative-sum pass, no per-pixel convolution
   loop and no scipy.ndimage dependency).
4. Downsample that detail map onto a coarse cell grid, scored only over
   each cell's *foreground* pixels (a cell that's mostly background
   doesn't get to look "busy" just from a sliver of real detail at its
   edge).
5. Threshold cells into "busy" (candidate design) vs "flat" (plain
   fabric/skin) using a percentile-based cutover computed only over
   foreground cells -- adapts to each photo's own contrast/lighting
   instead of one fixed absolute number.
6. Flood-fill the coarse grid to find connected "busy" clusters, and pick
   the one with the highest *total* detail score (not just the largest
   area) -- a small, genuinely dense logo should win over a diffuse,
   low-level noise pattern spread across a large area (e.g. visible woven
   fabric texture).
7. Return that cluster's bounding box, padded out a little (so the
   returned crop isn't drawn razor-tight against the actual design pixels
   -- leaves room for anti-aliased edges and for background_removal's own
   border-connectivity assumption to have real surrounding material inset
   from the crop's own edges, which is exactly what it needs to work well
   once this crop is applied).

Known limitations (documented, not silently hidden)
-----------------------------------------------------
- A heavily patterned/textured garment (stripes, plaid, a loud print
  covering the whole shirt) reads as "busy" everywhere, so there's no real
  flat-vs-detailed contrast to find -- `confidence` comes out low in this
  case (the busy cluster covers most of the foreground), signaling the
  caller to treat the suggested box as a rough starting point only.
- A very low-contrast or poorly-lit design (same color family as the
  garment) may under-detect -- again reflected as low confidence rather
  than a wrong-but-confident answer.
- This is coarse region detection, not segmentation -- it's meant to feed
  an adjustable user-confirmed crop step, not to be the final word.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from digitizer.segmentation.background_removal import compute_background_mask

# Working resolution for this detection pass -- deliberately small and
# independent of the pipeline's own DEFAULT_MAX_DIM/PREVIEW_MAX_DIM, same
# reasoning as classify_artwork.SAMPLE_MAX_DIM: this is about finding a
# coarse region, not resolving fine detail, so a coarse downsample is both
# sufficient and keeps this fast regardless of the caller's original
# upload size.
_WORK_MAX_DIM = 240

# Local-detail window size (pixels, at the _WORK_MAX_DIM working
# resolution) used for the box-filtered local standard deviation --small
# enough to pick out real texture/edge detail, large enough to not just
# measure single-pixel JPEG noise.
_DETAIL_WINDOW = 7

# Coarse scoring-grid resolution -- the design-region search happens over
# cells this size, not individual pixels, since the final result is a
# single bounding box, not a precise mask.
_GRID_SIZE = 24

# A cell needs at least this fraction of its pixels classified foreground
# to be considered at all (otherwise a cell that's 95% background with one
# stray bright foreground sliver could get an outsized score).
_MIN_FOREGROUND_FRACTION = 0.25

# Histogram bin count for the Otsu busy/flat threshold -- see
# `_otsu_threshold`'s docstring for why this replaces a fixed percentile.
_OTSU_BINS = 64

# Padding added around the detected cluster's bounding box, as a fraction
# of that box's own width/height -- see module docstring step 7.
_PADDING_FRACTION = 0.18


def _resize_for_detection(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    scale = min(1.0, _WORK_MAX_DIM / max(w, h))
    if scale < 1.0:
        rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    return rgb


def _local_std_map(gray: np.ndarray, window: int) -> np.ndarray:
    """Box-filtered local standard deviation of `gray` (`[h][w]` float
    array) over a `window`x`window` neighborhood, via the summed-area-table
    trick: local mean and local mean-of-squares are each a single O(1)
    lookup per pixel once their cumulative sums are built, so this whole
    function is O(h*w) with no scipy/OpenCV convolution needed.
    """
    h, w = gray.shape
    half = window // 2

    def _box_sum(arr: np.ndarray) -> np.ndarray:
        # Pad so every pixel gets a full (clipped-at-edges) window, then
        # integrate via cumulative sums along both axes.
        padded = np.pad(arr, half, mode="edge")
        csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        # Prepend a zero row/column so the standard inclusion-exclusion
        # rectangle-sum formula works uniformly for every output pixel,
        # including the first row/column.
        csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
        y0 = np.arange(h)
        y1 = y0 + window
        x0 = np.arange(w)
        x1 = x0 + window
        total = csum[np.ix_(y1, x1)] - csum[np.ix_(y0, x1)] - csum[np.ix_(y1, x0)] + csum[np.ix_(y0, x0)]
        return total

    counts = window * window
    mean = _box_sum(gray) / counts
    mean_sq = _box_sum(gray * gray) / counts
    variance = np.clip(mean_sq - mean * mean, 0.0, None)
    return np.sqrt(variance)


def _erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Shrinks `mask` (a `[h][w]` bool array) inward by `radius` pixels,
    via PIL's `MinFilter` -- no scipy/OpenCV needed, same approach
    background_removal.py's `_morphological_close` already uses for its
    dilate-then-erode closing pass.

    This exists specifically to strip the garment's own *outer* silhouette
    edge out of the detail-scoring step below: the boundary between the
    garment (foreground) and whatever's behind it (background) is itself a
    sharp color transition, which a local-std/detail measure reads as
    "busy" exactly as readily as genuine printed-design detail would --
    without this erosion, a plain, design-free garment's own outline would
    get misdetected as "the design" every time. Eroding first means only
    pixels comfortably *inside* the foreground (at least `radius` pixels
    from its own edge) are ever scored at all.
    """
    if radius <= 0:
        return mask
    h, w = mask.shape
    im = Image.new("L", (w, h))
    im.putdata((mask.astype(np.uint8) * 255).reshape(-1).tolist())
    eroded = im.filter(ImageFilter.MinFilter(radius * 2 + 1))
    return np.array(eroded, dtype=np.uint8).reshape(h, w) > 127


def _otsu_threshold(values: np.ndarray, bins: int = _OTSU_BINS) -> float:
    """Otsu's method: the threshold that maximizes between-class variance
    of `values` split into two groups -- i.e. the best automatic cutover
    point for a genuinely bimodal distribution, which is exactly the shape
    "flat garment/skin cells" vs. "busy design cells" scores tend to have
    (two separate clusters, not a smooth gradient). A *fixed* percentile
    (e.g. "top 30%") was tried first and rejected: it always classifies
    exactly that fraction of cells as busy regardless of whether the
    underlying distribution actually has two separate modes or is mostly
    continuous, which on a real bimodal case (a small busy logo patch on
    an otherwise uniform garment) classified much of the plain fabric as
    "busy" too, just because *something* has to fill out that top
    percentile. Otsu instead finds wherever the real gap between the two
    groups actually is, however small or large that top group turns out to
    be -- a small logo patch produces a small, tightly-separated busy
    group; a photo with no real design produces two very close groups (or
    effectively one), naturally giving few-to-no cells above threshold
    instead of forcing a fixed fraction to qualify either way.
    """
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        return vmin
    hist, edges = np.histogram(values, bins=bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    sum_total = float(np.sum(hist * centers))

    sum_b = 0.0
    w_b = 0.0
    best_thresh = vmin
    best_between_var = -1.0
    for i in range(bins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += hist[i] * centers[i]
        mean_b = sum_b / w_b
        mean_f = (sum_total - sum_b) / w_f
        between_var = w_b * w_f * (mean_b - mean_f) ** 2
        if between_var > best_between_var:
            best_between_var = between_var
            best_thresh = float(centers[i])
    return best_thresh


def _largest_busy_cluster(
    busy: np.ndarray, score: np.ndarray
) -> tuple[int, int, int, int, float, int] | None:
    """4-connected flood fill over the small boolean `busy` grid, scoring
    each cluster by the sum of `score` over its cells (not just cell
    count) -- see module docstring step 6 for why total detail beats raw
    area here. Returns `(row0, col0, row1_exclusive, col1_exclusive,
    total_score, cell_count)` for the winning cluster, or `None` if no
    cell is busy at all.
    """
    rows, cols = busy.shape
    visited = np.zeros_like(busy, dtype=bool)
    best: tuple[int, int, int, int, float, int] | None = None

    for r0 in range(rows):
        for c0 in range(cols):
            if not busy[r0, c0] or visited[r0, c0]:
                continue
            stack = [(r0, c0)]
            visited[r0, c0] = True
            min_r, max_r, min_c, max_c = r0, r0, c0, c0
            total = 0.0
            count = 0
            while stack:
                r, c = stack.pop()
                total += float(score[r, c])
                count += 1
                min_r, max_r = min(min_r, r), max(max_r, r)
                min_c, max_c = min(min_c, c), max(max_c, c)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and busy[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if best is None or total > best[4]:
                best = (min_r, min_c, max_r + 1, max_c + 1, total, count)
    return best


def detect_design_region(img: Image.Image) -> dict:
    """Suggests a bounding box for the actual printed/stitched design
    within `img` (e.g. a logo on a photo of a worn garment), as opposed to
    the whole garment/person/background.

    Returns:
      {"x0", "y0", "x1", "y1"}: normalized (0..1) box, `x1 > x0` and
        `y1 > y0` guaranteed, clamped to the image bounds.
      "confidence": 0..1 -- see module docstring's Known Limitations for
        what drives this down; callers should visibly flag a low-
        confidence suggestion rather than present it as certain.
      "foreground_detected": False only when `compute_background_mask`
        found no foreground at all (e.g. a totally blank/solid image) --
        the returned box is then the full image and confidence is 0.0.

    Never raises for a readable image; the full-image box + confidence 0.0
    fallback covers every case where a meaningful sub-region can't be
    identified, since this is always meant to feed a user-adjustable crop
    step, never to block the wizard outright.
    """
    work = _resize_for_detection(img)
    w, h = work.size
    if w < 4 or h < 4:
        return {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0, "confidence": 0.0, "foreground_detected": False}

    bg_mask_list = compute_background_mask(work.convert("RGBA"))
    bg_mask = np.array(bg_mask_list, dtype=bool)
    foreground = ~bg_mask
    if not foreground.any():
        return {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0, "confidence": 0.0, "foreground_detected": False}

    arr = np.asarray(work, dtype=np.float32)
    gray = arr.mean(axis=2)
    detail = _local_std_map(gray, _DETAIL_WINDOW)

    # Scoring uses the *eroded* foreground (see `_erode_mask`'s docstring)
    # so the garment's own outer silhouette edge -- itself a sharp color
    # transition, exactly what the detail map is designed to catch -- is
    # never mistaken for genuine design detail. Cell *eligibility* below
    # still uses the full (non-eroded) foreground, so a cell doesn't need
    # to contain much interior-only area to be considered, just real
    # garment content overall.
    #
    # The erosion radius is a full `_DETAIL_WINDOW`, not just its half-
    # width: a resized photo's fg/bg boundary is anti-aliased/blended over
    # a few pixels (not a single razor-sharp step), so a pixel can still
    # have a background-blended neighbor just beyond where the *boolean*
    # mask itself first flips to background. Eroding by only the window's
    # half-width left the detail map's own window still reaching into that
    # blended zone at its far corner -- confirmed as the actual cause of a
    # bogus "busy" hotspot right at a design-free rectangle's own corner
    # during this module's own testing -- so the margin needs enough extra
    # room to clear that blend band too, not just the window's own radius.
    interior_foreground = _erode_mask(foreground, _DETAIL_WINDOW)

    # Bin pixels onto the coarse scoring grid, aggregating per-cell mean
    # detail (over interior-foreground pixels only) and foreground fraction.
    grid_rows = min(_GRID_SIZE, h)
    grid_cols = min(_GRID_SIZE, w)
    row_edges = np.linspace(0, h, grid_rows + 1).astype(int)
    col_edges = np.linspace(0, w, grid_cols + 1).astype(int)

    cell_score = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    cell_fg_fraction = np.zeros((grid_rows, grid_cols), dtype=np.float64)
    for r in range(grid_rows):
        for c in range(grid_cols):
            r0, r1 = row_edges[r], max(row_edges[r] + 1, row_edges[r + 1])
            c0, c1 = col_edges[c], max(col_edges[c] + 1, col_edges[c + 1])
            fg_block = foreground[r0:r1, c0:c1]
            fg_count = int(fg_block.sum())
            total_count = fg_block.size
            cell_fg_fraction[r, c] = fg_count / total_count if total_count else 0.0
            interior_block = interior_foreground[r0:r1, c0:c1]
            if interior_block.any():
                cell_score[r, c] = float(detail[r0:r1, c0:c1][interior_block].mean())

    eligible = cell_fg_fraction >= _MIN_FOREGROUND_FRACTION
    if not eligible.any():
        # Foreground exists but is too sparse/speckled to form any
        # confidently-eligible cell -- fall back to its plain bounding box.
        fg_rows = np.any(foreground, axis=1)
        fg_cols = np.any(foreground, axis=0)
        y0, y1 = np.where(fg_rows)[0][[0, -1]]
        x0, x1 = np.where(fg_cols)[0][[0, -1]]
        return {
            "x0": float(x0) / w,
            "y0": float(y0) / h,
            "x1": float(x1 + 1) / w,
            "y1": float(y1 + 1) / h,
            "confidence": 0.2,
            "foreground_detected": True,
        }

    eligible_scores = cell_score[eligible]
    score_range = float(eligible_scores.max() - eligible_scores.min())

    # Otsu (below) always returns *some* split point, even across an
    # essentially uniform distribution where that split is meaningless --
    # a genuinely design-free garment (flat color, maybe a soft lighting
    # gradient) still has *some* tiny amount of per-cell score variation
    # from resize/compression noise, and Otsu will confidently carve that
    # noise into a "busy" vs "flat" half anyway. Below this absolute floor
    # (grayscale 0..255 local-std units), there's no real detail signal to
    # separate at all, so skip straight to the no-busy-region fallback
    # rather than let Otsu manufacture a split out of noise.
    _MIN_MEANINGFUL_SCORE_RANGE = 3.0
    if score_range < _MIN_MEANINGFUL_SCORE_RANGE:
        busy = np.zeros_like(eligible)
    else:
        cutoff = _otsu_threshold(eligible_scores)
        # Extra small floor above the eligible cells' own minimum, for the
        # same reason as above but scaled to this image's own score range.
        cutoff = max(cutoff, float(eligible_scores.min()) + 0.05 * score_range)
        busy = eligible & (cell_score > cutoff)

    if not busy.any():
        # Nothing stood out as meaningfully busier than the rest -- the
        # whole foreground is roughly uniform (could be a flat logo
        # already filling the frame, or a garment with no visible design
        # at all). Fall back to the full foreground bounding box with low
        # confidence rather than guessing at a sub-region.
        fg_rows = np.any(foreground, axis=1)
        fg_cols = np.any(foreground, axis=0)
        y0, y1 = np.where(fg_rows)[0][[0, -1]]
        x0, x1 = np.where(fg_cols)[0][[0, -1]]
        return {
            "x0": float(x0) / w,
            "y0": float(y0) / h,
            "x1": float(x1 + 1) / w,
            "y1": float(y1 + 1) / h,
            "confidence": 0.25,
            "foreground_detected": True,
        }

    cluster = _largest_busy_cluster(busy, cell_score)
    assert cluster is not None  # `busy.any()` was just checked above
    row0, col0, row1, col1, _total_score, busy_cell_count = cluster

    px0 = col_edges[col0]
    py0 = row_edges[row0]
    px1 = col_edges[col1] if col1 < len(col_edges) else w
    py1 = row_edges[row1] if row1 < len(row_edges) else h

    # Pad outward (see module docstring step 7), clamped to the image.
    box_w = px1 - px0
    box_h = py1 - py0
    pad_x = max(1.0, box_w * _PADDING_FRACTION)
    pad_y = max(1.0, box_h * _PADDING_FRACTION)
    px0 = max(0.0, px0 - pad_x)
    py0 = max(0.0, py0 - pad_y)
    px1 = min(float(w), px1 + pad_x)
    py1 = min(float(h), py1 + pad_y)

    # Confidence: how much the winning cluster actually stands out from
    # the rest of the foreground, by area -- a cluster covering nearly all
    # eligible foreground cells means there was no real flat-vs-busy
    # contrast to find (see Known Limitations), so confidence should be
    # low even though *a* box is still returned.
    eligible_count = int(eligible.sum())
    coverage = busy_cell_count / eligible_count if eligible_count else 1.0
    confidence = float(np.clip(1.0 - coverage, 0.15, 0.95))

    return {
        "x0": px0 / w,
        "y0": py0 / h,
        "x1": px1 / w,
        "y1": py1 / h,
        "confidence": confidence,
        "foreground_detected": True,
    }
