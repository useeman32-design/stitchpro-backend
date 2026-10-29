"""
Image -> EmbroideryDesign pipeline (Milestone 5: "real PNG -> segmentation
-> embroidery objects").

Unlike the earlier milestones (where the frontend already does the client-
side artwork analysis via stitch-pro's own stitchEngine.ts and this backend
only compiles an already-made stitch plan to a real file -- see
api/app.py's module docstring and formats/points_writer.py), this is the
first endpoint that does real server-side digitizing: given nothing but a
raw image, decide what the embroidery objects are.

Pipeline: downscale (`load_and_downscale` below) -> classify as flat
artwork or a photograph (digitizer.classifier.classify_artwork) -> for a
photograph, flatten it into something the rest of this pipeline already
knows how to handle well (digitizer.segmentation.artworkify: denoise,
contrast-stretch, posterize) -- this is the actual fix for "a real photo
of a design digitizes badly": the flat-art pipeline below was never the
problem, feeding it raw photographic gradients/lighting/texture was (see
`artworkify`'s and `color_quantize.build_label_grid`'s
`background_blend_tolerance` docstrings for exactly how and why that broke)
-> quantize + remove background/surrounding material via a border-connected
flood fill (digitizer.segmentation.color_quantize + background_removal --
robust to a real photographed garment's lighting/shading, not just a flat
studio backdrop) -> per-color connected regions with holes, curve-smoothed,
in mm space (digitizer.segmentation.polygons) -> one stitch object per
region, styled per `stitch_style`. Routing (digitizer.routing.router, used
automatically by formats/dst/writer.py:compile_design) then groups same-
color objects together and orders them for minimal travel, exactly like any
other design built this way.

`stitch_style` selects FILL (tatami rows, the default -- best for solid
color regions and most logos/graphics) or CROSS_STITCH (a grid of X's --
better for a deliberately pixelated/sampler-style design) for *every*
detected region in the design. When `stitch_style="fill"`, each region is
additionally checked for automatic SATIN classification (see
digitizer.stitches.satin.thin_region) before falling back to FILL: a thin,
elongated, hole-free region (a font glyph's straight stroke, a badge's
border bar, a ruled line) is a poor fit for tatami fill -- a handful of
short rows through a narrow shape reads as sparse scattered lines rather
than a clean stitched stroke, the single biggest gap found when comparing
our output against real professional reference files, where text/line-art
content is built almost entirely from satin columns. Classification is
deliberately conservative (see that module's docstring for exactly why) --
curved shapes and anything with a hole keep today's FILL(+border)
treatment, only genuinely straight-enough thin regions get the new
treatment -- so this is a purely additive quality improvement, never a
regression risk for shapes it doesn't apply to.
"""

from __future__ import annotations

import io

from PIL import Image
from shapely.geometry import Polygon

from digitizer.classifier.classify_artwork import classify_artwork
from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.segmentation.artworkify import artworkify
from digitizer.segmentation.background_removal import apply_mask_strokes, compute_background_mask
from digitizer.segmentation.color_quantize import build_label_grid, despeckle_label_grid
from digitizer.segmentation.polygons import label_grid_to_ring_groups
from digitizer.stitches.puff.eligibility import is_puff_eligible
from digitizer.stitches.satin.thin_region import (
    generate_satin_rungs_for_thin_region,
    is_satin_eligible,
    oriented_bbox_dims,
    thin_region_treatment,
)

# Rule 2's blend-tolerance inside color_quantize._merge_near_duplicate_colors,
# used specifically for a design this pipeline has run through `artworkify`
# -- see build_label_grid's `background_blend_tolerance` docstring for why
# the ordinary default (12.0, tuned for a single solid background color)
# reliably collapses a photographed design's genuinely distinct colors down
# to one once the background itself carries a dozen-plus gradient reference
# shades. 0.0 (fully disabling this specific merge rule, rather than just
# tightening it) was chosen only after benchmarking across this project's
# full 8-asset demo library (see session notes): a small-but-nonzero value
# like 2.0 looked sufficient on an initial handful of test images, but
# still silently collapsed monogram-m.jpg's subtler color distinctions down
# to a single color. 0.0 was the only value confirmed to preserve the full
# expected color count on all 8 assets. This is safe specifically *because*
# `artworkify`'s posterize step already does the anti-aliasing cleanup rule
# 2 exists for (collapsing a smooth AA gradient down to a handful of flat
# posterize levels before quantization ever runs), so rule 2's extra
# cross-color blend-line check is largely redundant for already-artworkified
# input, not just safe to loosen.
_PHOTO_BACKGROUND_BLEND_TOLERANCE = 0.0


def _classify_and_artworkify(
    img: Image.Image, auto_vectorize: bool, force_photo: bool
) -> tuple[Image.Image, str, float, dict[str, float]]:
    """Shared flat-vs-photo classification + conditional `artworkify` pass
    used by both `image_to_design` and `image_to_vector_regions` -- see
    their own docstrings (and this module's) for the full rationale.
    Returns `(possibly_artworkified_img, artwork_kind, artwork_confidence,
    label_grid_kwargs)`, where `label_grid_kwargs` is ready to splat
    straight into `build_label_grid(...)`.
    """
    if not auto_vectorize:
        return img, "flat (forced)", 1.0, {}

    if force_photo:
        artwork_kind = "photo (forced)"
        artwork_confidence = 1.0
    else:
        artwork_info = classify_artwork(img)
        artwork_kind = artwork_info["kind"]
        artwork_confidence = artwork_info["confidence"]

    label_grid_kwargs: dict[str, float] = {}
    if _is_photo_kind(artwork_kind):
        img = artworkify(img)
        label_grid_kwargs["background_blend_tolerance"] = _PHOTO_BACKGROUND_BLEND_TOLERANCE
    return img, artwork_kind, artwork_confidence, label_grid_kwargs


def _is_photo_kind(artwork_kind: str) -> bool:
    """True for either `'photo'` (classify_artwork's own vote) or `'photo
    (forced)'` (force_photo=True skipped the vote) -- every downstream
    photo-specific step (artworkify, the background-blend-tolerance
    override, despeckling) should trigger identically for both."""
    return artwork_kind.startswith("photo")


# Default resolution for the standalone background-removal *preview*
# (`remove_background_preview` below) -- deliberately much higher than
# DEFAULT_MAX_DIM (150), since this image is only ever shown to the user
# for a yes/no "does this look right" confirmation, never fed through the
# stitch generators, so there's no reason to pay the same aggressive
# downscale the full digitizing pipeline needs to stay cheap. Bounds are
# their own, separate range from MIN_MAX_DIM/MAX_MAX_DIM (which cap the
# *digitizing* pipeline's resolution, deliberately much lower) -- a caller
# can reasonably ask this preview-only endpoint for a crisper image than
# the stitch pipeline would ever accept.
PREVIEW_MAX_DIM = 900
PREVIEW_MIN_MAX_DIM = 16
PREVIEW_MAX_MAX_DIM = 2000

# Keep server-side processing cheap regardless of what the client uploads --
# see docs/embroidery-engine-research.md's cPanel CPU/timeout concerns. 150px
# matches the cap stitch-pro's own client-side stitchEngine.ts already uses
# for its rasterize step, so behavior stays consistent between the two
# digitizing paths.
DEFAULT_MAX_DIM = 150
MIN_MAX_DIM = 16
MAX_MAX_DIM = 400

STITCH_STYLES = ("fill", "cross")
_DEFAULT_DENSITY_MM_BY_STYLE = {
    "fill": 0.4,    # tatami row spacing
    "cross": 1.8,   # cross-stitch grid cell size -- a real cross-stitch
                    # texture needs cells large enough to actually read as
                    # individual X's rather than a solid-looking fill.
}
# A satin "edge walk" border traced around every region's outer boundary and
# holes (see digitizer/stitches/satin/border.py) hides the raw scanline/grid
# edge FILL and CROSS_STITCH otherwise leave behind -- the single biggest
# visible difference between an auto-digitized-looking fill and a
# professional one. FILL gets one on by default (same 1.2-1.4mm range real
# digitizers use for a logo's outline satin); CROSS_STITCH stays borderless
# by default since a crisp vector edge would fight its deliberately
# pixelated/sampler look.
_DEFAULT_BORDER_MM_BY_STYLE = {
    "fill": 1.2,
    "cross": 0.0,
}

# Underlay + pull compensation (digitizer/underlay/generator.py and the
# `pull_compensation_mm` plumbed through every stitches/*/generator.py) have
# existed and been tested since Milestone 6/the thin-region satin work, but
# were never actually turned on here -- every StitchSettings built by this
# pipeline left both at their do-nothing defaults (underlay=None,
# pull_compensation_mm=0.0), so none of that real, working code ever ran for
# an actual auto-digitized design. Every reference professional file this
# project studied uses both: underlay to stabilize the fabric before the top
# stitch lays down (without it, fill/satin pucker and shift -- a likely
# contributor to the "rough/inaccurate at high density" feedback), and pull
# compensation to counter the real, physical fact that thread tension pulls
# a stitched shape in narrower than its digitized geometry.
#
# SATIN (today, exclusively the auto-classified thin/elongated regions from
# thin_region.py -- narrow columns) gets "center_run": a single run stitch
# down the column's midline, the standard choice for narrow satin per every
# digitizing guide consulted (an edge-run underlay on a genuinely narrow
# column risks peeking out past the finished edge). FILL gets "edge_run":
# a running stitch just inside the outer boundary and every hole, which
# stabilizes the perimeter without materially changing the center fill
# rows. CROSS_STITCH gets no underlay -- its own grid pattern is already a
# self-stabilizing structure, and the underlay generator intentionally
# no-ops for it regardless (see underlay/generator.py's docstring).
_SATIN_UNDERLAY = "center_run"
_FILL_UNDERLAY = "edge_run"

# Pull compensation: how much every shape's boundary is expanded outward
# before stitching, so the real (tension-shrunk) result matches the
# intended size. Satin columns pull in the most (the full stitch crosses
# the narrow width every pass), fill pulls in less (most of its stitches
# run lengthwise, only the row ends feel the pull), and a cross-stitch grid
# barely moves at all (each X is short and self-anchoring) -- 0 is a
# deliberate, conservative choice there rather than an oversight.
_SATIN_PULL_COMPENSATION_MM = 0.2
_DEFAULT_PULL_COMPENSATION_MM_BY_STYLE = {
    "fill": 0.12,
    "cross": 0.0,
}

# New stitch type follow-up (Bean/Motif/3D Puff/Applique), all auto-detected
# like SATIN already was -- same scoping rule as SATIN: only ever considered
# when `stitch_style == "fill"`, never overriding a user's explicit
# CROSS_STITCH choice for every region.
#
# A single detected region's largest, simplest, most convex FILL shape
# becomes APPLIQUE only when it's genuinely dominant -- real appliqu\u00e9
# patches are almost always a design's main background/body silhouette, not
# an arbitrary detail shape, and wrongly auto-selecting a detail shape here
# would force a real machine-pause/fabric-handling workflow (see
# stitches/applique/generator.py) onto a design that never asked for it.
# Deliberately rare: most designs will have no APPLIQUE region at all.
_APPLIQUE_MIN_DOMINANT_AREA_FRACTION = 0.40
_APPLIQUE_MIN_CONVEXITY = 0.85


def _build_region_records(
    label_grid: list[list[int]],
    palette_hex: list[str],
    px_to_mm_x: float,
    px_to_mm_y: float,
    smooth_curves: bool,
) -> tuple[list[dict], float, list[str]]:
    """Traces every detected region (across all colors) in `label_grid` into
    mm-space polygon rings, plus each one's outer-boundary area -- the
    actual raster-to-vector step, shared by `image_to_design` (which
    immediately turns these into stitch objects) and `image_to_vector_regions`
    (which returns them as-is, for a client to review/edit *before* any
    stitch generation happens -- see that function's docstring).

    Returns `(region_records, total_foreground_area_mm2, threads)`:
      - `region_records[i]` is `{"color_index", "i", "color_hex", "rings",
        "area_mm2"}` -- `"rings"` is `[outer_ring, *hole_rings]`, each a list
        of `(x_mm, y_mm)` points.
      - `total_foreground_area_mm2` is the sum of every region's own area,
        used by the PUFF_3D/APPLIQUE heuristics to judge a region's size
        relative to the whole design, not in isolation.
      - `threads` is the subset of `palette_hex` that actually produced at
        least one traced region (a palette color can end up with zero
        pixels after quantization/despeckling).
    """
    region_records: list[dict] = []
    total_foreground_area_mm2 = 0.0
    threads: list[str] = []
    for color_index, color_hex in enumerate(palette_hex):
        ring_groups = label_grid_to_ring_groups(
            label_grid,
            color_index,
            px_to_mm_x,
            px_to_mm_y,
            smooth_iterations=3 if smooth_curves else 0,
        )
        if not ring_groups:
            continue
        threads.append(color_hex)
        for i, rings in enumerate(ring_groups):
            area_mm2 = 0.0
            try:
                poly = Polygon(rings[0])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                area_mm2 = poly.area
            except Exception:  # noqa: BLE001 - degenerate ring geometry, treat as zero area
                area_mm2 = 0.0
            total_foreground_area_mm2 += area_mm2
            region_records.append(
                {"color_index": color_index, "i": i, "color_hex": color_hex, "rings": rings, "area_mm2": area_mm2}
            )
    return region_records, total_foreground_area_mm2, threads


def _region_records_to_objects(
    region_records: list[dict],
    total_foreground_area_mm2: float,
    stitch_style: str,
    stitch_length_mm: float,
    effective_density_mm: float,
    angle_deg: float,
    effective_border_mm: float,
) -> list[EmbroideryObject]:
    """Turns traced vector regions (see `_build_region_records`, or a
    client's own edited region list via `image_to_vector_regions` ->
    `design_from_region_records`) into styled embroidery objects: per-shape
    SATIN/BEAN/MOTIF/PUFF_3D/APPLIQUE auto-classification exactly as
    `image_to_design` always did, now shared by both that function and
    `design_from_region_records` so a design built from a (possibly user-
    edited) region list gets identical stitch-type treatment to one built
    straight from an image.
    """
    object_type = ObjectType.FILL if stitch_style == "fill" else ObjectType.CROSS_STITCH

    objects: list[EmbroideryObject] = []
    applique_candidates: list[tuple[int, float]] = []  # (index into objects, area_mm2)
    seq = 0
    for rec in region_records:
        color_index, i, color_hex, rings = rec["color_index"], rec["i"], rec["color_hex"], rec["rings"]

        # Only FILL regions are considered for any of the auto-classified
        # treatments below -- CROSS_STITCH is a deliberate stylistic choice
        # the user made explicitly for every region, not something a
        # shape heuristic should override (see module docstring and
        # thin_region.py's).
        satin_rungs = None
        thin_treatment: str | None = None  # "SATIN" | "BEAN" | "MOTIF" | None
        if stitch_style == "fill" and is_satin_eligible(rings):
            satin_rungs = generate_satin_rungs_for_thin_region(rings[0])
            if satin_rungs is not None:
                dims = oriented_bbox_dims(rings[0])
                thin_treatment = thin_region_treatment(*dims) if dims is not None else "SATIN"

        if thin_treatment == "SATIN":
            objects.append(
                EmbroideryObject(
                    id=f"region-{color_index}-{i}",
                    type=ObjectType.SATIN,
                    geometry=satin_rungs,
                    color=color_hex,
                    stitch_settings=StitchSettings(
                        density_mm=effective_density_mm,
                        underlay=_SATIN_UNDERLAY,
                        pull_compensation_mm=_SATIN_PULL_COMPENSATION_MM,
                    ),
                    sequence=seq,
                    layer=0,
                )
            )
        elif thin_treatment in ("BEAN", "MOTIF"):
            # Reuse the already scan-validated satin rung pairs
            # (guaranteed well-behaved by generate_satin_rungs_for_thin_region
            # above) as a centerline path, rather than tracing the full
            # outer boundary -- a bean/motif stitch belongs down the middle
            # of a thin stroke, not around both of its long edges.
            rail_a = satin_rungs[0::2]
            rail_b = satin_rungs[1::2]
            midline = [((ax + bx) / 2, (ay + by) / 2) for (ax, ay), (bx, by) in zip(rail_a, rail_b)]
            if thin_treatment == "BEAN":
                objects.append(
                    EmbroideryObject(
                        id=f"region-{color_index}-{i}",
                        type=ObjectType.BEAN,
                        geometry=midline,
                        color=color_hex,
                        stitch_settings=StitchSettings(stitch_length_mm=stitch_length_mm),
                        sequence=seq,
                        layer=0,
                    )
                )
            else:
                objects.append(
                    EmbroideryObject(
                        id=f"region-{color_index}-{i}",
                        type=ObjectType.MOTIF,
                        geometry=midline,
                        color=color_hex,
                        stitch_settings=StitchSettings(stitch_length_mm=stitch_length_mm),
                        sequence=seq,
                        layer=0,
                    )
                )
        elif stitch_style == "fill" and is_puff_eligible(rings, total_foreground_area_mm2):
            objects.append(
                EmbroideryObject(
                    id=f"region-{color_index}-{i}",
                    type=ObjectType.PUFF_3D,
                    geometry=rings,
                    color=color_hex,
                    stitch_settings=StitchSettings(
                        density_mm=effective_density_mm,
                        angle_deg=angle_deg,
                    ),
                    sequence=seq,
                    layer=0,
                )
            )
        else:
            objects.append(
                EmbroideryObject(
                    id=f"region-{color_index}-{i}",
                    type=object_type,
                    geometry=rings,
                    color=color_hex,
                    stitch_settings=StitchSettings(
                        stitch_length_mm=stitch_length_mm,
                        density_mm=effective_density_mm,
                        angle_deg=angle_deg,
                        border_width_mm=effective_border_mm,
                        # Only ever applies to the FILL branch in practice --
                        # generate_underlay_stitches already no-ops for
                        # CROSS_STITCH regardless of this value, kept here
                        # rather than branched on object_type so this
                        # object-construction call doesn't need its own copy
                        # of that type check.
                        underlay=_FILL_UNDERLAY if object_type == ObjectType.FILL else None,
                        pull_compensation_mm=_DEFAULT_PULL_COMPENSATION_MM_BY_STYLE[stitch_style],
                    ),
                    sequence=seq,
                    layer=0,
                )
            )
            if object_type == ObjectType.FILL and len(rings) == 1:
                applique_candidates.append((len(objects) - 1, rec["area_mm2"]))
        seq += 1

    # At most one region in the whole design becomes APPLIQUE -- see
    # _APPLIQUE_MIN_DOMINANT_AREA_FRACTION's docstring above for why this
    # stays deliberately rare.
    if applique_candidates and total_foreground_area_mm2 > 0:
        best_idx, best_area = max(applique_candidates, key=lambda pair: pair[1])
        if best_area >= total_foreground_area_mm2 * _APPLIQUE_MIN_DOMINANT_AREA_FRACTION:
            candidate = objects[best_idx]
            poly = Polygon(candidate.geometry[0])
            if not poly.is_valid:
                poly = poly.buffer(0)
            hull_area = poly.convex_hull.area
            convexity = (poly.area / hull_area) if hull_area > 1e-9 else 0.0
            if convexity >= _APPLIQUE_MIN_CONVEXITY:
                candidate.type = ObjectType.APPLIQUE
                candidate.stitch_settings = StitchSettings(border_width_mm=candidate.stitch_settings.border_width_mm)

    if not objects:
        raise ValueError(
            "no foreground regions detected -- image may be blank, fully transparent, "
            "or a single solid color matching the detected background"
        )

    return objects


def load_and_downscale(image_bytes: bytes, max_dim: int = DEFAULT_MAX_DIM) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError("image has zero width or height")
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _apply_crop(img: Image.Image, crop: dict | None) -> Image.Image:
    """Crops `img` to a normalized (0..1) `{"x0", "y0", "x1", "y1"}` box
    (see digitizer.segmentation.design_region.detect_design_region) before
    anything else in the pipeline sees it -- the "extract only the design
    from a photo of someone wearing the garment" requirement: the rest of
    the pipeline (classification, background removal, quantization) is
    completely unchanged, it just now only ever sees the sub-image the
    user confirmed actually contains the design, instead of the whole
    person+garment+background frame.

    No-op when `crop` is `None` (every existing caller that doesn't pass
    it keeps today's exact behavior). Raises `ValueError` for a
    degenerate box (`x1 <= x0` or `y1 <= y0` after clamping to the image),
    same failure style as this module's other input validation.
    """
    if crop is None:
        return img
    w, h = img.size
    x0 = max(0.0, min(1.0, float(crop.get("x0", 0.0))))
    y0 = max(0.0, min(1.0, float(crop.get("y0", 0.0))))
    x1 = max(0.0, min(1.0, float(crop.get("x1", 1.0))))
    y1 = max(0.0, min(1.0, float(crop.get("y1", 1.0))))
    left = round(x0 * w)
    top = round(y0 * h)
    right = round(x1 * w)
    bottom = round(y1 * h)
    if right <= left or bottom <= top:
        raise ValueError("crop box is degenerate (x1/y1 must be greater than x0/y0 after clamping)")
    return img.crop((left, top, right, bottom))


def image_to_design(
    image_bytes: bytes,
    width_mm: float,
    height_mm: float,
    max_colors: int = 6,
    stitch_length_mm: float = 3.0,
    density_mm: float | None = None,
    max_dim: int = DEFAULT_MAX_DIM,
    angle_deg: float = 0.0,
    stitch_style: str = "fill",
    smooth_curves: bool = True,
    border_width_mm: float | None = None,
    remove_background: bool = True,
    auto_vectorize: bool = True,
    force_photo: bool = False,
    crop: dict | None = None,
) -> EmbroideryDesign:
    """Digitize a raw image into an EmbroideryDesign of stitch objects, one
    (or more, if a color appears in several disconnected regions) per
    detected color, sized to fit `width_mm` x `height_mm`.

    `auto_vectorize` (default `True`) runs `classify_artwork` on the
    (downscaled) image first and, only when it's classified `'photo'`, runs
    it through `artworkify` before anything else below ever sees it -- see
    the module docstring for why this is the actual fix for "a real photo
    digitizes badly", not a resolution or quantization-tuning problem. The
    detected `artwork_kind` (`'flat'` or `'photo'`, or `'flat (forced)'`
    when `auto_vectorize=False` skipped classification entirely) and its
    confidence are recorded on the returned design's `metadata` dict for
    callers that want to show the user what was detected. Pass `False`
    only to reproduce the exact pre-existing behavior (e.g. a regression
    test pinned to the old flat-art-only pipeline, or a caller that already
    knows its input is a clean vector export and wants to skip the
    classification work entirely).

    `force_photo` (default `False`) skips `classify_artwork`'s heuristic
    vote entirely and always treats the input as `'photo'` (recorded as
    `'photo (forced)'` in `metadata`, confidence `1.0`) -- for the "extract
    a design from a photo of an item that's already embroidered" flow
    (stitch-pro's dedicated upload entry point for that case), where the
    user has already told us this is a photograph of real stitching, not a
    flat logo export. This exists because the automatic classifier, while
    reliable on this project's benchmark library (see classify_artwork's
    own docstring), is still a heuristic -- a sharply, evenly lit close-up
    of a design with crisp thread edges can occasionally score closer to
    'flat' than a typical photo, which would skip the `artworkify`
    flattening pass this input actually needs. An explicit user signal is
    strictly more reliable than a guess and costs nothing when it agrees
    with the guess anyway. No-op (ignored) when `auto_vectorize=False` --
    that flag's whole point is to skip classification/`artworkify`
    entirely, which takes precedence over this one.

    `density_mm` left as `None` (the default) picks a sensible default for
    whichever `stitch_style` was chosen (see `_DEFAULT_DENSITY_MM_BY_STYLE`)
    rather than forcing one fixed number to do double duty as both a
    sensible fill-row spacing and a sensible cross-stitch cell size.

    `border_width_mm` likewise defaults (when left `None`) per
    `_DEFAULT_BORDER_MM_BY_STYLE` -- a satin edge-walk border around every
    region's outline and holes for FILL, none for CROSS_STITCH. Pass `0`
    explicitly to disable it for FILL too.

    `crop` (default `None`), when given, is a normalized (0..1)
    `{"x0", "y0", "x1", "y1"}` box applied before anything else below ever
    runs -- see digitizer.segmentation.design_region.detect_design_region
    and pipeline._apply_crop. This is the "extract only the design from a
    photo of someone wearing the garment" entry point: a caller first gets
    a suggested box from `detect_design_region`, shows it to the user as
    an adjustable crop they confirm or resize (never applied silently --
    same standing rule as `remove_background`'s own preview step), and
    only then passes the confirmed box here. `width_mm`/`height_mm` size
    the *cropped* result, exactly as if the uploaded photo had only ever
    contained that sub-region to begin with.

    `remove_background` (default `True`) runs the same border-connected
    flood-fill background/surrounding-material removal
    (digitizer.segmentation.background_removal) the standalone
    `remove_background_preview` function below shows the user ahead of
    time. Pass `False` only when the user explicitly chose "keep original"
    on that preview step -- every pixel is then treated as foreground
    (subject only to its own alpha channel), so the final digitized design
    matches exactly what they were shown and agreed to.

    Raises ValueError for an unreadable image, degenerate dimensions, an
    unrecognized `stitch_style`, or an image with no detectable foreground
    (e.g. a solid blank image, or one that is entirely surrounding
    material/background once that's removed).
    """
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("width_mm and height_mm must both be positive")
    if not (1 <= max_colors <= 16):
        raise ValueError("max_colors must be between 1 and 16")
    if not (MIN_MAX_DIM <= max_dim <= MAX_MAX_DIM):
        raise ValueError(f"max_dim must be between {MIN_MAX_DIM} and {MAX_MAX_DIM}")
    if stitch_style not in STITCH_STYLES:
        raise ValueError(f"stitch_style must be one of {STITCH_STYLES}, got {stitch_style!r}")

    effective_density_mm = density_mm if density_mm is not None else _DEFAULT_DENSITY_MM_BY_STYLE[stitch_style]
    if effective_density_mm <= 0:
        raise ValueError("density_mm must be > 0")

    effective_border_mm = border_width_mm if border_width_mm is not None else _DEFAULT_BORDER_MM_BY_STYLE[stitch_style]
    if effective_border_mm < 0:
        raise ValueError("border_width_mm must be >= 0")

    try:
        img = load_and_downscale(image_bytes, max_dim)
    except Exception as e:  # noqa: BLE001 - Pillow raises several distinct error types
        raise ValueError(f"could not read image: {e}") from e
    img = _apply_crop(img, crop)

    w, h = img.size
    px_to_mm_x = width_mm / w
    px_to_mm_y = height_mm / h

    img, artwork_kind, artwork_confidence, label_grid_kwargs = _classify_and_artworkify(
        img, auto_vectorize, force_photo
    )

    palette_hex, label_grid = build_label_grid(
        img,
        max_colors,
        remove_background=remove_background,
        **label_grid_kwargs,
    )
    if _is_photo_kind(artwork_kind):
        # Collapses the dozens of tiny texture-noise islands a close-up
        # photo's per-thread shading leaves behind (even after artworkify's
        # denoise+posterize) into their surrounding real region, instead of
        # each becoming its own near-useless stitch object -- see
        # despeckle_label_grid's own docstring for why this is a different
        # (and necessary, in addition to) fix from label_grid_to_ring_groups'
        # own min_area_px/min_bbox_mm filters below.
        label_grid = despeckle_label_grid(label_grid)

    region_records, total_foreground_area_mm2, threads = _build_region_records(
        label_grid, palette_hex, px_to_mm_x, px_to_mm_y, smooth_curves
    )

    objects = _region_records_to_objects(
        region_records,
        total_foreground_area_mm2,
        stitch_style,
        stitch_length_mm,
        effective_density_mm,
        angle_deg,
        effective_border_mm,
    )

    return EmbroideryDesign(
        width_mm=width_mm,
        height_mm=height_mm,
        objects=objects,
        threads=threads,
        metadata={"artwork_kind": artwork_kind, "artwork_confidence": artwork_confidence},
    )


def image_to_vector_regions(
    image_bytes: bytes,
    width_mm: float,
    height_mm: float,
    max_colors: int = 6,
    max_dim: int = DEFAULT_MAX_DIM,
    smooth_curves: bool = True,
    remove_background: bool = True,
    auto_vectorize: bool = True,
    bg_strokes: list[dict] | None = None,
    force_photo: bool = False,
    crop: dict | None = None,
) -> dict:
    """Runs everything `image_to_design` does *up to* (and not including)
    stitch generation -- classify/artworkify, background removal, color
    quantization, raster-to-polygon tracing -- and returns the traced
    regions themselves as plain, JSON-ready vector data instead of
    immediately turning them into stitch objects.

    This exists for the "the chosen background-removal image must become a
    real, editable vector, not just get handed to the digitizer as a
    raster" requirement: a client calls this once, shows the user the
    traced regions as editable shapes (select/delete/redraw), and only then
    calls `design_from_region_records` with whatever region list the user
    ended up confirming -- which may differ from what this function
    returned (a region removed, merged, or hand-edited).

    `bg_strokes`, when given, is a list of manual mask-correction brush
    strokes (see `background_removal.apply_mask_strokes` for the exact
    shape) applied on top of the automatically detected background mask
    before quantization -- the "adjustable-size brush to manually restore
    or erase parts of the background-removal mask" requirement. Strokes use
    normalized (0..1) coordinates/radius, so a client can record them
    against a preview image shown at any resolution and this function
    reproduces the same correction at whatever resolution it works at
    internally. Strokes still apply even when `remove_background=False`
    (every pixel starts foreground, but an "erase" stroke can still mark
    part of it background) -- the user may have chosen "keep original" and
    still want to manually knock out a stray bit of background by hand.

    Returns:
      {"width_mm", "height_mm", "artwork_kind", "artwork_confidence",
       "regions": [{"id", "color_hex", "area_mm2",
                     "rings": [[[x_mm, y_mm], ...], ...]}, ...]}

    `"id"` matches the `region-{color_index}-{i}` scheme `image_to_design`
    gives its own objects, so a client's region edits (e.g. "the user
    deleted region-2-0") can be matched straight through to
    `design_from_region_records` without any extra bookkeeping. `"rings"`
    is `[outer_ring, *hole_rings]`, exactly like an `EmbroideryObject`'s own
    `geometry` for a FILL/CROSS_STITCH/PUFF_3D region.

    `force_photo` -- see `image_to_design`'s docstring; same meaning here
    (the "extract a design from a photo of an already-embroidered item"
    upload entry point sets this unconditionally).

    `crop` -- see `image_to_design`'s docstring; same meaning here (the
    "extract only the design from a photo of someone wearing the garment"
    requirement -- a caller gets a suggested box from
    digitizer.segmentation.design_region.detect_design_region, the user
    confirms/adjusts it, and only the confirmed box is passed here).

    Raises ValueError under the same conditions as `image_to_design`
    (unreadable image, degenerate dimensions, no detectable foreground).
    """
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("width_mm and height_mm must both be positive")
    if not (1 <= max_colors <= 16):
        raise ValueError("max_colors must be between 1 and 16")
    if not (MIN_MAX_DIM <= max_dim <= MAX_MAX_DIM):
        raise ValueError(f"max_dim must be between {MIN_MAX_DIM} and {MAX_MAX_DIM}")

    try:
        img = load_and_downscale(image_bytes, max_dim)
    except Exception as e:  # noqa: BLE001 - Pillow raises several distinct error types
        raise ValueError(f"could not read image: {e}") from e
    img = _apply_crop(img, crop)

    w, h = img.size
    px_to_mm_x = width_mm / w
    px_to_mm_y = height_mm / h

    img, artwork_kind, artwork_confidence, _photo_label_grid_kwargs = _classify_and_artworkify(
        img, auto_vectorize, force_photo
    )

    rgba = img.convert("RGBA")
    if remove_background:
        bg_mask = compute_background_mask(rgba)
    else:
        bg_mask = [[False] * w for _ in range(h)]
    if bg_strokes:
        bg_mask = apply_mask_strokes(bg_mask, w, h, bg_strokes)

    # Same fix `image_to_design` already applies for photo-classified input
    # (see its own docstring and `build_label_grid`'s `background_blend_
    # tolerance` docstring for the full story): the default tolerance's
    # cross-color blend-line test assumes a flat logo on a single solid
    # background, and reliably finds *some* plausible blend path linking
    # any two unrelated colors once the background itself has many
    # reference shades (a photographed garment's lighting gradient) --
    # silently collapsing a genuinely multi-color design (confirmed here:
    # a two-thread photographed monogram collapsing to one washed-out
    # color and dozens of texture-noise fragments) down to one color.
    # `artworkify`'s posterize step already does this rule's job (flattening
    # real anti-aliasing to discrete levels before quantization), so fully
    # disabling it here is safe, not just tolerable -- exactly the same
    # reasoning `image_to_design` already relies on. Without this, this
    # function silently produced far worse regions than `image_to_design`
    # for the exact same photographed input, which defeats the whole point
    # of editing a vector that's supposed to represent the same design.
    palette_hex, label_grid = build_label_grid(
        img, max_colors, bg_mask_override=bg_mask, **_photo_label_grid_kwargs
    )
    if _is_photo_kind(artwork_kind):
        label_grid = despeckle_label_grid(label_grid)

    region_records, _total_area_mm2, _threads = _build_region_records(
        label_grid, palette_hex, px_to_mm_x, px_to_mm_y, smooth_curves
    )

    if not region_records:
        raise ValueError(
            "no foreground regions detected -- image may be blank, fully transparent, "
            "or a single solid color matching the detected background"
        )

    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "artwork_kind": artwork_kind,
        "artwork_confidence": artwork_confidence,
        "regions": [
            {
                "id": f"region-{rec['color_index']}-{rec['i']}",
                "color_hex": rec["color_hex"],
                "area_mm2": rec["area_mm2"],
                "rings": [[[round(x, 4), round(y, 4)] for x, y in ring] for ring in rec["rings"]],
            }
            for rec in region_records
        ],
    }


def design_from_region_records(
    regions: list[dict],
    width_mm: float,
    height_mm: float,
    stitch_length_mm: float = 3.0,
    density_mm: float | None = None,
    angle_deg: float = 0.0,
    stitch_style: str = "fill",
    border_width_mm: float | None = None,
    artwork_kind: str = "flat",
    artwork_confidence: float = 1.0,
) -> EmbroideryDesign:
    """The other half of the editable-vector pipeline (see
    `image_to_vector_regions`): turns a region list -- normally that
    function's own output, after a client has let the user select/delete/
    redraw regions on top of it -- into a real `EmbroideryDesign`, with
    identical per-region SATIN/BEAN/MOTIF/PUFF_3D/APPLIQUE auto-
    classification to `image_to_design`, so a design built this way looks
    exactly like one built straight from the image (for whatever regions
    the user chose to keep).

    `regions` is `[{"color_hex", "rings"}, ...]` -- `"id"`/`"area_mm2"`, if
    present (e.g. passed straight back from `image_to_vector_regions`), are
    ignored/recomputed here, so a client doesn't need to strip them before
    resubmitting. `"rings"` is `[outer_ring, *hole_rings]`, each ring a list
    of `[x_mm, y_mm]` (or `(x_mm, y_mm)`) pairs -- exactly the shape
    `image_to_vector_regions` returns, so its output (minus whatever the
    user removed/edited) can be passed straight back in.

    Raises ValueError for degenerate dimensions, an unrecognized
    `stitch_style`, an empty/invalid region list, or a region list that
    (after classification) produces no stitch objects at all.
    """
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("width_mm and height_mm must both be positive")
    if stitch_style not in STITCH_STYLES:
        raise ValueError(f"stitch_style must be one of {STITCH_STYLES}, got {stitch_style!r}")
    if not regions:
        raise ValueError("'regions' must be a non-empty list")

    effective_density_mm = density_mm if density_mm is not None else _DEFAULT_DENSITY_MM_BY_STYLE[stitch_style]
    if effective_density_mm <= 0:
        raise ValueError("density_mm must be > 0")
    effective_border_mm = border_width_mm if border_width_mm is not None else _DEFAULT_BORDER_MM_BY_STYLE[stitch_style]
    if effective_border_mm < 0:
        raise ValueError("border_width_mm must be >= 0")

    region_records: list[dict] = []
    total_foreground_area_mm2 = 0.0
    threads: list[str] = []
    seen_colors: dict[str, int] = {}
    counts_by_color: dict[int, int] = {}
    for region in regions:
        color_hex = str(region.get("color_hex") or "#000000")
        raw_rings = region.get("rings")
        if not raw_rings or not raw_rings[0] or len(raw_rings[0]) < 3:
            continue  # degenerate ring (no outer boundary, or too few points to be a polygon)
        rings = [[(float(x), float(y)) for x, y in ring] for ring in raw_rings]

        color_index = seen_colors.setdefault(color_hex, len(seen_colors))
        if color_index == len(threads):
            threads.append(color_hex)
        i = counts_by_color.get(color_index, 0)
        counts_by_color[color_index] = i + 1

        area_mm2 = 0.0
        try:
            poly = Polygon(rings[0])
            if not poly.is_valid:
                poly = poly.buffer(0)
            area_mm2 = poly.area
        except Exception:  # noqa: BLE001 - degenerate ring geometry, treat as zero area
            area_mm2 = 0.0
        total_foreground_area_mm2 += area_mm2
        region_records.append(
            {"color_index": color_index, "i": i, "color_hex": color_hex, "rings": rings, "area_mm2": area_mm2}
        )

    if not region_records:
        raise ValueError("'regions' contained no usable (non-degenerate) shapes")

    objects = _region_records_to_objects(
        region_records,
        total_foreground_area_mm2,
        stitch_style,
        stitch_length_mm,
        effective_density_mm,
        angle_deg,
        effective_border_mm,
    )

    return EmbroideryDesign(
        width_mm=width_mm,
        height_mm=height_mm,
        objects=objects,
        threads=threads,
        metadata={"artwork_kind": artwork_kind, "artwork_confidence": artwork_confidence},
    )


def remove_background_preview(image_bytes: bytes, max_dim: int = PREVIEW_MAX_DIM, crop: dict | None = None) -> bytes:
    """
    Standalone background-removal *preview* step (brief follow-up: "show
    the user what will be extracted from their photo and require an
    explicit confirmation before digitizing, never silently apply it").

    Runs the exact same border-connected flood-fill background detection
    `image_to_design` uses internally (digitizer.segmentation.
    background_removal.compute_background_mask) but, unlike the full
    pipeline, does nothing else with it -- no quantizing, no segmentation,
    no stitch generation -- and returns a real PNG with every detected
    background/surrounding-material pixel's alpha channel zeroed out, so a
    client can render it directly (e.g. over a checkerboard, the universal
    "this is transparent" convention) for the user to look at and approve.

    Deliberately processed at a much higher resolution (`PREVIEW_MAX_DIM`,
    900px by default) than the 150px the full digitizing pipeline caps
    itself to -- this image is only ever looked at by a human, not reduced
    to stitch geometry, so there's no reason to sacrifice visual fidelity
    for processing speed here the way the stitch pipeline must.

    `crop` -- see `image_to_design`'s docstring; same meaning here. Lets a
    caller preview the background mask against just the user-confirmed
    design region (create/auto-digitize/design-region.tsx) rather than the
    whole original photo, once that step has run.

    Raises ValueError for an unreadable image (same contract as
    `image_to_design`).
    """
    try:
        img = load_and_downscale(image_bytes, max_dim)
    except Exception as e:  # noqa: BLE001 - Pillow raises several distinct error types
        raise ValueError(f"could not read image: {e}") from e
    img = _apply_crop(img, crop)

    rgba = img.convert("RGBA")
    mask = compute_background_mask(rgba)
    w, h = rgba.size
    px = rgba.load()
    for y in range(h):
        for x in range(w):
            if mask[y][x]:
                r, g, b, _a = px[x, y]
                px[x, y] = (r, g, b, 0)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()
