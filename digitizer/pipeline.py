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

from digitizer.classifier.classify_artwork import classify_artwork
from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.segmentation.artworkify import artworkify
from digitizer.segmentation.background_removal import compute_background_mask
from digitizer.segmentation.color_quantize import build_label_grid, despeckle_label_grid
from digitizer.segmentation.polygons import label_grid_to_ring_groups
from digitizer.stitches.satin.thin_region import (
    generate_satin_rungs_for_thin_region,
    is_satin_eligible,
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

    `density_mm` left as `None` (the default) picks a sensible default for
    whichever `stitch_style` was chosen (see `_DEFAULT_DENSITY_MM_BY_STYLE`)
    rather than forcing one fixed number to do double duty as both a
    sensible fill-row spacing and a sensible cross-stitch cell size.

    `border_width_mm` likewise defaults (when left `None`) per
    `_DEFAULT_BORDER_MM_BY_STYLE` -- a satin edge-walk border around every
    region's outline and holes for FILL, none for CROSS_STITCH. Pass `0`
    explicitly to disable it for FILL too.

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

    w, h = img.size
    px_to_mm_x = width_mm / w
    px_to_mm_y = height_mm / h

    label_grid_kwargs: dict[str, float] = {}
    if auto_vectorize:
        artwork_info = classify_artwork(img)
        artwork_kind = artwork_info["kind"]
        artwork_confidence = artwork_info["confidence"]
        if artwork_kind == "photo":
            img = artworkify(img)
            label_grid_kwargs["background_blend_tolerance"] = _PHOTO_BACKGROUND_BLEND_TOLERANCE
    else:
        artwork_kind = "flat (forced)"
        artwork_confidence = 1.0

    palette_hex, label_grid = build_label_grid(
        img,
        max_colors,
        remove_background=remove_background,
        **label_grid_kwargs,
    )
    if artwork_kind == "photo":
        # Collapses the dozens of tiny texture-noise islands a close-up
        # photo's per-thread shading leaves behind (even after artworkify's
        # denoise+posterize) into their surrounding real region, instead of
        # each becoming its own near-useless stitch object -- see
        # despeckle_label_grid's own docstring for why this is a different
        # (and necessary, in addition to) fix from label_grid_to_ring_groups'
        # own min_area_px/min_bbox_mm filters below.
        label_grid = despeckle_label_grid(label_grid)

    object_type = ObjectType.FILL if stitch_style == "fill" else ObjectType.CROSS_STITCH

    objects: list[EmbroideryObject] = []
    threads: list[str] = []
    seq = 0
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
            # Only FILL regions are considered for auto-SATIN -- CROSS_STITCH
            # is a deliberate stylistic choice the user made explicitly for
            # every region, not something a thin-shape heuristic should
            # override (see module docstring and thin_region.py's).
            satin_rungs = None
            if stitch_style == "fill" and is_satin_eligible(rings):
                satin_rungs = generate_satin_rungs_for_thin_region(rings[0])

            if satin_rungs is not None:
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
                            # Only ever applies to the FILL branch in
                            # practice -- generate_underlay_stitches already
                            # no-ops for CROSS_STITCH regardless of this
                            # value, kept here rather than branched on
                            # object_type so this object-construction call
                            # doesn't need its own copy of that type check.
                            underlay=_FILL_UNDERLAY if object_type == ObjectType.FILL else None,
                            pull_compensation_mm=_DEFAULT_PULL_COMPENSATION_MM_BY_STYLE[stitch_style],
                        ),
                        sequence=seq,
                        layer=0,
                    )
                )
            seq += 1

    if not objects:
        raise ValueError(
            "no foreground regions detected -- image may be blank, fully transparent, "
            "or a single solid color matching the detected background"
        )

    return EmbroideryDesign(
        width_mm=width_mm,
        height_mm=height_mm,
        objects=objects,
        threads=threads,
        metadata={"artwork_kind": artwork_kind, "artwork_confidence": artwork_confidence},
    )


def remove_background_preview(image_bytes: bytes, max_dim: int = PREVIEW_MAX_DIM) -> bytes:
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

    Raises ValueError for an unreadable image (same contract as
    `image_to_design`).
    """
    try:
        img = load_and_downscale(image_bytes, max_dim)
    except Exception as e:  # noqa: BLE001 - Pillow raises several distinct error types
        raise ValueError(f"could not read image: {e}") from e

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
