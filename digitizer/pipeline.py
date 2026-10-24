"""
Image -> EmbroideryDesign pipeline (Milestone 5: "real PNG -> segmentation
-> embroidery objects").

Unlike the earlier milestones (where the frontend already does the client-
side artwork analysis via stitch-pro's own stitchEngine.ts and this backend
only compiles an already-made stitch plan to a real file -- see
api/app.py's module docstring and formats/points_writer.py), this is the
first endpoint that does real server-side digitizing: given nothing but a
raw image, decide what the embroidery objects are.

Pipeline: downscale (`load_and_downscale` below) -> quantize + remove
background/surrounding material via a border-connected flood fill
(digitizer.segmentation.color_quantize + background_removal -- robust to a
real photographed garment's lighting/shading, not just a flat studio
backdrop) -> per-color connected regions with holes, curve-smoothed, in mm
space (digitizer.segmentation.polygons) -> one stitch object per region,
styled per `stitch_style`. Routing (digitizer.routing.router, used
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

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.segmentation.background_removal import compute_background_mask
from digitizer.segmentation.color_quantize import build_label_grid
from digitizer.segmentation.polygons import label_grid_to_ring_groups
from digitizer.stitches.satin.thin_region import (
    generate_satin_rungs_for_thin_region,
    is_satin_eligible,
)


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
) -> EmbroideryDesign:
    """Digitize a raw image into an EmbroideryDesign of stitch objects, one
    (or more, if a color appears in several disconnected regions) per
    detected color, sized to fit `width_mm` x `height_mm`.

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

    palette_hex, label_grid = build_label_grid(img, max_colors, remove_background=remove_background)

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
            smooth_iterations=2 if smooth_curves else 0,
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
                        stitch_settings=StitchSettings(density_mm=effective_density_mm),
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
