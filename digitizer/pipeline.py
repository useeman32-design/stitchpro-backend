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
detected region in the design. Per-region automatic style classification
(e.g. recognizing a thin/elongated region and digitizing it as a SATIN
column instead) is a known follow-up, not implemented yet -- see
continue.md.
"""

from __future__ import annotations

import io

from PIL import Image

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.segmentation.color_quantize import build_label_grid
from digitizer.segmentation.polygons import label_grid_to_ring_groups

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

    palette_hex, label_grid = build_label_grid(img, max_colors)

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
