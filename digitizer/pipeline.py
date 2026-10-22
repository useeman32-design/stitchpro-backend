"""
Image -> EmbroideryDesign pipeline (Milestone 5: "real PNG -> segmentation
-> embroidery objects").

Unlike the earlier milestones (where the frontend already does the client-
side artwork analysis via stitch-pro's own stitchEngine.ts and this backend
only compiles an already-made stitch plan to a real file -- see
api/app.py's module docstring and formats/points_writer.py), this is the
first endpoint that does real server-side digitizing: given nothing but a
raw image, decide what the embroidery objects are.

Pipeline: downscale (`load_and_downscale` below) -> quantize + detect
background (digitizer.segmentation.color_quantize) -> per-color connected
regions with holes, in mm space (digitizer.segmentation.polygons) -> one
FILL EmbroideryObject per region. Routing (digitizer.routing.router, used
automatically by formats/dst/writer.py:compile_design) then groups same-
color objects together and orders them for minimal travel, exactly like any
other design built this way.

This first cut classifies every detected region as FILL. Thin/elongated
regions (e.g. a hand-drawn outline or a thin border) that would look better
as a SATIN column are a known follow-up (needs a skeleton/medial-axis-based
classifier) -- not implemented yet; see continue.md.
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
    density_mm: float = 0.4,
    max_dim: int = DEFAULT_MAX_DIM,
) -> EmbroideryDesign:
    """Digitize a raw image into an EmbroideryDesign of FILL objects, one
    (or more, if a color appears in several disconnected regions) per
    detected color, sized to fit `width_mm` x `height_mm`.

    Raises ValueError for an unreadable image, degenerate dimensions, or an
    image with no detectable foreground (e.g. a solid blank image).
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

    w, h = img.size
    px_to_mm_x = width_mm / w
    px_to_mm_y = height_mm / h

    palette_hex, label_grid = build_label_grid(img, max_colors)

    objects: list[EmbroideryObject] = []
    threads: list[str] = []
    seq = 0
    for color_index, color_hex in enumerate(palette_hex):
        ring_groups = label_grid_to_ring_groups(label_grid, color_index, px_to_mm_x, px_to_mm_y)
        if not ring_groups:
            continue
        threads.append(color_hex)
        for i, rings in enumerate(ring_groups):
            objects.append(
                EmbroideryObject(
                    id=f"region-{color_index}-{i}",
                    type=ObjectType.FILL,
                    geometry=rings,
                    color=color_hex,
                    stitch_settings=StitchSettings(
                        stitch_length_mm=stitch_length_mm,
                        density_mm=density_mm,
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
