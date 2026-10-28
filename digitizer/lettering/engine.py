"""
Text -> EmbroideryDesign, using the dedicated Bold Block lettering alphabet
(digitizer/lettering/alphabet_block.py) and the skeleton-to-satin-column
converter (digitizer/lettering/skeleton.py) -- NOT the generic photo/
artwork auto-digitizing pipeline (digitizer/pipeline.py). See this
package's docstring for why text gets its own path: we already know the
exact letterform geometry, so there's no classification/segmentation/
color-quantization guesswork to do at all, and the result is real
satin-column lettering (including clean curves on O/S/C/G/...) instead of
whatever FILL/SATIN mix the generic per-region classifier happens to land
on for a rasterized text image.

Scope for this version: straight-baseline text only. "Arc Up"/"Arc Down"
layouts keep using the older render-to-image + generic-auto-digitize path
(still real, still works, just not satin-column-clean) -- bending each
glyph's own skeleton around a tight embroidery-sized arc without
distorting individual letterforms needs more care than a first version
should rush; a future pass can add a proper per-glyph rigid-body arc
placement (rotate+translate each whole glyph to be tangent to a circle at
its own center, rather than warping skeleton points individually) without
changing this module's straight-layout path at all.
"""

from __future__ import annotations

from digitizer.lettering.alphabet_block import (
    CAP_HEIGHT_UNITS,
    DEFAULT_STROKE_WIDTH_UNITS,
    LETTER_GAP_UNITS,
    glyph_for,
    is_fully_supported,
)
from digitizer.lettering.skeleton import arc_points, polyline_points, stroke_to_satin_geometry
from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings

Point = tuple[float, float]

SUPPORTED_LAYOUTS = {"straight"}


def _stroke_skeleton(stroke: tuple, step_units: float) -> list[Point]:
    kind = stroke[0]
    if kind == "line":
        return polyline_points(stroke[1], step=step_units)
    if kind == "arc":
        _, cx, cy, rx, ry, start_deg, end_deg = stroke
        return arc_points(cx, cy, rx, ry, start_deg, end_deg, step_deg=8.0)
    raise ValueError(f"unknown stroke kind {kind!r}")


def build_text_design(
    text: str,
    size_mm: float,
    color: str = "#111111",
    layout: str = "straight",
    density_mm: float = 0.35,
    pull_compensation_mm: float = 0.08,
    stroke_width_ratio: float = DEFAULT_STROKE_WIDTH_UNITS / CAP_HEIGHT_UNITS,
) -> EmbroideryDesign:
    """Build a real EmbroideryDesign for `text` using the dedicated Bold
    Block satin-column lettering engine.

    `size_mm` is the cap height in millimeters (matches how embroidery
    lettering is normally sized/quoted, e.g. "1 inch letters"). Raises
    ValueError if `text` contains any character the Bold Block alphabet
    doesn't define, or if `layout` isn't supported yet -- callers should
    check `is_fully_supported(text)` / `layout in SUPPORTED_LAYOUTS` first
    and fall back to the generic render-and-auto-digitize path otherwise.
    """
    if not text:
        raise ValueError("text must be non-empty")
    if size_mm <= 0:
        raise ValueError("size_mm must be > 0")
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"unsupported layout {layout!r} for the dedicated lettering engine, expected 'straight'")
    if not is_fully_supported(text):
        unsupported = sorted({ch for ch in text if glyph_for(ch) is None})
        raise ValueError(f"characters not supported by the Bold Block alphabet: {unsupported!r}")

    scale = size_mm / CAP_HEIGHT_UNITS
    stroke_width_units = stroke_width_ratio * CAP_HEIGHT_UNITS

    # Lay out glyphs left-to-right in unit space; collect every stroke's
    # skeleton points (already offset by that glyph's x cursor).
    cursor_x = 0.0
    glyph_strokes: list[list[Point]] = []
    for ch in text:
        advance, strokes = glyph_for(ch)
        for stroke in strokes:
            pts = _stroke_skeleton(stroke, step_units=0.5)
            glyph_strokes.append([(cursor_x + x, y) for x, y in pts])
        cursor_x += advance + LETTER_GAP_UNITS
    text_width_units = max(0.0, cursor_x - LETTER_GAP_UNITS)

    # Scale to mm and center horizontally around x=0 for a predictable
    # design bounding box; build one SATIN object per stroke.
    half_width_mm = (text_width_units * scale) / 2.0
    objects: list[EmbroideryObject] = []
    settings = StitchSettings(density_mm=density_mm, pull_compensation_mm=pull_compensation_mm, angle_deg=0.0)
    for i, pts in enumerate(glyph_strokes):
        scaled = [(x * scale - half_width_mm, y * scale) for x, y in pts]
        geometry = stroke_to_satin_geometry(scaled, width_units=stroke_width_units * scale)
        objects.append(
            EmbroideryObject(
                id=f"letter-{i}",
                type=ObjectType.SATIN,
                geometry=geometry,
                color=color,
                stitch_settings=settings,
                sequence=i,
                layer=0,
            )
        )

    xs = [p[0] for pts in glyph_strokes for p in pts] if glyph_strokes else [0.0]
    ys = [p[1] for pts in glyph_strokes for p in pts] if glyph_strokes else [0.0]
    width_mm = (max(xs) - min(xs)) * scale if xs else size_mm
    height_mm = (max(ys) - min(ys)) * scale if ys else size_mm

    return EmbroideryDesign(
        width_mm=max(width_mm, 1.0),
        height_mm=max(height_mm, 1.0),
        objects=objects,
        threads=[color],
        metadata={"source": "lettering_engine", "font": "bold_block", "layout": layout},
    )
