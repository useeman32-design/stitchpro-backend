"""
"Bold Block" embroidery lettering alphabet (brief: dedicated lettering
engine, see digitizer/lettering/engine.py).

Each glyph is authored directly as a small set of monoline "strokes" on a
fixed unit grid -- cap height = 10 units, baseline at y=10, top of
capitals at y=0, +y down (matches the rest of the model). This is exactly
how real simple embroidery "block"/"varsity" digitizing fonts are built:
a skeleton centerline per stroke, not a filled TrueType outline, so every
stroke converts cleanly into a satin column via
digitizer.lettering.skeleton.stroke_to_satin_geometry regardless of
whether it's straight or curved -- no risk of the generic auto-digitizer's
thin_region classifier bailing out on a curved letter (O/S/C/G/...) or a
glyph with a hole (O/A/B/D/P/Q/R/...), because there IS no hole here: a
closed bowl like 'O' is just a ring-shaped stroke (a skeleton circle),
stitched as one continuous satin ring, exactly like a real digitized "O".

Each stroke is one of:
  ("line", [(x0, y0), (x1, y1), ...])      -- straight polyline skeleton
                                               (2+ vertices; a vertex is a
                                               deliberate corner, e.g. the
                                               apex of "A" or the zigzag
                                               points of "M"/"W"/"Z")
  ("arc", cx, cy, rx, ry, start_deg, end_deg)  -- elliptical arc skeleton
                                               (see skeleton.arc_points for
                                               the angle convention)

GLYPHS[ch] = (advance_width_units, [stroke, stroke, ...])

Only uppercase A-Z, digits 0-9, space, and a handful of common punctuation
are defined for this first version -- lowercase and the "Modern Sans" /
"Classic Serif" / "Script" font styles are a deliberate fast-follow (see
continue.md); those three still use the older generic image-digitizing
path (render-to-image, then auto-digitize) until they get their own
dedicated glyph sets the same way "Bold Block"/"Varsity" just did.
"""

from __future__ import annotations

Point = tuple[float, float]
Stroke = tuple  # ("line", [pts]) | ("arc", cx, cy, rx, ry, start, end)

CAP_HEIGHT_UNITS = 10.0
DEFAULT_STROKE_WIDTH_UNITS = 1.7
LETTER_GAP_UNITS = 1.6  # extra horizontal gap added between glyphs' advances
SPACE_WIDTH_UNITS = 4.5


def _line(*pts: Point) -> Stroke:
    return ("line", list(pts))


def _arc(cx: float, cy: float, r: float, start_deg: float, end_deg: float) -> Stroke:
    return ("arc", cx, cy, r, r, start_deg, end_deg)


GLYPHS: dict[str, tuple[float, list[Stroke]]] = {
    "A": (7.0, [_line((0, 10), (3.5, 0), (7, 10)), _line((1.5, 6.2), (5.5, 6.2))]),
    "B": (
        6.5,
        [
            _line((0, 0), (0, 10)),
            _arc(0, 2.6, 2.6, -90, 90),
            _arc(0, 7.4, 2.9, -90, 90),
        ],
    ),
    "C": (7.0, [_arc(3.7, 5, 5, 40, 320)]),
    "D": (7.2, [_line((0, 0), (0, 10)), _arc(0, 5, 5, -90, 90)]),
    "E": (6.2, [_line((0, 0), (0, 10)), _line((0, 0), (6.2, 0)), _line((0, 5), (5.2, 5)), _line((0, 10), (6.2, 10))]),
    "F": (6.0, [_line((0, 0), (0, 10)), _line((0, 0), (6, 0)), _line((0, 5), (5.2, 5))]),
    "G": (7.2, [_arc(3.8, 5, 5, 20, 320), _line((7.3, 5.2), (3.9, 5.2)), _line((7.3, 5.2), (7.3, 7.6))]),
    "H": (7.2, [_line((0, 0), (0, 10)), _line((7.2, 0), (7.2, 10)), _line((0, 5), (7.2, 5))]),
    "I": (2.4, [_line((1.2, 0), (1.2, 10))]),
    "J": (5.4, [_line((4.4, 0), (4.4, 6.8)), _arc(2.2, 6.8, 2.2, 0, 165)]),
    "K": (7.0, [_line((0, 0), (0, 10)), _line((7, 0), (0, 5.2)), _line((0, 5.2), (7, 10))]),
    "L": (6.0, [_line((0, 0), (0, 10)), _line((0, 10), (6, 10))]),
    "M": (8.6, [_line((0, 10), (0, 0), (4.3, 6.2), (8.6, 0), (8.6, 10))]),
    "N": (7.4, [_line((0, 10), (0, 0), (7.4, 10), (7.4, 0))]),
    "O": (7.6, [_arc(3.8, 5, 5, 0, 360)]),
    "P": (6.4, [_line((0, 0), (0, 10)), _arc(0, 2.7, 2.7, -90, 90)]),
    "Q": (7.6, [_arc(3.8, 5, 5, 0, 360), _line((4.6, 7.2), (8.1, 11.2))]),
    "R": (7.2, [_line((0, 0), (0, 10)), _arc(0, 2.7, 2.7, -90, 90), _line((2.2, 5.4), (7.2, 10))]),
    "S": (
        6.6,
        [
            _arc(3.5, 2.7, 2.7, 0, -270),
            _arc(3.5, 7.3, 2.7, 180, -90),
        ],
    ),
    "T": (6.8, [_line((0, 0), (6.8, 0)), _line((3.4, 0), (3.4, 10))]),
    "U": (7.2, [_line((0, 0), (0, 6.5)), _arc(3.6, 6.5, 3.6, 180, 0), _line((7.2, 6.5), (7.2, 0))]),
    "V": (7.0, [_line((0, 0), (3.5, 10), (7, 0))]),
    "W": (9.6, [_line((0, 0), (2.4, 10), (4.8, 3.5), (7.2, 10), (9.6, 0))]),
    "X": (7.0, [_line((0, 0), (7, 10)), _line((7, 0), (0, 10))]),
    "Y": (7.0, [_line((0, 0), (3.5, 5.5), (7, 0)), _line((3.5, 5.5), (3.5, 10))]),
    "Z": (6.8, [_line((0, 0), (6.8, 0), (0, 10), (6.8, 10))]),
    "0": (7.2, [_arc(3.6, 5, 5, 0, 360)]),
    "1": (4.2, [_line((0.6, 2.2), (2.6, 0), (2.6, 10))]),
    "2": (6.8, [_arc(3.4, 2.8, 2.8, -150, 70), _line((6.0, 5.2), (0, 10), (6.8, 10))]),
    "3": (6.6, [_arc(2.6, 2.7, 2.7, -110, 95), _arc(2.9, 7.3, 2.9, -95, 110)]),
    "4": (7.2, [_line((5.0, 0), (5.0, 10)), _line((5.0, 0), (0, 7.2), (7.2, 7.2))]),
    "5": (6.6, [_line((6.4, 0), (0, 0), (0, 5), ), _arc(2.9, 7.3, 2.9, -110, 110)]),
    "6": (6.6, [_line((5.2, 0), (2.1, 6.3)), _arc(3.3, 7.3, 3.0, 0, 360)]),
    "7": (6.6, [_line((0, 0), (6.6, 0), (2.4, 10))]),
    "8": (6.6, [_arc(3.3, 2.7, 2.7, -90, 270), _arc(3.3, 7.3, 2.9, -90, 270)]),
    "9": (6.8, [_arc(3.4, 3.2, 3.2, 0, 360), _line((6.5, 3.4), (3.9, 10))]),
    " ": (SPACE_WIDTH_UNITS, []),
    "'": (2.2, [_line((1.1, 0), (1.1, 2.6))]),
    ".": (2.4, [_line((1.2, 9.4), (1.2, 10))]),
    ",": (2.4, [_line((1.2, 9.4), (0.6, 11.2))]),
    "-": (4.0, [_line((0.4, 5.2), (3.6, 5.2))]),
    "&": (
        8.0,
        [
            _arc(3.2, 6.2, 3.2, -40, 250),
            _line((5.9, 3.6), (1.2, 9.4)),
            _arc(4.6, 2.2, 1.9, 60, 340),
        ],
    ),
    "!": (2.4, [_line((1.2, 0), (1.2, 6.8)), _line((1.2, 9.4), (1.2, 10))]),
}


def glyph_for(ch: str) -> tuple[float, list[Stroke]] | None:
    """Case-insensitive glyph lookup (the Bold Block alphabet only defines
    capitals -- lowercase input is upper-cased). Returns None for any
    character with no glyph defined (caller decides how to handle that,
    e.g. skip it or fall back to the generic pipeline for that string)."""
    return GLYPHS.get(ch.upper())


def is_fully_supported(text: str) -> bool:
    """True if every character in `text` has a Bold Block glyph -- the
    caller should use this to decide whether the dedicated lettering
    engine can handle the whole string, or must fall back to the generic
    render-and-auto-digitize path for an unsupported character (e.g.
    lowercase-only script marks, non-Latin text)."""
    return all(ch.upper() in GLYPHS for ch in text)
