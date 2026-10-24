"""
Automatic SATIN classification for thin/elongated regions (fill-quality
follow-up to the satin edge-walk border work in stitches/satin/border.py).

Why this exists
----------------
Visual study of real professional reference DST/DSB files (text wordmarks,
badge borders, decorative line art) showed that essentially none of their
text, border, or line-art content is built from tatami fill rows -- it's
built almost entirely from dense zigzag satin columns following the stroke
direction. Our pipeline, by contrast, forces *every* detected region
through FILL (or CROSS_STITCH) regardless of shape, including thin regions
like font glyph strokes, badge border bars, and logo line-art -- a tatami
fill run through a narrow shape visually reads as a handful of sparse
parallel lines rather than a clean stitched stroke, which is a strong
contributor to auto-digitized text/line-art output looking unprofessional.

This module decides, per region, whether it's a good candidate for a
*full* satin column (the entire shape stitched as one zigzag column,
replacing FILL+border entirely for that region) instead of just leaving it
to FILL's existing edge-walk border (stitches/satin/border.py) to dress up
the outline while the sparse interior fill remains the dominant look.

Deliberately conservative scope
--------------------------------
Converting an arbitrary polygon into satin rung geometry requires knowing
the shape's "long axis" and, at every point along it, exactly two boundary
crossings perpendicular to that axis (the two rails). This holds up well
for straight or gently-curved elongated shapes (a letter's straight stroke,
a badge's border bar, a ruled line) but breaks down for shapes that bend
sharply or loop back on themselves relative to a single fixed global axis
(a tightly curved "C"/"S"/"O" glyph, a shape with a hole) -- a perpendicular
scan line can cross such a shape's boundary more than twice at some
positions, which would produce an ambiguous or self-crossing satin column
if forced through anyway.

Rather than attempt a full medial-axis/skeleton extraction (a much larger,
higher-risk undertaking), this module uses the much simpler and cheaper
*oriented bounding box* of the region to estimate a long axis, then
conservatively validates that assumption by actually scanning the shape:
if any scan line's intersection with the polygon isn't exactly one clean
sub-segment, classification bails out and the caller falls back to the
existing FILL (+ border) path for that region -- never a broken or
self-crossing satin column. This means curved letters and shapes with
holes keep today's FILL behavior (itself already correct, per the prior
density/boustrophedon work); only genuinely straight-enough, hole-free,
sufficiently thin and elongated regions get the new satin treatment. That
subset still covers a meaningful fraction of real text (most sans-serif
capitals and digits are mostly straight strokes) and essentially all
badge/line-art border bars.
"""

from __future__ import annotations

import math

import shapely
from shapely.geometry import LineString, Polygon

Point = tuple[float, float]

# A region's oriented-bounding-box width must be at or under this to be a
# satin candidate at all -- comfortably covers real stroke widths (a few mm)
# without reaching into territory that's really a filled shape, not a
# stroke.
DEFAULT_MAX_SATIN_WIDTH_MM = 7.0

# length / width of the oriented bounding box must be at least this to
# count as "elongated" rather than a round/blobby thin shape (a small dot
# or short serif shouldn't become a satin column).
DEFAULT_MIN_ELONGATION_RATIO = 2.2

# How finely the shape is scanned along its long axis, both to build the
# final satin rungs and to validate the single-segment-per-scan-line
# assumption. Fine enough to follow real letter curvature, coarse enough
# to keep the eligibility check cheap.
DEFAULT_STEP_MM = 1.0

# Sub-segments shorter than this (mm) from a scan-line intersection are
# treated as numerical noise (e.g. a scan line grazing a vertex) rather
# than a real second crossing.
_MIN_SEGMENT_LEN_MM = 0.05


def _oriented_axis(outer_ring_mm: list[Point]) -> tuple[tuple[float, float], tuple[float, float], float, float] | None:
    """Returns `(direction_unit_vector, perpendicular_unit_vector,
    length_mm, width_mm)` for `outer_ring_mm`'s minimum-area oriented
    bounding rectangle, or `None` if the ring is degenerate. `direction` is
    along the rectangle's longer side (the shape's long axis)."""
    poly = Polygon(outer_ring_mm)
    if not poly.is_valid or poly.area < 1e-9:
        return None
    rect = shapely.oriented_envelope(poly)
    if rect.is_empty or rect.geom_type != "Polygon":
        return None
    coords = list(rect.exterior.coords)[:-1]
    if len(coords) != 4:
        return None

    side_lengths = []
    for i in range(4):
        a, b = coords[i], coords[(i + 1) % 4]
        side_lengths.append(math.hypot(b[0] - a[0], b[1] - a[1]))

    # Opposite sides of a rectangle are equal length; side 0 and side 2 are
    # one pair, side 1 and side 3 the other. Pick whichever pair is longer
    # as the long axis.
    pair_a_len = (side_lengths[0] + side_lengths[2]) / 2
    pair_b_len = (side_lengths[1] + side_lengths[3]) / 2
    if pair_a_len >= pair_b_len:
        length_mm, width_mm = pair_a_len, pair_b_len
        a, b = coords[0], coords[1]
    else:
        length_mm, width_mm = pair_b_len, pair_a_len
        a, b = coords[1], coords[2]

    dx, dy = b[0] - a[0], b[1] - a[1]
    dlen = math.hypot(dx, dy)
    if dlen < 1e-9:
        return None
    direction = (dx / dlen, dy / dlen)
    perpendicular = (-direction[1], direction[0])
    return direction, perpendicular, length_mm, width_mm


def is_satin_eligible(
    rings: list[list[Point]],
    max_satin_width_mm: float = DEFAULT_MAX_SATIN_WIDTH_MM,
    min_elongation_ratio: float = DEFAULT_MIN_ELONGATION_RATIO,
) -> bool:
    """Cheap, geometry-only pre-filter: should this region even be
    *considered* for satin classification? Does not guarantee
    `generate_satin_rungs_for_thin_region` will succeed -- that still does
    its own scan-based validation -- this just avoids paying for that scan
    on regions that obviously aren't candidates (anything with a hole, or
    not both thin and elongated).
    """
    if len(rings) != 1:
        return False  # has at least one hole -- v1 limitation, see module docstring
    outer = rings[0]
    if len(outer) < 3:
        return False
    axis = _oriented_axis(outer)
    if axis is None:
        return False
    _direction, _perp, length_mm, width_mm = axis
    if width_mm <= 0 or width_mm > max_satin_width_mm:
        return False
    if length_mm / width_mm < min_elongation_ratio:
        return False
    return True


def _as_line_segments(geom, min_len: float = _MIN_SEGMENT_LEN_MM) -> list[tuple[Point, Point]]:
    """Normalize a shapely intersection result (LineString, MultiLineString,
    GeometryCollection, Point, or empty) into a list of 2-point segments,
    dropping degenerate near-zero-length pieces (numerical noise from a
    scan line grazing a vertex) and ignoring zero-width Point touches."""
    if geom is None or geom.is_empty:
        return []
    segments: list[tuple[Point, Point]] = []
    gt = geom.geom_type
    if gt == "LineString":
        coords = list(geom.coords)
        if len(coords) >= 2:
            a, b = coords[0], coords[-1]
            if math.hypot(b[0] - a[0], b[1] - a[1]) >= min_len:
                segments.append((a, b))
    elif gt in ("MultiLineString", "GeometryCollection"):
        for part in geom.geoms:
            segments.extend(_as_line_segments(part, min_len))
    # Point/MultiPoint (a scan line just touching the boundary) contributes
    # no usable segment and is intentionally ignored.
    return segments


def generate_satin_rungs_for_thin_region(
    outer_ring_mm: list[Point],
    step_mm: float = DEFAULT_STEP_MM,
) -> list[Point] | None:
    """Convert a hole-free region's outer boundary into satin rung geometry
    (the flat `[A0, B0, A1, B1, ...]` shape `generate_satin_stitch` expects)
    by scanning perpendicular to the shape's oriented-bounding-box long
    axis at `step_mm` intervals.

    Returns `None` (meaning: not safe to treat as one satin column, caller
    should fall back to FILL) whenever any scan line's intersection with
    the polygon isn't exactly one clean sub-segment -- see module docstring
    for why this conservative bail-out is the right trade-off instead of
    guessing at an ambiguous or self-crossing column.
    """
    if step_mm <= 0:
        raise ValueError("step_mm must be > 0")
    if len(outer_ring_mm) < 3:
        return None

    poly = Polygon(outer_ring_mm)
    if not poly.is_valid or poly.area < 1e-9:
        return None

    axis = _oriented_axis(outer_ring_mm)
    if axis is None:
        return None
    direction, perpendicular, _length_mm, width_mm = axis
    dx, dy = direction
    px, py = perpendicular

    coords = list(poly.exterior.coords)
    t_values = [x * dx + y * dy for x, y in coords]
    s_values = [x * px + y * py for x, y in coords]
    t_min, t_max = min(t_values), max(t_values)
    span = t_max - t_min
    if span < step_mm:
        return None  # too short along its own long axis to be worth a column

    # Generous perpendicular half-length for the scan line so it always
    # fully crosses the shape even where it locally bulges past the
    # oriented bounding box's own width estimate.
    s_half = max(width_mm, max(s_values) - min(s_values)) * 2.0 + 1.0
    s_mid = (max(s_values) + min(s_values)) / 2.0

    def point_at(t: float, s: float) -> Point:
        return (t * dx + s * px, t * dy + s * py)

    n_steps = max(2, math.ceil(span / step_mm))
    rungs: list[Point] = []
    for k in range(n_steps + 1):
        t = t_min + span * k / n_steps
        scan_line = LineString([point_at(t, s_mid - s_half), point_at(t, s_mid + s_half)])
        segments = _as_line_segments(poly.intersection(scan_line))
        if len(segments) != 1:
            return None  # no clean crossing, or more than one -- bail
        a_pt, b_pt = segments[0]
        # Order consistently along the perpendicular axis so rail A/rail B
        # never swap sides from one rung to the next (which would twist the
        # column into a bowtie instead of a clean zigzag).
        sa = a_pt[0] * px + a_pt[1] * py
        sb = b_pt[0] * px + b_pt[1] * py
        if sa > sb:
            a_pt, b_pt = b_pt, a_pt
        rungs.append(a_pt)
        rungs.append(b_pt)

    if len(rungs) < 4:
        return None
    return rungs
