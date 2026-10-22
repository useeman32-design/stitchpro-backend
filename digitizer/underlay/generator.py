"""
Underlay stitch generator (brief section 13).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md). The three named strategies
(center-walk, contour/edge, sparse zig-zag) are documented, concepts-only,
in research/inkstitch-notes.md: underlay's job is to stabilize/flatten the
fabric *before* the top (satin/fill) layer is stitched, so the top layer
sits flatter and the finished design comes out closer to its intended size
and shape. It is always generated to stitch immediately before an object's
own top-stitches, in the same thread (no color change needed between
underlay and its object's top stitch).

Strategies (set via `StitchSettings.underlay`):
  - "center_run": a single running stitch down the middle of the shape.
    SATIN: the midline between the two rails. FILL: not really meaningful
    for an area (there's no "middle line" of a 2D region the way there is
    for a column) -- falls back to the same sparse cross-hatch pass as
    "zigzag", see below.
  - "edge_run": running stitch just inside the shape's own boundary.
    SATIN: both rails, each inset slightly toward the other so the
    underlay doesn't peek out past the finished satin edge. FILL: the
    outer boundary and every hole's boundary, inset inward/outward
    (respectively) by the same amount, computed correctly for holes via a
    single Shapely negative buffer on the whole polygon-with-holes.
  - "zigzag": a sparse, low-density pass of the object's own top-stitch
    type (satin or fill), at a much larger density than the final layer
    and (for fill) angled 45 degrees off the top layer's direction -- a
    standard "cross-hatch" underlay technique that locks the fabric in two
    directions instead of one.

RUNNING objects have no underlay strategy implemented (a running stitch is
usually itself an outline/detail line, not something that needs its own
stabilization pass) -- `generate_underlay_stitches` always returns `[]` for
them, regardless of `stitch_settings.underlay`.
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon

from digitizer.model import EmbroideryObject, ObjectType
from digitizer.stitches.fill.generator import generate_fill_stitch
from digitizer.stitches.running.generator import generate_running_stitch
from digitizer.stitches.satin.generator import generate_satin_stitch

Point = tuple[float, float]
Stitch = tuple[float, float, bool]

_DEFAULT_UNDERLAY_STITCH_LENGTH_MM = 2.5
_DEFAULT_EDGE_INSET_MM = 0.3


def _as_run(points: list[Point]) -> list[Stitch]:
    return [(x, y, i == 0) for i, (x, y) in enumerate(points)]


def _inset_point(p: Point, toward: Point, amount: float) -> Point:
    """Move `p` toward `toward` by `amount` mm (clamped to at most 40% of
    the distance between them, so a very narrow column's underlay can't
    cross past the midline)."""
    dx, dy = toward[0] - p[0], toward[1] - p[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return p
    t = min(amount, dist * 0.4) / dist
    return (p[0] + dx * t, p[1] + dy * t)


def _satin_underlay(obj: EmbroideryObject) -> list[Stitch]:
    strategy = obj.stitch_settings.underlay
    rail_a = obj.geometry[0::2]
    rail_b = obj.geometry[1::2]
    stitch_len = _DEFAULT_UNDERLAY_STITCH_LENGTH_MM

    if strategy == "center_run":
        midline = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(rail_a, rail_b)]
        return _as_run(generate_running_stitch(midline, stitch_len))

    if strategy == "edge_run":
        inset_a = [_inset_point(a, b, _DEFAULT_EDGE_INSET_MM) for a, b in zip(rail_a, rail_b)]
        inset_b = [_inset_point(b, a, _DEFAULT_EDGE_INSET_MM) for a, b in zip(rail_a, rail_b)]
        out: list[Stitch] = []
        out.extend(_as_run(generate_running_stitch(inset_a, stitch_len)))
        out.extend(_as_run(generate_running_stitch(inset_b, stitch_len)))
        return out

    if strategy == "zigzag":
        sparse_density = max(obj.stitch_settings.density_mm * 4, 1.5)
        return _as_run(generate_satin_stitch(obj.geometry, density_mm=sparse_density, pull_compensation_mm=0.0))

    return []


def _fill_underlay(obj: EmbroideryObject) -> list[Stitch]:
    strategy = obj.stitch_settings.underlay
    rings: list[list[Point]] = obj.geometry
    stitch_len = _DEFAULT_UNDERLAY_STITCH_LENGTH_MM

    if strategy == "edge_run":
        polygon = Polygon(shell=rings[0], holes=rings[1:])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        inset = polygon.buffer(-_DEFAULT_EDGE_INSET_MM, join_style=2)
        if inset.is_empty:
            return []
        geoms = list(inset.geoms) if hasattr(inset, "geoms") else [inset]
        out: list[Stitch] = []
        for g in geoms:
            if g.is_empty:
                continue
            contours = [list(g.exterior.coords)] + [list(r.coords) for r in g.interiors]
            for contour in contours:
                out.extend(_as_run(generate_running_stitch(contour, stitch_len)))
        return out

    if strategy in ("zigzag", "center_run"):
        # "center_run" has no literal meaning for a 2D area fill (there's no
        # single "middle line" the way there is for a satin column) -- it
        # falls back to the same sparse cross-hatch pass as "zigzag".
        sparse_density = max(obj.stitch_settings.density_mm * 5, 2.0)
        sparse_stitch_len = obj.stitch_settings.stitch_length_mm * 1.5
        cross_angle = obj.stitch_settings.angle_deg + 45
        return generate_fill_stitch(rings, sparse_stitch_len, sparse_density, cross_angle, 0.0)

    return []


def generate_underlay_stitches(obj: EmbroideryObject) -> list[Stitch]:
    """Return the (x, y, is_jump) underlay stitches for `obj`, or `[]` if
    `stitch_settings.underlay` is `None`, unrecognized, or the object's type
    has no underlay strategy implemented (RUNNING)."""
    if not obj.stitch_settings.underlay:
        return []
    if obj.type == ObjectType.SATIN:
        return _satin_underlay(obj)
    if obj.type == ObjectType.FILL:
        return _fill_underlay(obj)
    return []
