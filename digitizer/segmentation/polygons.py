"""
Label grid -> embroiderable polygon rings (Milestone 5, brief section 15:
"segmentation -> objects").

Technique: treat every pixel of a given label as a 1x1 axis-aligned square
in geometry space and take the union of all of them with Shapely. This one
call gets us three things we would otherwise have to hand-implement (and
would risk getting subtly wrong):

1. Connected-component splitting -- disjoint blobs of the same color don't
   touch, so they stay as separate Polygons in the resulting MultiPolygon.
2. Hole detection -- a fully-enclosed run of background pixels inside a
   blob (e.g. the counter of a letter "O") is never covered by a square, so
   GEOS (Shapely's geometry engine) represents it as an interior ring
   automatically, with no separate contour-tracing/border-following code
   needed.
3. A proper polygon boundary (not a pixel-by-pixel staircase we'd have to
   clean up ourselves) -- `Polygon.simplify()` then turns the inherent
   staircase from square-unioning into a much lower-vertex-count outline.

Verified independently against a synthetic "ring + separate blob" mask
before relying on it (see session notes) -- this is a standard use of
Shapely's own polygon algebra, not derived from any embroidery-specific
code.
"""

from __future__ import annotations

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

Point = tuple[float, float]
RingGroup = list[list[Point]]  # [outer_ring, hole_ring, hole_ring, ...]


def label_grid_to_ring_groups(
    label_grid: list[list[int]],
    label: int,
    px_to_mm_x: float,
    px_to_mm_y: float,
    min_area_px: float = 4.0,
    simplify_tolerance_mm: float = 0.25,
) -> list[RingGroup]:
    """Extract every connected region of `label` in `label_grid` as a
    ring-group ready to feed straight into `generate_fill_stitch` (brief
    section 9's FILL contract: `rings[0]` outer boundary, `rings[1:]`
    holes), in mm space.

    `min_area_px` discards noise specks (stray mis-quantized pixels) too
    small to be worth a stitch object. `simplify_tolerance_mm` reduces the
    unit-square staircase to a much smaller polygon (Douglas-Peucker via
    Shapely, applied in mm space so the tolerance means what it says
    regardless of the image's pixel density).
    """
    h = len(label_grid)
    w = len(label_grid[0]) if h else 0
    squares = [
        box(x, y, x + 1, y + 1)
        for y in range(h)
        for x in range(w)
        if label_grid[y][x] == label
    ]
    if not squares:
        return []

    merged = unary_union(squares)
    polys = list(merged.geoms) if hasattr(merged, "geoms") else [merged]

    ring_groups: list[RingGroup] = []
    for poly in polys:
        if poly.is_empty or poly.area < min_area_px:
            continue

        mm_exterior = [(px * px_to_mm_x, py * px_to_mm_y) for px, py in poly.exterior.coords]
        mm_holes = [
            [(px * px_to_mm_x, py * px_to_mm_y) for px, py in interior.coords]
            for interior in poly.interiors
        ]
        mm_poly = Polygon(shell=mm_exterior, holes=mm_holes)
        simplified = mm_poly.simplify(simplify_tolerance_mm, preserve_topology=True)

        if simplified.is_empty or simplified.area <= 0:
            continue
        if simplified.geom_type != "Polygon":
            # simplify() can rarely split degenerate input into a
            # MultiPolygon; keep only the largest piece rather than drop the
            # whole region.
            parts = [g for g in simplified.geoms if g.geom_type == "Polygon"]
            if not parts:
                continue
            simplified = max(parts, key=lambda p: p.area)

        outer = list(simplified.exterior.coords)
        holes = [list(interior.coords) for interior in simplified.interiors if len(interior.coords) >= 4]
        if len(outer) < 4:
            continue
        ring_groups.append([outer, *holes])

    return ring_groups
