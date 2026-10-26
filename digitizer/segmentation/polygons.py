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

from digitizer.geometry.smoothing import DEFAULT_SHARP_ANGLE_DEG, smooth_ring_group

Point = tuple[float, float]
RingGroup = list[list[Point]]  # [outer_ring, hole_ring, hole_ring, ...]


def label_grid_to_ring_groups(
    label_grid: list[list[int]],
    label: int,
    px_to_mm_x: float,
    px_to_mm_y: float,
    min_area_px: float = 4.0,
    simplify_tolerance_mm: float | None = None,
    smooth_iterations: int = 3,
    smooth_sharp_angle_deg: float = DEFAULT_SHARP_ANGLE_DEG,
    min_bbox_mm: float = 0.6,
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

    Leaving `simplify_tolerance_mm` as `None` (the default) picks
    `max(0.25, 2.0 * max(px_to_mm_x, px_to_mm_y))` -- i.e. at least ~2 source
    pixels' worth of simplification -- rather than a fixed 0.25mm. A fixed
    absolute tolerance was the real bug behind "curves still look rough"
    feedback: at the pipeline's actual working resolution (~150px across a
    real 50-90mm design, so roughly 0.3-0.6mm/pixel), 0.25mm is *sub-pixel*,
    so Douglas-Peucker could barely touch the raw unit-square staircase --
    confirmed by instrumenting a real rasterized circle, where 96% of its
    "simplified" vertices still had pixel-staircase turn angles (45,
    63.4, 90 degrees -- the exact signature of an unreduced Bresenham-style
    staircase) above `smooth_sharp_angle_deg`'s 35-degree cutoff. Since the
    angle-adaptive smoothing below treats anything above that cutoff as an
    intentional sharp design corner and never touches it, the staircase
    passed through essentially untouched regardless of `smooth_iterations`
    -- confirmed by the same instrumentation showing smoothing had *zero*
    effect on point count or roundness at the old default. Scaling the
    tolerance to the source resolution reliably collapses the staircase
    into a true gentle-curve polygon first, so the smoothing pass actually
    has a controllable number of gentle-turn vertices to work with, and
    this was verified to hold (round-trip tested on real rasterized
    circles) across design sizes from 20mm to 90mm at the pipeline's
    standard ~150px working resolution.

    `smooth_iterations` then runs adaptive, corner-preserving curve
    smoothing (digitizer/geometry/smoothing.py) on top of that simplified
    polygon -- genuinely curved edges (a circular badge, a rounded
    letterform) come out smooth instead of faceted, while intentionally
    sharp corners are left untouched. Pass `smooth_iterations=0` to skip
    this step entirely. Raised from 2 to 3 (the same real-circle
    instrumentation showed 3 iterations reliably beats 2 once the
    sharp-angle misclassification above is fixed; beyond 3 the curve has
    already converged and additional iterations only add redundant points
    along the same shape, so 3 is the efficient stopping point, not an
    arbitrary bump).

    `min_bbox_mm` is a second, physical-unit noise filter on top of
    `min_area_px`: a region whose *shorter* bounding-box side is below this
    many millimeters is discarded regardless of its pixel-space area. This
    specifically catches thin, elongated one-pixel-wide slivers along an
    anti-aliased curve edge -- a handful of boundary pixels that end up in
    their own quantized color bucket (a blend tone between the design and
    its background) can have *enough* pixel area to survive `min_area_px`
    while still being far too thin a sliver to be a real, intentional stitch
    detail. A real embroidery machine has a practical minimum stitch length
    around 1mm anyway, so a sub-millimeter region was never going to be
    meaningfully stitched regardless.
    """
    effective_tolerance_mm = (
        simplify_tolerance_mm
        if simplify_tolerance_mm is not None
        else max(0.25, 2.0 * max(px_to_mm_x, px_to_mm_y))
    )

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
        simplified = mm_poly.simplify(effective_tolerance_mm, preserve_topology=True)

        if simplified.is_empty or simplified.area <= 0:
            continue
        sx0, sy0, sx1, sy1 = simplified.bounds
        # Use the *shorter* bounding-box side, not the longer one: a thin
        # anti-aliased-edge sliver can be several mm long in one direction
        # while only a fraction of a mm wide in the other, which is exactly
        # the shape `min_bbox_mm` needs to catch -- using the longer side
        # here would let a long, hair-thin streak slip through untouched.
        if min(sx1 - sx0, sy1 - sy0) < min_bbox_mm:
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

        group = [outer, *holes]
        if smooth_iterations > 0:
            group = smooth_ring_group(group, smooth_iterations, smooth_sharp_angle_deg)
        ring_groups.append(group)

    return ring_groups
