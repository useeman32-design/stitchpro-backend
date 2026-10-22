"""
Basic stitch-order routing (brief section 12).

Three real constraints drive stitch order here, matching the routing
concepts noted (independently, no copied code) in
research/inkstitch-notes.md:

1. Layering: some objects must be fully stitched before others (e.g. an
   underlay pass before the satin/fill stitched on top of it). This is a
   hard constraint, expressed as `EmbroideryObject.layer` -- every object in
   a lower layer number is stitched before any object in a higher layer
   number, no exceptions.
2. Minimizing color changes (and so machine trims): within one layer, every
   object of the same color is grouped into one contiguous run instead of
   being interleaved with other colors, even if the caller's objects list
   happened to interleave them.
3. Minimizing travel between objects: within one color group, a greedy
   nearest-neighbor heuristic orders objects so each new object starts as
   close as possible to wherever the needle already is, cutting down total
   jump distance. This is deliberately simple (not a full TSP solve) --
   "basic routing" is what this milestone asks for; a better solver can
   replace just this one function later without changing its interface.

The caller's `EmbroideryObject.sequence` is used only as the starting order
and as a tie-break (so if objects were already given in a sensible order,
routing is close to a no-op); layering and color grouping always win over
it, and nearest-neighbor only ever reorders *within* one layer+color group.

Note: this computes each object's real stitches (via
`generate_object_stitches`) purely to find its physical start/end point for
distance comparisons -- the same stitches get computed again later when the
design is actually compiled to a format. That's a deliberate simplicity
trade-off for this "basic routing" milestone (correctness over caching);
revisit if routing large designs becomes a measured performance problem.
"""

from __future__ import annotations

import math

from digitizer.model import EmbroideryObject
from digitizer.stitches.dispatch import generate_object_stitches

Point = tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _object_endpoints(obj: EmbroideryObject) -> tuple[Point, Point]:
    """Real physical (start, end) points, from actually generating the
    object's stitches -- not approximated from raw geometry -- so routing
    decisions are based on where the needle truly starts and finishes."""
    stitches = generate_object_stitches(obj)
    if not stitches:
        return (0.0, 0.0), (0.0, 0.0)
    start = (stitches[0][0], stitches[0][1])
    end = (stitches[-1][0], stitches[-1][1])
    return start, end


def route_objects(objects: list[EmbroideryObject]) -> list[EmbroideryObject]:
    """Return `objects` reordered for stitch-out: grouped by layer
    (ascending, hard constraint), then by color (contiguous runs,
    first-appearance order preserved within each layer), then by greedy
    nearest-neighbor travel distance within each color group.

    Also reassigns each object's `.sequence` in place to match its final
    position, so `EmbroideryDesign.ordered_objects()` reflects the routed
    order too if the caller stores this result back onto the design."""
    if not objects:
        return []

    baseline = sorted(objects, key=lambda o: o.sequence)

    layers: dict[int, list[EmbroideryObject]] = {}
    for obj in baseline:
        layers.setdefault(obj.layer, []).append(obj)

    endpoints = {id(obj): _object_endpoints(obj) for obj in baseline}

    current_pos: Point = (0.0, 0.0)
    routed: list[EmbroideryObject] = []

    for layer_key in sorted(layers.keys()):
        layer_objects = layers[layer_key]

        # Group by color, preserving first-appearance order within this layer
        # -- this is what merges interleaved same-color objects into one
        # contiguous run without needing the caller to have pre-sorted them.
        color_order: list[str] = []
        color_groups: dict[str, list[EmbroideryObject]] = {}
        for obj in layer_objects:
            if obj.color not in color_groups:
                color_groups[obj.color] = []
                color_order.append(obj.color)
            color_groups[obj.color].append(obj)

        for color in color_order:
            remaining = list(color_groups[color])
            while remaining:
                best_idx = 0
                best_dist = float("inf")
                for i, obj in enumerate(remaining):
                    start, _ = endpoints[id(obj)]
                    d = _dist(current_pos, start)
                    if d < best_dist - 1e-9:
                        best_dist = d
                        best_idx = i
                chosen = remaining.pop(best_idx)
                routed.append(chosen)
                _, end = endpoints[id(chosen)]
                current_pos = end

    for i, obj in enumerate(routed):
        obj.sequence = i
    return routed
