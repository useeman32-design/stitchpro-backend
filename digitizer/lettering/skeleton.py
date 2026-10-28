"""
Skeleton-stroke -> satin rail/rung geometry (dedicated embroidery lettering
engine, independent of the generic photo/artwork auto-digitizer).

Why this module exists
-----------------------
`digitizer.stitches.satin.thin_region` already turns a region into a satin
column automatically, but it deliberately bails out (falls back to FILL)
for anything curved or hole-bearing, because it has to *guess* a shape's
long axis from nothing but an arbitrary polygon (see that module's
docstring). Typed text is a completely different situation: we don't have
to guess anything, because we're the ones generating the letterforms in the
first place. Each glyph here is authored directly as one or more
"skeleton strokes" -- a sampled centerline path (straight, or a circular/
elliptical arc) plus a width -- and this module turns that skeleton
directly into exactly the rail-A/rail-B rung geometry
`digitizer.stitches.satin.generator.generate_satin_stitch` already knows
how to turn into real zig-zag satin stitches. This works uniformly for
straight AND curved strokes (letter bowls, "S" curves, the ring of an "O"),
which `thin_region`'s oriented-bounding-box approach cannot do -- there is
no scanline/global-axis guessing here at all, just "offset the known
centerline left/right by half the stroke width, following its local
tangent at every sample point".
"""

from __future__ import annotations

import math

Point = tuple[float, float]


def line_points(p0: Point, p1: Point, step: float = 0.6) -> list[Point]:
    """Evenly spaced points from p0 to p1 inclusive, roughly `step' apart
    (always includes both endpoints exactly)."""
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    n = max(1, math.ceil(length / step))
    return [
        (p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n)
        for i in range(n + 1)
    ]


def polyline_points(vertices: list[Point], step: float = 0.6) -> list[Point]:
    """Resample a multi-vertex polyline at roughly `step` spacing while
    always keeping every original vertex exactly (so sharp corners, e.g.
    the apex of 'A' or the zigzag of 'Z'/'M'/'W', are never smoothed away).
    """
    if len(vertices) < 2:
        raise ValueError("polyline_points needs at least 2 vertices")
    out: list[Point] = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        seg = line_points(a, b, step)
        out.extend(seg[1:])  # skip seg[0]: already the last point in `out`
    return out


def arc_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    start_deg: float,
    end_deg: float,
    step_deg: float = 8.0,
) -> list[Point]:
    """Sample an elliptical arc's centerline from start_deg to end_deg
    (end_deg - start_deg may be negative, or exceed 360 for a full ring --
    e.g. 'O' uses a 0..360 sweep to close into a loop). Angle convention:
    x = cx + rx*cos(theta), y = cy + ry*sin(theta), theta in degrees --
    matches the model's own (+y down) point convention directly, no extra
    flipping needed since every glyph here is authored in that same frame.
    """
    span = end_deg - start_deg
    n = max(2, int(abs(span) / step_deg) + 1)
    return [
        (
            cx + rx * math.cos(math.radians(start_deg + span * i / (n - 1))),
            cy + ry * math.sin(math.radians(start_deg + span * i / (n - 1))),
        )
        for i in range(n)
    ]


def stroke_to_satin_geometry(skeleton: list[Point], width_units: float) -> list[Point]:
    """Convert a sampled skeleton centerline + uniform width into satin
    rail/rung geometry: a flat [A0,B0,A1,B1,...] list, exactly the contract
    `generate_satin_stitch` expects (model.py: "SATIN is a flat list of
    rail-A/rail-B rung pairs").

    At every skeleton sample, the local tangent is estimated with a
    central difference (forward/backward difference at the two open ends),
    then rotated 90 degrees to get the local outward normal; the two rails
    are the skeleton point offset +/- half the width along that normal.
    Because the normal is derived from the *local* tangent at every point
    rather than one single global axis, this naturally follows curves
    (arcs sample finely enough that the tangent rotates smoothly) as well
    as straight runs, and handles a sharp interior vertex (e.g. a
    multi-segment polyline's corner) by averaging the incoming/outgoing
    tangent there -- a small, deliberate miter approximation that is what
    real simple block-lettering digitizing does at a stroke's corner too.
    """
    n = len(skeleton)
    if n < 2:
        raise ValueError("skeleton must have at least 2 points")
    half = width_units / 2.0
    geometry: list[Point] = []
    for i in range(n):
        if i == 0:
            tx, ty = skeleton[1][0] - skeleton[0][0], skeleton[1][1] - skeleton[0][1]
        elif i == n - 1:
            tx, ty = skeleton[-1][0] - skeleton[-2][0], skeleton[-1][1] - skeleton[-2][1]
        else:
            tx, ty = skeleton[i + 1][0] - skeleton[i - 1][0], skeleton[i + 1][1] - skeleton[i - 1][1]
        length = math.hypot(tx, ty)
        if length < 1e-9:
            nx, ny = 1.0, 0.0
        else:
            # rotate tangent -90 degrees: (tx,ty) -> (ty,-tx), normalized
            nx, ny = ty / length, -tx / length
        px, py = skeleton[i]
        geometry.append((px + nx * half, py + ny * half))
        geometry.append((px - nx * half, py - ny * half))
    return geometry
