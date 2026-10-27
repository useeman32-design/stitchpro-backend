"""
EmbroideryDesign -> raw point-stream JSON (mirrors formats/dst/writer.py's
compile_design, but for the app's point-stream contract instead of a binary
pyembroidery EmbPattern).

Why this exists: stitch-pro's frontend already speaks a `DstPoint`
contract -- `{x, y, command}` in 0.1mm absolute units, screen-style (+Y
down) -- for its own client-side-generated stitch plans (see
formats/points_writer.py's docstring, and stitch-pro's own
src/utils/dstWriter.ts). This module produces the exact same shape, but
from a *server-digitized* EmbroideryDesign (Milestone 5's image_to_design,
or any future server-side digitizing path), so the frontend's existing
point-stream renderers/converters work completely unmodified regardless of
whether the plan came from the browser's own canvas analysis or from a real
backend digitize call -- it's the same wire format either way.

This is the one new thing this module adds beyond compile_design: a
`colorChange` record is emitted as its own point (at the needle's current
position, matching how pyembroidery/compile_design already calls
`pattern.color_change(0, 0)`), so a client can replay the stream and know
exactly when to switch which palette color it's drawing with, without
needing to understand `EmbroideryObject`/`EmbroideryDesign` at all.
"""

from __future__ import annotations

from digitizer.model import EmbroideryDesign
from digitizer.routing.router import route_objects
from digitizer.stitches.dispatch import generate_object_stitches

MAX_EXPORT_POINTS = 250_000  # matches formats/points_writer.py's own sanity cap


def design_to_points(design: EmbroideryDesign) -> dict:
    """Returns `{"width_mm", "height_mm", "threads": [hex, ...], "points":
    [{"x", "y", "command"}, ...], "stitchTypeBreakdown": [...]}` --
    `threads[0]` is the color in effect before the first `colorChange`
    point, `threads[1]` after the first one, and so on, exactly matching
    `points_to_file`'s own `colors` contract.

    `stitchTypeBreakdown` is purely informational metadata, *not* part of
    the `points` stream itself -- deliberately kept out of the point/command
    contract that `dstPoints` ultimately feeds into real DST/DSB file
    export (see stitch-pro's downloadStitchFile), so a client that doesn't
    know about it can safely ignore it with zero risk to the actual
    machine-readable output. It answers "which stitch types does this
    design actually use, and how much of it is each" -- e.g. a design is
    rarely pure FILL or pure SATIN; the auto-digitizer already classifies
    each region independently (thin_region.py), splitting a single color's
    regions across FILL/SATIN/CROSS_STITCH as appropriate per-shape, so a
    client wanting to show the user "this design automatically uses N
    stitch types across its regions" needs a different signal than the
    flat point stream to do it. Returned as a list (not a dict) ordered by
    descending stitch count, so a client can render "SATIN · 4 regions"
    chips in a sensible dominant-type-first order without re-sorting.
    """
    offset_x = -design.width_mm / 2
    offset_y = -design.height_mm / 2

    objects = route_objects(design.objects)
    threads: list[str] = []
    points: list[dict] = []
    current_color: str | None = None
    type_counts: dict[str, dict[str, int]] = {}

    for obj in objects:
        stitches = generate_object_stitches(obj)
        if not stitches:
            continue

        type_key = obj.type.value
        bucket = type_counts.setdefault(type_key, {"objectCount": 0, "stitchCount": 0})
        bucket["objectCount"] += 1
        bucket["stitchCount"] += sum(1 for *_xy, is_jump in stitches if not is_jump)

        if obj.color != current_color:
            threads.append(obj.color)
            if current_color is not None:
                last = points[-1] if points else None
                cx = last["x"] if last else 0
                cy = last["y"] if last else 0
                points.append({"x": cx, "y": cy, "command": "colorChange"})
            current_color = obj.color

        for x, y, is_jump in stitches:
            points.append(
                {
                    "x": round((x + offset_x) * 10),
                    "y": round((y + offset_y) * 10),
                    "command": "jump" if is_jump else "stitch",
                }
            )
            if len(points) > MAX_EXPORT_POINTS:
                raise ValueError(f"design too large to export as points (> {MAX_EXPORT_POINTS})")

    if not points:
        raise ValueError("design has no stitches to export")

    stitch_type_breakdown = [
        {"type": type_key, "objectCount": counts["objectCount"], "stitchCount": counts["stitchCount"]}
        for type_key, counts in sorted(type_counts.items(), key=lambda kv: -kv[1]["stitchCount"])
    ]

    return {
        "width_mm": design.width_mm,
        "height_mm": design.height_mm,
        "threads": threads,
        "points": points,
        "stitchTypeBreakdown": stitch_type_breakdown,
    }
