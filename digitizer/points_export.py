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
    [{"x", "y", "command"}, ...]}` -- `threads[0]` is the color in effect
    before the first `colorChange` point, `threads[1]` after the first one,
    and so on, exactly matching `points_to_file`'s own `colors` contract.
    """
    offset_x = -design.width_mm / 2
    offset_y = -design.height_mm / 2

    objects = route_objects(design.objects)
    threads: list[str] = []
    points: list[dict] = []
    current_color: str | None = None

    for obj in objects:
        stitches = generate_object_stitches(obj)
        if not stitches:
            continue

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

    return {
        "width_mm": design.width_mm,
        "height_mm": design.height_mm,
        "threads": threads,
        "points": points,
    }
