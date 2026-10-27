"""
Raw point-stream compiler -- a second, deliberately separate entry point
into pyembroidery alongside formats/dst/writer.py:compile_design.

Why this exists: the StitchPro mobile/web app already does REAL client-side
artwork analysis and stitch-plan math (canvas pixel scan -> per-color fill
runs -> a fully-formed, physical, 0.1mm-unit point stream with explicit
jump/stitch/colorChange commands -- see stitch-pro's src/utils/stitchEngine.ts
and src/utils/stitchPlanner.ts). That point stream is already a complete,
correct stitch plan; it doesn't need to go back through the
EmbroideryDesign -> ObjectType -> per-type stitch generator pipeline that
compile_design implements for brief-driven (future) server-side digitizing.

This module lets the API accept that already-computed point stream directly
and hand it to pyembroidery for real multi-format (DST/PES/JEF/VP3/EXP)
encoding -- the one piece the client genuinely cannot do itself, since
pyembroidery's binary format writers are Python-only. As more milestones
land server-side (satin, fill, real image-to-vector segmentation), new
designs should move to compile_design's EmbroideryDesign path instead; this
module is intentionally a thin, temporary bridge for the point-stream the
frontend already produces today.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from pyembroidery import COLOR_CHANGE, END, JUMP, STITCH, STOP, EmbPattern

from formats.threads import hex_to_thread

MAX_POINTS = 250_000  # sanity cap against pathological/malicious payloads

_COMMANDS = {
    "jump": JUMP,
    "stitch": STITCH,
    "colorChange": COLOR_CHANGE,
    # Real machine pause (PUFF_3D/APPLIQUE regions only -- see
    # digitizer/stitches/dispatch.py and formats/dst/writer.py's STOP
    # handling for why this is a real command, not simulation-only): a
    # server-digitized design's points stream (digitizer/points_export.py)
    # can include "stop" entries, and this is the module that re-compiles
    # that exact points stream into a real downloadable file (see
    # stitch-pro's downloadStitchFile.ts / backendClient.ts), so it must
    # understand the command too, or a Puff/Appliqu\u00e9 design would fail to
    # download with an "unknown command" error.
    "stop": STOP,
    "end": END,
}


def compile_points(points: Iterable[Mapping], colors: list[str] | None = None) -> EmbPattern:
    """Build a pyembroidery EmbPattern from an ordered list of
    {"x": int, "y": int, "command": "jump"|"stitch"|"colorChange"|"stop"|"end"}
    records, already in 0.1mm absolute units (screen-style, +Y down) -- the
    exact convention stitch-pro's own DstPoint/dstWriter.ts uses, and the
    same convention formats/dst/writer.py's compile_design feeds pyembroidery
    directly with no extra flip (verified by its own round-trip test).

    `colors` is an optional ordered list of hex thread colors, one per color
    block (colors[0] is used before the first colorChange, colors[1] after
    it, and so on) -- purely cosmetic metadata for color-aware formats like
    PES/JEF/VP3; DST itself does not store thread colors at all.
    """
    points = list(points)
    if not points:
        raise ValueError("points must not be empty")
    if len(points) > MAX_POINTS:
        raise ValueError(f"too many points ({len(points)} > {MAX_POINTS})")

    pattern = EmbPattern()
    for hex_color in colors or ["#1A1A1A"]:
        pattern.add_thread(hex_to_thread(hex_color))

    saw_end = False
    for i, p in enumerate(points):
        command = p.get("command")
        code = _COMMANDS.get(command)
        if code is None:
            raise ValueError(f"point {i}: unknown command {command!r}; expected one of {list(_COMMANDS)}")
        x, y = p.get("x"), p.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"point {i}: x/y must be numbers")
        pattern.add_stitch_absolute(code, x, y)
        if command == "end":
            saw_end = True

    if not saw_end:
        pattern.add_command(END)
    return pattern


def points_to_format_bytes(points: Iterable[Mapping], fmt_writer, colors: list[str] | None = None) -> bytes:
    import io

    pattern = compile_points(points, colors)
    buf = io.BytesIO()
    fmt_writer(pattern, buf)
    return buf.getvalue()
