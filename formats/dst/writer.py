"""
DST output layer -- the ONLY place in the backend allowed to import
pyembroidery / construct an EmbPattern / write binary embroidery bytes.

Per docs/THIRD_PARTY_LICENSES.md: pyembroidery is MIT licensed, used here
directly as a dependency (not modified/vendored). We do not reimplement the
DST binary format -- see docs/embroidery-engine-research.md section 1.
"""

from __future__ import annotations

from pyembroidery import (
    COLOR_CHANGE,
    END,
    JUMP,
    STITCH,
    STOP,
    EmbPattern,
    write_dst,
)

from digitizer.model import EmbroideryDesign
from digitizer.routing.router import route_objects
from digitizer.stitches.dispatch import generate_object_stitches
from formats.threads import hex_to_thread as _hex_to_thread

# generate_object_stitches' third tuple element is one of these three
# strings (see digitizer/stitches/dispatch.py) -- "stop" is a real machine
# pause (PUFF_3D/APPLIQUE only), encoded as pyembroidery's STOP command.
# At the raw DST byte level STOP and COLOR_CHANGE are the identical 3-byte
# command (see pyembroidery's own DstWriter) -- what makes ours read as "a
# stop" rather than a genuine thread change is simply that we never
# register a new thread for it below, exactly how every commercial
# digitizing program implements a mid-design pause in this format.
_CODE_BY_COMMAND = {"stitch": STITCH, "jump": JUMP, "stop": STOP}


def compile_design(design: EmbroideryDesign) -> EmbPattern:
    """Convert our internal EmbroideryDesign into a pyembroidery EmbPattern,
    in mm -> 0.1mm units, centered at the DST origin (0,0).

    Stitch order is decided by `route_objects` (Milestone 4), not by the
    design's raw object list order -- it enforces the layer constraint,
    groups same-color objects into contiguous runs to minimize color
    changes/trims, and applies a basic nearest-neighbor pass to cut down
    travel distance within each color group."""
    pattern = EmbPattern()

    offset_x = -design.width_mm / 2
    offset_y = -design.height_mm / 2

    objects = route_objects(design.objects)
    current_color: str | None = None

    for obj in objects:
        stitches = generate_object_stitches(obj)
        if not stitches:
            continue

        if obj.color != current_color:
            thread = _hex_to_thread(obj.color)
            pattern.add_thread(thread)
            if current_color is not None:
                pattern.color_change(0, 0)
            current_color = obj.color

        for x, y, command in stitches:
            code = _CODE_BY_COMMAND[command]
            pattern.add_stitch_absolute(code, (x + offset_x) * 10, (y + offset_y) * 10)

    pattern.add_command(END)
    return pattern


def design_to_dst_bytes(design: EmbroideryDesign) -> bytes:
    import io

    pattern = compile_design(design)
    buf = io.BytesIO()
    write_dst(pattern, buf)
    return buf.getvalue()


def write_design_dst(design: EmbroideryDesign, path: str) -> None:
    pattern = compile_design(design)
    write_dst(pattern, path)
