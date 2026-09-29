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
    EmbPattern,
    EmbThread,
    write_dst,
)

from digitizer.model import EmbroideryDesign
from digitizer.stitches.running.generator import generate_running_stitch


def _hex_to_thread(hex_color: str) -> EmbThread:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    thread = EmbThread()
    thread.set_color(r, g, b)
    return thread


def _object_stitch_points(obj) -> list[tuple[float, float]]:
    """Dispatch geometry -> physical stitch coordinates by object type.
    Only RUNNING is implemented as of Milestone 1; SATIN/FILL raise until
    their generators land (Milestones 2-3)."""
    from digitizer.model import ObjectType

    if obj.type == ObjectType.RUNNING:
        return generate_running_stitch(obj.geometry, obj.stitch_settings.stitch_length_mm)
    raise NotImplementedError(f"Stitch generation for {obj.type} not implemented yet")


def compile_design(design: EmbroideryDesign) -> EmbPattern:
    """Convert our internal EmbroideryDesign into a pyembroidery EmbPattern,
    in mm -> 0.1mm units, centered at the DST origin (0,0)."""
    pattern = EmbPattern()

    offset_x = -design.width_mm / 2
    offset_y = -design.height_mm / 2

    objects = design.ordered_objects()
    current_color: str | None = None

    for obj in objects:
        points = _object_stitch_points(obj)
        if not points:
            continue

        if obj.color != current_color:
            thread = _hex_to_thread(obj.color)
            pattern.add_thread(thread)
            if current_color is not None:
                pattern.color_change(0, 0)
            current_color = obj.color

        first_x, first_y = points[0]
        pattern.add_stitch_absolute(
            JUMP, (first_x + offset_x) * 10, (first_y + offset_y) * 10
        )
        for x, y in points:
            pattern.add_stitch_absolute(STITCH, (x + offset_x) * 10, (y + offset_y) * 10)

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
