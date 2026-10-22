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
    write_dst,
)

from digitizer.model import EmbroideryDesign
from digitizer.stitches.fill.generator import generate_fill_stitch
from digitizer.stitches.running.generator import generate_running_stitch
from digitizer.stitches.satin.generator import generate_satin_stitch
from formats.threads import hex_to_thread as _hex_to_thread


def _object_stitch_points(obj) -> list[tuple[float, float, bool]]:
    """Dispatch geometry -> physical stitch coordinates by object type.

    Returns a list of (x_mm, y_mm, is_jump) triples: `is_jump=True` means
    "lift the needle and jump here, starting a new disconnected stitch
    run" -- always true for at least the very first point of every object.
    RUNNING and SATIN each produce exactly one connected run (only their
    first point is a jump); FILL can produce several disconnected runs per
    object (concave polygons split a scan row into multiple spans) and
    marks each run's own first point as a jump individually.

    RUNNING (Milestone 1), SATIN (Milestone 2), and FILL (Milestone 3) are
    all implemented."""
    from digitizer.model import ObjectType

    if obj.type == ObjectType.RUNNING:
        points = generate_running_stitch(obj.geometry, obj.stitch_settings.stitch_length_mm)
        return [(x, y, i == 0) for i, (x, y) in enumerate(points)]
    if obj.type == ObjectType.SATIN:
        points = generate_satin_stitch(
            obj.geometry,
            obj.stitch_settings.density_mm,
            obj.stitch_settings.pull_compensation_mm,
        )
        return [(x, y, i == 0) for i, (x, y) in enumerate(points)]
    if obj.type == ObjectType.FILL:
        return generate_fill_stitch(
            obj.geometry,
            obj.stitch_settings.stitch_length_mm,
            obj.stitch_settings.density_mm,
            obj.stitch_settings.angle_deg,
            obj.stitch_settings.pull_compensation_mm,
        )
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
        stitches = _object_stitch_points(obj)
        if not stitches:
            continue

        if obj.color != current_color:
            thread = _hex_to_thread(obj.color)
            pattern.add_thread(thread)
            if current_color is not None:
                pattern.color_change(0, 0)
            current_color = obj.color

        for x, y, is_jump in stitches:
            code = JUMP if is_jump else STITCH
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
