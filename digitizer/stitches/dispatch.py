"""
Stitch generation dispatch -- the single place that maps an
EmbroideryObject's `type` to the right per-type stitch generator. Used by
both formats/dst/writer.py (to get real stitch coordinates to write out) and
digitizer/routing/router.py (to get an object's real physical start/end
points for travel-distance-based routing), so the two can never drift out of
sync about which generator handles which ObjectType.
"""

from __future__ import annotations

from digitizer.model import EmbroideryObject, ObjectType
from digitizer.stitches.fill.generator import generate_fill_stitch
from digitizer.stitches.running.generator import generate_running_stitch
from digitizer.stitches.satin.generator import generate_satin_stitch

Stitch = tuple[float, float, bool]  # (x_mm, y_mm, is_jump)


def generate_object_stitches(obj: EmbroideryObject) -> list[Stitch]:
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
