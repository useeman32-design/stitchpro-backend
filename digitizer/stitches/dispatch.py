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
from digitizer.underlay.generator import generate_underlay_stitches

Stitch = tuple[float, float, bool]  # (x_mm, y_mm, is_jump)


def _top_stitches(obj: EmbroideryObject) -> list[Stitch]:
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
        # obj.geometry is a list of rings: geometry[0] is the outer
        # boundary, geometry[1:] are holes (see digitizer/model.py).
        return generate_fill_stitch(
            obj.geometry,
            obj.stitch_settings.stitch_length_mm,
            obj.stitch_settings.density_mm,
            obj.stitch_settings.angle_deg,
            obj.stitch_settings.pull_compensation_mm,
        )
    raise NotImplementedError(f"Stitch generation for {obj.type} not implemented yet")


def generate_object_stitches(obj: EmbroideryObject) -> list[Stitch]:
    """Dispatch geometry -> physical stitch coordinates by object type.

    Returns a list of (x_mm, y_mm, is_jump) triples: `is_jump=True` means
    "lift the needle and jump here, starting a new disconnected stitch
    run" -- always true for at least the very first point of every object.

    If `obj.stitch_settings.underlay` is set, the object's underlay
    stitches (digitizer/underlay/generator.py) are generated and prepended
    before its own top stitches -- underlay always stitches first, in the
    same thread, no color change needed.

    RUNNING (Milestone 1), SATIN (Milestone 2), and FILL (Milestone 3) top
    stitches, plus underlay (Milestone 6) for SATIN/FILL, are all
    implemented."""
    underlay = generate_underlay_stitches(obj)
    top = _top_stitches(obj)
    return underlay + top
