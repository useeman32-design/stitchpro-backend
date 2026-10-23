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
from digitizer.stitches.cross.generator import generate_cross_stitch
from digitizer.stitches.fill.generator import generate_fill_stitch
from digitizer.stitches.running.generator import generate_running_stitch
from digitizer.stitches.satin.border import ring_to_satin_rungs
from digitizer.stitches.satin.generator import generate_satin_stitch
from digitizer.underlay.generator import generate_underlay_stitches

Stitch = tuple[float, float, bool]  # (x_mm, y_mm, is_jump)

# How finely a region's boundary ring is resampled before being handed to
# the satin generator as rung geometry -- see stitches/satin/border.py.
# Fine enough to follow real logo curvature, coarse enough that a typical
# region's border stays a small fraction of its total stitch count.
_BORDER_STEP_MM = 1.5


def _border_stitches(obj: EmbroideryObject) -> list[Stitch]:
    """Satin edge-walk border traced around `obj`'s outer boundary and every
    hole (see stitches/satin/border.py's module docstring for *why*), one
    independent closed satin loop per ring so holes get their own clean
    inner edge instead of being connected to the outer border by a visible
    long jump. No-op (returns []) when `border_width_mm` is 0, which is the
    default for any object built before this feature existed -- a purely
    additive change to FILL/CROSS_STITCH output otherwise identical to
    before.
    """
    border_mm = obj.stitch_settings.border_width_mm
    if border_mm <= 0:
        return []
    out: list[Stitch] = []
    for ring in obj.geometry:
        rungs = ring_to_satin_rungs(ring, border_mm, step_mm=_BORDER_STEP_MM)
        if len(rungs) < 4:
            continue
        points = generate_satin_stitch(rungs, density_mm=0.4)
        out.extend((x, y, i == 0) for i, (x, y) in enumerate(points))
    return out


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
        fill = generate_fill_stitch(
            obj.geometry,
            obj.stitch_settings.stitch_length_mm,
            obj.stitch_settings.density_mm,
            obj.stitch_settings.angle_deg,
            obj.stitch_settings.pull_compensation_mm,
        )
        return fill + _border_stitches(obj)
    if obj.type == ObjectType.CROSS_STITCH:
        # Same rings contract as FILL; `density_mm` is reused as the grid
        # cell size (see digitizer/model.py's StitchSettings docstring).
        cross = generate_cross_stitch(
            obj.geometry,
            obj.stitch_settings.density_mm,
            obj.stitch_settings.angle_deg,
            obj.stitch_settings.pull_compensation_mm,
        )
        return cross + _border_stitches(obj)
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
