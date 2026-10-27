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
from digitizer.stitches.applique.generator import generate_applique_stitches
from digitizer.stitches.bean.generator import generate_bean_stitch
from digitizer.stitches.cross.generator import generate_cross_stitch
from digitizer.stitches.fill.generator import generate_fill_stitch
from digitizer.stitches.motif.generator import generate_motif_stitch
from digitizer.stitches.puff.generator import generate_puff_stitches
from digitizer.stitches.running.generator import generate_running_stitch
from digitizer.stitches.satin.border import rings_to_border_satin_stitches
from digitizer.stitches.satin.generator import generate_satin_stitch
from digitizer.underlay.generator import generate_underlay_stitches

Stitch = tuple[float, float, bool]  # (x_mm, y_mm, is_jump) -- internal, per-type-generator contract
StitchEvent = tuple[float, float, str]  # (x_mm, y_mm, "stitch" | "jump" | "stop") -- generate_object_stitches' return contract

# Composite types that need to express a real mid-object pause (a STOP, for
# a human operator to place foam/fabric) which a plain bool `is_jump` flag
# cannot represent -- these bypass the ordinary underlay+top-stitch
# assembly below entirely and build their own full StitchEvent sequence,
# STOP included (see their own generator modules).
_COMPOSITE_GENERATORS = {
    ObjectType.PUFF_3D: generate_puff_stitches,
    ObjectType.APPLIQUE: generate_applique_stitches,
}

# How finely a region's boundary ring is resampled before being handed to
# the satin generator as rung geometry -- see stitches/satin/border.py.
# Fine enough to follow real logo curvature, coarse enough that a typical
# region's border stays a small fraction of its total stitch count.
_BORDER_STEP_MM = 1.5


def _to_events(stitches: list[Stitch]) -> list[StitchEvent]:
    return [(x, y, "jump" if is_jump else "stitch") for x, y, is_jump in stitches]


def _border_stitches(obj: EmbroideryObject) -> list[Stitch]:
    """Satin edge-walk border traced around `obj`'s outer boundary and every
    hole (see stitches/satin/border.py's module docstring for *why*). No-op
    (returns []) when `border_width_mm` is 0, which is the default for any
    object built before this feature existed -- a purely additive change to
    FILL/CROSS_STITCH output otherwise identical to before.
    """
    return rings_to_border_satin_stitches(obj.geometry, obj.stitch_settings.border_width_mm, step_mm=_BORDER_STEP_MM)


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
    if obj.type == ObjectType.BEAN:
        # obj.geometry is an open polyline path (see digitizer/model.py and
        # digitizer.pipeline's BEAN eligibility heuristic for how a thin
        # region's centerline becomes this path).
        return generate_bean_stitch(
            obj.geometry,
            obj.stitch_settings.stitch_length_mm,
            obj.stitch_settings.bean_passes,
        )
    if obj.type == ObjectType.MOTIF:
        # Same open-polyline contract as BEAN.
        return generate_motif_stitch(
            obj.geometry,
            obj.stitch_settings.motif_pattern,
            obj.stitch_settings.motif_spacing_mm,
            obj.stitch_settings.motif_scale,
            obj.stitch_settings.stitch_length_mm,
        )
    raise NotImplementedError(f"Stitch generation for {obj.type} not implemented yet")


def generate_object_stitches(obj: EmbroideryObject) -> list[StitchEvent]:
    """Dispatch geometry -> physical stitch coordinates by object type.

    Returns a list of (x_mm, y_mm, command) triples, `command` one of
    `"stitch"`, `"jump"` (lift the needle and jump here, starting a new
    disconnected stitch run -- always true for at least the very first
    point of every object), or `"stop"` (PUFF_3D/APPLIQUE only: a real
    machine pause for an operator to place foam/fabric -- see
    stitches/puff/generator.py's module docstring for why this is a real,
    not simulated, machine command).

    PUFF_3D and APPLIQUE build their own complete StitchEvent sequence
    (including any STOPs) via `_COMPOSITE_GENERATORS` and bypass the
    underlay+top-stitch assembly below entirely -- both already generate
    their own placement/tackdown passes in place of a generic underlay.

    For every other type: if `obj.stitch_settings.underlay` is set, the
    object's underlay stitches (digitizer/underlay/generator.py) are
    generated and prepended before its own top stitches -- underlay always
    stitches first, in the same thread, no color change needed.
    """
    composite = _COMPOSITE_GENERATORS.get(obj.type)
    if composite is not None:
        return composite(obj)
    underlay = generate_underlay_stitches(obj)
    top = _top_stitches(obj)
    return _to_events(underlay + top)
