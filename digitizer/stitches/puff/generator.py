"""
3D Puff stitch generator (new stitch type follow-up: Bean/Motif/3D Puff/Applique).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md).

Real "3D puff" embroidery raises a block of satin-style stitching off the
fabric by stitching it over a foam pad instead of directly on the garment --
almost always used for bold lettering/badges (varsity letters, cap logos),
never fine detail (see `digitizer.pipeline`'s PUFF_3D eligibility
heuristic). Placing and trimming the foam is a real physical step a machine
operator does mid-run, not something the embroidery machine itself can do,
so this generator produces the actual phase sequence a real puff file
needs:

  1. Placement running stitch: traces the region's outer boundary directly
     on the garment, marking exactly where to lay the foam patch.
  2. A real STOP command (confirmed with the user, not a simulation-only
     effect): the machine pauses and beeps here so the operator can lay a
     foam patch over the placement line and resume. At the DST byte level
     a STOP and a COLOR_CHANGE are literally the identical 3-byte command
     (see formats/dst/writer.py) -- what makes it read as "a stop" rather
     than a genuine thread change is simply that no new thread is
     registered for it, exactly how every commercial digitizing program
     implements a mid-design pause in this format.
  3. Two full-coverage tatami fill passes over the whole shape, offset 90
     degrees from each other (reusing the existing `generate_fill_stitch`,
     not a literal rail/rung satin column -- see scope note below), so the
     dense stitching hides the raised, slightly uneven foam surface from
     every angle instead of leaving thin gaps a single-direction fill can
     leave over it -- a standard real cross-fill puff coverage technique.
  4. A thin satin edge cap (`rings_to_border_satin_stitches`, the same
     edge-quality technique already used for FILL/CROSS_STITCH borders)
     tracing the real vector boundary, so the raised block's silhouette
     reads as one clean edge instead of the fill's jagged row-ends.

Scope note: a genuine satin-rail puff column (matching a thin bold letter
stroke's actual medial axis) is out of scope for this pass -- the same
straight/hole-free/thin-and-elongated constraints
`digitizer.stitches.satin.thin_region` documents for automatic SATIN
classification apply here too, and most real puff candidate shapes
(rounded bubble letters, badge blobs) don't satisfy them. The double-pass
cross-fill technique above is a legitimate, widely used alternative that
gives correct full foam coverage for *any* region shape, not just thin
straight ones.
"""

from __future__ import annotations

from digitizer.model import EmbroideryObject
from digitizer.stitches.fill.generator import generate_fill_stitch
from digitizer.stitches.running.generator import generate_running_stitch
from digitizer.stitches.satin.border import rings_to_border_satin_stitches

Point = tuple[float, float]
StitchEvent = tuple[float, float, str]  # (x_mm, y_mm, "stitch" | "jump" | "stop")

_PLACEMENT_STITCH_LENGTH_MM = 3.0
_COVERAGE_STITCH_LENGTH_MM = 2.5
# Puff coverage must never be sparser than this, regardless of the object's
# own `density_mm` -- too sparse and the foam shows through between rows.
_MIN_COVERAGE_DENSITY_MM = 0.35
_EDGE_CAP_WIDTH_MM = 1.0
_EDGE_CAP_DENSITY_MM = 0.4


def _run_to_events(points: list[Point]) -> list[StitchEvent]:
    return [(x, y, "jump" if i == 0 else "stitch") for i, (x, y) in enumerate(points)]


def _bool_to_events(stitches: list[tuple[float, float, bool]]) -> list[StitchEvent]:
    return [(x, y, "jump" if is_jump else "stitch") for x, y, is_jump in stitches]


def generate_puff_stitches(obj: EmbroideryObject) -> list[StitchEvent]:
    """Full placement -> STOP -> cross-fill coverage -> edge cap sequence
    for a PUFF_3D object. `obj.geometry` uses the same rings contract as
    FILL (`geometry[0]` = outer boundary, `geometry[1:]` = holes).
    """
    rings: list[list[Point]] = obj.geometry
    if not rings or len(rings[0]) < 3:
        return []
    settings = obj.stitch_settings

    out: list[StitchEvent] = []

    placement = generate_running_stitch(rings[0], _PLACEMENT_STITCH_LENGTH_MM)
    out.extend(_run_to_events(placement))
    if placement:
        out.append((placement[-1][0], placement[-1][1], "stop"))

    coverage_density = (
        min(settings.density_mm, _MIN_COVERAGE_DENSITY_MM)
        if settings.density_mm > 0
        else _MIN_COVERAGE_DENSITY_MM
    )
    pass_a = generate_fill_stitch(
        rings, _COVERAGE_STITCH_LENGTH_MM, coverage_density, settings.angle_deg, settings.pull_compensation_mm
    )
    pass_b = generate_fill_stitch(
        rings,
        _COVERAGE_STITCH_LENGTH_MM,
        coverage_density,
        settings.angle_deg + 90,
        settings.pull_compensation_mm,
    )
    out.extend(_bool_to_events(pass_a))
    out.extend(_bool_to_events(pass_b))

    border_mm = settings.border_width_mm if settings.border_width_mm > 0 else _EDGE_CAP_WIDTH_MM
    edge_cap = rings_to_border_satin_stitches(rings, border_mm, step_mm=1.5, density_mm=_EDGE_CAP_DENSITY_MM)
    out.extend(_bool_to_events(edge_cap))

    return out
