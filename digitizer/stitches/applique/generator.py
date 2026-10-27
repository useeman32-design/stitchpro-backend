"""
Appliqu\u00e9 stitch generator (new stitch type follow-up: Bean/Motif/3D Puff/Applique).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md).

Real appliqu\u00e9 embroidery sews a separate fabric patch onto the garment
instead of filling the shape with thread -- the machine's job is only to
mark where the patch goes, hold it down, and finish its raw edge; cutting
and placing the actual fabric is a human step the machine pauses for (see
`digitizer.pipeline`'s APPLIQUE eligibility heuristic for which single,
dominant, simple region in a design gets this treatment). This generator
produces the real four-phase sequence commercial digitizing software uses,
including real STOP commands (confirmed with the user, not a simulation-
only effect -- see stitches/puff/generator.py's module docstring for the
DST-level detail on why a STOP and a COLOR_CHANGE are byte-identical):

  1. Placement running stitch: traces the region's outer boundary directly
     on the garment, marking exactly where to lay the fabric patch.
  2. STOP #1: machine pauses so the operator can position the fabric patch
     over the placement line.
  3. Tackdown running stitch: re-traces the same boundary, now stitching
     through the new fabric too, securing it in place.
  4. STOP #2: machine pauses again so the operator can trim the excess
     fabric back close to the tackdown line -- this step only makes sense
     once the fabric is actually tacked down (trimming before tackdown
     would let the patch shift out of place).
  5. Finishing satin border: a satin column traced over the trimmed edge
     (`rings_to_border_satin_stitches`, the same edge-quality technique as
     FILL/CROSS_STITCH's own border and 3D Puff's edge cap, just wider --
     this is the stitch that actually covers the raw cut edge), covering
     every hole's edge too in case the patch shape has one.
"""

from __future__ import annotations

from digitizer.model import EmbroideryObject
from digitizer.stitches.running.generator import generate_running_stitch
from digitizer.stitches.satin.border import rings_to_border_satin_stitches

Point = tuple[float, float]
StitchEvent = tuple[float, float, str]  # (x_mm, y_mm, "stitch" | "jump" | "stop")

_PLACEMENT_STITCH_LENGTH_MM = 3.0
_TACKDOWN_STITCH_LENGTH_MM = 2.5
# Used only when the object wasn't given its own border_width_mm -- a real
# appliqu\u00e9 finishing border runs noticeably wider than FILL's thin edge-walk
# or Puff's thin edge cap, since it has to fully bury a real cut fabric edge.
_DEFAULT_BORDER_WIDTH_MM = 2.0
_BORDER_DENSITY_MM = 0.35


def _run_to_events(points: list[Point], mark_first_as_jump: bool) -> list[StitchEvent]:
    return [
        (x, y, "jump" if (mark_first_as_jump and i == 0) else "stitch") for i, (x, y) in enumerate(points)
    ]


def _bool_to_events(stitches: list[tuple[float, float, bool]]) -> list[StitchEvent]:
    return [(x, y, "jump" if is_jump else "stitch") for x, y, is_jump in stitches]


def generate_applique_stitches(obj: EmbroideryObject) -> list[StitchEvent]:
    """Full placement -> STOP -> tackdown -> STOP -> satin-border sequence
    for an APPLIQUE object. `obj.geometry` uses the same rings contract as
    FILL (`geometry[0]` = outer boundary, `geometry[1:]` = holes).
    """
    rings: list[list[Point]] = obj.geometry
    if not rings or len(rings[0]) < 3:
        return []
    outer = rings[0]
    settings = obj.stitch_settings

    out: list[StitchEvent] = []

    placement = generate_running_stitch(outer, _PLACEMENT_STITCH_LENGTH_MM)
    out.extend(_run_to_events(placement, mark_first_as_jump=True))
    if placement:
        out.append((placement[-1][0], placement[-1][1], "stop"))

    tackdown = generate_running_stitch(outer, _TACKDOWN_STITCH_LENGTH_MM)
    # Continues directly from the placement STOP's position -- the needle
    # is already sitting right there, so this is never a jump even though
    # it's logically a new pass.
    out.extend(_run_to_events(tackdown, mark_first_as_jump=False))
    if tackdown:
        out.append((tackdown[-1][0], tackdown[-1][1], "stop"))

    border_mm = settings.border_width_mm if settings.border_width_mm > 0 else _DEFAULT_BORDER_WIDTH_MM
    border = rings_to_border_satin_stitches(rings, border_mm, step_mm=1.5, density_mm=_BORDER_DENSITY_MM)
    out.extend(_bool_to_events(border))

    return out
