"""
StitchPro internal embroidery data model (brief section 8).

This is the ONLY representation the rest of the backend (and eventually the
mobile app, via the API) should think in terms of. Nothing outside
`formats/dst/writer.py` should ever touch a pyembroidery EmbPattern or raw
DST bytes directly -- that conversion happens exactly once, at the very end
of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObjectType(str, Enum):
    RUNNING = "RUNNING"
    SATIN = "SATIN"
    FILL = "FILL"
    CROSS_STITCH = "CROSS_STITCH"
    BEAN = "BEAN"               # reinforced multi-pass running stitch, for
                                 # hairline-thin strokes (see stitches/bean)
    MOTIF = "MOTIF"              # repeating decorative shape tiled along a
                                 # path (see stitches/motif)
    PUFF_3D = "PUFF_3D"          # satin-style coverage stitched over a foam
                                 # pad for a raised look (see stitches/puff)
    APPLIQUE = "APPLIQUE"        # fabric patch sewn on rather than filled
                                 # with thread (see stitches/applique)
    # Reserved for later milestones (brief section 8): COLUMN, SPECIAL.



Point = tuple[float, float]  # (x_mm, y_mm), +y down, matches pyembroidery's convention


@dataclass
class StitchSettings:
    """Per-object stitch parameters. Which fields matter depends on `type`."""

    stitch_length_mm: float = 3.0       # RUNNING, and satin/fill row target length
    density_mm: float = 0.4             # SATIN: spacing between crossing stitches;
                                         # FILL: spacing between rows;
                                         # CROSS_STITCH: grid cell size (reused field,
                                         # same "spacing between stitch events" idea)
    angle_deg: float = 0.0              # SATIN/FILL/CROSS_STITCH stitch direction
    underlay: str | None = None         # None | "edge_run" | "center_run" | "zigzag"
    pull_compensation_mm: float = 0.0   # SATIN/FILL outward expansion
    border_width_mm: float = 0.0        # FILL/CROSS_STITCH: width of a satin
                                         # "edge walk" column traced around the
                                         # region's outer boundary and every hole,
                                         # stitched on top of the fill/cross-stitch
                                         # to hide its jagged scanline/grid edge
                                         # under a clean vector-traced outline (see
                                         # stitches/satin/border.py). 0 = no border
                                         # (back-compat default for callers that
                                         # built a FILL object before this existed).
                                         # PUFF_3D/APPLIQUE: width of their own
                                         # finishing satin edge/border (see
                                         # stitches/puff/generator.py and
                                         # stitches/applique/generator.py); 0
                                         # falls back to each generator's own
                                         # sensible built-in default rather than
                                         # disabling the border outright, since
                                         # (unlike FILL/CROSS_STITCH) neither of
                                         # those two ever looks right with no
                                         # finishing edge at all.
    bean_passes: int = 3                 # BEAN only: how many full forward/
                                         # backward traversals of the same path
                                         # ("triple stitch" = 3, the classic
                                         # default; must be >= 1).
    motif_pattern: str = "diamond"       # MOTIF only: which built-in repeating
                                         # shape to tile (see
                                         # stitches/motif/generator.py's
                                         # MOTIF_PATTERNS for the full list).
    motif_spacing_mm: float = 4.0        # MOTIF only: center-to-center distance
                                         # between repeated motifs along the path.
    motif_scale: float = 1.0             # MOTIF only: size multiplier for each
                                         # repeated motif shape relative to its
                                         # nominal (spacing-derived) size.


@dataclass
class EmbroideryObject:
    id: str
    type: ObjectType
    geometry: Any                       # shape depends on `type`: RUNNING is a flat
                                         # list[Point] polyline; SATIN is a flat list of
                                         # rail-A/rail-B rung pairs [A0,B0,A1,B1,...]; FILL
                                         # is a list of rings (geometry[0]=outer boundary,
                                         # geometry[1:]=holes), each ring a list[Point].
                                         # See stitches/*/generator.py for exact contracts.
    color: str                          # hex, e.g. "#111111"
    stitch_settings: StitchSettings = field(default_factory=StitchSettings)
    sequence: int = 0                   # stitch-out order; routing.py may reassign this
    layer: int = 0                      # hard stitch-order constraint: every object in a
                                         # lower layer stitches fully before any object in a
                                         # higher layer (e.g. underlay=0 before satin/fill=1
                                         # on top of it). routing.py never reorders across
                                         # layers, only within one.


@dataclass
class EmbroideryDesign:
    width_mm: float
    height_mm: float
    hoop: str = "100x100"               # named hoop preset, e.g. "100x100", "130x180"
    objects: list[EmbroideryObject] = field(default_factory=list)
    threads: list[str] = field(default_factory=list)   # unique hex colors, stitch order
    metadata: dict[str, Any] = field(default_factory=dict)

    def ordered_objects(self) -> list[EmbroideryObject]:
        return sorted(self.objects, key=lambda o: o.sequence)
