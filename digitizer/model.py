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
    # Reserved for later milestones (brief section 8): COLUMN, APPLIQUE, SPECIAL.


Point = tuple[float, float]  # (x_mm, y_mm), +y down, matches pyembroidery's convention


@dataclass
class StitchSettings:
    """Per-object stitch parameters. Which fields matter depends on `type`."""

    stitch_length_mm: float = 3.0       # RUNNING, and satin/fill row target length
    density_mm: float = 0.4             # SATIN: spacing between crossing stitches;
                                         # FILL: spacing between rows
    angle_deg: float = 0.0              # SATIN/FILL stitch direction
    underlay: str | None = None         # None | "edge_run" | "center_run" | "zigzag"
    pull_compensation_mm: float = 0.0   # SATIN/FILL outward expansion


@dataclass
class EmbroideryObject:
    id: str
    type: ObjectType
    geometry: list[Point]               # meaning depends on `type` (see stitches/*)
    color: str                          # hex, e.g. "#111111"
    stitch_settings: StitchSettings = field(default_factory=StitchSettings)
    sequence: int = 0                   # stitch-out order; routing.py may reassign this


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
