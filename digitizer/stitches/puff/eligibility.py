"""
3D Puff region-eligibility heuristic (new stitch type follow-up: Bean/Motif/
3D Puff/Applique).

Decides, per detected region, whether it's a good automatic candidate for
PUFF_3D treatment instead of ordinary FILL -- see
digitizer/stitches/puff/generator.py for what that treatment actually does.

Real 3D puff embroidery is used almost exclusively on bold, blocky shapes
(varsity lettering, badge/crest blobs) raised over a foam pad -- never on
large sprawling background fills (foam can't follow a complicated outline
well) and never on the thin stroke shapes BEAN/MOTIF/SATIN already claim
(see digitizer/stitches/satin/thin_region.py -- callers must check that
first and only fall through to this heuristic for regions it rejected).
This looks for exactly that "bold, convex-ish, moderate-size blob" shape
using only cheap Shapely geometry, deliberately conservative in the same
spirit as thin_region.py's own SATIN check: when in doubt, fall back to
ordinary FILL rather than guess wrong on a feature with real physical
(foam-placement) consequences.
"""

from __future__ import annotations

import shapely
from shapely.geometry import Polygon

Point = tuple[float, float]

# A puff candidate must fill at least this fraction of its own oriented
# bounding box -- bold/blocky shapes (letters, badge blobs) do; sprawling,
# spindly, or deeply concave shapes don't.
DEFAULT_MIN_BBOX_FILL_RATIO = 0.55

# Too small and a raised foam block looks like a fuzzy blob, not a crisp
# letter/badge.
DEFAULT_MIN_AREA_MM2 = 25.0

# Too large a fraction of the whole design's foreground and it's much more
# likely a background/body panel (where foam placement and trimming are
# impractical) than a deliberately puffed accent.
DEFAULT_MAX_DESIGN_AREA_FRACTION = 0.35


def is_puff_eligible(
    rings: list[list[Point]],
    design_foreground_area_mm2: float,
    min_bbox_fill_ratio: float = DEFAULT_MIN_BBOX_FILL_RATIO,
    min_area_mm2: float = DEFAULT_MIN_AREA_MM2,
    max_design_area_fraction: float = DEFAULT_MAX_DESIGN_AREA_FRACTION,
) -> bool:
    """`rings` should already have failed BEAN/MOTIF/SATIN eligibility --
    see module docstring."""
    if len(rings) != 1:
        return False  # a hole-bearing shape rarely reads well raised over foam -- v1 limitation
    polygon = Polygon(rings[0])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area < 1e-9:
        return False

    area_mm2 = polygon.area
    if area_mm2 < min_area_mm2:
        return False
    if design_foreground_area_mm2 > 0 and area_mm2 > design_foreground_area_mm2 * max_design_area_fraction:
        return False

    bbox = shapely.oriented_envelope(polygon)
    if bbox.is_empty or bbox.area < 1e-9:
        return False
    if (area_mm2 / bbox.area) < min_bbox_fill_ratio:
        return False
    return True
