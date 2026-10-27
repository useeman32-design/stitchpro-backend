"""
New stitch type follow-up (Bean/Motif/3D Puff/Applique) tests: MOTIF stitch
generator + DST round-trip.
"""

import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.motif.generator import MOTIF_PATTERNS, generate_motif_stitch
from formats.dst.writer import design_to_dst_bytes

LONG_LINE = [(0, 0), (40, 0)]


def test_produces_multiple_repeats_along_a_long_path():
    stitches = generate_motif_stitch(LONG_LINE, pattern="diamond", spacing_mm=4.0)
    assert len(stitches) > 20  # several repeats, each several points


def test_unknown_pattern_name_falls_back_gracefully():
    stitches = generate_motif_stitch(LONG_LINE, pattern="not-a-real-pattern", spacing_mm=4.0)
    assert stitches  # doesn't raise, still produces something


def test_every_built_in_pattern_is_a_closed_shape():
    for name, points in MOTIF_PATTERNS.items():
        assert points[0] == points[-1], f"{name} is not closed"
        assert len(points) >= 4, f"{name} has too few points"


def test_zero_or_negative_spacing_raises():
    try:
        generate_motif_stitch(LONG_LINE, spacing_mm=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_no_single_stitch_exceeds_dst_max_length_regardless_of_spacing():
    # A deliberately large spacing must still be safely subdivided into
    # DST-legal (<=12.1mm) stitches by the inter-motif connector, not
    # emitted as one long illegal stitch.
    stitches = generate_motif_stitch(LONG_LINE, pattern="leaf", spacing_mm=25.0, stitch_length_mm=3.0)
    for (x0, y0, _), (x1, y1, _) in zip(stitches, stitches[1:]):
        assert math.hypot(x1 - x0, y1 - y0) <= 12.1 + 1e-6


def test_short_geometry_degenerates_without_crashing():
    assert generate_motif_stitch([(0, 0)]) == [(0, 0, True)]


def test_motif_object_round_trips_through_real_dst():
    obj = EmbroideryObject(
        id="motif-1",
        type=ObjectType.MOTIF,
        geometry=LONG_LINE,
        color="#2E8B57",
        stitch_settings=StitchSettings(motif_pattern="scallop", motif_spacing_mm=5.0),
    )
    design = EmbroideryDesign(width_mm=60, height_mm=20, objects=[obj], threads=["#2E8B57"])
    dst_bytes = design_to_dst_bytes(design)
    pattern = read_embroidery(io.BytesIO(dst_bytes))
    assert len(pattern.stitches) > 0
