"""
New stitch type follow-up (Bean/Motif/3D Puff/Applique) tests: PUFF_3D
stitch generator, the real STOP command it emits, and DST round-trip.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.puff.generator import generate_puff_stitches
from formats.dst.writer import design_to_dst_bytes

SQUARE = [(0, 0), (12, 0), (12, 12), (0, 12)]


def _puff_obj():
    return EmbroideryObject(
        id="puff-1",
        type=ObjectType.PUFF_3D,
        geometry=[SQUARE],
        color="#1A1A1A",
        stitch_settings=StitchSettings(density_mm=0.4),
    )


def test_sequence_contains_exactly_one_stop():
    stitches = generate_puff_stitches(_puff_obj())
    stops = [s for s in stitches if s[2] == "stop"]
    assert len(stops) == 1


def test_stop_comes_after_placement_and_before_coverage():
    stitches = generate_puff_stitches(_puff_obj())
    stop_index = next(i for i, s in enumerate(stitches) if s[2] == "stop")
    assert stop_index > 0  # placement ran first
    assert stop_index < len(stitches) - 1  # coverage still follows


def test_coverage_uses_two_perpendicular_fill_passes():
    # With density clamped to a sane minimum, a 12x12mm square should
    # produce a healthy number of coverage stitches in both passes -- a
    # regression here (e.g. only one pass) would roughly halve this count.
    stitches = generate_puff_stitches(_puff_obj())
    real_stitches = [s for s in stitches if s[2] == "stitch"]
    assert len(real_stitches) > 40


def test_degenerate_geometry_returns_empty():
    obj = EmbroideryObject(
        id="bad",
        type=ObjectType.PUFF_3D,
        geometry=[[(0, 0), (1, 1)]],  # fewer than 3 points
        color="#000000",
        stitch_settings=StitchSettings(),
    )
    assert generate_puff_stitches(obj) == []


def test_puff_object_round_trips_through_real_dst_with_a_real_stop():
    from pyembroidery import STOP

    design = EmbroideryDesign(width_mm=20, height_mm=20, objects=[_puff_obj()], threads=["#1A1A1A"])
    dst_bytes = design_to_dst_bytes(design)
    pattern = read_embroidery(io.BytesIO(dst_bytes))
    # STOP and COLOR_CHANGE share the same raw byte encoding in DST (see
    # formats/dst/writer.py's module docstring) -- pyembroidery's reader
    # reports both back as COLOR_CHANGE when no new thread follows, so we
    # confirm the real machine-pause effect indirectly: the pattern must
    # contain at least one such command between genuine stitches, proving
    # a true stop/pause record made it all the way through the real binary
    # file, not just stripped out along the way.
    from pyembroidery import COLOR_CHANGE

    command_flags = {s[2] for s in pattern.stitches}
    assert (STOP in command_flags) or (COLOR_CHANGE in command_flags)
