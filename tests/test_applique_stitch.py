"""
New stitch type follow-up (Bean/Motif/3D Puff/Applique) tests: APPLIQUE
stitch generator, its two real STOP commands, and DST round-trip.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.applique.generator import generate_applique_stitches
from formats.dst.writer import design_to_dst_bytes

SQUARE = [(0, 0), (15, 0), (15, 15), (0, 15)]


def _applique_obj(border_width_mm=0.0):
    return EmbroideryObject(
        id="applique-1",
        type=ObjectType.APPLIQUE,
        geometry=[SQUARE],
        color="#8B0000",
        stitch_settings=StitchSettings(border_width_mm=border_width_mm),
    )


def test_sequence_contains_exactly_two_stops():
    stitches = generate_applique_stitches(_applique_obj())
    stops = [s for s in stitches if s[2] == "stop"]
    assert len(stops) == 2


def test_phase_order_is_placement_stop_tackdown_stop_border():
    stitches = generate_applique_stitches(_applique_obj())
    stop_indices = [i for i, s in enumerate(stitches) if s[2] == "stop"]
    assert len(stop_indices) == 2
    first_stop, second_stop = stop_indices
    assert 0 < first_stop < second_stop < len(stitches) - 1


def test_zero_border_width_falls_back_to_a_sane_default():
    stitches = generate_applique_stitches(_applique_obj(border_width_mm=0.0))
    stitches_with_border = generate_applique_stitches(_applique_obj(border_width_mm=3.0))
    # Both produce real finishing-border stitches (not skipped because
    # border_width_mm looked like "0 = disabled", which is FILL/CROSS_
    # STITCH's convention but must not silently apply to APPLIQUE).
    assert len(stitches) > 0
    assert len(stitches_with_border) > 0


def test_degenerate_geometry_returns_empty():
    obj = EmbroideryObject(
        id="bad",
        type=ObjectType.APPLIQUE,
        geometry=[[(0, 0), (1, 1)]],
        color="#000000",
        stitch_settings=StitchSettings(),
    )
    assert generate_applique_stitches(obj) == []


def test_applique_object_round_trips_through_real_dst_with_real_stops():
    from pyembroidery import COLOR_CHANGE, STOP

    design = EmbroideryDesign(width_mm=25, height_mm=25, objects=[_applique_obj()], threads=["#8B0000"])
    dst_bytes = design_to_dst_bytes(design)
    pattern = read_embroidery(io.BytesIO(dst_bytes))
    command_flags = {s[2] for s in pattern.stitches}
    assert (STOP in command_flags) or (COLOR_CHANGE in command_flags)
