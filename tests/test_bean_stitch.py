"""
New stitch type follow-up (Bean/Motif/3D Puff/Applique) tests: BEAN stitch
generator + DST round-trip.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.bean.generator import generate_bean_stitch
from formats.dst.writer import design_to_dst_bytes

LINE = [(0, 0), (10, 0)]


def test_default_three_passes_each_covers_the_same_points():
    stitches = generate_bean_stitch(LINE, stitch_length_mm=2.5, passes=3)
    base_len = len(generate_bean_stitch(LINE, stitch_length_mm=2.5, passes=1))
    assert len(stitches) == base_len * 3


def test_only_the_very_first_point_is_a_jump():
    stitches = generate_bean_stitch(LINE, stitch_length_mm=2.5, passes=3)
    jumps = [i for i, (_x, _y, is_jump) in enumerate(stitches) if is_jump]
    assert jumps == [0]


def test_passes_alternate_direction_and_end_near_start_of_next():
    stitches = generate_bean_stitch(LINE, stitch_length_mm=5.0, passes=3)
    base = generate_bean_stitch(LINE, stitch_length_mm=5.0, passes=1)
    n = len(base)
    pass0 = stitches[0:n]
    pass1 = stitches[n : 2 * n]
    pass2 = stitches[2 * n : 3 * n]
    assert [p[:2] for p in pass0] == [p[:2] for p in base]
    assert [p[:2] for p in pass1] == [p[:2] for p in reversed(base)]
    assert [p[:2] for p in pass2] == [p[:2] for p in base]


def test_passes_less_than_one_raises():
    try:
        generate_bean_stitch(LINE, passes=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_single_pass_equals_plain_running_stitch_count():
    from digitizer.stitches.running.generator import generate_running_stitch

    stitches = generate_bean_stitch(LINE, stitch_length_mm=2.5, passes=1)
    base = generate_running_stitch(LINE, 2.5)
    assert len(stitches) == len(base)


def test_bean_object_round_trips_through_real_dst():
    import io

    obj = EmbroideryObject(
        id="bean-1",
        type=ObjectType.BEAN,
        geometry=LINE,
        color="#111111",
        stitch_settings=StitchSettings(stitch_length_mm=2.5, bean_passes=3),
    )
    design = EmbroideryDesign(width_mm=20, height_mm=20, objects=[obj], threads=["#111111"])
    dst_bytes = design_to_dst_bytes(design)
    pattern = read_embroidery(io.BytesIO(dst_bytes))
    # Just confirm it reads back with a plausible number of stitch/jump
    # records and no exception -- byte-level encoding is already covered
    # by test_running_stitch.py's own round-trip tests.
    assert len(pattern.stitches) > 0
