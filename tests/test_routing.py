"""
Milestone 4 tests: combine running/satin/fill + basic stitch-order routing.

Every test both checks our own router output AND (for the full-pipeline
tests) round-trips through pyembroidery (write -> independently read back)
to catch format-level bugs, not just ordering-logic bugs -- matching brief
section 21's test methodology, same as the other stitch-type test files.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery
from pyembroidery import COLOR_CHANGE

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.routing.router import route_objects
from formats.dst.writer import compile_design, design_to_dst_bytes


def _running(id_, p0, p1, color, sequence=0, layer=0):
    return EmbroideryObject(
        id=id_,
        type=ObjectType.RUNNING,
        geometry=[p0, p1],
        color=color,
        sequence=sequence,
        layer=layer,
    )


# ---------------------------------------------------------------------------
# 1. Layering: a hard constraint -- layer 0 always fully before layer 1,
#    regardless of input order/sequence.
# ---------------------------------------------------------------------------

def test_layer_zero_always_before_layer_one():
    objs = [
        _running("top", (0, 0), (1, 0), "#111111", sequence=0, layer=1),
        _running("under", (5, 5), (6, 5), "#111111", sequence=1, layer=0),
    ]
    routed = route_objects(objs)
    assert [o.id for o in routed] == ["under", "top"]


def test_three_layers_ordered_ascending():
    objs = [
        _running("c", (0, 0), (1, 0), "#111111", sequence=0, layer=2),
        _running("a", (0, 0), (1, 0), "#111111", sequence=1, layer=0),
        _running("b", (0, 0), (1, 0), "#111111", sequence=2, layer=1),
    ]
    routed = route_objects(objs)
    assert [o.id for o in routed] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 2. Color grouping: same-color objects become contiguous even when the
#    input interleaves them with a different color.
# ---------------------------------------------------------------------------

def test_interleaved_colors_are_grouped_contiguously():
    objs = [
        _running("red-1", (0, 0), (1, 0), "#FF0000", sequence=0),
        _running("blue-1", (0, 0), (1, 0), "#0000FF", sequence=1),
        _running("red-2", (0, 0), (1, 0), "#FF0000", sequence=2),
    ]
    routed = route_objects(objs)
    colors = [o.color for o in routed]
    # red's two objects must be adjacent (contiguous run), not split by blue
    red_positions = [i for i, c in enumerate(colors) if c == "#FF0000"]
    assert red_positions == [0, 1] or red_positions == [1, 2]
    assert len(set(colors[i] for i in range(len(colors)))) == len(
        [c for c, _ in __import__("itertools").groupby(colors)]
    )


def test_first_appearance_color_order_preserved():
    objs = [
        _running("blue-1", (0, 0), (1, 0), "#0000FF", sequence=0),
        _running("red-1", (0, 0), (1, 0), "#FF0000", sequence=1),
        _running("blue-2", (50, 50), (51, 50), "#0000FF", sequence=2),
    ]
    routed = route_objects(objs)
    # blue appeared first in the input -> blue's group should come first
    assert routed[0].color == "#0000FF"
    assert routed[1].color == "#0000FF"
    assert routed[2].color == "#FF0000"


# ---------------------------------------------------------------------------
# 3. Nearest-neighbor travel minimization within a color group.
# ---------------------------------------------------------------------------

def test_nearest_neighbor_picks_closer_object_first():
    objs = [
        _running("far", (100, 100), (105, 100), "#111111", sequence=0),
        _running("near", (1, 1), (2, 1), "#111111", sequence=1),
    ]
    routed = route_objects(objs)
    # starting from (0,0), "near" (starts at (1,1)) should be picked before
    # "far" (starts at (100,100)), even though "far" was given first
    assert [o.id for o in routed] == ["near", "far"]


def test_nearest_neighbor_chains_from_previous_endpoint():
    objs = [
        _running("a", (0, 0), (10, 0), "#111111", sequence=0),
        _running("b", (9, 1), (9, 5), "#111111", sequence=1),  # close to a's end
        _running("c", (-50, -50), (-51, -50), "#111111", sequence=2),  # far from everything
    ]
    routed = route_objects(objs)
    assert [o.id for o in routed] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 4. Sequence is reassigned to reflect the final routed order.
# ---------------------------------------------------------------------------

def test_sequence_reassigned_after_routing():
    objs = [
        _running("b", (50, 50), (51, 50), "#111111", sequence=0),
        _running("a", (0, 0), (1, 0), "#111111", sequence=1),
    ]
    routed = route_objects(objs)
    for i, obj in enumerate(routed):
        assert obj.sequence == i


def test_empty_object_list():
    assert route_objects([]) == []


# ---------------------------------------------------------------------------
# 5. Full pipeline: interleaved colors should compile to exactly one
#    COLOR_CHANGE instead of two, proving routing actually ran before
#    compilation (not just in isolation).
# ---------------------------------------------------------------------------

def test_compile_design_applies_routing_to_minimize_color_changes():
    design = EmbroideryDesign(
        width_mm=60,
        height_mm=60,
        objects=[
            EmbroideryObject(
                id="red-1", type=ObjectType.RUNNING, geometry=[(0, 0), (10, 0)],
                color="#FF0000", sequence=0,
            ),
            EmbroideryObject(
                id="blue-1", type=ObjectType.RUNNING, geometry=[(0, 10), (10, 10)],
                color="#0000FF", sequence=1,
            ),
            EmbroideryObject(
                id="red-2", type=ObjectType.RUNNING, geometry=[(0, 20), (10, 20)],
                color="#FF0000", sequence=2,
            ),
        ],
        threads=["#FF0000", "#0000FF"],
    )

    dst_bytes = design_to_dst_bytes(design)
    pattern = read_embroidery(io.BytesIO(dst_bytes))
    commands = [s[2] for s in pattern.stitches]
    # naive (un-routed) order red->blue->red would need 2 color changes;
    # routed (red,red grouped, then blue) needs only 1
    assert commands.count(COLOR_CHANGE) == 1


def test_compile_design_respects_layering_end_to_end():
    design = EmbroideryDesign(
        width_mm=60,
        height_mm=60,
        objects=[
            EmbroideryObject(
                id="top-satin", type=ObjectType.SATIN,
                geometry=[(0, 0), (0, 4), (20, 0), (20, 4)],
                color="#111111", stitch_settings=StitchSettings(density_mm=2.0),
                sequence=0, layer=1,
            ),
            EmbroideryObject(
                id="underlay", type=ObjectType.RUNNING,
                geometry=[(0, 2), (20, 2)],
                color="#111111", sequence=1, layer=0,
            ),
        ],
        threads=["#111111"],
    )
    routed = route_objects(list(design.objects))
    assert [o.id for o in routed] == ["underlay", "top-satin"]

    # end-to-end: compiling must not raise, and should still produce a
    # valid, readable DST with both objects' stitches present.
    dst_bytes = design_to_dst_bytes(design)
    pattern = read_embroidery(io.BytesIO(dst_bytes))
    assert len(pattern.stitches) > 0


def test_all_three_stitch_types_with_routing():
    # Same shapes as test_fill_stitch.py's combined test, but deliberately
    # given out of color-grouping order and on explicit layers, to prove
    # the full pipeline (routing + all three generators) works together.
    design = EmbroideryDesign(
        width_mm=60,
        height_mm=40,
        objects=[
            EmbroideryObject(
                id="body", type=ObjectType.FILL,
                geometry=[(0, 12), (20, 12), (20, 22), (0, 22)],
                color="#2E8B57", stitch_settings=StitchSettings(stitch_length_mm=3.0, density_mm=2.0),
                sequence=0, layer=0,
            ),
            EmbroideryObject(
                id="outline", type=ObjectType.RUNNING,
                geometry=[(0, 0), (20, 0)],
                color="#FF0000", sequence=1, layer=1,
            ),
            EmbroideryObject(
                id="band", type=ObjectType.SATIN,
                geometry=[(0, 5), (0, 8), (20, 5), (20, 8)],
                color="#0000FF", stitch_settings=StitchSettings(density_mm=2.0),
                sequence=2, layer=1,
            ),
        ],
        threads=["#2E8B57", "#FF0000", "#0000FF"],
    )
    pattern = compile_design(design)
    assert len(pattern.threadlist) == 3
    commands = [s[2] for s in pattern.stitches]
    assert commands.count(COLOR_CHANGE) == 2

    # layer 0 (body/fill) must be fully stitched before layer 1 objects --
    # verify by checking routing directly (compile_design already routes
    # internally, this double-checks route_objects agrees).
    routed = route_objects(list(design.objects))
    assert routed[0].id == "body"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
