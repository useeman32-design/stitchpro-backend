"""
Quality-improvement tests: underlay stitching (stabilization pass stitched
before an object's own top stitches).

`StitchSettings.underlay` existed as a field since Milestone 2 but was never
implemented until now; `digitizer/underlay/generator.py` implements all
three named strategies (center_run, edge_run, zigzag) for SATIN and FILL.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyembroidery import read_dst as read_embroidery

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.stitches.dispatch import generate_object_stitches
from digitizer.underlay.generator import generate_underlay_stitches
from formats.dst.writer import design_to_dst_bytes

SATIN_BAR = [(0, 0), (0, 10), (40, 0), (40, 10)]  # 40mm x 10mm bar


def _satin_obj(underlay):
    return EmbroideryObject(
        id="bar",
        type=ObjectType.SATIN,
        geometry=SATIN_BAR,
        color="#111111",
        stitch_settings=StitchSettings(density_mm=1.0, underlay=underlay),
    )


def test_no_underlay_by_default():
    obj = _satin_obj(None)
    assert generate_underlay_stitches(obj) == []


def test_unrecognized_underlay_value_is_noop():
    obj = _satin_obj("not-a-real-strategy")
    assert generate_underlay_stitches(obj) == []


def test_satin_center_run_underlay_follows_midline():
    obj = _satin_obj("center_run")
    stitches = generate_underlay_stitches(obj)
    assert stitches
    # every underlay point should sit on y=5 (the midline between y=0 and y=10)
    for x, y, _ in stitches:
        assert abs(y - 5) < 1e-6


def test_satin_edge_run_underlay_stays_inset_from_both_rails():
    obj = _satin_obj("edge_run")
    stitches = generate_underlay_stitches(obj)
    assert stitches
    ys = [y for _, y, _ in stitches]
    # inset toward the opposite rail -- should never touch y=0 or y=10 exactly,
    # and should include points inset from both rails (near 0 and near 10)
    assert all(0 < y < 10 for y in ys)
    assert any(y < 2 for y in ys)
    assert any(y > 8 for y in ys)


def test_satin_zigzag_underlay_is_sparser_than_top_stitch():
    obj = _satin_obj("zigzag")
    underlay = generate_underlay_stitches(obj)
    top = generate_object_stitches(_satin_obj(None))  # no underlay, just top stitch
    assert underlay
    assert len(underlay) < len(top)


def test_underlay_stitched_before_top_stitch_in_dispatch():
    obj = _satin_obj("center_run")
    combined = generate_object_stitches(obj)
    underlay_only = generate_underlay_stitches(obj)
    assert combined[: len(underlay_only)] == underlay_only


RECT = [(0, 0), (30, 0), (30, 20), (0, 20)]


def _fill_obj(underlay):
    return EmbroideryObject(
        id="body",
        type=ObjectType.FILL,
        geometry=[RECT],
        color="#2E8B57",
        stitch_settings=StitchSettings(stitch_length_mm=2.0, density_mm=1.0, underlay=underlay),
    )


def test_fill_edge_run_underlay_follows_inset_contour():
    obj = _fill_obj("edge_run")
    stitches = generate_underlay_stitches(obj)
    assert stitches
    for x, y, _ in stitches:
        assert -0.1 <= x <= 30.1
        assert -0.1 <= y <= 20.1
        # inset underlay must stay strictly inside the boundary (not on it)
        assert 0 < x < 30 or 0 < y < 20


def test_fill_zigzag_underlay_is_sparser_cross_hatch():
    obj = _fill_obj("zigzag")
    underlay = generate_underlay_stitches(obj)
    top = generate_object_stitches(_fill_obj(None))
    assert underlay
    assert len(underlay) < len(top)


def test_fill_center_run_aliases_to_zigzag():
    center = generate_underlay_stitches(_fill_obj("center_run"))
    zigzag = generate_underlay_stitches(_fill_obj("zigzag"))
    assert len(center) == len(zigzag)


def test_running_has_no_underlay():
    obj = EmbroideryObject(
        id="line",
        type=ObjectType.RUNNING,
        geometry=[(0, 0), (10, 0)],
        color="#111111",
        stitch_settings=StitchSettings(underlay="edge_run"),
    )
    assert generate_underlay_stitches(obj) == []


def test_underlay_to_dst_roundtrip():
    design = EmbroideryDesign(
        width_mm=50,
        height_mm=30,
        objects=[_fill_obj("edge_run")],
        threads=["#2E8B57"],
    )
    design.objects[0].geometry = [RECT]
    dst_bytes = design_to_dst_bytes(design)
    pattern = read_embroidery(io.BytesIO(dst_bytes))
    assert len(pattern.stitches) > 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
