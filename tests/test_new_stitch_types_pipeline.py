"""
New stitch type follow-up (Bean/Motif/3D Puff/Applique) tests: end-to-end
`image_to_design` auto-classification, exactly like test_satin_thin_region.py
covers SATIN at the geometry level and test_pipeline.py covers FILL/
CROSS_STITCH at the full-image level.

Every synthetic test image here is kept at or under the pipeline's own
150px downscale cap (`DEFAULT_MAX_DIM`) specifically so `width_mm`/
`height_mm` map onto pixels with simple, predictable arithmetic -- once an
image exceeds that cap and gets downscaled, a hairline few-pixel-wide
shape's measured width in mm becomes sensitive to resampling rounding,
which is a real (and already-documented, see thin_region.py) precision
ceiling of this heuristic approach, not something a test should fight.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from digitizer.model import ObjectType
from digitizer.pipeline import image_to_design


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_hairline_bar_becomes_bean():
    # ~0.6mm measured width (2px at this image's 0.2mm/px scale) -- below
    # BEAN_MAX_WIDTH_MM (1.0), comfortably above the 0.6mm noise floor.
    img = Image.new("RGB", (140, 40), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 19, 130, 21], fill=(10, 10, 10))
    design = image_to_design(_png_bytes(img), width_mm=28, height_mm=8, max_colors=2)
    assert len(design.objects) == 1
    assert design.objects[0].type == ObjectType.BEAN


def test_long_thin_band_becomes_motif():
    # ~2.2mm wide, ~24mm long -> elongation ratio ~11, over
    # MOTIF_MIN_ELONGATION_RATIO (10) and well above BEAN's width cutoff.
    img = Image.new("RGB", (140, 40), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 15, 130, 25], fill=(10, 10, 10))
    design = image_to_design(_png_bytes(img), width_mm=28, height_mm=8, max_colors=2)
    assert len(design.objects) == 1
    assert design.objects[0].type == ObjectType.MOTIF


def test_ordinary_stroke_width_still_becomes_satin_not_bean_or_motif():
    # Regression guard: the new BEAN/MOTIF split must not eat into SATIN's
    # existing, shipped territory for a normal-width, modestly-elongated
    # stroke (the common glyph/border-bar case).
    img = Image.new("RGB", (140, 40), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 12, 60, 28], fill=(10, 10, 10))  # ~10mm long, ~3.2mm wide
    design = image_to_design(_png_bytes(img), width_mm=28, height_mm=8, max_colors=2)
    assert len(design.objects) == 1
    assert design.objects[0].type == ObjectType.SATIN


def test_bold_blocky_non_dominant_shapes_can_become_puff():
    # Several similar-sized bold/blocky shapes -- none individually
    # dominant enough to win APPLIQUE, at least one bold/convex/moderate-
    # sized enough to win PUFF_3D.
    img = Image.new("RGB", (210, 70), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([5, 10, 60, 60], radius=8, fill=(10, 10, 10))
    d.rounded_rectangle([75, 10, 130, 60], radius=8, fill=(30, 30, 200))
    d.rounded_rectangle([145, 10, 200, 60], radius=8, fill=(30, 160, 30))
    design = image_to_design(_png_bytes(img), width_mm=60, height_mm=20, max_colors=4)
    types = {o.type for o in design.objects}
    assert ObjectType.PUFF_3D in types
    assert ObjectType.APPLIQUE not in types


def test_dominant_simple_background_becomes_applique():
    # One large, convex, hole-free shape covering most of the design plus
    # a small separate accent -- the real-world "main body silhouette
    # behind a small detail" shape APPLIQUE's heuristic targets.
    img = Image.new("RGB", (160, 100), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([5, 5, 120, 95], radius=10, fill=(10, 10, 10))
    d.rectangle([130, 10, 155, 35], fill=(255, 215, 0))
    design = image_to_design(_png_bytes(img), width_mm=48, height_mm=30, max_colors=3)
    types_by_color = {o.color: o.type for o in design.objects}
    assert ObjectType.APPLIQUE in types_by_color.values()


def test_cross_stitch_style_never_triggers_any_new_auto_type():
    # All four new heuristics are explicitly scoped to stitch_style="fill"
    # only -- CROSS_STITCH is a deliberate stylistic choice the heuristics
    # must never override (same existing rule SATIN already follows).
    img = Image.new("RGB", (140, 40), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 19, 130, 21], fill=(10, 10, 10))
    design = image_to_design(_png_bytes(img), width_mm=28, height_mm=8, max_colors=2, stitch_style="cross")
    assert len(design.objects) == 1
    assert design.objects[0].type == ObjectType.CROSS_STITCH


def test_letter_o_with_a_hole_is_never_auto_classified_as_applique_or_puff():
    # Regression guard matching test_pipeline.py's existing letter-O case:
    # a holed region must never become APPLIQUE or PUFF_3D (both require
    # len(rings) == 1), regardless of how dominant its area is.
    img = Image.new("RGB", (120, 120), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([10, 10, 110, 110], fill=(20, 20, 20))
    d.ellipse([35, 35, 85, 85], fill=(255, 255, 255))
    design = image_to_design(_png_bytes(img), width_mm=40, height_mm=40, max_colors=3)
    assert len(design.objects) == 1
    assert design.objects[0].type == ObjectType.FILL
