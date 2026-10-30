"""Regression tests for a real bug found during a technical audit of
`digitizer/segmentation/background_removal.py`:

    If any part of the real foreground design touches the image's outer
    border, that foreground color gets sampled into the "trusted" border
    seed-color palette (`_flood_fill_connected_background`'s `seed_colors`,
    reduced to `reference` in `compute_background_mask_and_reference`). The
    enclosed-hole pass then reclassifies *every* pixel anywhere in the
    image matching that color as background -- including real design
    pixels nowhere near the border, nowhere near any genuine enclosed hole.

    Reproduced during the audit with a synthetic cross-stitch-style pattern
    drawn edge-to-edge: 0 of 160,000 foreground pixels survived. The same
    pattern redrawn with a margin (touching no edge) correctly kept 17.6%
    foreground.

Each test below builds a real foreground "design" shape with a solid,
clearly background-*un*like color, deliberately placed so it touches one
or more image edges, and asserts that the bulk of that design survives
background removal as real foreground -- not just a token pixel or two.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from digitizer.segmentation.background_removal import compute_background_mask

BG = (245, 245, 245)  # off-white "fabric"/backdrop
FG = (20, 110, 200)  # a solid, clearly non-background blue "design" color
SIZE = 200


def _canvas() -> Image.Image:
    return Image.new("RGB", (SIZE, SIZE), BG)


def _foreground_survival_fraction(mask: list[list[bool]], box: tuple[int, int, int, int]) -> float:
    """Fraction of pixels inside `box` (x0, y0, x1, y1, half-open) that
    `mask` still calls foreground (`False`). `box` is expected to be
    (close to) the real design's own bounding box."""
    x0, y0, x1, y1 = box
    total = 0
    survived = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            total += 1
            if not mask[y][x]:
                survived += 1
    return survived / total if total else 0.0


class TestDesignTouchingASingleEdge:
    def test_design_touching_left_edge_mostly_survives(self):
        img = _canvas()
        box = (0, 70, 60, 130)  # flush against x=0
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=FG)
        mask = compute_background_mask(img)
        assert not mask[100][30]  # well inside the design, away from the touching edge
        assert _foreground_survival_fraction(mask, box) >= 0.9

    def test_design_touching_right_edge_mostly_survives(self):
        img = _canvas()
        box = (SIZE - 60, 70, SIZE, 130)  # flush against x=SIZE-1
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=FG)
        mask = compute_background_mask(img)
        assert not mask[100][SIZE - 30]
        assert _foreground_survival_fraction(mask, box) >= 0.9

    def test_design_touching_top_edge_mostly_survives(self):
        img = _canvas()
        box = (70, 0, 130, 60)  # flush against y=0
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=FG)
        mask = compute_background_mask(img)
        assert not mask[30][100]
        assert _foreground_survival_fraction(mask, box) >= 0.9

    def test_design_touching_bottom_edge_mostly_survives(self):
        img = _canvas()
        box = (70, SIZE - 60, 130, SIZE)  # flush against y=SIZE-1
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=FG)
        mask = compute_background_mask(img)
        assert not mask[SIZE - 30][100]
        assert _foreground_survival_fraction(mask, box) >= 0.9


class TestDesignTouchingMultipleEdges:
    def test_design_touching_two_adjacent_edges_mostly_survives(self):
        # An L-ish corner block flush against both the left and top edges.
        img = _canvas()
        box = (0, 0, 60, 60)
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=FG)
        mask = compute_background_mask(img)
        assert not mask[30][30]
        assert _foreground_survival_fraction(mask, box) >= 0.85

    def test_design_touching_three_edges_mostly_survives(self):
        # Three same-colored design pieces, one flush against each of the
        # top/left/bottom edges, each only over a modest stretch of that
        # edge -- real background is still clearly visible along the rest
        # of every edge (as opposed to a single edge being spanned in its
        # *entirety*, which has no real background signal left on that
        # side at all and is an inherent, undecidable limit of any
        # border-sampling technique -- see "remaining limitations").
        img = _canvas()
        d = ImageDraw.Draw(img)
        top_box = (80, 0, 120, 40)
        left_box = (0, 85, 30, 115)
        bottom_box = (80, SIZE - 40, 120, SIZE)
        for box in (top_box, left_box, bottom_box):
            d.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=FG)
        mask = compute_background_mask(img)
        assert not mask[20][100]  # top piece
        assert not mask[100][15]  # left piece
        assert not mask[SIZE - 20][100]  # bottom piece
        for box in (top_box, left_box, bottom_box):
            assert _foreground_survival_fraction(mask, box) >= 0.8

    def test_design_touching_all_four_edges_mostly_survives(self):
        # A plus/cross shape whose four arms each reach a different edge,
        # with real background still visible in the four corners -- the
        # realistic "touches every edge, but background is still clearly
        # present" case, as opposed to a design that fills the entire
        # frame (which is an inherent, undecidable limit of any
        # border-based technique -- see the module's own docstring and
        # this audit's "remaining limitations").
        img = _canvas()
        d = ImageDraw.Draw(img)
        # Horizontal arm: left edge to right edge.
        d.rectangle([0, 85, SIZE - 1, 115], fill=FG)
        # Vertical arm: top edge to bottom edge.
        d.rectangle([85, 0, 115, SIZE - 1], fill=FG)
        mask = compute_background_mask(img)
        assert not mask[100][100]  # center of the cross
        assert not mask[5][100]  # near the top tip
        assert not mask[195][100]  # near the bottom tip
        assert not mask[100][5]  # near the left tip
        assert not mask[100][195]  # near the right tip
        # Corners must still be real background (there's visible backdrop there).
        assert mask[5][5]
        assert mask[5][195]
        assert mask[195][5]
        assert mask[195][195]
        cross_box = (0, 0, SIZE, SIZE)
        # Measure survival only over the cross's own pixels, not the whole
        # canvas (most of the canvas is legitimately background here).
        cross_pixels = 0
        survived = 0
        for y in range(SIZE):
            for x in range(SIZE):
                if 85 <= y <= 115 or 85 <= x <= 115:
                    cross_pixels += 1
                    if not mask[y][x]:
                        survived += 1
        assert survived / cross_pixels >= 0.85


class TestOrdinaryDesignWithMargins:
    def test_inset_design_still_fully_survives(self):
        # Sanity / non-regression: the ordinary case every other test file
        # already covers, repeated here so this file is a complete,
        # self-contained reproduction of the bug report's exact 9 scenarios.
        img = _canvas()
        box = (60, 60, 140, 140)
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=FG)
        mask = compute_background_mask(img)
        assert _foreground_survival_fraction(mask, box) == 1.0
        assert mask[0][0] and mask[0][SIZE - 1] and mask[SIZE - 1][0] and mask[SIZE - 1][SIZE - 1]


class TestEnclosedHolesStillWork:
    def test_letter_like_counter_is_still_classified_as_a_hole(self):
        # Non-regression for the pre-existing, documented "enclosed hole"
        # feature (see test_background_removal.py) -- a fix for the
        # edge-touching bug above must not disable this.
        img = Image.new("RGB", (60, 60), BG)
        px = img.load()
        for y in range(10, 50):
            for x in range(10, 50):
                px[x, y] = (30, 30, 30)
        for y in range(22, 38):
            for x in range(22, 38):
                px[x, y] = BG
        mask = compute_background_mask(img)
        assert mask[30][30] is True  # the enclosed counter is still a hole
        assert mask[15][15] is False  # the ring itself is still foreground

    def test_enclosed_hole_survives_even_when_an_unrelated_design_touches_the_edge(self):
        # The real-world combination: a design with a genuine letter-style
        # counter/hole *and* a separate shape elsewhere that touches the
        # image border. The edge-touching fix must not break hole
        # classification, and the hole fix must not resurrect the
        # edge-touching bug.
        img = Image.new("RGB", (120, 60), BG)
        px = img.load()
        # Ring-with-a-hole on the left half (same shape as the test above).
        for y in range(10, 50):
            for x in range(10, 50):
                px[x, y] = (30, 30, 30)
        for y in range(22, 38):
            for x in range(22, 38):
                px[x, y] = BG
        # A separate blue block on the right half, flush against the right edge.
        box = (90, 15, 120, 45)
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=FG)
        mask = compute_background_mask(img)
        assert mask[30][30] is True  # counter is still a hole
        assert mask[15][15] is False  # ring is still foreground
        assert not mask[30][105]  # the edge-touching block mostly survives
        assert _foreground_survival_fraction(mask, box) >= 0.85


class TestForegroundSimilarToBackgroundColor:
    def test_edge_touching_design_close_in_color_to_background_still_mostly_survives(self):
        # A harder case: the design's color is close to (but distinctly
        # different from) the background's, and it touches an edge. This
        # must not be swallowed by the hole-tolerance check once (if) its
        # color leaks into the reference palette.
        near_bg_fg = (235, 180, 180)  # distinctly pink-ish, but close in brightness to BG
        img = _canvas()
        box = (0, 70, 60, 130)
        ImageDraw.Draw(img).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=near_bg_fg)
        mask = compute_background_mask(img)
        assert not mask[100][30]
        assert _foreground_survival_fraction(mask, box) >= 0.85

    def test_genuinely_background_colored_edge_touching_region_can_still_be_removed(self):
        # The flip side, to prove the fix doesn't just disable background
        # detection outright: a region that touches the edge and is
        # actually, truly the same color as the background (not just a
        # nearby foreground color) should still be treated as background.
        img = _canvas()
        # A same-color-as-background "tab" poking out from the border --
        # there is no real design here at all, just more backdrop.
        ImageDraw.Draw(img).rectangle([0, 70, 59, 130], fill=BG)
        mask = compute_background_mask(img)
        assert mask[100][30] is True
