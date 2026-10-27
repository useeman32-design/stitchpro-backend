"""
Color quantization + background detection (Milestone 5, brief section
14/15: "image -> segmentation").

This is the first real step of server-side digitizing: turn an arbitrary
raster image into a small palette of "thread colors" plus a per-pixel label
grid saying which palette color (or "background") each pixel belongs to.
Everything downstream (digitizer/segmentation/polygons.py) works off that
label grid, not the raw image.

Background/material detection itself lives in
digitizer/segmentation/background_removal.py (a border-connected flood fill,
robust to a real photographed garment's lighting/shading -- see that
module's docstring for why a flood fill, not a single global color
distance, is required for the "snap a photo of a design on a real shirt"
use case) -- this module just quantizes whatever that leaves as foreground.

Deliberately uses only Pillow (already a dependency) -- no OpenCV/
scikit-image, matching the conservative dependency stance from
docs/embroidery-engine-research.md until the cPanel-vs-VPS hosting question
is settled (OpenCV's wheel size and native dependencies are a much bigger
ask for shared hosting than Pillow).
"""

from __future__ import annotations

from collections import Counter, deque

from PIL import Image

from digitizer.segmentation.background_removal import compute_background_mask_and_reference

BACKGROUND_LABEL = -1

Rgb = tuple[int, int, int]

# How many palette slots the first quantize pass gets internally, before
# the anti-aliasing-aware merge below collapses it back down toward
# `max_colors`. A thin shape -- text, a logo's stroke outline -- is mostly
# *edge*: most of its foreground pixels sit within a pixel or two of the
# background, so a single font glyph can easily contain a hundred-plus
# distinct intermediate gray shades between "full ink" and "full
# background" even though only one real design color is involved.
# Quantizing straight to `max_colors` (as low as 2-3 for a simple logo)
# forces MEDIANCUT to spend most of its budget picking distinct
# representative points *along that one gradient* instead of separating
# genuinely different design colors -- left unmerged, that digitizes as
# several overlapping, oddly-shaped fills instead of one clean shape, which
# is exactly the "fill isn't solid" / "looks like scattered lines" failure
# mode a thin-stroke text logo triggers hardest. Over-sampling first and
# merging after (see `_merge_near_duplicate_colors`'s blend-line pass)
# fixes this without the caller needing to know in advance how much
# anti-aliasing a given image has.
_INTERMEDIATE_QUANTIZE_FLOOR = 24
_INTERMEDIATE_QUANTIZE_CAP = 64


def _dist2(a: Rgb, b: Rgb) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _point_segment_dist2(p: Rgb, a: Rgb, b: Rgb) -> float:
    """Squared distance from `p` to the closest point on segment `a`-`b` in
    RGB space (3-D point-to-segment distance, clamped to the segment)."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    seg_len2 = dx * dx + dy * dy + dz * dz
    if seg_len2 < 1e-9:
        return _dist2(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy + (p[2] - a[2]) * dz) / seg_len2
    t = max(0.0, min(1.0, t))
    cx, cy, cz = a[0] + t * dx, a[1] + t * dy, a[2] + t * dz
    return (p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2


def _merge_near_duplicate_colors(
    palette_rgb: list[Rgb],
    merge_tolerance: float,
    background_refs: tuple[Rgb, ...] = (),
    blend_tolerance: float = 12.0,
    counts: tuple[int, ...] = (),
    fg_blend_tolerance: float = 15.0,
    fg_blend_max_count_ratio: float = 0.15,
) -> tuple[list[Rgb], list[int]]:
    """Median-cut quantization can split what should be one solid color
    into two (or more) near-identical palette entries -- a real, observed
    case: a clean solid-blue logo badge photographed/exported with
    anti-aliased edges comes out with its interior blue plus a handful of
    edge-blend pixels, and `Image.quantize()` sometimes allocates a
    *second*, barely-different blue swatch to the mix rather than lumping
    them together. Left alone, this is exactly what produces the "fill
    isn't solid" symptom: two "different" thread colors, each only getting
    part of the shape, with a visible gap/seam between them instead of one
    clean fill.

    Two separate merge rules feed the same transitive union-find:

    1. Flat `merge_tolerance`: any two palette entries within that plain
       RGB distance of each other are the same color (catches adjacent
       quantize buckets along a gradient, or literal near-duplicates).

    2. Blend-line `blend_tolerance` (only active when `background_refs` is
       given): anti-aliasing linearly interpolates between a design color
       and whatever sits behind it, so a genuine edge-blend pixel's color
       always lies almost exactly *on* the straight line between its
       source color and the background it's blending into -- regardless of
       how far from either endpoint it sits. A palette entry within
       `blend_tolerance` of the segment connecting another entry to any
       known background reference color is folded into that entry. This is
       what lets a 100+-shade gray gradient from a single anti-aliased
       black glyph collapse into one group even though its lightest and
       darkest shades are nowhere near each other in plain RGB distance.

    3. Foreground blend-line `fg_blend_tolerance` (only active when
       `counts` is given): two *design* colors that touch each other
       directly -- a red shape overlapping a blue one, text sitting on a
       colored background swatch -- blend into each other along their
       shared boundary exactly the same way a shape blends into the
       background, but rule 2 above only ever tests against
       `background_refs`, so that boundary's blend pixels (a reddish-
       purple between a red and a blue region, say) previously survived
       quantization as their own "real" extra color. This is the actual
       mechanism behind the "color count shows more colors than the
       design really has" complaint for any multi-color design (a single
       foreground-color-on-white logo never hits this path, since its only
       blend line already goes through rule 2; it only shows up once two
       *different* design colors share an edge). A palette entry `j` is
       folded into whichever of two *other* entries `i`/`k` it's closer to
       when its color sits within `fg_blend_tolerance` of the segment
       between them -- gated by `fg_blend_max_count_ratio` so a genuinely
       separate (if minor) design color never gets eaten just because it
       happens to be roughly colinear with two bigger colors in RGB space:
       a real boundary-blend sliver's pixel count scales with the shared
       *perimeter* between two regions, not their area, so it is reliably
       much smaller than either neighboring region's own count, whereas an
       intentional small color accent (e.g. a logo's small highlight
       detail) is not specifically constrained to be small relative to
       *both* of two particular other colors at once. This is a
       population-only heuristic (deliberately not also requiring the
       candidate to be spatially touching `i`/`k`'s pixels -- a real
       smooth anti-aliasing gradient commonly passes through several
       intermediate quantize shades in a chain, so a strict "must directly
       border the pure parent colors" spatial check rejects genuine blend
       colors as often as it would reject a genuine accent, and was tried
       and reverted for exactly that reason -- see session notes). Known,
       accepted limitation: a small design color that happens to be both
       population-minor relative to two bigger colors *and* RGB-colinear
       with them can still be merged away; this trades a rare adversarial
       false-merge for reliably fixing the much more common real bug
       (genuine cross-color boundary blends surviving as bogus "extra"
       thread colors).

    Returns the deduplicated palette plus an `old_index -> new_index` remap
    list. When a group has more than one member and `background_refs` is
    given, the member *farthest* from every background reference color
    (i.e. least blended, most "pure") is used as that group's
    representative color, so the output thread color is the shape's actual
    ink color rather than some washed-out mid-gradient average. With no
    background refs (or a singleton group), the union-find root's own
    color is used, matching this function's original behavior exactly.
    """
    n = len(palette_rgb)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tol2 = merge_tolerance * merge_tolerance
    blend_tol2 = blend_tolerance * blend_tolerance
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if _dist2(palette_rgb[i], palette_rgb[j]) <= tol2:
                union(i, j)
                continue
            if not background_refs:
                continue
            for bg in background_refs:
                if _point_segment_dist2(palette_rgb[j], palette_rgb[i], bg) <= blend_tol2:
                    union(i, j)
                    break
                if _point_segment_dist2(palette_rgb[i], palette_rgb[j], bg) <= blend_tol2:
                    union(i, j)
                    break

    if counts and len(counts) == n:
        fg_blend_tol2 = fg_blend_tolerance * fg_blend_tolerance
        for i in range(n):
            for k in range(i + 1, n):
                for j in range(n):
                    if j == i or j == k:
                        continue
                    if find(j) == find(i) or find(j) == find(k):
                        continue
                    min_parent_count = min(counts[i], counts[k])
                    if min_parent_count <= 0 or counts[j] > fg_blend_max_count_ratio * min_parent_count:
                        continue
                    if _point_segment_dist2(palette_rgb[j], palette_rgb[i], palette_rgb[k]) > fg_blend_tol2:
                        continue
                    nearer = i if _dist2(palette_rgb[j], palette_rgb[i]) <= _dist2(palette_rgb[j], palette_rgb[k]) else k
                    union(j, nearer)

    groups: dict[int, list[int]] = {}
    seen_roots: list[int] = []
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
            seen_roots.append(root)
        groups[root].append(i)

    def _min_dist_to_bg(color: Rgb) -> float:
        return min((_dist2(color, bg) for bg in background_refs), default=0.0)

    root_to_new: dict[int, int] = {}
    new_palette: list[Rgb] = []
    remap = [0] * n
    for root in seen_roots:
        members = groups[root]
        if background_refs and len(members) > 1:
            rep = max(members, key=lambda idx: _min_dist_to_bg(palette_rgb[idx]))
        else:
            # `root` is itself a valid palette index (union-find roots are
            # drawn from the same 0..n-1 index space), so its own color is
            # a perfectly good, deterministic representative here.
            rep = root
        root_to_new[root] = len(new_palette)
        new_palette.append(palette_rgb[rep])
        for i in members:
            remap[i] = root_to_new[root]
    return new_palette, remap


def _reduce_to_max_colors(
    colors: list[Rgb], counts: list[int], max_colors: int
) -> tuple[list[Rgb], list[int]]:
    """After anti-aliasing merge, a design can still legitimately have more
    distinct colors than the caller's `max_colors` budget (e.g. a 10-color
    photo capped to 6 threads) -- unlike the anti-aliasing case, these really
    are different colors, so the right move is a second, population-weighted
    quantize pass over the *merged* representative colors, not another
    blend-line merge. `counts[i]` (foreground pixel count backing `colors[i]`)
    weights each color's influence so a tiny sliver of color doesn't get the
    same say as a color covering half the design.

    Returns `(final_colors, old_index -> new_index remap)`.
    """
    n = len(colors)
    if n <= max_colors:
        return colors, list(range(n))

    total = sum(counts) or 1
    sample_cap = 3000
    sample: list[Rgb] = []
    sample_group: list[int] = []
    for idx, (color, count) in enumerate(zip(colors, counts)):
        reps = max(1, round(count / total * sample_cap))
        sample.extend([color] * reps)
        sample_group.extend([idx] * reps)

    strip = Image.new("RGB", (len(sample), 1))
    strip.putdata(sample)
    quantized = strip.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    q_palette = quantized.getpalette() or []
    q_pixels = quantized.load()
    n_final = min(max_colors, len(q_palette) // 3)
    final_colors = [tuple(q_palette[i * 3 : i * 3 + 3]) for i in range(n_final)]

    votes: list[Counter] = [Counter() for _ in range(n)]
    for i, group_idx in enumerate(sample_group):
        votes[group_idx][q_pixels[i, 0]] += 1

    remap = [votes[idx].most_common(1)[0][0] if votes[idx] else 0 for idx in range(n)]
    return final_colors, remap


def build_label_grid(
    img: Image.Image,
    max_colors: int = 6,
    alpha_threshold: int = 16,
    bg_step_tolerance: float = 30.0,
    color_merge_tolerance: float = 18.0,
    remove_background: bool = True,
    background_blend_tolerance: float = 12.0,
) -> tuple[list[str], list[list[int]]]:
    """Reduce `img` (any PIL mode, already downscaled by the caller) to at
    most `max_colors` hex colors plus a label grid.

    Returns `(palette_hex, label_grid)`:
      - `palette_hex[i]` is the hex color for label `i`.
      - `label_grid[y][x]` is either an index into `palette_hex`, or
        `BACKGROUND_LABEL` for a pixel the border-connected flood fill
        (see background_removal.py) classified as background/surrounding
        material, transparent, or part of a small region the closing pass
        folded back into the background.

    Only genuine foreground pixels -- the ones the flood fill couldn't
    reach from any image border or transparent region through a chain of
    small local color steps -- compete for palette slots, so a design's
    real colors are never diluted by a shirt's lighting gradient or a
    photo's compression noise around its edges.

    Internally this quantizes to a generous intermediate palette (well
    above `max_colors`) first, then merges anti-aliasing shades back
    together using both a flat RGB-distance tolerance and a blend-line test
    against the detected background's own reference colors (see
    `_merge_near_duplicate_colors`), and only *then* -- if genuinely
    distinct colors still outnumber `max_colors` -- reduces to the caller's
    requested budget via a population-weighted second quantize pass (see
    `_reduce_to_max_colors`). A thin-stroke text logo or line-art badge is
    mostly anti-aliased edge pixels relative to its "core" ink color, so
    quantizing straight to a small `max_colors` without this intermediate
    step reliably fragments one real color into several, which is what
    made auto-digitized text/line art come out as scattered unmerged
    slivers instead of one clean shape.

    `remove_background=False` skips the flood fill entirely (every opaque
    pixel is treated as foreground, matching exactly what the
    `/api/design/remove-background` preview endpoint showed the user when
    they chose "keep original" instead of proceeding with the background
    removed) -- see digitizer/pipeline.py's `image_to_design` docstring for
    the end-to-end contract this flag preserves between preview and final
    digitized output.

    `background_blend_tolerance` is rule 2's own tolerance inside
    `_merge_near_duplicate_colors` (see that function's docstring) -- how
    close a palette color needs to lie to the line between another palette
    color and a detected background reference color to be folded into it,
    i.e. "this is just an anti-aliased edge blending into the background,
    not a real extra color". The default (12.0) is tuned for the common
    case this rule exists for: a flat logo on a *single* solid (or
    near-solid) background color. It breaks down for a background with
    many genuinely different reference shades across it (a photographed
    garment's lighting gradient, picked up as a dozen-plus distinct
    `background_refs` entries) -- with that many candidate lines to test,
    rule 2 reliably finds *some* plausible-looking blend path connecting
    almost any two unrelated foreground colors, transitively collapsing a
    genuinely multi-color design down to one washed-out average (confirmed
    on this project's own photographed sample assets -- see session notes).
    Callers that already know their source went through `artworkify.py`
    (i.e. was classified 'photo') should pass `0.0` -- fully disabling this
    specific rule. That was only settled on after benchmarking across a
    full 8-image sample library: an intermediate value (2.0) that looked
    sufficient on a few images still silently collapsed another real test
    image's subtler color distinctions down to one. This is safe
    specifically because `artworkify`'s own posterize step already performs
    the anti-aliasing cleanup this rule exists for (collapsing a smooth AA
    gradient to a handful of flat levels before quantization even runs), so
    rule 2's extra cross-color blend-line check is largely redundant -- not
    just tolerable to loosen -- on already-artworkified input. Do not pass
    `0.0` for ordinary flat-art callers that skip `artworkify`; that would
    reopen the original anti-aliasing fragmentation bug this rule exists to
    fix.
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    rgba_pixels = rgba.load()

    if remove_background:
        bg_mask, bg_reference = compute_background_mask_and_reference(
            rgba, alpha_threshold=alpha_threshold, step_tolerance=bg_step_tolerance
        )
    else:
        bg_mask = [[False] * w for _ in range(h)]
        bg_reference = []

    fg_coords: list[tuple[int, int]] = []
    fg_colors: list[Rgb] = []
    for y in range(h):
        for x in range(w):
            if bg_mask[y][x]:
                continue
            r, g, b, a = rgba_pixels[x, y]
            if a < alpha_threshold:
                continue
            fg_coords.append((x, y))
            fg_colors.append((r, g, b))

    label_grid: list[list[int]] = [[BACKGROUND_LABEL] * w for _ in range(h)]
    if not fg_colors:
        return [], label_grid

    # Quantize just the foreground pixels, packed into a 1-D strip image --
    # Image.quantize() needs an image, but the 2-D layout doesn't matter for
    # palette generation, only the multiset of colors does. Use a generous
    # intermediate budget (see module docstring / _INTERMEDIATE_QUANTIZE_*)
    # so the anti-aliasing merge below has fine-enough buckets to work with,
    # rather than quantizing straight to `max_colors`.
    intermediate_colors = max(max_colors, min(_INTERMEDIATE_QUANTIZE_CAP, max(_INTERMEDIATE_QUANTIZE_FLOOR, max_colors)))
    fg_strip = Image.new("RGB", (len(fg_colors), 1))
    fg_strip.putdata(fg_colors)
    quantized = fg_strip.quantize(colors=intermediate_colors, method=Image.Quantize.MEDIANCUT)
    q_palette = quantized.getpalette() or []
    q_pixels = quantized.load()

    n_colors = min(intermediate_colors, len(q_palette) // 3)
    palette_rgb = [tuple(q_palette[i * 3 : i * 3 + 3]) for i in range(n_colors)]

    # Per-intermediate-color pixel counts, needed by the foreground
    # blend-line merge rule (see _merge_near_duplicate_colors) to tell a
    # thin genuine boundary-blend sliver between two design colors apart
    # from a real, separate, minor design color.
    intermediate_counts = [0] * n_colors
    for i in range(len(fg_colors)):
        intermediate_counts[q_pixels[i, 0]] += 1

    merged_palette_rgb, remap = _merge_near_duplicate_colors(
        palette_rgb,
        color_merge_tolerance,
        background_refs=tuple(bg_reference),
        blend_tolerance=background_blend_tolerance,
        counts=tuple(intermediate_counts),
    )

    if len(merged_palette_rgb) > max_colors:
        merged_counts = [0] * len(merged_palette_rgb)
        for i in range(len(fg_colors)):
            merged_counts[remap[q_pixels[i, 0]]] += 1
        merged_palette_rgb, reduce_remap = _reduce_to_max_colors(merged_palette_rgb, merged_counts, max_colors)
        remap = [reduce_remap[merged_idx] for merged_idx in remap]

    palette_hex = ["#%02x%02x%02x" % color for color in merged_palette_rgb]

    for i, (x, y) in enumerate(fg_coords):
        label_grid[y][x] = remap[q_pixels[i, 0]]

    return palette_hex, label_grid


def despeckle_label_grid(label_grid: list[list[int]], min_component_px: int = 4) -> list[list[int]]:
    """Finds every 4-connected same-label component in `label_grid` and
    absorbs any component smaller than `min_component_px` pixels into
    whichever *other* label is most common immediately outside its border
    -- i.e. "this handful of pixels is noise sitting inside/against a
    bigger region, not its own real design feature."

    Why this exists in addition to `label_grid_to_ring_groups`'
    `min_area_px`/`min_bbox_mm` filters: those filters *discard* a small
    region outright (it just never becomes a stitch object), which is
    correct for genuine isolated noise specks sitting on a background, but
    wrong for a photo-derived label grid's dominant failure mode --
    dozens of tiny same-color islands scattered *inside* a real, larger
    foreground shape (a close-up product photo's per-thread
    highlight/shadow texture, surviving even after `artworkify`'s denoise
    and posterize). Discarding those leaves holes; this instead folds them
    into whatever real region actually surrounds them, so a macro photo's
    texture noise disappears into its enclosing shape rather than either
    staying as its own tiny stitch object or punching a hole in it.
    `BACKGROUND_LABEL` participates like any other label on both sides of
    this (a tiny background speck inside a foreground shape, or vice
    versa, gets absorbed the same way).

    Benchmarked end-to-end (the real `image_to_design` pipeline, counting
    actual surviving `design.threads`, not a palette-size proxy -- see
    session notes) across this project's full 8-asset photographed demo
    library: `min_component_px=4` is the largest value confirmed to never
    cost a single asset's real color count (a larger value like 6 traded
    away real colors on two separate assets, which this project's
    standing accuracy-first priority rules out). At 4 it gives a modest
    but real (~6%) reduction in total stitch-object count with zero
    accuracy cost -- a conservative, safe default rather than the more
    dramatic (but accuracy-costing) reduction a larger threshold would
    give. Pure-Python BFS (no numpy/scipy dependency needed) -- fast
    enough at this pipeline's working resolution (well under a second).
    """
    h = len(label_grid)
    w = len(label_grid[0]) if h else 0
    if h == 0 or w == 0:
        return label_grid

    visited = [[False] * w for _ in range(h)]
    grid = [row[:] for row in label_grid]

    for y0 in range(h):
        for x0 in range(w):
            if visited[y0][x0]:
                continue
            lbl = grid[y0][x0]
            component: list[tuple[int, int]] = []
            border_counts: dict[int, int] = {}
            queue: deque[tuple[int, int]] = deque([(x0, y0)])
            visited[y0][x0] = True
            while queue:
                x, y = queue.popleft()
                component.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if grid[ny][nx] == lbl:
                            if not visited[ny][nx]:
                                visited[ny][nx] = True
                                queue.append((nx, ny))
                        else:
                            border_counts[grid[ny][nx]] = border_counts.get(grid[ny][nx], 0) + 1
            if len(component) < min_component_px and border_counts:
                majority_label = max(border_counts, key=border_counts.get)
                for x, y in component:
                    grid[y][x] = majority_label

    return grid
