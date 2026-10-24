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

from PIL import Image

from digitizer.segmentation.background_removal import compute_background_mask

BACKGROUND_LABEL = -1

Rgb = tuple[int, int, int]


def _dist2(a: Rgb, b: Rgb) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _merge_near_duplicate_colors(
    palette_rgb: list[Rgb], merge_tolerance: float
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

    This greedily unions any two palette entries within `merge_tolerance`
    of each other (transitively, via a small union-find) and returns the
    deduplicated palette plus an `old_index -> new_index` remap list.
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
    for i in range(n):
        for j in range(i + 1, n):
            if _dist2(palette_rgb[i], palette_rgb[j]) <= tol2:
                union(i, j)

    root_to_new: dict[int, int] = {}
    new_palette: list[Rgb] = []
    remap = [0] * n
    for i in range(n):
        root = find(i)
        if root not in root_to_new:
            # `root` is itself a valid palette index (union-find roots are
            # drawn from the same 0..n-1 index space), so its own color is
            # a perfectly good, deterministic representative for the group
            # -- no need to search for one.
            root_to_new[root] = len(new_palette)
            new_palette.append(palette_rgb[root])
        remap[i] = root_to_new[root]
    return new_palette, remap


def build_label_grid(
    img: Image.Image,
    max_colors: int = 6,
    alpha_threshold: int = 16,
    bg_step_tolerance: float = 30.0,
    color_merge_tolerance: float = 18.0,
    remove_background: bool = True,
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

    `remove_background=False` skips the flood fill entirely (every opaque
    pixel is treated as foreground, matching exactly what the
    `/api/design/remove-background` preview endpoint showed the user when
    they chose "keep original" instead of proceeding with the background
    removed) -- see digitizer/pipeline.py's `image_to_design` docstring for
    the end-to-end contract this flag preserves between preview and final
    digitized output.
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    rgba_pixels = rgba.load()

    if remove_background:
        bg_mask = compute_background_mask(rgba, alpha_threshold=alpha_threshold, step_tolerance=bg_step_tolerance)
    else:
        bg_mask = [[False] * w for _ in range(h)]


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
    # palette generation, only the multiset of colors does.
    fg_strip = Image.new("RGB", (len(fg_colors), 1))
    fg_strip.putdata(fg_colors)
    quantized = fg_strip.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    q_palette = quantized.getpalette() or []
    q_pixels = quantized.load()

    n_colors = min(max_colors, len(q_palette) // 3)
    palette_rgb = [tuple(q_palette[i * 3 : i * 3 + 3]) for i in range(n_colors)]

    merged_palette_rgb, remap = _merge_near_duplicate_colors(palette_rgb, color_merge_tolerance)
    palette_hex = ["#%02x%02x%02x" % color for color in merged_palette_rgb]

    for i, (x, y) in enumerate(fg_coords):
        label_grid[y][x] = remap[q_pixels[i, 0]]

    return palette_hex, label_grid

