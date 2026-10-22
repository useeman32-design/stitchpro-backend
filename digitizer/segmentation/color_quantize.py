"""
Color quantization + background detection (Milestone 5, brief section
14/15: "image -> segmentation").

This is the first real step of server-side digitizing: turn an arbitrary
raster image into a small palette of "thread colors" plus a per-pixel label
grid saying which palette color (or "background") each pixel belongs to.
Everything downstream (digitizer/segmentation/polygons.py) works off that
label grid, not the raw image.

Deliberately uses only Pillow (already a dependency) -- no OpenCV/
scikit-image, matching the conservative dependency stance from
docs/embroidery-engine-research.md until the cPanel-vs-VPS hosting question
is settled (OpenCV's wheel size and native dependencies are a much bigger
ask for shared hosting than Pillow).
"""

from __future__ import annotations

import math

from PIL import Image

BACKGROUND_LABEL = -1

Rgb = tuple[int, int, int]


def _hex_to_rgb(h: str) -> Rgb:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _dist(a: Rgb, b: Rgb) -> float:
    return math.sqrt(sum((ac - bc) ** 2 for ac, bc in zip(a, b)))


def _detect_background_rgb(rgba: Image.Image) -> Rgb:
    """Find the dominant flat background color by sampling the whole image
    border (not just the 4 corner pixels -- a single stripe or shape that
    happens to touch one corner would otherwise throw the whole detection
    off) and taking the average of whichever color cluster covers the most
    of that perimeter. Real digitizing artwork (a logo/graphic on a plain
    surrounding color, per this pipeline's intended use -- see
    digitizer/pipeline.py) has its background covering most of the border
    even if a shape happens to touch one edge or corner."""
    w, h = rgba.size
    px = rgba.load()
    border: list[Rgb] = []
    for x in range(w):
        border.append(px[x, 0][:3])
        border.append(px[x, h - 1][:3])
    for y in range(h):
        border.append(px[0, y][:3])
        border.append(px[w - 1, y][:3])

    clusters: list[list[Rgb]] = []
    for c in border:
        for cluster in clusters:
            if _dist(c, cluster[0]) < 20:
                cluster.append(c)
                break
        else:
            clusters.append([c])

    largest = max(clusters, key=len)
    r = sum(c[0] for c in largest) // len(largest)
    g = sum(c[1] for c in largest) // len(largest)
    b = sum(c[2] for c in largest) // len(largest)
    return (r, g, b)


def build_label_grid(
    img: Image.Image,
    max_colors: int = 6,
    alpha_threshold: int = 16,
    bg_tolerance: float = 24.0,
) -> tuple[list[str], list[list[int]]]:
    """Reduce `img` (any PIL mode, already downscaled by the caller) to at
    most `max_colors` hex colors plus a label grid.

    Returns `(palette_hex, label_grid)`:
      - `palette_hex[i]` is the hex color for label `i`.
      - `label_grid[y][x]` is either an index into `palette_hex`, or
        `BACKGROUND_LABEL` for a pixel that is transparent (alpha below
        `alpha_threshold`) or close to the detected flat background color
        (within `bg_tolerance` in RGB Euclidean distance).

    A pixel's own quantized color is used if neither background condition
    applies, even if that quantized bucket is the one closest to the overall
    background color elsewhere in the image -- only pixels *individually*
    close to the background are excluded, which correctly keeps a shape
    whose fill color happens to be similar to (but distinct from) the
    background, while still dropping true background/anti-aliased-edge
    pixels in the halo around real shapes.
    """
    rgba = img.convert("RGBA")
    w, h = rgba.size
    bg_rgb = _detect_background_rgb(rgba)
    rgba_pixels = rgba.load()

    # Classify background *before* quantizing, using the same alpha/distance
    # rule the function has always used for its final label grid. This
    # matters because MEDIANCUT otherwise sees the whole image -- including
    # the halo of near-background anti-aliased edge pixels that downscaling
    # (e.g. pipeline.py's LANCZOS resize) produces around every shape -- and
    # can waste most or all of a small `max_colors` budget splitting hairs
    # between many subtly-different near-white/near-background shades,
    # instead of allocating palette slots to the real, visually distinct
    # foreground colors. Excluding background pixels from the quantizer's
    # input entirely fixes that: only genuine foreground pixels compete for
    # palette slots.
    fg_coords: list[tuple[int, int]] = []
    fg_colors: list[Rgb] = []
    for y in range(h):
        for x in range(w):
            r, g, b, a = rgba_pixels[x, y]
            if a < alpha_threshold or _dist((r, g, b), bg_rgb) < bg_tolerance:
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
    palette_hex = [
        "#%02x%02x%02x" % tuple(q_palette[i * 3 : i * 3 + 3]) for i in range(n_colors)
    ]

    for i, (x, y) in enumerate(fg_coords):
        label_grid[y][x] = q_pixels[i, 0]

    return palette_hex, label_grid
