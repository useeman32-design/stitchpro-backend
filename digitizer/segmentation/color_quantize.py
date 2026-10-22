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

    # Image.quantize() needs an RGB/P source; flatten onto white first for
    # quantization purposes only (true transparency is handled separately
    # below via the alpha channel, not lost here).
    rgb_flat = Image.new("RGB", (w, h), (255, 255, 255))
    rgb_flat.paste(rgba, mask=rgba.split()[3])
    quantized = rgb_flat.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    q_palette = quantized.getpalette() or []
    q_pixels = quantized.load()
    rgba_pixels = rgba.load()

    n_colors = min(max_colors, len(q_palette) // 3)
    palette_hex = [
        "#%02x%02x%02x" % tuple(q_palette[i * 3 : i * 3 + 3]) for i in range(n_colors)
    ]

    label_grid: list[list[int]] = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            r, g, b, a = rgba_pixels[x, y]
            if a < alpha_threshold or _dist((r, g, b), bg_rgb) < bg_tolerance:
                label_grid[y][x] = BACKGROUND_LABEL
            else:
                label_grid[y][x] = q_pixels[x, y]

    return palette_hex, label_grid
