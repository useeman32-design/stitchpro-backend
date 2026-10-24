"""
Background / surrounding-material removal via border-connected flood fill
(brief section 14/15 follow-up: "a user can photograph a design on a real
garment and we should extract only the design, never the fabric/material
behind it").

Why this replaces a single-global-color distance test
-------------------------------------------------------
The original approach (see color_quantize.py's git history) detected *one*
representative background RGB value (the dominant color across the image
border) and then classified any pixel within a fixed distance of that one
value as background. That works for a flat, perfectly uniform background
(a plain studio backdrop, a solid-fill canvas) but falls apart on exactly
the use case this module exists for: a *real photo* of a design already
stitched onto a real garment. Real fabric has lighting falloff across the
frame, folds, wrinkles, and seam shadows -- its color can legitimately
range well outside any single fixed tolerance from corner to corner, even
though every one of those pixels is still "just the shirt", not the design.

Three passes, not one
----------------------
1. **Connectivity pass** (`_flood_fill_connected_background`): every border
   pixel is background by definition (a real "extract the logo from a
   photo" shot has the design fully inset from the frame edges, with
   surrounding material visible all the way around it -- the one
   assumption this whole technique rests on; this is also the one
   fundamental limit classical border-relative segmentation always has, no
   different from e.g. GrabCut's own background-seeding assumption), plus
   any fully-transparent pixel anywhere. Background status then propagates
   inward, but *only* across small *local* (neighbor-to-neighbor) color
   jumps -- letting it walk across a wide *gradual* range spanning the
   whole frame (shirt fabric lighting falloff, a soft fold's shading) while
   still stopping dead at a genuine edge, the much sharper color jump where
   the stitched design actually begins.
2. **Tunnel-rejection pass** (`_reject_color_tunnels`): a real photo's
   design-to-background edge is anti-aliased/resized/JPEG-blurred, not a
   razor-sharp single-pixel step. On a shallow part of that edge (where the
   boundary runs close to diagonal relative to the pixel grid) the blend
   band can be several pixels wide, each individual step small enough to
   sneak under `step_tolerance` -- which lets the flood fill "tunnel"
   straight through the blend and spill into the solid interior of a
   same-ish-toned foreground region (verified with a real test case: a
   grayish-blue photo background right next to a blue badge). To catch
   this, every background-classified pixel is re-checked against a
   reference palette built *only* from the unambiguous seed pixels (the
   image border and any transparent pixels -- never anything the flood
   fill itself inferred, so this check can't be fooled by its own mistake).
   A flood-reached pixel whose color doesn't actually resemble any of those
   trusted seed colors gets reclassified back to foreground.
3. **Enclosed-hole pass** (`_classify_enclosed_holes`): an interior region
   that is *not* border-connected (a letter's counter -- the hole in an O,
   A, B, D, P, Q, R, 8, @ -- or any fully enclosed cutout) but whose color
   closely matches the background's own color palette is still correctly
   treated as background/a hole, not a new foreground thread color. This is
   what lets a real letterform's counter stay an unstitched hole instead of
   turning into its own (wrong) "foreground region" just because it happens
   not to touch the image border. Reuses the same trusted seed-only
   reference palette as pass 2.

A light morphological "closing" pass (dilate-then-erode on the resulting
binary mask, implemented with Pillow's own MaxFilter/MinFilter -- no
OpenCV/scikit-image needed, matching this project's existing dependency
stance) then smooths over small jagged flood-fill tendrils and 1-2px
anti-aliased-edge gaps before the mask is used for anything downstream.
"""

from __future__ import annotations

from collections import deque

from PIL import Image, ImageFilter

Rgb = tuple[int, int, int]


def _dist2(a: Rgb, b: Rgb) -> int:
    """Squared RGB distance -- avoids a sqrt call on every one of the
    (up to) hundreds of thousands of neighbor comparisons a flood fill over
    a 400x400 image can make."""
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _flood_fill_connected_background(
    img: Image.Image, alpha_threshold: int, step_tolerance: float
) -> tuple[list[list[bool]], list[list[bool]], list[Rgb]]:
    """Returns `(visited, trusted, seed_colors)`:
    - `visited[y][x]`: True if the pixel is background by connectivity.
    - `trusted[y][x]`: True only for the *unambiguous* seed pixels this
      pass started from (the image border, and any transparent pixel) --
      never for a pixel the flood fill merely grew into. Used downstream by
      `_reject_color_tunnels` to tell "definitely background" apart from
      "background according to a chain of local color-distance guesses
      that could have tunneled through a blended edge".
    - `seed_colors`: the RGB colors of the trusted *opaque* border seed
      pixels (a transparent pixel's RGB can be arbitrary/meaningless, so it
      never contributes a color sample) -- the raw material for building a
      reference palette of what the real background actually looks like.
    """
    w, h = img.size
    px = img.load()
    step_tol2 = step_tolerance * step_tolerance

    visited = [[False] * w for _ in range(h)]
    trusted = [[False] * w for _ in range(h)]
    seed_colors: list[Rgb] = []
    dq: deque[tuple[int, int]] = deque()

    def seed(x: int, y: int, *, is_trusted: bool = False, color: Rgb | None = None) -> None:
        if not visited[y][x]:
            visited[y][x] = True
            dq.append((x, y))
        if is_trusted:
            trusted[y][x] = True
        if color is not None:
            seed_colors.append(color)

    def border_seed(x: int, y: int) -> None:
        r, g, b, a = px[x, y]
        seed(x, y, is_trusted=True, color=(r, g, b) if a >= alpha_threshold else None)

    for x in range(w):
        border_seed(x, 0)
        border_seed(x, h - 1)
    for y in range(h):
        border_seed(0, y)
        border_seed(w - 1, y)

    for y in range(h):
        for x in range(w):
            if px[x, y][3] < alpha_threshold:
                seed(x, y, is_trusted=True)

    while dq:
        x, y = dq.popleft()
        r, g, b, _a = px[x, y]
        cur = (r, g, b)
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= w or ny >= h or visited[ny][nx]:
                continue
            nr, ng, nb, na = px[nx, ny]
            if na < alpha_threshold or _dist2((nr, ng, nb), cur) <= step_tol2:
                seed(nx, ny)

    return visited, trusted, seed_colors


def _reject_color_tunnels(
    img: Image.Image,
    mask: list[list[bool]],
    trusted: list[list[bool]],
    reference: list[Rgb],
    alpha_threshold: int,
    tunnel_tolerance: float,
) -> list[list[bool]]:
    """Un-marks any background-classified pixel that isn't actually trusted
    (i.e. was only reached via the flood fill's local-step chain, not a
    direct border/transparency seed) and whose color doesn't resemble any
    of the trusted reference colors -- see module docstring's "tunnel-
    rejection pass". Opaque trusted seeds are always kept regardless of
    color (they're background by definition, not by color match)."""
    if not reference:
        return mask
    w, h = img.size
    px = img.load()
    tol2 = tunnel_tolerance * tunnel_tolerance
    out = [row[:] for row in mask]
    for y in range(h):
        for x in range(w):
            if not out[y][x] or trusted[y][x]:
                continue
            r, g, b, a = px[x, y]
            if a < alpha_threshold:
                continue  # transparent, trust regardless of (meaningless) color
            if not any(_dist2((r, g, b), ref) <= tol2 for ref in reference):
                out[y][x] = False
    return out


def _representative_colors(colors: list[Rgb], max_colors: int) -> list[Rgb]:
    """Reduces `colors` (can be many thousands of samples from a gradient
    background) down to at most `max_colors` representative swatches, so
    the enclosed-hole pass below can cheaply check each interior pixel
    against a handful of reference colors instead of the whole background
    pixel population."""
    if not colors:
        return []
    if len(colors) <= max_colors:
        return list(dict.fromkeys(colors))
    strip = Image.new("RGB", (len(colors), 1))
    strip.putdata(colors)
    quantized = strip.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    n = min(max_colors, len(palette) // 3)
    return [(palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]) for i in range(n)]


def compute_background_mask(
    rgba: Image.Image,
    alpha_threshold: int = 16,
    step_tolerance: float = 30.0,
    hole_tolerance: float = 28.0,
    tunnel_tolerance: float = 45.0,
    closing_radius: int = 1,
) -> list[list[bool]]:
    """Returns a `[h][w]` grid of `True` = background / surrounding
    material (or an enclosed hole matching it), `False` = design
    foreground, for `rgba` (any PIL mode -- converted to RGBA internally).

    Thin wrapper around `compute_background_mask_and_reference` for callers
    that only need the mask -- see that function for the full parameter
    docs and for callers (color_quantize.py) that also need the detected
    background's own reference colors, e.g. to recognize an anti-aliased
    edge pixel as "a blend of some foreground color toward the background"
    rather than a new color in its own right.
    """
    mask, _reference = compute_background_mask_and_reference(
        rgba,
        alpha_threshold=alpha_threshold,
        step_tolerance=step_tolerance,
        hole_tolerance=hole_tolerance,
        tunnel_tolerance=tunnel_tolerance,
        closing_radius=closing_radius,
    )
    return mask


def compute_background_mask_and_reference(
    rgba: Image.Image,
    alpha_threshold: int = 16,
    step_tolerance: float = 30.0,
    hole_tolerance: float = 28.0,
    tunnel_tolerance: float = 45.0,
    closing_radius: int = 1,
) -> tuple[list[list[bool]], list[Rgb]]:
    """Same background-detection pipeline as `compute_background_mask`, but
    also returns the trusted reference palette (the image border/
    transparent seed pixels' own representative colors) it built along the
    way -- color_quantize.py's anti-aliasing-aware merge step needs this to
    recognize a graduated edge blend (text/logo color fading toward the
    *actual detected* background, not just "toward white") as one design
    color, not several.

    `step_tolerance` is a *local* (neighbor-to-neighbor) RGB-distance
    budget for the border-connectivity flood fill. `tunnel_tolerance` is a
    looser distance-to-trusted-seed-color budget used to catch and reject
    cases where that flood fill tunneled through a blended/anti-aliased
    edge into a solid foreground region it shouldn't have reached (see
    module docstring's tunnel-rejection pass) -- intentionally larger than
    `hole_tolerance` since it only needs to catch *obviously* wrong
    excursions, not finely discriminate. `hole_tolerance` is a plain
    distance-to-reference-color budget used only for pixels the flood fill
    couldn't reach at all, to catch enclosed background-colored holes.
    """

    img = rgba.convert("RGBA")
    w, h = img.size

    connected, trusted, seed_colors = _flood_fill_connected_background(img, alpha_threshold, step_tolerance)
    reference = _representative_colors(seed_colors, max_colors=16)

    mask = _reject_color_tunnels(img, connected, trusted, reference, alpha_threshold, tunnel_tolerance)

    if reference:
        px = img.load()
        tol2 = hole_tolerance * hole_tolerance
        for y in range(h):
            for x in range(w):
                if mask[y][x]:
                    continue
                r, g, b, a = px[x, y]
                if any(_dist2((r, g, b), ref) <= tol2 for ref in reference):
                    mask[y][x] = True

    if closing_radius > 0:
        mask = _morphological_close(mask, w, h, closing_radius)

    return mask, reference


def _morphological_close(mask: list[list[bool]], w: int, h: int, radius: int) -> list[list[bool]]:
    """Dilate then erode the background mask (standard "closing") to merge
    small jagged flood-fill tendrils and 1-2px anti-aliased-edge gaps into
    a cleaner boundary, using Pillow's own MaxFilter/MinFilter -- both
    operate on a single-band image, no OpenCV/numpy required."""
    size = radius * 2 + 1
    im = Image.new("L", (w, h))
    im.putdata([255 if mask[y][x] else 0 for y in range(h) for x in range(w)])
    closed = im.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    data = list(closed.get_flattened_data())
    return [[data[y * w + x] > 127 for x in range(w)] for y in range(h)]
