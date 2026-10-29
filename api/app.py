"""
StitchPro backend API — Flask (WSGI), chosen specifically for cPanel shared
hosting compatibility (Passenger only speaks WSGI, not ASGI). See
docs/embroidery-engine-research.md section 3 for the reasoning.

Milestone 1 exposed just enough to prove the mobile app could realistically
talk to this over HTTP (a point-array-only test endpoint). This milestone
adds the first real integration surface between the stitch-pro frontend
(which already does real client-side artwork analysis + stitch-plan math —
see stitch-pro's src/utils/stitchEngine.ts + stitchPlanner.ts) and this
Python engine, whose pyembroidery dependency is the one thing the frontend
genuinely cannot do itself: encode real, byte-accurate multi-format
embroidery files (DST/PES/JEF/VP3/EXP) from that already-computed stitch
plan. See formats/points_writer.py for why this is a deliberately separate
path from the EmbroideryDesign/ObjectType pipeline used by future
server-side digitizing milestones (satin, fill, real segmentation).

  GET  /api/health                  -> liveness check
  GET  /api/formats                 -> list of supported export format ids
  POST /api/design/line-to-dst      -> legacy Milestone-1 test endpoint,
                                        {"points": [[x,y], ...], ...} -> .dst
  POST /api/design/points-to-file   -> real integration endpoint, takes a
                                        complete point stream (as produced by
                                        stitch-pro's own stitchPlanner.ts) and
                                        returns a real file in any supported
                                        format. See request/response shape
                                        in the route's docstring below.
  POST /api/design/image-to-file    -> Milestone 5: real server-side
                                        digitizing. Takes a raw uploaded
                                        image and runs the whole
                                        segmentation -> objects -> stitches
                                        -> file pipeline itself
                                        (digitizer/pipeline.py), unlike the
                                        two routes above which only compile
                                        an already-digitized plan.
  POST /api/design/text-to-file     -> Dedicated embroidery lettering
                                        engine (digitizer/lettering):
                                        builds real satin-column
                                        letterforms directly from typed
                                        text, no image/classification step
                                        at all. See its own docstring.
  POST /api/design/text-to-points   -> Same lettering engine as above, JSON
                                        point-stream response for preview
                                        rendering (matches image-to-points).
  POST /api/design/vectorize        -> Real editable-vector pipeline: runs
                                        classification/background-removal/
                                        quantization/raster-to-polygon
                                        tracing (everything image-to-* does,
                                        up to but not including stitch
                                        generation) and returns the traced
                                        regions as plain JSON vector shapes,
                                        for a client to let the user review/
                                        edit (select, delete, redraw) before
                                        any stitches are generated. Also
                                        accepts manual background-mask
                                        correction brush strokes.
  POST /api/design/regions-to-points -> Builds a real digitized design from
                                        a (possibly user-edited) region list
                                        -- normally /api/design/vectorize's
                                        own output -- and returns the same
                                        JSON point-stream contract as
                                        /api/design/image-to-points.
  POST /api/design/regions-to-file  -> Same as regions-to-points but
                                        returns a real machine file, like
                                        /api/design/image-to-file.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, Response, jsonify, request, send_file

from digitizer.classifier.classify_artwork import classify_artwork
from digitizer.lettering.alphabet_block import glyph_for, is_fully_supported
from digitizer.lettering.engine import build_text_design
from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.pipeline import (
    MAX_MAX_DIM,
    MIN_MAX_DIM,
    PREVIEW_MAX_DIM,
    PREVIEW_MAX_MAX_DIM,
    PREVIEW_MIN_MAX_DIM,
    STITCH_STYLES,
    design_from_region_records,
    image_to_design,
    image_to_vector_regions,
    load_and_downscale,
    remove_background_preview,
)
from digitizer.segmentation.design_region import detect_design_region
from digitizer.points_export import design_to_points
from formats.dst.writer import design_to_dst_bytes
from formats.writer import SUPPORTED_FORMATS, design_to_format_bytes, points_to_format_bytes

app = Flask(__name__)

# CORS: this API is meant to be called directly from a browser-hosted
# frontend served from a different origin (e.g. GitHub Pages), which a
# cPanel-hosted Flask app under Passenger has no other way to allow. Kept
# wide open (`*`) for now since there's no user auth/session state yet to
# protect; tighten to the real production origin once one is fixed (see
# continue.md "Known pitfalls" for the current deployment status).
@app.after_request
def _add_cors_headers(resp: Response) -> Response:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "stitchpro-backend", "milestone": 5})


@app.get("/api/formats")
def formats():
    return jsonify({"formats": list(SUPPORTED_FORMATS)})


@app.post("/api/design/line-to-dst")
def line_to_dst():
    body = request.get_json(force=True, silent=False)
    points = [tuple(p) for p in body["points"]]
    stitch_length_mm = float(body.get("stitch_length_mm", 3.0))
    width_mm = float(body.get("width_mm", 100))
    height_mm = float(body.get("height_mm", 100))
    color = body.get("color", "#111111")

    design = EmbroideryDesign(
        width_mm=width_mm,
        height_mm=height_mm,
        objects=[
            EmbroideryObject(
                id="line-1",
                type=ObjectType.RUNNING,
                geometry=points,
                color=color,
                stitch_settings=StitchSettings(stitch_length_mm=stitch_length_mm),
                sequence=0,
            )
        ],
        threads=[color],
    )
    dst_bytes = design_to_dst_bytes(design)
    return send_file(
        io.BytesIO(dst_bytes),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name="design.dst",
    )


@app.post("/api/design/points-to-file")
def points_to_file():
    """
    Request JSON:
      {
        "points": [{"x": <0.1mm int>, "y": <0.1mm int>,
                    "command": "jump" | "stitch" | "colorChange" | "stop" | "end"}, ...],
        "colors": ["#RRGGBB", ...],   // optional, one per color block, in order
        "format": "dst" | "pes" | "jef" | "vp3" | "exp",   // default "dst"
        "name": "my-design"            // optional, used for the download filename
      }

    `points` must already be a complete, physical stitch plan in 0.1mm
    absolute units, screen-style (+Y down) — exactly what stitch-pro's own
    DstPoint[]/dstWriter.ts works with (see formats/points_writer.py for the
    coordinate-convention note). Nothing here re-digitizes the artwork; this
    endpoint's only job is turning an already-correct stitch plan into a
    real, byte-accurate file in whichever format was requested — the one
    thing pyembroidery can do that pure client-side JS cannot.

    Returns the file as a binary attachment download on success, or
    {"error": "..."} with a 400 status on a malformed/invalid request.
    """
    body = request.get_json(force=True, silent=False)
    if not isinstance(body, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400

    raw_points = body.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        return jsonify({"error": "'points' must be a non-empty array"}), 400

    fmt = str(body.get("format", "dst")).lower()
    if fmt not in SUPPORTED_FORMATS:
        return jsonify({"error": f"unsupported format {fmt!r}", "supported": list(SUPPORTED_FORMATS)}), 400

    colors = body.get("colors")
    if colors is not None and not isinstance(colors, list):
        return jsonify({"error": "'colors' must be an array of hex strings if provided"}), 400

    name = str(body.get("name") or "design")
    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "design"

    try:
        file_bytes = points_to_format_bytes(raw_points, fmt, colors)
    except (ValueError, KeyError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    return send_file(
        io.BytesIO(file_bytes),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"{safe_name}.{fmt}",
    )


def _digitize_image_from_request() -> tuple[EmbroideryDesign | None, tuple | None]:
    """Shared request parsing + digitizing for both image-to-file and
    image-to-points: validates the multipart form, runs `image_to_design`,
    and returns either `(design, None)` on success or `(None, (response,
    status))` on any validation/digitizing failure -- callers just need to
    `return err` when `err is not None`."""
    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return None, (jsonify({"error": "'image' file field is required"}), 400)

    form = request.form

    def _float_field(key: str, default: float) -> float:
        raw = form.get(key)
        return default if raw in (None, "") else float(raw)

    def _optional_float_field(key: str) -> float | None:
        raw = form.get(key)
        return None if raw in (None, "") else float(raw)

    def _int_field(key: str, default: int) -> int:
        raw = form.get(key)
        return default if raw in (None, "") else int(raw)

    def _bool_field(key: str, default: bool) -> bool:
        raw = form.get(key)
        return default if raw in (None, "") else str(raw).strip().lower() not in ("0", "false", "no")

    try:
        width_mm = _float_field("width_mm", 0)
        height_mm = _float_field("height_mm", 0)
        if width_mm <= 0 or height_mm <= 0:
            return None, (jsonify({"error": "'width_mm' and 'height_mm' are required and must be positive"}), 400)

        max_colors = _int_field("max_colors", 6)
        stitch_length_mm = _float_field("stitch_length_mm", 3.0)
        density_mm = _optional_float_field("density_mm")
        border_width_mm = _optional_float_field("border_width_mm")
        angle_deg = _float_field("angle_deg", 0.0)
        smooth_curves = _bool_field("smooth_curves", True)
        remove_background = _bool_field("remove_background", True)
        auto_vectorize = _bool_field("auto_vectorize", True)
        force_photo = _bool_field("force_photo", False)
        stitch_style = str(form.get("stitch_style") or "fill").strip().lower()
        if stitch_style not in STITCH_STYLES:
            return None, (jsonify({"error": f"'stitch_style' must be one of {list(STITCH_STYLES)}"}), 400)

        raw_crop = form.get("crop")
        crop = None
        if raw_crop not in (None, ""):
            try:
                crop = json.loads(raw_crop)
            except (TypeError, ValueError):
                return None, (jsonify({"error": "'crop' must be valid JSON"}), 400)
            if not isinstance(crop, dict) or not all(k in crop for k in ("x0", "y0", "x1", "y1")):
                return None, (jsonify({"error": "'crop' must be a JSON object with x0/y0/x1/y1"}), 400)

        # A caller-confirmed `crop` has already thrown away everything but
        # the actual design (the "extract only the design" requirement),
        # so the usual DEFAULT_MAX_DIM=150 -- sized to stay cheap when it's
        # still downscaling an *entire uncropped photo* -- is needlessly
        # low once there's only a small sub-image left to resize: see
        # digitizer.pipeline._MIN_FOREGROUND_AREA_MM2's own comment and
        # this project's session notes for the benchmarked before/after
        # (a cropped garment-photo logo going from unrecognizable to
        # clearly legible at MAX_MAX_DIM, at a near-identical ~0.6s/request
        # cost precisely because the crop already shrank the working area).
        # Only applies when the caller didn't already request a specific
        # max_dim of their own.
        default_max_dim = MAX_MAX_DIM if crop is not None else 150
        max_dim = _int_field("max_dim", default_max_dim)
        if not (MIN_MAX_DIM <= max_dim <= MAX_MAX_DIM):
            return None, (jsonify({"error": f"'max_dim' must be between {MIN_MAX_DIM} and {MAX_MAX_DIM}"}), 400)

        raw_crop = form.get("crop")
        crop = None
        if raw_crop not in (None, ""):
            try:
                crop = json.loads(raw_crop)
            except (TypeError, ValueError):
                return None, (jsonify({"error": "'crop' must be valid JSON"}), 400)
            if not isinstance(crop, dict) or not all(k in crop for k in ("x0", "y0", "x1", "y1")):
                return None, (jsonify({"error": "'crop' must be a JSON object with x0/y0/x1/y1"}), 400)
    except (TypeError, ValueError) as e:
        return None, (jsonify({"error": f"invalid numeric field: {e}"}), 400)

    try:
        design = image_to_design(
            image_file.read(),
            width_mm=width_mm,
            height_mm=height_mm,
            max_colors=max_colors,
            stitch_length_mm=stitch_length_mm,
            density_mm=density_mm,
            max_dim=max_dim,
            angle_deg=angle_deg,
            stitch_style=stitch_style,
            smooth_curves=smooth_curves,
            border_width_mm=border_width_mm,
            remove_background=remove_background,
            auto_vectorize=auto_vectorize,
            force_photo=force_photo,
            crop=crop,
        )
    except ValueError as e:
        return None, (jsonify({"error": str(e)}), 400)

    return design, None


@app.post("/api/design/image-to-file")
def image_to_file():
    """
    Milestone 5: real server-side digitizing. Unlike /api/design/points-to-
    file (which only compiles an already-digitized plan), this route does
    the actual image -> segmentation -> embroidery objects work itself --
    see digitizer/pipeline.py.

    Multipart form-data request:
      image           : required, the raw image file (PNG/JPEG/etc, whatever
                         Pillow can decode).
      width_mm        : required, target design width in mm.
      height_mm       : required, target design height in mm.
      format          : optional, one of SUPPORTED_FORMATS, default "dst".
      max_colors      : optional int, 1-16, default 6 -- how many distinct
                         thread colors to reduce the image to.
      stitch_length_mm: optional float, default 3.0.
      density_mm      : optional float -- FILL row spacing, or CROSS_STITCH
                         grid cell size (see `stitch_style`). Defaults to a
                         sensible value for whichever style is selected
                         (0.4mm fill rows, 1.8mm cross-stitch cells) if
                         omitted.
      angle_deg       : optional float, default 0.0 -- stitch direction for
                         FILL rows / CROSS_STITCH grid alignment.
      stitch_style    : optional, "fill" (default, tatami rows -- best for
                         solid-color logos/graphics) or "cross" (a grid of
                         X's -- better for a deliberately pixelated/sampler-
                         style design). Applies to every detected region in
                         the design.
      smooth_curves   : optional bool, default true -- adaptive corner-
                         preserving curve smoothing on detected region
                         boundaries (digitizer/geometry/smoothing.py).
                         Sharp corners are always preserved either way; set
                         to false only to get the raw simplified-polygon
                         boundary.
      remove_background: optional bool, default true -- removes the
                         detected background/surrounding material before
                         digitizing (see /api/design/remove-background and
                         digitizer/segmentation/background_removal.py).
                         Pass false only once the user has explicitly
                         chosen "keep original" on that preview step, so
                         the final design matches what they approved.
      auto_vectorize  : optional bool, default true -- classifies the image
                         as flat artwork or a photograph first
                         (digitizer/classifier/classify_artwork.py) and, for
                         a photograph, flattens it into clean flat-art-like
                         input before quantizing
                         (digitizer/segmentation/artworkify.py). This is
                         what actually fixes "a real photo digitizes badly"
                         -- the detected kind is in the response for
                         /api/design/image-to-points (not surfaced on this
                         binary-file endpoint). Pass false only to force the
                         old flat-art-only behavior.
      force_photo     : optional bool, default false -- skips the above
                         classification vote and always treats the image as
                         a photograph needing artworkify flattening. Set
                         this for the "extract a design from a photo of an
                         already-embroidered item" upload entry point,
                         where the user has already told us it's a photo
                         (see digitizer/pipeline.py's own docstring for
                         why an explicit signal beats a guess). Ignored
                         when auto_vectorize=false.
      max_dim         : optional int, caps the longest image side (in
                         pixels) actually processed, default 150 -- keeps
                         server-side processing cheap regardless of the
                         uploaded image's real resolution.
      crop            : optional JSON-encoded string: a normalized 0..1
                         {"x0", "y0", "x1", "y1"} box applied before
                         anything else runs -- see /api/design/
                         detect-region and digitizer.pipeline._apply_crop.
                         The "extract only the design from a photo of
                         someone wearing the garment" requirement; pass
                         only a box the user already confirmed/adjusted.
      name            : optional, used for the download filename.

    Returns the file as a binary attachment download on success, or
    {"error": "..."} with a 400 status on a malformed request, an unreadable
    image, or an image with no detectable foreground.
    """
    fmt = str(request.form.get("format", "dst")).lower()
    if fmt not in SUPPORTED_FORMATS:
        return jsonify({"error": f"unsupported format {fmt!r}", "supported": list(SUPPORTED_FORMATS)}), 400

    design, err = _digitize_image_from_request()
    if err is not None:
        return err

    name = str(request.form.get("name") or "design")
    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "design"

    try:
        file_bytes = design_to_format_bytes(design, fmt)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return send_file(
        io.BytesIO(file_bytes),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"{safe_name}.{fmt}",
    )


@app.post("/api/design/image-to-points")
def image_to_points():
    """
    Same digitizing pipeline as /api/design/image-to-file (Milestone 5), but
    returns the plain JSON point-stream contract
    (digitizer/points_export.py) instead of a binary file -- meant for
    client-side *preview rendering* (e.g. stitch-pro's Simulator screen),
    not for machine output. This lets any client (web or native -- neither
    needs a server-specific renderer) draw the exact same real,
    server-digitized stitch plan with its own renderer, using the identical
    {x, y, command} point contract the frontend already speaks for its own
    client-side-generated plans (see stitch-pro's src/utils/dstWriter.ts).

    Multipart form-data request: same fields as /api/design/image-to-file
    (image, width_mm, height_mm, max_colors, stitch_length_mm, density_mm,
    angle_deg, stitch_style, smooth_curves, max_dim, auto_vectorize), except
    there is no `format`/`name` field -- those only make sense for a file
    download.

    Response JSON on success:
      {"width_mm": ..., "height_mm": ...,
       "threads": ["#RRGGBB", ...],
       "points": [{"x": <0.1mm int>, "y": <0.1mm int>,
                   "command": "jump"|"stitch"|"colorChange"|"stop"}, ...],
       "stitchTypeBreakdown": [{"type": "SATIN"|"FILL"|"CROSS_STITCH"|"RUNNING"
                                 |"BEAN"|"MOTIF"|"PUFF_3D"|"APPLIQUE",
                                 "objectCount": int, "stitchCount": int}, ...],
       "artworkKind": "flat"|"photo"|"flat (forced)",
       "artworkConfidence": 0..1}

    `stitchTypeBreakdown` is purely informational (see
    digitizer/points_export.py's docstring) -- the auto-digitizer already
    classifies each region independently (thin_region.py decides FILL vs
    SATIN per-shape, e.g. thin strokes vs large fills), so a single design
    routinely mixes stitch types across its regions; this tells a client
    which types ended up used and how much of the design is each, sorted
    by descending stitch count, without needing to parse per-object detail
    out of the flat `points` stream.

    `artworkKind`/`artworkConfidence` come straight from
    `classify_artwork`/`auto_vectorize` (digitizer/pipeline.py) -- purely
    informational, e.g. for a client that wants to tell the user "detected
    as a photo, auto-converted to artwork" rather than silently changing
    behavior under them.

    Or {"error": "..."} with a 400 status on a malformed request, an
    unreadable image, or an image with no detectable foreground.
    """
    design, err = _digitize_image_from_request()
    if err is not None:
        return err

    try:
        result = design_to_points(design)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result["artworkKind"] = design.metadata.get("artwork_kind")
    result["artworkConfidence"] = design.metadata.get("artwork_confidence")
    return jsonify(result)


def _build_text_design_from_request() -> tuple[EmbroideryDesign | None, tuple | None]:
    """Shared request parsing for both text-to-file and text-to-points:
    validates the JSON body, builds a real EmbroideryDesign via the
    dedicated lettering engine (digitizer/lettering/engine.py -- satin-
    column letterforms, NOT the generic photo/artwork auto-digitizer), and
    returns either `(design, None)` on success or `(None, (response,
    status))` on a validation/engine error."""
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return None, (jsonify({"error": "request body must be a JSON object"}), 400)

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return None, (jsonify({"error": "'text' must be a non-empty string"}), 400)

    try:
        size_mm = float(body.get("size_mm", 20))
    except (TypeError, ValueError):
        return None, (jsonify({"error": "'size_mm' must be a number"}), 400)
    if not (3 <= size_mm <= 200):
        return None, (jsonify({"error": "'size_mm' must be between 3 and 200"}), 400)

    color = str(body.get("color") or "#111111")
    layout = str(body.get("layout") or "straight")

    if not is_fully_supported(text):
        unsupported = sorted({ch for ch in text if glyph_for(ch) is None})
        return None, (
            jsonify(
                {
                    "error": "text contains characters the dedicated lettering engine doesn't support yet",
                    "unsupportedCharacters": unsupported,
                }
            ),
            422,
        )

    try:
        design = build_text_design(text, size_mm=size_mm, color=color, layout=layout)
    except ValueError as e:
        return None, (jsonify({"error": str(e)}), 400)

    return design, None


@app.post("/api/design/text-to-file")
def text_to_file():
    """
    Real embroidery lettering, built from a dedicated satin-column
    alphabet (digitizer/lettering -- see its module docstrings for why
    this is a separate path from /api/design/image-to-file's generic
    photo/artwork auto-digitizer): every letter becomes a proper
    monoline-skeleton-derived satin column, including clean curves on
    letters like O/S/C/G, instead of whatever FILL/SATIN mix the generic
    per-region classifier would land on for a rasterized text image.

    Request JSON:
      {
        "text": "STITCHPRO",       // required; only characters in
                                    // digitizer/lettering/alphabet_block.py's
                                    // GLYPHS table are supported (A-Z,
                                    // 0-9, space, and a handful of common
                                    // punctuation) -- a 422 response lists
                                    // which characters aren't supported so
                                    // the caller can fall back to the
                                    // generic render-and-auto-digitize path
                                    // for this string.
        "size_mm": 20,              // optional, default 20 -- cap height
                                    // in mm, 3-200.
        "color": "#111111",         // optional hex thread color.
        "layout": "straight",       // optional, only "straight" is
                                    // supported so far -- "arc_up"/
                                    // "arc_down" still need the generic
                                    // pipeline (see engine.py docstring).
        "format": "dst",            // optional, one of SUPPORTED_FORMATS.
        "name": "my-text-design"    // optional, download filename.
      }

    Returns the file as a binary attachment download on success, a 422
    with {"error": ..., "unsupportedCharacters": [...]} if the text uses a
    character outside the Bold Block alphabet, or a 400 for any other
    malformed request.
    """
    body = request.get_json(force=True, silent=True) or {}
    fmt = str(body.get("format", "dst")).lower()
    if fmt not in SUPPORTED_FORMATS:
        return jsonify({"error": f"unsupported format {fmt!r}", "supported": list(SUPPORTED_FORMATS)}), 400

    design, err = _build_text_design_from_request()
    if err is not None:
        return err

    name = str(body.get("name") or "text-design")
    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "text-design"

    try:
        file_bytes = design_to_format_bytes(design, fmt)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return send_file(
        io.BytesIO(file_bytes),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"{safe_name}.{fmt}",
    )


@app.post("/api/design/text-to-points")
def text_to_points():
    """
    Same dedicated lettering engine as /api/design/text-to-file, but
    returns the plain JSON point-stream contract (digitizer/points_export.py)
    instead of a binary file -- for client-side preview rendering (e.g.
    stitch-pro's live text-design preview / Simulator), matching
    /api/design/image-to-points' response shape.

    Request JSON: same fields as /api/design/text-to-file except no
    `format`/`name` (those only make sense for a file download).

    Response JSON on success:
      {"width_mm": ..., "height_mm": ...,
       "threads": ["#RRGGBB", ...],
       "points": [{"x": <0.1mm int>, "y": <0.1mm int>,
                   "command": "jump"|"stitch"|"colorChange"}, ...],
       "stitchTypeBreakdown": [{"type": "SATIN", "objectCount": int,
                                 "stitchCount": int}]}

    Or a 422 with {"error": ..., "unsupportedCharacters": [...]} if the
    text uses a character outside the Bold Block alphabet, or a 400 for
    any other malformed request.
    """
    design, err = _build_text_design_from_request()
    if err is not None:
        return err

    try:
        result = design_to_points(design)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(result)


@app.post("/api/design/detect-region")
def detect_region():
    """
    Suggests a crop box isolating just the printed/stitched design within
    a photo -- brief: "when creating with [a] real world photo, example
    clothes on someone, it's supposed to detect it and extract only the
    design... but now it's just removing the background normally and
    trying to convert to vector which will give very bad result".

    Unlike /api/design/remove-background (which only strips *border-
    connected* surrounding material -- a studio backdrop, a flat-lay's
    table), a photo of someone *wearing* the garment has the shirt's own
    plain fabric (and skin/face/arms) filling most or all of the frame,
    none of which is "surrounding material" by that definition. This route
    runs digitizer.segmentation.design_region.detect_design_region, which
    looks *within* the foreground for the sub-region that actually looks
    like a design (more local color/edge detail than the large flat areas
    around it), and returns its bounding box as a plain JSON suggestion --
    never applied automatically. A client is expected to show this as an
    adjustable crop overlay the user confirms or resizes before anything
    is actually cropped, matching the same explicit-confirmation pattern
    /api/design/remove-background's own preview already establishes.

    Also runs the same flat-vs-photo classification
    (digitizer.classifier.classify_artwork) used elsewhere in this
    pipeline and includes it in the response: a clean flat-art logo export
    (not a photo at all) has no "garment vs. design" distinction to make,
    so a confidently-`'flat'`-classified image comes back with
    `"skip": true` and a harmless full-image box, signaling the client to
    skip showing this crop step entirely rather than risk cropping a
    flat design down to some arbitrary sub-region. Pass `force_photo=true`
    (same meaning as the digitize/vectorize endpoints) for an upload entry
    point that's already told us this is a photograph, to skip that
    classification vote and always run real detection.

    Multipart form-data request:
      image       : required, the raw image file (PNG/JPEG/etc).
      max_dim     : optional int, caps the longest processed side in
                    pixels before detection runs its own further internal
                    downsample, default 900 (digitizer.pipeline.
                    PREVIEW_MAX_DIM) -- detection quality is insensitive to
                    this beyond a few hundred px, it mainly exists so a
                    huge original upload doesn't get fully decoded/resized
                    twice at full resolution for no benefit.
      force_photo : optional bool, default false -- see above.

    Response JSON on success:
      {"x0", "y0", "x1", "y1": normalized 0..1 box,
       "confidence": 0..1,
       "skip": bool -- true when classified confidently 'flat' (see above),
       "artworkKind", "artworkConfidence": the classify_artwork result}

    Or {"error": "..."} with a 400 status on a missing/unreadable image or
    an out-of-range max_dim.
    """
    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return jsonify({"error": "'image' file field is required"}), 400

    raw_max_dim = request.form.get("max_dim")
    try:
        max_dim = PREVIEW_MAX_DIM if raw_max_dim in (None, "") else int(raw_max_dim)
    except (TypeError, ValueError):
        return jsonify({"error": "'max_dim' must be an integer"}), 400
    if not (PREVIEW_MIN_MAX_DIM <= max_dim <= PREVIEW_MAX_MAX_DIM):
        return (
            jsonify({"error": f"'max_dim' must be between {PREVIEW_MIN_MAX_DIM} and {PREVIEW_MAX_MAX_DIM}"}),
            400,
        )
    force_photo = str(request.form.get("force_photo", "")).strip().lower() not in ("", "0", "false", "no")

    try:
        img = load_and_downscale(image_file.read(), max_dim=max_dim)
    except Exception as e:  # noqa: BLE001 - Pillow raises several distinct error types
        return jsonify({"error": f"could not read image: {e}"}), 400

    if force_photo:
        artwork_kind, artwork_confidence = "photo (forced)", 1.0
    else:
        artwork_info = classify_artwork(img)
        artwork_kind, artwork_confidence = artwork_info["kind"], artwork_info["confidence"]

    is_confidently_flat = artwork_kind == "flat" and artwork_confidence >= 0.5
    if is_confidently_flat:
        return jsonify(
            {
                "x0": 0.0,
                "y0": 0.0,
                "x1": 1.0,
                "y1": 1.0,
                "confidence": 0.0,
                "skip": True,
                "artworkKind": artwork_kind,
                "artworkConfidence": artwork_confidence,
            }
        )

    region = detect_design_region(img)
    return jsonify(
        {
            "x0": region["x0"],
            "y0": region["y0"],
            "x1": region["x1"],
            "y1": region["y1"],
            "confidence": region["confidence"],
            "skip": False,
            "artworkKind": artwork_kind,
            "artworkConfidence": artwork_confidence,
        }
    )


@app.post("/api/design/remove-background")
def remove_background():
    """
    Standalone background-removal *preview* -- brief follow-up: "let a user
    see what will be extracted from their photo and require an explicit
    confirmation before digitizing, never silently apply it automatically".

    Unlike /api/design/image-to-file and /api/design/image-to-points, this
    route does no quantizing, segmentation, or stitch generation at all --
    it just runs the same background/surrounding-material detection those
    routes use internally (digitizer.segmentation.background_removal) and
    returns a real PNG with every detected background pixel's alpha channel
    zeroed out, at a much higher resolution than the digitizing pipeline's
    150px cap, since this image is only ever looked at by a human.

    Multipart form-data request:
      image   : required, the raw image file (PNG/JPEG/etc).
      max_dim : optional int, caps the longest processed side in pixels,
                default 900 (digitizer.pipeline.PREVIEW_MAX_DIM).
      crop    : optional JSON-encoded string: a normalized 0..1
                {"x0", "y0", "x1", "y1"} box applied before anything else
                runs -- see /api/design/detect-region and
                digitizer.pipeline._apply_crop. Lets a client preview the
                background mask against just the user-confirmed design
                region rather than the whole original photo, once that
                step has run.

    Returns the PNG directly (mimetype image/png, not a download
    attachment, so a client can drop it straight into an <img>/Image
    source) on success, or {"error": "..."} with a 400 status on a missing/
    unreadable image or an out-of-range max_dim.

    A client that calls this is expected to show the result to the user
    and let them choose between continuing with the background removed
    (pass `remove_background=true`, the default, to the digitize/export
    endpoints -- no need to re-upload the processed PNG, the server
    recomputes the identical mask) or keeping the original image untouched
    (pass `remove_background=false` to those same endpoints instead).
    """
    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return jsonify({"error": "'image' file field is required"}), 400

    raw_max_dim = request.form.get("max_dim")
    try:
        max_dim = PREVIEW_MAX_DIM if raw_max_dim in (None, "") else int(raw_max_dim)
    except (TypeError, ValueError):
        return jsonify({"error": "'max_dim' must be an integer"}), 400
    if not (PREVIEW_MIN_MAX_DIM <= max_dim <= PREVIEW_MAX_MAX_DIM):
        return (
            jsonify({"error": f"'max_dim' must be between {PREVIEW_MIN_MAX_DIM} and {PREVIEW_MAX_MAX_DIM}"}),
            400,
        )

    raw_crop = request.form.get("crop")
    crop = None
    if raw_crop not in (None, ""):
        try:
            crop = json.loads(raw_crop)
        except (TypeError, ValueError):
            return jsonify({"error": "'crop' must be valid JSON"}), 400
        if not isinstance(crop, dict) or not all(k in crop for k in ("x0", "y0", "x1", "y1")):
            return jsonify({"error": "'crop' must be a JSON object with x0/y0/x1/y1"}), 400

    try:
        png_bytes = remove_background_preview(image_file.read(), max_dim=max_dim, crop=crop)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return send_file(io.BytesIO(png_bytes), mimetype="image/png")


@app.post("/api/design/vectorize")
def vectorize():
    """
    Real editable-vector pipeline (brief follow-up: "whichever background-
    removal choice the user makes must be converted into a real, editable
    vector, not just handed to the digitizer as a raster").

    Runs the exact same classification/background-removal/quantization/
    raster-to-polygon tracing digitizer/pipeline.py's `image_to_design`
    does internally, but stops *before* any stitch generation and returns
    the traced regions themselves as plain JSON vector shapes -- meant for
    a client to render as editable paths (select/delete/redraw) before the
    user confirms a final design.

    Multipart form-data request:
      image            : required, the raw image file.
      width_mm         : required, target design width in mm.
      height_mm        : required, target design height in mm.
      max_colors       : optional int, 1-16, default 6.
      max_dim          : optional int, default 150 (digitizer.pipeline.
                          DEFAULT_MAX_DIM) -- may be raised up to
                          MAX_MAX_DIM for finer traced detail, since this
                          endpoint does no stitch generation itself and so
                          doesn't need to stay as cheap as image-to-points.
      smooth_curves    : optional bool, default true.
      remove_background: optional bool, default true -- see
                          /api/design/remove-background. Pass false for
                          "keep original"; manual `bg_strokes` (below) still
                          apply on top either way.
      auto_vectorize   : optional bool, default true -- see image-to-points.
      force_photo      : optional bool, default false -- see image-to-points.
      bg_strokes       : optional JSON-encoded string: a list of manual
                          background-mask correction brush strokes, each
                          {"mode": "restore"|"erase", "radius": 0..1,
                           "points": [[nx, ny], ...]} with every coordinate/
                          radius normalized 0..1 over the image -- the
                          "adjustable-size brush to manually restore or
                          erase parts of the background-removal mask"
                          requirement. See
                          digitizer.segmentation.background_removal.
                          apply_mask_strokes for the exact semantics.
      crop             : optional JSON-encoded string: a normalized 0..1
                          {"x0", "y0", "x1", "y1"} box applied before
                          anything else runs -- see /api/design/
                          detect-region and digitizer.pipeline._apply_crop.
                          The "extract only the design from a photo of
                          someone wearing the garment" requirement: pass
                          the box the user confirmed/adjusted on that
                          suggestion, never one applied without them
                          seeing/approving it first.

    Response JSON on success:
      {"widthMm", "heightMm", "artworkKind", "artworkConfidence",
       "regions": [{"id", "colorHex", "areaMm2",
                     "rings": [[[xMm, yMm], ...], ...]}, ...]}

    Pass the (optionally user-edited) `regions` array straight to
    /api/design/regions-to-points or /api/design/regions-to-file to finish
    building the design.

    Or {"error": "..."} with a 400 status on a malformed request, an
    unreadable image, invalid `bg_strokes` JSON, or an image with no
    detectable foreground.
    """
    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return jsonify({"error": "'image' file field is required"}), 400

    form = request.form

    def _float_field(key: str, default: float) -> float:
        raw = form.get(key)
        return default if raw in (None, "") else float(raw)

    def _int_field(key: str, default: int) -> int:
        raw = form.get(key)
        return default if raw in (None, "") else int(raw)

    def _bool_field(key: str, default: bool) -> bool:
        raw = form.get(key)
        return default if raw in (None, "") else str(raw).strip().lower() not in ("0", "false", "no")

    try:
        width_mm = _float_field("width_mm", 0)
        height_mm = _float_field("height_mm", 0)
        if width_mm <= 0 or height_mm <= 0:
            return jsonify({"error": "'width_mm' and 'height_mm' are required and must be positive"}), 400

        max_colors = _int_field("max_colors", 6)
        smooth_curves = _bool_field("smooth_curves", True)
        remove_background_flag = _bool_field("remove_background", True)
        auto_vectorize = _bool_field("auto_vectorize", True)
        force_photo = _bool_field("force_photo", False)

        raw_strokes = form.get("bg_strokes")
        bg_strokes = None
        if raw_strokes not in (None, ""):
            try:
                bg_strokes = json.loads(raw_strokes)
            except (TypeError, ValueError):
                return jsonify({"error": "'bg_strokes' must be valid JSON"}), 400
            if not isinstance(bg_strokes, list):
                return jsonify({"error": "'bg_strokes' must be a JSON array"}), 400

        raw_crop = form.get("crop")
        crop = None
        if raw_crop not in (None, ""):
            try:
                crop = json.loads(raw_crop)
            except (TypeError, ValueError):
                return jsonify({"error": "'crop' must be valid JSON"}), 400
            if not isinstance(crop, dict) or not all(k in crop for k in ("x0", "y0", "x1", "y1")):
                return jsonify({"error": "'crop' must be a JSON object with x0/y0/x1/y1"}), 400

        # See the matching comment in `_digitize_image_from_request` --
        # once a confirmed `crop` has already thrown away everything but
        # the actual design, this endpoint's own docstring already notes
        # max_dim "may be raised up to MAX_MAX_DIM for finer traced
        # detail" is cheap here; do that by default instead of requiring
        # every caller to know to ask for it explicitly.
        default_max_dim = MAX_MAX_DIM if crop is not None else 150
        max_dim = _int_field("max_dim", default_max_dim)
        if not (MIN_MAX_DIM <= max_dim <= MAX_MAX_DIM):
            return jsonify({"error": f"'max_dim' must be between {MIN_MAX_DIM} and {MAX_MAX_DIM}"}), 400
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"invalid numeric field: {e}"}), 400

    try:
        result = image_to_vector_regions(
            image_file.read(),
            width_mm=width_mm,
            height_mm=height_mm,
            max_colors=max_colors,
            max_dim=max_dim,
            smooth_curves=smooth_curves,
            remove_background=remove_background_flag,
            auto_vectorize=auto_vectorize,
            bg_strokes=bg_strokes,
            force_photo=force_photo,
            crop=crop,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(
        {
            "widthMm": result["width_mm"],
            "heightMm": result["height_mm"],
            "artworkKind": result["artwork_kind"],
            "artworkConfidence": result["artwork_confidence"],
            "regions": [
                {
                    "id": r["id"],
                    "colorHex": r["color_hex"],
                    "areaMm2": r["area_mm2"],
                    "rings": r["rings"],
                }
                for r in result["regions"]
            ],
        }
    )


def _design_from_regions_request() -> tuple[EmbroideryDesign | None, tuple | None]:
    """Shared JSON-body parsing + design-building for both regions-to-points
    and regions-to-file -- mirrors `_digitize_image_from_request` above, but
    for the no-image, region-list-in/design-out half of the editable-vector
    pipeline (see /api/design/vectorize)."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return None, (jsonify({"error": "request body must be a JSON object"}), 400)

    regions = body.get("regions")
    if not isinstance(regions, list) or not regions:
        return None, (jsonify({"error": "'regions' is required and must be a non-empty array"}), 400)

    def _num(key: str, default):
        raw = body.get(key, default)
        return default if raw is None else raw

    try:
        width_mm = float(_num("widthMm", 0))
        height_mm = float(_num("heightMm", 0))
        if width_mm <= 0 or height_mm <= 0:
            return None, (jsonify({"error": "'widthMm' and 'heightMm' are required and must be positive"}), 400)

        stitch_style = str(body.get("stitchStyle") or "fill").strip().lower()
        if stitch_style not in STITCH_STYLES:
            return None, (jsonify({"error": f"'stitchStyle' must be one of {list(STITCH_STYLES)}"}), 400)

        stitch_length_mm = float(_num("stitchLengthMm", 3.0))
        density_mm = body.get("densityMm")
        density_mm = None if density_mm in (None, "") else float(density_mm)
        angle_deg = float(_num("angleDeg", 0.0))
        border_width_mm = body.get("borderWidthMm")
        border_width_mm = None if border_width_mm in (None, "") else float(border_width_mm)
        artwork_kind = str(body.get("artworkKind") or "flat")
        artwork_confidence = float(_num("artworkConfidence", 1.0))
    except (TypeError, ValueError) as e:
        return None, (jsonify({"error": f"invalid numeric field: {e}"}), 400)

    normalized_regions = [
        {"color_hex": r.get("colorHex"), "rings": r.get("rings")} for r in regions if isinstance(r, dict)
    ]

    try:
        design = design_from_region_records(
            normalized_regions,
            width_mm=width_mm,
            height_mm=height_mm,
            stitch_length_mm=stitch_length_mm,
            density_mm=density_mm,
            angle_deg=angle_deg,
            stitch_style=stitch_style,
            border_width_mm=border_width_mm,
            artwork_kind=artwork_kind,
            artwork_confidence=artwork_confidence,
        )
    except ValueError as e:
        return None, (jsonify({"error": str(e)}), 400)

    return design, None


@app.post("/api/design/regions-to-points")
def regions_to_points():
    """
    Builds a real digitized design from a (possibly user-edited) region
    list -- normally /api/design/vectorize's own output, after a client has
    let the user select/delete/redraw regions on top of it -- and returns
    the same JSON point-stream contract as /api/design/image-to-points, for
    client-side preview rendering (e.g. stitch-pro's Simulator screen).

    JSON body:
      {"widthMm", "heightMm",
       "regions": [{"colorHex", "rings"}, ...],   // vectorize's own shape
       "stitchStyle": "fill"|"cross",              // optional, default fill
       "stitchLengthMm", "densityMm", "angleDeg", "borderWidthMm",  // all optional
       "artworkKind", "artworkConfidence"}         // optional passthrough, purely informational

    Response/error shape identical to /api/design/image-to-points.
    """
    design, err = _design_from_regions_request()
    if err is not None:
        return err

    try:
        result = design_to_points(design)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result["artworkKind"] = design.metadata.get("artwork_kind")
    result["artworkConfidence"] = design.metadata.get("artwork_confidence")
    return jsonify(result)


@app.post("/api/design/regions-to-file")
def regions_to_file():
    """
    Same region-list-in design-building as /api/design/regions-to-points,
    but returns a real machine file -- the editable-vector pipeline's
    equivalent of /api/design/image-to-file, for final export once the
    user has confirmed their edited region list.

    JSON body: same fields as /api/design/regions-to-points, plus:
      "format" : optional, one of SUPPORTED_FORMATS, default "dst".
      "name"   : optional, used for the download filename.

    Returns the file as a binary attachment download on success, or
    {"error": "..."} with a 400 status on a malformed request or a region
    list that produces no stitch objects.
    """
    body = request.get_json(silent=True) or {}
    fmt = str(body.get("format", "dst")).lower()
    if fmt not in SUPPORTED_FORMATS:
        return jsonify({"error": f"unsupported format {fmt!r}", "supported": list(SUPPORTED_FORMATS)}), 400

    design, err = _design_from_regions_request()
    if err is not None:
        return err

    name = str(body.get("name") or "design")
    safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "design"

    try:
        file_bytes = design_to_format_bytes(design, fmt)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return send_file(
        io.BytesIO(file_bytes),
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"{safe_name}.{fmt}",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
