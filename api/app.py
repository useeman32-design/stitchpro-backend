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
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, Response, jsonify, request, send_file

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from digitizer.pipeline import (
    MAX_MAX_DIM,
    MIN_MAX_DIM,
    PREVIEW_MAX_DIM,
    PREVIEW_MAX_MAX_DIM,
    PREVIEW_MIN_MAX_DIM,
    STITCH_STYLES,
    image_to_design,
    remove_background_preview,
)
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
                    "command": "jump" | "stitch" | "colorChange" | "end"}, ...],
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
        max_dim = _int_field("max_dim", 150)
        smooth_curves = _bool_field("smooth_curves", True)
        remove_background = _bool_field("remove_background", True)
        auto_vectorize = _bool_field("auto_vectorize", True)
        stitch_style = str(form.get("stitch_style") or "fill").strip().lower()
        if stitch_style not in STITCH_STYLES:
            return None, (jsonify({"error": f"'stitch_style' must be one of {list(STITCH_STYLES)}"}), 400)
        if not (MIN_MAX_DIM <= max_dim <= MAX_MAX_DIM):
            return None, (jsonify({"error": f"'max_dim' must be between {MIN_MAX_DIM} and {MAX_MAX_DIM}"}), 400)
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
      max_dim         : optional int, caps the longest image side (in
                         pixels) actually processed, default 150 -- keeps
                         server-side processing cheap regardless of the
                         uploaded image's real resolution.
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
                   "command": "jump"|"stitch"|"colorChange"}, ...],
       "artworkKind": "flat"|"photo"|"flat (forced)",
       "artworkConfidence": 0..1}

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

    try:
        png_bytes = remove_background_preview(image_file.read(), max_dim=max_dim)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return send_file(io.BytesIO(png_bytes), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
