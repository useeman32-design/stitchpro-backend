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
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, Response, jsonify, request, send_file

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from formats.dst.writer import design_to_dst_bytes
from formats.writer import SUPPORTED_FORMATS, points_to_format_bytes

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
    return jsonify({"status": "ok", "service": "stitchpro-backend", "milestone": 1})


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
