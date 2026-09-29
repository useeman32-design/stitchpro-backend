"""
StitchPro backend API — Flask (WSGI), chosen specifically for cPanel shared
hosting compatibility (Passenger only speaks WSGI, not ASGI). See
docs/embroidery-engine-research.md section 3 for the reasoning.

This is deliberately minimal right now: brief section 24's full API surface
(/api/digitize, /api/preview, /api/design/:id, etc.) will grow here as more
milestones land. For Milestone 1, we expose just enough to prove the
mobile app could realistically talk to this over HTTP:

  GET  /api/health              -> liveness check
  POST /api/design/line-to-dst  -> {"points": [[x,y], ...], "stitch_length_mm": 3,
                                     "width_mm": 100, "height_mm": 100, "color": "#111111"}
                                    returns a .dst file download
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, jsonify, request, send_file
import io

from digitizer.model import EmbroideryDesign, EmbroideryObject, ObjectType, StitchSettings
from formats.dst.writer import design_to_dst_bytes

app = Flask(__name__)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "stitchpro-backend", "milestone": 1})


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
