"""Tiny shared helper: hex color string -> pyembroidery EmbThread.

Split out of formats/dst/writer.py so formats/points_writer.py (the raw
point-stream compiler used by the HTTP API) can reuse it without creating a
circular import between the two writer modules.
"""

from __future__ import annotations

from pyembroidery import EmbThread


def hex_to_thread(hex_color: str) -> EmbThread:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    thread = EmbThread()
    thread.set_color(r, g, b)
    return thread
