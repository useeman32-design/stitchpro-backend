"""
General multi-format output layer -- the ONLY place in the backend (besides
the now-legacy formats/dst/writer.py:compile_design, which this module
reuses) allowed to construct a pyembroidery EmbPattern or write binary
embroidery bytes.

Per docs/THIRD_PARTY_LICENSES.md: pyembroidery is MIT licensed, used here
directly as a dependency (not modified/vendored). We do not reimplement any
binary embroidery format -- pyembroidery already ships correct writers for
every format listed below.

Brief section 24 (API surface) calls for DST, PES, JEF, VP3 and EXP as the
first wave of supported export formats -- matching what every competitor in
this space (e.g. EmbroideryTrace) already ships. All five share the exact
same EmbPattern compilation step; only the final write call differs.
"""

from __future__ import annotations

import io
from typing import Literal

from pyembroidery import write_dst, write_exp, write_jef, write_pes, write_vp3

from digitizer.model import EmbroideryDesign
from formats.dst.writer import compile_design

Format = Literal["dst", "pes", "jef", "vp3", "exp"]

_WRITERS = {
    "dst": write_dst,
    "pes": write_pes,
    "jef": write_jef,
    "vp3": write_vp3,
    "exp": write_exp,
}

SUPPORTED_FORMATS: tuple[Format, ...] = tuple(_WRITERS.keys())  # type: ignore[assignment]


def design_to_format_bytes(design: EmbroideryDesign, fmt: Format) -> bytes:
    """Compile `design` once and write it out as `fmt`. Raises ValueError for
    an unsupported format name (always check `SUPPORTED_FORMATS` first if the
    format string comes from outside, e.g. an HTTP request)."""
    fmt = fmt.lower()  # type: ignore[assignment]
    writer = _WRITERS.get(fmt)
    if writer is None:
        raise ValueError(f"Unsupported format {fmt!r}; supported: {SUPPORTED_FORMATS}")

    pattern = compile_design(design)
    buf = io.BytesIO()
    writer(pattern, buf)
    return buf.getvalue()


def points_to_format_bytes(points, fmt: Format, colors: list[str] | None = None) -> bytes:
    """Same format dispatch as `design_to_format_bytes`, but for an
    already-compiled raw point stream (see formats/points_writer.py) rather
    than an EmbroideryDesign. Used by the /api/design/points-to-file route."""
    from formats.points_writer import points_to_format_bytes as _points_to_format_bytes

    fmt = fmt.lower()  # type: ignore[assignment]
    writer = _WRITERS.get(fmt)
    if writer is None:
        raise ValueError(f"Unsupported format {fmt!r}; supported: {SUPPORTED_FORMATS}")
    return _points_to_format_bytes(points, writer, colors)
