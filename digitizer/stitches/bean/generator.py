"""
Bean stitch generator (new stitch type follow-up: Bean/Motif/3D Puff/Applique).

Independent implementation -- no code taken from Ink/Stitch or any GPL/ACSL
project (see docs/THIRD_PARTY_LICENSES.md).

A "bean" (aka triple, or N-pass) stitch is a plain running stitch repeated
back and forth over the *exact same* path multiple times (classically 3
passes: forward, back, forward again) before moving on. Real digitizers
reach for it on hairline-thin outlines, stems, and lettering details that
need to read as a bolder, more defined line than a single running-stitch
pass gives, without the stiffness/thread cost of a full satin column --
see `digitizer.pipeline`'s BEAN eligibility heuristic for exactly which
auto-detected regions get this treatment instead of SATIN or FILL.

This reuses `generate_running_stitch` for the actual per-pass point
sampling (identical corner-preservation / max-stitch-length guarantees as
every other RUNNING-based output in this codebase), then just walks the
resulting point list forward and backward alternately `passes` times.
"""

from __future__ import annotations

from digitizer.stitches.running.generator import generate_running_stitch

Point = tuple[float, float]
Stitch = tuple[float, float, bool]  # (x, y, is_jump)


def generate_bean_stitch(
    geometry: list[Point],
    stitch_length_mm: float = 3.0,
    passes: int = 3,
) -> list[Stitch]:
    """Convert `geometry` (an open polyline, same contract as RUNNING, or a
    closed ring's point list treated as a path -- see the pipeline's BEAN
    heuristic for how a thin closed region becomes a centerline path before
    reaching here) into bean-stitch coordinates: `passes` full traversals of
    the same running-stitch point sequence, alternating direction each pass
    (forward, back, forward, ...) so the needle ends up back near wherever
    the next pass should continue from, instead of jumping back to the
    start every time.

    Guarantees:
    - Every pass uses the exact same resampled point sequence (computed
      once), so all `passes` lie exactly on top of each other -- a real
      bean stitch's defining visual trait, not `passes` independently-
      resampled near-duplicates that could drift apart on a curve.
    - Only the very first point of the very first pass is marked as a jump
      (the needle traveling in from wherever it was before this object);
      every later pass boundary is a normal connected stitch, since the
      needle is already sitting exactly where the next pass starts.

    `passes < 1` raises `ValueError`. `passes == 1` is accepted as a
    (rarely useful) escape hatch equivalent to a plain running stitch, not
    an error -- some callers may want to force a single pass.
    """
    if passes < 1:
        raise ValueError("passes must be >= 1")
    base = generate_running_stitch(geometry, stitch_length_mm)
    if not base:
        return []

    out: list[Stitch] = []
    for p in range(passes):
        pass_points = base if p % 2 == 0 else list(reversed(base))
        for i, (x, y) in enumerate(pass_points):
            is_jump = p == 0 and i == 0
            out.append((x, y, is_jump))
    return out
