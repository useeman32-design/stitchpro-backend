#!/usr/bin/env python3
"""
Benchmarks `digitizer.pipeline.image_to_design` end-to-end against this
project's real photographed demo asset library (stitch-pro's
`assets/embroidery/*.jpg` -- the actual images shown to users in the Auto
Digitize wizard's sample-artwork picker and template gallery, see
stitch-pro's `src/constants/sampleArtworks.ts` and
`src/services/templates/index.ts`).

Why this exists as a standalone script rather than a pytest test: these
assets live in the sibling `stitch-pro` repo, not this one, so a hard
dependency on that checkout existing at a specific relative path would be
fragile in any environment/CI that only has this repo checked out. Run it
manually (from a workspace that has both repos side by side) any time
`digitizer/segmentation/{artworkify,color_quantize}.py` or
`digitizer/classifier/classify_artwork.py` change, to catch exactly the
kind of regression this project has already hit once: a change that looks
like a pure improvement on one or two images can silently cost a real
color on another (see artworkify.py's own POSTERIZE_BITS docstring for
the concrete story -- always measure actual `design.threads` survival
across the *whole* asset library, never a palette-size proxy on a
handful of hand-picked images).

Usage:
    python3 scripts/benchmark_photo_pipeline.py [--assets-dir PATH]

Exits non-zero if any asset's extracted thread count regresses below the
`EXPECTED_MIN_THREADS` baseline recorded below (last updated when this
script was committed -- see docs/photo-pipeline-benchmark.md for the full
results table and how it was produced).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digitizer.pipeline import image_to_design  # noqa: E402

DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "stitch-pro" / "assets" / "embroidery"

# The 8 real photographed assets shown in stitch-pro's Auto Digitize wizard
# and template gallery (see this script's module docstring). Each must
# retain at least `EXPECTED_MIN_THREADS` distinct thread colors -- the
# number every one of them was confirmed to produce once the color-collapse
# bug (see digitizer/pipeline.py's module docstring) was fixed. A future
# change that drops any of these below their baseline is a real regression,
# not noise -- investigate before assuming a tuning change is an improvement.
EXPECTED_MIN_THREADS: dict[str, int] = {
    "lion-patch.jpg": 6,
    "monogram-m.jpg": 6,
    "company-logo.jpg": 6,
    "floral-design.jpg": 6,
    "cap-logo.jpg": 6,
    "badge-crest.jpg": 6,
    "wordmark-text.jpg": 6,
    "seasonal-snowflake.jpg": 6,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--max-dim", type=int, default=150, help="matches the API's own default")
    args = parser.parse_args()

    if not args.assets_dir.is_dir():
        print(f"assets dir not found: {args.assets_dir} (pass --assets-dir)", file=sys.stderr)
        return 2

    ok = True
    total_objects = 0
    for fname, expected_min in EXPECTED_MIN_THREADS.items():
        path = args.assets_dir / fname
        if not path.is_file():
            print(f"SKIP {fname}: not found at {path}")
            continue
        image_bytes = path.read_bytes()
        t0 = time.time()
        design = image_to_design(image_bytes, width_mm=80, height_mm=80, max_colors=6, max_dim=args.max_dim)
        dt = time.time() - t0
        n_threads = len(design.threads)
        n_objects = len(design.objects)
        total_objects += n_objects
        kind = design.metadata.get("artwork_kind")
        status = "OK" if n_threads >= expected_min else "REGRESSION"
        if n_threads < expected_min:
            ok = False
        print(
            f"{status:10s} {fname:28s} kind={kind:14s} threads={n_threads} "
            f"(expected >= {expected_min}) objects={n_objects:4d} time={dt:.3f}s"
        )

    print(f"\nTotal stitch objects across all assets: {total_objects}")
    print("PASS" if ok else "FAIL -- see REGRESSION lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
