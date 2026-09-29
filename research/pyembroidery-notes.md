# pyembroidery — hands-on notes

Verified hands-on this session (not just docs-reading): installed `pyembroidery` via
pip in a scratch environment, generated a synthetic stitch pattern (a filled circle +
a ring/annulus, two colors, forcing genuine jump insertion mid-row), wrote it as a
Tajima `.dst` with a hand-rolled binary encoder, then independently decoded that same
file with `pyembroidery.read()` and compared:

- `pattern.bounds()` returned `(-369, -169, 381, 181)` in 0.1mm units — matched the
  hand-calculated geometry exactly (e.g. circle left edge:
  `offsetX(-45mm) + (20+0.5)*1.25mm - 14*1.25mm = -36.9mm = -369` units).
- `pyembroidery.write_png(pattern, ...)` rendered the design; visually confirmed
  correct fill zigzag, correct ring/hole shape (disjoint per-row runs + jump gap
  rendered correctly), correct color separation at the color-change boundary, no
  flip/mirror artifacts.
- Re-ran the same check later against real (non-synthetic) output from the actual
  StitchPro app's in-progress pipeline (the "Lion Emblem" and "Monogram M" built-in
  samples) — `pyembroidery.read()` + `Counter(s[2] for s in pattern.stitches)` gives an
  immediate breakdown of STITCH (0) / JUMP (1) / TRIM (2) / COLOR_CHANGE (5) / END (4)
  counts, which is the fastest way to sanity-check a generated pattern before
  eyeballing a render.

## API surface we'll actually call from the backend

```python
from pyembroidery import EmbPattern, EmbThread, write_dst, STITCH, JUMP, TRIM, COLOR_CHANGE

pattern = EmbPattern()
pattern.add_thread(EmbThread(0x111111))          # or EmbThread("#111111")
pattern.add_stitch_absolute(STITCH, x_mm * 10, y_mm * 10)   # units: 0.1mm
pattern.add_stitch_absolute(JUMP, x_mm * 10, y_mm * 10)
pattern.add_stitch_absolute(COLOR_CHANGE, x_mm * 10, y_mm * 10)
pattern.add_stitch_absolute(TRIM, x_mm * 10, y_mm * 10)
pattern.end()

write_dst(pattern, "design.dst", {"tie_on": True, "tie_off": True})
```

Key details that matter for correctness:
- Units are **0.1mm**, not mm — every coordinate we hand it must be pre-multiplied.
- Coordinate system has +y down (image-style), matching what our stitch generators
  will already produce if we keep the same convention as the current client-side
  `stitchEngine.ts`/`stitchPlanner.ts` — no flip needed going into pyembroidery.
- `write_dst` applies DST's 12.1mm max-stitch-length constraint and tie-on/tie-off
  contingencies automatically unless overridden in the settings dict — we should NOT
  duplicate that clamping ourselves upstream, just make sure our stitch generators
  don't rely on stitches longer than that by design (the fill/satin engines should
  already respect density-driven stitch lengths well under that ceiling).
- `EmbPattern.get_as_stitchblock()` is the convenient way to iterate color-grouped
  blocks for our stitch simulator, instead of re-deriving color groups from raw
  command flags ourselves.

## Decision

Use pyembroidery as the backend's format I/O layer (read + write), starting with DST
per the brief's priority, with PES/EXP/JEF/VP3 available "for free" later since the
same `EmbPattern` object writes all of them — we do not need separate work per format
once our internal stitch pattern → `EmbPattern` conversion exists.
