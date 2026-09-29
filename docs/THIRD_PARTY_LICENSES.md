# Third-Party Library License Audit — StitchPro Embroidery Engine

Status: initial audit, based on direct inspection of each project's repository/license
file and PyPI metadata (checked 2026-09-29). Re-verify exact license text in the
vendored copy before shipping — licenses can change between versions.

Classification key:
- 🟢 GREEN — permissive (MIT/BSD/Apache-2.0/zlib). Safe to add as a dependency and
  ship inside our proprietary backend, subject to the stated attribution terms.
- 🟡 YELLOW — needs a specific legal/engineering decision before use (e.g. usable only
  as a build tool, or usable only if dynamically linked, or ambiguous).
- 🔴 RED — do not incorporate the code into our proprietary backend under any
  circumstance without explicit legal sign-off. Research/study only.

---

## 1. pyembroidery

- **Repo:** https://github.com/EmbroidePy/pyembroidery
- **Version audited:** current `main` / PyPI `pyembroidery` (author: Tatarize)
- **License:** MIT
- **Classification:** 🟢 GREEN
- **What it does:** Pure-Python library. Reads ~46 embroidery formats, writes 10-20
  (including DST, PES, EXP, JEF, VP3, SVG, PNG, CSV, G-code). Provides an `EmbPattern`
  object model (stitches as `(x, y, command)` tuples in 0.1mm units, a thread list,
  metadata/"extras"), a command set (`STITCH`, `JUMP`, `TRIM`, `STOP`, `END`,
  `COLOR_CHANGE`, `NEEDLE_SET`, `SEQUIN_MODE`, `SEQUIN_EJECT`, `SLOW`/`FAST`), format
  auto-conversion (`pyembroidery.convert(...)`), and settings like `max_stitch`,
  `max_jump`, `tie_on`/`tie_off` contingencies applied automatically per format.
- **Commercial use permitted:** Yes, unrestricted.
- **Modification permitted:** Yes.
- **Attribution requirement:** Keep the MIT license/copyright notice if we redistribute
  its source (standard MIT boilerplate) — no attribution required in-app UI.
- **Source disclosure requirement:** None.
- **Decision:** **Use directly as our DST (and other format) read/write layer.**
  It already solves exactly what section 1/18 of the brief asks for. We will NOT
  reimplement DST binary encoding — see `research/pyembroidery-notes.md` for the exact
  API surface we plan to call from our stitch pipeline (already hand-verified this
  session against real generated `.dst` output).

## 2. libembroidery

- **Repo:** https://github.com/Embroidermodder/libembroidery
- **Version audited:** v1.0-alpha (still under construction per their README)
- **License:** zlib
- **Classification:** 🟢 GREEN
- **What it does:** Single-header C library (`embroidery.h`) — pattern data structures,
  ~45 format readers/writers, a `libembroidery-convert` CLI, and geometry/stitch
  utility functions. Used internally by Embroidermodder 2.
- **Commercial use permitted:** Yes.
- **Modification permitted:** Yes.
- **Attribution requirement:** Must keep the license notice inside `embroidery.h`
  itself if vendoring the file; not required to redistribute a separate LICENSE file.
- **Source disclosure requirement:** None.
- **Decision:** **Do not add as a second dependency right now.** It is C, still
  alpha/"under construction" per its own README, and pyembroidery already covers every
  format and command we need in the language (Python) our backend will actually be
  written in. Re-evaluate only if we hit something pyembroidery genuinely can't do
  (e.g. a format pyembroidery doesn't read that a customer's machine needs).

## 3. Ink/Stitch (inkstitch)

- **Repo:** https://github.com/inkstitch/inkstitch
- **License:** **GPL-3.0**
- **Classification:** 🔴 RED — study only, zero code reuse
- **What it does (for research purposes only):** An Inkscape extension implementing a
  full manual/semi-automatic digitizing workflow: running stitch (from dashed
  strokes), satin columns (zig-zag between two rails, with "auto-route satin" for
  sequencing + under-pathing), tatami/fill stitch (parallel rows clipped to a
  polygon, staggered to avoid "valley" alignment), three underlay types for satin
  (center-walk, contour, zig-zag) and fill underlay (perpendicular fill at wider
  spacing), pull compensation (both a fixed-mm expansion and a percentage-of-width
  expansion, applied per row/column), lettering (via prepared font files), and a
  jump/trim-aware stitch simulator.
- **Why RED:** GPL-3.0 is a strong copyleft license. Incorporating GPL code (even a
  small function) into our proprietary commercial backend would obligate us to
  release our backend's source under GPL-3.0 too. That is incompatible with the
  "own and operate ourselves" commercial goal stated in the brief.
- **What we take instead:** Concepts and parameter *names/shapes* only (these are not
  copyrightable) — e.g. the idea that fill needs a perpendicular underlay layer, that
  satin pull compensation should be expressed as "extra mm added symmetrically (or
  asymmetrically) to each rail," that staggering fill rows avoids visible seams. Our
  `research/inkstitch-notes.md` documents these concepts in our own words with no
  copied code, and our satin/fill engines (sections 9-11 of the brief) will be
  independent implementations validated against Ink/Stitch's *output* (visual/metric
  comparison per section 22), never against its source.

## 4. PEmbroider

- **Repo:** https://github.com/CreativeInquiry/PEmbroider
- **License:** Dual-licensed — **GPLv3 OR the Anti-Capitalist Software License (ACSL) v1.4**
- **Classification:** 🔴 RED — cannot use at all for our purposes, stronger than Ink/Stitch
- **Critical finding:** The ACSL **explicitly forbids commercial/for-profit use**. Its
  permitted-user clause only covers: (a) an individual laboring for themselves, (b) a
  non-profit, (c) an educational institution, or (d) a shared-profit organization that
  lets non-members set their own labor cost. StitchPro, a commercial embroidery SaaS,
  does not qualify under any of these. The project's own README goes further and
  **names commercial embroidery-digitizing software by name** (Hatch, DRAWings, DIME,
  etc.) as exactly the kind of "profiteering" use it forbids — i.e. our own product
  category is called out by the authors as disallowed. The GPLv3 alternative doesn't
  help either — same copyleft problem as Ink/Stitch, and PEmbroider is Java/Processing,
  not Python, so there's no practical reuse path anyway.
- **Decision:** Do not use any PEmbroider code, do not vendor it, do not closely port
  its API. Treat its public documentation/examples purely as background reading on
  "what an embroidery geometry library's feature set looks like," nothing more.

## 5. Image processing / geometry libraries

| Library | Repo | License | Class | Purpose | Notes |
|---|---|---|---|---|---|
| **NumPy** | numpy/numpy | BSD-3-Clause | 🟢 GREEN | array math substrate for every other lib | already a transitive dependency of the others |
| **Pillow** | python-pillow/Pillow | HPND (permissive, MIT-like) | 🟢 GREEN | image decode (PNG/JPG/SVG-rasterized), resizing, basic pixel ops | lightweight, pure-C-extension but small; good default for constrained hosting |
| **OpenCV** (`opencv-python-headless`) | opencv/opencv | Apache-2.0 (since 4.5, was BSD-3 before) | 🟢 GREEN (license) / 🟡 YELLOW (ops) | contour detection, connected components, morphology, color quantization | license is fine; **hosting is the concern** — large native binary (~60-90MB wheel), heavier CPU/RAM footprint, see hosting section below |
| **scikit-image** | scikit-image/scikit-image | BSD-3-Clause | 🟢 GREEN | segmentation, morphology, contour extraction as a lighter-weight OpenCV alternative | depends on SciPy (also BSD, but a heavy compiled wheel) |
| **Shapely** | shapely/shapely | BSD-3-Clause | 🟢 GREEN | polygon boolean ops, buffering (pull-compensation offsets), cleanup, hole handling | wraps GEOS (also permissive, LGPL-2.1 — see note below) |
| **GEOS** (Shapely's native dependency) | libgeos/geos | LGPL-2.1-only | 🟡 YELLOW, but standard/safe pattern | geometry engine | LGPL is fine when **dynamically linked** (which is exactly how Shapely's wheel ships it) — we are not statically linking or modifying GEOS itself, so no source-disclosure obligation attaches to our code. Flag only if we ever vendor/statically link GEOS ourselves. |

Recommendation: start with **Pillow + Shapely only**. Add OpenCV or scikit-image
later, and only if the hosting environment can actually run them (see below) — don't
install either speculatively.

---

## Summary table

| # | Project | License | Class | Use |
|---|---|---|---|---|
| 1 | pyembroidery | MIT | 🟢 | Direct dependency — our DST/format I/O layer |
| 2 | libembroidery | zlib | 🟢 | Not used for now (redundant with pyembroidery); re-evaluate only for format gaps |
| 3 | Ink/Stitch | GPL-3.0 | 🔴 | Research/concepts only, zero code reuse |
| 4 | PEmbroider | GPLv3 / ACSL v1.4 | 🔴 | Forbidden for commercial use outright — docs/examples only, no code, no close API mirroring |
| 5 | NumPy | BSD-3-Clause | 🟢 | Dependency |
| 6 | Pillow | HPND | 🟢 | Dependency |
| 7 | OpenCV | Apache-2.0 | 🟢 (license) / 🟡 (ops cost) | Optional, hosting-dependent |
| 8 | scikit-image | BSD-3-Clause | 🟢 (license) / 🟡 (ops cost) | Optional, hosting-dependent |
| 9 | Shapely (+ GEOS) | BSD-3-Clause (+ LGPL-2.1 dynamic) | 🟢 | Dependency |

No RED-classified code will be copied into, or closely derived from, the StitchPro
backend. Anything under GPL/ACSL is used exclusively as a research reference.
