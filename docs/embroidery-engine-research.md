# StitchPro Embroidery Engine — Research Report & Architecture Recommendation

This is the deliverable for brief section 28 ("your first task right now"). It covers
what each reference project actually provides, what we will reuse vs. build
ourselves, and the recommended architecture — including the hosting constraint
(cPanel shared hosting) that changes some of the defaults below.

Full license terms and classification: `docs/THIRD_PARTY_LICENSES.md`.

## 1. What each project solves (and what we take from it)

### pyembroidery (MIT) — **used directly, as a dependency**
Hand-tested this session (not just read about): built a synthetic stitch pattern,
wrote it as DST with our own encoder, and independently decoded it with
`pyembroidery.read()` to verify correctness — see `research/pyembroidery-notes.md`
for the exact API calls and behavior we're relying on. It already provides:
- `EmbPattern` — the pattern object: a flat list of `(x, y, command)` stitches in
  0.1mm units, plus a `threadlist` of `EmbThread` (color/brand/catalog metadata) and a
  metadata "extras" dict.
- Every command we need: `STITCH`, `JUMP`, `TRIM`, `COLOR_CHANGE`, `STOP`, `END`,
  `SEQUIN_MODE`/`SEQUIN_EJECT`, `SLOW`/`FAST`.
- Full DST reader **and** writer (also PES/EXP/JEF/VP3/XXX/CSV/SVG/PNG/G-code), plus
  `pyembroidery.convert()` for format-to-format conversion.
- Sane per-format defaults applied automatically at write time (DST's 12.1mm max
  stitch length, tie-on/tie-off contingencies, coordinate flipping, etc.) — exactly
  the fiddly format-spec correctness work we don't want to reinvent.

**Decision: our backend will build a structured stitch pattern (our own
`EmbroideryDesign`/stitch-sequence model, see §4) and hand it to pyembroidery's
`EmbPattern` + `write_dst()` as the very last step.** This directly satisfies brief
section 1's instruction: "If PyEmbroidery already reliably writes DST files, do not
waste time reinventing the DST binary format." Our existing hand-rolled TypeScript
DST writer (`stitch-pro/src/utils/dstWriter.ts`, client-side, built and verified
earlier this project) was a reasonable stopgap for a client-only prototype, but the
backend rebuild should standardize on pyembroidery so we're not maintaining two
independent DST encoders long-term.

### libembroidery (zlib) — **not adopted, re-evaluate later**
A single-header C library backing Embroidermodder 2, permissively licensed (zlib), and
capable of ~45 formats via its own `EmbPattern`-equivalent struct and a
`libembroidery-convert` CLI. Fully fine to use commercially. We're not adding it
because: (a) it's C, our backend is Python; (b) it's explicitly "v1.0-alpha, under
construction" per its own README as of this check; (c) pyembroidery already covers
every format in the brief's requirement list. Keep on the radar only if we ever need a
format pyembroidery doesn't read.

### Ink/Stitch (GPL-3.0) — **concepts only, no code**
This is where almost all of the *algorithmic* research value is, and also the
project we're legally barred from copying from. Documented independently in
`research/inkstitch-notes.md`; the load-bearing concepts we're taking (as ideas, not
code) into our own implementation:
- **Running stitch**: subdivide a path into fixed-length segments; corners need a
  stitch placed AT the corner (not skipped over) so the thread actually turns there.
- **Satin column**: two "rail" boundary curves + a zig-zag between them; density is
  expressed as spacing between crossing stitches, not a stitch count; width comes
  from the perpendicular distance between rails at each point, not a fixed value.
- **Fill/tatami**: parallel rows at a chosen angle, clipped to the polygon (rows are
  computed on the *unrotated* polygon, then everything is rotated to the target
  angle — much simpler than rotating scan logic itself); rows are staggered
  (row N's stitch phase offset from row N+1) specifically to avoid a visible
  straight "valley" seam running perpendicular to the rows.
- **Underlay**: always stitched *before* the top layer; three named strategies
  (center-walk = one line of running stitch down the middle, contour = running
  stitch just inside each rail, zig-zag = a sparse low-density satin/fill pass) —
  its job is to stabilize/flatten the fabric so the top layer sits correctly.
- **Pull compensation**: fabric physically pulls inward as it's stitched, so the
  finished design comes out narrower than drawn. Compensation is a small outward
  expansion applied per-row (fill) or per-rail (satin), expressible as a fixed mm
  amount, a percentage of width, or both combined.
- **Routing**: object stitch order is not just "nearest neighbor" — color changes,
  layering (what has to be stitched under what), and minimizing trims all factor in.

We will implement our own versions of all of the above as new, independent code —
validated by comparing *output* (stitch counts, dimensions, visual renders) against
Ink/Stitch per brief section 22, never by reading its implementation line-by-line
while writing ours.

### PEmbroider (GPLv3 / ACSL) — **not usable, stronger restriction than expected**
Initially assumed "GPL, so same treatment as Ink/Stitch" — turned out to be worse: its
alternate license (Anti-Capitalist Software License) **explicitly names commercial
embroidery-digitizing software as a forbidden use case**. This one is fully off the
table, including for "study the algorithm, write our own version" purposes with any
direct code reference. See the license audit doc for the exact clause. Its
documentation is fine to read for feature-list inspiration only (e.g., "this is a
reasonable set of features an embroidery library needs"), same as we'd read any
product's marketing page.

### Image processing (OpenCV / scikit-image / Shapely) — **selectively adopted**
All permissively licensed (Apache-2.0 / BSD-3 / BSD-3+LGPL-dynamic — see audit doc).
The real constraint here isn't licensing, it's the **hosting environment** (§3 below).
Recommendation: start with Pillow (image decode/basic ops) + Shapely (polygon
boolean ops — union/difference for holes, `buffer()` for pull-compensation offsets,
`simplify()` for polygon cleanup). Hold off on OpenCV/scikit-image until we confirm
the hosting plan can actually run them.

## 2. What we build ourselves (no existing library covers this)

- Our internal `EmbroideryDesign` / `EmbroideryObject` data model (brief §8) — nothing
  external models "a design as a set of typed embroidery objects with settings," only
  "a flat stitch list" (pyembroidery) or "a vector drawing" (Ink/Stitch's Inkscape
  document model, which is GPL and Inkscape-specific anyway).
- The running/satin/fill stitch generators (brief §9-11) — original implementations,
  informed by the concepts above.
- Routing/sequencing, underlay, pull compensation as configurable, independent modules
  (brief §12-14).
- The image → embroidery-objects pipeline and the geometry-based classifier (brief
  §15-16) — deterministic, no ML, using Pillow/Shapely (+ OpenCV/scikit-image once
  hosting allows) purely as low-level geometry/pixel primitives.
- Quality validation and the stitch simulator (brief §19-20) — validation rules and
  simulator playback logic are product-specific; pyembroidery's pattern object gives us
  the stitch/jump/trim data to drive both, but the rules and the player are ours.

## 3. Hosting constraint: cPanel shared hosting — this changes real decisions

You mentioned the target deployment is **cPanel shared hosting**. This materially
affects the backend's tech choices, so here's what's actually true about that
environment (researched, not assumed):

- Shared cPanel plans that offer "Setup Python App" run Python through **Phusion
  Passenger as a WSGI application behind Apache** — your app never binds its own port
  and never runs as a standalone process; Apache/Passenger starts, manages, and can
  kill/restart it. This means:
  - **FastAPI (ASGI) is not natively supported.** It needs a WSGI adapter shim
    (`a2wsgi`) to run at all, which works but loses FastAPI's native async
    concurrency benefits in that mode. **Flask (WSGI-native) is the better fit** for
    a cPanel-hosted Python backend.
  - **No persistent background workers.** Celery/RQ-style long-running worker
    processes generally aren't viable on shared hosting — there's no mechanism to keep
    a daemon alive outside of Passenger's request-driven lifecycle. The realistic
    substitute is: (a) process digitization synchronously within the request if it's
    fast enough, or (b) use cPanel's **Cron Jobs** feature to run a short script every
    minute that pulls pending jobs from a database/file queue and processes one — a
    "poor man's async queue," not true background workers.
  - **CPU/RAM are shared and typically capped** (often 1 CPU core / ~512MB-1GB burst
    limits on entry shared plans, enforced via CloudLinux CageFS/LVE). Apache also
    enforces request timeouts (commonly 30-120s depending on host config). Real
    image-processing work (contour detection, polygon extraction, stitch generation
    on a full-resolution photo) can realistically blow past both on a cheap plan.
  - Installing packages is done through cPanel's Python App pip interface, which
    generally works for pure-Python or small-wheel packages, but heavy compiled
    packages (OpenCV, SciPy, scikit-image) are the most likely to fail to install, be
    extremely slow to import, or eat the memory ceiling — this is a real risk, not
    hypothetical.

**Recommendation (needs your decision, this is a real fork):**
1. **If we must stay 100% on your current cPanel plan:** build the backend as a Flask
   WSGI app, keep dependencies to pyembroidery + Pillow + Shapely only (skip OpenCV/
   scikit-image), cap input image resolution aggressively before processing (we
   already downscale to a small grid client-side — do the same server-side), keep
   each digitize request's processing time low enough to finish inside Apache's
   timeout, and use a cron-driven queue for anything that can't finish synchronously.
   This is workable for the "simple logos/names/monograms, 1-4 flat colors" MVP scope
   in brief §25, but will be a ceiling on quality/performance later (multi-object
   complex logos, larger hoop sizes, heavier segmentation).
2. **If a small VPS is an option** ($5-6/mo tier: DigitalOcean, Hetzner, Linode, or a
   PaaS like Render/Railway/Fly.io): run the embroidery backend there as a proper
   FastAPI service with a real background worker (Celery/RQ + Redis) for anything
   compute-heavy, and keep cPanel for whatever else it's currently hosting (website,
   email, etc.). The mobile app just calls whichever backend URL we configure — the
   API contract (brief §24) is identical either way, so this isn't a decision that
   locks us in; the code we write now (deterministic engine, stitch generators, data
   model) is 100% reusable in either hosting scenario. Only the *web framework glue*
   (Flask+WSGI vs FastAPI+ASGI) and *which image libraries we allow ourselves* change.

**I need to know which of these two you want before I pick Flask vs FastAPI and
before I decide whether OpenCV is on the table**, since switching frameworks later is
pure waste. Everything else below (data model, stitch engines, milestones 1-4) is
identical regardless of that answer, so I'm starting there.

## 4. Recommended architecture

```
stitchpro/
  mobile/                      # existing React Native + Expo app (unchanged scope)
  backend/
    api/                       # Flask or FastAPI routes (framework TBD, see §3)
    digitizer/
      preprocessing/           # decode, downscale, background handling (Pillow)
      segmentation/            # color reduction/segmentation, connected components
      geometry/                # contour->polygon, cleanup, Shapely ops
      classifier/              # deterministic geometry-based object-type decision
      stitches/
        running/
        satin/
        fill/
      routing/
      underlay/
      compensation/
      validation/
      simulation/
    formats/
      dst/                     # thin wrapper around pyembroidery's EmbPattern I/O
    tests/
  research/
    pyembroidery-notes.md      # hands-on findings from this session
    inkstitch-notes.md         # concepts-only notes, no copied code
    libembroidery-notes.md
  docs/
    architecture.md
    embroidery-model.md
    digitization.md
    THIRD_PARTY_LICENSES.md
    embroidery-engine-research.md   # this file
```

The `EmbroideryDesign` / `EmbroideryObject` model (brief §8) lives entirely inside
`backend/digitizer` and is what every stitch generator, router, and validator passes
around internally; pyembroidery's `EmbPattern` only gets constructed at the very final
`formats/dst` step. The mobile app talks to `backend/api` only (brief §24) and never
touches DST bytes or raw stitch coordinates directly — it receives a design ID, a
preview render, and a download URL.

## 5. Milestone plan (brief §26, restated as immediate next steps)

Once hosting is confirmed, the plan is exactly the milestone order in the brief:
1. Line → running stitch → DST (via pyembroidery) — smallest possible proof the new
   backend pipeline works end-to-end.
2. Polygon → fill stitch → DST.
3. Satin column → satin stitch → DST.
4. Combine all three + basic routing.
5. PNG → segmentation → embroidery objects (this reuses/upgrades the pixel-analysis
   logic already proven client-side this session, but moved server-side with real
   contour/polygon extraction instead of a scanline grid).
Milestones 6-10 follow after 1-5 are solid, per the brief.

Nothing in Milestone 1-4 is blocked by the hosting question — they're pure
stitch-math + pyembroidery, no image libraries involved yet. I can start those
immediately while you decide on hosting, since that decision only matters starting at
Milestone 5.
