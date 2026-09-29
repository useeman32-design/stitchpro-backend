# Ink/Stitch — concept notes (GPL-3.0 — NO CODE COPIED, see THIRD_PARTY_LICENSES.md)

Ink/Stitch is GPL-3.0. These notes describe embroidery *concepts and terminology* we
learned by reading its public documentation (inkstitch.org/docs) — not its source
code — in our own words, for the purpose of designing an independent implementation.
Do not paste any Ink/Stitch source into the StitchPro backend.

## Running stitch
- Built from a path; a dashed stroke style signals "use running stitch" in their
  Inkscape workflow (an authoring-tool convention, not relevant to us since our input
  is raster artwork, not hand-drawn vector paths).
- Concept worth keeping: stitch spacing is a target length, but corners/sharp turns
  need an explicit stitch placed at the vertex itself, otherwise the thread cuts the
  corner short and rounds it off visually.

## Satin column
- Defined by two boundary "rails" plus optional "rungs" (cross-lines marking where one
  logical section ends/begins, e.g. for routing letters as separate segments).
- Zig-zags alternate left-rail → right-rail → left-rail...
- "Zig-zag spacing" is their density control — distance between successive crossing
  stitches along the column, not a stitch count.
- Multiple underlay types, stitched before the satin: **center-walk** (single line of
  running stitch down the column's middle and back), **contour** (running stitch
  offset in from each rail), **zig-zag** (a sparse low-density satin-like pass). Wider
  columns/trickier fabrics benefit from combining more than one.
- Pull compensation on satin: expands the column width outward from center by a
  configured mm amount per side, because the finished stitched width comes out
  narrower than the drawn width due to thread pulling the fabric in.
- "Auto-route satin columns" sequences multiple satin objects, inserting
  under-pathing/jump stitches as needed and splitting a column into pieces to avoid
  a jump when it can instead travel along already-stitched satin.

## Fill / tatami stitch
- Parallel stitch rows at a configurable angle, clipped to the polygon boundary
  (including holes).
- Rows are staggered — each row's stitch phase is offset from its neighbor by a
  configurable fraction of stitch length, specifically to avoid a visible straight
  seam ("valley") running across the fill where every row happens to break at the
  same x position.
- Underlay for fill is typically the *same fill algorithm, rotated ~90° and at a much
  wider row spacing* — its purpose is to flatten/stabilize the fabric under the dense
  top layer, not to be visible itself.
- Pull compensation for fill: expand each row outward from its row-center by a fixed
  mm and/or a percentage of the row's width; more advanced handling reshapes the
  overall polygon slightly (Ink/Stitch's own PR history shows this is still an active,
  imperfect area even for them — a useful signal that our first version doesn't need
  to be perfect either, just configurable and testable, exactly as brief §14 says).

## Routing / sequencing
- Object stitch order isn't purely nearest-neighbor distance minimization — color
  grouping, what must be stitched under what (layering), and minimizing trims/jumps
  are all inputs. This matches brief §12's explicit instruction not to optimize for
  shortest travel distance alone.

## Takeaway for our implementation
Every one of the above concepts maps directly onto a brief section (§9 running, §10
satin, §11 fill, §12 routing, §13 underlay, §14 compensation) and will be implemented
as new, independent StitchPro code — validated against Ink/Stitch's *rendered output*
for comparable simple test shapes (brief §22), never against its source.
