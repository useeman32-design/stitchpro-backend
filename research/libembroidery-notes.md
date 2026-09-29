# libembroidery — notes

zlib licensed (see THIRD_PARTY_LICENSES.md — fully commercial-safe). Single-header C
library (`embroidery.h`) backing Embroidermodder 2. Provides pattern data structures,
~45 format readers/writers, a `libembroidery-convert` CLI, and some geometry/stitch
utility functions.

Not adopted for the current backend because:
1. It's C — our backend is Python (pyembroidery already gives us equivalent
   read/write coverage in the language we're actually using).
2. Its own README describes it as "v1.0-alpha, under construction."
3. No format in the brief's requirement list (DST primarily, PES/EXP/JEF/VP3 later)
   is missing from pyembroidery.

Re-evaluate if/when we need a format pyembroidery doesn't support, or if we ever need
a C/native extension for performance reasons pyembroidery can't meet (unlikely at
StitchPro's current scale — DST files are small, stitch counts in the thousands, not
millions).
