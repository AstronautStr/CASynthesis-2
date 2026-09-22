#!/usr/bin/env python3
"""A full garbage collection must not be able to stall the instrument.

Both user sessions of 2026-09-22 (artifacts/perf/perf_20260922_2246*.npz and
_2256*.npz) show one full collection of 27-30 ms -- at 15 s and at 116 s of
play -- and underruns right there: the collector holds the GIL, so the render
thread and the device callback wait with it, and 30 ms is most of the 40 ms
look-ahead.  The cost is the size of the object graph the collector walks:
~173 000 tracked objects after start-up (numpy, scipy, numba, pygame, the
compiled kernels, the pattern library), none of which the instrument ever frees.

gc.freeze() moves what exists at that moment to the permanent generation, and
a full collection then walks only what was made since -- the frame's own
garbage.  This test runs the prototype headless for a few seconds the way it
starts, and then times one full collection over its object graph: 25 ms before
the freeze, well under a millisecond after.

    python tests/test_gc_pause.py
"""
import gc
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("CASYNTH_VOLUME", "0.01")
os.environ["CASYNTH_RUN_SECONDS"] = "3"
os.environ["CASYNTH_NO_PERSIST"] = "1"

PAUSE_CEILING_MS = 3.0            # measured: 25-36 ms before the freeze, ~0.5 after


class FullCollectionIsCheap(unittest.TestCase):

    def test_a_full_collection_after_startup(self):
        import gol_synth
        gol_synth.main()                          # start, play three seconds, quit
        worst = 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            gc.collect()
            worst = max(worst, (time.perf_counter() - t0) * 1000.0)
        self.assertLess(
            worst, PAUSE_CEILING_MS,
            f"a full collection over the instrument's object graph takes {worst:.1f} ms "
            f"({len(gc.get_objects())} tracked objects): every one of them stalls the "
            f"render thread and the device callback for that long")


if __name__ == '__main__':
    unittest.main(verbosity=2)
