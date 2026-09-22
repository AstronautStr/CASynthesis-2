#!/usr/bin/env python3
"""A frame in which nothing changed must cost almost nothing to draw.

The prototype's UI thread paints at 60 Hz and holds the GIL while it does; the
audit of 2026-09-22 measured 3.3-4.5 ms per frame -- 540-580 rectangles, 415-440
lines and 141 text renders on EVERY frame, of which nearly everything is static:
the grid, the legend gradient, every label, the pattern column, the field itself
between two generations, the spectrum strip between two analyses.  Under a drag
the render thread was busy 100 % of real time live against 62 % offline: the
difference is this thread's Python, sharing the GIL (user session 23:12).

This test runs the prototype headless with the field standing still and COUNTS
the text renders and drawing primitives of the last frames (deterministic).
A frame that changed nothing may render a handful of texts (the cost readout,
the meter) and draw the widgets whose state the mouse can change.

    python tests/test_frame_work.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("CASYNTH_VOLUME", "0.01")
os.environ["CASYNTH_RUN_SECONDS"] = "3"
os.environ["CASYNTH_NO_PERSIST"] = "1"

import pygame                                                       # noqa: E402

TEXT_CEILING = 12          # the readouts that really change: cost, meter, status
PRIMITIVE_CEILING = 220    # widgets the mouse can change; not the field, not the grid

COUNT = {'text': 0, 'prim': 0}


class _FontProxy:
    def __init__(self, f):
        self._f = f

    def render(self, *a, **k):
        COUNT['text'] += 1
        return self._f.render(*a, **k)

    def __getattr__(self, name):
        return getattr(self._f, name)


def _install_counters():
    real_sysfont = pygame.font.SysFont
    pygame.font.SysFont = lambda *a, **k: _FontProxy(real_sysfont(*a, **k))
    for name in ('rect', 'line', 'circle', 'polygon', 'aaline'):
        real = getattr(pygame.draw, name)

        def counted(*a, _real=real, **k):
            COUNT['prim'] += 1
            return _real(*a, **k)
        setattr(pygame.draw, name, counted)


class SteadyFrameIsCheap(unittest.TestCase):

    def test_the_last_frames_of_a_standing_field(self):
        _install_counters()
        import gol_synth
        per_frame = []
        real_draw = gol_synth.draw_frame

        def draw(*a, **k):
            COUNT['text'] = COUNT['prim'] = 0
            real_draw(*a, **k)
            per_frame.append((COUNT['text'], COUNT['prim']))
        gol_synth.draw_frame = draw
        gol_synth.main()                                  # three seconds, nothing touched
        self.assertGreater(len(per_frame), 60, "the run drew too few frames to judge")
        tail = per_frame[-20:]
        texts = max(t for t, _p in tail)
        prims = max(p for _t, p in tail)
        self.assertLessEqual(
            texts, TEXT_CEILING,
            f"a frame in which nothing changed rendered {texts} texts (ceiling {TEXT_CEILING}): "
            f"the labels are being rendered again every frame")
        self.assertLessEqual(
            prims, PRIMITIVE_CEILING,
            f"a frame in which nothing changed drew {prims} primitives (ceiling "
            f"{PRIMITIVE_CEILING}): the grid, the field or the legend are being redrawn "
            f"from scratch every frame")


if __name__ == '__main__':
    unittest.main(verbosity=2)
