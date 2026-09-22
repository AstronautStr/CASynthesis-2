#!/usr/bin/env python3
"""A spectrum knob must not make the instrument decompose the field again.

The audit of 2026-09-22 (memory/log/2026-09-22-perf-audit.md) found that dragging
ANY spectrum knob -- harm, shape, spread, alpha, n -- is the most expensive thing
the instrument does, on every cell including the default one.  One of the two
reasons: the eigen-decomposition of every figure's Laplacian is run again on
every knob value, on the UI thread (the display's analyse), on the render thread
(the Env cells' set_params) and, for the Events articulation, in _retune_all --
although the matrix L depends only on the field, and the knobs act only on the
cheap part after it (mode selection, the harmonic pull, the amplitude law).

These tests COUNT the decompositions (deterministic -- no milliseconds, so they
do not flap with the machine's temperature): twenty knob values on one standing
field may cost at most one decomposition per figure, not twenty.

    python tests/test_knob_drag_work.py

Stdlib runner (pytest is not installed)."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import casynth_core                                                        # noqa: E402
from casynth_config import SR, CHUNK_S, MASTER_GAIN                        # noqa: E402
from casynth_engine import analyse, analyse_cache_clear, midi_to_freq      # noqa: E402
from casynth_engines import registry                                       # noqa: E402
from casynth_engines import figures as fg                                  # noqa: E402
from casynth_engines.engine_api import EngineContext, PAN_FIELD_MODE       # noqa: E402
import test_live_budget as tlb                                             # noqa: E402

BLOCK = int(CHUNK_S * SR)
SPECTRUM = {k: tlb.SETTINGS[k] for k in tlb.SPECTRUM_KEYS}
KNOB_VALUES = [i / 20.0 for i in range(21)]


class _CountEigen:
    """Counts the LAPACK eigen-solves the code under test really runs: both the
    values-only and the vectors path, whichever the settings pick."""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._eigh = np.linalg.eigh
        self._eigvalsh = np.linalg.eigvalsh

        def eigh(a, *args, **kw):
            self.n += 1
            return self._eigh(a, *args, **kw)

        def eigvalsh(a, *args, **kw):
            self.n += 1
            return self._eigvalsh(a, *args, **kw)
        np.linalg.eigh = eigh
        np.linalg.eigvalsh = eigvalsh
        return self

    def __exit__(self, *exc):
        np.linalg.eigh = self._eigh
        np.linalg.eigvalsh = self._eigvalsh
        return False


def figures_of(grid):
    return sum(1 for c in fg.components(grid) if len(c) >= 2)


class SpectrumKnobDoesNotRedecompose(unittest.TestCase):

    def setUp(self):
        self.grid = tlb.field()
        self.exc = None
        self.n_fig = figures_of(self.grid)
        analyse_cache_clear()
        if hasattr(casynth_core, 'eigen_cache_clear'):
            casynth_core.eigen_cache_clear()

    def _drag(self, knob, fn):
        """fn(params) for every value of `knob`; -> eigen-solves counted."""
        with _CountEigen() as c:
            for v in KNOB_VALUES:
                p = dict(SPECTRUM)
                p[knob] = v if knob != 'n' else max(1, int(round(1 + v * 19)))
                fn(p)
        return c.n

    def _assert_once_per_figure(self, what, knob, n_calls):
        self.assertLessEqual(
            n_calls, self.n_fig,
            f"{what}: dragging `{knob}` over {len(KNOB_VALUES)} values on a standing field "
            f"of {self.n_fig} figures ran {n_calls} eigen-decompositions -- the matrix does "
            f"not depend on the knob, so at most {self.n_fig} (one per figure) are needed")

    # -- the display / the Env cells: casynth_engine.analyse --------------------
    def test_analyse_harm(self):
        n = self._drag('harm', lambda p: analyse(self.grid, midi_to_freq(48), 'laplacian', p,
                                                 exc=self.exc))
        self._assert_once_per_figure('analyse', 'harm', n)

    def test_analyse_shape(self):
        """shape > 0 needs eigenVECTORS; shape = 0 only values -- both must be remembered."""
        n = self._drag('shape', lambda p: analyse(self.grid, midi_to_freq(48), 'laplacian', p,
                                                  exc=self.exc))
        # the drag crosses shape = 0 once: the values-only path may add one solve per figure
        self.assertLessEqual(n, 2 * self.n_fig,
                             f"analyse: dragging `shape` ran {n} eigen-decompositions on "
                             f"{self.n_fig} figures (at most two per figure: values and vectors)")

    def test_analyse_n(self):
        n = self._drag('n', lambda p: analyse(self.grid, midi_to_freq(48), 'laplacian', p,
                                              exc=self.exc))
        self._assert_once_per_figure('analyse', 'n', n)

    # -- the Events articulation: object_resonators._retune_all -----------------
    def test_events_retune_harm(self):
        ctx = EngineContext(SR, BLOCK, 2, midi_to_freq(48), 1.0, tlb.STEPS_PER_S,
                            pan=PAN_FIELD_MODE)
        p = dict(tlb.SETTINGS, artic=1, voice=0, waveform=0)
        eng = registry.create('laplace_unified', ctx, dict(p))
        eng.init(self.grid, self.exc, MASTER_GAIN * 0.7)
        eng.render(MASTER_GAIN * 0.7, 0)              # the first boundary: every figure struck
        n = self._drag('harm', lambda q: eng.set_params(dict(p, **q)))
        self._assert_once_per_figure('Events set_params', 'harm', n)


if __name__ == '__main__':
    unittest.main(verbosity=2)
