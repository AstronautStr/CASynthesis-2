#!/usr/bin/env python3
"""Switching an axis of Laplace+ must not compile a kernel on the render thread.

In both user sessions of 2026-09-22 (artifacts/perf/perf_20260922_2246*.npz and
_2256*.npz) the click on `artic` cost 159-189 ms inside set_params ON THE RENDER
THREAD and a burst of 26-27 underruns: the Events cell is created there, and
creating it loads (or compiles) the numba kernels of the Objects bank and of the
Events voicings for the first time in the process.  The host already builds the
ENGINE on the UI thread for exactly this reason (gol_synth._hand_engine), but
the cells inside the unified engine are built lazily, on the first switch.

So the engine has a warm(): the host calls it right after creating the engine,
on its own thread, and every kernel a later switch could need is compiled then.
This test COUNTS compiled signatures (deterministic): after create + init + warm,
every Events kernel has one; and a switch to Events + Square and to Events + FM
then renders without a single new compilation.

    python tests/test_events_kernels_warm.py
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, CHUNK_S, MASTER_GAIN, NOTE_DEFAULT          # noqa: E402
from casynth_engine import midi_to_freq                                   # noqa: E402
from casynth_engines import registry                                      # noqa: E402
from casynth_engines.engine_api import EngineContext, PAN_FIELD_MODE      # noqa: E402

BLOCK = int(CHUNK_S * SR)
GAIN = MASTER_GAIN * 0.7


def kernels():
    """Every compiled kernel an Events cell can run, by name."""
    import casynth_engines.unified_events as uev
    import casynth_engines.object_resonators as orz
    import casynth_engines.figures as fg
    out = {'orz._render': orz._render,
           'fg._fill_laplacian': fg._fill_laplacian,
           'fg._periodic_mean': fg._periodic_mean}
    for name in ('_live_slots', '_wave_ramps', '_wave_slots', '_wave_mix', '_amps',
                 '_amps_ramps', '_amps_slots', '_fm_sum', '_fm_slots', '_fm_mix'):
        out['uev.' + name] = getattr(uev, name)
    return out


def compiled():
    return {k: len(getattr(f, 'signatures', ())) for k, f in kernels().items()}


class EventsKernelsAreWarm(unittest.TestCase):

    def test_warm_compiles_every_events_kernel_before_the_first_switch(self):
        try:
            import numba  # noqa: F401
        except ImportError:                       # pragma: no cover
            self.skipTest("numba is not installed: the kernels are plain Python")
        ctx = EngineContext(SR, BLOCK, 2, midi_to_freq(NOTE_DEFAULT), 1.0, 2.0,
                            pan=PAN_FIELD_MODE)
        p = dict(registry.defaults('laplace_unified'))
        eng = registry.create('laplace_unified', ctx, dict(p))     # Env + Bank + Sine
        g = np.zeros((30, 52), np.uint8)
        g[10:13, 10] = 1
        g[20, 30:33] = 1
        eng.init(g, None, GAIN)
        eng.render(GAIN, 0)
        # what the host does right after building the engine on its own thread
        eng.warm()
        cold = [k for k, n in compiled().items() if n == 0]
        self.assertEqual(cold, [], f"after warm() these kernels are still not compiled -- "
                                   f"the first switch would compile them on the render "
                                   f"thread: {cold}")
        before = compiled()
        # the switches the player makes: Events + Square, then Events + FM
        for axes in (dict(artic=1, voice=0, waveform=2), dict(artic=1, voice=1)):
            eng.set_params(dict(p, **axes))
            for i in range(3):
                eng.render(GAIN, (i + 1) * BLOCK)
        after = compiled()
        grew = [k for k in after if after[k] != before[k]]
        self.assertEqual(grew, [], f"switching the axes compiled new signatures on the "
                                   f"render path: {grew}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
