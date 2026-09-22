#!/usr/bin/env python3
"""The tail pool must not be walked slot by slot in Python on every block.

The audit of 2026-09-22 (memory/log/2026-09-22-perf-audit.md) found the second
reason a spectrum knob starves the device on the DEFAULT cell: on every block,
SlotPool walks its 1920 tail slots one numpy scalar at a time -- the tail budget
and the release countdown always, and _acquire_tail once for EVERY mode whose
frequency moved (all 480 of them on every frame of a drag).  At the player's
tempo (120 BPM, 1/4: a release lasts six blocks) the pool holds 500-800 ringing
tails and each acquisition scans hundreds of them: 4 ms of pure Python per block
under the GIL, doubled by the UI thread.

This test COUNTS the scalar reads of the release counter during one update on
that very case (deterministic; no milliseconds).  A pool that finds its free
and its quietest slots with vector operations makes a handful of them.

    python tests/test_slot_pool_work.py
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import (SR, CHUNK_S, MASTER_GAIN, GRID_H, GRID_W,        # noqa: E402
                            RANDOM_DENSITY, N_ACTIVE)
from casynth_engine import step, events_field, midi_to_freq, SlotPool         # noqa: E402
from casynth_engines import registry                                          # noqa: E402
from casynth_engines.engine_api import EngineContext, PAN_FIELD_MODE          # noqa: E402
import test_live_budget as tlb                                                # noqa: E402

BLOCK = int(CHUNK_S * SR)
STEPS_PER_S = 2.0                 # 120 BPM at 1/4 -- the tempo the instrument opens on
GAIN = MASTER_GAIN * 0.7
SCALAR_READS_CEILING = 64         # a vector pool reads the counter a few times per update


class _Counting(np.ndarray):
    """An ndarray view that counts SCALAR element reads (`a[s]` with one index)."""
    reads = 0

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            _Counting.reads += 1
        return np.ndarray.__getitem__(self, key)


def random_field():
    rng = np.random.default_rng(7)
    return (rng.random((GRID_H, GRID_W)) < RANDOM_DENSITY).astype(np.uint8)


class TailPoolIsNotWalkedInPython(unittest.TestCase):

    def test_a_dragged_knob_at_the_players_tempo(self):
        ctx = EngineContext(SR, BLOCK, 2, midi_to_freq(48), 1.0, STEPS_PER_S,
                            pan=PAN_FIELD_MODE)
        p = dict(tlb.SETTINGS, artic=0, voice=0, waveform=0)
        eng = registry.create('laplace_unified', ctx, dict(p))
        eng.set_rate(STEPS_PER_S)
        eng.set_envelope(0.0, 0.0, 1.0, 0.1, False)
        g = random_field()
        eng.init(g, None, GAIN)
        pool = eng._layers[0].cell.bank.pool if eng._layers else None
        if pool is None:
            eng.render(GAIN, 0)
            pool = eng._layers[0].cell.bank.pool
        self.assertIsInstance(pool, SlotPool)
        pool._release_cnt = pool._release_cnt.view(_Counting)
        # drag `harm` at 60 Hz for 60 blocks: the pool fills with ringing tails
        sps = SR / STEPS_PER_S
        cum = gen = 0
        val, way = 1.0, -1 / 48
        worst = 0
        ringing = 0
        for i in range(90):
            if cum >= (gen + 1) * sps:
                new = step(g)
                eng.update_field(new, events_field(g, new))
                g = new
                gen += 1
            if i % 2 == 0:
                val += way
                if val <= 0.0 or val >= 1.0:
                    way = -way
                    val = min(1.0, max(0.0, val))
                eng.set_params(dict(p, harm=val))
            _Counting.reads = 0
            eng.render(GAIN, cum, gain_prev=GAIN)
            if i >= 30:
                worst = max(worst, _Counting.reads)
                ringing = max(ringing, int(np.count_nonzero(
                    np.asarray(pool._release_cnt)[N_ACTIVE + 1:] > 0)))
            cum += BLOCK
        self.assertGreater(ringing, 300, "the case did not fill the pool -- it proves nothing")
        self.assertLessEqual(
            worst, SCALAR_READS_CEILING,
            f"one block of a `harm` drag at 120 BPM 1/4 read the release counter "
            f"{worst} times one slot at a time (pool holding up to {ringing} ringing "
            f"tails) -- the pool is being walked in Python; ceiling {SCALAR_READS_CEILING}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
