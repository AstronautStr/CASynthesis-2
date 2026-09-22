#!/usr/bin/env python3
"""Drag a spectrum knob on the Events wave bank until the table pool overflows.

THE CRASH (user, 2026-09-22).  The instrument died with no traceback -- the window
simply closed -- and the Windows event log had python.exe, exception 0xc0000005
(access violation), faulting module "unknown".  That is a compiled kernel reading
past an array: unified_events kept a SECOND pool of wave tables on top of
laplace_carriers', and when it filled, its clear() reallocated the buffer at 16
rows.  The rows already written into tab_a / tab_b for the block in flight then
pointed past that buffer, and the readout has no bounds check.

Dragging `harm` reaches it in a second: it pulls every mode of every figure
toward its own whole harmonic, so one block asks for a table at ~1130 distinct
frequencies where the pool holds 512.

This file is the repro, kept as the child process of the gate in
tests/test_live_budget.py (WavePoolSurvives): run it and it either finishes and
prints one line, or the process dies of the bug.

    python tests/wave_pool_repro.py [blocks]

Exit 0 and "SURVIVED ..." = the pool held.  Anything else -- and in particular a
return code that is not a Python exception at all -- is the bug.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, NOTE_DEFAULT                              # noqa: E402
from casynth_engine import step, events_field, midi_to_freq              # noqa: E402
from casynth_engines import registry                                     # noqa: E402
from casynth_engines import laplace_carriers as lc                       # noqa: E402
from casynth_engines.engine_api import EngineContext, PAN_FIELD_MODE     # noqa: E402

import test_live_budget as T                                            # noqa: E402

BLOCKS = int(sys.argv[1]) if len(sys.argv) > 1 else 240
STEPS_PER_S = 12.0


def main():
    params = dict(T.SETTINGS, artic=T.ARTIC_EVENTS, waveform=2)   # Events + Square
    ctx = EngineContext(SR, T.BLOCK, 2, midi_to_freq(NOTE_DEFAULT), 1.0,
                        STEPS_PER_S, pan=PAN_FIELD_MODE)
    eng = registry.create('laplace_unified', ctx, dict(params))
    eng.set_rate(STEPS_PER_S)
    eng.set_envelope(*T.GEN_ENV)
    g = T.field()
    exc = None
    eng.init(g, exc, T.GAIN)
    samples_per_step = SR / STEPS_PER_S
    cum, gen = 0, 0
    harm, way = float(params['harm']), -T.DRAG_STEP
    worst_row, peak_rows = -1, 0
    for i in range(BLOCKS):
        if cum >= (gen + 1) * samples_per_step:
            new = step(g)
            exc = events_field(g, new)
            g = new
            gen += 1
            eng.update_field(g, exc)
        if i % 2 == 0:                       # a 60 Hz drag against 125 Hz blocks
            harm += way
            if harm <= 0.0 or harm >= 1.0:
                way = -way
                harm = min(1.0, max(0.0, harm))
            params['harm'] = harm
            eng.set_params(dict(params))
        eng.render(T.GAIN, cum, gain_prev=T.GAIN, transpose=1.0)
        cum += T.BLOCK
        # what the block just read: the row numbers handed to the kernel, against
        # the buffer it was given.  A row at or past it is the crash.
        for lay in eng._layers:
            cell = lay.cell
            tab_a = getattr(cell, 'tab_a', None)
            if tab_a is None:
                continue
            have = cell.tables.buf.shape[0]
            peak_rows = max(peak_rows, have)
            top = max(int(tab_a.max()), int(cell.tab_b.max()))
            if top >= have:
                print(f"FAILED: block {i} pointed a mode at table row {top}, "
                      f"but the buffer holds {have}")
                return 1
            worst_row = max(worst_row, top)
    declined = int(lc._TAB_DECLINED[0])
    print(f"SURVIVED {BLOCKS} blocks: highest row used {worst_row} of a "
          f"{peak_rows}-row buffer, {len(lc._TAB_ROW)} tables held, "
          f"{declined} modes sent back to Re(z) by the cap")
    return 0


if __name__ == '__main__':
    sys.exit(main())
