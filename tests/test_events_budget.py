#!/usr/bin/env python3
"""The Events articulation has to hold real time on a Random field.

THE CASE (user session 2026-09-22, artifacts/perf/perf_20260922_032708.npz +
_session_20260922_032707.npz).  The player was on Laplace+ with the instrument's
start settings, on a Random field at 12 generations a second, and the sound was
clean.  At t = 23.8 s he moved ONE knob -- `artic`, Env -> Events -- and the
device starved: 504 underruns in the next ten seconds and the look-ahead the
host buys with dropouts went 5 -> 20 chunks (40 -> 160 ms of latency).  At
t = 43.5 s he moved it back and the underruns stopped dead.

What the telemetry of that session measured, per rendered block (budget 8.00 ms):

    Env    (before, 19 s)   mean 2.45   p50 1.82   p90  5.11   over budget  1.2%
    Events (after,  9 s)    mean 9.98   p50 8.32   p90 21.80   over budget 58.5%
    Env    (again,  9 s)    mean 2.53   p50 2.09   p90  4.75   over budget  1.0%

The render thread delivered 97 blocks a second where the device asks for 125.

WHY IT IS THE ARTICULATION AND NOT THE FIELD.  The Env cell analyses the field
when the field changes (update_field) and the block itself is then cheap.  The
Events cell analyses INSIDE the block: object_resonators._boundary -> _track ->
the eigen-decomposition of every figure that was born, all of it on the render
thread between two blocks the device is already waiting for.  On a Random field
at 12 generations a second that boundary lands on roughly every tenth block and
costs more than the whole block budget.

So the gate below is not "is the arithmetic right" -- tests/test_laplace_unified
already pins that byte for byte -- it is "does it come in time":

    test_events_holds_real_time     the render thread's work for one second of
                                    sound takes well under one second
    test_env_holds_real_time        the same field and the same settings on the
                                    articulation that was fine, so a fix to
                                    Events cannot be a slowdown of Env

    python tests/test_events_budget.py

Stdlib runner (pytest is not installed).  Measures time: @timing_test, so
check.py runs it alone (tests/timing_gate.py)."""
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from timing_gate import timing_test                                      # noqa: E402
from casynth_config import (SR, CHUNK_S, MASTER_GAIN, NOTE_DEFAULT,      # noqa: E402
                            AUDIO_LOOKAHEAD_MS)
from casynth_engine import step, events_field, midi_to_freq              # noqa: E402
from casynth_engines import registry                                     # noqa: E402
from casynth_engines.engine_api import EngineContext, PAN_FIELD_MODE     # noqa: E402

BLOCK = int(CHUNK_S * SR)                   # 352 samples = 7.98 ms of sound
BUDGET_MS = BLOCK / SR * 1000.0

# The field the player had, read out of the session at the frame the first
# underrun was logged (replay_grids[1394], 265 live cells of 30x52).
FIELD = (
    '#.....#..#.........##......###.##............#.##.#.',
    '##.......#........#..#......#######..#..............',
    '.#......#..........#..........#...#...#...........#.',
    '#........###.......#..................#......###...#',
    '.........###.........##.............#.........##....',
    '........#..##.........#.............................',
    '.......###..........................................',
    '........#..#..#.....................................',
    '.........##.##......#...#...........................',
    '.............#..#......#............................',
    '..........#...#......#..............................',
    '..........#.........#....................#.#........',
    '..............###......#................##.##.......',
    '..............#...#...#.#.............#...#.#.......',
    '...............###...#.#..............#.....#.......',
    '...............###...##.....##........#....#........',
    '........#.....................#........#...#.....#..',
    '.......###...................#...........#.##...#.##',
    '......##..#.............##.##.............#.#.......',
    '........##..............#..........#......#.#.......',
    '...#.............###.....#...#....##......##.....###',
    '...##.......#...##..#....#.........#....#..#.....##.',
    '....#...#.#####.##.##..##.#..#..........#....##.....',
    '....#...#....#....###...#...##...###..........#....#',
    '.##..#......###......##.#......#.##........##.##..#.',
    '............###...#..####..#..................##.##.',
    '...............#...##..#..#.........................',
    '..#.#..###......#...#...#...........................',
    '..#..##..........#..#..##.....#...................#.',
    '......####..........####....#####............#..###.',
)

# ...and the knobs it was played with (replay_controls[1394]); `artic` is the one
# the test moves, everything else is exactly what the session recorded.
SETTINGS = dict(n=20, spread=1.0, alpha=0.0, shape=1.0, harm=1.0, fullshape=1,
                dyn=1.0, voice=0, waveform=2, events=0, fm_depth=1.0,
                radius_mul=1.0, decay_s=0.8, attack_ms=0.0)
ARTIC_ENV, ARTIC_EVENTS = 0, 1
STEPS_PER_S = 12.0            # bpm 120 at 1/16T, the division the session was on
GAIN = MASTER_GAIN * 0.7      # master gain at the volume the player had
GEN_ENV = (0.0, 0.0, 1.0, 0.1, False)     # gen attack / decay / sustain / release / slew

# WHAT "IN TIME" MEANS -- three things, none of them a number fitted to today's
# machine:
#
#   1. one second of sound must cost well under one second of work.  The render
#      thread does not own the machine: the UI thread paints at 60 Hz (about 20%
#      of a core in the same session) and holds the GIL while it does.  Sixty per
#      cent of the budget leaves that room; the case as the user hit it spent 115%.
#   2. an ORDINARY block -- one that carries no new generation -- must fit in the
#      budget.  These are 90% of all blocks; if they do not fit, nothing can.
#   3. a block that DOES carry a new generation is allowed to cost more (the
#      Objects law tracks its figures there), but not more than the look-ahead the
#      host keeps -- one boundary may lean on the ring, never empty it.
MEAN_CEILING = 0.6 * BUDGET_MS
BOUNDARY_CEILING = float(AUDIO_LOOKAHEAD_MS)


def field():
    g = np.zeros((len(FIELD), len(FIELD[0])), np.uint8)
    for r, row in enumerate(FIELD):
        for c, ch in enumerate(row):
            if ch == '#':
                g[r, c] = 1
    return g


def run_blocks(artic, blocks=250, warmup=30):
    """Render `blocks` blocks the way the prototype's render thread does, on the
    field above, advancing the automaton at STEPS_PER_S.

    -> (plain, boundary): [ms] of the blocks that carry no new generation and of
    the ones that do.  Both are the work of the RENDER THREAD only -- update_field
    when the generation changed, plus the block itself.  The automaton step and
    the analysis the UI thread does for the display are not counted: they are the
    other thread's, and this gate is about what the device waits for."""
    params = dict(SETTINGS, artic=artic)
    ctx = EngineContext(SR, BLOCK, 2, midi_to_freq(NOTE_DEFAULT), 1.0,
                        STEPS_PER_S, pan=PAN_FIELD_MODE)
    eng = registry.create('laplace_unified', ctx, dict(params))
    eng.set_rate(STEPS_PER_S)
    eng.set_envelope(*GEN_ENV)
    g = field()
    exc = None
    eng.init(g, exc, GAIN)
    samples_per_step = SR / STEPS_PER_S
    cum, gen = 0, 0
    plain, boundary = [], []
    for i in range(blocks + warmup):
        fresh = cum >= (gen + 1) * samples_per_step
        t0 = time.perf_counter()
        if fresh:
            new = step(g)
            exc = events_field(g, new)
            g = new
            gen += 1
            eng.update_field(g, exc)
        eng.render(GAIN, cum, gain_prev=GAIN, transpose=1.0)
        dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:                 # the first blocks compile the kernel
            (boundary if fresh else plain).append(dt)
        cum += BLOCK
    return plain, boundary


def report(name, plain, boundary):
    p = np.sort(np.asarray(plain))
    b = np.sort(np.asarray(boundary))
    allv = np.sort(np.concatenate([p, b]))
    n = len(p)
    print(f"  {name:6s} ordinary n={n:4d} mean={p.mean():6.2f} p50={p[n // 2]:6.2f} "
          f"p99={p[int(n * .99)]:6.2f} max={p[-1]:6.2f}")
    print(f"  {'':6s} new field n={len(b):4d} mean={b.mean():6.2f} "
          f"max={b[-1]:6.2f} (look-ahead {BOUNDARY_CEILING:.0f} ms)")
    print(f"  {'':6s} all blocks mean={allv.mean():6.2f} of {BUDGET_MS:.2f} ms"
          f"  -> one second of sound costs "
          f"{allv.mean() * (1.0 / CHUNK_S) / 1000:.2f} s")
    return allv.mean(), float(p[int(n * .99)]), float(b.max())


@timing_test
class EventsArticulationBudget(unittest.TestCase):
    """The user's 2026-09-22 case: one knob (`artic`) starved the device."""

    def _check(self, name, artic):
        mean, plain_p99, worst_boundary = report(name, *run_blocks(artic))
        self.assertLess(mean, MEAN_CEILING,
                        f"{name}: a block costs {mean:.2f} ms of the {BUDGET_MS:.2f} ms "
                        f"budget on average, over the {MEAN_CEILING:.2f} ms ceiling -- "
                        f"one second of sound costs "
                        f"{mean * (1.0 / CHUNK_S) / 1000:.2f} s")
        self.assertLess(plain_p99, BUDGET_MS,
                        f"{name}: an ordinary block (no new generation) reaches "
                        f"{plain_p99:.2f} ms at p99, over the {BUDGET_MS:.2f} ms budget")
        self.assertLess(worst_boundary, BOUNDARY_CEILING,
                        f"{name}: the block that takes a new generation costs "
                        f"{worst_boundary:.2f} ms, more than the {BOUNDARY_CEILING:.0f} ms "
                        f"of look-ahead the host keeps -- one boundary empties the ring")

    def test_env_holds_real_time(self):
        """The articulation that was fine stays fine -- the anchor a fix to
        Events must not move."""
        self._check('Env', ARTIC_ENV)

    def test_events_holds_real_time(self):
        """Events on the same field and the same settings: what the player heard
        break when he moved `artic`."""
        self._check('Events', ARTIC_EVENTS)


if __name__ == '__main__':
    unittest.main(verbosity=2)
