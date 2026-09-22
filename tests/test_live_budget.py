#!/usr/bin/env python3
"""The instrument has to come in time on the fields the user actually played.

Two cases, both read out of his own sessions of 2026-09-22 (the telemetry of
artifacts/perf/perf_*.npz beside the session log the prototype wrote):

    EventsArticulationBudget   switching `artic` from Env to Events starved the
                               device -- 504 underruns in ten seconds
    HarmDragBudget             dragging `harm` starved it again -- 1000 underruns
                               in 103 seconds, `rend` reaching 165 ms

Each one carries the field, the knobs, the tempo and the note of the moment the
underruns began, and asks the render thread for what it owes the device.

THE FIRST CASE (user session 2026-09-22 03:27, artifacts/perf/perf_20260922_032708.npz +
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

    python tests/test_live_budget.py

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
from casynth_engine import step, events_field, midi_to_freq, analyse     # noqa: E402
from casynth_engines.laplace_unified import SPECTRUM_KEYS                # noqa: E402
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
# ...and for a knob being DRAGGED, the same first thing said of the whole render
# thread: everything it owes the device -- the blocks, the knob it is told about
# and the field it is handed -- in well under real time.
RENDER_SHARE_CEILING = 0.6

UI_FRAME_S = 1.0 / 60.0       # the prototype's frame loop, so a drag posts at 60 Hz
DRAG_STEP = 1.0 / 48.0        # a hand moving about 2 px of a 96 px track per frame

# ── THE SECOND CASE: dragging `harm` (user session 2026-09-22 05:04) ─────────
#
# 1000 underruns in 103 s, the look-ahead again 5 -> 20 chunks, and `rend` reaching
# 165 ms.  The telemetry puts the worst of it under three knobs -- `harm` at
# t = 76..100 s, `fm_depth` at t = 52..62 -- and the articulation was Env this
# time, not Events: the engine is Env + Bank + Square (and Env + FM before t = 90).
# Measured on the field and the knobs below, 200 blocks, dragging `harm`:
#
#     Env + Bank + Square, knobs still   0.42 s of work per second of sound
#     Env + Bank + Square, harm dragged  1.18 s
#     Env + FM,            knobs still   0.46 s
#     Env + FM,            harm dragged  1.06 s
#
# and of the 1886 ms it spent for 1596 ms of sound: 1260 in the blocks, 278 in
# set_params and 291 in the UI thread's analyse -- the last two being the SAME
# analysis of the SAME field, run twice.
HARM_FIELD = (
    '.#............#..#.##.....###.....................#.',
    '##...........##.....#............................##.',
    '..................#..............................##.',
    '..........#..##...#..............................###',
    '...........##..........#####........................',
    '...............#......#...#.................#.......',
    '...............#..........#.#..............####.....',
    '...............#.......###................##...#....',
    '.............................................#...##.',
    '.................................................#..',
    '..............................................##.#..',
    '...............................................#.#..',
    '.......................##.......................##..',
    '.......................##...........................',
    '...........................................#........',
    '.###.......................................##.......',
    '...##.....................#...............#..#......',
    '......#....#.............#.#...............#.#......',
    '......#....#..............#................##...##..',
    '....#.#....#..............#......##.............#.#.',
    '..............##........#.#......##..............#..',
    '.#..###......#....###...............................',
    '..########....#.....#.........................#.....',
    '....##...#......#.#..#...##..............##..#.##...',
    '.........#.......#..##...##...............#.........',
    '.......................................#..#.##.##...',
    '..........###............##............#.##.#####...',
    '.........##..#...........#..##.........#...#..##....',
    '#.............#.....##.....###..............####...#',
    '.#......#..###...#...#.....#.#...............#......',
)
HARM_SETTINGS = dict(n=20, spread=1.0, alpha=0.0, shape=1.0, harm=0.75,
                     fullshape=1, dyn=1.0, artic=ARTIC_ENV, voice=0, waveform=2,
                     fm_depth=0.8333333333333334, events=0, radius_mul=1.0,
                     decay_s=0.8, attack_ms=0.0)
VOICE_BANK, VOICE_FM = 0, 1


def grid_of(rows):
    g = np.zeros((len(rows), len(rows[0])), np.uint8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == '#':
                g[r, c] = 1
    return g


def field():
    return grid_of(FIELD)


def run_blocks(params, blocks=250, warmup=30, start=None, drag=None,
               ui_analyse=False):
    """Render `blocks` blocks the way the prototype's render thread does, on
    `start` (the field above by default), advancing the automaton at STEPS_PER_S.

    drag       : the name of a knob to sweep, as a player's hand sweeps it -- one
                 DRAG_STEP per UI frame, turning around at the ends, so the whole
                 window is spent dragging.  A knob change is what the prototype
                 does with one: set_params on the engine, and analyse() on the UI
                 thread for the display.
    ui_analyse : also time the analysis the UI thread does for the display.  It
                 is the OTHER thread's work, so it is reported and not asserted on
                 -- but it is the same analyse() of the same field, and knowing
                 that is half of what this gate is for.

    -> dict of [ms]: `plain` / `boundary` (the blocks that carry no new generation
    and the ones that do), `setp` and `upd` (the rest of the render thread's work)
    and `anlys` (the UI thread's)."""
    ctx = EngineContext(SR, BLOCK, 2, midi_to_freq(NOTE_DEFAULT), 1.0,
                        STEPS_PER_S, pan=PAN_FIELD_MODE)
    p = dict(params)
    eng = registry.create('laplace_unified', ctx, dict(p))
    eng.set_rate(STEPS_PER_S)
    eng.set_envelope(*GEN_ENV)
    g = field() if start is None else np.array(start, np.uint8, copy=True)
    exc = None
    eng.init(g, exc, GAIN)
    samples_per_step = SR / STEPS_PER_S
    blocks_per_frame = max(1, int(round(UI_FRAME_S / CHUNK_S)))
    cum, gen = 0, 0
    way, val = -DRAG_STEP, float(p.get(drag, 0.0)) if drag else 0.0
    out = dict(plain=[], boundary=[], setp=[], upd=[], anlys=[])
    for i in range(blocks + warmup):
        live = i >= warmup              # the first blocks compile the kernels
        fresh = cum >= (gen + 1) * samples_per_step
        if fresh:
            new = step(g)
            exc = events_field(g, new)
            g = new
            gen += 1
            t0 = time.perf_counter()
            eng.update_field(g, exc)
            if live:
                out['upd'].append((time.perf_counter() - t0) * 1000.0)
        if drag and i % blocks_per_frame == 0:
            val += way
            if val <= 0.0 or val >= 1.0:
                way = -way
                val = min(1.0, max(0.0, val))
            p[drag] = val
            if ui_analyse:
                t0 = time.perf_counter()
                analyse(g, midi_to_freq(NOTE_DEFAULT), 'laplacian',
                        {k: p[k] for k in SPECTRUM_KEYS}, exc=exc)
                if live:
                    out['anlys'].append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            eng.set_params(dict(p))
            if live:
                out['setp'].append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        eng.render(GAIN, cum, gain_prev=GAIN, transpose=1.0)
        dt = (time.perf_counter() - t0) * 1000.0
        if live:
            out['boundary' if fresh else 'plain'].append(dt)
        cum += BLOCK
    return out


def report(name, got):
    plain, boundary = got['plain'], got['boundary']
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
        mean, plain_p99, worst_boundary = report(
            name, run_blocks(dict(SETTINGS, artic=artic)))
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


@timing_test
class HarmDragBudget(unittest.TestCase):
    """The user's 2026-09-22 05:04 case: dragging `harm` starved the device.

    A spectrum knob is the worst kind to drag: every mode of every figure changes
    frequency on every UI frame, so the whole field is re-analysed AND the wave
    bank needs a band-limited table for every new frequency.  The gate asks of the
    render thread what it owes the device -- the blocks, the knobs it is told
    about and the fields it is handed -- and reports what the UI thread spends on
    the same analysis beside it."""

    def _check(self, name, params):
        got = run_blocks(params, start=grid_of(HARM_FIELD), drag='harm',
                         ui_analyse=True)
        blocks = len(got['plain']) + len(got['boundary'])
        real_ms = blocks * BUDGET_MS
        render_ms = sum(sum(got[k]) for k in ('plain', 'boundary', 'setp', 'upd'))
        ui_ms = sum(got['anlys'])
        share = render_ms / real_ms
        print(f"  {name}, dragging harm: {blocks} blocks = {real_ms:.0f} ms of sound")
        for k in ('plain', 'boundary', 'setp', 'upd', 'anlys'):
            v = np.sort(np.asarray(got[k])) if got[k] else None
            if v is None or not len(v):
                continue
            print(f"      {k:9s} n={len(v):4d} mean={v.mean():7.2f} "
                  f"max={v[-1]:7.2f}  total={v.sum():7.0f} ms")
        print(f"      render thread {render_ms:.0f} ms = {share:.2f} of real time"
              f"   (UI thread's analyse {ui_ms:.0f} ms = {ui_ms / real_ms:.2f})\n")
        self.assertLess(share, RENDER_SHARE_CEILING,
                        f"{name}: dragging harm, the render thread spends "
                        f"{share:.2f} of real time (ceiling {RENDER_SHARE_CEILING}) "
                        f"-- this is the drag that starved the device")

    def test_bank_square_holds_real_time_while_harm_is_dragged(self):
        """Env + Bank + Square, the engine the last ten seconds of that session
        were played on."""
        self._check('Env+Bank+Square', dict(HARM_SETTINGS, voice=VOICE_BANK))

    def test_fm_holds_real_time_while_harm_is_dragged(self):
        """Env + FM, the engine the rest of it was played on."""
        self._check('Env+FM', dict(HARM_SETTINGS, voice=VOICE_FM))


if __name__ == '__main__':
    unittest.main(verbosity=2)
