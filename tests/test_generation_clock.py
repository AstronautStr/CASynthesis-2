#!/usr/bin/env python3
"""The automaton must keep its tempo on the AUDIO clock, even while the render
thread stalls.

The user's session of 2026-09-22 23:12 ("при тормозах не соблюдается BPM"): the
automaton stepped on the wall clock to the millisecond (125.0 +- 8 ms at 1/16),
but in the SOUND a new generation lands on whichever block the render thread
happens to render next after it sees the new field.  When the thread has fallen
behind and then refills the ring in a burst, those blocks are bunched: measured
on the output-sample clock the generation intervals in the stuttering stretches
were 113 +- 34 ms (max 279) and 103 +- 18 against 125 in the clean ones.

This test runs the prototype headless with Random + Play at the default tempo
(120 BPM, 1/4: 22 050 samples a generation), makes the render thread stall for
70 ms every twelfth block -- longer than the 40 ms look-ahead, so the ring
runs dry and refills in bursts -- and reads back the DEVICE-time position at
which every generation was applied (the render thread logs it).  Every interval
must be one generation long to within a block, and the run must not drift.

Measures real time (the stall is a sleep): @timing_test, run alone by check.py.

    python tests/test_generation_clock.py
"""
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
os.environ.setdefault("CASYNTH_VOLUME", "0.01")
os.environ["CASYNTH_RUN_SECONDS"] = "14"
os.environ["CASYNTH_NO_PERSIST"] = "1"
os.environ["CASYNTH_RECORD"] = "1"          # the session is captured, not written

from timing_gate import timing_test                                    # noqa: E402
from casynth_config import SR, CHUNK_S, BPM_DEFAULT, NOTE_DIVS, DIV_DEFAULT   # noqa: E402

BLOCK = int(CHUNK_S * SR)
STEP_SAMPLES = NOTE_DIVS[DIV_DEFAULT][1] * 60.0 / BPM_DEFAULT * SR
STALL_S = 0.07
STALL_EVERY = 12


@timing_test
class GenerationsKeepTheTempoOnTheAudioClock(unittest.TestCase):

    def test_intervals_under_render_stalls(self):
        import pygame
        import gol_synth
        captured = {}
        gol_synth._dump_session = lambda rec, prefix='_session': captured.update(rec) or 'captured'

        real_create = gol_synth.engines.create
        count = {'n': 0}

        def create(eid, ctx, params):
            e = real_create(eid, ctx, params)
            real_render = e.render

            def render(*a, **k):
                count['n'] += 1
                if count['n'] % STALL_EVERY == 0:
                    time.sleep(STALL_S)
                return real_render(*a, **k)
            e.render = render
            return e
        gol_synth.engines.create = create

        frames = [0]
        real_get = pygame.event.get
        lay_box = {}

        def get(*a, **k):
            evs = real_get(*a, **k)
            f = frames[0]
            frames[0] += 1
            lay = lay_box.get('lay')
            if f == 5 and lay is not None:
                for bid in ('random', 'play'):
                    b = next(b for b in lay.buttons if b['id'] == bid)
                    evs.append(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=b['rect'].center,
                                                  button=1))
                    evs.append(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=b['rect'].center,
                                                  button=1))
            return evs
        pygame.event.get = get
        real_draw = gol_synth.draw_frame

        def draw(screen, fonts, state, lay, rt):
            lay_box['lay'] = lay
            real_draw(screen, fonts, state, lay, rt)
        gol_synth.draw_frame = draw
        np.random.seed(11)
        gol_synth.main()

        applied = captured.get('field_applied')
        self.assertTrue(applied, "the render thread did not log where it applied the fields "
                                 "(rec['field_applied'])")
        # (serial, cum, silence, gen, steps, look-ahead): the generations only,
        # in device time, with the ring depth the host had at that moment
        by_gen = {}
        for (_serial, c, s, g, _steps, la) in applied:
            if g > 0:
                by_gen.setdefault(g, (c + s, s, la))
        order = sorted(by_gen)
        self.assertGreater(len(order), 12, f"only {len(order)} generations sounded in the run")
        pos = np.array([by_gen[g][0] for g in order], float)
        sil = np.array([by_gen[g][1] for g in order], float)
        la = np.array([by_gen[g][2] for g in order], float)
        gens = np.array(order, float)
        iv = np.diff(pos) / np.diff(gens)
        # A generation is put on the beat plus the ring's look-ahead, so the
        # beat moves later by exactly one block each time a dropout deepens the
        # ring (that is the buffer getting deeper, not the tempo moving); apart
        # from that, an interval may miss the beat by one block of quantisation
        # and a millisecond or two of clock slack.  A block that was already in
        # the ring when a dropout inserted silence is heard later than the
        # render thread could know, so across an underrun the slack grows by
        # that silence.  Anything more is the automaton being applied when the
        # render thread happens to see it, not on its beat.
        expected = STEP_SAMPLES * np.diff(gens) + np.diff(la) * BLOCK
        dev = np.diff(pos) - expected
        slack = BLOCK + 0.004 * SR + np.diff(sil)
        over = np.abs(dev) - slack
        grown = int((np.diff(la) > 0).sum())
        print(f"  {len(order)} generations; intervals mean {iv.mean() / SR * 1000:.1f} ms "
              f"sd {iv.std() / SR * 1000:.1f}; off the beat by at most "
              f"{np.abs(dev).max() / SR * 1000:.1f} ms; the ring deepened {grown} times; "
              f"underruns {captured.get('underruns')}")
        self.assertLessEqual(
            float(over.max()), 0.0,
            f"a generation lands off its beat by {np.abs(dev).max() / SR * 1000:.1f} ms on the "
            f"audio clock, {over.max() / SR * 1000:.1f} ms beyond the block and the silence "
            f"around it -- the automaton is applied when the render thread happens to see "
            f"it, not on its beat")


if __name__ == '__main__':
    unittest.main(verbosity=2)
