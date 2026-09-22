#!/usr/bin/env python3
"""The wave-table pool must hold every frequency a standing field sounds.

The user's session of 2026-09-22 22:46 (artifacts/perf/perf_20260922_224642.npz,
_session_20260922_224642.npz): Laplace+ on Events + Saw at 8 generations a
second, `harm` set once to 0.4375 and then nothing touched -- and the instrument
starved anyway: 276 underruns in 36 s, blocks of 20-45 ms.  With `harm` below 1
every mode of every figure AND of every ringing tail has a frequency of its own,
so this field sounds 630-1000 distinct frequencies at once; the pool held 512
tables, so every generation rebuilt hundreds of them and 50-600 modes a block
fell back to the sine line (`table_rows` returns -1 for what the pool cannot
serve).  Replayed offline: 6.3 ms a block on average with a p90 of 19.8 at 512
rows; 4.8 / 5.9 at 1024, with nothing declined and tables built only where new
figures are born.

This test replays that field from the frame the knobs stopped moving and COUNTS
(deterministic; no milliseconds) what the pool could not serve and what it had
to rebuild between two generations -- both must be zero.

    python tests/test_wave_pool_holds_the_field.py
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, CHUNK_S, MASTER_GAIN, NOTE_DEFAULT                # noqa: E402
from casynth_engine import step, events_field, midi_to_freq                     # noqa: E402
from casynth_engines import registry                                            # noqa: E402
from casynth_engines import laplace_carriers as lc                              # noqa: E402
from casynth_engines.engine_api import EngineContext, PAN_FIELD_MODE            # noqa: E402

BLOCK = int(CHUNK_S * SR)
STEPS_PER_S = 8.0                    # 120 BPM at 1/16, the division of the session
GAIN = MASTER_GAIN * 0.7
# replay_grids[1340] of the session, 192 cells; the knobs it was played with
FIELD = (
    '........#...#.#...........##.#.............##.#.....',
    '........#.#.............##.#.##............#..#.....',
    '.#.......##..####.........#...#............##.#....#',
    '##.......##...###.....###.##.#......#......###......',
    '.#...................#####..#......#.#....#.........',
    '.....................#...###...##..#.#....#.........',
    '......................####....#..#..#...............',
    '.......................#......#..#..................',
    '...###.......................##.##..................',
    '...###........................##....................',
    '....................................................',
    '......................##........................#..#',
    '.#......##............#.#......................#....',
    '.#......##.............#............................',
    '....................................................',
    '..............................................#...##',
    '...................................##..#.##.........',
    '#..................................#.#...###...##...',
    '#.....##............#.............##.....#.##.......',
    '.....#..#..........#.#.............#.##...##.......#',
    '......#.#.........#..#..............###...#.........',
    '.......#..........###...............................',
    '............#....##.................................',
    '..........####......................................',
    '..............#.....................................',
    '.......#.####..##...................................',
    '......###....#.#..#.................................',
    '.........####.#.#..#................................',
    '......##...#...#..##................................',
    '.......#..##.###............................##......',
)
SETTINGS = dict(n=20, spread=1.0, alpha=0.0, shape=1.0, harm=0.4375, fullshape=1, dyn=1.0,
                artic=1, voice=0, waveform=1, fm_depth=1.0, events=0, radius_mul=1.0,
                decay_s=0.8, attack_ms=0.0)


def grid_of(rows):
    g = np.zeros((len(rows), len(rows[0])), np.uint8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == '#':
                g[r, c] = 1
    return g


class WavePoolHoldsTheField(unittest.TestCase):

    def test_the_untouched_field_of_the_22_46_session(self):
        ctx = EngineContext(SR, BLOCK, 2, midi_to_freq(NOTE_DEFAULT), 1.0, STEPS_PER_S,
                            pan=PAN_FIELD_MODE)
        eng = registry.create('laplace_unified', ctx, dict(SETTINGS))
        eng.set_rate(STEPS_PER_S)
        eng.set_envelope(0.0, 0.0, 1.0, 0.1, False)
        g = grid_of(FIELD)
        exc = None
        eng.init(g, exc, GAIN)
        sps = SR / STEPS_PER_S
        cum = gen = 0
        warm = 40
        declined = 0
        rebuilt_between = 0
        distinct = 0
        for i in range(300 + warm):
            fresh = cum >= (gen + 1) * sps
            if fresh:
                new = step(g)
                exc = events_field(g, new)
                g = new
                gen += 1
                eng.update_field(g, exc)
            d0, b0 = lc._TAB_DECLINED[0], lc._TAB_BUILT[0]
            eng.render(GAIN, cum, transpose=1.0)
            if i >= warm:
                declined += lc._TAB_DECLINED[0] - d0
                if not fresh:
                    rebuilt_between += lc._TAB_BUILT[0] - b0
                cell = eng._layers[0].cell
                sel, freq = cell._selection(1.0, audible_only=False)
                distinct = max(distinct, int(len(np.unique(freq[sel]))))
            cum += BLOCK
        self.assertGreater(distinct, 600, "the field did not sound 600+ distinct waves -- "
                                          "the case proves nothing")
        self.assertEqual(
            declined, 0,
            f"{declined} mode-readouts fell back to the sine line over 300 blocks of a field "
            f"nobody touched: the pool ({lc.TABLE_CACHE_MAX} rows) is smaller than the "
            f"{distinct} distinct waves the field sounds")
        self.assertEqual(
            rebuilt_between, 0,
            f"{rebuilt_between} tables were rebuilt on blocks that carried no new generation: "
            f"the pool ({lc.TABLE_CACHE_MAX} rows) is thrashing on a standing field")


if __name__ == '__main__':
    unittest.main(verbosity=2)
