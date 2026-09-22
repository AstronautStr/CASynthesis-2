#!/usr/bin/env python3
"""The wave-table pool must hold every frequency a standing field sounds.

TWO of the user's sessions of 2026-09-22 (artifacts/perf/perf_20260922_2246*.npz
and perf_20260922_2256*.npz with their _session_*.npz): Laplace+ on Events + Saw
at eight generations a second, `harm` set once below 1 and then nothing touched
-- and the instrument starved anyway.  With `harm` below 1 every mode of every
figure AND of every ringing tail has a frequency of its own, so a field sounds
600-1400 distinct waves at once; a pool smaller than that rebuilds hundreds of
tables every generation and sends the rest of the modes to the sine line
(`table_rows` returns -1 for what it cannot serve).

    22:46, 192 cells: 826 distinct waves.  512 rows: 6.3 ms a block, p90 19.8;
                      1024 rows: 4.8 / 5.9, nothing declined.
    22:56, 364 cells: up to 1434 distinct waves.  1024 rows: p90 8.7, up to
                      444 modes a block declined; 2048 rows: p90 7.4, none.

Both cases replay the field from the frame the knobs stopped moving and COUNT
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
STEPS_PER_S = 8.0                    # 120 BPM at 1/16, the division of both sessions
GAIN = MASTER_GAIN * 0.7
SETTINGS = dict(n=20, spread=1.0, alpha=0.0, shape=1.0, fullshape=1, dyn=1.0,
                artic=1, voice=0, waveform=1, fm_depth=1.0, events=0, radius_mul=1.0,
                decay_s=0.8, attack_ms=0.0)

# replay_grids[1340] of the 22:46 session, 192 cells, harm 0.4375
FIELD_2246 = (
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
# replay_grids[10088] of the 22:56 session (t = 169 s), 364 cells, harm 0.375
FIELD_2256 = (
    '....#..#...##........#............#.#....#.##.......',
    '.....##......#.....##........#.#.###.#...##...#....#',
    '..........#........##.........#......##.#.....##....',
    '...........#...##...#...............####............',
    '............##....##........###....##...#..##.......',
    '.............##..###....#.#####.....##...##........#',
    '..#..##.......#####....#..#.........##...........###',
    '.##.###................###..#..#..#.#...#..#........',
    '.##..##........................#....#..#.#...#......',
    '...#...#....................#....#.#......##........',
    '.##...##...##.............###..#####................',
    '.##.#.##.................#.#..##...##..#...#........',
    '.#.#....................#####......###..#...........',
    '........................#..#.........#..#..#........',
    '........................#..#......#.#...........###.',
    '...##.............#....##.##.....#..#...#...#...#...',
    '.#...#.........#.#.#...#####.....#.....#####........',
    '#....#........####.##..#...#..............##.##.....',
    '#..#.........#......##...##......#.#...###..........',
    '#..............######.....#.......#......##.##......',
    '..###..............###...#.....##........###.#.##...',
    '...##...........##.#.#...#....#.##....#..#.#....#...',
    '...#...............######....##.#....##...#..###....',
    '##...............###..#......##.##...#..#..#####....',
    '#....................##......##.#.....####.#........',
    '#####.....................##...........###.#........',
    '.###......................###...#...................',
    '###.....................##.......##...#.#.....#....#',
    '##.#............#.##.#............#.######....#.#...',
    '#...#.##..........##.#..#......#.#..#...#.....##....',
)
CASES = [('22:46', FIELD_2246, 0.4375, 600), ('22:56', FIELD_2256, 0.375, 1100)]


def grid_of(rows):
    g = np.zeros((len(rows), len(rows[0])), np.uint8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == '#':
                g[r, c] = 1
    return g


def replay(rows, harm, blocks=300, warmup=40):
    """-> (declined readouts, tables rebuilt on blocks without a new generation,
    the most distinct waves any block asked for)."""
    ctx = EngineContext(SR, BLOCK, 2, midi_to_freq(NOTE_DEFAULT), 1.0, STEPS_PER_S,
                        pan=PAN_FIELD_MODE)
    eng = registry.create('laplace_unified', ctx, dict(SETTINGS, harm=harm))
    eng.set_rate(STEPS_PER_S)
    eng.set_envelope(0.0, 0.0, 1.0, 0.1, False)
    g = grid_of(rows)
    exc = None
    eng.init(g, exc, GAIN)
    sps = SR / STEPS_PER_S
    cum = gen = 0
    declined = rebuilt_between = distinct = 0
    for i in range(blocks + warmup):
        fresh = cum >= (gen + 1) * sps
        if fresh:
            new = step(g)
            exc = events_field(g, new)
            g = new
            gen += 1
            eng.update_field(g, exc)
        d0, b0 = lc._TAB_DECLINED[0], lc._TAB_BUILT[0]
        eng.render(GAIN, cum, transpose=1.0)
        if i >= warmup:
            declined += lc._TAB_DECLINED[0] - d0
            if not fresh:
                rebuilt_between += lc._TAB_BUILT[0] - b0
            cell = eng._layers[0].cell
            sel, freq = cell._selection(1.0, audible_only=False)
            distinct = max(distinct, int(len(np.unique(freq[sel]))))
        cum += BLOCK
    return declined, rebuilt_between, distinct


class WavePoolHoldsTheField(unittest.TestCase):

    def _check(self, name, rows, harm, at_least):
        declined, rebuilt_between, distinct = replay(rows, harm)
        self.assertGreater(distinct, at_least, f"{name}: the field did not sound {at_least}+ "
                                               f"distinct waves -- the case proves nothing")
        self.assertEqual(
            declined, 0,
            f"{name}: {declined} mode-readouts fell back to the sine line over 300 blocks of "
            f"a field nobody touched: the pool ({lc.TABLE_CACHE_MAX} rows) is smaller than "
            f"the {distinct} distinct waves the field sounds")
        self.assertEqual(
            rebuilt_between, 0,
            f"{name}: {rebuilt_between} tables were rebuilt on blocks that carried no new "
            f"generation: the pool ({lc.TABLE_CACHE_MAX} rows) is thrashing on a standing field")

    def test_the_untouched_field_of_the_22_46_session(self):
        self._check(*CASES[0])

    def test_the_denser_untouched_field_of_the_22_56_session(self):
        self._check(*CASES[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
