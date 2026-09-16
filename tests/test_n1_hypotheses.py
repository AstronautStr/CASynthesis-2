"""Revised N1: independent model agreement, live field controls, saved experiments."""
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from casynth_lab import BLOCK, DemoRunner, scene_from_doc, registry
from casynth_lab.engine_api import EngineContext
from casynth_lab.catalog import Catalog
from casynth_lab import gutter_field as gf
from casynth_lab.gutter_controls import DAMP_ID, LINKS_ID
from demos.build_n1_hypotheses import CASES, PARAMS, blinkers, scene_for, record_offline, build
from demos.network_reference_n0.gutter_network import GutterNetwork

CTX = EngineContext(44100, BLOCK, 2, 110.0, 1.0, 4.0)


def grid(shift=0):
    g = np.zeros((32, 32), np.uint8)
    for r, c in blinkers(shift):
        g[r, c] = 1
    return g


def engine(eid, g):
    e = registry.create(eid, CTX, PARAMS)
    e.init(g, None, .028)
    return e


def reference_messages(net, g, eid):
    # Independent geometry and formulas; do not call production mapping helpers.
    counts = [sum(int(g[r, c]) for r in range(16 * y, 16 * (y + 1))
                  for c in range(8 * x, 8 * (x + 1))) for y in range(2) for x in range(4)]
    u = [n / (n + 2) for n in counts]
    for i in range(8):
        ratio = 2 ** (2 * u[i] - 1)
        net.set_filters(i, [f * ratio for f in net.cfg['filters_hz'][i]])
    if eid == DAMP_ID:
        for i in range(8):
            net.set_slider('damp', 160 + 80 * u[i], nodes=[i])
    else:
        net.set_matrix([[net.cfg['matrix'][i][j] * (.15 + 2.85 * u[i])
                         for j in range(8)] for i in range(8)])


class HypothesesTests(unittest.TestCase):
    def test_new_channels_match_independent_slow_model_and_keep_state(self):
        for eid in (DAMP_ID, LINKS_ID):
            e = engine(eid, grid())
            net = GutterNetwork(e.cfg)
            for shift in (0, 8, 0, 16):
                g = grid(shift)
                if shift != 0 or net.k:
                    before = e.node.copy()
                    e.update_field(g, None)
                    np.testing.assert_array_equal(e.node, before)
                reference_messages(net, g, eid)
                fast = np.concatenate([e.raw_block() for _ in range(9)])
                slow, _ = net.render(len(fast))
                np.testing.assert_array_equal(fast, slow)
            self.assertEqual(int(e.resets.sum()), 0)

    def test_every_region_controls_each_variant_and_restore_keeps_inflight_ramps(self):
        for eid in (gf.ENGINE_ID, DAMP_ID, LINKS_ID):
            for y in range(2):
                for x in range(4):
                    g = grid()
                    e = engine(eid, g)
                    for _ in range(5):
                        e.raw_block()
                    control = engine(eid, g)
                    control.restore_state(g, None, e.export_state())
                    changed = g.copy()
                    changed[16 * y, 8 * x] = 1
                    e.update_field(changed, None)
                    self.assertFalse(np.array_equal(e.ffreq, control.ffreq), eid)
                    if eid == DAMP_ID:
                        self.assertFalse(np.array_equal(e.ramps[gf.R_DAMP, 1],
                                                        control.ramps[gf.R_DAMP, 1]))
                    if eid == LINKS_ID:
                        self.assertFalse(np.array_equal(e.G_tgt, control.G_tgt))
                    first = e.raw_block()
                    self.assertFalse(np.array_equal(first, control.raw_block()), eid)
                    restored = engine(eid, changed)
                    restored.restore_state(changed, None, e.export_state())
                    for _ in range(9):
                        np.testing.assert_array_equal(e.raw_block(), restored.raw_block())

    def test_catalog_notes_replay_continue_and_no_overwrite(self):
        art = ROOT / 'artifacts' / 'n1_revision'
        art.mkdir(parents=True, exist_ok=True)
        hashes = []
        with tempfile.TemporaryDirectory(dir=art) as tmp:
            cat = Catalog(tmp, repo_root=None)
            for case in CASES:
                doc = scene_for(case)
                self.assertNotEqual(doc['variants']['A']['engine_id'],
                                    doc['variants']['B']['engine_id'])
                for s in ('A', 'B'):
                    p = doc['variants'][s]['engine_params']
                    self.assertEqual(p['freeze_ca'], 0)
                    self.assertGreater(p['depth'], 0)
                rid, _ = record_offline(cat, doc, .4, [], case['title'], case['hypothesis'])
                cat.write_notes(rid, case['hypothesis'] + '\n\nUser feedback: test.')
                rec = cat.load(rid)
                self.assertTrue(rec.notes.startswith('Гипотеза - '))
                result = cat.replay(rid, yield_cpu=False)
                self.assertEqual(result.status, 'match', result.reason)
                continued, state, _ = cat.continue_runner(rid)
                control = DemoRunner(scene_from_doc(doc))
                control.post('start', at=0)
                for _ in range(math.ceil(.4 * 44100 / BLOCK)):
                    control.next_block()
                first = None
                for _ in range(65):
                    a, b = continued.next_block(), control.next_block()
                    for s in ('A', 'B', 'monitor'):
                        np.testing.assert_array_equal(a.get(s), b.get(s))
                    if first is None:
                        first = a.A.tobytes()
                hashes.append(first)
                with self.assertRaises(SystemExit):
                    build(tmp)
                self.assertIn('User feedback: test.', cat.load(rid).notes)
            self.assertEqual(len(set(hashes)), len(CASES))


if __name__ == '__main__':
    unittest.main()
