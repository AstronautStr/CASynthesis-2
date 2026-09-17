"""Objects -- Birth strength (the force of the birth position) gates (REQ
memory/req-objects-event-source-modal-2026-09-17.md section 5, "Приёмка дополнения"):

  - registry: `birth_strength` 0..4 float, default 1, absent key = 1, finite and
    inside the range (registry, engine set_params, snapshot); stored but inactive
    with Uniform (hint with its value); every Objects scene loads
  - the law on the M1 phases: the profiles at s = 0 / 0.5 / 1 / 2 / 4 of every
    preflight bank and synthetic case (independent preflight JSON of 2026-09-18),
    the norm sum c^2 = m, exact zeros kept for s >= 1 and appearing for 0 < s < 1,
    equal b stay equal, one mode c = 1, s = 1 returns b itself, invariance to the
    sign / basis of b's eigenvectors inherited (c is a function of b only)
  - five engines on M1 side by side (s = 0 / 0.5 / 1 / 2 / 4): frequencies, output
    weights, packet moments, a and the tracker equal; the per-packet coefficients of
    every phase equal the preflight profile of that s
  - the PCM path: Birth position at s = 0 from the same start equals Uniform bit for
    bit; Uniform is the same at any stored s; s = 1 is the delivered v4 Birth
    position (the pinned M1 record of 2026-09-17 replays bit for bit); s changed on a
    still field makes no packet and leaves the ringing tail alone (through 0 and 1);
    a sequence of different s across transitions keeps the history of both paths and
    survives Continue; snapshot v5 continues exactly, a v4 snapshot imports as s = 1,
    refusals (v4 version with model v5, s outside the range)
  - the M2 scene: sides differ only in birth_strength (1 / 4), settings of M1, the
    side gain constant, on disk == builder, runner Continue exact (A / B / monitor),
    a live knob change through the runner journal continues exactly, copy_side
    carries the knob; the bench panel: 17 rows, the window fits an 880 px desktop,
    the hint line fits the panel, the knob is a slider with Birth position and an
    inactive text with Uniform

    python tests/test_objects_birth_strength.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_obs_tests/.
"""
import copy
import json
import math
import os
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, MASTER_GAIN                                    # noqa: E402
from casynth_engine import step, events_field                                 # noqa: E402
from casynth_lab import BLOCK, registry, DemoRunner, load_scene, scene_from_doc   # noqa: E402
from casynth_lab.engine_api import EngineContext                              # noqa: E402
from casynth_lab import object_resonators as orz                             # noqa: E402
from demos.build_objects_event_source import preflight_params, case_cells    # noqa: E402
from demos.build_objects_birth_strength import (CASES, PREFLIGHT_STRENGTH, SIDE_GAIN, STRENGTHS,   # noqa: E402
                                                scene_for, side_params)

ART = os.path.join(ROOT, 'artifacts', '_obs_tests')
F0 = 110.0
RATE = 6.0
CTX = EngineContext(SR, BLOCK, 2, F0, 1.0, RATE)
GAIN = MASTER_GAIN * 0.7
EID = orz.ENGINE_ID
ROWS = COLS = 32
BIRTHS, DEATHS = orz.EV_BIRTHS, orz.EV_DEATHS
UNIFORM, POSITION = orz.EXC_UNIFORM, orz.EXC_POSITION
OES_CATALOG = os.path.join(ROOT, 'lab_catalog', 'objects_event_source_modal_2026_09_17')
OBJECT_SCENES = ('n4_spectrum', 'n4_neighbor', 'ol_glider', 'ol_galaxy', 'ol_neighbor',
                 'ora_r1', 'ora_r2', 'ora_a1', 'ora_user034', 'oes_e1', 'oes_m1', 'obs_m2')


def preflight_strength():
    with open(PREFLIGHT_STRENGTH, encoding='utf-8') as f:
        return json.load(f)


def params(strength=1.0, ex=POSITION, **over):
    p = preflight_params(BIRTHS, ex)
    p['birth_strength'] = float(strength)
    p.update(over)
    return p


def grid(cells, rows=ROWS, cols=COLS):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def engine(cells, p, gain=GAIN):
    e = registry.create(EID, CTX, p)
    e.init(grid(cells), None, gain)
    return e


def run(e, n_blocks, gain=GAIN):
    out = []
    for _ in range(n_blocks):
        y, _pk, _nc = e.render_float(gain)
        out.append(y)
    return np.concatenate(out) if out else np.zeros((0, 2))


def expected_profile(b, s):
    """The REQ 5.2 law written independently of the engine (for the norm checks)."""
    b = np.asarray(b, np.float64)
    m = len(b)
    if s == 0:
        return np.ones(m)
    if s == 1:
        return b.copy()
    v = (1 - s) + s * b if s < 1 else b ** s
    return math.sqrt(m) * v / math.sqrt(float((v * v).sum()))


def rendered_like(rec):
    """Render the record's embedded scene (a 2026-09-17 record: no birth_strength key ->
    the default 1 added, journal = one 'start' at 0) with the CURRENT code and compare
    every output with the stored WAV bit for bit; {output: equal}."""
    doc = copy.deepcopy(rec.meta['scene'])
    for group in ('variants', 'factory_variants'):
        for side, var in (doc.get(group) or {}).items():
            if var.get('engine_id') == EID:
                var['engine_params'].setdefault('birth_strength', 1.0)
    for side, per_engine in (doc.get('param_memory') or {}).items():
        if EID in per_engine:
            per_engine[EID].setdefault('birth_strength', 1.0)
    cmds = [j for j in rec.meta['journal'] if j['kind'] != 'step']
    assert [(j['kind'], j['out_sample']) for j in cmds] == [('start', 0)], cmds
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    ref = {o: rec.pcm(o) for o in ('A', 'B', 'monitor')}
    n = ref['A'].shape[0] // BLOCK
    out = {o: [] for o in ref}
    for _ in range(n):
        blk = runner.next_block()
        for o in ref:
            out[o].append(np.array(blk.get(o), copy=True))
    return {o: bool(np.array_equal(np.concatenate(out[o]), ref[o])) for o in ref}


class RegistryTests(unittest.TestCase):
    def test_spec_defaults_validation_hints_and_every_objects_scene(self):
        spec = registry.get(EID)
        self.assertEqual(spec.spec_of('birth_strength'), ('birth_strength', 'Birth strength', 0.0, 4.0, False, 1.0))
        names = [p[0] for p in spec.params]
        self.assertEqual(names.index('birth_strength'), names.index('excitation') + 1)
        self.assertEqual(orz.OPTIONAL_PARAMS['birth_strength'], 1.0)
        self.assertEqual((orz.MODEL_VERSION, orz.STATE_VERSION), ('ca_object_resonators_n4_v5', 5))
        self.assertEqual(orz.COMPATIBLE_STATES[4], 'ca_object_resonators_n4_v4')
        self.assertEqual(spec.defaults()['birth_strength'], 1.0)
        self.assertEqual(spec.defaults()['excitation'], UNIFORM)                # Uniform stays the default
        # absent key = 1 (older parameter sets), stored value read back
        e = registry.create(EID, CTX, dict(detector=0, frequency_scale=220.0, decay_s=0.8, attack_ms=0.0))
        self.assertEqual(e.params['birth_strength'], 1.0)
        self.assertEqual(e._birth_strength(), 1.0)
        e.set_params(dict(e.params, birth_strength=2.5))
        self.assertEqual(e._birth_strength(), 2.5)
        # finite and inside 0..4: registry, engine
        for bad in (-0.001, 4.001, 5, float('nan'), float('inf'), -float('inf'), 'x', True):
            with self.assertRaises(ValueError):
                registry.validate_param(EID, 'birth_strength', bad)
            self.assertFalse(orz.valid_birth_strength(bad))
        for bad in (-0.001, 4.001, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                e.set_params(dict(e.params, birth_strength=bad))
            self.assertEqual(e._birth_strength(), 2.5)                        # the old parameters stay
        for ok in (0, 0.0, 0.5, 1, 4.0):
            self.assertEqual(registry.validate_param(EID, 'birth_strength', ok), float(ok))
            self.assertTrue(orz.valid_birth_strength(ok))
        # inactive with Uniform (value shown), active with Birth position
        ina = spec.inactive(params(2.5, UNIFORM))
        self.assertEqual(ina['birth_strength'], '2.50  Birth position only')
        self.assertNotIn('birth_strength', spec.inactive(params(2.5, POSITION)))
        self.assertNotIn('birth_strength', spec.inactive(params(2.5, POSITION, detector=orz.DET_DISK)))
        self.assertFalse(orz.position_supported(params(0.0, POSITION, detector=orz.DET_DISK)))   # s = 0 no bypass
        d = engine(case_cells('M1'), params(3.0)).display()
        self.assertEqual(d['birth_strength'], 3.0)
        self.assertEqual(d['birth_strength_hint'], orz.BIRTH_STRENGTH_HINT)
        for name in OBJECT_SCENES:
            sc = load_scene(os.path.join(ROOT, 'demos', name + '.json'))
            for side in ('A', 'B'):
                eid, p = sc.variants[side]
                if eid == EID:
                    self.assertIn('birth_strength', p)
                    self.assertEqual(p['birth_strength'], 4.0 if (name, side) == ('obs_m2', 'B') else 1.0)


class LawTests(unittest.TestCase):
    def test_profiles_match_the_independent_preflight_and_the_edge_cases(self):
        pf = preflight_strength()
        n = 0
        for case in pf['cases'] + pf['synthetic']:
            b = np.asarray(case.get('weights_spatial', case.get('b')), np.float64)
            for prof in case['profiles']:
                s = float(prof['strength'])
                want = np.asarray(prof['b'], np.float64)
                if s == 0.0:
                    # the engine takes the Uniform path at s = 0 (c = 1 even with zero participation)
                    got = np.ones(len(b)) if int(case.get('births', 1)) > 0 else np.zeros(len(b))
                elif not b.any():
                    got = np.zeros(len(b))                                     # no packet, never blended
                else:
                    got = orz.birth_strength_profile(b, s)
                np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12, err_msg=f"{case} s={s}")
                np.testing.assert_allclose(got, expected_profile(b, s) if b.any() else got, rtol=1e-12, atol=1e-12)
                if b.any():
                    self.assertAlmostEqual(float((got * got).sum()), float(len(b)), places=9)   # sum c^2 = m
                n += 1
        self.assertGreaterEqual(n, 50)
        b16 = np.asarray([0.7814996203212149, 0.7037460155649663, 1.3762266851846143])
        table = {0.5: [0.901214, 0.861881, 1.202071], 1: [0.781500, 0.703746, 1.376227],
                 2: [0.515832, 0.418295, 1.599671], 4: [0.178722, 0.117524, 1.718792]}   # REQ 5.2 control example
        for s, want in table.items():
            np.testing.assert_allclose(orz.birth_strength_profile(b16, s), want, atol=6e-7)
        self.assertIs(orz.birth_strength_profile(b16, 1.0), b16)                # s = 1: b itself, no arithmetic
        zero = np.asarray([0.0, 1.0, math.sqrt(2.0)])
        for s in (1.0, 1.5, 2.0, 4.0):
            self.assertEqual(float(orz.birth_strength_profile(zero, s)[0]), 0.0)   # an exact zero stays zero
        for s in (0.25, 0.5, 0.75):
            self.assertGreater(float(orz.birth_strength_profile(zero, s)[0]), 0.0)   # ... and appears below 1
        for s in (0.5, 2.0, 4.0):
            np.testing.assert_allclose(orz.birth_strength_profile(np.ones(3), s), np.ones(3), atol=1e-15)
            np.testing.assert_allclose(orz.birth_strength_profile(np.ones(1), s), [1.0], atol=1e-15)
            c = orz.birth_strength_profile(np.asarray([1.2, 1.2, 0.3]), s)
            self.assertEqual(float(c[0]), float(c[1]))                         # equal b stay equal
            self.assertTrue(np.all(c >= 0.0))
        for bad in (0.0, -1.0, 4.5, float('nan')):
            with self.assertRaises(ValueError):
                orz.birth_strength_profile(b16, bad)
        # c depends on b only: the same b from a sign-flipped / rotated basis gives the same c
        np.testing.assert_array_equal(orz.birth_strength_profile(b16.copy(), 2.0), orz.birth_strength_profile(b16, 2.0))

    def test_five_engines_on_m1_share_everything_but_the_coefficients_which_match_the_preflight(self):
        cells = case_cells('M1')
        pf = preflight_strength()
        engines = {s: engine(cells, params(s)) for s in STRENGTHS}
        ref = engines[1.0]
        g = grid(cells)
        seen = {s: {} for s in STRENGTHS}
        for t in range(0, 40):
            if t > 0:
                prev, g = g, step(g)
                for e in engines.values():
                    e.update_field(g, events_field(prev, g))
            for _ in range(int(round(SR / RATE / BLOCK))):
                for e in engines.values():
                    e.render_float(GAIN)
                for s, e in engines.items():
                    self.assertEqual(sorted(e.figures), sorted(ref.figures))
                    for fid, fr in ref.figures.items():
                        sr_, se = fr.slot, e.figures[fid].slot
                        self.assertEqual(sr_, se)
                        if sr_ < 0:
                            continue
                        np.testing.assert_array_equal(ref.ffreq[sr_], e.ffreq[se])      # frequencies
                        np.testing.assert_array_equal(ref.wtgt[sr_], e.wtgt[se])        # output weights
                        self.assertEqual(float(ref.last_e[sr_]), float(e.last_e[se]))   # packet moments
                        self.assertEqual(float(ref.last_a[sr_]), float(e.last_a[se]))   # a
            dr = ref.display()
            for s, e in engines.items():
                d = e.display()
                self.assertEqual([f['id'] for f in d['figures']], [f['id'] for f in dr['figures']])
                self.assertEqual(d['zero_participation'], 0)
                self.assertEqual(d['unsupported_packets'], 0)
                for f in d['figures']:
                    if f['slot'] < 0 or f['e'] <= 0.0:
                        continue
                    c = np.asarray(f['b'])
                    self.assertAlmostEqual(float((c * c).sum()), float(f['modes']), places=9)
                    seen[s].setdefault(((t - 1) % 3 + 1, f['n'], int(f['e'])), c)
        # every preflight phase at every strength
        checked = 0
        for case in pf['cases']:
            if int(case['births']) <= 0:
                continue
            key = (int(case['generation']), int(case['cells']), int(case['births']))
            for prof in case['profiles']:
                s = float(prof['strength'])
                self.assertIn(key, seen[s])
                np.testing.assert_allclose(seen[s][key], prof['b'], rtol=1e-9, atol=1e-9)
                checked += 1
        self.assertGreaterEqual(checked, 25)
        # s = 0: the uniform packet (no per-mode states at all); s = 1: per-mode states only
        self.assertEqual(float(np.abs(engines[0.0].zfm).max()), 0.0)
        self.assertGreater(float(np.abs(engines[0.0].zf).max()), 0.0)
        self.assertGreater(float(np.abs(engines[1.0].zfm).max()), 0.0)


class PathTests(unittest.TestCase):
    def _gens(self, cells, n):
        g = grid(cells)
        out = [g]
        for _ in range(n):
            g = step(g)
            out.append(g)
        return out

    def _drive(self, e, gens, blocks=5):
        y = [run(e, blocks)]
        for k in range(1, len(gens)):
            e.update_field(gens[k], events_field(gens[k - 1], gens[k]))
            y.append(run(e, blocks))
        return np.concatenate(y)

    def test_s0_is_uniform_uniform_ignores_s_and_s1_replays_the_pinned_m1_record(self):
        cells = case_cells('M1')
        gens = self._gens(cells, 12)
        y_uniform = self._drive(engine(cells, params(1.0, UNIFORM)), gens)
        y_uniform_s4 = self._drive(engine(cells, params(4.0, UNIFORM)), gens)
        y_pos_s0 = self._drive(engine(cells, params(0.0, POSITION)), gens)
        y_pos_s1 = self._drive(engine(cells, params(1.0, POSITION)), gens)
        np.testing.assert_array_equal(y_uniform, y_uniform_s4)               # Uniform: any stored s
        np.testing.assert_array_equal(y_uniform, y_pos_s0)                   # s = 0: the Uniform path, bit for bit
        self.assertGreater(float(np.abs(y_uniform - y_pos_s1).max()), 1e-6)
        # s = 1 is the delivered Birth position and Uniform is untouched: the pinned
        # records of 2026-09-17 (scenes without the key; by the user's rule they are
        # never rebuilt with substituted values, so this is a direct comparison of
        # their stored WAVs with the current code on the same scene + the default s)
        from casynth_lab.catalog import Catalog
        if not os.path.isdir(OES_CATALOG):
            self.skipTest('the 2026-09-17 catalog is not present')
        cat = Catalog(OES_CATALOG)
        recs = [rec for rec, err in cat.list() if err is None]
        self.assertEqual(len(recs), 2)
        for rec in recs:
            self.assertEqual(rendered_like(rec), dict(A=True, B=True, monitor=True), rec.title)

    def test_s_changed_on_a_still_field_makes_no_packet_and_a_sequence_keeps_both_histories(self):
        cells = case_cells('M1')
        gens = self._gens(cells, 2)
        e = engine(cells, params(1.0))
        twin = engine(cells, params(1.0))
        for x in (e, twin):
            run(x, 3)
            x.update_field(gens[1], events_field(gens[0], gens[1]))
            run(x, 2)                                                  # mid-pulse per-mode states
        zfm0 = e.zfm.copy()
        for s in (0.0, 1.0, 4.0, 0.5, 1.0, 0.0, 2.0):                   # through 0 and 1
            e.set_params(dict(e.params, birth_strength=s))
            np.testing.assert_array_equal(e.render_float(GAIN)[0], twin.render_float(GAIN)[0])
            self.assertEqual(e.display()['changes'], twin.display()['changes'])
        np.testing.assert_array_equal(e.zf, twin.zf)
        np.testing.assert_array_equal(e.zfm, twin.zfm)                  # states untouched (decayed alike)
        self.assertLess(float(np.abs(e.zfm).max()), float(np.abs(zfm0).max()))
        # a sequence of different s across transitions: each packet takes its own s,
        # the per-mode history of s > 0 is not erased by s = 0 (the uniform path adds to zf)
        cells = case_cells('E1')                                      # one bank the whole period
        gens = self._gens(cells, 6)
        seq = [4.0, 0.0, 0.5, 1.0, 2.0, 0.0]
        e = engine(cells, params(seq[0]))
        run(e, 3)
        cs = []
        kept = 0
        for k in range(1, len(gens)):
            e.set_params(dict(e.params, birth_strength=seq[k - 1]))
            zfm_before = e.zfm.copy()
            zf_before = e.zf.copy()
            e.update_field(gens[k], events_field(gens[k - 1], gens[k]))
            run(e, 1)
            d = e.display()
            for f in d['figures']:
                if f['slot'] >= 0 and f['e'] > 0:
                    cs.append((seq[k - 1], np.asarray(f['b'])))
            if seq[k - 1] == 0.0 and float(np.abs(zfm_before).max()) > 0.0 and d['figures'][0]['e'] > 0:
                self.assertGreater(float(np.abs(e.zfm).max()), 0.0)          # per-mode history kept at s = 0
                self.assertGreater(float(np.abs(e.zf).max()), float(np.abs(zf_before).max()) * 0.5)   # uniform path fed
                kept += 1
        self.assertGreaterEqual(len(cs), 4)
        for s, c in cs:
            if s == 0.0:
                np.testing.assert_array_equal(c, np.ones(len(c)))
            else:
                self.assertAlmostEqual(float((c * c).sum()), float(len(c)), places=9)
        self.assertTrue(any(s == 0.0 for s, _c in cs))
        self.assertTrue(any(s > 1.0 for s, _c in cs))
        self.assertGreaterEqual(kept, 1)

    def test_snapshot_v5_continues_v4_imports_as_s1_and_refusals(self):
        cells = case_cells('M1')
        gens = self._gens(cells, 4)
        e = engine(cells, params(4.0))
        run(e, 2)
        e.update_field(gens[1], events_field(gens[0], gens[1]))
        run(e, 2)
        st = e.export_state()
        self.assertEqual((st['version'], st['model_version']), (5, orz.MODEL_VERSION))
        self.assertEqual(st['params']['birth_strength'], 4.0)
        twin = registry.create(EID, CTX, dict(e.params))
        twin.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
        twin.restore_state(e._grid, e._exc, copy.deepcopy(st))
        self.assertEqual(twin._birth_strength(), 4.0)
        for k in range(2, 4):
            for x in (e, twin):
                x.update_field(gens[k], events_field(gens[k - 1], gens[k]))
            for _ in range(20):
                np.testing.assert_array_equal(e.render_float(GAIN)[0], twin.render_float(GAIN)[0])
        # a v4 snapshot (no key) of an s = 1 run imports as s = 1: the same sound
        e1 = engine(cells, params(1.0))
        run(e1, 2)
        e1.update_field(gens[1], events_field(gens[0], gens[1]))
        run(e1, 2)
        st4 = e1.export_state()
        st4['version'] = 4
        st4['model_version'] = orz.COMPATIBLE_STATES[4]
        st4['params'] = {k: v for k, v in st4['params'].items() if k != 'birth_strength'}
        old = registry.create(EID, CTX, dict(e1.params))
        old.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
        old.restore_state(e1._grid, e1._exc, copy.deepcopy(st4))
        self.assertEqual(old._birth_strength(), 1.0)
        for k in range(2, 4):
            for x in (e1, old):
                x.update_field(gens[k], events_field(gens[k - 1], gens[k]))
            for _ in range(20):
                np.testing.assert_array_equal(e1.render_float(GAIN)[0], old.render_float(GAIN)[0])
        # refusals
        for bad_state in (dict(copy.deepcopy(st), version=4),                             # v4 must say model v4
                          dict(copy.deepcopy(st), params=dict(st['params'], birth_strength=4.5)),
                          dict(copy.deepcopy(st), params=dict(st['params'], birth_strength=float('nan')))):
            fresh = registry.create(EID, CTX, dict(e.params))
            fresh.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
            with self.assertRaises(ValueError):
                fresh.restore_state(e._grid, e._exc, bad_state)


class SceneTests(unittest.TestCase):
    def test_m2_scene_sides_differ_only_in_the_strength_and_the_runner_continues_exactly(self):
        case = CASES[0]
        doc = scene_for(case)
        sc = scene_from_doc(doc)
        self.assertEqual(doc['cells'], case_cells('M1'))
        self.assertEqual(doc['rate_hz'], 6.0)
        self.assertTrue(case['hypothesis'].startswith('Гипотеза - '))
        self.assertIn('Ручку Birth strength можно двигать', case['hypothesis'])
        (ea, pa), (eb, pb) = sc.variants['A'], sc.variants['B']
        self.assertEqual((ea, eb), (EID, EID))
        self.assertEqual({k for k in pa if pa[k] != pb[k]}, {'birth_strength'})
        self.assertEqual((pa['birth_strength'], pb['birth_strength']), (1.0, 4.0))
        self.assertEqual((pa['events'], pa['excitation'], pa['detector'], pa['spectrum'], pa['fullshape']),
                         (BIRTHS, POSITION, 0, 1, 1))
        self.assertEqual((pa['attack_ms'], pa['decay_s'], pa['n'], pa['spread'], pa['harm']), (4.0, 1.39, 3, 1.0, 0.87))
        self.assertEqual((pa['shape'], pa['alpha'], pa['dyn']), (0.0, 0.0, 0.0))
        self.assertEqual(pa, side_params(1.0))
        self.assertEqual(doc['audio']['side_gain'], SIDE_GAIN['obs_m2'])
        self.assertNotEqual(SIDE_GAIN['obs_m2']['B'], 0.93)                    # measured for 1 / 4, not carried over
        on_disk = load_scene(os.path.join(ROOT, 'demos', 'obs_m2.json'))
        self.assertEqual(on_disk.variants, sc.variants)
        self.assertEqual(on_disk.side_gain, doc['audio']['side_gain'])
        runner = DemoRunner(sc)
        runner.post('start', at=0)
        n = int(6.0 * SR / BLOCK)
        for _ in range(n):
            runner.next_block()
        st = runner.export_state()
        twin = DemoRunner.from_state(st)
        gen0 = twin.gen
        for _ in range(200):
            a, b = runner.next_block(), twin.next_block()
            for s in ('A', 'B', 'monitor'):
                np.testing.assert_array_equal(a.get(s), b.get(s))
        self.assertGreater(twin.gen, gen0)
        snap = runner.snapshot()
        self.assertEqual(snap['clip_blocks'], dict(A=0, B=0))
        for side, s in (('A', 1.0), ('B', 4.0)):
            d = snap['display'][side]
            self.assertEqual((d['excitation_name'], d['birth_strength'], d['unsupported_packets']),
                             ('Birth position', s, 0))
            self.assertTrue(d['position_supported'])
        # a live knob change through the journal (the bench's path): validated, applied,
        # continues exactly from a snapshot taken after it, and copy_side carries it
        with self.assertRaises(ValueError):
            runner.post('set_param', side='B', name='birth_strength', value=4.5)
        runner.post('set_param', side='B', name='birth_strength', value=2.0)
        runner.post('set_param', side='B', name='birth_strength', value=0.0, at=runner.out_samples + 40 * BLOCK)
        for _ in range(80):
            runner.next_block()
        self.assertEqual(runner.snapshot()['sides']['B'][1]['birth_strength'], 0.0)
        st2 = runner.export_state()
        twin2 = DemoRunner.from_state(st2)
        self.assertEqual(twin2.snapshot()['sides']['B'][1]['birth_strength'], 0.0)
        for _ in range(120):
            a, b = runner.next_block(), twin2.next_block()
            for s in ('A', 'B', 'monitor'):
                np.testing.assert_array_equal(a.get(s), b.get(s))
        runner.post('copy_side', src='B', dst='A')
        runner.next_block()
        self.assertEqual(runner.snapshot()['sides']['A'][1]['birth_strength'], 0.0)
        for _ in range(20):
            self.assertTrue(np.all(np.isfinite(runner.next_block().get('B'))))

    def test_headless_bench_panel_rows_height_hint_line_and_the_knob_states(self):
        import pygame
        import demo_bench as db
        from casynth_lab.audio_out import LiveEngine
        scene = scene_from_doc(scene_for(CASES[0]))
        runner = DemoRunner(scene)
        eng = LiveEngine(runner, sink=lambda m, b: None)
        pygame.init()
        app = db.BenchApp(scene, eng)
        n_rows = len(registry.get(EID).params) + len(registry.get(EID).ranges)
        self.assertEqual(n_rows, 17)
        self.assertEqual(app._params_height(EID), n_rows * db.ROW_H)
        low = db.BenchApp(scene, eng, max_height=880)
        self.assertLessEqual(low.height, 880)
        self.assertGreaterEqual(low.figure_rows, db.FIGURE_ROWS_MIN)
        self.assertEqual(db.FIGURE_HEAD_H, 62)
        rows = {spec[0]: rect for spec, rect in app._param_rows(EID)}
        self.assertIn('birth_strength', rows)
        # the rows do not overlap: a choice button and a range field end inside their row
        self.assertLessEqual(db.CHOICE_H + 3, db.ROW_H + 2)
        self.assertLessEqual(db.RANGE_FIELD_H, db.ROW_H)
        screen = pygame.Surface((app.width, app.height))
        font, small = pygame.font.SysFont(db.FONT_NAMES, 17), pygame.font.SysFont(db.FONT_NAMES, 14)
        limit = db.PANEL_W + db.MARGIN - 2                                     # up to the pattern column
        self.assertLessEqual(small.size(db.BIRTH_STRENGTH_LINE)[0], limit)
        self.assertLessEqual(small.size("Birth strength 4.00 stored (Birth position only)")[0], limit)
        self.assertLessEqual(small.size('Birth strength')[0], app.slider_x - app.panel_x)
        eng.start()
        try:
            eng.post('start')
            eng.post('select', side='B')
            time.sleep(0.4)
            app.draw(screen, font, small)
            snap = eng.snapshot()
            self.assertEqual(snap['display']['B']['birth_strength'], 4.0)
            self.assertNotIn('birth_strength', app._inactive(EID, snap['sides']['B'][1]))
            sx, sy, sw, sh = rows['birth_strength']
            arr = pygame.surfarray.array3d(screen).astype(int)
            acc = np.asarray(db.C_ACCENT)
            band = arr[sx:sx + sw, sy:sy + sh]
            self.assertTrue((np.abs(band - acc).sum(axis=-1) < 30).any())     # the slider is drawn (accent fill)
            # the hint line under the Objects header (4th line) has ink in both knob states
            y0 = app.params_y + n_rows * db.ROW_H + 6

            def ink(y_lo, y_hi):
                return int((arr[app.panel_x:app.panel_x + db.PANEL_W, y_lo:y_hi].sum(axis=-1)
                            > np.asarray(db.C_BG).sum() + 60).sum())
            self.assertGreater(ink(y0 + 42, y0 + 56), 50)
            # the knob drag posts a valid value (the slider spans 0..4)
            self.assertEqual(app.press((sx + sw // 2, sy + 2), 1), 'param:birth_strength')
            app.release()
            time.sleep(0.4)
            v = eng.snapshot()['sides']['B'][1]['birth_strength']
            self.assertTrue(0.0 <= v <= 4.0 and abs(v - 2.0) < 0.2, v)
            # Uniform side: the knob is an inactive text with its value, the header says stored
            eng.post('select', side='A')
            eng.post('set_param', side='A', name='excitation', value=UNIFORM)
            time.sleep(0.4)
            screen.fill(db.C_BG)
            app.draw(screen, font, small)
            snap = eng.snapshot()
            self.assertEqual(app._inactive(EID, snap['sides']['A'][1])['birth_strength'], '1.00  Birth position only')
            arr = pygame.surfarray.array3d(screen).astype(int)
            band = arr[sx:sx + sw, sy:sy + sh]
            self.assertFalse((np.abs(band - acc).sum(axis=-1) < 30).any())    # no slider fill
            self.assertGreater(ink(y0 + 42, y0 + 56), 50)
            self.assertIsNone(app.press((sx + sw // 2, sy + 2), 1))            # not editable
            app.release()
        finally:
            eng.stop()
            pygame.quit()


if __name__ == '__main__':
    os.makedirs(ART, exist_ok=True)
    unittest.main(verbosity=1)
