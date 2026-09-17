#!/usr/bin/env python3
"""Objects -- radius range and excitation attack gates
(REQ memory/req-objects-radius-attack-2026-09-17.md):

  - Radius x has no technical ceiling: any finite value >= 0 validates (registry,
    scene), the slider covers a user range (registry hint `ranges`, default
    0.25..4) edited per side through the runner command set_range: validation,
    A / B independence, survival across engine switches, snapshot / Continue,
    scene `param_ranges`, records (save -> bench_scene -> continue); an older scene
    / snapshot without a range gets 0.25 / 4; a range change never touches the sound
  - the Disk mask at x8 / x32 equals an independent torus-distance computation, a
    radius from figures.full_cover_radius on is the equivalent full mask (no
    squaring of huge radii), R = 0 stays one cell; `covers_all` in display()
  - the two radius pairs of the REQ on the preflight fields (memory/research/
    objects-radius-attack-preflight-2026-09-17.json): every packet of every bank
    equals the independent count of changed cells inside the figure's own circle
    (births in the current circle, deaths in the previous one), the foreign part
    (cells of the other family) per bank and per generation, the packet
    distribution, the preflight totals; the joint evolution equals the union of the
    two independent evolutions; a figure shifted out of the circle stops
    contributing; the knob / range alone never makes a packet
  - the bench: Min / Max fields (commit by Enter / leaving, Esc, Tab, refused
    numbers), the slider over a narrow range with 0.001 steps, the one-time clamp,
    hotkeys inert while typing, the full-cover frame, panel height
  - Attack (part 2 of the REQ): the coefficient formula and its 10 -> 90 % rise time,
    Attack 0 = the v1/v2 scalar reference of the N4 gate bit for bit, the kernel
    against an independent scalar reference of the smoothing through strikes, ramps
    and zero crossings (exact bypass after a ramp back to 0), silence without events,
    unchanged packet moments and pulse states, Continue mid-pulse / mid-ramp / with
    overlapping events, a new front without a bank reset, tails without excitation,
    a v2 snapshot = Attack 0, the level / brightness change of the probe

    python tests/test_objects_radius_attack.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_ora_tests/.
"""
import copy
import json
import math
import os
import sys
import tempfile
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
from casynth_lab.registry import EngineSpec                                   # noqa: E402
from casynth_lab.scene import SceneError                                      # noqa: E402
from casynth_lab import figures as fg                                        # noqa: E402
from casynth_lab import object_resonators as orz                             # noqa: E402

ART = os.path.join(ROOT, 'artifacts', '_ora_tests')
F0 = 110.0
RATE = 6.0
CTX = EngineContext(SR, BLOCK, 2, F0, 1.0, RATE)
GAIN = MASTER_GAIN * 0.7
TOL = 1e-12
PREFLIGHT = os.path.join(ROOT, 'memory', 'research', 'objects-radius-attack-preflight-2026-09-17.json')
EID = orz.ENGINE_ID
ROWS = COLS = 32


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def groups_of(case):
    """The two independently placed families of a preflight case: [[(r, c)...], ...]."""
    return [[(int(r), int(c)) for r, c in g] for g in preflight()['cases'][case]['groups']]


def screenshot_params(**over):
    """The user's settings of the REQ screenshot (Disk, Laplace, the seven, Decay 1.39)."""
    pf = preflight()['settings']
    p = dict(detector=int(pf['detector']), radius_mul=float(pf['radius_mul']), spectrum=int(pf['spectrum']),
             frequency_scale=float(pf['frequency_scale']), decay_s=float(pf['decay_s']),
             n=int(pf['n']), spread=float(pf['spread']), alpha=float(pf['alpha']), shape=float(pf['shape']),
             harm=float(pf['harm']), fullshape=int(pf['fullshape']), dyn=float(pf['dyn']))
    for k, v in orz.OPTIONAL_PARAMS.items():
        p.setdefault(k, v)
    p.update(over)
    return p


def grid(cells, rows=ROWS, cols=COLS):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def engine(cells, params=None, gain=GAIN, exc=None):
    e = registry.create(EID, CTX, screenshot_params() if params is None else params)
    e.init(grid(cells), exc, gain)
    return e


def run(e, n_blocks, gain=GAIN):
    out = []
    for _ in range(n_blocks):
        y, _pk, _nc = e.render_float(gain)
        out.append(y)
    return np.concatenate(out) if out else np.zeros((0, 2))


def torus_disk(centre, radius, rows=ROWS, cols=COLS):
    """An INDEPENDENT disk mask: every cell whose shortest torus distance to the
    centre is <= radius (+ 1e-9), computed cell by cell without figures.py."""
    m = np.zeros((rows, cols), bool)
    cy, cx = float(centre[0]), float(centre[1])
    for r in range(rows):
        dr = abs(r - cy)
        dr = min(dr, rows - dr)
        for c in range(cols):
            dc = abs(c - cx)
            dc = min(dc, cols - dc)
            m[r, c] = math.hypot(dr, dc) <= radius + 1e-9
    return m


def evolve(cells, transitions):
    g = grid(cells)
    out = [g]
    for _ in range(transitions):
        g = step(g)
        out.append(g)
    return out


def engine_generations(cells, params, transitions, on_boundary=None):
    """Run the engine generation by generation (one block per generation, exc of
    the transition): [(grid, display)] for generations 0..transitions."""
    e = registry.create(EID, CTX, params)
    g = grid(cells)
    e.init(g, None, GAIN)
    e.render_float(GAIN)
    out = [(g.copy(), e.display())]
    for _ in range(transitions):
        prev, g = g, step(g)
        e.update_field(g, events_field(prev, g))
        if on_boundary is not None:
            on_boundary(e)
        e.render_float(GAIN)
        out.append((g.copy(), e.display()))
    return e, out


def independent_events(gens, families):
    """The REQ rule computed without the engine: for every generation t >= 1 and
    every figure id of the engine's display, expected packet e = changed cells
    inside its circle (births in the current circle, deaths in the previous one;
    a new id: births only) and the FOREIGN part (changed cells outside the
    family's own independent field).  `families` = [[grid per generation]].
    Returns {t: {id: (e, foreign, family_index)}}."""
    out = {}
    for t in range(1, len(gens)):
        g_prev, d_prev = gens[t - 1]
        g_cur, d_cur = gens[t]
        births = (g_cur != 0) & (g_prev == 0)
        deaths = (g_prev != 0) & (g_cur == 0)
        prev_by_id = {f['id']: f for f in d_prev['figures']}
        row = {}
        for f in d_cur['figures']:
            cells = np.asarray(f['cells'])
            fam = [k for k, fields in enumerate(families) if fields[t][cells[:, 0], cells[:, 1]].all()]
            assert len(fam) == 1, f"figure {f['id']} at generation {t} belongs to {fam}"
            fam = fam[0]
            own_cur = families[fam][t] != 0
            own_prev = families[fam][t - 1] != 0
            k_cur = torus_disk(f['centre'], f['radius'])
            e = int(np.count_nonzero(births & k_cur))
            foreign = int(np.count_nonzero(births & k_cur & ~own_cur))
            p = prev_by_id.get(f['id'])
            if p is not None:
                k_prev = torus_disk(p['centre'], p['radius'])
                e += int(np.count_nonzero(deaths & k_prev))
                foreign += int(np.count_nonzero(deaths & k_prev & ~own_prev))
            row[f['id']] = (e, foreign, fam)
        out[t] = row
    return out


# ======================================================================================
class RegistryAndValidationTests(unittest.TestCase):
    def test_radius_has_no_ceiling_and_a_default_user_range(self):
        spec = registry.get(EID)
        self.assertEqual(spec.spec_of('radius_mul'), ('radius_mul', 'Radius x', 0.0, math.inf, False, 1.0))
        self.assertEqual(spec.ranges, {'radius_mul': (0.25, 4.0)})
        self.assertEqual(orz.RADIUS_RANGE, (0.25, 4.0))
        self.assertEqual(registry.default_range(EID, 'radius_mul'), (0.25, 4.0))
        self.assertIsNone(registry.default_range(EID, 'decay_s'))
        for v in (0, 0.01, 0.34, 4.0, 8, 32, 1e6, 1e300):
            self.assertEqual(registry.validate_param(EID, 'radius_mul', v), float(v))
        for v in (-0.1, -1, math.inf, -math.inf, math.nan, True, '2'):
            with self.assertRaises(ValueError):
                registry.validate_param(EID, 'radius_mul', v)
        # scenes: the same rule
        doc = scene_doc(dict(radius_mul=32.0))
        scene_from_doc(doc)
        for v in (-1.0, math.inf):
            bad = copy.deepcopy(doc)
            bad['variants']['B']['engine_params']['radius_mul'] = v
            with self.assertRaises(SceneError):
                scene_from_doc(bad)

    def test_validate_range_and_the_registry_hint_rules(self):
        self.assertEqual(registry.validate_range(EID, 'radius_mul', 0.3, 0.6), (0.3, 0.6))
        self.assertEqual(registry.validate_range(EID, 'radius_mul', 0, 1e9), (0.0, 1e9))
        self.assertEqual(registry.validate_range(EID, 'radius_mul', 1, 32), (1.0, 32.0))
        for lo, hi in ((0.6, 0.3), (0.5, 0.5), (-0.1, 1.0), (0.0, math.inf), (math.nan, 1.0), (True, 2.0), ('0', 1)):
            with self.assertRaises(ValueError):
                registry.validate_range(EID, 'radius_mul', lo, hi)
        with self.assertRaises(ValueError):
            registry.validate_range(EID, 'decay_s', 0.2, 1.0)          # no user range
        with self.assertRaises(ValueError):
            registry.validate_range('laplacian', 'n', 1, 2)
        with self.assertRaises(ValueError):
            registry.validate_range('no_such_engine', 'radius_mul', 1, 2)
        # a hint must name a float parameter and sit inside its validation bounds
        params = [('x', 'X', 0.0, 10.0, False, 1.0), ('k', 'K', 0, 3, True, 1)]
        for bad in ({'x': (5.0, 5.0)}, {'x': (-1.0, 5.0)}, {'x': (1.0, 11.0)}, {'k': (0, 2)},
                    {'y': (0.0, 1.0)}, {'x': (0.0, math.inf)}):
            with self.assertRaises(ValueError):
                registry.register(EngineSpec('_ora_test', 'T', params, lambda c, p: None, ranges=bad))
            self.assertNotIn('_ora_test', registry.REGISTRY)
        registry.register(EngineSpec('_ora_test', 'T', params, lambda c, p: None, ranges={'x': (1, 2)}))
        try:
            self.assertEqual(registry.get('_ora_test').ranges, {'x': (1.0, 2.0)})
        finally:
            registry.unregister('_ora_test')


def scene_doc(over_b=None, over_a=None, cells=None, ranges=None, side_gain=None):
    pa, pb = screenshot_params(**(over_a or {})), screenshot_params(**(over_b or {}))
    d = dict(format=2, id='ora_test', title='ora test', grid=dict(rows=ROWS, cols=COLS),
             cells=[[int(r), int(c)] for r, c in (cells if cells is not None else sum(groups_of('receiver'), []))],
             rule='B3/S23', boundary='torus', rate_hz=RATE,
             audio=dict(f0_hz=F0, level=1.0, side_gain=dict(side_gain or dict(A=1.0, B=1.0))),
             variants=dict(A=dict(engine_id=EID, engine_params=pa), B=dict(engine_id=EID, engine_params=pb)),
             initial_side='A', listen='test')
    if ranges is not None:
        d['param_ranges'] = ranges
    return d


# ======================================================================================
class FullMaskTests(unittest.TestCase):
    def test_disk_mask_equals_the_independent_distance_rule_and_the_full_cover_bound(self):
        self.assertAlmostEqual(fg.full_cover_radius(32, 32), math.hypot(16, 16), places=12)
        self.assertAlmostEqual(fg.full_cover_radius(16, 40), math.hypot(8, 20), places=12)
        e, gens = engine_generations(sum(groups_of('p3_p5'), []), screenshot_params(radius_mul=1.0), 3)
        for _g, d in gens:
            for f in d['figures']:
                for mul in (1.0, 2.0, 8.0, 32.0):
                    r = float(f['radius_geom']) * mul
                    np.testing.assert_array_equal(fg.disk_mask(tuple(f['centre']), r, 32, 32),
                                                  torus_disk(f['centre'], r))
        # from the bound on: the whole field, whatever the radius (no squaring of it)
        for r in (fg.full_cover_radius(32, 32), 22.63, 32.0 * 0.745, 1e6, 5e300, math.inf):
            self.assertTrue(fg.covers_field(r, 32, 32))
            with np.errstate(all='raise'):
                self.assertTrue(fg.disk_mask((3.2, 4.1), r, 32, 32).all())
        self.assertFalse(fg.covers_field(22.0, 32, 32))
        self.assertFalse(fg.disk_mask((0.0, 0.0), 22.0, 32, 32).all())         # (16, 16) is 22.63 away
        self.assertTrue(fg.disk_mask((0.0, 0.0), 22.0, 32, 32)[15, 15])
        self.assertEqual(int(fg.disk_mask((4.0, 4.0), 0.0, 32, 32).sum()), 1)   # R = 0 stays one cell

    def test_engine_at_x32_covers_the_field_and_hears_every_change(self):
        cells = sum(groups_of('p3_p5'), [])
        e, gens = engine_generations(cells, screenshot_params(radius_mul=32.0), 12)
        for g, d in gens:
            self.assertTrue(all(f['covers_all'] for f in d['figures']))
        for t in range(1, len(gens)):
            g_prev, g_cur = gens[t - 1][0], gens[t][0]
            changed = int(np.count_nonzero(g_prev != g_cur))
            prev_ids = {f['id'] for f in gens[t - 1][1]['figures']}
            for f in gens[t][1]['figures']:
                if f['slot'] < 0:
                    continue
                if f['id'] in prev_ids:
                    self.assertEqual(f['e'], float(changed))          # births + deaths of the whole field
                else:
                    self.assertEqual(f['e'], float(np.count_nonzero((g_cur != 0) & (g_prev == 0))))
        # Own ignores the radius: no covers_all, e = own changes only
        e, gens = engine_generations(cells, screenshot_params(radius_mul=32.0, detector=orz.DET_OWN), 3)
        self.assertFalse(any(f['covers_all'] for _g, d in gens for f in d['figures']))
        # a single cell keeps R = 0 -> R_eff = 0 at any multiplier
        e = engine(cells + [(30, 2)], screenshot_params(radius_mul=1e6))
        run(e, 1)
        one = [f for f in e.display()['figures'] if f['n'] == 1][0]
        self.assertEqual((one['radius'], one['radius_geom'], one['covers_all']), (0.0, 0.0, False))


# ======================================================================================
class RadiusPairsTests(unittest.TestCase):
    """The two radius A/B of the REQ against independent masks."""
    N_TRANSITIONS = 71

    def _families(self, case):
        groups = groups_of(case)
        fam = [evolve(g, self.N_TRANSITIONS) for g in groups]
        joint = evolve(sum(groups, []), self.N_TRANSITIONS)
        # the joint evolution IS the union of the independent ones (the figures never touch)
        for t in range(self.N_TRANSITIONS + 1):
            self.assertFalse((fam[0][t] & fam[1][t]).any())
            np.testing.assert_array_equal(joint[t], fam[0][t] | fam[1][t])
        return groups, fam

    def _check(self, case, mul):
        groups, fam = self._families(case)
        e, gens = engine_generations(sum(groups, []), screenshot_params(radius_mul=mul), self.N_TRANSITIONS)
        expect = independent_events(gens, fam)
        foreign_sum = [0, 0]
        packets = [0, 0]
        distribution = {}
        for t in range(1, len(gens)):
            got = {f['id']: f for f in gens[t][1]['figures']}
            self.assertEqual(sorted(got), sorted(expect[t]))
            for fid, (ev, foreign, k) in expect[t].items():
                f = got[fid]
                if f['slot'] < 0:
                    continue                                     # tracked without a slot: no packet
                self.assertEqual(f['e'], float(ev), (case, mul, t, fid))
                self.assertAlmostEqual(f['a'], ev / (ev + 2.0) if ev else 0.0, places=12)
                foreign_sum[k] += foreign
                if ev > 0:
                    packets[k] += 1
                distribution.setdefault(k, {}).setdefault(int(ev), 0)
                distribution[k][int(ev)] += 1
        return foreign_sum, packets, distribution

    def test_r1_receiver_hears_the_blinker_only_at_x2(self):
        pf = preflight()['cases']['receiver']['runs']
        f1, p1, _d = self._check('receiver', 1.0)
        f2, p2, d2 = self._check('receiver', 2.0)
        self.assertEqual((f1, p1), (pf['1.0']['foreign_events_sum'], pf['1.0']['packets']))
        self.assertEqual((f2, p2), (pf['2.0']['foreign_events_sum'], pf['2.0']['packets']))
        self.assertEqual((f1, p1, f2, p2), ([0, 0], [0, 71], [284, 0], [71, 71]))
        self.assertEqual(d2[0], {4: 71})                     # the receiver: four foreign changes every generation
        self.assertEqual(d2[1], {4: 71})                     # the blinker: its own four, no foreign ones

    def test_r1_the_blinker_shifted_out_of_the_x2_circle_stops_contributing(self):
        recv, blink = groups_of('receiver')
        far = [(r, c + 8) for r, c in blink]                 # 13.3 .. 14.1 cells from the centre: outside x2 (10.66)
        e, gens = engine_generations(recv + far, screenshot_params(radius_mul=2.0), 12)
        fam = [evolve(recv, 12), evolve(far, 12)]
        expect = independent_events(gens, fam)
        receiver_foreign = sum(v[1] for t in expect for v in expect[t].values() if v[2] == 0)
        self.assertEqual(receiver_foreign, 0)
        receiver_e = [f['e'] for _g, d in gens[1:] for f in d['figures'] if f['n'] == 17]
        self.assertEqual(set(receiver_e), {0.0})
        # ... and x3 reaches it again (the same independent rule, a different circle)
        e, gens = engine_generations(recv + far, screenshot_params(radius_mul=3.0), 6)
        expect = independent_events(gens, fam[:1] + [evolve(far, 6)])
        self.assertEqual(sum(v[1] for t in expect for v in expect[t].values() if v[2] == 0), 4 * 6)
        self.assertEqual({f['e'] for _g, d in gens[1:] for f in d['figures'] if f['n'] == 17}, {4.0})

    def test_r2_two_periods_receive_each_other_only_at_x8_and_x32(self):
        pf = preflight()['cases']['p3_p5']['runs']
        results = {}
        for mul in (1.0, 8.0, 32.0):
            results[mul] = self._check('p3_p5', mul)
            foreign, packets, _d = results[mul]
            self.assertEqual(foreign, pf[f'{mul}']['foreign_events_sum'], mul)
            self.assertEqual(packets, pf[f'{mul}']['packets'], mul)
        self.assertEqual(results[1.0][:2], ([0, 0], [118, 71]))
        self.assertEqual(results[8.0][:2], ([392, 519], [141, 71]))
        self.assertEqual(results[32.0][:2], ([1312, 519], [141, 71]))
        # the distribution of packet sizes moves up with the radius (not only "non-zero")
        for k in (0, 1):
            self.assertLess(max(results[1.0][2][k]), max(results[32.0][2][k]))

    def test_the_knob_and_the_range_never_make_a_packet(self):
        cells = sum(groups_of('receiver'), [])
        e = engine(cells, screenshot_params(radius_mul=1.0))
        run(e, 2)
        for mul in (0.5, 32.0, 1e6, 0.0, 1.0):
            zf = e.zf.copy()
            e.set_params(dict(e.params, radius_mul=mul))
            self.assertTrue(np.array_equal(zf, e.zf))
            run(e, 1)
            self.assertTrue((e.zf <= zf).all())
            self.assertEqual(e.display()['radius_mul'], mul)
        # the runner: set_range leaves the engine and the sound untouched
        doc = scene_doc()
        a, b = DemoRunner(scene_from_doc(doc)), DemoRunner(scene_from_doc(doc))
        for r in (a, b):
            r.post('start', at=0)
        for _ in range(20):
            a.next_block()
            b.next_block()
        b.post('set_range', side='B', name='radius_mul', lo=0.3, hi=0.6)
        b.post('set_range', side='A', name='radius_mul', lo=1.0, hi=32.0)
        for _ in range(60):
            x, y = a.next_block(), b.next_block()
            for s in ('A', 'B', 'monitor'):
                np.testing.assert_array_equal(x.get(s), y.get(s))
        self.assertEqual(b.side_ranges(), {'A': {'radius_mul': (1.0, 32.0)}, 'B': {'radius_mul': (0.3, 0.6)}})
        self.assertEqual(b.sides['B'].params['radius_mul'], 1.0)          # the value is not clamped by the runner
        self.assertEqual(a.sides['B'].engine.export_state()['params'], b.sides['B'].engine.export_state()['params'])


# ======================================================================================
class RunnerSceneRecordTests(unittest.TestCase):
    def test_set_range_validation_storage_independence_and_engine_switches(self):
        r = DemoRunner(scene_from_doc(scene_doc()))
        self.assertEqual(r.side_ranges(), {'A': {'radius_mul': (0.25, 4.0)}, 'B': {'radius_mul': (0.25, 4.0)}})
        self.assertEqual(r.param_ranges(), {})
        for kw in (dict(side='C', name='radius_mul', lo=0, hi=1), dict(side='A', name='decay_s', lo=0, hi=1),
                   dict(side='A', name='radius_mul', lo=1, hi=1), dict(side='A', name='radius_mul', lo=-1, hi=1),
                   dict(side='A', name='radius_mul', lo=0, hi=math.inf), dict(side='A', name='nope', lo=0, hi=1)):
            with self.assertRaises(ValueError):
                r.post('set_range', **kw)
        self.assertEqual(r._pending, [])
        r.post('set_range', side='A', name='radius_mul', lo=0.3, hi=0.6)
        r.post('set_range', side='B', name='radius_mul', lo=1, hi=32)
        r.next_block()
        self.assertEqual(r.side_ranges(), {'A': {'radius_mul': (0.3, 0.6)}, 'B': {'radius_mul': (1.0, 32.0)}})
        self.assertEqual(r.param_ranges(), {'A': {EID: {'radius_mul': [0.3, 0.6]}},
                                            'B': {EID: {'radius_mul': [1.0, 32.0]}}})
        self.assertEqual(r.range_of('A', 'radius_mul'), (0.3, 0.6))
        self.assertIsNone(r.range_of('A', 'decay_s'))
        # a queued engine change counts: the range is checked against the engine the side WILL have
        r.post('set_engine', side='A', engine_id='laplacian')
        with self.assertRaises(ValueError):
            r.post('set_range', side='A', name='radius_mul', lo=0, hi=1)
        r.next_block()
        self.assertEqual(r.side_ranges()['A'], {})
        self.assertEqual(r.param_ranges()['A'], {EID: {'radius_mul': [0.3, 0.6]}})   # remembered
        r.post('set_engine', side='A', engine_id=EID)
        r.next_block()
        self.assertEqual(r.side_ranges()['A'], {'radius_mul': (0.3, 0.6)})
        r.post('factory')
        r.next_block()
        self.assertEqual(r.side_ranges()['A'], {'radius_mul': (0.3, 0.6)})           # factory: parameters, not ranges
        self.assertIn(('set_range', {'side': 'B', 'name': 'radius_mul', 'lo': 1.0, 'hi': 32.0}),
                      [(k, a) for (_t, _s, k, a) in r.journal])
        # snapshot: the ranges travel; an older snapshot without them = the defaults
        st = r.export_state()
        self.assertEqual(st['ranges'], r.param_ranges())
        twin = DemoRunner.from_state(st)
        self.assertEqual(twin.side_ranges(), r.side_ranges())
        old = dict(st)
        del old['ranges']
        self.assertEqual(DemoRunner.from_state(old).side_ranges(),
                         {'A': {'radius_mul': (0.25, 4.0)}, 'B': {'radius_mul': (0.25, 4.0)}})
        for bad in ({'A': {EID: {'radius_mul': [2, 1]}}}, {'C': {}}, {'A': {'nope': {}}}, {'A': {EID: {'radius_mul': [1]}}}):
            with self.assertRaises(ValueError):
                DemoRunner.from_state(dict(st, ranges=bad))

    def test_scene_param_ranges_validation_and_load(self):
        doc = scene_doc(ranges={'B': {EID: {'radius_mul': [0.25, 32.0]}}})
        sc = scene_from_doc(doc)
        self.assertEqual(sc.param_ranges, {'B': {EID: {'radius_mul': (0.25, 32.0)}}})
        r = DemoRunner(sc)
        self.assertEqual(r.side_ranges(), {'A': {'radius_mul': (0.25, 4.0)}, 'B': {'radius_mul': (0.25, 32.0)}})
        self.assertEqual(scene_from_doc(scene_doc()).param_ranges, {})
        for bad in ({'C': {}}, {'A': []}, {'A': {'nope': {'radius_mul': [0, 1]}}}, {'A': {EID: [0, 1]}},
                    {'A': {EID: {'radius_mul': [1, 1]}}}, {'A': {EID: {'radius_mul': [0.5]}}},
                    {'A': {EID: {'decay_s': [0.2, 1.0]}}}, {'A': {EID: {'radius_mul': [-1, 1]}}}):
            with self.assertRaises(SceneError):
                scene_from_doc(scene_doc(ranges=bad))
        # the existing scenes of the previous catalogs carry no ranges: the defaults
        sc = load_scene(os.path.join(ROOT, 'demos', 'ol_neighbor.json'))
        self.assertEqual(sc.param_ranges, {})
        self.assertEqual(DemoRunner(sc).side_ranges()['B'], {'radius_mul': (0.25, 4.0)})

    def test_records_keep_the_ranges_through_save_open_and_continue(self):
        from casynth_lab.catalog import Catalog, bench_scene
        from demos.build_n1_demos import record_offline
        os.makedirs(ART, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            cat = Catalog(tmp, repo_root=None)
            doc = scene_doc(ranges={'A': {EID: {'radius_mul': [0.3, 0.6]}}})
            cmds = [('set_range', 2 * BLOCK, dict(side='B', name='radius_mul', lo=1.0, hi=32.0)),
                    ('set_param', 3 * BLOCK, dict(side='B', name='radius_mul', value=32.0))]
            rid, snap = record_offline(cat, doc, 1.0, cmds, 'ranges', '')
            self.assertEqual(snap['ranges'], {'A': {'radius_mul': (0.3, 0.6)}, 'B': {'radius_mul': (1.0, 32.0)}})
            rec = cat.load(rid)
            self.assertEqual(rec.meta['scene']['param_ranges'], {'A': {EID: {'radius_mul': [0.3, 0.6]}}})
            self.assertEqual(rec.meta['state_at_end']['param_ranges'],
                             {'A': {EID: {'radius_mul': [0.3, 0.6]}}, 'B': {EID: {'radius_mul': [1.0, 32.0]}}})
            self.assertEqual([j['kind'] for j in rec.meta['journal'] if j['kind'] != 'step'],
                             ['start', 'set_range', 'set_param'])
            self.assertEqual(cat.replay(rid, yield_cpu=False).status, 'match')
            scene, _vol = bench_scene(rec)
            self.assertEqual(scene.param_ranges, {'A': {EID: {'radius_mul': (0.3, 0.6)}},
                                                  'B': {EID: {'radius_mul': (1.0, 32.0)}}})
            self.assertEqual(scene.variants['B'][1]['radius_mul'], 32.0)
            continued, _state, _ = cat.continue_runner(rid)
            self.assertEqual(continued.side_ranges(), {'A': {'radius_mul': (0.3, 0.6)}, 'B': {'radius_mul': (1.0, 32.0)}})
            # an older record without ranges (its documents predate them): the defaults
            meta = copy.deepcopy(rec.meta)
            meta['scene'].pop('param_ranges', None)
            meta['state_at_end'].pop('param_ranges', None)
            scene, _vol = bench_scene(cat.load(rid).__class__(tmp, rid, meta))
            self.assertEqual(scene.param_ranges, {})


# ======================================================================================
class BenchTests(unittest.TestCase):
    @staticmethod
    def _wait(eng, pred, timeout=4.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if pred(eng.snapshot()):
                return True
            time.sleep(0.02)
        return False

    def test_min_max_fields_slider_clamp_hotkeys_and_the_full_cover_frame(self):
        import pygame
        import demo_bench as db
        from casynth_lab.audio_out import LiveEngine
        scene = scene_from_doc(scene_doc(cells=sum(groups_of('p3_p5'), [])))
        runner = DemoRunner(scene)
        eng = LiveEngine(runner, sink=lambda m, b: None)
        pygame.init()
        app = db.BenchApp(scene, eng)
        n_rows = len(registry.get(EID).params) + len(registry.get(EID).ranges)
        self.assertEqual(app._params_height(EID), n_rows * db.ROW_H)
        self.assertGreaterEqual(app.footer_y, app.params_y + n_rows * db.ROW_H + 6 + db.FIGURE_HEAD_H + app.figure_rows * db.DISPLAY_ROW_H)
        self.assertLessEqual(app.height, 1000)
        self.assertEqual(app.figure_rows, db.FIGURE_ROWS_MAX)
        # a low desktop (960 px, 2026-09-17): the figure rows give way, the window fits
        low = db.BenchApp(scene, eng, max_height=880)
        self.assertLessEqual(low.height, 880)
        self.assertGreaterEqual(low.figure_rows, db.FIGURE_ROWS_MIN)
        self.assertLess(low.figure_rows, app.figure_rows)
        self.assertEqual(low.width, app.width)
        rows = {spec[0]: rect for spec, rect in app._param_rows(EID)}
        rr = app._range_rects(EID)['radius_mul']
        self.assertEqual(rows['spectrum'][1] - rows['radius_mul'][1], 2 * db.ROW_H)   # the extra row
        self.assertTrue(rows['radius_mul'][1] < rr['y'] < rows['spectrum'][1])
        screen = pygame.Surface((app.width, app.height))
        font, small = pygame.font.SysFont(db.FONT_NAMES, 17), pygame.font.SysFont(db.FONT_NAMES, 14)
        eng.start()
        try:
            eng.post('start')
            self.assertTrue(self._wait(eng, lambda s: s['gen'] >= 1))
            app.press((app.tabs['B'][0] + 2, app.tabs['B'][1] + 2), 1)
            self.assertTrue(self._wait(eng, lambda s: s['selected'] == 'B'))
            app.draw(screen, font, small)
            # Max -> 32 (Enter); the slider end is now 32 and the field covers all
            self.assertEqual(app.press((rr['max'][0] + 5, rr['max'][1] + 5), 1), 'range:radius_mul:max')
            app.release()
            self.assertEqual(app.key('c'), 'range:typing')                 # hotkeys inert while typing
            self.assertEqual(app.key('r'), 'range:typing')
            self.assertEqual(app.range_edit['edit'].text, '4')               # (letters arrive as text input)
            self.assertEqual(app.key('a', ctrl=True), 'range:caret')
            for ch in '32':
                app.text_input(ch)
            self.assertEqual(app.key('return'), 'range:commit')
            self.assertIsNone(app.range_edit)
            self.assertTrue(self._wait(eng, lambda s: s['ranges']['B']['radius_mul'] == (0.25, 32.0)))
            self.assertEqual(eng.snapshot()['ranges']['A']['radius_mul'], (0.25, 4.0))   # A untouched
            sx, sy, sw, _sh = rows['radius_mul']
            self.assertEqual(app.press((sx + sw, sy + 2), 1), 'param:radius_mul')
            app.release()
            self.assertTrue(self._wait(eng, lambda s: s['sides']['B'][1]['radius_mul'] == 32.0))
            app.draw(screen, font, small)
            figs = eng.snapshot()['display']['B']['figures']
            self.assertTrue(all(f['covers_all'] for f in figs))
            col = db.C_FIGURES[figs[0]['color'] % len(db.C_FIGURES)]
            self.assertEqual(tuple(screen.get_at((app.field_x, app.field_y))[:3]), col)      # the frame
            # a refused number: the range stays, the reason is shown, Esc cancels
            self.assertEqual(app.press((rr['min'][0] + 5, rr['min'][1] + 5), 1), 'range:radius_mul:min')
            app.release()
            for ch in 'abc':
                app.text_input(ch)
            self.assertEqual(app.key('return'), 'range:error')
            self.assertEqual(app.range_error, ('radius_mul', 'not a number'))
            self.assertIsNotNone(app.range_edit)
            app.draw(screen, font, small)
            self.assertEqual(app.key('escape'), 'range:cancel')
            self.assertIsNone(app.range_edit)
            for text, err in (('-1', 'min < 0'), ('40', 'min must be < max'), ('inf', 'not a number')):
                app.press((rr['min'][0] + 5, rr['min'][1] + 5), 1)
                app.release()
                for ch in text:
                    app.text_input(ch)
                self.assertEqual(app.key('return'), 'range:error', text)
                self.assertEqual(app.range_error, ('radius_mul', err))
                app.key('escape')
            self.assertEqual(eng.snapshot()['ranges']['B']['radius_mul'], (0.25, 32.0))
            # Max 0.6 committed by LEAVING the field (click into Min), Min 0.3 by Enter:
            # the range narrows, the value 32 is clamped ONCE to 0.6
            app.press((rr['max'][0] + 5, rr['max'][1] + 5), 1)
            app.release()
            for ch in '0.6':
                app.text_input(ch)
            self.assertEqual(app.press((rr['min'][0] + 5, rr['min'][1] + 5), 1), 'range:radius_mul:min')
            app.release()
            for ch in '0.3':
                app.text_input(ch)
            self.assertEqual(app.key('return'), 'range:commit')
            self.assertTrue(self._wait(eng, lambda s: s['ranges']['B']['radius_mul'] == (0.3, 0.6)
                                       and s['sides']['B'][1]['radius_mul'] == 0.6))
            kinds = [k for (_t, _s, k, _a) in runner.journal]
            self.assertEqual(kinds.count('set_range'), 3)
            clamps = [a for (_t, _s, k, a) in runner.journal if k == 'set_param' and a['name'] == 'radius_mul']
            self.assertEqual([a['value'] for a in clamps], [32.0, 0.6])
            # the narrow range: the slider middle is 0.45, steps of 0.001 are reachable
            app.press((sx + sw // 2, sy + 2), 1)
            app.release()
            self.assertTrue(self._wait(eng, lambda s: s['sides']['B'][1]['radius_mul'] == 0.45))
            app.drag_param = 'radius_mul'
            app._set_param_from_x(sx + sw // 2 + 1)
            app.release()
            self.assertTrue(self._wait(eng, lambda s: s['sides']['B'][1]['radius_mul'] == 0.453))   # 0.4525 -> 3 decimals
            # Tab moves to the other bound (committing the first), Enter commits it
            app.press((rr['min'][0] + 5, rr['min'][1] + 5), 1)
            app.release()
            for ch in '0.31':
                app.text_input(ch)
            self.assertEqual(app.key('tab'), 'range:radius_mul:max')
            self.assertEqual(app.range_edit['which'], 'max')
            for ch in '0.62':
                app.text_input(ch)
            self.assertEqual(app.key('return'), 'range:commit')
            self.assertTrue(self._wait(eng, lambda s: s['ranges']['B']['radius_mul'] == (0.31, 0.62)))
            self.assertEqual(eng.snapshot()['sides']['B'][1]['radius_mul'], 0.453)    # inside: not touched
            app.draw(screen, font, small)
            # the fields of the other side: their own range
            app.press((app.tabs['A'][0] + 2, app.tabs['A'][1] + 2), 1)
            self.assertTrue(self._wait(eng, lambda s: s['selected'] == 'A'))
            app.draw(screen, font, small)
            self.assertEqual(app._range_of(eng.snapshot(), 'A', 'radius_mul'), (0.25, 4.0))
            self.assertEqual(app.key('r'), 'reset')                        # hotkeys back
        finally:
            eng.stop()
            pygame.quit()


# ======================================================================================
# Attack (part 2 of the REQ): one causal pole on the excitation pulse before the bank
# ======================================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_n4_object_resonators import bare_engine, setup_slot, scalar_n4     # noqa: E402  (the v1/v2 reference)


def scalar_attack(slots, blocks, sr=SR, ramp_n=882, hp_hz=20.0, out_scale=orz.OUT_SCALE, g0=GAIN,
                  decay_s=0.8, attack0=0.0):
    """An INDEPENDENT scalar sample path of the Attack smoothing (REQ formula):
    slots = [dict(freqs, weights, pan_p, ndrive)] all driven, fixed tuning;
    blocks = [{'inject': {slot: a}, 'attack': ms, 'gain': g}] (boundary actions).
        p = 0.75 ((1-qf) zf - (1-qs) zs) / (qs-qf);  u = q u + (1-q) p;  bank driven by u
        q = exp(-ln 9 / (sr ms / 1000)) (0 at ms = 0), a change ramps q linearly over ramp_n."""
    qf = math.exp(-1.0 / (sr * 0.00025))
    qs = math.exp(-1.0 / (sr * 0.002))
    hp_h = math.exp(-2.0 * math.pi * hp_hz / sr)
    r = 10.0 ** (-3.0 / (sr * decay_s))
    S = []
    for sd in slots:
        n = len(sd['freqs'])
        S.append(dict(re=[0.0] * n, im=[0.0] * n, c=[math.cos(2 * math.pi * f / sr) for f in sd['freqs']],
                      s=[math.sin(2 * math.pi * f / sr) for f in sd['freqs']], w=list(sd['weights']),
                      nd=int(sd['ndrive']), zf=0.0, zs=0.0, u=0.0,
                      L=math.cos(math.pi * sd['pan_p'] / 2), R=math.sin(math.pi * sd['pan_p'] / 2)))

    def q_of(ms):
        return 0.0 if ms <= 0.0 else math.exp(-math.log(9.0) / (sr * ms / 1000.0))
    q_cur = q_tgt = q_of(attack0)
    q_inc, q_left = 0.0, 0
    g_cur = g_tgt = g0
    g_inc, g_left = 0.0, 0
    hp = [[0.0, 0.0], [0.0, 0.0]]
    out = []
    for blk in blocks:
        if 'attack' in blk and q_of(blk['attack']) != q_tgt:
            q_tgt = q_of(blk['attack'])
            q_inc = (q_tgt - q_cur) / ramp_n
            q_left = ramp_n
        for si, a in blk.get('inject', {}).items():
            S[si]['zf'] += a
            S[si]['zs'] += a
        if 'gain' in blk and blk['gain'] != g_tgt:
            g_tgt = blk['gain']
            g_inc = (g_tgt - g_cur) / ramp_n
            g_left = ramp_n
        for _t in range(BLOCK):
            if g_left > 0:
                g_left -= 1
                g_cur = g_tgt if g_left == 0 else g_cur + g_inc
            if q_left > 0:
                q_left -= 1
                q_cur = q_tgt if q_left == 0 else q_cur + q_inc
            L = 0.0
            R = 0.0
            for st in S:
                p = 0.75 * ((1.0 - qf) * st['zf'] - (1.0 - qs) * st['zs']) / (qs - qf)
                st['zf'] *= qf
                st['zs'] *= qs
                u = q_cur * st['u'] + (1.0 - q_cur) * p
                st['u'] = u
                acc = 0.0
                for j in range(len(st['re'])):
                    re, im, c, sn = st['re'][j], st['im'][j], st['c'][j], st['s'][j]
                    nre = r * (c * re - sn * im) + (u if j < st['nd'] else 0.0)
                    nim = r * (sn * re + c * im)
                    st['re'][j], st['im'][j] = nre, nim
                    acc += st['w'][j] * nre
                L += st['L'] * acc
                R += st['R'] * acc
            yL = hp_h * ((hp[0][0] + L) - hp[0][1])
            hp[0] = [yL, L]
            yR = hp_h * ((hp[1][0] + R) - hp[1][1])
            hp[1] = [yR, R]
            out.append((yL * out_scale * g_cur, yR * out_scale * g_cur))
    return np.array(out)


PROBE_FREQS = [110.0, 440.0, 880.0]           # the Researcher's scalar probe (preflight attack_proposal_probe)
PROBE_WEIGHTS = [1.0, 0.4, 0.1]


def probe_engine(attack_ms, decay_s=1.39):
    e = bare_engine()
    e.set_params(dict(e.params, decay_s=decay_s, attack_ms=attack_ms))
    e.ints[orz.I_R_LEFT] = 0                        # the settings are in force from the first sample
    e.rr[:] = (e.rr[orz.R_TGT], e.rr[orz.R_TGT], 0.0)
    e.qq[:] = (e.qq[orz.R_TGT], e.qq[orz.R_TGT], 0.0)
    e.ints[orz.I_Q_LEFT] = 0
    setup_slot(e, 0, PROBE_FREQS, 3, 0.5)
    e.wcur[0, :3] = PROBE_WEIGHTS
    e.wtgt[0, :3] = PROBE_WEIGHTS
    return e


class AttackTests(unittest.TestCase):
    def test_coefficient_formula_and_the_10_to_90_percent_rise_time(self):
        self.assertEqual(orz.attack_q(0.0), 0.0)
        self.assertEqual(orz.attack_q(-1.0), 0.0)
        for ms in (1.0, 4.0, 10.0, 20.0):
            q = orz.attack_q(ms)
            self.assertAlmostEqual(q, math.exp(-math.log(9.0) / (SR * ms / 1000.0)), places=15)
            # the step response of u = q u + (1 - q) p: 10 % -> 90 % in attack_ms (+- one sample)
            u, t10, t90 = 0.0, None, None
            for k in range(int(SR * 0.1)):
                u = q * u + (1.0 - q)
                if t10 is None and u >= 0.1:
                    t10 = k
                if t90 is None and u >= 0.9:
                    t90 = k
                    break
            self.assertAlmostEqual((t90 - t10) / SR * 1000.0, ms, delta=1000.0 / SR * 1.5)
        pf = preflight()['attack_proposal_probe']['results']
        for ms, row in pf.items():
            self.assertAlmostEqual(orz.attack_q(float(ms)), row['q'], places=12)
        spec = registry.get(EID).spec_of('attack_ms')
        self.assertEqual(spec, ('attack_ms', 'Attack', 0.0, 20.0, False, 0.0))
        self.assertEqual(orz.OPTIONAL_PARAMS['attack_ms'], 0.0)
        self.assertEqual((orz.MODEL_VERSION, orz.STATE_VERSION), ('ca_object_resonators_n4_v5', 5))   # v4 until Birth strength (2026-09-18)

    def test_attack_zero_is_the_previous_path_bit_for_bit(self):
        """The v1/v2 scalar reference of the N4 gate (no attack in its formulas) equals the
        v3 kernel at Attack 0 through every ramp, tune, pan and strike of that gate."""
        freqs_a = [220.0, 381.05, 512.3, 777.7, 1234.5]
        freqs_b = [80.8, 130.1, 190.2]
        e = bare_engine()
        self.assertEqual(float(e.qq[orz.R_CUR]), 0.0)
        setup_slot(e, 0, freqs_a, 5, 0.3)
        setup_slot(e, 1, freqs_b + [640.0, 990.0], 3, 0.9,
                   zre=[0.0, 0.0, 0.0, 0.02, -0.01], zim=[0.0, 0.0, 0.0, 0.01, 0.03])
        slots = [dict(freqs=freqs_a, weights=[0.2] * 5, pan_p=0.3, ndrive=5),
                 dict(freqs=freqs_b + [640.0, 990.0], weights=[1 / 3] * 3 + [0.2, 0.2], pan_p=0.9,
                      ndrive=3, zre=[0.0, 0.0, 0.0, 0.02, -0.01], zim=[0.0, 0.0, 0.0, 0.01, 0.03])]
        e.wcur[1, 3:5] = 0.2
        e.wtgt[1, 3:5] = 0.2
        blocks = [dict(inject={0: 0.6, 1: 0.75}), {}, dict(gain=GAIN * 1.5), {}, dict(decay=1.3),
                  dict(inject={1: 0.4}), dict(tune={0: ([220.0, 381.05, 512.3, 777.7, 1234.5, 1500.0], 6)}),
                  {}, dict(pan={1: 0.1}), {}, dict(tune={1: ([80.8, 130.1], 2)}), {}, {},
                  dict(inject={0: 0.2, 1: 0.2}, decay=0.5, gain=GAIN), {}, {}, {}, {}]
        ref = scalar_n4(slots, blocks)
        got = []
        for blk in blocks:
            for si, (freqs, n) in blk.get('tune', {}).items():
                e._tune_slot(si, np.asarray(freqs), np.full(n, 1.0 / n), np.asarray(freqs) / 220.0,
                             orz.SPEC_FIGURE, ramp=True)
            for si, p in blk.get('pan', {}).items():
                e._set_pan(si, p * 31.0, ramp=True)
            if 'decay' in blk:
                e.set_params(dict(e.params, decay_s=blk['decay']))
            for si, a in blk.get('inject', {}).items():
                e.inject(si, a)
            got.append(e.render_float(blk.get('gain', e.gg[orz.R_TGT]))[0])
        got = np.concatenate(got)
        self.assertGreater(float(np.abs(ref).max()), 1e-4)
        self.assertEqual(float(np.abs(got - ref).max()), 0.0)          # exact, not merely close

    def test_kernel_matches_the_attack_reference_through_strikes_ramps_and_zero_crossings(self):
        freqs = [PROBE_FREQS, [150.0, 333.3, 520.0, 1010.0]]
        weights = [PROBE_WEIGHTS, [0.3, 0.3, 0.2, 0.2]]
        pans = [0.5, 0.15]
        e = bare_engine()
        for s in range(2):
            setup_slot(e, s, freqs[s], len(freqs[s]), pans[s])
            e.wcur[s, :len(weights[s])] = weights[s]
            e.wtgt[s, :len(weights[s])] = weights[s]
        slots = [dict(freqs=freqs[s], weights=weights[s], pan_p=pans[s], ndrive=len(freqs[s])) for s in range(2)]
        blocks = [dict(inject={0: 0.6}), dict(attack=4.0), dict(inject={1: 0.75}), {}, {},
                  dict(inject={0: 0.3, 1: 0.2}), dict(attack=1.0, inject={0: 0.9}), {}, {}, dict(gain=GAIN * 1.4),
                  dict(attack=20.0), {}, {}, dict(inject={1: 0.5}), dict(attack=0.0), {}, {}, {},
                  dict(inject={0: 0.4}), {}, dict(attack=4.0), dict(attack=0.0), {}, {}, dict(inject={1: 0.3}),
                  {}, {}, {}]
        ref = scalar_attack(slots, blocks)
        got = []
        for blk in blocks:
            if 'attack' in blk:
                e.set_params(dict(e.params, attack_ms=blk['attack']))
            for si, a in blk.get('inject', {}).items():
                e.inject(si, a)
            got.append(e.render_float(blk.get('gain', e.gg[orz.R_TGT]))[0])
        got = np.concatenate(got)
        self.assertGreater(float(np.abs(ref).max()), 1e-4)
        self.assertLessEqual(float(np.abs(got - ref).max()), TOL)
        self.assertEqual(float(e.qq[orz.R_CUR]), 0.0)                  # back at zero: exactly 0
        self.assertEqual(int(e.ints[orz.I_Q_LEFT]), 0)
        # from here the previous path: an engine with the same state but the attack
        # state cleared (zu = 0 is never read at q = 0) renders the same blocks
        twin = bare_engine()
        for name in e._ARRAYS + ('cth', 'sth'):
            getattr(twin, name)[...] = getattr(e, name)
        twin.params = dict(e.params)
        twin.zu[:] = 0.0
        self.assertTrue((e.zu[:2] != 0.0).all())
        for _ in range(30):
            np.testing.assert_array_equal(e.render_float(GAIN)[0], twin.render_float(GAIN)[0])

    def test_silence_without_events_and_the_moments_of_the_packets_do_not_move(self):
        e = bare_engine()
        e.set_params(dict(e.params, attack_ms=4.0))
        setup_slot(e, 0, PROBE_FREQS, 3, 0.5)
        y = run(e, 60)
        self.assertEqual(float(np.abs(y).max()), 0.0)
        # the p3/p5 field at Attack 0 and 4: the same packets at the same boundaries,
        # the same pulse states block by block (u never feeds back into them)
        cells = sum(groups_of('p3_p5'), [])
        e0 = engine(cells, screenshot_params(attack_ms=0.0))
        e4 = engine(cells, screenshot_params(attack_ms=4.0))
        g = grid(cells)
        for b in range(int(3.0 * SR / BLOCK)):
            if b and b % 12 == 0:
                prev, g = g, step(g)
                for e in (e0, e4):
                    e.update_field(g, events_field(prev, g))
            y0 = e0.render_float(GAIN)[0]
            y4 = e4.render_float(GAIN)[0]
            np.testing.assert_array_equal(e0.zf, e4.zf)
            np.testing.assert_array_equal(e0.zs, e4.zs)
            np.testing.assert_array_equal(e0.last_e, e4.last_e)
            self.assertEqual([f['id'] for f in e0.display()['figures']], [f['id'] for f in e4.display()['figures']])
            if float(np.abs(y0).max()) > 0.0:
                self.assertGreater(float(np.abs(y0 - y4).max()), 0.0)   # but the sound differs

    def test_snapshot_mid_pulse_mid_ramp_and_overlapping_events_continue_exactly(self):
        cells = sum(groups_of('receiver'), [])
        e = engine(cells, screenshot_params(attack_ms=4.0))
        e.render_float(GAIN)                                            # the start packet: mid-pulse
        e.set_params(dict(e.params, attack_ms=10.0))
        e.render_float(GAIN)                                            # 8 ms into the 20 ms q ramp
        e.inject(0, 0.5)                                                # a new strike on the ringing bank
        self.assertGreater(int(e.ints[orz.I_Q_LEFT]), 0)
        self.assertNotEqual(float(e.zu[0]), 0.0)
        st = e.export_state()
        self.assertEqual(st['version'], 5)
        for name in ('zu', 'qq'):
            self.assertIn(name, st)
        twin = registry.create(EID, CTX, dict(e.params))
        twin.init(np.zeros((32, 32), np.uint8), None, 0.0)
        twin.restore_state(e._grid, e._exc, st)
        self.assertEqual(float(twin.zu[0]), float(e.zu[0]))
        for _ in range(60):
            np.testing.assert_array_equal(e.render_float(GAIN)[0], twin.render_float(GAIN)[0])
        # a strike never resets u or the bank: the smoothed pulse just gets a new front
        u_before, z_before = float(e.zu[0]), e.zre[0, :3].copy()
        e.inject(0, 0.4)
        self.assertEqual(float(e.zu[0]), u_before)
        np.testing.assert_array_equal(e.zre[0, :3], z_before)
        # a bank that becomes a tail: pulse and attack states zeroed, no excitation any more
        e._to_tail(0)
        tails = [s for s in range(orz.N_ACTIVE, orz.N_SLOTS) if e.role[s] == orz.ROLE_TAIL]
        self.assertEqual(len(tails), 1)
        self.assertEqual((float(e.zu[tails[0]]), float(e.zf[tails[0]]), float(e.zs[tails[0]])), (0.0, 0.0, 0.0))
        self.assertEqual(int(e.ndrive[tails[0]]), 0)

    def test_older_v2_snapshot_means_attack_zero(self):
        cells = sum(groups_of('receiver'), [])
        e = engine(cells, screenshot_params(attack_ms=0.0))
        e.render_float(GAIN)
        st = e.export_state()
        old = dict(st, version=2, model_version='ca_object_resonators_n4_v2', ints=st['ints'][:3].copy(),
                   params={k: v for k, v in st['params'].items() if k != 'attack_ms'})
        del old['zu']
        del old['qq']

        def fresh():
            t = registry.create(EID, CTX, dict(e.params))
            t.init(np.zeros((32, 32), np.uint8), None, 0.0)
            return t
        twin = fresh()
        twin.restore_state(e._grid, e._exc, old)
        self.assertEqual(twin.params['attack_ms'], 0.0)
        self.assertEqual(float(twin.qq[orz.R_CUR]), 0.0)
        self.assertEqual(tuple(twin.ints.shape), (4,))
        for _ in range(40):
            np.testing.assert_array_equal(e.render_float(GAIN)[0], twin.render_float(GAIN)[0])
        with self.assertRaises(ValueError):                             # a v2 state must say model v2
            fresh().restore_state(e._grid, e._exc, dict(old, model_version=orz.MODEL_VERSION))
        with self.assertRaises(ValueError):
            fresh().restore_state(e._grid, e._exc, dict(st, version=1))
        with self.assertRaises(ValueError):                             # q outside [0, 1)
            fresh().restore_state(e._grid, e._exc, dict(st, qq=np.array([1.0, 1.0, 0.0])))
        # a parameter set without the key: 0 (the registry default is also 0)
        e2 = registry.create(EID, CTX, dict(detector=1, frequency_scale=220.0, decay_s=0.8))
        self.assertEqual(e2.params['attack_ms'], 0.0)
        self.assertEqual(registry.defaults(EID)['attack_ms'], 0.0)

    def test_levels_and_brightness_change_as_the_probe_predicts(self):
        """The Researcher's scalar probe (preflight): Attack 4 ms softens the first
        sample, removes energy above 3 kHz and lowers the level; measured here on
        the engine's own output of the same three resonators."""
        out = {}
        for ms in (0.0, 4.0):
            e = probe_engine(ms)
            e.inject(0, 0.6)
            y = run(e, int(1.0 * SR / BLOCK))[:, 0]
            spec = np.abs(np.fft.rfft(y)) ** 2
            f = np.fft.rfftfreq(len(y), 1.0 / SR)
            out[ms] = dict(first=float(np.abs(y[:3]).max()), rms=float(np.sqrt(np.mean(y * y))),
                           hf=float(spec[f > 3000.0].sum()), step=float(np.abs(np.diff(y)).max()))
        self.assertLess(out[4.0]['first'], out[0.0]['first'] * 0.1)
        self.assertLess(10.0 * math.log10(out[4.0]['hf'] / out[0.0]['hf']), -20.0)
        self.assertLess(out[4.0]['rms'], out[0.0]['rms'])                # no compensation: the level falls
        step_eq = out[4.0]['step'] / out[4.0]['rms'] / (out[0.0]['step'] / out[0.0]['rms'])
        self.assertLess(20.0 * math.log10(step_eq), -8.0)
        # the knob in the panel: display, ramp, the seven / detector / radius untouched
        e = engine(sum(groups_of('receiver'), []))
        run(e, 2)
        before = dict(e.params)
        e.set_params(dict(e.params, attack_ms=4.0))
        d = e.display()
        self.assertEqual((d['attack_ms'], d['attack_ramp_left']), (4.0, e.ramp_n))
        self.assertAlmostEqual(d['q_target'], orz.attack_q(4.0), places=15)
        self.assertEqual({k: v for k, v in e.params.items() if k != 'attack_ms'},
                         {k: v for k, v in before.items() if k != 'attack_ms'})
        run(e, 3)
        self.assertEqual((e.display()['attack_ramp_left'], e.display()['q']), (0, orz.attack_q(4.0)))


if __name__ == '__main__':
    unittest.main()
