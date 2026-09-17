"""Objects -- event source (Births / Deaths) and modal excitation (Birth position)
gates (REQ memory/req-objects-event-source-modal-2026-09-17.md, "Сверка"):

  - registry: `events` Both / Births / Deaths and `excitation` Uniform / Birth position,
    defaults Both / Uniform, absent keys = the defaults, the older Objects scenes load
  - E1 (Octagon II p5, preflight cells): the packets of the ONE bank of the figure
    over repeated periods equal independent masks (births inside the figure's
    current cells, deaths inside its previous cells): Births [8,16,0,8,8], Deaths
    [0,16,8,0,16], Both = the sum; Births / Both start with one packet of every cell,
    Deaths with none; a uniform packet never touches the per-mode states
  - a manual birth / death on a still figure strikes only its own side; the last
    figure vanishing in Deaths takes ONE last packet into its tail (not lost by
    the move to a tail slot, not repeated, the tail stops feeding once the pulse
    is below TAIL_FLOOR); in Births / Both the tail gets none (the previous cycle)
  - M1 (Jam p3): A Uniform and B Birth position share frequencies, output weights,
    packet moments and a block by block; b of every packet has sum b^2 = m and the
    preflight coefficients on the 11- / 16- / 3-cell banks
  - the Birth position law: a single birth moved between two cells of one fixed
    asymmetric geometry changes the relative coefficients (preflight
    independent_checks), zero participation -> no packet (counted), the mean over
    a degenerate group is invariant to the sign and to a rotation of the basis
    inside the group, the same figure translated (across the seam too) and
    rotated gives the same b
  - the kernel against an independent scalar reference with per-mode packets,
    uniform packets, Attack (0 / 4 ms and a ramp) and a fed tail (<= 1e-12);
    packets enter the resonator INPUT: responses add, the output weights never
    change with a strike
  - switching Events / Excitation on a still field makes no packet; Birth position
    outside its combination makes no packet (counted `unsupported_packets`, never a
    uniform strike); snapshot v4 (per-mode states, fed tails) continues exactly, a
    v3 snapshot is accepted as Both / Uniform, inconsistent fed counts are refused;
    the two scenes and their runner Continue are exact; the bench panel (16 rows,
    the wider 'Birth position' button, the status line) fits a 960 px desktop

    python tests/test_objects_event_source.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_oes_tests/.
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
from casynth_lab import figures as fg                                        # noqa: E402
from casynth_lab import object_resonators as orz                             # noqa: E402
from demos.build_objects_event_source import (CASES, preflight_params, case_cells, scene_for,   # noqa: E402
                                              SIDE_GAIN)

ART = os.path.join(ROOT, 'artifacts', '_oes_tests')
F0 = 110.0
RATE = 6.0
CTX = EngineContext(SR, BLOCK, 2, F0, 1.0, RATE)
GAIN = MASTER_GAIN * 0.7
TOL = 1e-12
PREFLIGHT = os.path.join(ROOT, 'memory', 'research', 'objects-event-source-modal-preflight-2026-09-17.json')
EID = orz.ENGINE_ID
ROWS = COLS = 32
BOTH, BIRTHS, DEATHS = orz.EV_BOTH, orz.EV_BIRTHS, orz.EV_DEATHS
UNIFORM, POSITION = orz.EXC_UNIFORM, orz.EXC_POSITION


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def params(ev=BIRTHS, ex=UNIFORM, **over):
    p = preflight_params(over.pop('events', ev), over.pop('excitation', ex))
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


def generations(e, g, transitions, blocks_per_gen=1):
    """Feed `transitions` Conway steps (one boundary each, `blocks_per_gen` blocks
    rendered per generation); returns [(grid, display)] from the initial field on."""
    out = [(g.copy(), e.display())]
    for _ in range(transitions):
        prev, g = g, step(g)
        e.update_field(g, events_field(prev, g))
        for _b in range(blocks_per_gen):
            e.render_float(GAIN)
        out.append((g.copy(), e.display()))
    return out


def own_packets(gens, events):
    """Independent packets of every sounding figure per generation: births inside its
    current cells, deaths inside its previous cells (Own masks), by the selector."""
    rows = []
    for t in range(1, len(gens)):
        g_prev, d_prev = gens[t - 1]
        g_cur, d_cur = gens[t]
        births = (g_cur != 0) & (g_prev == 0)
        deaths = (g_prev != 0) & (g_cur == 0)
        prev_by_id = {f['id']: f for f in d_prev['figures']}
        row = {}
        for f in d_cur['figures']:
            if f['slot'] < 0:
                continue
            cur = fg.own_mask(np.asarray(f['cells']), ROWS, COLS)
            e_b = int(np.count_nonzero(births & cur))
            p = prev_by_id.get(f['id'])
            e_d = int(np.count_nonzero(deaths & fg.own_mask(np.asarray(p['cells']), ROWS, COLS))) if p else 0
            new = p is None
            if events == BOTH:
                e = e_b + e_d
            elif events == BIRTHS:
                e = e_b
            else:
                e = 0 if new else e_d
            row[f['id']] = (e, float(f['e']))
        rows.append(row)
    return rows


class RegistryTests(unittest.TestCase):
    def test_choices_defaults_older_parameter_sets_and_older_scenes(self):
        spec = registry.get(EID)
        self.assertEqual(spec.choices['events'], ('Both', 'Births', 'Deaths'))
        self.assertEqual(spec.choices['excitation'], ('Uniform', 'Birth position'))
        self.assertEqual((spec.defaults()['events'], spec.defaults()['excitation']), (0, 0))
        names = [p[0] for p in spec.params]
        self.assertEqual(names[:5], ['detector', 'radius_mul', 'spectrum', 'events', 'excitation'])
        e = registry.create(EID, CTX, dict(detector=1, frequency_scale=220.0, decay_s=0.8))
        self.assertEqual((e.params['events'], e.params['excitation']), (0, 0))
        self.assertEqual(orz.MODEL_VERSION, 'ca_object_resonators_n4_v4')
        self.assertEqual(orz.STATE_VERSION, 4)
        for name in ('n4_spectrum', 'n4_neighbor', 'ol_glider', 'ol_galaxy', 'ol_neighbor', 'ora_r1', 'ora_r2',
                     'ora_a1', 'ora_user034', 'oes_e1', 'oes_m1'):
            sc = load_scene(os.path.join(ROOT, 'demos', name + '.json'))
            for side in ('A', 'B'):
                eid, p = sc.variants[side]
                if eid == EID:
                    self.assertIn('events', p)
                    self.assertIn('excitation', p)
        text = orz.overlay(params(DEATHS, POSITION), 32, 32)['text']
        self.assertIn('[Own, Laplace] [Deaths, Birth position]', text)
        self.assertTrue(orz.position_supported(params(BIRTHS, POSITION)))
        for bad in (dict(events=BOTH), dict(detector=1), dict(spectrum=0), dict(fullshape=0)):
            self.assertFalse(orz.position_supported(params(BIRTHS, POSITION, **bad)))


class EventSourceTests(unittest.TestCase):
    def test_e1_packets_of_the_one_bank_follow_independent_masks(self):
        cells = case_cells('E1')
        pf = preflight()['cases']['E1']
        self.assertEqual(pf['period'], 5)
        expect = {BIRTHS: [8, 16, 0, 8, 8], DEATHS: [0, 16, 8, 0, 16]}
        expect[BOTH] = [a + b for a, b in zip(expect[BIRTHS], expect[DEATHS])]
        for events in (BOTH, BIRTHS, DEATHS):
            e = engine(cells, params(events, UNIFORM))
            e.render_float(GAIN)                                   # the start: the field against zeros
            d0 = e.display()
            self.assertEqual(len(d0['figures']), 1)
            fid = d0['figures'][0]['id']
            self.assertEqual(float(d0['figures'][0]['e']), 16.0 if events != DEATHS else 0.0)
            gens = generations(e, grid(cells), 15)
            ids = {f['id'] for _g, d in gens for f in d['figures']}
            self.assertEqual(ids, {fid})                           # one bank the whole period
            rows = own_packets(gens, events)
            seq = [row[fid][1] for row in rows]
            for row in rows:
                self.assertEqual(row[fid][1], float(row[fid][0]))
            self.assertEqual(seq, [float(v) for v in expect[events] * 3])
            self.assertEqual(int(np.count_nonzero(e.zfm)) + int(np.count_nonzero(e.zsm)) + int(np.count_nonzero(e.zum)), 0)
            self.assertEqual(int(e.display()['n_tails_fed']), 0)
            self.assertEqual(int(e.display()['unsupported_packets']), 0)
        # Births + Deaths of the independent masks = Both, generation by generation
        eb = engine(cells, params(BIRTHS, UNIFORM))
        eb.render_float(GAIN)
        rows_b = own_packets(generations(eb, grid(cells), 10), BIRTHS)
        ed = engine(cells, params(DEATHS, UNIFORM))
        ed.render_float(GAIN)
        rows_d = own_packets(generations(ed, grid(cells), 10), DEATHS)
        eo = engine(cells, params(BOTH, UNIFORM))
        eo.render_float(GAIN)
        rows_o = own_packets(generations(eo, grid(cells), 10), BOTH)
        for rb, rd, ro in zip(rows_b, rows_d, rows_o):
            (kb,), (kd,), (ko,) = rb, rd, ro
            self.assertEqual(rb[kb][1] + rd[kd][1], ro[ko][1])

    def test_manual_birth_death_isolation_and_the_last_figure_in_deaths(self):
        block = [(5, 5), (5, 6), (6, 5), (6, 6)]
        fields = [grid(block), grid(block + [(4, 4)]), grid(block), np.zeros((ROWS, COLS), np.uint8)]
        got = {}
        for events in (BOTH, BIRTHS, DEATHS):
            e = engine(block, params(events, UNIFORM))
            e.render_float(GAIN)
            self.assertEqual(float(e.display()['figures'][0]['e']), 4.0 if events != DEATHS else 0.0)
            seq = []
            for k in range(1, 3):
                e.update_field(fields[k], events_field(fields[k - 1], fields[k]))
                e.render_float(GAIN)
                seq.append(float(e.display()['figures'][0]['e']))
            got[events] = seq
            # the figure leaves: its bank becomes a tail
            fid = e.display()['figures'][0]['id']
            s_active = e.figures[fid].slot
            zf_res, zs_res = float(e.zf[s_active]), float(e.zs[s_active])   # the rest of the earlier packets
            e.update_field(fields[3], events_field(fields[2], fields[3]))
            e._boundary()                                           # the boundary alone, then look
            self.assertEqual(len(e.figures), 0)
            self.assertEqual(int(e.role[s_active]), orz.ROLE_FREE)
            tails = np.nonzero(e.role == orz.ROLE_TAIL)[0]
            self.assertEqual(len(tails), 1)
            t = int(tails[0])
            if events == DEATHS:
                a = 4.0 / (4.0 + 2.0)
                self.assertEqual(float(e.zf[t]), zf_res + a)       # the strike travelled with the bank
                self.assertEqual(float(e.zs[t]), zs_res + a)       # (on top of the rest of the earlier ones)
                self.assertEqual(int(e.npulse[t]), 3)              # ... and feeds its three modes
                self.assertEqual(int(e.ndrive[t]), 0)
                self.assertEqual(e.display()['n_tails_fed'], 1)
                y0 = e.raw_block(events=False)
                self.assertGreater(float(np.abs(y0).max()), 1e-4)  # the last packet sounds
                zf_prev = float(e.zf[t])
                fed = []
                for _ in range(12):
                    y, _pk, _nc = e.render_float(GAIN)             # further boundaries: no packet again
                    self.assertLess(float(e.zf[t]), zf_prev if zf_prev > 0.0 else 1.0)
                    zf_prev = float(e.zf[t])
                    fed.append(e.display()['n_tails_fed'])
                self.assertEqual(fed[0], 1)
                self.assertEqual(fed[-1], 0)                        # the pulse decayed below TAIL_FLOOR
                self.assertEqual(int(e.npulse[t]), 0)
                self.assertEqual(int(e.nlive[t]), 3)                # the response rings on
                self.assertGreater(float(np.abs(y).max()), 1e-6)
            else:
                self.assertEqual(float(e.zf[t]) + float(e.zs[t]) + float(e.zu[t]), 0.0)
                self.assertEqual(int(e.npulse[t]), 0)
                self.assertEqual(e.display()['n_tails_fed'], 0)
        self.assertEqual(got[BIRTHS], [1.0, 0.0])
        self.assertEqual(got[DEATHS], [0.0, 1.0])
        self.assertEqual(got[BOTH], [1.0, 1.0])

    def test_deaths_tail_strike_survives_an_eviction_and_a_split(self):
        """Every tail slot busy: the struck bank still reaches a tail slot (the quietest
        tail is evicted to a fading slot) with its pulse; a split in Deaths strikes the
        old bank's tail once with the deaths of its previous mask."""
        cells = case_cells('M1')
        e = engine(cells, params(DEATHS, UNIFORM))
        e.render_float(GAIN)
        for s in range(orz.N_ACTIVE, orz.N_ACTIVE + orz.N_TAILS):   # fill the tail pool with quiet ringing tails
            e.role[s] = orz.ROLE_TAIL
            e.nlive[s] = 1
            e.zre[s, 0] = 1e-3 * (1 + s)
            e.ffreq[s, 0] = 200.0 + s
            c, sn = orz.trig_of([200.0 + s], SR)
            e.cth[s, 0], e.sth[s, 0] = float(c[0]), float(sn[0])
            e.wcur[s, 0] = e.wtgt[s, 0] = 0.1
        g0 = grid(cells)
        g1 = step(g0)                                                # Jam gen 1: 2 components (split), 4 deaths
        births = (g1 != 0) & (g0 == 0)
        deaths = (g0 != 0) & (g1 == 0)
        self.assertEqual(int(deaths.sum()), 4)
        old = {f['id']: f for f in e.display()['figures']}
        self.assertEqual(sorted(f['n'] for f in old.values()), [3, 3, 7])   # Jam: three components at first
        e.update_field(g1, events_field(g0, g1))
        e._boundary()
        d = e.display()
        self.assertEqual(d['n_figures'], 2)                          # 11 (a merge: two old banks to tails) + 3
        lost = [f for fid, f in old.items() if fid not in {x['id'] for x in d['figures']}]
        self.assertEqual(sorted(f['n'] for f in lost), [3, 7])
        for f in d['figures']:
            if f['id'] in old:                                       # continuing: the deaths of its previous mask
                self.assertEqual(float(f['e']), float(np.count_nonzero(
                    deaths & fg.own_mask(np.asarray(old[f['id']]['cells']), ROWS, COLS))))
            else:
                self.assertEqual(float(f['e']), 0.0)                 # a new bank: no packet in Deaths
        self.assertGreaterEqual(d['evictions'], 1)
        fed = [int(s) for s in np.nonzero(e.npulse > 0)[0]]
        want = {}
        for f in lost:
            e_last = float(np.count_nonzero(deaths & fg.own_mask(np.asarray(f['cells']), ROWS, COLS)))
            if e_last > 0.0:
                want[e_last / (e_last + 2.0)] = f['modes']
        self.assertEqual(len(fed), len(want))
        self.assertGreater(len(fed), 0)
        for t in fed:
            self.assertTrue(orz.N_ACTIVE <= t < orz.N_ACTIVE + orz.N_TAILS)
            self.assertEqual(int(e.role[t]), orz.ROLE_TAIL)
            a = round(float(e.zf[t]), 9)
            match = [k for k in want if round(k, 9) == a]
            self.assertEqual(len(match), 1)
            self.assertEqual(int(e.npulse[t]), want[match[0]])
        self.assertEqual(int(np.count_nonzero(births)), 5)


class BirthPositionTests(unittest.TestCase):
    def test_m1_sides_share_everything_but_the_distribution_and_b_matches_preflight(self):
        cells = case_cells('M1')
        pf = preflight()['cases']['M1']
        ea = engine(cells, params(BIRTHS, UNIFORM))
        eb = engine(cells, params(BIRTHS, POSITION))
        g = grid(cells)
        seen = {}
        for t in range(0, 40):
            if t > 0:
                prev, g = g, step(g)
                for e in (ea, eb):
                    e.update_field(g, events_field(prev, g))
            for _ in range(int(round(SR / RATE / BLOCK))):
                ea.render_float(GAIN)
                eb.render_float(GAIN)
                self.assertEqual(sorted(ea.figures), sorted(eb.figures))
                for fid, fa in ea.figures.items():                         # the banks of the figures
                    sa, sb = fa.slot, eb.figures[fid].slot
                    self.assertEqual(sa, sb)
                    if sa < 0:
                        continue
                    np.testing.assert_array_equal(ea.ffreq[sa], eb.ffreq[sb])      # frequencies
                    np.testing.assert_array_equal(ea.wtgt[sa], eb.wtgt[sb])        # output weights
                    np.testing.assert_array_equal(ea.wcur[sa], eb.wcur[sb])
                    self.assertEqual(float(ea.last_e[sa]), float(eb.last_e[sb]))   # packet moments and sizes
                    self.assertEqual(float(ea.last_a[sa]), float(eb.last_a[sb]))   # a
                    self.assertEqual(int(ea.ndrive[sa]), int(eb.ndrive[sb]))
                # (tail slots may differ in index: the quietest tail is evicted by energy)
            da, db_ = ea.display(), eb.display()
            self.assertEqual([f['id'] for f in da['figures']], [f['id'] for f in db_['figures']])   # same tracker
            for fa, fb in zip(da['figures'], db_['figures']):
                if fa['slot'] < 0 or fa['e'] <= 0.0:
                    continue
                m = fa['modes']
                self.assertEqual(fa['b'], [1.0] * m)
                b = np.asarray(fb['b'])
                self.assertAlmostEqual(float((b * b).sum()), float(m), places=9)
                seen.setdefault(((t - 1) % pf['period'] + 1, fa['n'], int(fa['e'])), b)
        self.assertEqual(int(eb.display()['zero_participation']), 0)
        self.assertEqual(int(eb.display()['unsupported_packets']), 0)
        # the preflight coefficients of the 11-, 16- and 3-cell banks (gen 1 / 2 / 3)
        for ph in pf['phases']:
            for bank in ph['banks']:
                if bank['births'] <= 0:
                    continue
                key = (int(ph['generation']), len(bank['cells']), int(bank['births']))
                self.assertIn(key, seen)
                np.testing.assert_allclose(seen[key], bank['weights_spatial'], rtol=1e-9, atol=1e-9)
                self.assertEqual(len(bank['frequencies_hz']), len(bank['weights_spatial']))
        self.assertGreaterEqual(len(seen), 3)
        # the sounds differ once a transition strikes (the start packet is uniform on both)
        ys = {}
        for ex in (UNIFORM, POSITION):
            e = engine(cells, params(BIRTHS, ex))
            y0 = run(e, 20)
            g0 = grid(cells)
            g1 = step(g0)
            e.update_field(g1, events_field(g0, g1))
            ys[ex] = (y0, run(e, 20))
        # (a whole-figure birth gives b = 1 up to rounding: the same sound to 1e-12, not bit for bit)
        np.testing.assert_allclose(ys[UNIFORM][0], ys[POSITION][0], rtol=0, atol=1e-12)
        self.assertGreater(float(np.abs(ys[UNIFORM][1] - ys[POSITION][1]).max()), 1e-6)

    def _graph(self, cells):
        p = params(BIRTHS, POSITION)
        freqs, amps, graph = orz.laplace_modes_of(np.asarray(cells), ROWS, COLS, F0, orz.laplace_settings(p),
                                                  None, with_graph=True)
        return freqs, amps, graph

    def _born(self, cells, born_cells, graph):
        cells = np.asarray(cells)
        born = np.zeros((ROWS, COLS), bool)
        for r, c in born_cells:
            born[r, c] = True
        return born[cells[graph['order'], 0], cells[graph['order'], 1]]

    def test_law_single_birth_zero_participation_degeneracy_sign_rotation_translation(self):
        pf = preflight()
        chk = pf['independent_checks']['single_birth_same_11_cell_geometry']
        cells11 = [tuple(c) for c in pf['cases']['M1']['phases'][0]['banks'][0]['cells']]
        self.assertEqual(len(cells11), 11)
        freqs, _amps, graph = self._graph(cells11)
        self.assertEqual(len(graph['idx']), 3)
        np.testing.assert_allclose(freqs, pf['cases']['M1']['phases'][0]['banks'][0]['frequencies_hz'], rtol=1e-9)
        got = []
        for cell in chk['cells']:
            b, total = orz.birth_position_weights(graph['L'], graph['idx'], self._born(cells11, [tuple(cell)], graph))
            self.assertGreater(total, orz.ZERO_PART)
            got.append(b)
        np.testing.assert_allclose(got, chk['input_weights'], rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.norm(got[0] - got[1])), chk['max_profile_distance'], places=9)
        self.assertGreater(float(np.abs(got[0] - got[1]).max()), 1.0)     # the place changes the profile
        # zero participation: no births -> no packet (b = 0, total 0)
        b0, total0 = orz.birth_position_weights(graph['L'], graph['idx'], np.zeros(11, bool))
        self.assertEqual(total0, 0.0)
        np.testing.assert_array_equal(b0, np.zeros(3))
        self.assertEqual(list(pf['independent_checks']['zero_event_profile']), [0.0, 0.0, 0.0])
        # degenerate groups: E1 gen 1 (24 cells) has ranks [2, 8, 1] -- the group mean is
        # invariant to the sign of the eigenvectors and to a rotation inside the group
        cells24 = [tuple(c) for c in pf['cases']['E1']['phases'][0]['banks'][0]['cells']]
        _f, _a, g24 = self._graph(cells24)
        lam, V = np.linalg.eigh(g24['L'])
        groups = orz.degenerate_groups(lam, g24['idx'])
        self.assertEqual([len(q) for q in groups], pf['cases']['E1']['phases'][0]['banks'][0]['degenerate_ranks'])
        born = self._born(cells24, [tuple(c) for c in cells24[:5]], g24)
        b_ref, total_ref = orz.birth_position_weights(g24['L'], g24['idx'], born)
        rng = np.random.default_rng(7)

        def p_of(Vx):
            part = (Vx[born, :] ** 2).sum(axis=0)
            p = np.array([part[q].sum() / len(q) for q in groups])
            return np.sqrt(len(p) * p / p.sum())
        Vs = V * rng.choice([-1.0, 1.0], size=V.shape[1])[None, :]
        np.testing.assert_allclose(p_of(Vs), b_ref, rtol=0, atol=1e-12)
        Vr = V.copy()
        for q in groups:
            if len(q) > 1:
                Q, _r = np.linalg.qr(rng.normal(size=(len(q), len(q))))
                Vr[:, q] = V[:, q] @ Q
        np.testing.assert_allclose(p_of(Vr), b_ref, rtol=0, atol=1e-12)
        # a translation (across the seam) and a 90 degree rotation of the figure with
        # its births give the same b
        for dr, dc in ((0, 0), (20, 25), (-9, 13)):
            moved = [((r + dr) % ROWS, (c + dc) % COLS) for r, c in cells11]
            _f, _a, gm = self._graph(moved)
            bm, _t = orz.birth_position_weights(gm['L'], gm['idx'],
                                                self._born(moved, [((8 + dr) % ROWS, (9 + dc) % COLS)], gm))
            np.testing.assert_allclose(bm, got[0], rtol=0, atol=1e-9)
        rot = [(c, ROWS - 1 - r) for r, c in cells11]
        _f, _a, gr = self._graph(rot)
        br, _t = orz.birth_position_weights(gr['L'], gr['idx'], self._born(rot, [(9, ROWS - 1 - 8)], gr))
        np.testing.assert_allclose(br, got[0], rtol=0, atol=1e-9)
        # a whole-figure birth (the start) is the uniform packet
        ball, _t = orz.birth_position_weights(graph['L'], graph['idx'], np.ones(11, bool))
        np.testing.assert_allclose(ball, np.ones(3), rtol=0, atol=1e-12)

    def test_unsupported_combination_makes_no_packet_and_is_reported(self):
        cells = case_cells('E1')
        for bad in (dict(detector=1), dict(events=BOTH), dict(events=DEATHS), dict(spectrum=0), dict(fullshape=0)):
            e = engine(cells, params(BIRTHS, POSITION, **bad))
            e.render_float(GAIN)
            d = e.display()
            self.assertFalse(d['position_supported'])
            self.assertEqual(d['position_condition'], orz.POSITION_CONDITION)
            gens = generations(e, grid(cells), 6)
            d = e.display()
            self.assertGreaterEqual(int(d['unsupported_packets']), 1)   # every refused packet is counted
            self.assertEqual(float(np.abs(e.zf).sum() + np.abs(e.zs).sum() + np.abs(e.zfm).sum()), 0.0)
            for _g, dd in gens:
                for f in dd['figures']:
                    self.assertEqual(float(f['e']), 0.0)
            y = run(e, 5)
            self.assertEqual(float(np.abs(y).max()), 0.0)          # silence: never a uniform strike instead
        e = engine(cells, params(BIRTHS, POSITION))
        e.render_float(GAIN)
        self.assertTrue(e.display()['position_supported'])
        self.assertEqual(e.display()['unsupported_packets'], 0)
        self.assertGreater(float(np.abs(e.zfm).sum()), 0.0)


def scalar_v4(slots, blocks, sr=SR, ramp_n=882, hp_hz=20.0, out_scale=orz.OUT_SCALE, g0=GAIN,
              decay_s=0.8, attack0=0.0):
    """An INDEPENDENT scalar sample path of the v4 excitation (REQ formulas):
    slots = [dict(freqs, weights, pan_p, nd)] active banks, fixed tuning;
    blocks = [{'inject': {slot: a}, 'inject_modes': {slot: [delta_j]}, 'attack': ms,
               'gain': g, 'tail': {slot: e}}]  (boundary actions; 'tail' = a last
    uniform packet e and the bank becomes a tail that keeps feeding its nd modes
    until every pulse state is below 1e-7 at a block boundary)
        p = 0.75 ((1-qf) zf - (1-qs) zs) / (qs-qf);   u = q u + (1-q) p      (slot)
        pm_j, um_j the same on the per-mode states;   mode j < fed gets u + um_j
        q = exp(-ln 9 / (sr ms / 1000)) (0 at ms = 0), a change ramps q linearly."""
    qf = math.exp(-1.0 / (sr * 0.00025))
    qs = math.exp(-1.0 / (sr * 0.002))
    hp_h = math.exp(-2.0 * math.pi * hp_hz / sr)
    r = 10.0 ** (-3.0 / (sr * decay_s))
    S = []
    for sd in slots:
        n = len(sd['freqs'])
        S.append(dict(re=[0.0] * n, im=[0.0] * n, c=[math.cos(2 * math.pi * f / sr) for f in sd['freqs']],
                      s=[math.sin(2 * math.pi * f / sr) for f in sd['freqs']], w=list(sd['weights']),
                      fed=int(sd['nd']), tail=False, zf=0.0, zs=0.0, u=0.0,
                      zfm=[0.0] * n, zsm=[0.0] * n, um=[0.0] * n,
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
        for st in S:                                                 # the boundary: a finished last packet
            if st['tail'] and st['fed'] > 0:
                if (max(abs(st['zf']), abs(st['zs']), abs(st['u'])) < 1e-7
                        and max(abs(v) for v in st['zfm'][:st['fed']] + st['zsm'][:st['fed']] + st['um'][:st['fed']]) < 1e-7):
                    st['zf'] = st['zs'] = st['u'] = 0.0
                    st['zfm'] = [0.0] * len(st['zfm'])
                    st['zsm'] = [0.0] * len(st['zsm'])
                    st['um'] = [0.0] * len(st['um'])
                    st['fed'] = 0
        if 'attack' in blk and q_of(blk['attack']) != q_tgt:
            q_tgt = q_of(blk['attack'])
            q_inc = (q_tgt - q_cur) / ramp_n
            q_left = ramp_n
        for si, a in blk.get('inject', {}).items():
            S[si]['zf'] += a
            S[si]['zs'] += a
        for si, d in blk.get('inject_modes', {}).items():
            for j, v in enumerate(d):
                S[si]['zfm'][j] += v
                S[si]['zsm'][j] += v
        for si, e in blk.get('tail', {}).items():
            a = e / (e + 2.0)
            S[si]['zf'] += a
            S[si]['zs'] += a
            S[si]['tail'] = True
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
                    if j < st['fed']:
                        pm = 0.75 * ((1.0 - qf) * st['zfm'][j] - (1.0 - qs) * st['zsm'][j]) / (qs - qf)
                        st['zfm'][j] *= qf
                        st['zsm'][j] *= qs
                        um = q_cur * st['um'][j] + (1.0 - q_cur) * pm
                        st['um'][j] = um
                        nre = r * (c * re - sn * im) + (u + um)
                    else:
                        nre = r * (c * re - sn * im)
                    nim = r * (sn * re + c * im)
                    st['re'][j], st['im'][j] = nre, nim
                    acc += st['w'][j] * nre
                L += st['L'] * acc
                R += st['R'] * acc
            yl = hp_h * ((hp[0][0] + L) - hp[0][1])
            hp[0] = [yl, L]
            yr = hp_h * ((hp[1][0] + R) - hp[1][1])
            hp[1] = [yr, R]
            out.append((yl * out_scale * g_cur, yr * out_scale * g_cur))
    return np.asarray(out)


def bare_engine(p=None):
    e = registry.create(EID, CTX, p or dict(detector=1, frequency_scale=220.0, decay_s=0.8, attack_ms=0.0,
                                            events=0, excitation=0))
    e.init(np.zeros((ROWS, COLS), np.uint8), None, GAIN)
    return e


def setup_slot(e, s, freqs, nd, pan_p, weights=None):
    n = len(freqs)
    e.role[s] = orz.ROLE_ACTIVE
    e.slot_id[s] = 1000 + s
    e.ndrive[s] = nd
    e.nlive[s] = n
    e.ffreq[s, :n] = freqs
    c, sn = orz.trig_of(np.asarray(freqs), e.sr)
    e.cth[s, :n] = c
    e.sth[s, :n] = sn
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights)
    e.wcur[s, :n] = w
    e.wtgt[s, :n] = w
    L, R = orz.pan_of(pan_p * (COLS - 1), COLS)
    e.pan[s] = (L, 0.0, L, R, 0.0, R)


class KernelTests(unittest.TestCase):
    def test_kernel_matches_the_scalar_reference_with_per_mode_packets_attack_and_a_fed_tail(self):
        freqs_a = [110.0, 434.66, 551.01, 777.7]
        freqs_b = [80.8, 130.1, 190.2]
        e = bare_engine()
        setup_slot(e, 0, freqs_a, 4, 0.3)
        setup_slot(e, 1, freqs_b, 3, 0.9)
        slots = [dict(freqs=freqs_a, weights=[0.25] * 4, pan_p=0.3, nd=4),
                 dict(freqs=freqs_b, weights=[1 / 3] * 3, pan_p=0.9, nd=3)]
        blocks = [dict(inject_modes={0: [0.6 * 1.289, 0.0, 0.6 * 1.157, 0.1]}, inject={1: 0.5}), {},
                  dict(attack=4.0), dict(inject_modes={0: [0.2, 0.3, 0.0, 0.0]}), {}, {},
                  dict(inject={0: 0.3}, inject_modes={1: [0.1, 0.2, 0.3]}), dict(gain=GAIN * 1.4), {},
                  dict(attack=0.0), {}, dict(inject_modes={0: [0.4, 0.0, 0.0, 0.2]}), {}, {}, {},
                  dict(tail={1: 4.0}), {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
        ref = scalar_v4(slots, blocks)
        got = []
        for blk in blocks:
            if 'attack' in blk:
                e.set_params(dict(e.params, attack_ms=blk['attack']))
            for si, a in blk.get('inject', {}).items():
                e.inject(si, a)
            for si, d in blk.get('inject_modes', {}).items():
                e.inject_modes(si, d)
            for si, ev in blk.get('tail', {}).items():
                e._to_tail(si, ev)
            y, _pk, _nc = e.render_float(blk.get('gain', e.gg[orz.R_TGT]))
            got.append(y)
        got = np.concatenate(got)
        self.assertLessEqual(float(np.abs(got - ref).max()), TOL)
        self.assertGreater(float(np.abs(got).max()), 1e-3)
        t = int(np.nonzero(e.role == orz.ROLE_TAIL)[0][0])
        self.assertEqual(int(e.npulse[t]), 0)                       # the tail's last packet has finished
        self.assertEqual(int(e.nlive[t]), 3)
        # Attack 0 and a uniform packet only: the per-mode states stay 0 and the path is the v3 one
        self.assertEqual(float(np.abs(e.zfm[0]).max()), float(np.abs(e.zfm[0]).max()))

    def test_packets_enter_the_resonator_input_responses_add_and_weights_never_change(self):
        freqs = [110.0, 434.66, 551.01]
        w = [0.7, 0.35, 0.2]

        def fresh():
            e = bare_engine(dict(detector=0, frequency_scale=220.0, decay_s=1.39, attack_ms=4.0, events=1, excitation=1))
            setup_slot(e, 0, freqs, 3, 0.5, w)
            return e
        d1, d2 = [0.6 * 1.289, 0.0, 0.6 * 1.157], [0.4 * 0.78, 0.4 * 0.70, 0.4 * 1.38]
        y1 = run_with(fresh(), {0: d1}, 40)
        y2 = run_with(fresh(), {6: d2}, 40)
        e12 = fresh()
        y12 = run_with(e12, {0: d1, 6: d2}, 40)
        self.assertLessEqual(float(np.abs(y12 - (y1 + y2)).max()), TOL)     # linear at the input
        np.testing.assert_array_equal(e12.wcur[0, :3], w)                     # a strike never recolours
        np.testing.assert_array_equal(e12.wtgt[0, :3], w)
        self.assertEqual(int(e12.wleft[0]), 0)
        # a zero coefficient leaves that mode silent, however hard the packet
        e0 = fresh()
        e0.inject_modes(0, [0.0, 3.0, 0.0])
        run(e0, 20)
        self.assertEqual(float(np.hypot(e0.zre[0, 0], e0.zim[0, 0])), 0.0)
        self.assertEqual(float(np.hypot(e0.zre[0, 2], e0.zim[0, 2])), 0.0)
        self.assertGreater(float(np.hypot(e0.zre[0, 1], e0.zim[0, 1])), 1e-3)
        # Attack acts on the per-mode packet: 4 ms softens the first sample of the response
        first = {}
        for ms in (0.0, 4.0):
            e = bare_engine(dict(detector=0, frequency_scale=220.0, decay_s=1.39, attack_ms=ms, events=1, excitation=1))
            setup_slot(e, 0, freqs, 3, 0.0, w)
            e.inject_modes(0, d1)
            y = e.raw_block(events=False)
            first[ms] = abs(float(y[0, 0]))
        self.assertLess(first[4.0], first[0.0] * 0.1)


def run_with(e, packets, n_blocks):
    out = []
    for k in range(n_blocks):
        if k in packets:
            e.inject_modes(0, packets[k])
        y, _pk, _nc = e.render_float(GAIN)
        out.append(y)
    return np.concatenate(out)


class StateTests(unittest.TestCase):
    def test_switching_events_or_excitation_on_a_still_field_makes_no_packet(self):
        cells = case_cells('E1')
        e = engine(cells, params(BIRTHS, UNIFORM))
        twin = engine(cells, params(BIRTHS, UNIFORM))
        run(e, 3)
        run(twin, 3)
        for p in (dict(events=DEATHS), dict(events=BOTH), dict(excitation=POSITION), dict(events=BIRTHS),
                  dict(excitation=UNIFORM)):
            e.set_params(dict(e.params, **p))
            np.testing.assert_array_equal(e.render_float(GAIN)[0], twin.render_float(GAIN)[0])
            self.assertEqual(e.display()['changes'], twin.display()['changes'])
        np.testing.assert_array_equal(e.zf, twin.zf)
        np.testing.assert_array_equal(e.zfm, twin.zfm)

    def test_snapshot_v4_continues_exactly_v3_is_accepted_and_bad_fed_counts_are_refused(self):
        cells = case_cells('M1')
        e = engine(cells, params(BIRTHS, POSITION))
        g = grid(cells)
        gens = [g]
        for _ in range(3):
            g = step(g)
            gens.append(g)
        e.render_float(GAIN)
        e.update_field(gens[1], events_field(gens[0], gens[1]))
        e.render_float(GAIN)                                       # mid-pulse with per-mode states
        self.assertGreater(float(np.abs(e.zfm).max()), 0.0)
        st = e.export_state()
        self.assertEqual(st['version'], 4)
        self.assertEqual(st['model_version'], orz.MODEL_VERSION)
        for name in ('zfm', 'zsm', 'zum', 'npulse', 'last_b'):
            self.assertIn(name, st)
        self.assertEqual(st['counters'].shape, (orz.N_COUNTERS,))
        twin = registry.create(EID, CTX, dict(e.params))
        twin.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
        twin.restore_state(e._grid, e._exc, copy.deepcopy(st))
        for k in range(2, 4):
            for x in (e, twin):
                x.update_field(gens[k], events_field(gens[k - 1], gens[k]))
            for _ in range(20):
                np.testing.assert_array_equal(e.render_float(GAIN)[0], twin.render_float(GAIN)[0])
        # a Deaths tail mid-packet continues exactly too
        ed = engine(cells, params(DEATHS, UNIFORM))
        ed.render_float(GAIN)
        ed.update_field(gens[1], events_field(gens[0], gens[1]))
        ed.render_float(GAIN)
        self.assertEqual(ed.display()['n_tails_fed'], 1)
        st2 = ed.export_state()
        td = registry.create(EID, CTX, dict(ed.params))
        td.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
        td.restore_state(ed._grid, ed._exc, copy.deepcopy(st2))
        for _ in range(30):
            np.testing.assert_array_equal(ed.render_float(GAIN)[0], td.render_float(GAIN)[0])
        # refusals
        bad = copy.deepcopy(st)
        s0 = int(np.nonzero(bad['role'] == orz.ROLE_ACTIVE)[0][0])
        bad['npulse'][s0] = 1                                       # an active slot never has a tail count
        with self.assertRaises(ValueError):
            fresh = registry.create(EID, CTX, dict(e.params))
            fresh.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
            fresh.restore_state(e._grid, e._exc, bad)
        with self.assertRaises(ValueError):                          # a v3 state must say model v3
            fresh = registry.create(EID, CTX, dict(e.params))
            fresh.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
            fresh.restore_state(e._grid, e._exc, dict(copy.deepcopy(st), version=3))
        # a v3 snapshot (no Events / Excitation) of a Both / Uniform run: the same sound
        eo = engine(cells, params(BOTH, UNIFORM))
        eo.render_float(GAIN)
        eo.update_field(gens[1], events_field(gens[0], gens[1]))
        eo.render_float(GAIN)
        st3 = eo.export_state()
        old = {k: v for k, v in st3.items() if k not in ('zfm', 'zsm', 'zum', 'npulse', 'last_b')}
        old.update(version=3, model_version='ca_object_resonators_n4_v3', counters=st3['counters'][:5].copy(),
                   params={k: v for k, v in st3['params'].items() if k not in ('events', 'excitation')})
        t3 = registry.create(EID, CTX, dict(eo.params))
        t3.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
        t3.restore_state(eo._grid, eo._exc, old)
        self.assertEqual((t3.params['events'], t3.params['excitation']), (0, 0))
        self.assertEqual(t3.counters.shape, (orz.N_COUNTERS,))
        for k in range(2, 4):
            for x in (eo, t3):
                x.update_field(gens[k], events_field(gens[k - 1], gens[k]))
            for _ in range(20):
                np.testing.assert_array_equal(eo.render_float(GAIN)[0], t3.render_float(GAIN)[0])


class SceneTests(unittest.TestCase):
    def test_scenes_sides_differ_only_in_the_selector_and_continue_exactly(self):
        pf = preflight()
        for case in CASES:
            doc = scene_for(case)
            sc = scene_from_doc(doc)
            self.assertEqual(doc['cells'], [[int(r), int(c)] for r, c in pf['cases'][case['case']]['cells']])
            self.assertEqual(doc['rate_hz'], 6.0)
            self.assertTrue(case['hypothesis'].startswith('Гипотеза - '))
            (ea, pa), (eb, pb) = sc.variants['A'], sc.variants['B']
            self.assertEqual((ea, eb), (EID, EID))
            diff = {k for k in pa if pa[k] != pb[k]}
            self.assertEqual(diff, {'events'} if case['id'] == 'oes_e1' else {'excitation'})
            self.assertEqual((pa['detector'], pa['spectrum'], pa['fullshape'], pa['attack_ms'], pa['decay_s']),
                             (0, 1, 1, 4.0, 1.39))
            self.assertEqual((pa['shape'], pa['alpha'], pa['dyn'], pa['n'], pa['spread'], pa['harm']),
                             (0.0, 0.0, 0.0, 3, 1.0, 0.87))
            self.assertEqual(doc['audio']['side_gain'], SIDE_GAIN[case['id']])
            on_disk = load_scene(os.path.join(ROOT, 'demos', case['id'] + '.json'))
            self.assertEqual(on_disk.variants, sc.variants)
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
            db_ = snap['display']['B']
            if case['id'] == 'oes_m1':
                self.assertEqual(db_['excitation_name'], 'Birth position')
                self.assertTrue(db_['position_supported'])
                self.assertEqual(db_['unsupported_packets'], 0)
            else:
                self.assertEqual(db_['events_name'], 'Deaths')

    def test_headless_bench_panel_rows_choice_widths_and_the_status_line(self):
        import pygame
        import demo_bench as db
        from casynth_lab.audio_out import LiveEngine
        scene = scene_from_doc(scene_for(CASES[1]))
        runner = DemoRunner(scene)
        eng = LiveEngine(runner, sink=lambda m, b: None)
        pygame.init()
        app = db.BenchApp(scene, eng)
        n_rows = len(registry.get(EID).params) + len(registry.get(EID).ranges)
        self.assertEqual(n_rows, 16)
        self.assertEqual(app._params_height(EID), n_rows * db.ROW_H)
        self.assertLessEqual(app.height, 1000)
        low = db.BenchApp(scene, eng, max_height=880)
        self.assertLessEqual(low.height, 880)
        self.assertGreaterEqual(low.figure_rows, db.FIGURE_ROWS_MIN)
        rows = {spec[0]: rect for spec, rect in app._param_rows(EID)}
        ex = app._choice_rects(EID, 'excitation', rows['excitation'][1])
        ev = app._choice_rects(EID, 'events', rows['events'][1])
        self.assertEqual([r[2] for _v, r in ex], [92, 92])                    # 'Birth position' fits
        self.assertEqual([r[2] for _v, r in ev], [60, 60, 60])
        for _v, r in ex + ev:
            self.assertLessEqual(r[0] + r[2], app.panel_x + db.PANEL_W)
        screen = pygame.Surface((app.width, app.height))
        font, small = pygame.font.SysFont(db.FONT_NAMES, 17), pygame.font.SysFont(db.FONT_NAMES, 14)
        eng.start()
        try:
            eng.post('start')
            eng.post('select', side='B')
            time.sleep(0.4)
            app.draw(screen, font, small)
            snap = eng.snapshot()
            self.assertEqual(snap['display']['B']['excitation_name'], 'Birth position')
            # the status line of an unsupported combination is drawn in the error colour
            fake = dict(snap['display']['B'])
            fake.update(position_supported=False, excitation=1, unsupported_packets=3)
            y0 = app.params_y + n_rows * db.ROW_H + 6
            screen.fill(db.C_BG)
            app._draw_display(screen, small, fake, app.panel_x, y0)
            arr = pygame.surfarray.array3d(screen).astype(int)

            def reddish(y_lo, y_hi):                                # C_ERR (230, 120, 90) and its blends
                band = arr[app.panel_x:app.panel_x + db.PANEL_W, y_lo:y_hi]
                return bool(((band[..., 0] > 120) & (band[..., 0] > band[..., 1] + 40)).any())
            self.assertTrue(reddish(y0 + 26, y0 + 44))               # the status line
            self.assertFalse(reddish(y0, y0 + 14))                   # the first line is plain
            self.assertEqual(db.FIGURE_HEAD_H, 48)
        finally:
            eng.stop()
            pygame.quit()


if __name__ == '__main__':
    os.makedirs(ART, exist_ok=True)
    unittest.main(verbosity=1)
