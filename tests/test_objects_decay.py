"""Objects -- Decay law (losses from the history of the cells) gates, REQ
memory/req-objects-decay-2026-09-18.md section 5 "Техническая приёмка":

  - registry: `decay_law` Fixed / Common / Modal (0..2, default 0) after `decay_s`;
    the combination rule (an adaptive law needs Laplace + full) refused by the
    registry, the engine, the scene loader, the runner (set_param / copy_spectrum,
    before anything changes; a live refusal is not journaled) and the snapshot;
    the bench locks (inactive texts) both ways; every Objects scene loads
  - the independent history: births / death and re-birth / continuous life / a
    glider through the seam / an unchanged field / pause and resume / several
    edits in one block / a re-set value / a knob / restore / an extra update_field
    -- the engine's age equals an independent numpy model block by block; the
    targets q / T at the preflight's transition samples (D1, D2 with its pauses,
    D3 with split / merge) equal the preflight for Common and Modal
  - the mathematics: P rows sum to 1, sign / basis invariance of a degenerate
    group, the blinker's exact P; translation across the seam and a 90 degree
    rotation with the history carried give the same targets; equal ages ->
    Common == Modal, one mode -> the same; two identical geometries with
    different artificial histories: losses differ, frequencies / weights / the
    next packet do not; the order q -> T -> gamma (Common = the mean RATE), the
    clip tolerance and the error bound of q
  - the audio reference: the adaptive kernel against a scalar reference (the 20 ms
    smoother, the update order, per-mode r) <= 1e-12 through a target change; an
    amplitude T60 of a free mode at frozen gamma; zero input never gains energy
    from a loss change; the D2 modal envelopes of the REAL engine in both pauses:
    upper over lower grows in Modal and stays flat in Common
  - the isolation of the three experiments: per transition the sides share ids,
    frequencies, weights, packets, a, Attack; only the losses differ
  - the memory of the sound: a law / Decay change makes no packet, no reset, no
    revival; a vanished bank takes its applied / target gamma into its tail and
    holds it; a returning figure starts fresh; a switch back to Fixed settles the
    bank onto the global r
  - compatibility: the pinned WAVs of the 2026-09-17 / 09-18 records (E1, M1, M2)
    equal the current code with the default law bit for bit; a v5 snapshot imports
    as Fixed with u = 1 on the live cells (the same sound)
  - snapshots: v6 continues exactly from inside a D2 pause, during a gamma
    transition (D1 after a law switch) and after split / merge (D3); refusals
  - the D1 control: the mean target and applied gamma of the Common side over
    5..20 s within 1 % of ln(1000) / 0.947926
  - scenes: cells == preflight, rates, seconds, the sides differ only in the law
    (D1: and Decay), side gains, on disk == builder, the D2 pause commands at the
    exact samples in the runner's journal, the level windows within 1 dB,
    the bench panel (18 rows, <= 880 px, the Decay law buttons, the row text)

    python tests/test_objects_decay.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_od_tests/.
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
from casynth_lab.scene import SceneError                                      # noqa: E402
from casynth_lab import object_resonators as orz                             # noqa: E402
from casynth_lab import figures as fg                                        # noqa: E402
from casynth_lab import event_network as en                                  # noqa: E402
from demos.build_objects_decay import (CASES, PREFLIGHT, SIDE_GAIN, STABLE_DECAY_S, D2_PAUSES,   # noqa: E402
                                       scene_for, side_params, case_cells, preflight_case, script_of,
                                       window_mask, render_sides)
from demos.build_objects_event_source import rms_db                          # noqa: E402

ART = os.path.join(ROOT, 'artifacts', '_od_tests')
F0 = 110.0
GAIN = MASTER_GAIN * 0.7
EID = orz.ENGINE_ID
ROWS = COLS = 32
FIXED, COMMON, MODAL = orz.LAW_FIXED, orz.LAW_COMMON, orz.LAW_MODAL
OBJECT_SCENES = ('n4_spectrum', 'n4_neighbor', 'ol_glider', 'ol_galaxy', 'ol_neighbor',
                 'ora_r1', 'ora_r2', 'ora_a1', 'ora_user034', 'oes_e1', 'oes_m1', 'obs_m2',
                 'od_d1', 'od_d2', 'od_d3')
PINNED = (('objects_event_source_modal_2026_09_17', 2), ('objects_birth_strength_2026_09_18', 1))


def ctx(rate=2.0):
    return EngineContext(SR, BLOCK, 2, F0, 1.0, rate)


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def params(law=FIXED, decay=STABLE_DECAY_S, **over):
    p = side_params(law, decay)
    p.update(over)
    return p


def grid(cells, rows=ROWS, cols=COLS):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def engine(cells, p, gain=GAIN, rate=2.0):
    e = registry.create(EID, ctx(rate), p)
    e.init(grid(cells), None, gain)
    return e


def run(e, n_blocks, gain=GAIN):
    out = []
    for _ in range(n_blocks):
        y, _pk, _nc = e.render_float(gain)
        out.append(y)
    return np.concatenate(out) if out else np.zeros((0, 2))


def gens_of(cells, n):
    g = grid(cells)
    out = [g]
    for _ in range(n):
        g = step(g)
        out.append(g)
    return out


class AgeModel:
    """The history of the REQ written independently of the engine: u = 1 on a born
    position, 0 on a dead one, survivors keep theirs; after every rendered block
    u *= exp(-B / (SR tau)); the accepted transition = the last rendered field
    against the field at the render."""
    def __init__(self, shape):
        self.u = np.zeros(shape)
        self.prev = np.zeros(shape, np.uint8)
        self.k = math.exp(-BLOCK / (SR * 0.5))

    def render(self, grid_now):
        g = np.asarray(grid_now, np.uint8)
        born = (g != 0) & (self.prev == 0)
        dead = (self.prev != 0) & (g == 0)
        self.u[born] = 1.0
        self.u[dead] = 0.0
        self.prev = g.copy()
        u_at_boundary = self.u.copy()
        self.u *= self.k
        return u_at_boundary


def drive_case(case_id, law, gain=GAIN, on_block=None, decay=STABLE_DECAY_S):
    """Drive one preflight case through the engine on the sample clock of the REQ
    (steps at ca >= (gen + 1) SR / rate, the pause schedule of the case); on_block(e,
    out_samples, gen) after every block.  Returns the engine."""
    case = preflight_case(case_id)
    e = engine(case['cells'], params(law, decay), gain, rate=case['rate_hz'])
    g = grid(case['cells'])
    gen = 0
    ca = 0
    out = 0
    stepn = SR / case['rate_hz']
    sched = {int(round(t * SR)): on for t, on in case['pause_schedule']}
    paused = False
    for _ in range(int(math.ceil(case['seconds'] * SR / BLOCK))):
        if out in sched:
            paused = sched[out]
        if not paused and ca >= (gen + 1) * stepn:
            prev, g = g, step(g)
            gen += 1
            e.update_field(g, events_field(prev, g))
        e.render_float(gain)
        if on_block is not None:
            on_block(e, out, gen)
        out += BLOCK
        if not paused:
            ca += BLOCK
    return e


def rendered_like(rec):
    """Render the record's embedded scene (older records: the keys absent then get
    their bit-exact defaults, journal = one 'start' at 0) with the CURRENT code and
    compare every output with the stored WAV bit for bit; {output: equal}."""
    doc = copy.deepcopy(rec.meta['scene'])
    for group in ('variants', 'factory_variants'):
        for side, var in (doc.get(group) or {}).items():
            if var.get('engine_id') == EID:
                for k, v in orz.OPTIONAL_PARAMS.items():
                    var['engine_params'].setdefault(k, v)
    for side, per_engine in (doc.get('param_memory') or {}).items():
        if EID in per_engine:
            for k, v in orz.OPTIONAL_PARAMS.items():
                per_engine[EID].setdefault(k, v)
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
    def test_spec_defaults_hints_combination_rule_and_every_objects_scene(self):
        spec = registry.get(EID)
        self.assertEqual(spec.spec_of('decay_law'), ('decay_law', 'Decay law', 0, 2, True, 0))
        names = [p[0] for p in spec.params]
        self.assertEqual(names.index('decay_law'), names.index('decay_s') + 1)
        self.assertEqual(spec.choices['decay_law'], ('Fixed', 'Common', 'Modal'))
        self.assertEqual(orz.LAW_NAMES, ('Fixed', 'Common age', 'Modal age'))
        self.assertEqual(orz.OPTIONAL_PARAMS['decay_law'], 0)
        self.assertEqual((orz.MODEL_VERSION, orz.STATE_VERSION), ('ca_object_resonators_n4_v6', 6))
        self.assertEqual(orz.COMPATIBLE_STATES[5], 'ca_object_resonators_n4_v5')
        self.assertEqual((orz.T_FRESH_S, orz.AGE_TAU_S, orz.GAMMA_SMOOTH_S), (0.08, 0.5, 0.020))
        # absent key = Fixed
        e = registry.create(EID, ctx(), dict(detector=0, frequency_scale=220.0, decay_s=0.8, attack_ms=0.0))
        self.assertEqual(e.params['decay_law'], 0)
        # the combination rule: registry, engine (the old parameters stay), scene, snapshot
        for law in (COMMON, MODAL):
            for bad in (dict(spectrum=0), dict(fullshape=0), dict(spectrum=0, fullshape=0)):
                with self.assertRaises(ValueError):
                    registry.validate_params(EID, params(law, **bad))
                with self.assertRaises(ValueError):
                    orz.validate_params(params(law, **bad))
                with self.assertRaises(SceneError):
                    doc = scene_for(CASES[0])
                    doc['variants']['B']['engine_params'] = params(law, **bad)
                    scene_from_doc(doc)
            registry.validate_params(EID, params(law))                        # Laplace + full: fine
            self.assertFalse(orz.decay_law_supported(params(law, spectrum=0)))
        registry.validate_params(EID, params(FIXED, spectrum=0, fullshape=0))   # Fixed: any combination
        e = engine(case_cells('D2'), params(COMMON))
        run(e, 3)
        for bad in (dict(spectrum=0), dict(fullshape=0), dict(decay_law=MODAL, spectrum=0)):
            with self.assertRaises(ValueError):
                e.set_params(dict(e.params, **bad))
            self.assertEqual((e.params['decay_law'], e.params['spectrum'], e.params['fullshape']), (COMMON, 1, 1))
        with self.assertRaises(ValueError):
            engine(case_cells('D2'), params(MODAL, spectrum=0))
        # the bench locks: Fixed + Figure / 8x8 -> the law buttons are text with the reason;
        # an adaptive law -> Spectrum and full are text (set Fixed first)
        ina = orz.inactive(params(FIXED, spectrum=0))
        self.assertIn('decay_law', ina)
        self.assertIn(orz.LAW_CONDITION, ina['decay_law'])
        self.assertIn('decay_law', orz.inactive(params(FIXED, fullshape=0)))
        self.assertNotIn('decay_law', orz.inactive(params(FIXED)))
        for law in (COMMON, MODAL):
            ina = orz.inactive(params(law))
            self.assertIn('spectrum', ina)
            self.assertIn('fullshape', ina)
            self.assertNotIn('decay_law', ina)
        # display
        d = engine(case_cells('D2'), params(MODAL)).display()
        for key in ('decay_law', 'decay_law_name', 'law_supported', 'law_condition', 't_fresh_s', 'age_tau_s',
                    'gamma_smooth_s', 'n_adaptive'):
            self.assertIn(key, d)
        self.assertEqual((d['decay_law'], d['decay_law_name'], d['law_supported']), (MODAL, 'Modal age', True))
        for f in d['figures']:
            for key in ('t_lo', 't_hi', 't_applied', 'adaptive'):
                self.assertIn(key, f)
        # every Objects scene loads with the key
        want = {('od_d1', 'B'): COMMON, ('od_d2', 'A'): COMMON, ('od_d2', 'B'): MODAL, ('od_d3', 'B'): MODAL}
        for name in OBJECT_SCENES:
            sc = load_scene(os.path.join(ROOT, 'demos', name + '.json'))
            for side in ('A', 'B'):
                eid, p = sc.variants[side]
                if eid == EID:
                    self.assertEqual(p['decay_law'], want.get((name, side), FIXED), (name, side))

    def test_runner_refuses_the_combination_before_any_change_and_never_journals_it(self):
        sc = scene_from_doc(scene_for(CASES[1]))                    # D2: A Common, B Modal
        runner = DemoRunner(sc)
        runner.post('start', at=0)
        for _ in range(20):
            runner.next_block()
        before = runner.snapshot()['sides']
        n_journal = len(runner.journal)
        with self.assertRaises(ValueError):
            runner.post('set_param', side='B', name='spectrum', value=0)   # Figure under Modal age
        with self.assertRaises(ValueError):
            runner.post('set_param', side='B', name='fullshape', value=0)
        # a queued Fixed makes the same change valid (the state it WILL have counts)
        runner.post('set_param', side='B', name='decay_law', value=FIXED)
        runner.post('set_param', side='B', name='spectrum', value=0)
        # ... and a queued Figure forbids the law again
        with self.assertRaises(ValueError):
            runner.post('set_param', side='B', name='decay_law', value=MODAL)
        runner.next_block()
        self.assertEqual((runner.sides['B'].params['decay_law'], runner.sides['B'].params['spectrum']), (FIXED, 0))
        # copy_spectrum of an 8x8 window (full = 0, a shared spectrum setting) onto the
        # adaptive side is refused at post time (Spectrum itself is not a shared setting)
        runner.post('set_param', side='B', name='fullshape', value=0)
        runner.next_block()
        with self.assertRaises(ValueError):
            runner.post('copy_spectrum', src='B', dst='A')
        self.assertEqual(runner.snapshot()['sides']['A'], before['A'])
        runner.post('set_param', side='B', name='fullshape', value=1)
        # back to the valid pair; an apply-time refusal (forced past _check) leaves no journal entry
        runner.post('set_param', side='B', name='spectrum', value=1)
        runner.post('set_param', side='B', name='decay_law', value=MODAL)
        runner.next_block()
        n_journal = len(runner.journal)
        runner._pending.append((runner._seq + 1, None, 'set_param', dict(side='B', name='spectrum', value=0)))
        runner._seq += 1
        runner.next_block()
        self.assertEqual(len(runner.journal), n_journal)
        self.assertEqual(runner.sides['B'].params['spectrum'], 1)
        self.assertEqual([j for j in runner.journal if j[2] == 'set_param' and j[3].get('value') == 0
                          and j[3].get('name') == 'spectrum' and j[1] > n_journal], [])


class HistoryTests(unittest.TestCase):
    def test_age_equals_an_independent_model_through_every_kind_of_transition(self):
        # a glider through the seam (16 gen/s): births, deaths, survivors, the seam
        cells = [[30, 29], [30, 30], [30, 31], [29, 31], [28, 30]]
        e = engine(cells, params(MODAL), rate=16.0)
        model = AgeModel((ROWS, COLS))
        g = grid(cells)
        checked = 0
        for t in range(0, 40):
            if t > 0:
                prev, g = g, step(g)
                e.update_field(g, events_field(prev, g))
            for _ in range(int(round(SR / 16.0 / BLOCK))):
                u = model.render(g)
                e.render_float(GAIN)
                np.testing.assert_allclose(e.age * 0 + e.age, model.u, rtol=0, atol=1e-15)   # after the block
                checked += 1
        self.assertGreater(checked, 100)
        self.assertLess(float(e.age.max()), 0.9)                                     # nothing stays fresh
        # a still block: death and re-birth (new history), continuous life, an unchanged
        # field, a knob, an extra update_field, restore, several edits in one block
        cells = [[5, 5], [5, 6], [6, 5], [6, 6]]
        e = engine(cells, params(MODAL))
        model = AgeModel((ROWS, COLS))
        g = grid(cells)
        for _ in range(10):
            model.render(g)
            e.render_float(GAIN)
        np.testing.assert_allclose(e.age, model.u, atol=1e-15)
        e.update_field(g, None)                                        # the same field again: no event
        e.set_params(dict(e.params, decay_s=1.0))                      # a knob: no event
        st = e.export_state()
        twin = registry.create(EID, ctx(), dict(e.params))
        twin.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
        twin.restore_state(e._grid, e._exc, copy.deepcopy(st))         # restore: no event
        for x in (e, twin):
            x.render_float(GAIN)
        model.render(g)
        np.testing.assert_allclose(e.age, model.u, atol=1e-15)
        np.testing.assert_allclose(twin.age, model.u, atol=1e-15)
        g2 = g.copy()
        g2[5, 5] = 0                                                   # a death ...
        e.update_field(g2, events_field(g, g2))
        g3 = g2.copy()
        g3[5, 5] = 1                                                   # ... undone before the render: nothing
        e.update_field(g3, events_field(g2, g3))
        model.render(g3)
        e.render_float(GAIN)
        np.testing.assert_allclose(e.age, model.u, atol=1e-15)
        self.assertLess(float(e.age[5, 5]), 0.9)                       # the history of (5, 5) went on
        g4 = g3.copy()
        g4[5, 5] = 0
        e.update_field(g4, events_field(g3, g4))
        model.render(g4)
        e.render_float(GAIN)
        self.assertEqual(float(e.age[5, 5]), 0.0)                      # dead: no history
        g5 = g4.copy()
        g5[5, 5] = 1
        g5[9, 9] = 1
        e.update_field(g5, events_field(g4, g5))
        u = model.render(g5)
        e.render_float(GAIN)
        self.assertEqual(float(u[5, 5]), 1.0)                          # re-born: a NEW history (fresh)
        np.testing.assert_allclose(e.age, model.u, atol=1e-15)
        self.assertEqual(e.display()['changes'], 3)                    # 2 accepted transitions + the start
        # 6 gen/s and 2 gen/s give the same rule (only the count of blocks between steps differs)
        for rate in (6.0, 2.0):
            case = preflight_case('D3')
            e = engine(case['cells'], params(COMMON), rate=rate)
            model = AgeModel((ROWS, COLS))
            g = grid(case['cells'])
            for t in range(12):
                if t > 0:
                    prev, g = g, step(g)
                    e.update_field(g, events_field(prev, g))
                for _ in range(int(round(SR / rate / BLOCK))):
                    model.render(g)
                    e.render_float(GAIN)
            np.testing.assert_allclose(e.age, model.u, atol=1e-15)

    def test_pause_and_resume_keep_the_history_running(self):
        # the runner's Pause CA freezes the field only: the engine's age keeps decaying
        # and equals the model driven with the same (unchanged) field
        sc = scene_from_doc(scene_for(CASES[1]))
        runner = DemoRunner(sc)
        runner.post('start', at=0)                                     # the scene script queues the pauses
        model = AgeModel((ROWS, COLS))
        n = int(6.5 * SR / BLOCK)
        for _ in range(n):
            runner.next_block()
            model.render(runner.grid)
            np.testing.assert_allclose(runner.sides['B'].engine.age, model.u, atol=1e-15)
        self.assertFalse(runner.paused)
        self.assertEqual(runner.gen, 6 + 1)                            # gen 7 came after the first pause
        for j in runner.journal:
            if j[2] == 'pause':
                self.assertIn((j[0], j[3]['on']), set(D2_PAUSES))

    def test_targets_at_the_preflight_samples_for_all_three_cases_and_both_laws(self):
        pf = preflight()
        for case in pf['cases']:
            tr = {t['t_samples']: t for t in case['transitions']}
            pb = {p['t_samples']: p for p in case.get('pause_boundaries', [])}
            for law in (COMMON, MODAL):
                worst = dict(T=0.0, f=0.0, n=0)

                def check(e, out, gen, tr=tr, pb=pb, law=law, worst=worst):
                    ref = pb.get(out) or tr.get(out)
                    if ref is None:
                        return
                    banks = {tuple(map(tuple, b['cells'])): b for b in ref['banks']}
                    for f in e.figures.values():
                        if f.slot < 0:
                            continue
                        b = banks[tuple(map(tuple, f.cells.tolist()))]
                        nd = int(e.ndrive[f.slot])
                        T = orz.LN1000 / e.gam_t[f.slot, :nd]
                        want = np.asarray(b['local_T60_s']) if law == MODAL else np.full(nd, b['common_T60_s'])
                        self.assertEqual(nd, len(want))
                        worst['T'] = max(worst['T'], float(np.max(np.abs(T - want))))
                        worst['f'] = max(worst['f'], float(np.max(np.abs(e.ffreq[f.slot, :nd] - np.asarray(b['freqs'])))))
                        worst['n'] += 1
                drive_case(case['id'], law, on_block=check)
                self.assertGreaterEqual(worst['n'], len(case['transitions']))
                self.assertLess(worst['T'], 1e-9, (case['id'], law))
                self.assertLess(worst['f'], 1e-6, (case['id'], law))


class MathTests(unittest.TestCase):
    def test_participation_rows_invariance_and_the_blinker(self):
        rows, cols = ROWS, COLS
        cells = np.asarray(case_cells('D2'), np.int64)
        _f, _a, graph = orz.laplace_modes_of(cells, rows, cols, F0, orz.laplace_settings(params()), None, with_graph=True)
        P = orz.mode_participation(graph['L'], graph['idx'])
        self.assertEqual(P.shape, (2, 3))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        # the path graph: mode 1 (lambda 1) lives on the ends, mode 2 (lambda 3) 1/6, 2/3, 1/6
        node_of = {tuple(cells[graph['order'][k]].tolist()): k for k in range(3)}
        c = node_of[(10, 10)]
        np.testing.assert_allclose(P[0], [0.5 if k != c else 0.0 for k in range(3)], atol=1e-12)
        np.testing.assert_allclose(P[1], [1.0 / 6.0 if k != c else 2.0 / 3.0 for k in range(3)], atol=1e-12)
        # a degenerate group (Octagon II): P is the same for a sign flip and a rotation of the basis
        cells = np.asarray(case_cells('D1'), np.int64)
        _f, _a, graph = orz.laplace_modes_of(cells, rows, cols, F0, orz.laplace_settings(params()), None, with_graph=True)
        L = graph['L']
        lam, vecs = np.linalg.eigh(L)
        groups = orz.degenerate_groups(lam, graph['idx'])
        ranks = [len(g) for g in groups]
        self.assertGreater(max(ranks), 1)
        P = orz.mode_participation(L, graph['idx'])
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        rng = np.random.default_rng(3)
        for t, group in enumerate(groups):
            d = len(group)
            V = vecs[:, group]
            Q = np.linalg.qr(rng.standard_normal((d, d)))[0]            # a rotation inside the group
            W = V @ Q
            W[:, 0] *= -1.0                                             # and a sign flip
            np.testing.assert_allclose((W * W).sum(axis=1) / d, P[t], atol=1e-12)
        # an empty selection
        self.assertEqual(orz.mode_participation(L, []).shape, (0, len(cells)))

    def test_translation_rotation_equal_ages_one_mode_and_two_histories(self):
        rows, cols = ROWS, COLS
        base = np.asarray(case_cells('D1'), np.int64)
        rng = np.random.default_rng(7)
        u_base = rng.uniform(0.05, 1.0, size=len(base))                 # an artificial history per cell

        def targets(cells, u_cells):
            _f, _a, graph = orz.laplace_modes_of(cells, rows, cols, F0, orz.laplace_settings(params()), None,
                                                 with_graph=True)
            P = orz.mode_participation(graph['L'], graph['idx'])
            q, T, gamma = orz.age_targets(P, u_cells[graph['order']], STABLE_DECAY_S)
            return q, T, gamma

        q0, T0, g0 = targets(base, u_base)
        # translation across the seam: the same cells shifted, the history shifted with them
        for dr, dc in ((21, 25), (-9, 17), (3, -12)):
            moved = np.stack([np.mod(base[:, 0] + dr, rows), np.mod(base[:, 1] + dc, cols)], axis=1)
            order = np.lexsort((moved[:, 1], moved[:, 0]))
            q1, T1, g1 = targets(moved[order], u_base[order])
            np.testing.assert_allclose(q1, q0, atol=1e-12)
            np.testing.assert_allclose(g1, g0, atol=1e-9)
        # a 90 degree rotation about the field's centre (with the history)
        rot = np.stack([base[:, 1], np.mod(-base[:, 0], rows)], axis=1)
        order = np.lexsort((rot[:, 1], rot[:, 0]))
        q1, T1, g1 = targets(rot[order], u_base[order])
        np.testing.assert_allclose(sorted(q1), sorted(q0), atol=1e-12)   # a rotation may reorder equal modes
        # equal ages: Common == Modal targets (every mode the same T), one mode: the same
        for u in (0.0, 0.37, 1.0):
            q, T, gamma = targets(base, np.full(len(base), u))
            np.testing.assert_allclose(q, u, atol=1e-12)
            np.testing.assert_allclose(T, STABLE_DECAY_S - (STABLE_DECAY_S - 0.08) * u, atol=1e-12)
            self.assertAlmostEqual(float(gamma.mean()), float(gamma[0]), places=9)
        q, T, gamma = targets(np.asarray([[4, 4], [4, 5]], np.int64), np.asarray([1.0, 0.2]))   # a 2-cell figure: 1 mode
        self.assertEqual(len(gamma), 1)
        np.testing.assert_allclose(q, [0.6], atol=1e-12)
        # two identical geometries with different artificial histories: losses differ,
        # frequencies / weights / the next packet do not
        cells = case_cells('D1')
        gens = gens_of(cells, 3)
        a, b = engine(cells, params(MODAL)), engine(cells, params(MODAL))
        run(a, 3)
        run(b, 3)
        b.age[b.age > 0] = 0.15                                        # an older history on b
        for k in (1, 2):
            for x in (a, b):
                x.update_field(gens[k], events_field(gens[k - 1], gens[k]))
                x.render_float(GAIN)
            sa, sb = a.figures[1].slot, b.figures[1].slot
            nd = int(a.ndrive[sa])
            np.testing.assert_array_equal(a.ffreq[sa], b.ffreq[sb])
            np.testing.assert_array_equal(a.wtgt[sa], b.wtgt[sb])
            self.assertEqual((float(a.last_e[sa]), float(a.last_a[sa])), (float(b.last_e[sb]), float(b.last_a[sb])))
            np.testing.assert_array_equal(a.last_b[sa], b.last_b[sb])
            self.assertGreater(float(np.abs(a.gam_t[sa, :nd] - b.gam_t[sb, :nd]).max()), 1.0)

    def test_the_order_q_then_t_then_gamma_and_the_bounds(self):
        P = np.asarray([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]])
        u = np.asarray([1.0, 0.0, 0.5])
        q, T, gamma = orz.age_targets(P, u, 1.39)
        np.testing.assert_allclose(q, [0.5, 0.45], atol=1e-15)
        np.testing.assert_allclose(T, 1.39 - 1.31 * q, atol=1e-15)
        np.testing.assert_allclose(gamma, math.log(1000.0) / T, atol=1e-12)
        common = float(gamma.mean())
        self.assertNotAlmostEqual(common, math.log(1000.0) / float(T.mean()), places=3)   # the mean RATE, not time
        # the two-mode figure of D2 at the first pause: the REQ's numbers
        cells = np.asarray(case_cells('D2'), np.int64)
        _f, _a, graph = orz.laplace_modes_of(cells, ROWS, COLS, F0, orz.laplace_settings(params()), None, with_graph=True)
        Pm = orz.mode_participation(graph['L'], graph['idx'])
        pb = preflight_case('D2')['pause_boundaries'][0]['banks'][0]
        u = np.zeros(3)
        node_of = {tuple(cells[graph['order'][k]].tolist()): k for k in range(3)}
        k = math.exp(-BLOCK / (SR * 0.5))
        u[node_of[(10, 9)]] = u[node_of[(10, 11)]] = k
        u[node_of[(10, 10)]] = k ** 377
        q, T, gamma = orz.age_targets(Pm, u, 1.39)
        np.testing.assert_allclose(T, pb['local_T60_s'], atol=1e-12)
        self.assertAlmostEqual(math.log(1000.0) / float(gamma.mean()), pb['common_T60_s'], places=12)
        # clip within the tolerance, an error beyond it
        q, T, g = orz.age_targets(np.asarray([[1.0 + 5e-10, 0.0]]), np.asarray([1.0, 0.0]), 1.39)
        self.assertEqual(float(q[0]), 1.0)
        with self.assertRaises(RuntimeError):
            orz.age_targets(np.asarray([[1.0, 0.0]]), np.asarray([1.0 + 2e-6, 0.0]), 1.39)
        with self.assertRaises(RuntimeError):
            orz.age_targets(np.asarray([[1.0, 0.0]]), np.asarray([-2e-6, 0.0]), 1.39)
        self.assertAlmostEqual(orz.gamma_of_r(orz.decay_r(1.39)), math.log(1000.0) / 1.39, places=9)


def scalar_reference(e, n_blocks, targets=None):
    """The sample path of the module docstring for the slots in use of engine e (a
    scalar Python loop over a copy of its state; the gain ramp is finished, no
    events): `targets` {block: (slot, gamma array)} sets gam_t before that block.
    Returns the stereo output (n, 2)."""
    S, M = orz.N_SLOTS, orz.N_BANK
    st = {name: np.array(getattr(e, name), copy=True) for name in
          ('role', 'ndrive', 'npulse', 'nlive', 'zre', 'zim', 'cth', 'sth', 'wcur', 'zf', 'zs', 'zu', 'zfm', 'zsm',
           'zum', 'pan', 'slaw', 'gam_a', 'gam_t', 'hp')}
    qf, qs, strength = e.consts[en.C_QF], e.consts[en.C_QS], e.consts[en.C_STRENGTH]
    r = float(e.rr[orz.R_CUR])
    g = float(e.gg[orz.R_CUR])
    q = float(e.qq[orz.R_CUR])
    ksm = e.ksm
    sr = e.sr
    out = np.zeros((n_blocks * BLOCK, 2))
    k = 0
    for blk in range(n_blocks):
        if targets and blk in targets:
            s, gt = targets[blk]
            st['gam_t'][s, :len(gt)] = gt
        for _t in range(BLOCK):
            L = R = 0.0
            for s in range(S):
                if st['role'][s] == orz.ROLE_FREE:
                    continue
                p = strength * ((1.0 - qf) * st['zf'][s] - (1.0 - qs) * st['zs'][s]) / (qs - qf)
                st['zf'][s] *= qf
                st['zs'][s] *= qs
                u = q * st['zu'][s] + (1.0 - q) * p
                st['zu'][s] = u
                nd = st['ndrive'][s] if st['role'][s] == orz.ROLE_ACTIVE else st['npulse'][s]
                acc = 0.0
                for j in range(int(st['nlive'][s])):
                    if st['slaw'][s] == 1:
                        ga = ksm * st['gam_a'][s, j] + (1.0 - ksm) * st['gam_t'][s, j]
                        st['gam_a'][s, j] = ga
                        rj = math.exp(-ga / sr)
                    else:
                        rj = r
                    re, im = st['zre'][s, j], st['zim'][s, j]
                    c, sn = st['cth'][s, j], st['sth'][s, j]
                    if j < nd:
                        pm = strength * ((1.0 - qf) * st['zfm'][s, j] - (1.0 - qs) * st['zsm'][s, j]) / (qs - qf)
                        st['zfm'][s, j] *= qf
                        st['zsm'][s, j] *= qs
                        um = q * st['zum'][s, j] + (1.0 - q) * pm
                        st['zum'][s, j] = um
                        nre = rj * (c * re - sn * im) + (u + um)
                    else:
                        nre = rj * (c * re - sn * im)
                    nim = rj * (sn * re + c * im)
                    st['zre'][s, j], st['zim'][s, j] = nre, nim
                    acc += st['wcur'][s, j] * nre
                L += st['pan'][s, orz.P_LCUR] * acc
                R += st['pan'][s, orz.P_RCUR] * acc
            for ch, v in ((0, L), (1, R)):
                y = e.hp_h * ((st['hp'][ch, orz.H_PREV] + v) - st['hp'][ch, orz.H_MIX])
                st['hp'][ch, orz.H_PREV] = y
                st['hp'][ch, orz.H_MIX] = v
                out[k, ch] = y * orz.OUT_SCALE * g
            k += 1
    return out, st


class KernelTests(unittest.TestCase):
    def test_adaptive_kernel_equals_the_scalar_reference_through_a_target_change(self):
        cells = case_cells('D1')
        e = engine(cells, params(MODAL))
        run(e, 8)                                                      # the start packet, the gain ramp done
        s = e.figures[1].slot
        self.assertEqual(int(e.slaw[s]), 1)
        self.assertEqual(int(e.ints[orz.I_G_LEFT]), 0)
        nd = int(e.ndrive[s])
        new_t = e.gam_t[s, :nd] * np.asarray([0.5, 2.0, 1.3])[:nd]
        ref, st_ref = scalar_reference(e, 6, targets={3: (s, new_t)})
        got = []
        for blk in range(6):
            if blk == 3:
                e.gam_t[s, :nd] = new_t
            got.append(e.raw_block(events=False))
        got = np.concatenate(got)
        self.assertLess(float(np.abs(got - ref).max()), 1e-12)
        np.testing.assert_allclose(e.gam_a[s, :nd], st_ref['gam_a'][s, :nd], rtol=0, atol=1e-9)
        self.assertGreater(float(np.abs(got).max()), 1e-4)
        # the applied gamma moved towards the new targets with the 20 ms pole
        for j in range(nd):
            self.assertNotAlmostEqual(float(e.gam_a[s, j]), float(new_t[j]), places=6)
            self.assertLess(abs(float(e.gam_a[s, j]) - float(new_t[j])), abs(float(e.gam_a[s, j]) - float(e.gam_t[s, j])) + 1e-9)

    def test_t60_at_frozen_gamma_and_no_energy_from_a_loss_change_at_zero_input(self):
        # one free mode: |z| after T seconds at frozen gamma = -60 dB
        cells = [[4, 4], [4, 5]]                                       # 2 cells: exactly one mode
        e = engine(cells, params(MODAL))
        run(e, 60)
        s = e.figures[1].slot
        self.assertEqual(int(e.ndrive[s]), 1)
        e._zero_pulse(s)
        e.zfm[s] = 0.0
        e.zsm[s] = 0.0
        e.zum[s] = 0.0
        T = 0.3
        gamma = orz.LN1000 / T
        e.gam_a[s, 0] = gamma
        e.gam_t[s, 0] = gamma
        e.zre[s, 0], e.zim[s, 0] = 1.0, 0.0
        n = int(round(T * SR / BLOCK))
        for _ in range(n):
            e.raw_block(events=False)
        mag = math.hypot(float(e.zre[s, 0]), float(e.zim[s, 0]))
        expected_db = -20.0 * math.log10(math.e) * gamma * (n * BLOCK) / SR
        self.assertAlmostEqual(20.0 * math.log10(mag), expected_db, places=6)
        self.assertAlmostEqual(expected_db, -60.0, delta=1.0)          # n blocks ~ T (block granularity)
        # zero input: whatever the targets do, |z| never grows
        e.zre[s, 0], e.zim[s, 0] = 0.3, -0.2
        last = math.hypot(0.3, -0.2)
        rng = np.random.default_rng(1)
        for blk in range(40):
            e.gam_t[s, 0] = orz.LN1000 / float(rng.uniform(0.08, 1.39))
            e.raw_block(events=False)
            cur = math.hypot(float(e.zre[s, 0]), float(e.zim[s, 0]))
            self.assertLessEqual(cur, last * (1.0 + 1e-12))
            last = cur
        self.assertGreater(last, 0.0)
        self.assertEqual(float(np.abs(e.zf).max()), 0.0)

    def test_d2_modal_envelopes_of_the_real_engine_in_both_pauses(self):
        # the REQ's acceptance for D2: the upper component keeps relative to the lower
        # one longer in Modal than in Common -- measured on the real engine's resonator
        # states in BOTH pauses (a side gain never changes this ratio)
        curves = {}
        for law in (COMMON, MODAL):
            rows = []

            def probe(e, out, gen, rows=rows):
                f = e.figures.get(1)
                if f is None or f.slot < 0:
                    return
                mag = np.hypot(e.zre[f.slot, :2], e.zim[f.slot, :2])
                rows.append((out, gen, mag[0], mag[1]))
            drive_case('D2', law, on_block=probe)
            curves[law] = rows
        for p_on, p_off in ((D2_PAUSES[0][0], D2_PAUSES[1][0]), (D2_PAUSES[2][0], D2_PAUSES[3][0])):
            ratio = {}
            for law, rows in curves.items():
                inside = [(out, lo, hi) for out, _g, lo, hi in rows if p_on <= out < p_off]
                self.assertGreater(len(inside), 200)
                start = inside[0]
                at = {int(round(0.10 * SR / BLOCK)): None, int(round(0.25 * SR / BLOCK)): None,
                      int(round(0.50 * SR / BLOCK)): None}
                for k in at:
                    out, lo, hi = inside[k]
                    at[k] = 20.0 * math.log10(hi / lo) - 20.0 * math.log10(start[2] / start[1])
                ratio[law] = at
            for k in ratio[MODAL]:
                self.assertGreater(ratio[MODAL][k], ratio[COMMON][k] + 10.0)   # tens of dB apart
                self.assertLess(abs(ratio[COMMON][k]), 1.0)                    # Common: together
            self.assertGreater(ratio[MODAL][max(ratio[MODAL])], 30.0)
        # the same in both pauses (repeat after the resume)
        first = [r for r in curves[MODAL] if D2_PAUSES[0][0] <= r[0] < D2_PAUSES[1][0]]
        second = [r for r in curves[MODAL] if D2_PAUSES[2][0] <= r[0] < D2_PAUSES[3][0]]
        d1 = 20 * math.log10(first[60][3] / first[60][2]) - 20 * math.log10(first[0][3] / first[0][2])
        d2 = 20 * math.log10(second[60][3] / second[60][2]) - 20 * math.log10(second[0][3] / second[0][2])
        self.assertAlmostEqual(d1, d2, delta=1.0)


class IsolationTests(unittest.TestCase):
    def test_the_sides_of_d1_d2_d3_share_everything_but_the_losses(self):
        for case in CASES:
            pfc = preflight_case(case['case'])
            a = engine(pfc['cells'], params(case['law']['A'], case['decay']['A']), rate=pfc['rate_hz'])
            b = engine(pfc['cells'], params(case['law']['B'], case['decay']['B']), rate=pfc['rate_hz'])
            g = grid(pfc['cells'])
            per_gen = int(round(SR / pfc['rate_hz'] / BLOCK))
            n_gen = 24 if case['case'] == 'D3' else 12
            diff_losses = 0
            for t in range(n_gen + 1):
                if t > 0:
                    prev, g = g, step(g)
                    for x in (a, b):
                        x.update_field(g, events_field(prev, g))
                for _ in range(per_gen):
                    for x in (a, b):
                        x.render_float(GAIN)
                self.assertEqual(sorted(a.figures), sorted(b.figures))
                for fid, fa in a.figures.items():
                    fb = b.figures[fid]
                    self.assertEqual(fa.slot, fb.slot)
                    np.testing.assert_array_equal(fa.cells, fb.cells)
                    if fa.slot < 0:
                        continue
                    s = fa.slot
                    self.assertEqual(int(a.ndrive[s]), int(b.ndrive[s]))
                    np.testing.assert_array_equal(a.ffreq[s], b.ffreq[s])
                    np.testing.assert_array_equal(a.wtgt[s], b.wtgt[s])
                    self.assertEqual(float(a.last_e[s]), float(b.last_e[s]))
                    self.assertEqual(float(a.last_a[s]), float(b.last_a[s]))
                    np.testing.assert_array_equal(a.last_b[s], b.last_b[s])
                    np.testing.assert_array_equal(a.qq, b.qq)
                    nd = int(a.ndrive[s])
                    ra = (np.exp(-a.gam_a[s, :nd] / SR) if a.slaw[s] else np.full(nd, a.rr[orz.R_CUR]))
                    rb = (np.exp(-b.gam_a[s, :nd] / SR) if b.slaw[s] else np.full(nd, b.rr[orz.R_CUR]))
                    if float(np.abs(ra - rb).max()) > 0.0:
                        diff_losses += 1
                self.assertEqual(a.display()['changes'], b.display()['changes'])
                for x in (a, b):                                    # an adaptive slot never runs undamped
                    for s in np.nonzero(x.slaw == 1)[0]:
                        nl = int(x.nlive[s])
                        self.assertTrue(bool(np.all(x.gam_a[s, :nl] > 0.0)) and bool(np.all(x.gam_t[s, :nl] > 0.0)))
                # (tails die out at different times under different losses: their count may differ)
            self.assertGreater(diff_losses, 5)


class StateTests(unittest.TestCase):
    def test_law_and_decay_changes_make_no_packet_no_reset_no_revival(self):
        cells = case_cells('D1')
        gens = gens_of(cells, 2)
        e = engine(cells, params(FIXED))
        run(e, 3)
        e.update_field(gens[1], events_field(gens[0], gens[1]))
        run(e, 4)
        s = e.figures[1].slot
        changes = e.display()['changes']
        last = (float(e.last_e[s]), float(e.last_a[s]))
        prev_out = None
        for law, decay in ((COMMON, 1.39), (MODAL, 1.39), (MODAL, 0.6), (COMMON, 0.6), (FIXED, 0.6), (MODAL, 1.2), (FIXED, 1.39)):
            z_before = np.hypot(e.zre[s], e.zim[s]).copy()
            zf_before = float(np.abs(e.zf).max())
            e.set_params(dict(e.params, decay_law=law, decay_s=decay))
            y = e.render_float(GAIN)[0]
            self.assertEqual(e.display()['changes'], changes)                     # no event
            self.assertEqual((float(e.last_e[s]), float(e.last_a[s])), last)     # no packet
            self.assertLessEqual(float(np.abs(e.zf).max()), zf_before)              # the pulse states only decay
            z_after = np.hypot(e.zre[s], e.zim[s])
            self.assertTrue(np.all(z_after <= z_before * 1.0 + 1e-12))          # no energy added
            self.assertTrue(np.all(z_after[z_before > 0] > 0.0))                 # no reset
            if prev_out is not None:
                self.assertLess(abs(float(y[0, 0]) - float(prev_out[-1, 0])), 0.05)   # continuous
            prev_out = y
            self.assertTrue(np.all(np.isfinite(y)))
        # a decayed tail is never revived: once every slot is free, a law change leaves the
        # output equal to a twin's (only the DC filter's own residual, never a bank again)
        empty = np.zeros((ROWS, COLS), np.uint8)
        e2, twin = engine(cells, params(FIXED)), engine(cells, params(FIXED))
        for x in (e2, twin):
            run(x, 3)
            x.update_field(empty, events_field(gens[0], empty))
            run(x, int(4.0 * SR / BLOCK))                                        # the tail dies out
            self.assertEqual(int(np.count_nonzero(x.role != orz.ROLE_FREE)), 0)
        for law in (COMMON, MODAL, FIXED):
            e2.set_params(dict(e2.params, decay_law=law))
            y2, yt = run(e2, 5), run(twin, 5)
            np.testing.assert_array_equal(y2, yt)
            self.assertLess(float(np.abs(y2).max()), 1e-30)
            self.assertEqual(int(np.count_nonzero(e2.role != orz.ROLE_FREE)), 0)

    def test_vanished_banks_hold_their_gamma_and_returning_figures_start_fresh(self):
        cells = case_cells('D1')
        gens = gens_of(cells, 2)
        e = engine(cells, params(MODAL))
        run(e, 30)
        s = e.figures[1].slot
        nd = int(e.ndrive[s])
        gt = e.gam_t[s, :nd].copy()
        ga = e.gam_a[s, :nd].copy()
        empty = np.zeros((ROWS, COLS), np.uint8)
        e.update_field(empty, events_field(gens[0], empty))
        e.render_float(GAIN)
        tails = np.nonzero(e.role == orz.ROLE_TAIL)[0]
        self.assertEqual(len(tails), 1)
        t = int(tails[0])
        self.assertEqual(int(e.slaw[t]), 1)
        np.testing.assert_array_equal(e.gam_t[t, :nd], gt)                       # the target travelled with it
        self.assertLess(float(np.abs(e.gam_a[t, :nd] - ga).max()), 0.05 * float(ga.max()))   # one block of smoothing
        for _ in range(20):
            e.set_params(dict(e.params, decay_s=0.3 if e.params['decay_s'] > 1.0 else 1.39))
            e.render_float(GAIN)
            np.testing.assert_array_equal(e.gam_t[t, :nd], gt)                   # held: no geometry left
        np.testing.assert_allclose(e.gam_a[t, :nd], gt, rtol=1e-3)               # the smoothing (20 ms) finished
        # the figure comes back: a NEW bank, applied == target before its first sample
        e.update_field(gens[0], events_field(empty, gens[0]))
        e.render_float(GAIN)
        f = e.figures[max(e.figures)]
        self.assertGreater(f.id, 1)
        s2 = f.slot
        self.assertEqual(int(e.slaw[s2]), 1)
        self.assertNotEqual(s2, t)
        self.assertTrue(np.all(e.gam_t[s2, :nd] > 0.0))
        # after one block the applied value has moved only by the smoother from its target (started there)
        self.assertLess(float(np.abs(e.gam_a[s2, :nd] - e.gam_t[s2, :nd]).max()), 1e-9 * float(e.gam_t[s2, :nd].max()) + 1e-12)
        # fresh: q = 1 on every mode -> T = 0.08 s
        np.testing.assert_allclose(orz.LN1000 / e.gam_t[s2, :nd], orz.T_FRESH_S, atol=1e-12)

    def test_a_fixed_era_tail_holds_the_decay_of_the_switch(self):
        cells = case_cells('D1')
        gens = gens_of(cells, 2)
        empty = np.zeros((ROWS, COLS), np.uint8)
        e = engine(cells, params(FIXED, decay=1.39))
        run(e, 10)
        e.update_field(empty, events_field(gens[0], empty))
        run(e, 1)
        t = int(np.nonzero(e.role == orz.ROLE_TAIL)[0][0])
        self.assertEqual(int(e.slaw[t]), 0)                                      # Fixed: the global r
        e.set_params(dict(e.params, decay_law=MODAL))
        g = orz.gamma_of_r(orz.decay_r(1.39))
        self.assertEqual(int(e.slaw[t]), 1)
        nl = int(e.nlive[t])
        np.testing.assert_allclose(e.gam_t[t, :nl], g)
        e.set_params(dict(e.params, decay_s=0.3))                                # a later Decay move ...
        run(e, 30)
        np.testing.assert_allclose(e.gam_t[t, :nl], g)                           # ... does not reach the tail
        np.testing.assert_allclose(e.gam_a[t, :nl], g)

    def test_switch_back_to_fixed_settles_onto_the_global_r(self):
        cells = case_cells('D1')
        e = engine(cells, params(MODAL))
        run(e, 30)
        s = e.figures[1].slot
        e.set_params(dict(e.params, decay_law=FIXED))
        self.assertEqual(int(e.slaw[s]), 1)
        g = orz.gamma_of_r(e.rr[orz.R_TGT], SR)
        np.testing.assert_allclose(e.gam_t[s, :int(e.nlive[s])], g)
        settled_at = None
        for blk in range(120):
            e.render_float(GAIN)
            if int(e.slaw[s]) == 0:
                settled_at = blk
                break
        self.assertIsNotNone(settled_at)
        self.assertGreater(settled_at, 5)                                        # smoothly, not at once
        self.assertLess(settled_at, 120)
        self.assertEqual(int(e.slaw[s]), 0)
        self.assertEqual(e.display()['n_adaptive'], 0)
        # the Fixed path from here: identical to a twin that never left Fixed, up to the history
        # of the losses (not bit-exact, the REQ asks a smooth transition only)
        self.assertTrue(np.all(np.isfinite(run(e, 10))))

    def test_pinned_records_replay_bit_for_bit_and_a_v5_snapshot_imports_as_fixed(self):
        from casynth_lab.catalog import Catalog
        for name, n_records in PINNED:
            root = os.path.join(ROOT, 'lab_catalog', name)
            if not os.path.isdir(root):
                continue
            cat = Catalog(root)
            recs = [rec for rec, err in cat.list() if err is None]
            self.assertEqual(len(recs), n_records)
            for rec in recs:
                self.assertEqual(rendered_like(rec), dict(A=True, B=True, monitor=True), rec.title)
        # a v5 snapshot (no new keys / arrays) of a Fixed run continues with the same sound
        cells = case_cells('D3')
        gens = gens_of(cells, 6)
        e = engine(cells, params(FIXED), rate=6.0)
        run(e, 3)
        for k in (1, 2):
            e.update_field(gens[k], events_field(gens[k - 1], gens[k]))
            run(e, 3)
        st5 = e.export_state()
        st5['version'] = 5
        st5['model_version'] = orz.COMPATIBLE_STATES[5]
        for key in ('slaw', 'gam_a', 'gam_t', 'age', 'fig_part', 'fig_part_modes', 'fig_part_offsets', 'law_key'):
            st5.pop(key)
        st5['params'] = {k: v for k, v in st5['params'].items() if k != 'decay_law'}
        old = registry.create(EID, ctx(6.0), dict(e.params))
        old.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
        old.restore_state(e._grid, e._exc, copy.deepcopy(st5))
        self.assertEqual(old.params['decay_law'], FIXED)
        np.testing.assert_array_equal(old.age, (e.G_prev != 0).astype(float))   # u = 1 on the live cells
        self.assertEqual(int(np.count_nonzero(old.slaw)), 0)
        for k in (3, 4, 5):
            for x in (e, old):
                x.update_field(gens[k], events_field(gens[k - 1], gens[k]))
            for _ in range(20):
                np.testing.assert_array_equal(e.render_float(GAIN)[0], old.render_float(GAIN)[0])
        # refusals
        st = e.export_state()
        good = registry.create(EID, ctx(6.0), dict(e.params))
        good.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
        good.restore_state(e._grid, e._exc, copy.deepcopy(st))
        bad_age = copy.deepcopy(st)
        bad_age['age'][0, 0] = 0.5                                              # a dead cell with a history
        bad_law = copy.deepcopy(st)
        bad_law['params'] = dict(st['params'], decay_law=MODAL, spectrum=0)
        bad_slaw = copy.deepcopy(st)
        bad_slaw['slaw'][orz.N_SLOTS - 1] = 1                                   # a free slot with a law
        bad_neg = copy.deepcopy(st)
        bad_neg['gam_t'][0, 0] = -1.0
        for bad in (dict(copy.deepcopy(st), version=5), bad_age, bad_law, bad_slaw, bad_neg):
            fresh = registry.create(EID, ctx(6.0), dict(e.params))
            fresh.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
            with self.assertRaises(ValueError):
                fresh.restore_state(e._grid, e._exc, bad)

    def test_snapshot_v6_continues_exactly_mid_pause_mid_transition_and_after_split_merge(self):
        # D2: from inside the first pause (the runner carries the queued pause off)
        sc = scene_from_doc(scene_for(CASES[1]))
        runner = DemoRunner(sc)
        runner.post('start', at=0)
        while runner.out_samples < D2_PAUSES[0][0] + 100 * BLOCK:
            runner.next_block()
        self.assertTrue(runner.paused)
        st = runner.export_state()
        self.assertEqual(st['sides']['B']['engine']['version'], 6)
        self.assertGreater(len(st['pending']), 0)
        twin = DemoRunner.from_state(st)
        for _ in range(int(3.0 * SR / BLOCK)):
            a, b = runner.next_block(), twin.next_block()
            for o in ('A', 'B', 'monitor'):
                np.testing.assert_array_equal(a.get(o), b.get(o))
        self.assertFalse(runner.paused)
        self.assertEqual(runner.snapshot()['display']['B']['n_adaptive'], twin.snapshot()['display']['B']['n_adaptive'])
        # D1: a law switch on side A, the snapshot taken during the gamma transition
        sc = scene_from_doc(scene_for(CASES[0]))
        runner = DemoRunner(sc)
        runner.post('start', at=0)
        for _ in range(int(2.0 * SR / BLOCK)):
            runner.next_block()
        runner.post('set_param', side='A', name='decay_law', value=MODAL)
        for _ in range(3):
            runner.next_block()
        st = runner.export_state()
        ea = runner.sides['A'].engine
        s = ea.figures[1].slot
        self.assertEqual(int(ea.slaw[s]), 1)
        self.assertGreater(float(np.abs(ea.gam_a[s, :3] - ea.gam_t[s, :3]).max()), 1.0)   # mid transition
        twin = DemoRunner.from_state(st)
        for _ in range(200):
            a, b = runner.next_block(), twin.next_block()
            for o in ('A', 'B', 'monitor'):
                np.testing.assert_array_equal(a.get(o), b.get(o))
        # D3: after split / merge (tails carrying their gamma)
        sc = scene_from_doc(scene_for(CASES[2]))
        runner = DemoRunner(sc)
        runner.post('start', at=0)
        for _ in range(int(6.0 * SR / BLOCK)):
            runner.next_block()
        eb = runner.sides['B'].engine
        self.assertGreater(int(np.count_nonzero((eb.role == orz.ROLE_TAIL) & (eb.slaw == 1))), 0)
        st = runner.export_state()
        twin = DemoRunner.from_state(st)
        for _ in range(int(2.0 * SR / BLOCK)):
            a, b = runner.next_block(), twin.next_block()
            for o in ('A', 'B', 'monitor'):
                np.testing.assert_array_equal(a.get(o), b.get(o))
        self.assertGreater(runner.gen, 36)


class ScriptTests(unittest.TestCase):
    """The scene script (the D2 pauses): queued at every start from the beginning,
    journaled with the mark, skipped by the replay, never repeated by Continue."""

    def test_restart_stop_and_live_start_requeue_the_script_and_bad_scripts_are_refused(self):
        sc = scene_from_doc(scene_for(CASES[1]))
        self.assertEqual(sc.script, [(at, 'pause', dict(on=on)) for at, on in D2_PAUSES])
        runner = DemoRunner(sc)
        runner.post('start', at=0)
        for _ in range(40):
            runner.next_block()
        self.assertEqual(len(runner._pending), 4)
        runner.post('reset')                                           # Restart: the scene begins again
        runner.next_block()
        t0 = runner.journal[-1][0] if runner.journal[-1][2] == 'reset' else None
        resets = [j for j in runner.journal if j[2] == 'reset']
        t0 = resets[-1][0]
        self.assertEqual(sorted(a for _q, a, _k, _x in runner._pending), [t0 + at for at, _on in D2_PAUSES])
        runner.post('stop')                                            # Stop: nothing stays queued
        runner.next_block()
        self.assertEqual(runner._pending, [])
        runner.post('pause', on=False)                                 # the live start of a stopped scene
        runner.next_block()
        starts = [j for j in runner.journal if j[2] == 'pause' and j[3].get('on') is False and not j[3].get('script')]
        t1 = starts[-1][0]
        self.assertEqual(sorted(a for _q, a, _k, _x in runner._pending), [t1 + at for at, _on in D2_PAUSES])
        while runner.out_samples <= t1 + D2_PAUSES[0][0]:
            runner.next_block()
        self.assertTrue(runner.paused)
        self.assertEqual(runner.t_samples, D2_PAUSES[0][0] + BLOCK)   # the pause stands on the scene clock
        # a start while running queues nothing more; scenes without a script queue nothing
        n = len(runner._pending)
        runner.post('start')
        runner.next_block()
        self.assertEqual(len(runner._pending), n)
        r1 = DemoRunner(scene_from_doc(scene_for(CASES[0])))
        r1.post('start', at=0)
        r1.next_block()
        self.assertEqual(r1._pending, [])
        # refused scripts
        for bad in ([dict(at=0, kind='pause', args=dict(on=True))], [dict(at=-352, kind='pause', args=dict(on=True))],
                    [dict(at=1.5, kind='pause', args=dict(on=True))], [dict(at=352, kind='set_cell', args=dict(on=True))],
                    [dict(at=352, kind='pause', args=dict(on=1))], [dict(at=352, kind='pause', args=dict(on=True, x=1))],
                    [dict(at=352, kind='pause', args=dict(on=True), extra=1)], dict(at=352)):
            doc = scene_for(CASES[1])
            doc['script'] = bad
            with self.assertRaises(SceneError):
                scene_from_doc(doc)

    def test_a_record_with_the_script_replays_exactly_and_continue_does_not_repeat_it(self):
        from casynth_lab.catalog import Catalog
        from demos.build_n1_demos import record_offline
        import shutil
        root = os.path.join(ART, 'catalog_script')
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(os.path.join(root, '.tmp'))
        cat = Catalog(root)
        doc = scene_for(CASES[1])
        rid, snap = record_offline(cat, doc, 12.0, [], 'script', 'script test')
        rec = cat.load(rid)
        pauses = [(j['out_sample'], j['args']['on'], j['args'].get('script')) for j in rec.meta['journal']
                  if j['kind'] == 'pause']
        self.assertEqual(pauses, [(at, on, True) for at, on in D2_PAUSES])   # the commands are in the journal
        res = cat.replay(rid, yield_cpu=False)
        self.assertEqual(res.status, 'match', res.reason)
        # Continue from the end: nothing scripted is queued again, no pause comes back
        from casynth_lab.snapshot import load_state
        st = load_state(rec.dir, 'state_end')
        twin = DemoRunner.from_state(st)
        self.assertEqual(twin._pending, [])
        self.assertTrue(twin.running)
        self.assertFalse(twin.paused)
        gen0 = twin.gen
        for _ in range(int(4.0 * SR / BLOCK)):
            twin.next_block()
        self.assertFalse(twin.paused)
        self.assertEqual([j for j in twin.journal if j[2] == 'pause'], [])
        self.assertGreater(twin.gen, gen0 + 6)


class ControlTests(unittest.TestCase):
    def test_d1_control_constant_is_the_mean_rate_of_the_common_side_over_5_to_20_s(self):
        gt, ga = [], []

        def probe(e, out, gen):
            if 5.0 * SR <= out < 20.0 * SR:
                f = e.figures[1]
                nd = int(e.ndrive[f.slot])
                gt.append(float(e.gam_t[f.slot, :nd].mean()))
                ga.append(float(e.gam_a[f.slot, :nd].mean()))
        drive_case('D1', COMMON, on_block=probe)
        control = orz.LN1000 / CASES[0]['decay']['A']
        self.assertEqual(CASES[0]['decay']['A'], preflight_case('D1')['fixed_T60_s'])
        self.assertLess(abs(float(np.mean(gt)) - control) / control, 0.01)
        self.assertLess(abs(float(np.mean(ga)) - control) / control, 0.01)
        # the mature cycle: Common T of the REQ (0.651 / 0.415 / 1.156 / 0.853 / 0.735 s) at its transitions
        T_at = {}

        def probe2(e, out, gen):
            if gen >= 10 and gen not in T_at and out == int(math.ceil(gen * (SR / 2.0) / BLOCK)) * BLOCK:
                f = e.figures[1]
                T_at[gen] = orz.LN1000 / float(e.gam_t[f.slot, 0])
        drive_case('D1', COMMON, on_block=probe2)
        cyc = sorted(T_at[g] for g in range(20, 25))
        for got, want in zip(cyc, sorted([0.651, 0.415, 1.156, 0.853, 0.735])):
            self.assertAlmostEqual(got, want, delta=0.01)                       # the REQ's rounded numbers


class SceneTests(unittest.TestCase):
    def test_scenes_conditions_journal_and_the_level_windows(self):
        for case in CASES:
            doc = scene_for(case)
            sc = scene_from_doc(doc)
            pfc = preflight_case(case['case'])
            self.assertEqual(doc['cells'], case_cells(case['case']))
            self.assertEqual(doc['rate_hz'], pfc['rate_hz'])
            self.assertEqual(case['seconds'], pfc['seconds'])
            self.assertTrue(case['hypothesis'].startswith('Гипотеза - '))
            (ea, pa), (eb, pb) = sc.variants['A'], sc.variants['B']
            self.assertEqual((ea, eb), (EID, EID))
            diff = {k for k in pa if pa[k] != pb[k]}
            self.assertEqual(diff, {'decay_law', 'decay_s'} if case['id'] == 'od_d1' else {'decay_law'})
            self.assertEqual((pa['decay_law'], pb['decay_law']), (case['law']['A'], case['law']['B']))
            self.assertEqual((pa['decay_s'], pb['decay_s']), (case['decay']['A'], case['decay']['B']))
            self.assertEqual((pa['detector'], pa['spectrum'], pa['fullshape'], pa['events'], pa['excitation']),
                             (0, 1, 1, orz.EV_BIRTHS, orz.EXC_UNIFORM))
            self.assertEqual((pa['attack_ms'], pa['n'], pa['spread'], pa['harm'], pa['shape'], pa['alpha'], pa['dyn'],
                              pa['birth_strength'], pa['radius_mul']), (4.0, 3, 1.0, 0.87, 0.0, 0.0, 0.0, 1.0, 1.0))
            self.assertEqual(doc['audio']['side_gain'], SIDE_GAIN[case['id']])
            on_disk = load_scene(os.path.join(ROOT, 'demos', case['id'] + '.json'))
            self.assertEqual(on_disk.variants, sc.variants)
            self.assertEqual(on_disk.side_gain, doc['audio']['side_gain'])
        # D2: the pause commands land at the exact samples, one block after the last strike
        self.assertEqual(tuple(D2_PAUSES), ((132704, True), (242880, False), (375232, True), (485408, False)))
        for at, _on in D2_PAUSES:
            self.assertEqual(at % BLOCK, 0)
        self.assertEqual(int(math.ceil(6 * (SR / 2.0) / BLOCK)) * BLOCK + BLOCK, D2_PAUSES[0][0])
        y, peak, clip, runner = render_sides(scene_for(CASES[1]), CASES[1]['seconds'])
        pauses = [(j[0], j[3]['on'], j[3].get('script')) for j in runner.journal if j[2] == 'pause']
        self.assertEqual(pauses, [(at, on, True) for at, on in D2_PAUSES])
        self.assertEqual(clip, dict(A=0, B=0))
        # the level windows with the delivered gains: A - B within 1 dB; D2's ratio is a
        # property of the engine (the side gain scales both components alike)
        for case, yy in ((CASES[1], y),):
            m = window_mask(case, yy['A'].shape[0])
            self.assertLess(abs(rms_db(yy['A'][m]) - rms_db(yy['B'][m])), 1.0)
        for case in (CASES[0], CASES[2]):
            self.assertNotIn('script', scene_for(case))
            yy, peak, clip, _r = render_sides(scene_for(case), case['seconds'])
            m = window_mask(case, yy['A'].shape[0])
            self.assertLess(abs(rms_db(yy['A'][m]) - rms_db(yy['B'][m])), 1.0, case['id'])
            self.assertEqual(clip, dict(A=0, B=0))
            self.assertTrue(all(np.all(np.isfinite(yy[s])) for s in ('A', 'B')))

    def test_headless_bench_rows_height_law_buttons_locks_and_row_text(self):
        import pygame
        import demo_bench as db
        from casynth_lab.audio_out import LiveEngine
        scene = scene_from_doc(scene_for(CASES[1]))
        runner = DemoRunner(scene)
        eng = LiveEngine(runner, sink=lambda m, b: None)
        pygame.init()
        app = db.BenchApp(scene, eng)
        n_rows = len(registry.get(EID).params) + len(registry.get(EID).ranges)
        self.assertEqual(n_rows, 18)
        self.assertEqual(app._params_height(EID), n_rows * db.ROW_H)
        low = db.BenchApp(scene, eng, max_height=880)
        self.assertLessEqual(low.height, 880)
        self.assertGreaterEqual(low.figure_rows, db.FIGURE_ROWS_MIN)
        self.assertLessEqual(db.CHOICE_H + 3, db.ROW_H + 2)
        self.assertLessEqual(db.RANGE_FIELD_H, db.ROW_H)
        rows = {spec[0]: rect for spec, rect in app._param_rows(EID)}
        self.assertIn('decay_law', rows)
        sx, sy, _sw, _sh = rows['decay_law']
        btns = app._choice_rects(EID, 'decay_law', sy)
        self.assertEqual(len(btns), 3)
        for _v, (bx, _by, bw, _bh) in btns:
            self.assertLessEqual(bx + bw, app.panel_x + db.PANEL_W)
        small = pygame.font.SysFont(db.FONT_NAMES, 14)
        for word in registry.get(EID).choices['decay_law']:
            self.assertLessEqual(small.size(word)[0], btns[0][1][2] - 2)        # the words fit the buttons
        limit = db.PANEL_W + db.MARGIN - 2
        x_text = 82 + 60 + 6                                                    # bar_x + bar_w + 6 of a figure row
        for f in (dict(adaptive=True, t_lo=0.08, t_hi=1.39), dict(adaptive=True, t_lo=0.42, t_hi=0.421), dict(adaptive=False)):
            txt = f"110 Hz  {db.figure_decay_text(f, 0.75)}"
            self.assertLessEqual(x_text + small.size(txt)[0], limit)
        self.assertEqual(db.figure_decay_text(dict(adaptive=False), 0.75), 'a 0.75')
        # the header lines of the Objects display fit left of the pattern column
        for law in (FIXED, COMMON, MODAL):
            for det in (0, 1):
                disp = dict(decay_law=law, decay_law_name=orz.LAW_NAMES[law], decay_s=1.39, detector=det,
                            radius_mul=10.25, spectrum=1, f0=110.0, laplace=dict(n=12, fullshape=1), ramp_left=5,
                            frequency_scale=880.0)
                self.assertLessEqual(small.size(db.decay_law_line(disp))[0], limit)
                # the fallback without Radius x always fits (the bench draws the first that fits)
                dec = ('decay 1.39 s*', 'Common*', 'Modal*')[law]
                self.assertLessEqual(small.size(f"Laplace 110 Hz n 12 full   {dec}")[0], limit)
                self.assertLessEqual(db.fit_text(small, [f"R x10.250   Laplace 110 Hz n 12 full   {dec}",
                                                         f"Laplace 110 Hz n 12 full   {dec}"], limit, (0, 0, 0)).get_width(),
                                     limit)
        for text in (orz.inactive(params(FIXED, spectrum=0))['decay_law'], orz.inactive(params(MODAL))['spectrum'],
                     orz.inactive(params(MODAL))['fullshape']):
            self.assertLessEqual(88 + small.size(text)[0], limit, text)
        self.assertEqual(db.figure_decay_text(dict(adaptive=True, t_lo=0.42, t_hi=1.16), 0.75), 'T 0.42-1.16 s')
        self.assertEqual(db.figure_decay_text(dict(adaptive=True, t_lo=0.42, t_hi=0.421), 0.75), 'T 0.42 s')
        screen = pygame.Surface((app.width, app.height))
        font = pygame.font.SysFont(db.FONT_NAMES, 17)
        eng.start()
        try:
            eng.post('start')
            eng.post('select', side='B')
            time.sleep(0.4)
            app.draw(screen, font, small)
            snap = eng.snapshot()
            self.assertEqual(snap['display']['B']['decay_law_name'], 'Modal age')
            ina = app._inactive(EID, snap['sides']['B'][1])
            self.assertIn('spectrum', ina)                                      # locked by the law
            self.assertIn('fullshape', ina)
            self.assertNotIn('decay_law', ina)
            # the Common button switches the law; the Spectrum row is not editable
            for v, rect in btns:
                if v == COMMON:
                    self.assertEqual(app.press((rect[0] + 2, rect[1] + 1), 1), 'param:decay_law')   # the top edge too
            app.release()
            ssx, ssy, _w, _h = rows['spectrum']
            self.assertIsNone(app.press((ssx + 10, ssy + 2), 1))
            time.sleep(0.4)
            self.assertEqual(eng.snapshot()['sides']['B'][1]['decay_law'], COMMON)
            # Fixed + Figure: the law buttons are locked with the reason
            eng.post('set_param', side='B', name='decay_law', value=FIXED)
            time.sleep(0.3)
            eng.post('set_param', side='B', name='spectrum', value=0)
            time.sleep(0.4)
            ina = app._inactive(EID, eng.snapshot()['sides']['B'][1])
            self.assertIn('decay_law', ina)
            self.assertIn(orz.LAW_CONDITION, ina['decay_law'])
            for v, rect in btns:
                if v == MODAL:
                    self.assertIsNone(app.press((rect[0] + 2, rect[1] + 8), 1))
            app.release()
            time.sleep(0.3)
            self.assertEqual(eng.snapshot()['sides']['B'][1]['decay_law'], FIXED)
            screen.fill(db.C_BG)
            app.draw(screen, font, small)
        finally:
            eng.stop()
            pygame.quit()


if __name__ == '__main__':
    os.makedirs(ART, exist_ok=True)
    unittest.main(verbosity=1)
