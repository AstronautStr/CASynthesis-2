#!/usr/bin/env python3
"""Objects / Laplace gates (REQ memory/req-objects-laplace-comparison-2026-09-17.md and the
three P2 of memory/research/object-resonators-n4-review-2026-09-16.md):

  - the Laplace spectrum law of the Objects engine equals the old map_laplacian on one
    compact component away from the seam for the same seven settings, f0 and
    excitation (every generation of the three scenes, several settings, |diff| <= 1e-10
    -- in fact 0), fullshape 0 and 1, dyn with the events field
  - every one of the seven settings changes the spectral data of the galaxy in at least
    one generation (spread / fullshape in the generations the preflight names), spread
    selects from the whole spectrum, shape needs the eigenvectors
  - the engine in the Laplace mode: lowest mode f0 of the scene, weights = LAPLACE_GAIN
    times the map_laplacian amplitudes, Freq scale inactive; a setting / spectrum /
    excitation change retunes at once, ramps the weights over 20 ms and never strikes
  - v2 tail rules: a returning mode starts from zero while the vanished modes ring on in
    a tail of their own (the review's 4 -> 2 -> 4 case, Continue in the middle of the
    transition), exhausted pools fade in place (no hard drop, linear 20 ms ramp)
  - registry hints, copy_spectrum (only the shared seven, no engine / detector / radius /
    decay change), older parameter sets / snapshots mean Figure
  - scene side_gain: validation, absent = bit-exact, exact effect, Continue
  - the three scenes against the REQ / preflight, L3 receiver events at x1 / x1.5, the
    catalog build / replay / continue / notes, the bench panel (buttons, inactive
    settings, height), the block budget of both sides after warm-up

    python tests/test_objects_laplace.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_ol_tests/.
"""
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
from casynth_core import map_laplacian, extract, laplacian_modes, PATCH_SIZE  # noqa: E402
from casynth_engine import step, events_field, analyse, _crop_like_extract    # noqa: E402
from casynth_lab import BLOCK, registry, DemoRunner, load_scene, scene_from_doc   # noqa: E402
from casynth_lab.engine_api import EngineContext                              # noqa: E402
from casynth_lab.scene import SceneError                                      # noqa: E402
from casynth_lab.snapshot import save_state, load_state                      # noqa: E402
from casynth_lab import figures as fg                                        # noqa: E402
from casynth_lab import object_resonators as orz                             # noqa: E402

ART = os.path.join(ROOT, 'artifacts', '_ol_tests')
F0 = 110.0
CTX = EngineContext(SR, BLOCK, 2, F0, 1.0, 6.0)
GAIN = MASTER_GAIN * 0.7
TOL = 1e-10
PREFLIGHT = os.path.join(ROOT, 'memory', 'research', 'object-resonators-laplace-preflight-2026-09-17.json')
SETTINGS = [dict(n=12, spread=0.0, alpha=1.0, shape=0.0, harm=0.0, fullshape=1, dyn=0.0),   # the REQ table
            dict(n=4, spread=1.0, alpha=2.0, shape=1.0, harm=1.0, fullshape=0, dyn=1.0),
            dict(n=20, spread=0.5, alpha=0.5, shape=0.6, harm=0.3, fullshape=1, dyn=0.7),
            dict(n=1, spread=0.0, alpha=0.0, shape=0.2, harm=0.0, fullshape=0, dyn=0.0)]


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def cells_of(case):
    return [(int(r), int(c)) for r, c in preflight()['cases'][case]['cells']]


def grid(cells, rows=32, cols=32):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def laplace_params(spectrum=orz.SPEC_LAPLACE, detector=1, radius_mul=1.0, **sets):
    p = dict(detector=detector, radius_mul=radius_mul, spectrum=spectrum, frequency_scale=220.0,
             decay_s=0.8, **orz.laplace_settings({}))
    p.update(sets)
    return p


def engine(cells, params=None, gain=GAIN, exc=None):
    e = registry.create(orz.ENGINE_ID, CTX, laplace_params() if params is None else params)
    e.init(grid(cells), exc, gain)
    return e


def run(e, n_blocks, gain=GAIN):
    out = []
    for _ in range(n_blocks):
        y, _pk, _nc = e.render_float(gain)
        out.append(y)
    return np.concatenate(out) if out else np.zeros((0, 2))


def old_laplace_on_bbox(cells, sets, exc):
    """map_laplacian exactly as the old synth's analyse() calls it for one object."""
    cells = np.asarray(cells)
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), np.uint8)
    sub[cells[:, 0] - r0, cells[:, 1] - c0] = 1
    if sets['fullshape']:
        patch = sub
        ex = None if exc is None else exc[r0:r1 + 1, c0:c1 + 1] * sub
    else:
        patch = extract(sub, PATCH_SIZE)
        ex = None if exc is None else _crop_like_extract(exc[r0:r1 + 1, c0:c1 + 1] * sub, sub, PATCH_SIZE)
    fr, am = map_laplacian(patch, F0, sets['n'], sets['spread'], sets['alpha'], sets['shape'], sets['harm'],
                           bool(sets['fullshape']), sets['dyn'], ex)
    num = int(np.count_nonzero(fr))
    return fr[:num], am[:num]


def evolve_fields(cells, generations):
    """[(grid, exc)] of the generations 0..generations (exc of the transition, None first)."""
    g = grid(cells)
    out = [(g, None)]
    for _ in range(generations):
        prev, g = g, step(g)
        out.append((g, events_field(prev, g)))
    return out


# ======================================================================================
class LaplaceLawTests(unittest.TestCase):
    def test_laplace_law_equals_map_laplacian_on_every_compact_component(self):
        worst = 0.0
        count = 0
        for case in ('glider', 'galaxy', 'neighbor'):
            for g, exc in evolve_fields(cells_of(case), 9):
                for sets in SETTINGS:
                    for cells in fg.components(g):
                        f, a = orz.laplace_modes_of(cells, 32, 32, F0, sets, exc)
                        fr, am = old_laplace_on_bbox(cells, sets, exc)
                        self.assertEqual(len(f), len(fr))
                        self.assertEqual(len(a), len(f))
                        if len(f):
                            worst = max(worst, float(np.abs(f - fr).max()), float(np.abs(a - am).max()))
                            self.assertAlmostEqual(float(f[0]), F0, delta=1e-9)      # lowest mode = f0
                        count += 1
        self.assertGreater(count, 300)
        self.assertLessEqual(worst, TOL)
        # the analyse() path of the old synth (its segmentation + bbox) agrees too where
        # the field has no seam: the neighbor field, both objects
        g = grid(cells_of('neighbor'))
        _l, voices, _c = analyse(g, F0, 'laplacian', dict(SETTINGS[0]))
        comps = fg.components(g)
        got = {len(c): orz.laplace_modes_of(c, 32, 32, F0, SETTINGS[0])[0] for c in comps}
        for v in voices:
            num = int(np.count_nonzero(v['freqs']))
            match = [f for f in got.values() if len(f) == num and np.array_equal(f, v['freqs'][:num])]
            self.assertTrue(match)

    def test_every_setting_changes_the_galaxy_spectral_data_spread_and_shape_semantics(self):
        pf = preflight()
        fields = evolve_fields(cells_of('galaxy'), 7)                # one period: generations 0..7
        base = dict(SETTINGS[0])
        changed = {}
        for k, v in ((k, pf['galaxy_controls_over_period'][k]['value']) for k in orz.SPECTRUM_KEYS):
            alt = dict(base, **{k: v})
            if k == 'dyn':
                alt['shape'] = 1.0                                     # dyn acts only with shape > 0
                ref = dict(base, shape=1.0)
            else:
                ref = base
            gens = []
            for gen, (g, exc) in enumerate(fields):
                differs = False
                for cells in fg.components(g):
                    f0_, a0 = orz.laplace_modes_of(cells, 32, 32, F0, ref, exc)
                    f1, a1 = orz.laplace_modes_of(cells, 32, 32, F0, alt, exc)
                    if len(f0_) != len(f1) or not np.array_equal(f0_, f1) or not np.array_equal(a0, a1):
                        differs = True
                if differs:
                    gens.append(gen)
            changed[k] = gens
            self.assertTrue(gens, f"{k} changes nothing")
        self.assertEqual(changed['spread'], [2, 4, 5])                   # the preflight's generations
        self.assertEqual(changed['fullshape'], [2, 4, 5])
        for k in ('alpha', 'shape', 'harm'):
            self.assertEqual(changed[k], list(range(8)))
        self.assertEqual(changed['dyn'], list(range(1, 8)))              # generation 0 has no events yet
        # n = 4 against 12: generations 3 and 7 hold only 4-cell components (3 modes each),
        # both values select the same three -- the preflight counted the padded array
        self.assertEqual(changed['n'], [0, 1, 2, 4, 5, 6])
        # spread = 1 selects across the WHOLE spectrum before the count limit: the 64-cell
        # phase has 63 modes, n = 12 of them decimated up to the top
        big = [c for c in fg.components(fields[2][0]) if len(c) == 64][0]
        f_low, _ = orz.laplace_modes_of(big, 32, 32, F0, dict(base, spread=0.0))
        f_all, _ = orz.laplace_modes_of(big, 32, 32, F0, dict(base, spread=1.0))
        sq = fg.spectrum(big, 32, 32, 63)
        self.assertEqual(len(sq), 63)
        self.assertEqual(f_low[0], f_all[0])
        self.assertGreater(f_all[-1], f_low[-1])
        self.assertAlmostEqual(f_all[-1] / f_all[0], float(sq[-1] / sq[0]), places=9)
        # shape > 0 uses eigenvectors: the weights differ from the rolloff, dyn blends the
        # events field (a figure without events falls back to the static projection)
        _f, a_roll = orz.laplace_modes_of(big, 32, 32, F0, dict(base, shape=0.0))
        _f, a_shape = orz.laplace_modes_of(big, 32, 32, F0, dict(base, shape=1.0))
        self.assertFalse(np.allclose(a_roll, a_shape))
        exc = fields[2][1]
        _f, a_dyn = orz.laplace_modes_of(big, 32, 32, F0, dict(base, shape=1.0, dyn=1.0), exc)
        _f, a_dyn0 = orz.laplace_modes_of(big, 32, 32, F0, dict(base, shape=1.0, dyn=1.0), np.zeros((32, 32)))
        self.assertFalse(np.allclose(a_dyn, a_shape))
        np.testing.assert_allclose(a_dyn0, a_shape, rtol=1e-9, atol=1e-9)  # fallback to deg (normalised)
        # a wrapping figure keeps the torus graph (the old bbox would lose the wrap edges)
        ring = [(0, c) for c in range(32)]
        f_ring, _ = orz.laplace_modes_of(np.array(ring), 32, 32, F0, base)
        L = fg.laplacian_matrix(fg.canonical_cells(np.array(ring), 32, 32), 32, 32)
        self.assertEqual(float(L.diagonal().min()), 2.0)                # every cell has two neighbours
        fr, _ = laplacian_modes(L, F0, 12)
        np.testing.assert_array_equal(f_ring, fr[:int(np.count_nonzero(fr))])

    def test_engine_laplace_mode_frequencies_weights_and_inactive_scale(self):
        cells = cells_of('neighbor')
        e = engine(cells)
        run(e, 1)
        d = e.display()
        self.assertEqual(d['spectrum_name'], 'Laplace')
        self.assertEqual(d['f0'], F0)
        for f in d['figures']:
            s = f['slot']
            fr, am = old_laplace_on_bbox(np.array(f['cells']), SETTINGS[0], None)
            self.assertEqual(f['modes'], len(fr))
            np.testing.assert_array_equal(e.ffreq[s, :len(fr)], fr)
            np.testing.assert_allclose(e.wtgt[s, :len(fr)], orz.LAPLACE_GAIN * am, rtol=0, atol=1e-15)
            self.assertEqual(int(e.smode[s]), orz.SPEC_LAPLACE)
            self.assertAlmostEqual(f['f_low'], F0, delta=1e-9)
        self.assertEqual([f['modes'] for f in d['figures']], [12, 2])
        # Freq scale does nothing in this mode (and no packet)
        before = e.ffreq.copy()
        e.set_params(dict(e.params, frequency_scale=440.0))
        np.testing.assert_array_equal(e.ffreq, before)
        self.assertEqual(float(np.abs(e.zf).max()), float(np.abs(e.zf).max()))
        # the same field in the Figure law: scale * sqrt(lambda), weights 1 / n
        e2 = engine(cells, laplace_params(spectrum=orz.SPEC_FIGURE))
        run(e2, 1)
        f = e2.display()['figures'][0]
        sq = fg.spectrum(np.array(f['cells']), 32, 32, orz.N_BANK)
        np.testing.assert_array_equal(e2.ffreq[f['slot'], :len(sq)], 220.0 * sq)
        self.assertEqual(f['modes'], 16)
        self.assertEqual(float(e2.wtgt[f['slot'], 0]), 1.0 / 16)

    def test_setting_spectrum_and_excitation_changes_retune_without_a_packet(self):
        cells = cells_of('neighbor')
        e = engine(cells)
        run(e, 4)                                                       # the start packet is over the ramps
        s = e.display()['figures'][0]['slot']
        zf0, zs0 = e.zf.copy(), e.zs.copy()
        z0 = e.zre.copy()
        # spread: frequencies change at once, states kept, weights ramp 20 ms, no packet
        e.set_params(dict(e.params, spread=1.0))
        fr, am = old_laplace_on_bbox(np.array(cells_of('neighbor')[:2]), dict(SETTINGS[0], spread=1.0), None)
        f_recv, a_recv = orz.laplace_modes_of(e.figures[1].cells, 32, 32, F0, dict(SETTINGS[0], spread=1.0))
        np.testing.assert_array_equal(e.ffreq[s, :12], f_recv)
        np.testing.assert_array_equal(e.zre, z0)
        np.testing.assert_array_equal(e.zf, zf0)
        np.testing.assert_array_equal(e.zs, zs0)
        self.assertEqual(int(e.wleft[s]), 882)
        np.testing.assert_allclose(e.wtgt[s, :12], orz.LAPLACE_GAIN * a_recv, atol=1e-15)
        y = run(e, 3)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertEqual(int(e.wleft[s]), 0)
        np.testing.assert_allclose(e.wcur[s, :12], orz.LAPLACE_GAIN * a_recv, atol=1e-15)
        # n 12 -> 4: eight modes vanish into a tail, the bank keeps four; n 4 -> 12: eight
        # NEW modes from zero, the tail rings on
        e.set_params(dict(e.params, n=4))
        self.assertEqual(int(e.ndrive[s]), 4)
        tails = [t for t in range(orz.N_SLOTS) if e.role[t] == orz.ROLE_TAIL]
        self.assertEqual(len(tails), 1)
        self.assertEqual(int(e.nlive[tails[0]]), 8)
        np.testing.assert_array_equal(e.ffreq[tails[0], :8], f_recv[4:])
        run(e, 1)
        e.set_params(dict(e.params, n=12))
        self.assertEqual(int(e.ndrive[s]), 12)
        self.assertEqual(float(np.abs(e.zre[s, 4:12]).max()), 0.0)
        self.assertEqual(float(np.abs(e.zim[s, 4:12]).max()), 0.0)
        self.assertGreater(float(np.abs(e.zre[tails[0], :8]).max()), 0.0)
        self.assertEqual(int(e.ndrive[tails[0]]), 0)
        # the spectrum switch Laplace -> Figure -> Laplace: retune, no packet, ids kept
        ids = sorted(e.figures)
        zf1, zs1 = e.zf.copy(), e.zs.copy()
        e.set_params(dict(e.params, spectrum=orz.SPEC_FIGURE))
        self.assertEqual(int(e.smode[s]), orz.SPEC_FIGURE)
        self.assertEqual(int(e.ndrive[s]), 16)
        np.testing.assert_array_equal(e.zf, zf1)
        np.testing.assert_array_equal(e.zs, zs1)
        self.assertEqual(sorted(e.figures), ids)
        run(e, 1)
        zf1, zs1 = e.zf.copy(), e.zs.copy()
        e.set_params(dict(e.params, spectrum=orz.SPEC_LAPLACE))
        self.assertEqual(int(e.ndrive[s]), 12)
        self.assertEqual(sorted(e.figures), ids)
        np.testing.assert_array_equal(e.zf, zf1)
        np.testing.assert_array_equal(e.zs, zs1)
        # a new excitation on the SAME field retunes only when dyn acts (shape > 0, dyn > 0)
        g = grid(cells)
        exc = np.zeros((32, 32))
        exc[8, 8] = 1.7
        e.set_params(dict(e.params, shape=0.0, dyn=1.0))
        w_before = e.wtgt.copy()
        e.update_field(g, exc)
        run(e, 1)
        np.testing.assert_array_equal(e.wtgt, w_before)                 # dyn without shape: no effect
        e.set_params(dict(e.params, shape=1.0, dyn=1.0))
        run(e, 3)
        w_shape = e.wtgt[s, :12].copy()
        exc2 = np.zeros((32, 32))
        exc2[16, 15] = 1.0
        z_before = e.zre.copy()
        zf1, zs1 = e.zf.copy(), e.zs.copy()
        e.update_field(g, exc2)
        e._boundary()                                                    # processed, before any sample
        self.assertFalse(np.array_equal(e.wtgt[s, :12], w_shape))
        f_dyn, a_dyn = orz.laplace_modes_of(e.figures[1].cells, 32, 32, F0, orz.laplace_settings(e.params), exc2)
        np.testing.assert_allclose(e.wtgt[s, :12], orz.LAPLACE_GAIN * a_dyn, atol=1e-15)
        np.testing.assert_array_equal(e.zf, zf1)                         # weights, not a strike
        np.testing.assert_array_equal(e.zs, zs1)
        np.testing.assert_array_equal(e.zre, z_before)
        self.assertEqual(int(e.wleft[s]), 882)
        self.assertEqual(int(e.counters[orz.C_CHANGES]), 1)              # no field change counted
        run(e, 1)
        # Restore of the same field / excitation adds nothing
        st = e.export_state()
        run(e, 2)
        y1 = run(e, 3)
        other = engine(cells)
        other.restore_state(g, exc2, st)
        run(other, 2)
        y2 = run(other, 3)
        np.testing.assert_array_equal(y1, y2)

    def test_registry_hints_copy_spectrum_and_older_parameter_sets(self):
        spec = registry.get(orz.ENGINE_ID)
        self.assertEqual([p[0] for p in spec.params][-7:], list(orz.SPECTRUM_KEYS))
        core = {p[0]: p for p in registry.get('laplacian').params}
        for p in spec.params[-7:]:
            self.assertEqual(p, core[p[0]])                             # ranges / names / defaults of the old Laplace
        self.assertEqual(spec.defaults()['spectrum'], orz.SPEC_LAPLACE)
        self.assertEqual(registry.value_text(orz.ENGINE_ID, 'spectrum', 0), 'Figure')
        self.assertEqual(registry.value_text(orz.ENGINE_ID, 'spectrum', 1), 'Laplace')
        inactive = spec.inactive(laplace_params())
        self.assertIn('frequency_scale', inactive)
        self.assertIn('dyn', inactive)                                  # shape = 0: dyn cannot act
        self.assertNotIn('n', inactive)
        self.assertEqual(spec.inactive(laplace_params(shape=0.5)), {'frequency_scale': inactive['frequency_scale']})
        fig = spec.inactive(laplace_params(spectrum=orz.SPEC_FIGURE))
        self.assertEqual(sorted(fig), sorted(orz.SPECTRUM_KEYS))
        self.assertIn('[Disk, Laplace]', spec.overlay(laplace_params(), 32, 32)['text'])
        self.assertEqual(registry.spectrum_keys('laplacian', orz.ENGINE_ID), orz.SPECTRUM_KEYS)
        self.assertEqual(registry.spectrum_keys('fft2d', orz.ENGINE_ID), ('n',))
        self.assertEqual(registry.spectrum_keys('pm_network', orz.ENGINE_ID), ())
        # copy_spectrum: the seven shared settings only, both directions, no restart
        scene = load_scene(os.path.join(ROOT, 'demos', 'ol_glider.json'))
        r = DemoRunner(scene)
        r.post('start', at=0)
        r.post('set_param', at=0, side='A', name='spread', value=0.7)
        r.post('set_param', at=0, side='A', name='n', value=5)
        r.post('set_param', at=0, side='B', name='radius_mul', value=2.0)
        r.post('set_param', at=0, side='B', name='harm', value=0.4)
        for _ in range(3):
            r.next_block()
        r.post('copy_spectrum', src='A', dst='B')
        r.next_block()
        a, b = r.side_settings()['A'][1], r.side_settings()['B'][1]
        self.assertEqual({k: b[k] for k in orz.SPECTRUM_KEYS}, {k: a[k] for k in orz.SPECTRUM_KEYS})
        self.assertEqual((b['detector'], b['radius_mul'], b['spectrum'], b['decay_s'], b['frequency_scale']),
                         (1, 2.0, 1, 0.8, 220.0))
        self.assertEqual(r.side_settings()['B'][0], orz.ENGINE_ID)
        self.assertEqual(r.side_settings()['A'][0], 'laplacian')
        self.assertEqual(r.journal[-1][2], 'copy_spectrum')
        r.post('set_param', at=None, side='B', name='alpha', value=1.7)
        r.post('copy_spectrum', src='B', dst='A')
        r.next_block()
        self.assertEqual(r.side_settings()['A'][1]['alpha'], 1.7)
        self.assertEqual(r.side_settings()['A'][0], 'laplacian')
        with self.assertRaises(ValueError):
            r.post('copy_spectrum', src='A', dst='A')
        r.post('set_engine', at=None, side='A', engine_id='pm_network')
        with self.assertRaises(ValueError):                             # nothing shared -> rejected, no journal entry
            r.post('copy_spectrum', src='B', dst='A')
        # an engine created without the new keys, and a snapshot without them, mean Figure
        e = registry.create(orz.ENGINE_ID, CTX, dict(detector=1, frequency_scale=220.0, decay_s=0.8))
        self.assertEqual(e.params['spectrum'], orz.SPEC_FIGURE)
        self.assertEqual(orz.laplace_settings(e.params), orz.laplace_settings({}))
        e.init(grid(cells_of('glider')), None, GAIN)
        run(e, 2)
        st = e.export_state()
        old = dict(st)
        old['params'] = {k: v for k, v in st['params'].items() if k not in ('spectrum',) + orz.SPECTRUM_KEYS}
        other = registry.create(orz.ENGINE_ID, CTX, registry.defaults(orz.ENGINE_ID))
        other.init(np.zeros((32, 32), np.uint8), None, 0.0)
        other.restore_state(grid(cells_of('glider')), None, old)
        self.assertEqual(other.params, e.params)
        np.testing.assert_array_equal(run(other, 3), run(e, 3))
        with self.assertRaises(ValueError):
            registry.create(orz.ENGINE_ID, CTX, laplace_params()).restore_state(
                grid(cells_of('glider')), None, dict(st, laplace_gain=0.1))


# ======================================================================================
class TailRuleTests(unittest.TestCase):
    def test_returning_modes_start_from_zero_and_old_tails_ring_undriven(self):
        """The review's case: five cells -> three -> five (Figure law, 4 -> 2 -> 4 modes)."""
        os.makedirs(ART, exist_ok=True)
        five = [(9, 14), (9, 15), (9, 16), (10, 15), (10, 16)]
        e = engine(five, laplace_params(spectrum=orz.SPEC_FIGURE))
        run(e, 12)
        s = e.display()['figures'][0]['slot']
        f_old = e.ffreq[s, :4].copy()
        z_old = np.hypot(e.zre[s, 2:4], e.zim[s, 2:4])
        self.assertGreater(float(z_old.min()), 1.0)
        e.update_field(grid(five[:3]), None)
        run(e, 1)
        d = e.display()
        self.assertEqual((d['figures'][0]['id'], d['figures'][0]['modes'], d['n_tails']), (1, 2, 1))
        t = [k for k in range(orz.N_SLOTS) if e.role[k] == orz.ROLE_TAIL][0]
        self.assertEqual(int(e.ndrive[t]), 0)
        np.testing.assert_array_equal(e.ffreq[t, :2], f_old[2:4])         # the old frequencies
        self.assertEqual(int(e.nlive[s]), 2)
        e.update_field(grid(five), None)
        st_mid = e.export_state()                                        # pending regrowth, not processed
        e._boundary()                                                    # processed, before any sample
        d = e.display()
        self.assertEqual((d['figures'][0]['id'], d['figures'][0]['modes']), (1, 4))
        self.assertEqual(float(np.abs(e.zre[s, 2:4]).max()), 0.0)        # new modes from zero
        self.assertEqual(float(np.abs(e.zim[s, 2:4]).max()), 0.0)
        self.assertEqual(float(e.wcur[s, 2]), 0.0)
        self.assertEqual(int(e.wleft[s]), 882)
        self.assertGreater(float(e.last_a[s]), 0.0)                      # the regrowth strikes the bank ...
        self.assertEqual(float(e.zf[t]), 0.0)                            # ... never the old tail
        self.assertEqual(int(e.role[t]), orz.ROLE_TAIL)
        zt = np.hypot(e.zre[t, :2], e.zim[t, :2])
        self.assertTrue(np.all(zt > 0.0))
        y = run(e, int(4.0 * SR / BLOCK))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertEqual(e.display()['n_tails'], 0)                      # the old tail decayed and was freed
        # Continue in the middle of the transition equals the continuous run
        e2 = engine(five, laplace_params(spectrum=orz.SPEC_FIGURE))
        e2.restore_state(grid(five), None, st_mid)
        e3 = engine(five, laplace_params(spectrum=orz.SPEC_FIGURE))
        run(e3, 12)
        e3.update_field(grid(five[:3]), None)
        run(e3, 1)
        e3.update_field(grid(five), None)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            save_state(tmp, 'mid', st_mid)
            back = load_state(tmp, 'mid')
        e4 = engine(five, laplace_params(spectrum=orz.SPEC_FIGURE))
        e4.restore_state(grid(five), None, back)
        for _ in range(20):
            a = e2.render_float(GAIN)[0]
            b = e3.render_float(GAIN)[0]
            c = e4.render_float(GAIN)[0]
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, c)
        # the same rule in the Laplace law with n as the lever (12 -> 6 -> 12)
        e = engine(cells_of('neighbor'))
        run(e, 6)
        s = e.display()['figures'][0]['slot']
        e.set_params(dict(e.params, n=6))
        run(e, 1)
        e.set_params(dict(e.params, n=12))
        self.assertEqual(float(np.abs(e.zre[s, 6:12]).max()), 0.0)
        self.assertEqual(sum(1 for k in range(orz.N_SLOTS) if e.role[k] == orz.ROLE_TAIL), 1)

    def test_exhausted_pools_fade_in_place_and_nothing_is_cut(self):
        many = []
        for r in range(0, 32, 4):
            for c in range(0, 30, 5):
                many += [(r + 1, c), (r + 1, c + 1), (r + 1, c + 2)]
        g = grid(many)
        e = engine(many, laplace_params(spectrum=orz.SPEC_FIGURE), gain=0.04)
        run(e, 1, 0.04)
        peak = 0.0
        for k in range(8):
            e.update_field(np.zeros_like(g), None)
            y, pk, _ = e.render_float(0.04)
            peak = max(peak, pk)
            e.update_field(g, None)
            y, pk, _ = e.render_float(0.04)
            peak = max(peak, pk)
        d = e.display()
        self.assertEqual((d['n_tails'], d['n_fading']), (orz.N_TAILS, orz.N_FADING))
        self.assertGreater(d['evictions'], 0)
        self.assertGreater(d['inplace_fades'], 0)
        self.assertEqual(d['drops'], 0)
        self.assertTrue(np.isfinite(peak))
        # one controlled in-place fade: every tail and fading slot busy, a figure leaves ->
        # its active slot ramps its weights to 0 linearly over 882 samples and is freed
        self.assertGreater(d['unvoiced_blocks'], 0)                       # in-place fades hold active slots
        e = engine(many, laplace_params(spectrum=orz.SPEC_FIGURE), gain=0.04)
        run(e, 1, 0.04)
        for k in range(8):
            e.update_field(np.zeros_like(g), None)
            run(e, 1, 0.04)
            e.update_field(g, None)
            run(e, 1, 0.04)
        run(e, 3, 0.04)                                                   # every fade finished
        self.assertEqual(e.display()['n_fading'], 0)
        e.update_field(g, None)
        run(e, 1, 0.04)                                                   # 24 figures sound
        e.update_field(np.zeros_like(g), None)
        run(e, 1, 0.04)                                                   # -> 24 evictions fill the fading pool
        self.assertEqual(e.display()['n_fading'], orz.N_FADING)
        e.update_field(g, None)
        run(e, 1, 0.04)                                                   # active slots free: 24 figures again
        self.assertEqual(e.display()['n_sounding'], orz.N_ACTIVE)
        e.update_field(np.zeros_like(g), None)
        e._boundary()                                                     # they leave: every pool still busy
        s = [k for k in range(orz.N_ACTIVE) if e.role[k] == orz.ROLE_FADING and e.wleft[k] == 882]
        self.assertTrue(s)
        s = s[0]
        w0 = e.wcur[s, :].copy()
        self.assertGreater(float(w0.max()), 0.0)
        e._set_gain(0.04)
        e._kernel(BLOCK, e._out)
        np.testing.assert_allclose(e.wcur[s], w0 + BLOCK * (0.0 - w0) / 882, rtol=0, atol=1e-12)
        self.assertEqual(int(e.wleft[s]), 882 - BLOCK)
        run(e, 3, 0.04)
        self.assertEqual(int(e.role[s]), orz.ROLE_FREE)
        self.assertEqual(e.display()['drops'], 0)
        # whole-field replacement every block on a dense field: finite, no hard drop
        rng = np.random.default_rng(7)
        dense = (rng.random((32, 32)) < 0.5).astype(np.uint8)
        for spectrum in (orz.SPEC_FIGURE, orz.SPEC_LAPLACE):
            e = registry.create(orz.ENGINE_ID, CTX, laplace_params(spectrum=spectrum))
            e.init(dense, None, MASTER_GAIN)
            for b in range(60):
                gg = np.zeros_like(dense) if b % 2 else dense
                e.update_field(gg, None)
                y, pk, _nc = e.render_float(MASTER_GAIN)
                self.assertTrue(np.all(np.isfinite(y)))
            self.assertEqual(e.display()['drops'], 0)


# ======================================================================================
class SideGainAndBenchTests(unittest.TestCase):
    def test_scene_side_gain_validation_absent_is_exact_and_the_effect_is_exact(self):
        path = os.path.join(ROOT, 'demos', 'ol_galaxy.json')
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        self.assertEqual(doc['audio']['side_gain'], dict(A=1.0, B=0.74))
        for bad in ({'C': 1.0}, {'A': 0.0}, {'B': -1.0}, {'B': 'x'}, {'B': True}, [1.0], {'A': 17.0}):
            d = json.loads(json.dumps(doc))
            d['audio']['side_gain'] = bad
            with self.assertRaises(SceneError):
                scene_from_doc(d)
        # absent = 1.0 on both sides, bit-exact with an explicit 1.0
        d1 = json.loads(json.dumps(doc))
        del d1['audio']['side_gain']
        d2 = json.loads(json.dumps(doc))
        d2['audio']['side_gain'] = dict(A=1.0, B=1.0)
        s1, s2 = scene_from_doc(d1), scene_from_doc(d2)
        self.assertEqual(s1.side_gain, dict(A=1.0, B=1.0))
        r1, r2 = DemoRunner(s1), DemoRunner(s2)
        for r in (r1, r2):
            r.post('start', at=0)
        for _ in range(30):
            a, b = r1.next_block(), r2.next_block()
            for o in ('A', 'B', 'monitor'):
                np.testing.assert_array_equal(a.get(o), b.get(o))
        # B at 0.5 equals the same scene at half the volume on side B (both engines apply
        # the gain once, before the clip): the Laplace side too
        d3 = json.loads(json.dumps(doc))
        d3['audio']['side_gain'] = dict(A=0.5, B=0.5)
        r3 = DemoRunner(scene_from_doc(d3), vol=0.7)
        r4 = DemoRunner(s1, vol=0.35)
        for r in (r3, r4):
            r.post('start', at=0)
        for _ in range(60):
            a, b = r3.next_block(), r4.next_block()
            np.testing.assert_array_equal(a.A, b.A)
            np.testing.assert_array_equal(a.B, b.B)
        # part of the snapshot: Continue keeps the calibration
        st = r3.export_state()
        twin = DemoRunner.from_state(st)
        self.assertEqual(twin.scene.side_gain, dict(A=0.5, B=0.5))
        for _ in range(20):
            a, b = r3.next_block(), twin.next_block()
            for o in ('A', 'B', 'monitor'):
                np.testing.assert_array_equal(a.get(o), b.get(o))

    @staticmethod
    def _wait(eng, cond, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout and not cond(eng.snapshot()):
            time.sleep(0.02)
        return cond(eng.snapshot())

    def test_headless_bench_panel_buttons_inactive_settings_and_height(self):
        import pygame
        import demo_bench as db
        from casynth_lab.audio_out import LiveEngine
        scene = load_scene(os.path.join(ROOT, 'demos', 'ol_neighbor.json'))
        runner = DemoRunner(scene)
        eng = LiveEngine(runner, sink=lambda m, b: None)
        pygame.init()
        app = db.BenchApp(scene, eng)
        # the whole Objects panel fits: 13 setting rows + the range row + the figure rows + the footer
        n_rows = len(registry.get(orz.ENGINE_ID).params) + len(registry.get(orz.ENGINE_ID).ranges)
        self.assertEqual(n_rows, 16)                                    # 14 until Events / Excitation (2026-09-17)
        self.assertGreaterEqual(app.footer_y, app.params_y + n_rows * db.ROW_H + 6 + db.FIGURE_HEAD_H + app.figure_rows * db.DISPLAY_ROW_H)
        self.assertGreaterEqual(app.height, app.footer_y + 40 + db.MARGIN)
        self.assertLessEqual(app.height, 1000)
        screen = pygame.Surface((app.width, app.height))
        font, small = pygame.font.SysFont(db.FONT_NAMES, 17), pygame.font.SysFont(db.FONT_NAMES, 14)
        eng.start()
        try:
            eng.post('start')
            self.assertTrue(self._wait(eng, lambda s: s['gen'] >= 1))
            app.press((app.tabs['B'][0] + 2, app.tabs['B'][1] + 2), 1)
            self.assertTrue(self._wait(eng, lambda s: s['selected'] == 'B'))
            app.draw(screen, font, small)
            rows = {spec[0]: rect for spec, rect in app._param_rows(orz.ENGINE_ID)}
            # side B in the Laplace mode: Freq scale and dyn are shown inactive (no slider)
            sx, sy, _w, _h = rows['frequency_scale']
            self.assertIsNone(app.press((sx + 10, sy + 2), 1))
            sx, sy, _w, _h = rows['dyn']
            self.assertIsNone(app.press((sx + 10, sy + 2), 1))
            sx, sy, _w, _h = rows['spread']
            self.assertEqual(app.press((sx + 60, sy + 2), 1), 'param:spread')
            app.release()
            self.assertTrue(self._wait(eng, lambda s: s['sides']['B'][1]['spread'] > 0.0))
            for value, rect in app._choice_rects(orz.ENGINE_ID, 'spectrum', rows['spectrum'][1]):
                if value == orz.SPEC_FIGURE:
                    self.assertEqual(app.press((rect[0] + 2, rect[1] + 2), 1), 'param:spectrum')
            self.assertTrue(self._wait(eng, lambda s: s['sides']['B'][1]['spectrum'] == orz.SPEC_FIGURE))
            app.draw(screen, font, small)
            inactive = app._inactive(orz.ENGINE_ID, eng.snapshot()['sides']['B'][1])
            self.assertEqual(sorted(inactive), sorted(orz.SPECTRUM_KEYS))
            # copy spectrum buttons: B -> A puts the seven onto the Laplace side, A -> B back
            rect = app.spec_btns[('B', 'A')]
            self.assertEqual(app.press((rect[0] + 3, rect[1] + 3), 1), 'spectrum:BA')
            self.assertTrue(self._wait(eng, lambda s: s['sides']['A'][1]['spread'] == s['sides']['B'][1]['spread']))
            self.assertEqual(eng.snapshot()['sides']['A'][0], 'laplacian')
            self.assertEqual(eng.snapshot()['sides']['B'][0], orz.ENGINE_ID)
            rect = app.spec_btns[('A', 'B')]
            self.assertEqual(app.press((rect[0] + 3, rect[1] + 3), 1), 'spectrum:AB')
            time.sleep(0.1)
            app.draw(screen, font, small)
            # the figure overlay of the listened side is drawn in the field in the figure's colour
            snap = eng.snapshot()
            f = snap['display']['B']['figures'][0]
            r, c = f['cells'][0]
            px = tuple(screen.get_at((app.field_x + c * db.CELL + 3, app.field_y + r * db.CELL + 3))[:3])
            col = db.C_FIGURES[f['color'] % len(db.C_FIGURES)]
            self.assertIn(px, (col, tuple((u + v) // 2 for u, v in zip(col, db.C_ALIVE_PAUSED))))
        finally:
            eng.stop()
            pygame.quit()


# ======================================================================================
class ScenesAndCatalogTests(unittest.TestCase):
    def test_scenes_match_the_req_and_the_preflight(self):
        from demos.build_objects_laplace import CASES, scene_for, RATE_HZ, SECONDS, SPECTRUM, SIDE_GAIN
        pf = preflight()
        self.assertEqual((RATE_HZ, SECONDS), (6.0, 12.0))
        self.assertEqual(SPECTRUM, orz.laplace_settings({}))
        self.assertEqual(SPECTRUM, pf['params'])
        for case, name, radius in zip(CASES, ('glider', 'galaxy', 'neighbor'), (1.0, 1.0, 1.5)):
            doc = scene_for(case)
            path = os.path.join(ROOT, 'demos', case['id'] + '.json')
            with open(path, encoding='utf-8') as f:
                self.assertEqual(json.load(f), doc)
            self.assertEqual(doc['cells'], [[int(r), int(c)] for r, c in pf['cases'][name]['cells']])
            self.assertEqual((doc['rate_hz'], doc['audio']['f0_hz'], doc['grid']), (6.0, 110.0, dict(rows=32, cols=32)))
            self.assertTrue(case['hypothesis'].startswith('Гипотеза - '))
            a, b = doc['variants']['A'], doc['variants']['B']
            self.assertEqual(a, dict(engine_id='laplacian', engine_params=dict(SPECTRUM)))
            self.assertEqual(b['engine_id'], orz.ENGINE_ID)
            self.assertEqual({k: b['engine_params'][k] for k in orz.SPECTRUM_KEYS}, SPECTRUM)
            self.assertEqual((b['engine_params']['detector'], b['engine_params']['radius_mul'],
                              b['engine_params']['spectrum'], b['engine_params']['decay_s']),
                             (orz.DET_DISK, radius, orz.SPEC_LAPLACE, 0.8))
            self.assertEqual(doc['audio']['side_gain'], SIDE_GAIN[case['id']])
            self.assertEqual(doc['audio']['side_gain']['A'], 1.0)
        # no seam in any of the 73 states, the component sizes of the preflight
        for name in ('glider', 'galaxy', 'neighbor'):
            g = grid(cells_of(name))
            sizes = set()
            for _ in range(73):
                self.assertFalse(g[0].any() or g[-1].any() or g[:, 0].any() or g[:, -1].any())
                sizes.add(tuple(sorted(len(c) for c in fg.components(g))))
                g = step(g)
            want = set(tuple(sorted(s)) for s in pf['cases'][name]['component_size_sets'])
            self.assertEqual(sizes, want)
        with open(os.path.join(ROOT, 'run_objects_laplace.bat'), encoding='ascii') as f:
            bat = f.read()
        self.assertIn('ol_glider.json', bat)
        self.assertIn('objects_laplace_2026_09_17', bat)
        self.assertNotIn('--live', bat)
        # the N4 catalog entry and scenes are untouched in spirit: still the Figure law
        from demos.build_n4_objects import N4_DISK, N4_OWN
        self.assertEqual((N4_DISK['spectrum'], N4_OWN['spectrum']), (orz.SPEC_FIGURE, orz.SPEC_FIGURE))

    def test_l3_receiver_hears_the_blinker_only_at_radius_1p5_and_the_blinker_itself_at_both(self):
        pf = preflight()
        want = pf['cases']['neighbor']['receiver_events_all_steps']
        for mul, per_gen in ((1.0, want['radius_1']), (1.5, want['radius_1p5'])):
            e = engine(cells_of('neighbor'), laplace_params(radius_mul=mul))
            g = grid(cells_of('neighbor'))
            run(e, 1)
            recv = [f for f in e.display()['figures'] if f['n'] == 17][0]['id']
            blk = [f for f in e.display()['figures'] if f['n'] == 3][0]['id']
            e_recv, e_blk = [], []
            for gen in range(12):
                g = step(g)
                e.update_field(g, None)
                run(e, 1)
                d = {f['id']: f for f in e.display()['figures']}
                self.assertEqual(sorted(d), sorted([recv, blk]))         # both keep their identity
                e_recv.append(d[recv]['e'])
                e_blk.append(d[blk]['e'])
            self.assertEqual(e_recv, [float(per_gen)] * 12)
            self.assertEqual(e_blk, [4.0] * 12)
        # the live lever: 1.5 -> 1.0 during the run stops the receiver's packets, the
        # blinker goes on; no packet from the knob itself
        e = engine(cells_of('neighbor'), laplace_params(radius_mul=1.5))
        g = grid(cells_of('neighbor'))
        run(e, 1)
        for _ in range(3):
            g = step(g)
            e.update_field(g, None)
            run(e, 1)
        recv = [f for f in e.display()['figures'] if f['n'] == 17][0]
        self.assertEqual(recv['e'], 4.0)
        zf = e.zf.copy()
        e.set_params(dict(e.params, radius_mul=1.0))
        np.testing.assert_array_equal(e.zf, zf)
        g = step(g)
        e.update_field(g, None)
        run(e, 1)
        d = {f['n']: f for f in e.display()['figures']}
        self.assertEqual((d[17]['e'], d[3]['e']), (0.0, 4.0))

    def test_catalog_build_replay_continue_notes_and_no_overwrite(self):
        from casynth_lab.catalog import Catalog
        from demos.build_n1_demos import record_offline
        from demos.build_objects_laplace import CASES, scene_for, build
        os.makedirs(ART, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            cat = Catalog(tmp, repo_root=None)
            firsts = []
            for case in CASES:
                doc = scene_for(case)
                rid, snap = record_offline(cat, doc, 0.5, [], case['title'], case['note'])
                self.assertEqual(snap['clip_blocks'], dict(A=0, B=0))
                cat.write_notes(rid, case['hypothesis'] + '\n\nUser feedback: test.')
                rec = cat.load(rid)
                self.assertTrue(rec.notes.startswith('Гипотеза - '))
                result = cat.replay(rid, yield_cpu=False)
                self.assertEqual(result.status, 'match', result.reason)
                continued, state, _ = cat.continue_runner(rid)
                self.assertEqual(continued.scene.side_gain, doc['audio']['side_gain'])
                control = DemoRunner(scene_from_doc(doc))
                control.post('start', at=0)
                for _ in range(math.ceil(0.5 * SR / BLOCK)):
                    control.next_block()
                gen0 = continued.gen
                first = None
                for _ in range(40):
                    a, b = continued.next_block(), control.next_block()
                    for s in ('A', 'B', 'monitor'):
                        np.testing.assert_array_equal(a.get(s), b.get(s))
                    if first is None:
                        first = a.B.tobytes()
                self.assertGreater(continued.gen, gen0)
                firsts.append(first)
                with self.assertRaises(SystemExit):
                    build(tmp)
                self.assertIn('User feedback: test.', cat.load(rid).notes)
            self.assertEqual(len(set(firsts)), len(CASES))

    def test_old_n4_records_are_pinned_and_run_in_their_own_version(self):
        """Records made by the v1 engine: their scene documents lack the new Objects
        parameters and the CURRENT code refuses them (no silent defaults); they are
        pinned to the commit of their sound files, so the bench runs them in a
        separate bench of that version (S6); the version's own check matches the
        saved audio when the repository and the audio are present."""
        from casynth_lab.catalog import Catalog
        from casynth_lab import provenance as prov
        root = os.path.join(ROOT, 'lab_catalog', 'object_resonators_n4_2026_09_16')
        if not os.path.isdir(root):
            self.skipTest('N4 catalog not present')
        cat = Catalog(root)
        entries = [r for r, err in cat.list() if err is None]
        self.assertEqual(len(entries), 2)
        for rec in entries:
            doc = rec.meta['scene']
            for side in ('A', 'B'):
                if doc['variants'][side]['engine_id'] == orz.ENGINE_ID:
                    self.assertNotIn('spectrum', doc['variants'][side]['engine_params'])
            with self.assertRaises(SceneError):                           # strict: no defaults invented
                scene_from_doc(doc)
            self.assertEqual(rec.status, 'pinned')
            self.assertTrue(rec.commit)
            if cat.repo_root is None or not prov.commit_exists(cat.repo_root, rec.commit):
                continue
            plan = cat.source_plan(rec)
            self.assertEqual(plan['mode'], 'worktree', plan)               # sound code differs -> its version
            self.assertIn('casynth_lab/object_resonators.py', plan['differs'])
            if not os.path.isfile(os.path.join(root, rec.id, 'monitor.wav')):
                continue                                                  # audio is not in git
            try:
                child = cat.run_version(rec, 'check')
            except Exception as e:                                        # noqa: BLE001  (no git worktree here)
                self.skipTest(f'version bench unavailable: {e}')
            self.assertTrue(child.wait(600))
            self.assertIsNotNone(child.result, child.error)
            self.assertEqual(child.result.get('status'), 'match', child.result)
            self.assertEqual(child.result.get('commit'), rec.commit)

    def test_block_budget_of_both_sides_on_the_three_scenes(self):
        budget_ms = BLOCK / SR * 1000.0
        for sid in ('ol_glider', 'ol_galaxy', 'ol_neighbor'):
            scene = load_scene(os.path.join(ROOT, 'demos', sid + '.json'))
            runner = DemoRunner(scene)
            runner.post('start', at=0)
            times = []
            n = int(6.0 * SR / BLOCK)
            for b in range(n):
                t0 = time.perf_counter()
                runner.next_block()
                times.append((time.perf_counter() - t0) * 1000.0)
            warm = np.array(times[int(1.0 * SR / BLOCK):])
            p99 = float(np.percentile(warm, 99))
            self.assertLess(p99, budget_ms, f"{sid}: p99 {p99:.2f} ms")
            self.assertEqual(runner.snapshot()['clip_blocks'], dict(A=0, B=0))


if __name__ == '__main__':
    unittest.main()
