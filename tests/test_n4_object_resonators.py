#!/usr/bin/env python3
"""N4 gates (REQ memory/req-object-resonators-n4-2026-09-16.md, "Приёмка Developer и
Researcher" 2-5): figure geometry and spectrum (blinker {0, 1, 3}, invariance across
the seam, the glider's two spectral phases, circle / centre / R, R = 0, the ambiguous
centre of a torus-spanning figure, the non-periodic ratios of the core Laplacian),
the matching (movement, one-cell shift, area change, a big newcomer, split / merge),
excitation semantics (zero from zero, unchanged field / parameters / Restore add no
packet, births and deaths on old / new geometry, the N4.2 receiver Own / Disk /
outside), the whole sample path against an independent scalar reference through the
decay / weight / pan / gain ramps (<= 1e-12), the life cycle (translation, growth,
Tumbler split / merge, clear all + quick re-add, tails release), the limits (24 banks,
96 tails, evictions visible), snapshot / Continue exactness, levels / timing, the
bench panel and overlay, and the prepared scenes / catalog.

    python tests/test_n4_object_resonators.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_n4_tests/.
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
from casynth_engine import step                                               # noqa: E402
from casynth_lab import BLOCK, registry                                       # noqa: E402
from casynth_lab.engine_api import EngineContext, supports_snapshot          # noqa: E402
from casynth_lab.snapshot import save_state, load_state                      # noqa: E402
from casynth_lab import event_network as en                                  # noqa: E402
from casynth_lab import figures as fg                                        # noqa: E402
from casynth_lab import object_resonators as orz                             # noqa: E402

ART = os.path.join(ROOT, 'artifacts', '_n4_tests')
CTX = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 6.0)
GAIN = MASTER_GAIN * 0.7                     # 0.028, the bench default
TOL = 1e-12
PREFLIGHT = os.path.join(ROOT, 'memory', 'research', 'object-resonators-n4-preflight-2026-09-16.json')


def grid(cells, rows=32, cols=32):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r % rows, c % cols] = 1
    return g


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def glider_cells():
    return [tuple(c) for c in preflight()['cases']['glider']['cells']]


def neighbor_cells():
    return [tuple(c) for c in preflight()['cases']['neighbors']['cells']]


def receiver_cells():
    """The still 17-cell figure of N4.2 (the neighbour blinker is (9, 14..16))."""
    return [c for c in neighbor_cells() if c not in blinker(9, 15)]


def block5(r0=20, c0=20):
    return [(r0 + r, c0 + c) for r in range(5) for c in range(5)]


def blinker(r=9, c=15):
    return [(r, c - 1), (r, c), (r, c + 1)]


def tumbler(r0=13, c0=12):
    rel = ((0, 0), (0, 1), (0, 5), (0, 6), (1, 0), (1, 2), (1, 4), (1, 6),
           (2, 0), (2, 2), (2, 4), (2, 6), (3, 2), (3, 4),
           (4, 1), (4, 2), (4, 4), (4, 5), (5, 1), (5, 2), (5, 4), (5, 5))
    return [(r0 + r, c0 + c) for r, c in rel]


def engine(cells, detector=1, scale=220.0, decay=0.8, gain=GAIN, rows=32, cols=32, radius_mul=1.0):
    e = registry.create(orz.ENGINE_ID, CTX, dict(detector=detector, radius_mul=radius_mul,
                                                 frequency_scale=scale, decay_s=decay))
    e.init(grid(cells, rows, cols), None, gain)
    return e


def run(e, n_blocks, gain=GAIN):
    out = []
    for _ in range(n_blocks):
        y, _pk, _nc = e.render_float(gain)
        out.append(y)
    return np.concatenate(out) if out else np.zeros((0, 2))


def evolve(e, g, seconds, rate=6.0, gain=GAIN, on_block=None):
    """Drive the engine like the bench: CA steps at `rate`, one render per block."""
    n_blocks = int(math.ceil(seconds * SR / BLOCK))
    step_s = SR / rate
    ca, gen = 0, 0
    ys = []
    for b in range(n_blocks):
        if ca >= (gen + 1) * step_s:
            g = step(g)
            gen += 1
            e.update_field(g, None)
        y, _pk, _nc = e.render_float(gain)
        ys.append(y)
        if on_block is not None:
            on_block(b, gen, e)
        ca += BLOCK
    return np.concatenate(ys), g


def fig_by_id(e, fid):
    for f in e.display()['figures']:
        if f['id'] == fid:
            return f
    return None


# -- the independent scalar reference ------------------------------------------------
def scalar_n4(slots, blocks, sr=SR, ramp_n=882, hp_hz=20.0, out_scale=orz.OUT_SCALE,
              r0=None, g0=GAIN):
    """Pure-Python sample path of the module docstring.  `slots`: list of dicts
    (freqs, weights, pan_p, ndrive, zre, zim) -- ringing modes beyond ndrive allowed;
    `blocks`: list of per-block actions {'inject': {slot: a}, 'decay': T60,
    'gain': g, 'tune': {slot: (freqs, n)}, 'pan': {slot: p}} applied at the block
    boundary before its samples (the same order as the engine: tune / pan / decay /
    inject, then gain).  v2 tune rules: a shrink moves the vanished driven modes into
    a tail slot of their own (states, frequencies, weights, panning copied, undriven,
    appended after the existing slots), every other undriven mode left in the slot
    ramps to weight 0; new modes start from zero.  Returns (n, 2)."""
    qf = math.exp(-1.0 / (sr * 0.00025))
    qs = math.exp(-1.0 / (sr * 0.002))
    strength = 0.75
    hp_h = math.exp(-2.0 * math.pi * hp_hz / sr)
    M = orz.N_BANK

    def padded(v, fill=0.0):
        v = list(v)
        return v + [fill] * (M - len(v))
    S = []
    for sd in slots:
        n = len(sd['freqs'])
        st = dict(re=padded(sd.get('zre', [0.0] * n)), im=padded(sd.get('zim', [0.0] * n)),
                  c=padded([math.cos(2.0 * math.pi * f / sr) for f in sd['freqs']], 1.0),
                  s=padded([math.sin(2.0 * math.pi * f / sr) for f in sd['freqs']]),
                  w=padded(sd['weights']), winc=[0.0] * M, wtgt=padded(sd['weights']), wleft=0,
                  nd=int(sd['ndrive']), zf=0.0, zs=0.0, pleft=0)
        L = math.cos(math.pi * sd['pan_p'] / 2.0)
        R = math.sin(math.pi * sd['pan_p'] / 2.0)
        st.update(L=L, Linc=0.0, Ltgt=L, R=R, Rinc=0.0, Rtgt=R)
        S.append(st)
    r_cur = r_tgt = (10.0 ** (-3.0 / (sr * 0.8)) if r0 is None else r0)
    r_inc, r_left = 0.0, 0
    g_cur = g_tgt = g0
    g_inc, g_left = 0.0, 0
    hp = [[0.0, 0.0], [0.0, 0.0]]
    out = []
    for blk in blocks:
        for si, (freqs, n) in blk.get('tune', {}).items():
            st = S[si]
            n_old = st['nd']
            if n < n_old:
                tail = dict(re=[0.0] * M, im=[0.0] * M, c=[1.0] * M, s=[0.0] * M, w=[0.0] * M,
                            winc=[0.0] * M, wtgt=[0.0] * M, wleft=st['wleft'], nd=0, zf=0.0, zs=0.0,
                            pleft=st['pleft'], L=st['L'], Linc=st['Linc'], Ltgt=st['Ltgt'],
                            R=st['R'], Rinc=st['Rinc'], Rtgt=st['Rtgt'])
                for k, j in enumerate(range(n, n_old)):
                    for key in ('re', 'im', 'c', 's', 'w', 'winc', 'wtgt'):
                        tail[key][k] = st[key][j]
                    st['re'][j] = st['im'][j] = 0.0
                    st['w'][j] = st['winc'][j] = st['wtgt'][j] = 0.0
                S.append(tail)
            for j, f in enumerate(freqs):
                st['c'][j] = math.cos(2.0 * math.pi * f / sr)
                st['s'][j] = math.sin(2.0 * math.pi * f / sr)
                if j >= n_old:
                    st['re'][j] = st['im'][j] = 0.0
            st['nd'] = n
            w = [1.0 / n if j < n else 0.0 for j in range(len(st['w']))]
            st['wtgt'] = w
            st['winc'] = [(w[j] - st['w'][j]) / ramp_n for j in range(len(w))]
            st['wleft'] = ramp_n
        for si, p in blk.get('pan', {}).items():
            st = S[si]
            st['Ltgt'] = math.cos(math.pi * p / 2.0)
            st['Rtgt'] = math.sin(math.pi * p / 2.0)
            st['Linc'] = (st['Ltgt'] - st['L']) / ramp_n
            st['Rinc'] = (st['Rtgt'] - st['R']) / ramp_n
            st['pleft'] = ramp_n
        if 'decay' in blk:
            r_tgt = 10.0 ** (-3.0 / (sr * blk['decay']))
            r_inc = (r_tgt - r_cur) / ramp_n
            r_left = ramp_n
        for si, a in blk.get('inject', {}).items():
            S[si]['zf'] += a
            S[si]['zs'] += a
        if 'gain' in blk and blk['gain'] != g_tgt:
            g_tgt = blk['gain']
            g_inc = (g_tgt - g_cur) / ramp_n
            g_left = ramp_n
        for _t in range(BLOCK):
            if r_left > 0:
                r_left -= 1
                r_cur = r_tgt if r_left == 0 else r_cur + r_inc
            if g_left > 0:
                g_left -= 1
                g_cur = g_tgt if g_left == 0 else g_cur + g_inc
            L = 0.0
            R = 0.0
            for st in S:
                n = len(st['re'])
                if st['wleft'] > 0:
                    st['wleft'] -= 1
                    if st['wleft'] == 0:
                        st['w'] = list(st['wtgt'])
                    else:
                        st['w'] = [st['w'][j] + st['winc'][j] for j in range(n)]
                if st['pleft'] > 0:
                    st['pleft'] -= 1
                    if st['pleft'] == 0:
                        st['L'], st['R'] = st['Ltgt'], st['Rtgt']
                    else:
                        st['L'] = st['L'] + st['Linc']
                        st['R'] = st['R'] + st['Rinc']
                p = strength * ((1.0 - qf) * st['zf'] - (1.0 - qs) * st['zs']) / (qs - qf)
                st['zf'] = st['zf'] * qf
                st['zs'] = st['zs'] * qs
                acc = 0.0
                for j in range(n):
                    re, im, c, sn = st['re'][j], st['im'][j], st['c'][j], st['s'][j]
                    nre = r_cur * (c * re - sn * im) + p if j < st['nd'] else r_cur * (c * re - sn * im)
                    nim = r_cur * (sn * re + c * im)
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


def bare_engine(gain=GAIN):
    """An engine on an empty field (no figures): slots are set up by hand."""
    e = registry.create(orz.ENGINE_ID, CTX, dict(detector=1, frequency_scale=220.0, decay_s=0.8))
    e.init(np.zeros((32, 32), np.uint8), None, gain)
    return e


def setup_slot(e, s, freqs, ndrive, pan_p, zre=None, zim=None):
    n = len(freqs)
    e.role[s] = orz.ROLE_ACTIVE
    e.slot_id[s] = 1000 + s
    e.ndrive[s] = ndrive
    e.nlive[s] = n
    e.sqrtlam[s, :n] = np.asarray(freqs) / 220.0
    e.ffreq[s, :n] = 220.0 * e.sqrtlam[s, :n]
    c, sn = orz.trig_of(e.ffreq[s, :n], e.sr)
    e.cth[s, :n] = c
    e.sth[s, :n] = sn
    w = np.zeros(orz.N_BANK)
    w[:n] = 1.0 / ndrive if ndrive else 0.0
    e.wcur[s] = w
    e.wtgt[s] = w
    e.pan[s] = (math.cos(math.pi * pan_p / 2), 0.0, math.cos(math.pi * pan_p / 2),
                math.sin(math.pi * pan_p / 2), 0.0, math.sin(math.pi * pan_p / 2))
    if zre is not None:
        e.zre[s, :n] = zre
        e.zim[s, :n] = zim


# ======================================================================================
class GeometryAndSpectrumTests(unittest.TestCase):
    def test_blinker_spectrum_and_invariance_across_the_seam(self):
        g = grid(blinker())
        comps = fg.components(g)
        self.assertEqual(len(comps), 1)
        lam = np.linalg.eigvalsh(fg.laplacian_matrix(comps[0], 32, 32))
        np.testing.assert_allclose(lam, [0.0, 1.0, 3.0], atol=1e-12)
        sq = fg.spectrum(comps[0], 32, 32)
        np.testing.assert_allclose(sq, [1.0, math.sqrt(3.0)], atol=1e-12)
        ref_key = fg.shape_key(comps[0], 32, 32)
        for cells in (blinker(9, 0), blinker(0, 15), blinker(31, 31), [(8, 15), (9, 15), (10, 15)],
                      [(31, 5), (0, 5), (1, 5)], [(5, 31), (5, 0), (5, 1)]):
            c = fg.components(grid(cells))
            self.assertEqual(len(c), 1, cells)
            np.testing.assert_allclose(fg.spectrum(c[0], 32, 32), sq, atol=1e-12)
            if cells[0][0] == cells[1][0]:                      # horizontal: same key
                self.assertEqual(fg.shape_key(c[0], 32, 32), ref_key)
        # diagonal pair across the corner: one component, one mode sqrt(2)
        c = fg.components(grid([(31, 31), (0, 0)]))
        self.assertEqual(len(c), 1)
        np.testing.assert_allclose(fg.spectrum(c[0], 32, 32), [math.sqrt(2.0)], atol=1e-12)
        self.assertEqual(fg.centre_of(c[0], 32, 32), (31.5, 31.5))
        # rotation / reflection of a larger asymmetric shape
        base = [(10, 10), (10, 11), (10, 12), (11, 10), (12, 11), (13, 12), (13, 13)]
        sq0 = fg.spectrum(fg.components(grid(base))[0], 32, 32)
        for tf in (lambda r, c: (c, r), lambda r, c: (r, -c), lambda r, c: (-r, c), lambda r, c: (c, -r)):
            cells = [tf(r, c) for r, c in base]
            cells = [(r + 20, c + 20) for r, c in cells]
            np.testing.assert_allclose(fg.spectrum(fg.components(grid(cells))[0], 32, 32), sq0, atol=1e-10)

    def test_glider_has_two_spectral_phases_repeating_every_other_generation(self):
        g = grid(glider_cells())
        specs = []
        for _ in range(8):
            c = fg.components(g)
            self.assertEqual(len(c), 1)
            self.assertEqual(len(c[0]), 5)
            specs.append(fg.spectrum(c[0], 32, 32))
            g = step(g)
        for k in range(8):
            self.assertEqual(len(specs[k]), 4)
            np.testing.assert_allclose(specs[k], specs[k % 2], atol=1e-10)
        self.assertGreater(float(np.abs(specs[0] - specs[1]).max()), 0.05)

    def test_nonperiodic_ratios_equal_the_core_laplacian(self):
        from casynth_core import map_laplacian
        for cells in (glider_cells(), receiver_cells(), block5()):
            comps = fg.components(grid(cells))
            self.assertEqual(len(comps), 1)
            comp = comps[0]
            sq = fg.spectrum(comp, 32, 32)
            n = len(sq)
            freqs, _amps = map_laplacian(grid(cells).astype(float), 100.0, n=n, harm=0.0, fullshape=True)
            freqs = np.asarray(freqs)[:n]
            np.testing.assert_allclose(freqs / freqs[0], sq / sq[0], rtol=1e-9)

    def test_centre_radius_disk_and_the_req_numbers(self):
        cells = neighbor_cells()
        comps = fg.components(grid(cells))
        self.assertEqual([len(c) for c in comps], [17, 3])
        big = comps[0]
        cy, cx = fg.centre_of(big, 32, 32)
        self.assertAlmostEqual(cy, 11.7647058824, places=9)
        self.assertAlmostEqual(cx, 11.7647058824, places=9)
        R = fg.radius_of(big, (cy, cx), 32, 32)
        self.assertAlmostEqual(R, 5.32962, places=5)
        mask = fg.disk_mask((cy, cx), R, 32, 32)
        self.assertTrue(mask[big[:, 0], big[:, 1]].all())          # the circle holds its own cells
        self.assertTrue(mask[comps[1][:, 0], comps[1][:, 1]].all())   # and the blinker
        self.assertFalse(mask[0, 0])
        far = np.nonzero(~mask)
        dr = fg.torus_delta(far[0], cy, 32)
        dc = fg.torus_delta(far[1], cx, 32)
        self.assertTrue((np.hypot(dr, dc) > R).all())
        # the cell that defines R sits exactly on the boundary and is inside
        self.assertEqual(int(np.count_nonzero(mask)), 92)
        # own mask = the cells
        own = fg.own_mask(big, 32, 32)
        self.assertEqual(int(own.sum()), 17)
        self.assertTrue(own[big[:, 0], big[:, 1]].all())
        # single cell: R = 0, disk = itself, no modes
        one = fg.components(grid([(4, 4)]))[0]
        self.assertEqual(fg.centre_of(one, 32, 32), (4.0, 4.0))
        self.assertEqual(fg.radius_of(one, (4.0, 4.0), 32, 32), 0.0)
        self.assertEqual(int(fg.disk_mask((4.0, 4.0), 0.0, 32, 32).sum()), 1)
        self.assertEqual(len(fg.spectrum(one, 32, 32)), 0)
        # the seam glider from the preflight: centre in the unwrapped sense, R < 2
        g = grid(glider_cells())
        for _ in range(40):
            g = step(g)
        c = fg.components(g)[0]
        cy, cx = fg.centre_of(c, 32, 32)
        dr = fg.torus_delta(c[:, 0], cy, 32)
        dc = fg.torus_delta(c[:, 1], cx, 32)
        self.assertAlmostEqual(float(dr.sum()), 0.0, places=9)   # the unwrapped mean
        self.assertAlmostEqual(float(dc.sum()), 0.0, places=9)
        self.assertLess(fg.radius_of(c, (cy, cx), 32, 32), 2.0)

    def test_ambiguous_centre_of_a_torus_spanning_figure(self):
        ring = fg.components(grid([(5, c) for c in range(32)]))[0]
        self.assertEqual(len(ring), 32)
        self.assertEqual(fg.centre_of(ring, 32, 32), (5.0, 0.5))               # new: smallest
        self.assertEqual(fg.centre_of(ring, 32, 32, prev=(5.0, 17.2)), (5.0, 17.5))   # nearest to prev
        self.assertEqual(fg.centre_of(ring, 32, 32, prev=(5.0, 31.9)), (5.0, 31.5))
        R = fg.radius_of(ring, (5.0, 0.5), 32, 32)
        self.assertEqual(R, 15.5)
        self.assertEqual(len(fg.spectrum(ring, 32, 32)), 24)
        self.assertEqual(fg.periodic_mean([28, 29, 30], 32), 29.0)
        self.assertEqual(fg.periodic_mean([31, 0, 1], 32), 0.0)
        self.assertEqual(fg.periodic_mean([30, 31, 0], 32), 31.0)
        self.assertEqual(fg.periodic_mean([0, 16], 32), 8.0)                  # two minima: smallest
        self.assertEqual(fg.periodic_mean([0, 16], 32, prev=25.0), 24.0)

    def test_matching_rules(self):
        rows = cols = 32
        # the glider keeps its identity while moving (including across the seam)
        g = grid(glider_cells())
        prev = fg.components(g)
        prev_c = [fg.centre_of(c, rows, cols) for c in prev]
        for _ in range(140):
            g = step(g)
            comps = fg.components(g)
            cont, tails, new = fg.match(prev, prev_c, comps, rows, cols)
            self.assertEqual((cont, tails, new), ({0: 0}, [], []))
            prev, prev_c = comps, [fg.centre_of(c, rows, cols, prev_c[0]) for c in comps]
        # one-cell shift of a single cell (no overlap): continued by displacement
        cont, tails, new = fg.match([np.array([[5, 5]])], [(5.0, 5.0)], [np.array([[5, 6]])], rows, cols)
        self.assertEqual(cont, {0: 0})
        cont, tails, new = fg.match([np.array([[5, 5]])], [(5.0, 5.0)], [np.array([[5, 8]])], rows, cols)
        self.assertEqual((cont, tails, new), ({}, [0], [0]))
        # area change (a cell added) keeps the identity; a big newcomer elsewhere is new
        prev = fg.components(grid(blinker()))
        prev_c = [fg.centre_of(c, rows, cols) for c in prev]
        comps = fg.components(grid(blinker() + [(10, 15)] + block5()))
        cont, tails, new = fg.match(prev, prev_c, comps, rows, cols)
        self.assertEqual(len(comps), 2)
        self.assertEqual(tails, [])
        self.assertEqual(len(cont), 1)
        self.assertEqual(len(new), 1)
        self.assertEqual(len(comps[next(iter(cont))]), 4)
        # merge: two figures become one component -> both to tails, one new
        prev = fg.components(grid(blinker(9, 15) + blinker(9, 20)))
        prev_c = [fg.centre_of(c, rows, cols) for c in prev]
        comps = fg.components(grid([(9, c) for c in range(14, 22)]))
        self.assertEqual((len(prev), len(comps)), (2, 1))
        cont, tails, new = fg.match(prev, prev_c, comps, rows, cols)
        self.assertEqual((cont, tails, new), ({}, [0, 1], [0]))
        # split: one figure into two components -> tail + two new
        cont, tails, new = fg.match(comps, [fg.centre_of(comps[0], rows, cols)], prev, rows, cols)
        self.assertEqual((cont, tails, new), ({}, [0], [0, 1]))

    def test_spectrum_is_bit_identical_for_every_placement(self):
        # the glider repeats its shape every 4 generations, translated by (1, 1): the
        # canonical placement, the matrix and the eigenvalues are then identical bit
        # for bit anywhere on the torus (phases 2 apart are mirror images: equal only
        # up to rounding)
        g = grid(glider_cells())
        phases = []
        for k in range(132):
            c = fg.components(g)[0]
            phases.append((fg.canonical_cells(c, 32, 32), fg.spectrum(c, 32, 32)))
            g = step(g)
        for k in range(4, 132):
            np.testing.assert_array_equal(phases[k][0], phases[k - 4][0])
            np.testing.assert_array_equal(phases[k][1], phases[k - 4][1])
            np.testing.assert_allclose(phases[k][1], phases[k - 2][1], rtol=1e-12)

    def test_spectrum_cache_never_changes_a_result(self):
        cache = fg.SpectrumCache(24)
        a = fg.components(grid(glider_cells()))[0]
        b = fg.components(grid([(r - 20, c - 20) for r, c in glider_cells()]))[0]
        s1 = cache.get(a, 32, 32)
        s2 = cache.get(b, 32, 32)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(s1, fg.spectrum(a, 32, 32))
        s2[:] = 0.0
        np.testing.assert_array_equal(cache.get(a, 32, 32), s1)


# ======================================================================================
class ExcitationTests(unittest.TestCase):
    def test_zero_from_zero_initial_packet_and_no_packet_without_a_change(self):
        e = engine([])
        y = run(e, 20)
        self.assertEqual(float(np.abs(y).max()), 0.0)                         # exact zero
        self.assertEqual(e.display()['n_figures'], 0)
        e = engine(blinker())
        y0 = run(e, 1)
        d = e.display()
        self.assertEqual((d['n_figures'], d['n_sounding']), (1, 1))
        self.assertEqual(d['figures'][0]['e'], 3.0)                            # 3 births in its detector
        self.assertAlmostEqual(d['figures'][0]['a'], 3.0 / 5.0)
        self.assertGreater(float(np.abs(y0).max()), 0.0)
        zf_after = float(e.zf[0])
        # unchanged field, parameter changes, update_field with the same field: no packet
        e.update_field(grid(blinker()), None)
        e.set_params(dict(detector=0, radius_mul=2.0, frequency_scale=330.0, decay_s=1.2))
        run(e, 5)
        self.assertLess(float(e.zf[0]), zf_after)
        self.assertEqual(e.display()['figures'][0]['e'], 3.0)                 # the last packet shown
        self.assertEqual(int(e.counters[3]), 1)                                # one field change seen
        # decay afterwards
        y = run(e, int(3.0 * SR / BLOCK))
        first = float(np.abs(y[:BLOCK]).max())
        last = float(np.abs(y[-BLOCK:]).max())
        self.assertGreater(first, 0.0)
        self.assertLess(last, first * 1e-3)

    def test_births_and_deaths_on_old_and_new_geometry_blinker(self):
        for det in (0, 1):
            e = engine(blinker(), detector=det)
            run(e, 1)
            g = step(grid(blinker()))
            e.update_field(g, None)
            run(e, 1)
            f = e.display()['figures'][0]
            self.assertEqual(f['id'], 1)
            self.assertEqual(f['e'], 4.0)                # 2 births (new mask) + 2 deaths (old mask)
            self.assertAlmostEqual(f['a'], 4.0 / 6.0)
            # only the deaths of the old geometry count when the field is emptied: the
            # figure leaves -> no packet, a tail
            e.update_field(np.zeros((32, 32), np.uint8), None)
            run(e, 1)
            d = e.display()
            self.assertEqual((d['n_figures'], d['n_sounding'], d['n_tails']), (0, 0, 1))
            self.assertEqual(int(e.role[0]), orz.ROLE_FREE)

    def test_n42_receiver_own_none_disk_four_outside_zero(self):
        cells = neighbor_cells()
        for det, want in ((0, 0.0), (1, 4.0)):
            e = engine(cells, detector=det)
            g = grid(cells)
            run(e, 1)
            d = e.display()
            self.assertEqual([(f['id'], f['n'], f['modes']) for f in d['figures']], [(1, 17, 16), (2, 3, 2)])
            self.assertEqual(d['figures'][0]['e'], 17.0 if det == 0 else 20.0)   # start packet
            self.assertEqual(d['figures'][1]['e'], 3.0)
            for _ in range(12):
                g = step(g)
                e.update_field(g, None)
                run(e, 1)
                recv, blk = fig_by_id(e, 1), fig_by_id(e, 2)
                self.assertEqual(recv['e'], want)
                self.assertEqual(blk['e'], 4.0)
                self.assertEqual(recv['n'], 17)
                self.assertEqual(recv['modes'], 16)
            # the blinker moved outside the circle: the old blinker -> tail, a new figure;
            # the receiver sees the deaths of the old one (Disk) once, then nothing
            far = grid(receiver_cells() + [(9, 24), (9, 25), (9, 26)])
            e.update_field(far, None)
            run(e, 1)
            d = e.display()
            self.assertEqual(d['n_tails'], 1)
            self.assertEqual([f['id'] for f in d['figures']], [1, 3])
            self.assertEqual(fig_by_id(e, 1)['e'], 3.0 if det == 1 else 0.0)
            for _ in range(6):
                far = step(far)
                e.update_field(far, None)
                run(e, 1)
                self.assertEqual(fig_by_id(e, 1)['e'], 0.0)
                self.assertEqual(fig_by_id(e, 3)['e'], 4.0)

    def test_several_edits_in_one_block_merge_and_the_detector_switch_is_no_hit(self):
        e = engine(blinker())
        run(e, 1)
        g = grid(blinker())
        g[20, 20] = 1
        g[20, 20] = 0                                    # edited and undone before the block
        e.update_field(g, None)
        zf = float(e.zf[0])
        run(e, 1)
        self.assertEqual(int(e.counters[3]), 1)
        self.assertLess(float(e.zf[0]), zf)
        e.set_params(dict(detector=0, radius_mul=1.0, frequency_scale=220.0, decay_s=0.8))
        e.set_params(dict(detector=1, radius_mul=3.0, frequency_scale=220.0, decay_s=0.8))
        zf = float(e.zf[0])
        run(e, 1)
        self.assertLess(float(e.zf[0]), zf)

    def test_radius_multiplier_scales_the_disk_only_and_is_no_hit(self):
        cells = neighbor_cells()
        # x0.5: the receiver's circle (R 5.33 -> 2.66) no longer holds the blinker -> Disk hears 0
        e = engine(cells, detector=1, radius_mul=0.5)
        g = grid(cells)
        run(e, 1)
        recv = fig_by_id(e, 1)
        self.assertAlmostEqual(recv['radius'], 0.5 * 5.32962, places=4)
        self.assertAlmostEqual(recv['radius_geom'], 5.32962, places=4)
        inside = fg.disk_mask(tuple(recv['centre']), recv['radius'], 32, 32)
        own_inside = int(inside[grid(receiver_cells()) == 1].sum())
        self.assertTrue(0 < own_inside < 17)                 # a shrunk circle no longer holds all own cells
        self.assertEqual(recv['e'], float(own_inside))       # start packet: births inside the circle only
        for _ in range(6):
            g = step(g)
            e.update_field(g, None)
            run(e, 1)
            self.assertEqual(fig_by_id(e, 1)['e'], 0.0)      # the blinker is outside the shrunk circle
            self.assertEqual(fig_by_id(e, 2)['e'], 0.0)      # blinker R 1 -> 0.5: its births / deaths at distance 1 fall outside
        # x3: the blinker's circle (R 1 -> 3) reaches the still receiver, which has no changes:
        # still 0 for the receiver's sake; the receiver's circle x3 hears the blinker as before
        e = engine(cells, detector=1, radius_mul=3.0)
        g = grid(cells)
        run(e, 1)
        for _ in range(4):
            g = step(g)
            e.update_field(g, None)
            run(e, 1)
            self.assertEqual(fig_by_id(e, 1)['e'], 4.0)
            self.assertEqual(fig_by_id(e, 2)['e'], 4.0)
        # Own ignores the multiplier; a single cell keeps R = 0; the knob itself is no hit
        e = engine(cells + [(25, 25)], detector=0, radius_mul=4.0)
        run(e, 1)
        one = [f for f in e.display()['figures'] if f['n'] == 1][0]
        self.assertEqual((one['radius'], one['radius_geom'], one['slot']), (0.0, 0.0, -1))
        self.assertEqual(fig_by_id(e, 1)['e'], 17.0)
        zf = e.zf.copy()
        e.set_params(dict(e.params, radius_mul=0.25))
        run(e, 1)
        self.assertTrue((e.zf <= zf).all())
        self.assertEqual(e.display()['radius_mul'], 0.25)
        # a change of the multiplier alone never makes an event on still cells (Disk)
        e = engine(receiver_cells(), detector=1)
        run(e, 2)
        for mul in (0.5, 4.0, 1.0):
            e.set_params(dict(e.params, radius_mul=mul))
            zf = float(e.zf[0])
            run(e, 1)
            self.assertLess(float(e.zf[0]), zf)
            self.assertEqual(int(e.counters[3]), 1)


# ======================================================================================
class ScalarReferenceTests(unittest.TestCase):
    def test_kernel_matches_the_scalar_reference_through_every_ramp(self):
        freqs_a = [220.0, 381.05, 512.3, 777.7, 1234.5]
        freqs_b = [80.8, 130.1, 190.2]
        e = bare_engine()
        setup_slot(e, 0, freqs_a, 5, 0.3)
        setup_slot(e, 1, freqs_b + [640.0, 990.0], 3, 0.9,
                   zre=[0.0, 0.0, 0.0, 0.02, -0.01], zim=[0.0, 0.0, 0.0, 0.01, 0.03])   # ringing extras
        slots = [dict(freqs=freqs_a, weights=[0.2] * 5, pan_p=0.3, ndrive=5),
                 dict(freqs=freqs_b + [640.0, 990.0], weights=[1 / 3] * 3 + [0.0, 0.0], pan_p=0.9,
                      ndrive=3, zre=[0.0, 0.0, 0.0, 0.02, -0.01], zim=[0.0, 0.0, 0.0, 0.01, 0.03])]
        # the ringing extras of slot 1 keep a weight (they rang with 1/5 before): set both
        e.wcur[1, 3:5] = 0.2
        e.wtgt[1, 3:5] = 0.2
        slots[1]['weights'] = [1 / 3] * 3 + [0.2, 0.2]
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
            y, _pk, _nc = e.render_float(blk.get('gain', e.gg[orz.R_TGT]))
            got.append(y)
        got = np.concatenate(got)
        self.assertEqual(got.shape, ref.shape)
        self.assertGreater(float(np.abs(ref).max()), 1e-4)
        self.assertLessEqual(float(np.abs(got - ref).max()), TOL)
        self.assertEqual(int(e.ndrive[0]), 6)
        # v2: the vanished DRIVEN mode of slot 1 (3 -> 2) rings on in a tail slot of its
        # own; the two ringing extras (never driven) faded in place and were zeroed
        self.assertEqual((int(e.nlive[1]), int(e.ndrive[1])), (2, 2))
        tails = [s for s in range(orz.N_ACTIVE, orz.N_ACTIVE + orz.N_TAILS) if e.role[s] == orz.ROLE_TAIL]
        self.assertEqual(len(tails), 1)
        self.assertEqual(int(e.nlive[tails[0]]), 1)
        self.assertEqual(float(e.ffreq[tails[0], 0]), 190.2)

    def test_decay_values_gain_ramp_and_r_ramp_bookkeeping(self):
        self.assertAlmostEqual(orz.decay_r(0.8), 10 ** (-3 / (SR * 0.8)), places=15)
        e = bare_engine(gain=0.01)
        self.assertEqual(list(e.gg), [0.01, 0.01, 0.0])
        e.render_float(0.02)
        self.assertEqual(int(e.ints[orz.I_G_LEFT]), 882 - BLOCK)
        e.render_float(0.02)
        e.render_float(0.02)
        self.assertEqual(int(e.ints[orz.I_G_LEFT]), 0)
        self.assertEqual(float(e.gg[orz.R_CUR]), 0.02)
        e.set_params(dict(e.params, decay_s=0.2))
        self.assertEqual(int(e.ints[orz.I_R_LEFT]), 882)
        run(e, 3)
        self.assertEqual(float(e.rr[orz.R_CUR]), orz.decay_r(0.2))
        self.assertEqual(orz.pan_of(0.0, 32), (1.0, 0.0))
        L, R = orz.pan_of(31.0, 32)
        self.assertAlmostEqual(L, 0.0, places=15)
        self.assertEqual(R, 1.0)
        self.assertEqual(orz.pan_of(31.5, 32), orz.pan_of(31.0, 32))            # clipped


# ======================================================================================
class LifeCycleTests(unittest.TestCase):
    def test_translation_growth_and_tumbler_split_merge_keep_or_retire_identities(self):
        e = engine(glider_cells())
        g = grid(glider_cells())
        ids = set()

        def on_block(b, gen, eng):
            d = eng.display()
            ids.update(f['id'] for f in d['figures'])
            self.assertEqual(d['n_sounding'], 1)
        evolve(e, g, 12.0, rate=16.0, on_block=on_block)
        self.assertEqual(ids, {1})                                              # the glider keeps its id
        # growth: cells painted next to it keep the identity, modes grow, weights ramp
        e = engine(blinker())
        run(e, 2)
        e.update_field(grid(blinker() + [(10, 15), (10, 16)]), None)
        run(e, 1)
        f = e.display()['figures'][0]
        self.assertEqual((f['id'], f['n'], f['modes']), (1, 5, 4))
        self.assertGreater(int(e.wleft[0]), 0)
        self.assertEqual(f['e'], 2.0)
        # tumbler: period 14 with splits / merges -> retired identities become tails,
        # everything finite, every block sounds
        e = engine(tumbler())
        y, _g = evolve(e, grid(tumbler()), 6.0)
        self.assertTrue(np.all(np.isfinite(y)))
        d = e.display()
        self.assertGreater(e.next_id, 2)
        self.assertGreaterEqual(d['n_figures'], 1)
        self.assertEqual(d['drops'], 0)

    def test_clear_all_and_quick_re_add_tails_release(self):
        e = engine(neighbor_cells())
        run(e, 3)
        e.update_field(np.zeros((32, 32), np.uint8), None)
        run(e, 1)
        d = e.display()
        self.assertEqual((d['n_figures'], d['n_tails']), (0, 2))
        self.assertEqual(int(np.count_nonzero(e.role == orz.ROLE_ACTIVE)), 0)
        e.update_field(grid(neighbor_cells()), None)
        run(e, 1)
        d = e.display()
        self.assertEqual([f['id'] for f in d['figures']], [3, 4])            # new identities
        self.assertEqual(d['n_tails'], 2)
        self.assertEqual(d['figures'][0]['e'], 20.0)                            # a fresh start packet
        e.update_field(np.zeros((32, 32), np.uint8), None)
        y = run(e, int(3.0 * SR / BLOCK))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertEqual(e.display()['n_tails'], 0)                             # released after decay
        self.assertEqual(int(np.count_nonzero(e.role)), 0)
        self.assertEqual(float(np.abs(e.zre).max()), 0.0)

    def test_limits_are_visible_and_never_silent(self):
        many = []
        for r in range(0, 32, 4):
            for c in range(0, 32, 5):
                many += blinker(r + 1, c + 1)
        g = grid(many)
        n_fig = len(fg.components(g))
        self.assertGreater(n_fig, orz.N_ACTIVE)
        e = engine(many)
        run(e, 1)
        d = e.display()
        self.assertEqual((d['n_figures'], d['n_sounding']), (n_fig, orz.N_ACTIVE))
        self.assertGreater(d['unvoiced_blocks'], 0)
        # clear / re-add cycles pile tails beyond 96: the quietest fade (counted)
        for k in range(6):
            e.update_field(np.zeros((32, 32), np.uint8), None)
            run(e, 1)
            e.update_field(g, None)
            run(e, 1)
        d = e.display()
        self.assertEqual(d['n_tails'], orz.N_TAILS)
        self.assertGreater(d['evictions'], 0)
        y = run(e, 30)
        self.assertTrue(np.all(np.isfinite(y)))
        # a freed slot goes to a waiting figure (largest first) without a packet
        e = engine(many)
        run(e, 1)
        waiting = [f for f in e.display()['figures'] if f['slot'] < 0]
        self.assertTrue(waiting)
        g2 = g.copy()
        first = e.display()['figures'][0]
        for r, c in first['cells']:
            g2[r, c] = 0
        e.update_field(g2, None)
        run(e, 1)
        d = e.display()
        self.assertEqual(d['n_sounding'], orz.N_ACTIVE)
        got = fig_by_id(e, waiting[0]['id'])
        self.assertGreaterEqual(got['slot'], 0)
        self.assertEqual(got['e'], 0.0)

    def test_full_toggle_every_block_dense_field_and_knob_ends_are_finite(self):
        rng = np.random.default_rng(7)
        dense = (rng.random((32, 32)) < 0.5).astype(np.uint8)
        for det in (0, 1):
            for scale, decay in ((55.0, 0.2), (880.0, 1.5)):
                e = registry.create(orz.ENGINE_ID, CTX, dict(detector=det, frequency_scale=scale, decay_s=decay))
                e.init(dense, None, MASTER_GAIN)
                peak = 0.0
                g = dense
                for b in range(60):
                    g = np.zeros_like(dense) if b % 2 else dense
                    e.update_field(g, None)
                    y, pk, _nc = e.render_float(MASTER_GAIN)
                    self.assertTrue(np.all(np.isfinite(y)))
                    peak = max(peak, pk)
                self.assertLess(peak, 4.0)                              # finite, bounded (stress, not a scene)
                self.assertEqual(int(e.counters[1]), 0)


# ======================================================================================
class SnapshotTests(unittest.TestCase):
    def test_snapshot_mid_tail_mid_ramps_restores_exactly_without_a_packet(self):
        os.makedirs(ART, exist_ok=True)
        e = engine(neighbor_cells())
        g = grid(neighbor_cells())
        _y, g = evolve(e, g, 1.0)
        e.set_params(dict(e.params, decay_s=1.1))                       # r ramp in flight
        g2 = g.copy()
        g2[9, 14:17] = 0                                                 # the blinker gone -> a tail
        g2[20, 20:23] = 1                                                # a new blinker
        e.update_field(g2, None)
        run(e, 1)
        e.update_field(step(g2), None)
        run(e, 1)                                                        # weight / pan ramps in flight
        g3 = step(step(g2))
        e.update_field(g3, None)                                         # pending, not rendered yet
        self.assertTrue(supports_snapshot(e))
        st = e.export_state()
        self.assertGreater(int(st['ints'][orz.I_R_LEFT]), 0)
        self.assertEqual(int(np.count_nonzero(st['role'] == orz.ROLE_TAIL)), 1)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            save_state(tmp, 'n4', st)
            back = load_state(tmp, 'n4')
        other = registry.create(orz.ENGINE_ID, CTX, dict(detector=1, frequency_scale=220.0, decay_s=0.8))
        other.init(np.zeros((32, 32), np.uint8), None, 0.0)
        self.assertEqual(other.params['radius_mul'], 1.0)          # optional parameter filled in
        other.restore_state(g3, None, back)
        # a snapshot written before the radius knob existed restores with the default 1.0
        old = dict(back)
        old['params'] = {k: v for k, v in back['params'].items() if k != 'radius_mul'}
        older = registry.create(orz.ENGINE_ID, CTX, registry.defaults(orz.ENGINE_ID))
        older.init(np.zeros((32, 32), np.uint8), None, 0.0)
        older.restore_state(g3, None, old)
        self.assertEqual(older.params, other.params)
        self.assertEqual(other.params, e.params)
        self.assertEqual(sorted(other.figures), sorted(e.figures))
        gg = g3
        for _ in range(30):
            a, _p1, _c1 = e.render_float(GAIN)
            b, _p2, _c2 = other.render_float(GAIN)
            np.testing.assert_array_equal(a, b)
            gg = step(gg)
            e.update_field(gg, None)
            other.update_field(gg, None)
        self.assertEqual(e.display(), other.display())
        # restore rejects: wrong model, tampered spectrum, a foreign pending field
        bad = dict(back)
        bad['model_version'] = 'x'
        with self.assertRaises(ValueError):
            other.restore_state(g3, None, bad)
        bad = dict(back)
        bad['sqrtlam'] = back['sqrtlam'].copy()
        bad['sqrtlam'][0, 0] += 1e-3
        bad['ffreq'] = 220.0 * bad['sqrtlam']
        with self.assertRaises(ValueError):
            other.restore_state(g3, None, bad)
        with self.assertRaises(ValueError):
            other.restore_state(step(g3), None, back)

    def test_runner_continue_equals_the_continuous_run_for_both_scenes(self):
        from casynth_lab import DemoRunner, load_scene
        for sc in ('n4_spectrum', 'n4_neighbor'):
            scene = load_scene(os.path.join(ROOT, 'demos', sc + '.json'))
            runner = DemoRunner(scene)
            runner.post('start', at=0)
            runner.post('set_param', at=int(0.7 * SR), side='B', name='decay_s', value=1.2)
            runner.post('set_param', at=int(0.9 * SR), side='B', name='frequency_scale', value=330.0)
            runner.post('set_param', at=int(0.95 * SR), side='B', name='radius_mul', value=2.5)
            for _ in range(int(1.0 * SR / BLOCK)):
                runner.next_block()
            st = runner.export_state()
            with tempfile.TemporaryDirectory(dir=ART) as tmp:
                save_state(tmp, 'runner', st)
                back = load_state(tmp, 'runner')
            twin = DemoRunner.from_state(back)
            for _ in range(60):
                a, b = runner.next_block(), twin.next_block()
                for s in ('A', 'B', 'monitor'):
                    np.testing.assert_array_equal(a.get(s), b.get(s))
            self.assertEqual(runner.snapshot()['display']['B'], twin.snapshot()['display']['B'])


# ======================================================================================
class LevelsTimingAndContractTests(unittest.TestCase):
    def test_scene_levels_match_the_preflight_and_stay_far_from_the_clip(self):
        pf = preflight()
        for name, det in (('glider', 1), ('neighbors', 0), ('neighbors', 1)):
            cells = [tuple(c) for c in pf['cases'][name]['cells']]
            e = engine(cells, detector=det)
            y, _g = evolve(e, grid(cells), 12.0)
            rms = 20 * math.log10(float(np.sqrt(np.mean(y ** 2))))
            ref = pf['cases'][name]['modes']['own' if det == 0 else 'disk']
            self.assertLess(abs(rms - ref['rms_at_scale_0p5_gain_0p028_db']), 0.1)
            self.assertLess(abs(float(np.abs(y).max()) - ref['peak_at_scale_0p5_gain_0p028']), 1e-3)
            self.assertLess(float(np.abs(y).max()), 0.1)

    def test_engine_timing_budget_on_both_scenes(self):
        from casynth_lab import DemoRunner, load_scene
        budget_ms = BLOCK / SR * 1000.0
        for sc in ('n4_spectrum', 'n4_neighbor'):
            scene = load_scene(os.path.join(ROOT, 'demos', sc + '.json'))
            runner = DemoRunner(scene)
            runner.post('start', at=0)
            times = []
            for b in range(int(6.0 * SR / BLOCK)):
                t0 = time.perf_counter()
                runner.next_block()
                times.append((time.perf_counter() - t0) * 1000.0)
            warm = np.array(times[60:])
            self.assertLess(float(np.percentile(warm, 99)), budget_ms, sc)

    def test_rejects_other_contexts_registry_hints_and_display(self):
        for ctx in (EngineContext(48000, BLOCK, 2, 110.0, 1.0, 6.0), EngineContext(SR, 256, 2, 110.0, 1.0, 6.0),
                    EngineContext(SR, BLOCK, 1, 110.0, 1.0, 6.0)):
            with self.assertRaises(ValueError):
                registry.create(orz.ENGINE_ID, ctx, registry.defaults(orz.ENGINE_ID))
        spec = registry.get(orz.ENGINE_ID)
        self.assertEqual(spec.defaults(), dict(detector=1, radius_mul=1.0, spectrum=1, frequency_scale=220.0,
                                               decay_s=0.8, attack_ms=0.0, events=0, excitation=0, birth_strength=1.0, decay_law=0, **orz.laplace_settings({})))
        self.assertEqual(registry.value_text(orz.ENGINE_ID, 'detector', 0), 'Own')
        self.assertEqual(registry.value_text(orz.ENGINE_ID, 'detector', 1), 'Disk')
        with self.assertRaises(ValueError):
            registry.validate_param(orz.ENGINE_ID, 'frequency_scale', 1000.0)
        ov = spec.overlay(dict(detector=0), 32, 32)
        self.assertIn('[Own, Figure]', ov['text'])
        self.assertNotIn('circles', ov)
        e = engine(neighbor_cells())
        run(e, 1)
        d = e.display()
        for key in ('figures', 'n_figures', 'n_sounding', 'n_tails', 'detector', 'detector_name',
                    'frequency_scale', 'decay_s', 'evictions', 'drops', 'model'):
            self.assertIn(key, d)
        f = d['figures'][0]
        for key in ('id', 'color', 'slot', 'n', 'cells', 'centre', 'radius', 'modes', 'f_low', 'e', 'a', 'level'):
            self.assertIn(key, f)
        self.assertEqual(len(f['cells']), 17)
        self.assertAlmostEqual(f['radius'], 5.32962, places=5)
        self.assertAlmostEqual(f['f_low'], 220.0 * float(fg.spectrum(np.array(f['cells']), 32, 32)[0]))
        self.assertEqual(d['model'], orz.MODEL_VERSION)
        json.dumps(d)                                                   # plain data only

    def test_existing_records_of_n3_and_laplace_still_replay(self):
        """Item 1 of the acceptance: saved N3 / Laplace records replay exactly with the
        current code (the catalogs are in git; skipped when they are absent)."""
        from casynth_lab.catalog import Catalog
        roots = [os.path.join(ROOT, 'lab_catalog', 'network_n3_combined_2026_09_16'),
                 os.path.join(ROOT, 'tests', 's7_demo_catalog.py')]
        n3 = roots[0]
        if not os.path.isdir(n3):
            self.skipTest('N3 catalog not present')
        cat = Catalog(n3, repo_root=None)
        entries = [r for r, err in cat.list() if err is None]
        self.assertTrue(entries)
        rec = entries[-1]
        if not os.path.isfile(os.path.join(n3, rec.id, 'monitor.wav')):
            self.skipTest('N3 audio not present (audio is not in git)')
        result = cat.replay(rec.id, yield_cpu=False)
        self.assertEqual(result.status, 'match', result.reason)


# ======================================================================================
class BenchTests(unittest.TestCase):
    def test_headless_draw_figures_overlay_and_panel_for_both_scenes(self):
        import pygame
        import demo_bench as db
        from casynth_lab import DemoRunner, load_scene
        from casynth_lab.audio_out import LiveEngine
        for sc, want_a in (('n4_spectrum', False), ('n4_neighbor', True)):
            scene = load_scene(os.path.join(ROOT, 'demos', sc + '.json'))
            runner = DemoRunner(scene)
            eng = LiveEngine(runner, sink=lambda m, b: None)
            pygame.init()
            app = db.BenchApp(scene, eng)
            self.assertEqual(app.engine_rows, 4)          # 4 engines per row since 2026-09-17 (was 5 rows of 3)
            self.assertEqual(len(app.engine_btns), 16)          # + laplace_carriers (2026-09-20)
            self.assertIn(orz.ENGINE_ID, app.engine_btns)
            screen = pygame.Surface((app.width, app.height))
            font, small = pygame.font.SysFont(db.FONT_NAMES, 17), pygame.font.SysFont(db.FONT_NAMES, 14)
            eng.start()
            try:
                eng.post('start')
                t0 = time.time()
                while time.time() - t0 < 5 and eng.snapshot()['gen'] < 2:
                    time.sleep(0.02)
                app.draw(screen, font, small)
                snap = eng.snapshot()
                self.assertEqual('figures' in snap['display']['A'], want_a)
                self.assertIn('figures', snap['display']['B'])
                eng.post('select', side='B')
                t0 = time.time()
                while time.time() - t0 < 5 and eng.snapshot()['selected'] != 'B':
                    time.sleep(0.02)
                app.draw(screen, font, small)
                # the figure's cells carry its colour on screen (listened side B)
                disp = eng.snapshot()['display']['B']
                f = disp['figures'][0]
                col = db.C_FIGURES[f['color'] % len(db.C_FIGURES)]
                r, c = f['cells'][0]
                px = screen.get_at((app.field_x + c * db.CELL + 3, app.field_y + r * db.CELL + 3))[:3]
                self.assertEqual(tuple(px), col)
                eng.post('pause', on=True)
                time.sleep(0.2)
                app.draw(screen, font, small)
                fake = dict(figures=[dict(id=1, color=0, slot=0, n=5, cells=[[0, 0], [31, 31]], centre=[31.5, 31.5],
                                          radius=1.4, modes=4, f_low=220.0, e=4.0, a=0.67, level=2.0)],
                            n_figures=1, n_sounding=1, n_single=0, n_tails=3, n_fading=0, evictions=1,
                            drops=0, unvoiced_blocks=0, changes=1, detector=0, detector_name='Own',
                            rows=32, cols=32, radius_mul=1.0, frequency_scale=220.0, decay_s=0.8, r=0.99, r_target=0.98,
                            ramp_left=100, gain=0.028, model='x')
                app._draw_figures(screen, small, fake, False)
                app._draw_display(screen, small, fake, app.panel_x, app.params_y + 4 * db.ROW_H + 6)
            finally:
                eng.stop()
                pygame.quit()


# ======================================================================================
class CatalogTests(unittest.TestCase):
    def test_scenes_match_the_req_and_the_preflight(self):
        from demos.build_n4_objects import CASES, scene_for, RATE_HZ, SECONDS
        from casynth_lab import tuned_events as te
        pf = preflight()
        self.assertEqual(RATE_HZ, 6.0)
        self.assertEqual(SECONDS, 12.0)
        for case, name in zip(CASES, ('glider', 'neighbors')):
            doc = scene_for(case)
            path = os.path.join(ROOT, 'demos', case['id'] + '.json')
            with open(path, encoding='utf-8') as f:
                self.assertEqual(json.load(f), doc)
            self.assertEqual(doc['cells'], [[int(r), int(c)] for r, c in pf['cases'][name]['cells']])
            self.assertEqual(doc['rate_hz'], 6.0)
            self.assertTrue(case['hypothesis'].startswith('Гипотеза - '))
        a, b = CASES[0]['A'], CASES[0]['B']
        self.assertEqual(a, (te.ENGINE_ID, dict(field_tuning=1, decay_s=0.8)))
        lap = orz.laplace_settings({})                       # the two N4 scenes keep the Figure law
        self.assertEqual(b, (orz.ENGINE_ID, dict(detector=1, radius_mul=1.0, spectrum=0, frequency_scale=220.0,
                                                 decay_s=0.8, attack_ms=0.0, events=0, excitation=0, birth_strength=1.0, decay_law=0, **lap)))
        a, b = CASES[1]['A'], CASES[1]['B']
        self.assertEqual(a, (orz.ENGINE_ID, dict(detector=0, radius_mul=1.0, spectrum=0, frequency_scale=220.0,
                                                 decay_s=0.8, attack_ms=0.0, events=0, excitation=0, birth_strength=1.0, decay_law=0, **lap)))
        self.assertEqual(b, (orz.ENGINE_ID, dict(detector=1, radius_mul=1.0, spectrum=0, frequency_scale=220.0,
                                                 decay_s=0.8, attack_ms=0.0, events=0, excitation=0, birth_strength=1.0, decay_law=0, **lap)))
        # the N4.2 figures evolve as independently placed figures and never merge
        g = grid(neighbor_cells())
        recv = grid(receiver_cells())
        blk = grid(blinker(9, 15))
        self.assertEqual(len(receiver_cells()), 17)
        for _ in range(72):
            self.assertEqual(len(fg.components(g)), 2)
            np.testing.assert_array_equal(g, np.maximum(recv, blk))
            g, recv, blk = step(g), step(recv), step(blk)
        with open(os.path.join(ROOT, 'run_object_resonators_n4.bat'), encoding='ascii') as f:
            bat = f.read()
        self.assertIn('n4_spectrum.json', bat)
        self.assertIn('object_resonators_n4_2026_09_16', bat)
        self.assertNotIn('--live', bat)

    def test_catalog_build_replay_continue_notes_and_no_overwrite(self):
        from casynth_lab import DemoRunner, scene_from_doc
        from casynth_lab.catalog import Catalog
        from demos.build_n1_demos import record_offline
        from demos.build_n4_objects import CASES, scene_for, build
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


if __name__ == '__main__':
    unittest.main()
