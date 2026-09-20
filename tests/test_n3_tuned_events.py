#!/usr/bin/env python3
"""N3 gates (REQ memory/req-network-combined-n3-2026-09-16.md, "Проверки до передачи"):
the whole chain events -> delays -> tuned banks -> DC filters -> output against an
independent scalar reference (both modes, through a decay ramp), the network states
against ca_event_network, the frequency law against gutter_field_periodic, event
semantics, the two control probes, silent retune / excited tail, snapshot mid-tail
and mid-ramp, limits, timing, the bench panel and the prepared scenes / catalog.

    python tests/test_n3_tuned_events.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_n3_tests/.
"""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, MASTER_GAIN                                    # noqa: E402
from casynth_lab import BLOCK, registry                                       # noqa: E402
from casynth_lab.engine_api import EngineContext, supports_snapshot          # noqa: E402
from casynth_lab.snapshot import save_state, load_state                      # noqa: E402
from casynth_lab import periodic_readout as pr                               # noqa: E402
from casynth_lab import gutter_field as gf                                   # noqa: E402
from casynth_lab import gutter_field_periodic as gp                          # noqa: E402
from casynth_lab import event_network as en                                  # noqa: E402
from casynth_lab import tuned_events as te                                   # noqa: E402
from casynth_lab.gutter_field_n1_config import CONFIG as N1_CONFIG           # noqa: E402

ART = os.path.join(ROOT, 'artifacts', '_n3_tests')
CTX = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 6.0)
GAIN = MASTER_GAIN * 0.7
TOL = 1e-12
PULSAR_LIKE = [(9, 11), (9, 12), (9, 13), (11, 9), (12, 9), (13, 9), (25, 20), (0, 0), (31, 31)]


def grid(cells, rows=32, cols=32):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r % rows, c % cols] = 1
    return g


def ref_weights(rows, cols):
    K = np.zeros((8, rows, cols))
    for i in range(8):
        j, l = i % 4, i // 4
        ax, ay = (j + 0.5) / 4, (l + 0.5) / 2
        for r in range(rows):
            for c in range(cols):
                x, y = (c + 0.5) / cols, (r + 0.5) / rows
                dx = ((x - ax + 0.5) % 1.0) - 0.5
                dy = ((y - ay + 0.5) % 1.0) - 0.5
                hx = (1 + math.cos(math.pi * dx / 0.25)) / 2 if abs(dx) <= 0.25 else 0.0
                hy = (1 + math.cos(math.pi * dy / 0.5)) / 2 if abs(dy) <= 0.5 else 0.0
                K[i, r, c] = hx * hy
    return K


def ref_packet(prev, cur):
    K = ref_weights(*np.asarray(cur).shape)
    E = (np.asarray(prev) != np.asarray(cur)).astype(float)
    e = np.tensordot(K, E, axes=([1, 2], [0, 1]))
    return e / (e + 2.0)


def ref_freqs(g, tuning):
    """The REQ law written out: n_i = sum K_i G, u = n / (n + 2), ratio = 2^(2u-1) or 1,
    f = float32(min(19000, base * ratio))."""
    K = ref_weights(*g.shape)
    n = np.tensordot(K, g.astype(float), axes=([1, 2], [0, 1]))
    u = n / (n + 2.0)
    f = np.zeros((8, 24))
    for i in range(8):
        ratio = 2.0 ** (2.0 * u[i] - 1.0) if tuning else 1.0
        for j in range(24):
            f[i, j] = float(np.float32(min(19000.0, N1_CONFIG['filters_hz'][i][j] * ratio)))
    return f


def ref_r(t60):
    return 10.0 ** (-3.0 / (SR * t60))


def scalar_n3(packets, freqs_of_block, n_samples, r, sr=SR):
    """The REQ chain per sample, plain Python floats.  packets: {sample: a (8,)} added
    to both pulse states before that sample; freqs_of_block(b) -> (8, 24) frequencies
    applied at the first sample of block b; r: float or per-sample array.
    Returns (out (n, 2) after the DC filters, l trace (n, 8), (zre, zim))."""
    D = [149, 211, 293, 401, 547, 743, 1009, 1361]
    q = math.exp(-2 * math.pi * 6000.0 / sr)
    qf = math.exp(-1.0 / (sr * 0.00025))
    qs = math.exp(-1.0 / (sr * 0.002))
    rho = 0.88
    h = math.exp(-2 * math.pi * 20.0 / sr)
    M = [[0.25 - (1.0 if i == j else 0.0) for j in range(8)] for i in range(8)]
    hist = [[0.0] * D[i] for i in range(8)]
    l = [0.0] * 8
    zf = [0.0] * 8
    zs = [0.0] * 8
    zre = [[0.0] * 24 for _ in range(8)]
    zim = [[0.0] * 24 for _ in range(8)]
    hp_prev, mix_prev = [0.0, 0.0], [0.0, 0.0]
    p = [(j + 0.5) / 4 for _ in range(2) for j in range(4)]
    cL = [math.cos(math.pi * x / 2) for x in p]
    cR = [math.sin(math.pi * x / 2) for x in p]
    r_k = (lambda k: float(r)) if np.isscalar(r) else (lambda k: float(r[k]))
    cth = sth = None
    out = np.zeros((n_samples, 2))
    ltr = np.zeros((n_samples, 8))
    for k in range(n_samples):
        if k % BLOCK == 0:
            f = freqs_of_block(k // BLOCK)
            cth = [[math.cos(2.0 * math.pi * float(f[i][j]) / float(sr)) for j in range(24)] for i in range(8)]
            sth = [[math.sin(2.0 * math.pi * float(f[i][j]) / float(sr)) for j in range(24)] for i in range(8)]
        if k in packets:
            for i in range(8):
                zf[i] += float(packets[k][i])
                zs[i] += float(packets[k][i])
        rk = r_k(k)
        exc = [0.75 * ((1 - qf) * zf[i] - (1 - qs) * zs[i]) / (qs - qf) for i in range(8)]
        for i in range(8):
            zf[i] *= qf
            zs[i] *= qs
        d = [hist[i][k % D[i]] for i in range(8)]
        for i in range(8):
            l[i] = (1 - q) * d[i] + q * l[i]
        for i in range(8):
            acc = 0.0
            for j in range(8):
                acc += M[i][j] * l[j]
            hist[i][k % D[i]] = math.tanh(exc[i] + rho * acc)
        L = 0.0
        R = 0.0
        for i in range(8):
            acc = 0.0
            for j in range(24):
                re, im = zre[i][j], zim[i][j]
                c, s = cth[i][j], sth[i][j]
                nre = rk * (c * re - s * im) + l[i]
                nim = rk * (s * re + c * im)
                zre[i][j], zim[i][j] = nre, nim
                acc += nre
            b = acc / 24
            L += cL[i] * b
            R += cR[i] * b
        mix = [L / 8, R / 8]
        for ch in range(2):
            y = h * ((hp_prev[ch] + mix[ch]) - mix_prev[ch])
            hp_prev[ch] = y
            mix_prev[ch] = mix[ch]
            out[k, ch] = y
        ltr[k] = l
    return out, ltr, (np.array(zre), np.array(zim))


def engine(g, tuning=1, decay=0.8, gain=GAIN):
    e = registry.create(te.ENGINE_ID, CTX, dict(field_tuning=tuning, decay_s=decay))
    e.init(g, None, gain)
    return e


def n2_engine(g, gain=GAIN):
    e = registry.create(en.ENGINE_ID, CTX, dict(rho=0.88))
    e.init(g, None, gain)
    return e


def net_of(e):
    if isinstance(e, te.TunedEventsEngine):
        return e.network_state()
    return dict(ring=e.ring.copy(), lfilt=e.lfilt.copy(), zf=e.zf.copy(), zs=e.zs.copy(), k=int(e.ints[0]))


def same_net(a, b):
    return (all(np.array_equal(a[k], b[k]) for k in ('ring', 'lfilt', 'zf', 'zs')) and a['k'] == b['k'])


class ChainReferenceTests(unittest.TestCase):
    def _history(self):
        g0 = grid(PULSAR_LIKE)
        g1 = grid(PULSAR_LIKE[:-2] + [(9, 14), (25, 21)])       # deaths + births
        return g0, g1

    def test_kernel_matches_scalar_reference_in_both_modes(self):
        g0, g1 = self._history()
        for tuning in (0, 1):
            e = engine(g0, tuning=tuning)
            blocks = []
            packets = {0: ref_packet(np.zeros_like(g0), g0)}
            a2 = None
            for b in range(9):
                if b == 2:                                            # overlapping packet + retune
                    e.update_field(g1, None)
                    packets[2 * BLOCK] = ref_packet(g0, g1)
                blocks.append(e.raw_block())
                if b == 2:
                    a2 = e.last_a.copy()
            fast = np.concatenate(blocks)
            f0, f1 = ref_freqs(g0, tuning), ref_freqs(g1, tuning)
            slow, ltr, (zre, zim) = scalar_n3(packets, lambda b: f0 if b < 2 else f1, 9 * BLOCK, ref_r(0.8))
            self.assertLessEqual(float(np.abs(fast - slow).max()), TOL, tuning)
            self.assertGreater(np.abs(fast).max(), 0.0)
            self.assertLessEqual(float(np.abs(e.zre - zre).max()), TOL)
            self.assertLessEqual(float(np.abs(e.zim - zim).max()), TOL)
            np.testing.assert_array_equal(e.ffreq, f1)
            self.assertLess(np.abs(a2 - packets[2 * BLOCK]).max(), 1e-12)
            self.assertEqual(e.last_a.tolist(), [0.0] * 8)

    def test_kernel_matches_scalar_reference_through_a_decay_ramp(self):
        g0, _ = self._history()
        e = engine(g0, tuning=1, decay=0.8)
        first = e.raw_block()
        e.set_params(dict(field_tuning=1, decay_s=0.3))               # 20 ms linear ramp of r
        self.assertEqual(int(e.ints[te.I_R_LEFT]), 882)
        rest = [e.raw_block() for _ in range(6)]
        fast = np.concatenate([first] + rest)
        n = 7 * BLOCK
        r0, r1 = ref_r(0.8), ref_r(0.3)
        r = np.full(n, r0)
        inc = (r1 - r0) / 882
        cur = r0
        for m in range(1, 882):
            cur = cur + inc
            r[BLOCK + m - 1] = cur
        r[BLOCK + 881:] = r1
        f = ref_freqs(g0, 1)
        slow, _, _ = scalar_n3({0: ref_packet(np.zeros_like(g0), g0)}, lambda b: f, n, r)
        self.assertLessEqual(float(np.abs(fast - slow).max()), TOL)
        self.assertEqual(float(e.rr[te.R_CUR]), r1)
        self.assertEqual(int(e.ints[te.I_R_LEFT]), 0)

    def test_network_states_equal_n2_and_do_not_depend_on_the_tuning(self):
        g0, g1 = self._history()
        a, b, n2 = engine(g0, tuning=0), engine(g0, tuning=1), n2_engine(g0)
        outs = {0: [], 1: []}
        for blk in range(12):
            if blk in (3, 7):
                g = g1 if blk == 3 else g0
                for x in (a, b, n2):
                    x.update_field(g, None)
            outs[0].append(a.raw_block())
            outs[1].append(b.raw_block())
            n2.raw_block()
            self.assertTrue(same_net(net_of(a), net_of(n2)), blk)
            self.assertTrue(same_net(net_of(b), net_of(n2)), blk)
            np.testing.assert_array_equal(a.last_a, n2.last_a)
            np.testing.assert_array_equal(b.last_a, n2.last_a)
        ya, yb = np.concatenate(outs[0]), np.concatenate(outs[1])
        self.assertGreater(float(np.abs(ya - yb).max()), 1e-3)          # the banks differ
        self.assertEqual(int(a.ints[te.I_K]), 12 * BLOCK)

    def test_frequencies_follow_the_a_law_and_fixed_mode_is_ratio_one(self):
        g = grid([(9, 11), (9, 12), (9, 13), (25, 20), (3, 30)])
        b = engine(g, tuning=1)
        a_side = registry.create(gp.ENGINE_ID, CTX, dict(scale=1.0, depth=1.0, interaction=127, freeze_ca=0))
        a_side.init(g, None, GAIN)
        np.testing.assert_array_equal(b.ffreq, a_side.ffreq)              # the A law, same field
        np.testing.assert_array_equal(b.ffreq, ref_freqs(g, 1))
        np.testing.assert_array_equal(b.ratio, gf.ratio_of(gf.field_u(pr.weighted_counts(pr.weights(32, 32), g)), 1.0, 1.0))
        a = engine(g, tuning=0)
        np.testing.assert_array_equal(a.ratio, np.ones(8))
        np.testing.assert_array_equal(a.ffreq, np.array(N1_CONFIG['filters_hz']).astype(np.float32).astype(np.float64))
        np.testing.assert_array_equal(a.ffreq, ref_freqs(g, 0))
        self.assertEqual(a.ffreq.shape, (8, 24))
        self.assertLess(float(a.ffreq.max()), 19000.0)
        self.assertEqual(te.BASE_HZ, tuple(tuple(row) for row in N1_CONFIG['filters_hz']))
        # the tables are exactly math.cos / math.sin of the float64 angles
        for i, j in ((0, 0), (3, 17), (7, 23)):
            th = 2.0 * math.pi * b.ffreq[i, j] / float(SR)
            self.assertEqual(b.cth[i, j], math.cos(th))
            self.assertEqual(b.sth[i, j], math.sin(th))
        # a retune by the switch: frequencies change, nothing else, no event
        b.raw_block()                                             # the initial packet is spent
        zre, zim, net = b.zre.copy(), b.zim.copy(), net_of(b)
        b.set_params(dict(field_tuning=0, decay_s=0.8))
        np.testing.assert_array_equal(b.ffreq, a.ffreq)
        np.testing.assert_array_equal(b.zre, zre)
        np.testing.assert_array_equal(b.zim, zim)
        self.assertTrue(same_net(net_of(b), net))
        b.raw_block()
        self.assertEqual(b.last_a.tolist(), [0.0] * 8)


class EventSemanticsTests(unittest.TestCase):
    def test_event_semantics(self):
        g0 = grid(PULSAR_LIKE)
        e = engine(g0)
        e.raw_block()
        a0 = e.last_a.copy()
        self.assertLess(np.abs(a0 - ref_packet(np.zeros_like(g0), g0)).max(), 1e-12)
        zf, zs = e.zf.copy(), e.zs.copy()
        e.update_field(g0, np.ones_like(g0, dtype=float))         # same field, exc ignored
        e.raw_block()
        self.assertEqual(e.last_a.tolist(), [0.0] * 8)
        self.assertTrue(np.all(e.zf < zf) and np.all(e.zs < zs))
        dead = g0.copy(); dead[9, 11] = 0
        born = g0.copy(); born[9, 10] = 1
        both = dead.copy(); both[9, 10] = 1
        made = {}
        for name, cur in (('dead', dead), ('born', born), ('both', both)):
            x = engine(g0)
            x.raw_block()
            x.update_field(cur, None)
            x.raw_block()
            self.assertLess(np.abs(x.last_a - ref_packet(g0, cur)).max(), 1e-12)
            self.assertGreater(x.last_a.max(), 0.0)
            made[name] = x.last_e.sum()
        self.assertLess(abs(made['both'] - (made['dead'] + made['born'])), 1e-12)
        y = engine(g0)
        y.raw_block()
        y.update_field(dead, None)
        y.update_field(g0, None)                                  # several edits inside a block
        y.raw_block()
        self.assertEqual(y.last_a.tolist(), [0.0] * 8)
        y.set_params(dict(field_tuning=0, decay_s=0.5))           # parameters are not events
        y.raw_block()
        self.assertEqual(y.last_a.tolist(), [0.0] * 8)
        y.reset(GAIN)                                             # init again: one packet
        self.assertEqual(float(np.abs(y.ring).max()), 0.0)
        self.assertEqual(float(np.abs(y.zre).max() + np.abs(y.hp).max()), 0.0)
        y.raw_block()
        self.assertLess(np.abs(y.last_a - a0).max(), 1e-12)

    def test_control_probes_hold_frequencies_or_hold_impulses(self):
        g0 = grid(PULSAR_LIKE)
        g1 = np.roll(g0, 9, axis=1)
        n = 40
        # (a) frequencies held (fixed mode), the impulse history changes with the field
        a0, a1 = engine(g0, tuning=0), engine(g1, tuning=0)
        y0 = np.concatenate([a0.raw_block() for _ in range(n)])
        y1 = np.concatenate([a1.raw_block() for _ in range(n)])
        np.testing.assert_array_equal(a0.ffreq, a1.ffreq)
        self.assertFalse(same_net(net_of(a0), net_of(a1)))
        self.assertGreater(float(np.abs(y0 - y1).max()), 1e-3)
        # (b) the impulse history held (same field), only the frequencies change
        b0, b1 = engine(g0, tuning=0), engine(g0, tuning=1)
        z0 = np.concatenate([b0.raw_block() for _ in range(n)])
        z1 = np.concatenate([b1.raw_block() for _ in range(n)])
        self.assertTrue(same_net(net_of(b0), net_of(b1)))
        np.testing.assert_array_equal(b0.last_a, b1.last_a)
        self.assertFalse(np.array_equal(b0.ffreq, b1.ffreq))
        self.assertGreater(float(np.abs(z0 - z1).max()), 1e-3)
        np.testing.assert_array_equal(z0, y0)                     # the same probe, reproducible

    def test_silent_retune_makes_no_sound_and_an_excited_tail_keeps_its_state(self):
        e = engine(np.zeros((32, 32), np.uint8))
        for blk in range(30):
            if blk % 7 == 3:
                e.set_params(dict(field_tuning=blk % 2, decay_s=0.8))   # retunes (u = 0 -> ratio 1/2)
            y, peak, n_clip = e.render_float(GAIN)
            self.assertEqual(float(np.abs(y).max()), 0.0)
        self.assertEqual(float(np.abs(e.zre).max() + np.abs(e.zim).max()), 0.0)
        # an excited tail: the packet arrives, the field stays, then a retune by the switch
        e.inject(np.full(8, 0.5))
        for _ in range(int(0.5 * SR / BLOCK)):
            e.render_float(GAIN)
        zre, zim, net, ff = e.zre.copy(), e.zim.copy(), net_of(e), e.ffreq.copy()
        e.set_params(dict(field_tuning=1, decay_s=0.8))
        np.testing.assert_array_equal(e.zre, zre)                 # states kept across the retune
        np.testing.assert_array_equal(e.zim, zim)
        self.assertTrue(same_net(net_of(e), net))
        self.assertFalse(np.array_equal(e.ffreq, ff))
        rms = []
        for sec in range(6):
            acc = np.concatenate([e.render_float(GAIN)[0] for _ in range(int(SR / BLOCK))]).mean(1)
            rms.append(20 * math.log10(max(float(np.sqrt(np.mean(acc ** 2))), 1e-30)))
            self.assertEqual(e.last_a.tolist(), [0.0] * 8)        # no new events
        self.assertGreater(rms[0], -80.0)                         # still sounding after the retune
        self.assertGreater(rms[0], rms[2])
        self.assertGreater(rms[2], rms[5])
        self.assertLess(rms[5], -100.0)                           # decays without new excitation
        self.assertTrue(np.all(np.isfinite(e.zre)) and np.all(np.isfinite(e.hp)))

    def test_decay_values_and_ramp(self):
        self.assertEqual(te.decay_r(0.8), 10.0 ** (-3.0 / (SR * 0.8)))
        e = engine(grid(PULSAR_LIKE), decay=0.8)
        self.assertEqual(float(e.rr[te.R_CUR]), te.decay_r(0.8))
        f = engine(grid(PULSAR_LIKE), decay=0.2)
        self.assertEqual(float(f.rr[te.R_CUR]), te.decay_r(0.2))       # init starts at the tuned r
        self.assertEqual(int(f.ints[te.I_R_LEFT]), 0)
        e.set_params(dict(field_tuning=1, decay_s=1.5))
        self.assertEqual(int(e.ints[te.I_R_LEFT]), 882)
        self.assertEqual(float(e.rr[te.R_CUR]), te.decay_r(0.8))       # unchanged until the samples run
        e.raw_block(); e.raw_block()
        self.assertGreater(int(e.ints[te.I_R_LEFT]), 0)
        e.raw_block()
        self.assertEqual(int(e.ints[te.I_R_LEFT]), 0)
        self.assertEqual(float(e.rr[te.R_CUR]), te.decay_r(1.5))       # lands exactly on the target
        self.assertEqual(e.display()['ramp_left'], 0)

    def test_snapshot_mid_tail_and_mid_ramp_restores_exactly_without_a_packet(self):
        g = grid(PULSAR_LIKE)
        e = engine(g, tuning=1, decay=0.8)
        self.assertTrue(supports_snapshot(e))
        for _ in range(int(0.3 * SR / BLOCK)):                    # an excited tail
            e.render(GAIN, 0)
        e.set_params(dict(field_tuning=1, decay_s=0.4))
        e.render(GAIN, 0)                                         # mid-ramp
        self.assertGreater(int(e.ints[te.I_R_LEFT]), 0)
        self.assertGreater(float(np.abs(e.zre).max()), 0.0)
        os.makedirs(ART, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            save_state(tmp, 's', e.export_state())
            st = load_state(tmp, 's')
        for key in ('ring', 'lfilt', 'zf', 'zs', 'M', 'D', 'ints', 'rr', 'zre', 'zim', 'ffreq', 'hp',
                    'grid_prev', 'grid_pending', 'gain_prev', 'params', 'model_version', 'rho', 'out_scale'):
            self.assertIn(key, st)
        r = registry.create(te.ENGINE_ID, CTX, dict(field_tuning=0, decay_s=1.2))
        r.restore_state(g, None, st)
        self.assertEqual(r.params, dict(field_tuning=1, decay_s=0.4))
        np.testing.assert_array_equal(r.zf, e.zf)                # no packet was added
        np.testing.assert_array_equal(r.cth, e.cth)
        for _ in range(12):
            np.testing.assert_array_equal(e.render(GAIN, 0)[0], r.render(GAIN, 0)[0])
        self.assertEqual(r.last_a.tolist(), [0.0] * 8)
        self.assertEqual(int(r.ints[te.I_R_LEFT]), int(e.ints[te.I_R_LEFT]))
        other = g.copy(); other[0, 5] = 1
        fresh = lambda: registry.create(te.ENGINE_ID, CTX, dict(field_tuning=1, decay_s=0.8))   # noqa: E731
        with self.assertRaises(ValueError):
            fresh().restore_state(other, None, st)
        with self.assertRaises(ValueError):
            fresh().restore_state(g, None, dict(st, model_version='x'))
        with self.assertRaises(ValueError):
            fresh().restore_state(g, None, dict(st, ffreq=st['ffreq'] * 1.01))      # not the field's law
        with self.assertRaises(ValueError):
            fresh().restore_state(g, None, dict(st, params=dict(field_tuning=0, decay_s=0.4)))
        with self.assertRaises(ValueError):
            fresh().restore_state(g, None, dict(st, rho=0.9))
        with self.assertRaises(ValueError):
            registry.create(en.ENGINE_ID, CTX, dict(rho=0.88)).restore_state(g, None, st)


class LimitsAndContractTests(unittest.TestCase):
    def test_limits_full_toggle_random_edits_and_decay_ends_at_max_gain(self):
        rng = np.random.default_rng(20260916)
        gain = MASTER_GAIN * 1.0                                   # vol 1, level 1 -> 0.04
        worst = {}
        for decay in (0.20, 1.50):
            for mode in ('full_toggle', 'random', 'edits', 'knob'):
                g = np.zeros((32, 32), np.uint8)
                x = engine(g, tuning=1, decay=decay, gain=gain)
                peak = 0.0
                for b in range(int(4 * SR / BLOCK)):
                    if mode == 'full_toggle':
                        g = 1 - g
                    elif mode == 'random':
                        g = (rng.random((32, 32)) < 0.5).astype(np.uint8)
                    elif b % 12 == 0:
                        r0, c0 = int(rng.integers(0, 27)), int(rng.integers(0, 27))
                        g = g.copy(); g[r0:r0 + 5, c0:c0 + 5] = 1 - g[r0:r0 + 5, c0:c0 + 5]
                        if mode == 'knob':
                            x.set_params(dict(field_tuning=(b // 12) % 2,
                                              decay_s=(0.2 if (b // 12) % 3 else 1.5)))
                    x.update_field(g, None)
                    y, p, n_clip = x.render_float(gain)
                    self.assertEqual(n_clip, 0, (decay, mode))
                    self.assertTrue(np.all(np.isfinite(y)))
                    peak = max(peak, p)
                self.assertLess(peak, 1.0, (decay, mode))
                worst[(decay, mode)] = peak
        self.assertGreater(max(worst.values()), 0.1)              # the probes did excite the model

    def test_rejects_other_contexts_registry_hints_and_display(self):
        params = dict(field_tuning=1, decay_s=0.8)
        with self.assertRaises(ValueError):
            registry.create(te.ENGINE_ID, EngineContext(48000, BLOCK, 2, 110.0, 1.0, 6.0), params)
        with self.assertRaises(ValueError):
            registry.create(te.ENGINE_ID, EngineContext(SR, 256, 2, 110.0, 1.0, 6.0), params)
        spec = registry.get(te.ENGINE_ID)
        self.assertEqual([p[0] for p in spec.params], ['field_tuning', 'decay_s'])
        self.assertEqual(spec.params[0][2:], (0, 1, True, 1))
        self.assertEqual(spec.params[1][2:], (0.20, 1.50, False, 0.80))
        self.assertEqual(spec.choices['field_tuning'], ('Fixed', 'Field'))
        ov = spec.overlay(params, 32, 32)
        self.assertEqual(len(ov['labels']), 8)
        self.assertEqual(len(ov['circles']), 16)
        self.assertNotIn('lines', ov)
        e = engine(grid(PULSAR_LIKE))
        e.render(GAIN, 0)
        d = e.display()
        for key in ('events', 'e', 'level', 'ratio', 'counts', 'tuning', 'decay_s', 'r', 'r_target',
                    'ramp_left', 'model'):
            self.assertIn(key, d)
        self.assertEqual(len(d['events']), 8)
        self.assertEqual(len(d['ratio']), 8)
        self.assertGreater(max(d['level']), 0.0)
        self.assertEqual(d['model'], te.MODEL_VERSION)
        self.assertEqual(te.RHO, 0.88)
        self.assertEqual(te.OUT_SCALE, 4.0)

    def test_engine_timing_budget(self):
        import time
        g = grid(PULSAR_LIKE)
        e = engine(g)
        times = []
        for b in range(400):
            if b % 7 == 0:
                g = np.roll(g, 1, axis=1)
                e.update_field(g, None)
            t0 = time.perf_counter()
            e.render(GAIN, 0)
            times.append((time.perf_counter() - t0) * 1000.0)
        self.assertLess(float(np.percentile(times[100:], 99)), 0.5 * BLOCK / SR * 1000.0)


class BenchTests(unittest.TestCase):
    def test_headless_draw_overlay_and_panel_for_both_sides(self):
        import time
        import pygame
        import demo_bench as db
        from casynth_lab import DemoRunner, load_scene
        from casynth_lab.audio_out import LiveEngine
        scene = load_scene(os.path.join(ROOT, 'demos', 'n3_cycles.json'))
        runner = DemoRunner(scene)
        eng = LiveEngine(runner, sink=lambda m, b: None)
        pygame.init()
        app = db.BenchApp(scene, eng)
        self.assertEqual(app.engine_rows, 5)          # 4 engines per row; 5 rows since laplace_fm
        self.assertEqual(len(app.engine_btns), 17)          # + laplace_fm (2026-09-20)
        self.assertIn(te.ENGINE_ID, app.engine_btns)
        self.assertGreater(app.height - app.footer_y, 40)
        data = app._overlay_runs(te.ENGINE_ID, dict(field_tuning=1, decay_s=0.8))
        self.assertEqual(data['lines'], [])
        self.assertEqual(len(data['circles']), 16)
        self.assertEqual(len(data['labels']), 8)
        self.assertTrue(data['text'])
        self.assertEqual(registry.value_text(te.ENGINE_ID, 'field_tuning', 0), 'Fixed')
        self.assertEqual(registry.value_text(te.ENGINE_ID, 'field_tuning', 1), 'Field')
        screen = pygame.Surface((app.width, app.height))
        font, small = pygame.font.SysFont(db.FONT_NAMES, 17), pygame.font.SysFont(db.FONT_NAMES, 14)
        eng.start()
        try:
            app.draw(screen, font, small)
            snap = eng.snapshot()
            self.assertIs(snap['display']['A']['tuning'], False)
            self.assertIs(snap['display']['B']['tuning'], True)
            for s in ('A', 'B'):
                for key in ('events', 'level', 'ratio', 'counts', 'decay_s'):
                    self.assertIn(key, snap['display'][s])
            eng.post('select', side='B')
            t0 = time.time()
            while time.time() - t0 < 5 and eng.snapshot()['selected'] != 'B':
                time.sleep(0.02)
            self.assertEqual(eng.snapshot()['selected'], 'B')
            app.draw(screen, font, small)
            app._draw_display(screen, small, dict(events=[0.5] * 8, e=[1.0] * 8, level=[2.0] * 8,
                                                  ratio=[1.2] * 8, counts=[3.5] * 8, tuning=True,
                                                  decay_s=0.8, r=0.99, r_target=0.98, ramp_left=100,
                                                  model='x'),
                              app.panel_x, app.params_y + 4 * db.ROW_H + 6)
        finally:
            eng.stop()
            pygame.quit()


class CatalogTests(unittest.TestCase):
    def _fixtures(self):
        path = os.path.join(ROOT, 'memory', 'research', 'network-n3-preflight-2026-09-16.json')
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding='utf-8') as f:
            pf = json.load(f)
        return {c['name']: sorted(map(tuple, c['cells'])) for c in pf['cases']}

    def _evolution(self, cells, gens=140):
        from casynth_engine import step
        g = grid(cells)
        seen = {g.tobytes(): 0}
        cur, events, pops, period = g, [], [int(g.sum())], None
        for gen in range(1, gens):
            nxt = step(cur)
            events.append(int((nxt != cur).sum()))
            pops.append(int(nxt.sum()))
            if period is None and nxt.tobytes() in seen:
                period = gen - seen[nxt.tobytes()]
            seen.setdefault(nxt.tobytes(), gen)
            cur = nxt
        return period, events, pops

    def test_scenes_fixtures_periods_and_the_swap_command_match_the_req(self):
        from demos.build_n3_combined import (CASES, scene_for, octagon_ii, tumbler, glider,
                                             r_pentomino, PARAMS_A, PARAMS_B, SWAP_AT_S, ROWS, COLS)
        ref = self._fixtures()
        self.assertEqual(sorted(map(tuple, octagon_ii())), ref['Octagon II'])
        self.assertEqual(sorted(map(tuple, tumbler())), ref['Tumbler'])
        self.assertEqual(sorted(map(tuple, glider())), ref['Glider'])
        self.assertEqual(sorted(map(tuple, r_pentomino())), ref['R-pentomino'])
        self.assertEqual(self._evolution(octagon_ii())[0], 5)
        self.assertEqual(self._evolution(tumbler())[0], 14)
        period, events, pops = self._evolution(glider())
        self.assertEqual(period, 128)
        self.assertEqual((min(events[:128]), max(events[:128])), (4, 4))
        self.assertEqual((min(pops), max(pops)), (5, 5))
        self.assertIsNone(self._evolution(r_pentomino(), 130)[0])
        want = dict(n3_cycles=(6.0, 24.0, 1), n3_travel=(16.0, 12.0, 0), n3_growth=(6.0, 12.0, 0))
        for case in CASES:
            doc = scene_for(case)
            path = os.path.join(ROOT, 'demos', case['id'] + '.json')
            with open(path, encoding='utf-8') as f:
                self.assertEqual(json.load(f), doc)
            self.assertEqual((doc['variants']['A']['engine_id'], doc['variants']['B']['engine_id']),
                             (te.ENGINE_ID, te.ENGINE_ID))
            self.assertEqual(doc['variants']['A']['engine_params'], dict(field_tuning=0, decay_s=0.8))
            self.assertEqual(doc['variants']['B']['engine_params'], dict(field_tuning=1, decay_s=0.8))
            self.assertEqual(PARAMS_A['decay_s'], PARAMS_B['decay_s'])
            rate, seconds, n_cmd = want[case['id']]
            self.assertEqual((case['rate'], case['seconds'], len(case['commands'])), (rate, seconds, n_cmd))
            self.assertTrue(case['hypothesis'].startswith('Гипотеза - '))
        kind, at, args = CASES[0]['commands'][0]
        self.assertEqual((kind, at), ('set_cells', int(round(SWAP_AT_S * SR))))
        self.assertEqual(len(args['cells']), ROWS * COLS)                  # the WHOLE field, one command
        self.assertEqual(sorted((r, c) for r, c, v in args['cells'] if v), ref['Tumbler'])
        with open(os.path.join(ROOT, 'run_network_n3.bat'), encoding='ascii') as f:
            bat = f.read()
        self.assertIn('n3_cycles.json', bat)
        self.assertIn('network_n3_combined_2026_09_16', bat)
        self.assertNotIn('--live', bat)

    def test_the_swap_is_one_journalled_command_that_keeps_audio_and_the_ca_clock(self):
        from casynth_engine import step
        from casynth_lab import DemoRunner, scene_from_doc
        from demos.build_n3_combined import CASES, scene_for, tumbler
        case = CASES[0]
        runner = DemoRunner(scene_from_doc(scene_for(case)))
        runner.post('start', at=0)
        for kind, at, args in case['commands']:
            runner.post(kind, at=at, **args)
        swap_at = case['commands'][0][1]
        boundary = -(-swap_at // BLOCK) * BLOCK
        for _ in range(boundary // BLOCK):
            runner.next_block()
        self.assertEqual(runner.out_samples, boundary)
        self.assertEqual(runner.journal[-1][2], 'step')                   # nothing swapped yet
        gen_before, k_before = int(runner.gen), int(runner.sides['A'].engine.ints[te.I_K])
        zre_before = runner.sides['A'].engine.zre.copy()
        self.assertGreater(float(np.abs(zre_before).max()), 0.0)          # an excited tail
        runner.next_block()
        swaps = [(o, k, a) for o, q, k, a in runner.journal if k == 'set_cells']
        self.assertEqual(len(swaps), 1)
        self.assertEqual(swaps[0][0], boundary)
        self.assertEqual(len(swaps[0][2]['cells']), 32 * 32)
        # the swap is applied before the CA step of that block: the field is the Tumbler
        # advanced by the generations the clock owes at that boundary (no clock reset)
        t = grid(tumbler())
        gens = int(runner.gen) - gen_before
        for _ in range(gens):
            t = step(t)
        np.testing.assert_array_equal(runner.grid, t)
        self.assertEqual(int(runner.sides['A'].engine.ints[te.I_K]), k_before + BLOCK)   # no audio reset
        self.assertEqual(runner.t_samples, runner.out_samples)
        for s in ('A', 'B'):
            self.assertGreater(float(runner.sides[s].engine.last_a.max()), 0.0)          # one packet
            self.assertTrue(np.all(np.isfinite(runner.sides[s].engine.zre)))
        self.assertFalse(np.array_equal(runner.sides['A'].engine.zre, zre_before))
        self.assertGreater(float(np.abs(runner.sides['A'].engine.ring).max()), 0.0)

    def test_catalog_build_replay_continue_notes_and_no_overwrite(self):
        from casynth_lab import DemoRunner, scene_from_doc
        from casynth_lab.catalog import Catalog
        from demos.build_n1_demos import record_offline
        from demos.build_n3_combined import CASES, scene_for, build
        os.makedirs(ART, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            cat = Catalog(tmp, repo_root=None)
            firsts = []
            for case in CASES:
                doc = scene_for(case)
                rid, snap = record_offline(cat, doc, 0.5, list(case['commands']), case['title'], case['note'])
                self.assertEqual(snap['clip_blocks'], dict(A=0, B=0))
                cat.write_notes(rid, case['hypothesis'] + '\n\nUser feedback: test.')
                rec = cat.load(rid)
                self.assertTrue(rec.notes.startswith('Гипотеза - '))
                result = cat.replay(rid, yield_cpu=False)
                self.assertEqual(result.status, 'match', result.reason)
                continued, state, _ = cat.continue_runner(rid)
                control = DemoRunner(scene_from_doc(doc))
                control.post('start', at=0)
                for kind, at, args in case['commands']:
                    control.post(kind, at=at, **args)
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
                self.assertGreater(continued.gen, gen0)                  # the CA keeps evolving
                firsts.append(first)
                with self.assertRaises(SystemExit):
                    build(tmp)
                self.assertIn('User feedback: test.', cat.load(rid).notes)
            self.assertEqual(len(set(firsts)), len(CASES))


if __name__ == '__main__':
    unittest.main()
