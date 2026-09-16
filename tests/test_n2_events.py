#!/usr/bin/env python3
"""N2 gates (REQ memory/req-network-events-n2-2026-09-16.md, "Проверки до передачи"):
the shared periodic readout, side A (gutter_field_periodic) against gutter_field,
side B (ca_event_network) against an independent scalar reference, event
semantics, causality through the delays, snapshot / continue, limits, timing,
the three scenes and the prepared catalog.

    python tests/test_n2_events.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_n2_tests/.
"""
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

ART = os.path.join(ROOT, 'artifacts', '_n2_tests')
CTX = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 6.0)
GAIN = MASTER_GAIN * 0.7
GUTTER_PARAMS = dict(scale=1.0, depth=1.0, interaction=127, freeze_ca=0)


def grid(cells, rows=32, cols=32):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r % rows, c % cols] = 1
    return g


def ref_weights(rows, cols):
    """The REQ formulas written out per cell (independent of periodic_readout)."""
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


class PeriodicReadoutTests(unittest.TestCase):
    def test_partition_of_unity_and_formulas(self):
        for rows, cols in ((32, 32), (16, 24), (7, 9)):
            K = pr.weights(rows, cols)
            self.assertEqual(K.shape, (8, rows, cols))
            self.assertTrue(np.all(K >= 0.0))
            self.assertLess(np.abs(K.sum(0) - 1.0).max(), 1e-12)
            self.assertLess(np.abs(K - ref_weights(rows, cols)).max(), 1e-12)

    def test_periodic_on_the_torus(self):
        K = pr.weights(32, 32)
        # a shift by one node spacing (8 columns / 16 rows) maps node i onto its neighbour
        for i in range(8):
            j, l = i % 4, i // 4
            right = 4 * l + (j + 1) % 4
            self.assertLess(np.abs(np.roll(K[i], 8, axis=1) - K[right]).max(), 1e-12)
            down = 4 * ((l + 1) % 2) + j
            self.assertLess(np.abs(np.roll(K[i], 16, axis=0) - K[down]).max(), 1e-12)
        # continuity across the outer edge: the step between columns 31 and 0 is the
        # same size as any interior step of the same weight
        steps = np.abs(np.diff(np.concatenate([K[0, 16], K[0, 16, :1]])))
        self.assertLess(steps[-1], steps.max() + 1e-12)
        self.assertGreater(K[0, 16, 0], 0.0)          # node 0 (centre column 3.5) reaches column 0
        self.assertGreater(K[3, 16, 0], 0.0)          # ... and so does node 3 across the edge

    def test_weighted_counts_and_centres(self):
        K = pr.weights(32, 32)
        g = grid([(16, 3), (16, 4), (0, 31)])
        n = pr.weighted_counts(K, g)
        self.assertLess(abs(n.sum() - 3.0), 1e-12)
        self.assertGreater(n[0], n[1])
        self.assertEqual(pr.centre_cells(32, 32)[0], (3.5, 7.5))
        self.assertEqual(pr.centre_cells(32, 32)[7], (27.5, 23.5))
        with self.assertRaises(ValueError):
            pr.weighted_counts(K, np.zeros((16, 16)))


class GutterPeriodicTests(unittest.TestCase):
    def _pair(self, g, **over):
        params = dict(GUTTER_PARAMS)
        params.update(over)
        a = registry.create(gp.ENGINE_ID, CTX, params)
        b = registry.create(gf.ENGINE_ID, CTX, params)
        a.init(g, None, GAIN)
        b.init(g, None, GAIN)
        return a, b

    def test_same_model_as_gutter_field_only_the_readout_and_trim_differ(self):
        # empty field: every count is 0 on both readouts -> identical messages, identical raw
        a, b = self._pair(np.zeros((32, 32), np.uint8))
        self.assertEqual(a.model_version, gp.MODEL_VERSION)
        self.assertNotEqual(a.model_version, b.model_version)
        for _ in range(6):
            np.testing.assert_array_equal(a.raw_block(), b.raw_block())
        ya, _, _ = a.render_float(GAIN)
        yb, _, _ = b.render_float(GAIN)
        self.assertGreater(np.abs(yb).max(), 0.0)
        self.assertLess(np.abs(ya - yb * gp.A_TRIM).max(), 1e-15)
        self.assertAlmostEqual(20 * math.log10(gp.A_TRIM), -16.5, places=12)

    def test_weighted_counts_drive_the_frequencies(self):
        g = grid([(9, 11), (9, 12), (9, 13), (25, 20)])
        a, b = self._pair(g)
        K = ref_weights(32, 32)
        want = np.tensordot(K, g.astype(float), axes=([1, 2], [0, 1]))
        self.assertLess(np.abs(a.counts - want).max(), 1e-12)
        self.assertLess(np.abs(a.u_field - want / (want + 2.0)).max(), 1e-12)
        ratio = np.power(2.0, np.clip(2.0 * a.u_field - 1.0, -1.0, 1.0))
        base = np.array(a.cfg['filters_hz'])
        np.testing.assert_array_equal(a.ffreq, np.minimum(base * ratio[:, None], 19000.0)
                                      .astype(np.float32).astype(np.float64))
        # a cell moved INSIDE one N1 region: N1 does not see it, the periodic readout does
        moved = grid([(9, 11), (9, 12), (9, 14), (25, 20)])
        a.update_field(moved, None)
        b.update_field(moved, None)
        self.assertLess(np.abs(b.u_field - gf.field_u(gf.region_counts(g))).max(), 1e-15)
        self.assertNotEqual(list(a.counts), list(want))
        self.assertEqual(a.display()['readout'], 'periodic')
        self.assertIsInstance(a.display()['counts'][0], float)

    def test_snapshot_roundtrip_and_registry(self):
        spec = registry.get(gp.ENGINE_ID)
        self.assertEqual([p[0] for p in spec.params], [p[0] for p in gf.PARAMS])
        ov = spec.overlay(GUTTER_PARAMS, 32, 32)
        self.assertEqual(len(ov['labels']), 8)
        self.assertNotIn('lines', ov)
        g = grid([(9, 11), (9, 12), (9, 13)])
        a, _ = self._pair(g)
        self.assertTrue(supports_snapshot(a))
        for _ in range(3):
            a.render(GAIN, 0)
        a.set_params(dict(GUTTER_PARAMS, interaction=200))       # in-flight 50 ms ramp
        a.render(GAIN, 0)
        os.makedirs(ART, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            save_state(tmp, 's', a.export_state())
            st = load_state(tmp, 's')
        r = registry.create(gp.ENGINE_ID, CTX, GUTTER_PARAMS)
        r.restore_state(g, None, st)
        for _ in range(8):
            np.testing.assert_array_equal(a.render(GAIN, 0)[0], r.render(GAIN, 0)[0])
        bad = dict(st, model_version=gf.MODEL_VERSION)
        with self.assertRaises(ValueError):
            registry.create(gp.ENGINE_ID, CTX, GUTTER_PARAMS).restore_state(g, None, bad)


# -- side B: an independent scalar reference of the REQ formulas -------------------------
def scalar_b(packets, n_samples, rho, sr=SR):
    """packets: {sample index: a (8,)} added to both pulse states before that sample;
    rho: float or per-sample array.  Returns (out (n, 2), l trace (n, 8))."""
    D = [149, 211, 293, 401, 547, 743, 1009, 1361]
    q = math.exp(-2 * math.pi * 6000.0 / sr)
    qf = math.exp(-1.0 / (sr * 0.00025))
    qs = math.exp(-1.0 / (sr * 0.002))
    M = [[0.25 - (1.0 if i == j else 0.0) for j in range(8)] for i in range(8)]
    hist = [[0.0] * D[i] for i in range(8)]
    l = [0.0] * 8
    zf = [0.0] * 8
    zs = [0.0] * 8
    p = [(j + 0.5) / 4 for _ in range(2) for j in range(4)]
    cL = [math.cos(math.pi * x / 2) for x in p]
    cR = [math.sin(math.pi * x / 2) for x in p]
    rho_k = (lambda k: float(rho)) if np.isscalar(rho) else (lambda k: float(rho[k]))
    out = np.zeros((n_samples, 2))
    ltr = np.zeros((n_samples, 8))
    for k in range(n_samples):
        if k in packets:
            for i in range(8):
                zf[i] += float(packets[k][i])
                zs[i] += float(packets[k][i])
        exc = [0.75 * ((1 - qf) * zf[i] - (1 - qs) * zs[i]) / (qs - qf) for i in range(8)]
        for i in range(8):
            zf[i] *= qf
            zs[i] *= qs
        d = [hist[i][k % D[i]] for i in range(8)]
        for i in range(8):
            l[i] = (1 - q) * d[i] + q * l[i]
        r = rho_k(k)
        for i in range(8):
            acc = 0.0
            for j in range(8):
                acc += M[i][j] * l[j]
            hist[i][k % D[i]] = math.tanh(exc[i] + r * acc)
        L = 0.0
        R = 0.0
        for i in range(8):
            L += cL[i] * l[i]
            R += cR[i] * l[i]
        out[k] = (L / 8, R / 8)
        ltr[k] = l
    return out, ltr


def ref_packet(prev, cur):
    K = ref_weights(*np.asarray(cur).shape)
    E = (np.asarray(prev) != np.asarray(cur)).astype(float)
    e = np.tensordot(K, E, axes=([1, 2], [0, 1]))
    return e / (e + 2.0)


def b_engine(g, rho=0.88, gain=GAIN):
    e = registry.create(en.ENGINE_ID, CTX, dict(rho=rho))
    e.init(g, None, gain)
    return e


PULSAR_LIKE = [(9, 11), (9, 12), (9, 13), (11, 9), (12, 9), (13, 9), (25, 20), (0, 0), (31, 31)]


class EventNetworkTests(unittest.TestCase):
    def test_kernel_matches_scalar_reference_single_and_overlapping_packets(self):
        g0 = grid(PULSAR_LIKE)
        g1 = grid(PULSAR_LIKE[:-2] + [(9, 14), (25, 21)])       # deaths + births
        e = b_engine(g0)
        blocks = []
        packets = {0: ref_packet(np.zeros_like(g0), g0)}
        for b in range(9):
            if b == 2:                                            # overlapping packet
                e.update_field(g1, None)
                packets[2 * BLOCK] = ref_packet(g0, g1)
            blocks.append(e.raw_block())
        fast = np.concatenate(blocks)
        slow, _ = scalar_b(packets, 9 * BLOCK, 0.88)
        np.testing.assert_array_equal(fast, slow)
        self.assertGreater(np.abs(fast).max(), 0.0)
        e2 = b_engine(g0)
        e2.raw_block()
        self.assertLess(np.abs(e2.last_a - packets[0]).max(), 1e-12)

    def test_kernel_matches_scalar_reference_through_a_response_ramp(self):
        g0 = grid(PULSAR_LIKE)
        e = b_engine(g0, rho=0.70)
        first = e.raw_block()
        e.set_params(dict(rho=0.95))                              # 20 ms linear ramp
        self.assertEqual(int(e.ints[en.I_RHO_LEFT]), 882)
        rest = [e.raw_block() for _ in range(6)]
        fast = np.concatenate([first] + rest)
        n = 7 * BLOCK
        rho = np.full(n, 0.70)
        inc = (0.95 - 0.70) / 882
        cur = 0.70
        for m in range(1, 882):
            cur = cur + inc
            rho[BLOCK + m - 1] = cur
        rho[BLOCK + 881:] = 0.95
        slow, _ = scalar_b({0: ref_packet(np.zeros_like(g0), g0)}, n, rho)
        np.testing.assert_array_equal(fast, slow)
        self.assertEqual(float(e.rho[en.R_CUR]), 0.95)
        self.assertEqual(int(e.ints[en.I_RHO_LEFT]), 0)

    def test_event_semantics(self):
        g0 = grid(PULSAR_LIKE)
        e = b_engine(g0)
        e.raw_block()
        a0 = e.last_a.copy()
        self.assertTrue(np.all(a0 >= 0.0) and a0.max() > 0.0)
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
            x = b_engine(g0)
            x.raw_block()
            x.update_field(cur, None)
            x.raw_block()
            self.assertLess(np.abs(x.last_a - ref_packet(g0, cur)).max(), 1e-12)
            self.assertGreater(x.last_a.max(), 0.0)
            made[name] = x.last_e.sum()
        self.assertLess(abs(made['both'] - (made['dead'] + made['born'])), 1e-12)   # no cancellation
        y = b_engine(g0)
        y.raw_block()
        y.update_field(dead, None)
        y.update_field(g0, None)                                  # back inside the block
        y.raw_block()
        self.assertEqual(y.last_a.tolist(), [0.0] * 8)
        y.set_params(dict(rho=0.6))                               # not an event
        y.raw_block()
        self.assertEqual(y.last_a.tolist(), [0.0] * 8)
        y.reset(GAIN)                                             # init again: one packet
        self.assertEqual(float(np.abs(y.ring).max()), 0.0)
        y.raw_block()
        self.assertLess(np.abs(y.last_a - a0).max(), 1e-12)

    def test_causality_through_the_delays_by_states(self):
        e = b_engine(np.zeros((32, 32), np.uint8))
        e.raw_block()
        self.assertEqual(float(np.abs(e.lfilt).max()), 0.0)
        D = list(en.DELAYS)
        for src in (0, 5):
            x = b_engine(np.zeros((32, 32), np.uint8))
            x.raw_block()
            a = np.zeros(8); a[src] = 0.5
            x.inject(a)
            n = D[src] + 1361 + 4
            trace = np.zeros((n, 8))
            out = np.zeros((1, 2))
            for k in range(n):
                x._kernel(1, out)
                trace[k] = x.lfilt
            first = [int(np.argmax(np.abs(trace[:, i]) > 0.0)) for i in range(8)]
            for i in range(8):
                self.assertEqual(first[i], D[src] if i == src else D[src] + D[i], (src, i))
            _, ltr = scalar_b({0: a}, n, 0.88)
            np.testing.assert_array_equal(trace, ltr)

    def test_zero_stays_zero_and_the_tail_decays_without_reset(self):
        e = b_engine(np.zeros((32, 32), np.uint8))
        for _ in range(50):
            y, peak, n_clip = e.render_float(GAIN)
            self.assertEqual(float(np.abs(y).max()), 0.0)
        self.assertEqual(peak, 0.0)
        e.update_field(grid(PULSAR_LIKE), None)
        rms = []
        for sec in range(12):
            acc = np.concatenate([e.render_float(GAIN)[0] for _ in range(int(SR / BLOCK))]).mean(1)
            rms.append(20 * math.log10(max(float(np.sqrt(np.mean(acc ** 2))), 1e-30)))
        self.assertGreater(rms[0], rms[4])
        self.assertGreater(rms[4], rms[11])
        self.assertLess(rms[11], -100.0)
        self.assertTrue(np.all(np.isfinite(e.ring)) and np.all(np.isfinite(e.lfilt)))

    def test_snapshot_restore_adds_no_packet_and_continues_exactly(self):
        g = grid(PULSAR_LIKE)
        e = b_engine(g)
        self.assertTrue(supports_snapshot(e))
        for _ in range(3):
            e.render(GAIN, 0)
        e.set_params(dict(rho=0.65))
        e.render(GAIN, 0)                                         # mid-ramp
        self.assertGreater(int(e.ints[en.I_RHO_LEFT]), 0)
        os.makedirs(ART, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ART) as tmp:
            save_state(tmp, 's', e.export_state())
            st = load_state(tmp, 's')
        for key in ('ring', 'lfilt', 'zf', 'zs', 'M', 'D', 'ints', 'rho', 'grid_prev',
                    'grid_pending', 'gain_prev', 'params', 'model_version'):
            self.assertIn(key, st)
        r = registry.create(en.ENGINE_ID, CTX, dict(rho=0.88))
        r.restore_state(g, None, st)
        self.assertEqual(r.params, dict(rho=0.65))
        np.testing.assert_array_equal(r.zf, e.zf)                # no packet was added
        for _ in range(12):
            np.testing.assert_array_equal(e.render(GAIN, 0)[0], r.render(GAIN, 0)[0])
        self.assertEqual(r.last_a.tolist(), [0.0] * 8)
        other = g.copy(); other[0, 5] = 1
        with self.assertRaises(ValueError):
            registry.create(en.ENGINE_ID, CTX, dict(rho=0.88)).restore_state(other, None, st)
        with self.assertRaises(ValueError):
            registry.create(en.ENGINE_ID, CTX, dict(rho=0.88)).restore_state(
                g, None, dict(st, model_version='x'))

    def test_transfer_of_the_same_event_between_nodes_changes_the_pattern(self):
        outs = {}
        for src in (0, 3, 5):
            x = b_engine(np.zeros((32, 32), np.uint8))
            x.raw_block()
            a = np.zeros(8); a[src] = 0.5
            x.inject(a)
            outs[src] = np.concatenate([x.raw_block(events=False) for _ in range(int(SR / BLOCK))])
        mono = [outs[0].mean(1), outs[5].mean(1)]
        mono = [m / np.sqrt(np.mean(m ** 2)) for m in mono]               # energy equalised
        env = [np.sqrt((m.reshape(-1, BLOCK) ** 2).mean(1)) for m in mono]
        self.assertLess(np.corrcoef(env[0], env[1])[0, 1], 0.995)
        spec = [20 * np.log10(np.abs(np.fft.rfft(m)) + 1e-12) for m in mono]
        self.assertGreater(float(np.sqrt(np.mean((spec[0] - spec[1]) ** 2))), 3.0)
        # the stereo image follows the node position before the coupling arrives:
        # node 0 (p = 0.125) sits left, node 3 (p = 0.875) right
        D = list(en.DELAYS)
        w0, w3 = outs[0][D[0]:D[0] + 140], outs[3][D[3]:D[3] + 140]
        self.assertGreater(np.abs(w0[:, 0]).mean(), 3 * np.abs(w0[:, 1]).mean())
        self.assertLess(3 * np.abs(w3[:, 0]).mean(), np.abs(w3[:, 1]).mean())

    def test_limits_full_toggle_random_and_response_ends_at_max_gain(self):
        rng = np.random.default_rng(20260916)
        gain = MASTER_GAIN * 1.0                                   # vol 1, level 1
        for rho in (0.60, 0.95):
            for mode in ('full_toggle', 'random'):
                g = np.zeros((32, 32), np.uint8)
                x = b_engine(g, rho=rho, gain=gain)
                peak = 0.0
                for b in range(int(4 * SR / BLOCK)):
                    g = (1 - g) if mode == 'full_toggle' else (rng.random((32, 32)) < 0.5).astype(np.uint8)
                    x.update_field(g, None)
                    y, p, n_clip = x.render_float(gain)
                    self.assertEqual(n_clip, 0, (rho, mode))
                    self.assertTrue(np.all(np.isfinite(y)))
                    peak = max(peak, p)
                self.assertLess(peak, 1.0, (rho, mode))

    def test_rejects_other_contexts_and_registry_hints(self):
        with self.assertRaises(ValueError):
            registry.create(en.ENGINE_ID, EngineContext(48000, BLOCK, 2, 110.0, 1.0, 6.0), dict(rho=0.88))
        with self.assertRaises(ValueError):
            registry.create(en.ENGINE_ID, EngineContext(SR, 256, 2, 110.0, 1.0, 6.0), dict(rho=0.88))
        spec = registry.get(en.ENGINE_ID)
        self.assertEqual([p[0] for p in spec.params], ['rho'])
        self.assertEqual(spec.params[0][2:], (0.60, 0.95, False, 0.88))
        ov = spec.overlay(dict(rho=0.88), 32, 32)
        self.assertEqual(len(ov['labels']), 8)
        self.assertEqual(len(ov['circles']), 16)
        e = b_engine(grid(PULSAR_LIKE))
        e.render(GAIN, 0)
        d = e.display()
        for key in ('events', 'e', 'level', 'rho', 'rho_target', 'ramp_left', 'model'):
            self.assertIn(key, d)
        self.assertEqual(len(d['events']), 8)
        self.assertGreater(max(d['level']), 0.0)

    def test_engine_timing_budget(self):
        import time
        g = grid(PULSAR_LIKE)
        e = b_engine(g)
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
    def test_headless_draw_overlay_and_display_for_both_sides(self):
        import time
        import pygame
        import demo_bench as db
        from casynth_lab import DemoRunner, load_scene
        from casynth_lab.audio_out import LiveEngine
        scene = load_scene(os.path.join(ROOT, 'demos', 'n2_rhythm.json'))
        runner = DemoRunner(scene)
        eng = LiveEngine(runner, sink=lambda m, b: None)
        pygame.init()
        app = db.BenchApp(scene, eng)
        self.assertEqual(app.engine_rows, 5)
        self.assertIn(en.ENGINE_ID, app.engine_btns)
        self.assertIn(gp.ENGINE_ID, app.engine_btns)
        self.assertGreater(app.height - app.footer_y, 40)
        for eid, params in ((gp.ENGINE_ID, GUTTER_PARAMS), (en.ENGINE_ID, dict(rho=0.88))):
            data = app._overlay_runs(eid, params)
            self.assertEqual(data['lines'], [])
            self.assertEqual(len(data['circles']), 16)
            self.assertEqual(len(data['labels']), 8)
            self.assertTrue(data['text'])
        screen = pygame.Surface((app.width, app.height))
        font, small = pygame.font.SysFont(db.FONT_NAMES, 17), pygame.font.SysFont(db.FONT_NAMES, 14)
        eng.start()
        try:
            app.draw(screen, font, small)
            snap = eng.snapshot()
            self.assertEqual(snap['display']['A']['readout'], 'periodic')
            self.assertIn('events', snap['display']['B'])
            eng.post('select', side='B')
            t0 = time.time()
            while time.time() - t0 < 5 and eng.snapshot()['selected'] != 'B':
                time.sleep(0.02)
            self.assertEqual(eng.snapshot()['selected'], 'B')
            app.draw(screen, font, small)
            app._draw_display(screen, small, dict(events=[0.5] * 8, e=[1.0] * 8, level=[0.1] * 8,
                                                  rho=0.7, rho_target=0.9, ramp_left=100, model='x'),
                              app.panel_x, app.params_y + 4 * db.ROW_H + 6)
        finally:
            eng.stop()
            pygame.quit()


class CatalogTests(unittest.TestCase):
    def test_scenes_periods_and_events_match_the_req(self):
        from casynth_engine import step
        from demos.build_n2_events import CASES, scene_for, write_scenes
        import json
        pf_path = os.path.join(ROOT, 'memory', 'research', 'network-n2-preflight-2026-09-16.json')
        pf = json.load(open(pf_path, encoding='utf-8')) if os.path.isfile(pf_path) else None
        names = dict(pulse='n2_rhythm', travel='n2_travel', growth='n2_growth')
        want = dict(n2_rhythm=(3, 6.0, (32, 56)), n2_travel=(128, 16.0, (4, 4)), n2_growth=(None, 6.0, None))
        for case in CASES:
            doc = scene_for(case)
            path = os.path.join(ROOT, 'demos', case['id'] + '.json')
            self.assertEqual(json.load(open(path, encoding='utf-8')), doc)
            self.assertEqual((doc['variants']['A']['engine_id'], doc['variants']['B']['engine_id']),
                             (gp.ENGINE_ID, en.ENGINE_ID))
            self.assertEqual(doc['variants']['B']['engine_params'], dict(rho=0.88))
            self.assertEqual(doc['variants']['A']['engine_params'],
                             dict(scale=1.0, depth=1.0, interaction=127, freeze_ca=0))
            g = grid(case['cells'])
            period, rate, ev_range = want[case['id']]
            self.assertEqual(case['rate'], rate)
            seen = {g.tobytes(): 0}
            cur, events, pops = g, [], [int(g.sum())]
            found = None
            for gen in range(1, 130):
                nxt = step(cur)
                events.append(int((nxt != cur).sum()))
                pops.append(int(nxt.sum()))
                if found is None and nxt.tobytes() in seen:
                    found = gen - seen[nxt.tobytes()]
                seen.setdefault(nxt.tobytes(), gen)
                cur = nxt
            self.assertEqual(found, period, case['id'])
            if ev_range:
                self.assertEqual((min(events[:72]), max(events[:72])), ev_range)
            if case['id'] == 'n2_growth':
                self.assertEqual((pops[72], events[72]), (71, 66))
            if pf:
                ref = next(c for c in pf['cases'] if names[c['id']] == case['id'])
                self.assertEqual(sorted(map(tuple, ref['cells'])), sorted(map(tuple, case['cells'])))
        with open(os.path.join(ROOT, 'run_network_n2.bat'), encoding='ascii') as f:
            bat = f.read()
        self.assertIn('n2_rhythm.json', bat)
        self.assertIn('network_n2_events_2026_09_16', bat)
        self.assertNotIn('--live', bat)

    def test_catalog_build_replay_continue_notes_and_no_overwrite(self):
        from casynth_lab import DemoRunner, scene_from_doc
        from casynth_lab.catalog import Catalog
        from demos.build_n1_demos import record_offline
        from demos.build_n2_events import CASES, scene_for, build
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
                self.assertGreater(continued.gen, gen0)                  # the CA keeps evolving
                firsts.append(first)
                with self.assertRaises(SystemExit):
                    build(tmp)
                self.assertIn('User feedback: test.', cat.load(rid).notes)
            self.assertEqual(len(set(firsts)), len(CASES))


if __name__ == '__main__':
    unittest.main()
