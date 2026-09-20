#!/usr/bin/env python3
"""Laplace carriers gates (REQ memory/req-laplace-carriers-2026-09-20.md section 7):

  - the shared spectral base: the engine's (f_j, a_j) are the baseline analysis, equal to
    the Researcher preflight on every phase of the Jam, and every one of the seven
    settings acts (dyn checked where it acts: shape > 0 with a non-empty events field)
  - the waveform law: Wave bank / Sine is the existing `laplacian` engine BYTE-EXACT over
    the whole 12 s Jam (generation changes and tails included); Saw / Square equal the
    direct band-limited finite sum with coherent h*theta phases; the table error stays
    below -60 dB at low, high and non-integer frequencies; b(nu) is applied per WAVE, not
    per f0 (1760 Hz and a high mode); coincident modes add with their phases
  - Filter: the mask, A = sqrt(sum a^2) and the first 16 harmonic amplitudes equal the
    preflight and an independent direct computation; one mask serves Saw and Square;
    D = 0 is the plain wave at that scale; an empty / one-cell figure stays silent; width
    and depth act on a paused field without re-initialising the carrier phase
  - time: no strike from a repeated analysis, a smooth waveform / method change that a new
    change turns around instead of cutting, the paused field keeps sounding, Stop is
    silent, Continue (also from the middle of a crossfade) is byte-exact
  - the six scenes: variants, per-variant side gain, levels within 1 dB, no clip, and the
    block budget of both sides after warm-up

    python tests/test_laplace_carriers.py

Stdlib runner (pytest is not installed).  Scratch files: artifacts/_lc_tests/.
"""
import json
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from casynth_config import SR, MASTER_GAIN, MAX_VOICES                        # noqa: E402
from casynth_engine import step, events_field, analyse                        # noqa: E402
from casynth_lab import BLOCK, registry, DemoRunner, load_scene, scene_from_doc, render_offline  # noqa: E402
from casynth_lab.engine_api import EngineContext, supports_snapshot           # noqa: E402
from casynth_lab import laplace_carriers as lc                                # noqa: E402
from demos.build_laplace_carriers import (CASES, SIDE_GAIN, SPECTRUM, variant, jam_cells,  # noqa: E402
                                          F0_HZ, RATE_HZ, SECONDS, WIDTH_OCT, DEPTH_DB)

PREFLIGHT = os.path.join(ROOT, 'memory', 'research', 'laplace-carriers-preflight-2026-09-20.json')
F0 = F0_HZ
CTX = EngineContext(SR, BLOCK, 2, F0, 1.0, RATE_HZ)
GAIN = MASTER_GAIN * 0.7
BLOCKS_PER_GEN = int(round(SR / RATE_HZ / BLOCK))


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def jam_grid():
    g = np.zeros((32, 32), np.uint8)
    for r, c in jam_cells():
        g[r, c] = 1
    return g


def make(method, waveform, spectrum=None, width=WIDTH_OCT, depth=DEPTH_DB):
    p = dict(method=method, waveform=waveform, filter_width_oct=width,
             filter_depth_db=depth, **(spectrum or SPECTRUM))
    return registry.create(lc.ENGINE_ID, CTX, p)


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-300))


def rms(y):
    y = np.asarray(y, np.float64)
    return math.sqrt(float(np.mean(y * y))) if y.size else 0.0


class SpectralBase(unittest.TestCase):
    """REQ 7.1: the same (f_j, a_j) as the baseline and the preflight."""

    def test_modes_equal_the_preflight_on_every_phase(self):
        doc = preflight()
        grid = jam_grid()
        for ph in doc['phases']:
            e = make(lc.METHOD_BANK, lc.WF_SAW)
            e.init(grid, None, GAIN)
            self.assertEqual(len(e.voices), len(ph['objects']))
            for v, o in zip(e.voices, ph['objects']):
                fr = np.asarray(v['freqs'])
                am = np.asarray(v['amps'])
                nz = fr > 0
                got = (fr[nz] / F0).tolist()
                self.assertEqual(len(got), len(o['ratios']))
                for a, b in zip(got, o['ratios']):
                    self.assertLess(abs(a - b), 1e-12)
                for a, b in zip(am[nz].tolist(), o['amplitudes']):
                    self.assertLess(abs(a - b), 1e-12)
            grid = step(grid)

    def test_both_methods_read_one_analysis(self):
        grid = jam_grid()
        ref = analyse(grid, F0, 'laplacian', dict(SPECTRUM), exc=None)[1]
        for method in (lc.METHOD_FILTER, lc.METHOD_BANK):
            e = make(method, lc.WF_SQUARE)
            e.init(grid, None, GAIN)
            self.assertEqual(len(e.voices), len(ref))
            for v, r in zip(e.voices, ref):
                self.assertTrue(np.array_equal(np.asarray(v['freqs']), np.asarray(r['freqs'])))
                self.assertTrue(np.array_equal(np.asarray(v['amps']), np.asarray(r['amps'])))

    def test_every_spectrum_setting_acts(self):
        # each setting on a field where it ACTS: fullshape needs a figure larger than the
        # 8x8 extract window, every other one moves the Jam itself
        jam = step(jam_grid())
        jam_exc = events_field(jam_grid(), jam)
        big = np.zeros((32, 32), np.uint8)
        big[8:20, 8:20] = 1
        big[9, 9] = 0                                  # not a perfectly regular graph
        base = dict(n=4, spread=0.5, alpha=0.5, shape=0.5, harm=0.2, fullshape=1, dyn=0.0)
        changes = dict(n=6, spread=1.0, alpha=1.5, shape=1.0, harm=1.0, fullshape=0, dyn=1.0)

        def spectra(sp, grid, exc):
            e = make(lc.METHOD_FILTER, lc.WF_SAW, spectrum=sp)
            e.init(grid, exc, GAIN)
            return ([np.asarray(v['freqs']).copy() for v in e.voices],
                    [np.asarray(v['amps']).copy() for v in e.voices])

        for key, value in changes.items():
            grid, exc = (big, None) if key == 'fullshape' else (jam, jam_exc)
            f0s, a0s = spectra(base, grid, exc)
            f1s, a1s = spectra(dict(base, **{key: value}), grid, exc)
            moved = (len(f1s) != len(f0s)
                     or any(not np.array_equal(x, y) for x, y in zip(f0s, f1s))
                     or any(not np.array_equal(x, y) for x, y in zip(a0s, a1s)))
            self.assertTrue(moved, f"setting {key} changed nothing")

    def test_dyn_acts_with_shape_and_events(self):
        grid = jam_grid()
        new = step(grid)
        exc = events_field(grid, new)
        self.assertGreater(float(exc.sum()), 0.0)
        sp = dict(n=4, spread=0.0, alpha=1.0, shape=0.8, harm=0.0, fullshape=1, dyn=0.0)
        e0 = make(lc.METHOD_BANK, lc.WF_SAW, spectrum=sp)
        e0.init(new, exc, GAIN)
        sp1 = dict(sp, dyn=1.0)
        e1 = make(lc.METHOD_BANK, lc.WF_SAW, spectrum=sp1)
        e1.init(new, exc, GAIN)
        diff = max(float(np.max(np.abs(np.asarray(a['amps']) - np.asarray(b['amps']))))
                   for a, b in zip(e0.voices, e1.voices))
        self.assertGreater(diff, 1e-6)
        self.assertTrue(all(np.array_equal(np.asarray(a['freqs']), np.asarray(b['freqs']))
                            for a, b in zip(e0.voices, e1.voices)))


class WaveLaw(unittest.TestCase):
    """REQ 7.2 / 7.4: the waveform, its band limit and the table quality."""

    def test_band_limit_law(self):
        self.assertEqual(lc.band_limit1(1000.0), 1.0)
        self.assertEqual(lc.band_limit1(0.40 * SR), 1.0)
        self.assertEqual(lc.band_limit1(0.45 * SR), 0.0)
        self.assertEqual(lc.band_limit1(0.46 * SR), 0.0)
        mid = lc.band_limit1(0.425 * SR)
        self.assertAlmostEqual(mid, 0.5, places=12)
        self.assertTrue(0.0 < lc.band_limit1(0.41 * SR) < 1.0)

    def test_harmonics_are_limited_per_wave_not_per_f0(self):
        # every wave gets its OWN ceiling: h * f < 0.45 * sr
        for f, expect in ((110.0, 180), (1760.0, 11), (5000.0, 3), (19844.0, 1), (20000.0, 0)):
            h = lc.harmonic_indices(lc.WF_SAW, f)
            self.assertEqual(len(h), expect, f"saw at {f} Hz")
            if len(h):
                self.assertLess(h[-1] * f, 0.45 * SR)
                self.assertGreaterEqual((h[-1] + 1) * f, 0.45 * SR)
        self.assertEqual(len(lc.harmonic_indices(lc.WF_SQUARE, 1760.0)), 6)   # odd only
        self.assertEqual(len(lc.harmonic_indices(lc.WF_SINE, 110.0)), 1)

    def test_coefficients_are_c_h_times_band_limit(self):
        for wf in (lc.WF_SINE, lc.WF_SAW, lc.WF_SQUARE):
            for f in (110.0, 987.65):
                h, c = lc.harmonic_coeffs(wf, f)
                if wf == lc.WF_SINE:
                    self.assertEqual(list(h), [1])
                else:
                    if wf == lc.WF_SQUARE:
                        self.assertTrue(all(x % 2 == 1 for x in h))
                    want = (1.0 / h) * lc.band_limit(h * f)
                    self.assertTrue(np.allclose(c, want, rtol=0, atol=0))
                self.assertAlmostEqual(float(c[0]), 1.0, places=15)   # fundamental = 1

    def test_table_error_below_minus_60_db(self):
        for wf in (lc.WF_SAW, lc.WF_SQUARE):
            for f in (55.0, 110.0, 440.0, 1760.0, 3457.31, 9871.3):
                n = 4096
                ref = lc.direct_wave(wf, f, 0.3, n)
                got = lc.table_wave(lc.wavetable(wf, f), 0.3, 2 * np.pi * f / SR, n)
                rel = db(rms(got - ref) / max(rms(ref), 1e-300))
                self.assertLess(rel, -60.0, f"{lc.WAVE_NAMES[wf]} at {f} Hz: {rel:.1f} dB")

    def test_no_energy_above_the_band_limit(self):
        for wf in (lc.WF_SAW, lc.WF_SQUARE):
            f = 1760.0
            n = 1 << 15
            y = lc.table_wave(lc.wavetable(wf, f), 0.0, 2 * np.pi * f / SR, n)
            Y = np.abs(np.fft.rfft(y * np.hanning(n)))
            fr = np.fft.rfftfreq(n, 1.0 / SR)
            above = Y[fr >= 0.45 * SR]
            self.assertLess(db(float(above.max()) / float(Y.max())), -80.0)

    def test_sine_bank_is_the_sum_of_modes(self):
        grid = jam_grid()
        e = make(lc.METHOD_BANK, lc.WF_SINE)
        e.init(grid, None, GAIN)
        for _ in range(30):
            e.render(GAIN, 0)
        pool = e.bank.pool
        slots = [k for k in range(1, len(pool.freq_slots)) if pool.freq_slots[k] > 0 and e.bank.amp_cur[k] > 1e-4]
        ph = e.bank.phase.copy()
        amp = e.bank.amp_cur.copy()
        L, R = e.bank.render(lc.WF_SINE, lc.WF_SINE, BLOCK, SR)
        idx = np.arange(BLOCK, dtype=float)
        ref = np.zeros(BLOCK)
        for k in slots:
            f = pool.freq_slots[k]
            a = amp[k] + (pool.amp_tgt[k] - amp[k]) * 0.0     # steady state: amp_cur == amp_tgt
            self.assertAlmostEqual(amp[k], pool.amp_tgt[k], places=12)
            ref += a * lc.band_limit1(f) * np.sin(ph[k] + 2 * np.pi * f / SR * idx)
        ref *= math.cos(0.5 * np.pi / 2.0)
        self.assertLess(float(np.max(np.abs(L - ref))), 1e-12)

    def test_bank_slot_equals_the_direct_sum(self):
        grid = jam_grid()
        for wf in (lc.WF_SAW, lc.WF_SQUARE):
            e = make(lc.METHOD_BANK, wf)
            e.init(grid, None, GAIN)
            for _ in range(30):
                e.render(GAIN, 0)
            pool = e.bank.pool
            slots = [k for k in range(1, len(pool.freq_slots))
                     if pool.freq_slots[k] > 0 and e.bank.amp_cur[k] > 1e-4]
            self.assertGreaterEqual(len(slots), 3)
            ph = e.bank.phase.copy()
            amp = e.bank.amp_cur.copy()
            L, _R = e.bank.render(wf, wf, BLOCK, SR)
            ref = np.zeros(BLOCK)
            for k in slots:
                ref += amp[k] * lc.direct_wave(wf, pool.freq_slots[k], ph[k], BLOCK)
            ref *= math.cos(0.5 * np.pi / 2.0)
            rel = db(rms(L - ref) / max(rms(ref), 1e-300))
            self.assertLess(rel, -60.0, f"{lc.WAVE_NAMES[wf]}: {rel:.1f} dB")

    def test_bank_lines_sit_on_h_times_mode_frequency(self):
        # the Jam phase with a non-integer ratio 1.80194 -- no forced common grid
        grid = jam_grid()
        e = make(lc.METHOD_BANK, lc.WF_SAW)
        e.init(grid, None, GAIN)
        y = []
        for _ in range(400):
            e.update_field(grid, None)          # frozen field: a steady spectrum
            buf, _p, _c = e.render(GAIN, 0)
            y.append(buf[:, 0].astype(float))
        y = np.concatenate(y)[BLOCK * 100:]
        n = 1 << 16
        Y = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
        fr = np.fft.rfftfreq(n, 1.0 / SR)
        f_mode = F0 * 1.8019377358048378
        for target in (f_mode, 2 * f_mode, 3 * f_mode):
            band = (fr > target - 6) & (fr < target + 6)
            self.assertGreater(db(float(Y[band].max()) / float(Y.max())), -60.0,
                               f"no line at {target:.1f} Hz")

    def test_coincident_modes_add_with_phase(self):
        # generation 1 of the Jam holds a figure with two equal mode frequencies
        grid = step(jam_grid())
        e = make(lc.METHOD_BANK, lc.WF_SINE)
        e.init(grid, None, GAIN)
        pairs = [v for v in e.voices
                 if np.count_nonzero(np.asarray(v['freqs']) > 0) == 2
                 and abs(float(v['freqs'][0]) - float(v['freqs'][1])) < 1e-9]
        self.assertTrue(pairs, "the preflight phase with two coincident modes is gone")
        for _ in range(30):
            e.render(GAIN, 0)
        pool = e.bank.pool
        same = [k for k in range(1, len(pool.freq_slots))
                if abs(pool.freq_slots[k] - float(pairs[0]['freqs'][0])) < 1e-9
                and e.bank.amp_cur[k] > 1e-4]
        self.assertGreaterEqual(len(same), 2, "coincident modes were merged away")
        self.assertLess(abs(e.bank.phase[same[0]] - e.bank.phase[same[1]]), 1e-9)


class FilterLaw(unittest.TestCase):
    """REQ 7.3: the mask, its scale and the silence rules."""

    def test_mask_and_amplitudes_equal_the_preflight(self):
        doc = preflight()
        p = doc['params']
        for ph in doc['phases']:
            for o in ph['objects']:
                fr = np.array([F0 * r for r in o['ratios']])
                am = np.array(o['amplitudes'])
                H, A = lc.filter_mask(fr, am, F0, p['filter_width_oct'], p['filter_depth_db'])
                self.assertIsNotNone(H)
                self.assertAlmostEqual(A, math.sqrt(float((am * am).sum())), places=15)
                got_db = 20.0 * np.log10(H[:16])
                self.assertLess(float(np.max(np.abs(got_db - np.array(o['filter_db_first16'])))), 1e-9)
                for name, wf in (('sine', lc.WF_SINE), ('saw', lc.WF_SAW), ('square', lc.WF_SQUARE)):
                    g = lc.carrier_coeffs(wf, F0)
                    amp16 = (A * g * H)[:16]
                    want = np.array(o['waveforms'][name]['filter_amplitudes_first16'])
                    self.assertLess(float(np.max(np.abs(amp16 - want))), 1e-12)
                    self.assertEqual(int(np.count_nonzero(A * g * H > 0)),
                                     o['waveforms'][name]['filter_nonzero_harmonics'])

    def test_mask_is_an_independent_direct_computation(self):
        fr = np.array([110.0, 198.213, 254.77])
        am = np.array([1.0, 0.6, 0.25])
        sigma, depth = 0.42, 18.0
        H, A = lc.filter_mask(fr, am, F0, sigma, depth)
        K = lc.carrier_harmonics(F0)
        want = np.zeros(K)
        for i in range(K):
            k = i + 1
            want[i] = sum(a * math.exp(-0.5 * (math.log2(k / (f / F0)) / sigma) ** 2)
                          for f, a in zip(fr, am))
        want = np.power(10.0, -depth * (1.0 - want / want.max()) / 20.0)
        self.assertLess(float(np.max(np.abs(H - want))), 1e-12)
        self.assertLessEqual(float(H.max()), 1.0 + 1e-15)
        self.assertGreaterEqual(float(H.min()), 10 ** (-depth / 20.0) - 1e-15)

    def test_one_mask_serves_saw_and_square(self):
        grid = jam_grid()
        specs = {}
        for wf in (lc.WF_SAW, lc.WF_SQUARE):
            e = make(lc.METHOD_FILTER, wf)
            e.init(grid, None, GAIN)
            g = lc.carrier_coeffs(wf, F0)
            specs[wf] = [None if s is None else s / np.where(g > 0, g, 1.0) for s in e._specs]
        for a, b in zip(specs[lc.WF_SAW], specs[lc.WF_SQUARE]):
            if a is None:
                self.assertIsNone(b)
                continue
            odd = np.arange(len(a)) % 2 == 0          # k odd (index k-1) -> present in both
            self.assertTrue(np.allclose(a[odd], b[odd], rtol=0, atol=1e-15))

    def test_depth_zero_is_the_plain_wave(self):
        grid = jam_grid()
        e = make(lc.METHOD_FILTER, lc.WF_SAW, depth=0.0)
        e.init(grid, None, GAIN)
        for _ in range(20):
            e.render(GAIN, 0)
        live = [v for v in range(MAX_VOICES) if e.filt.amp_cur[v] > 1e-4]
        self.assertTrue(live)
        th = e.filt.th.copy()
        amp = e.filt.amp_cur.copy()
        L, _R = e.filt.render(BLOCK, SR)
        ref = np.zeros(BLOCK)
        for v in live:                       # D = 0: every carrier is the plain wave x A
            ref += amp[v] * lc.direct_wave(lc.WF_SAW, F0, th[v], BLOCK)
        ref *= math.cos(0.5 * np.pi / 2.0)
        self.assertLess(float(np.max(np.abs(L - ref))), 1e-12)

    def test_empty_and_single_cell_are_silent(self):
        for cells in ([], [(5, 5)]):
            g = np.zeros((32, 32), np.uint8)
            for r, c in cells:
                g[r, c] = 1
            for method in (lc.METHOD_FILTER, lc.METHOD_BANK):
                e = make(method, lc.WF_SAW, depth=0.0)
                e.init(g, None, GAIN)
                for _ in range(10):
                    buf, peak, n_clip = e.render(GAIN, 0)
                    self.assertEqual(int(np.abs(buf).max()), 0)
                    self.assertEqual(n_clip, 0)

    def test_filter_lines_sit_on_the_carrier_harmonics(self):
        grid = jam_grid()
        for wf, evens in ((lc.WF_SAW, True), (lc.WF_SQUARE, False)):
            e = make(lc.METHOD_FILTER, wf, depth=6.0)
            e.init(grid, None, GAIN)
            y = []
            for _ in range(400):
                e.update_field(grid, None)
                buf, _p, _c = e.render(GAIN, 0)
                y.append(buf[:, 0].astype(float))
            y = np.concatenate(y)[BLOCK * 100:]
            n = 1 << 16
            Y = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
            fr = np.fft.rfftfreq(n, 1.0 / SR)

            def rel(target):
                band = (fr > target - 6) & (fr < target + 6)
                return db(float(Y[band].max()) / float(Y.max()))

            self.assertGreater(rel(F0), -40.0)
            self.assertGreater(rel(3 * F0), -60.0)
            if evens:
                self.assertGreater(rel(2 * F0), -60.0)
            else:
                self.assertLess(rel(2 * F0), -70.0)
            # never a line on the inharmonic mode itself
            self.assertLess(rel(F0 * 1.8019377358048378), -70.0)

    def test_width_and_depth_act_on_a_paused_field_without_touching_the_phase(self):
        grid = jam_grid()
        e = make(lc.METHOD_FILTER, lc.WF_SAW)
        e.init(grid, None, GAIN)
        for _ in range(20):
            e.render(GAIN, 0)
        th = e.filt.th.copy()
        spec = [None if s is None else s.copy() for s in e._specs]
        for key, value in (('filter_depth_db', 6.0), ('filter_width_oct', 0.9)):
            p = dict(e.params)
            p[key] = value
            e.set_params(p)                      # no update_field: the field is paused
            self.assertTrue(np.array_equal(th, e.filt.th))
            self.assertFalse(np.array_equal(spec[0], e._specs[0]), f"{key} changed nothing")
            spec = [None if s is None else s.copy() for s in e._specs]


class TimeBehaviour(unittest.TestCase):
    """REQ 7.5 / 7.7: envelopes, smooth changes, snapshot."""

    def test_bank_sine_is_the_baseline_byte_exact(self):
        grid = jam_grid()
        base = registry.create('laplacian', CTX, dict(SPECTRUM))
        cand = make(lc.METHOD_BANK, lc.WF_SINE)
        base.init(grid, None, GAIN)
        cand.init(grid, None, GAIN)
        for gen in range(int(SECONDS * RATE_HZ)):
            for _ in range(BLOCKS_PER_GEN):
                a, _pa, _ca = base.render(GAIN, 0)
                b, _pb, _cb = cand.render(GAIN, 0)
                self.assertTrue(np.array_equal(a, b), f"generation {gen}")
            new = step(grid)
            exc = events_field(grid, new)
            grid = new
            base.update_field(grid, exc)
            cand.update_field(grid, exc)

    def test_repeated_analysis_of_an_unchanged_field_changes_nothing(self):
        grid = jam_grid()
        quiet = make(lc.METHOD_FILTER, lc.WF_SAW)
        noisy = make(lc.METHOD_FILTER, lc.WF_SAW)
        quiet.init(grid, None, GAIN)
        noisy.init(grid, None, GAIN)
        for _ in range(60):
            noisy.update_field(grid, None)       # the bench re-analysing the same field
            noisy.set_params(dict(noisy.params))
            a, _p, _c = quiet.render(GAIN, 0)
            b, _p2, _c2 = noisy.render(GAIN, 0)
            self.assertTrue(np.array_equal(a, b))

    def test_method_and_waveform_changes_are_smooth(self):
        grid = jam_grid()
        for switch in (dict(waveform=lc.WF_SQUARE), dict(method=lc.METHOD_FILTER)):
            e = make(lc.METHOD_BANK, lc.WF_SAW)
            e.init(grid, None, GAIN)
            y = []
            for i in range(60):
                if i == 30:
                    p = dict(e.params)
                    p.update(switch)
                    e.set_params(p)
                buf, _p, _c = e.render(GAIN, 0)
                y.append(buf[:, 0].astype(float))
            y = np.concatenate(y)
            d = np.abs(np.diff(y))
            seam = float(d[30 * BLOCK - 1])
            self.assertLessEqual(seam, float(np.percentile(d, 99.9)) * 1.5,
                                 f"{switch}: a step of {seam} at the switch")

    def test_a_second_switch_turns_the_crossfade_around(self):
        grid = jam_grid()
        e = make(lc.METHOD_BANK, lc.WF_SAW)
        e.init(grid, None, GAIN)
        for _ in range(10):
            e.render(GAIN, 0)
        self.assertEqual(e._mix, 1.0)
        p = dict(e.params, method=lc.METHOD_FILTER)
        e.set_params(p)
        e.render(GAIN, 0)
        mid = e._mix
        self.assertTrue(0.0 < mid < 1.0, "the method switch is not a crossfade")
        e.set_params(dict(e.params, method=lc.METHOD_BANK))
        e.render(GAIN, 0)
        self.assertGreater(e._mix, mid)          # continued from where it was, not cut
        self.assertLessEqual(e._mix, 1.0)

    def test_snapshot_continue_is_byte_exact(self):
        for method in (lc.METHOD_FILTER, lc.METHOD_BANK):
            for wf in (lc.WF_SINE, lc.WF_SAW, lc.WF_SQUARE):
                self._continue_case(method, wf, switch_at=None)

    def test_snapshot_continue_inside_a_crossfade(self):
        self._continue_case(lc.METHOD_BANK, lc.WF_SAW, switch_at=BLOCKS_PER_GEN * 2 + 3)

    def _continue_case(self, method, wf, switch_at):
        grid = jam_grid()
        e = make(method, wf)
        e.init(grid, None, GAIN)
        self.assertTrue(supports_snapshot(e))
        cut = BLOCKS_PER_GEN * 2 + 4
        seq, state, cut_grid, cut_exc = [], None, None, None
        exc = None
        nb = 0
        for gen in range(6):
            for _ in range(BLOCKS_PER_GEN):
                if switch_at is not None and nb == switch_at:
                    e.set_params(dict(e.params, method=lc.METHOD_FILTER))
                if nb == cut:
                    state = e.export_state()
                    cut_grid = grid.copy()
                    cut_exc = None if exc is None else exc.copy()
                buf, _p, _c = e.render(GAIN, 0)
                seq.append(buf)
                nb += 1
            new = step(grid)
            exc = events_field(grid, new)
            grid = new
            e.update_field(grid, exc)
        cont = make(method, wf)
        cont.restore_state(cut_grid, cut_exc, state)
        grid2 = cut_grid.copy()
        nb2 = cut
        for _ in range(BLOCKS_PER_GEN - (cut % BLOCKS_PER_GEN)):
            buf, _p, _c = cont.render(GAIN, 0)
            self.assertTrue(np.array_equal(buf, seq[nb2]), f"{method}/{wf} block {nb2}")
            nb2 += 1
        for _gen in range(cut // BLOCKS_PER_GEN + 1, 6):
            new = step(grid2)
            exc2 = events_field(grid2, new)
            grid2 = new
            cont.update_field(grid2, exc2)
            for _ in range(BLOCKS_PER_GEN):
                buf, _p, _c = cont.render(GAIN, 0)
                self.assertTrue(np.array_equal(buf, seq[nb2]), f"{method}/{wf} block {nb2}")
                nb2 += 1

    def test_restore_rejects_a_foreign_state(self):
        grid = jam_grid()
        e = make(lc.METHOD_BANK, lc.WF_SAW)
        e.init(grid, None, GAIN)
        st = e.export_state()
        bad = dict(st, version=99)
        with self.assertRaises(ValueError):
            make(lc.METHOD_BANK, lc.WF_SAW).restore_state(grid, None, bad)
        bad = dict(st, engine_id='laplacian')
        with self.assertRaises(ValueError):
            make(lc.METHOD_BANK, lc.WF_SAW).restore_state(grid, None, bad)
        other = EngineContext(SR, BLOCK, 2, 55.0, 1.0, RATE_HZ)
        with self.assertRaises(ValueError):
            registry.create(lc.ENGINE_ID, other, dict(e.params)).restore_state(grid, None, st)


class BenchIntegration(unittest.TestCase):
    """The registry entry, the scenes and the bench transport."""

    def test_registry_entry(self):
        spec = registry.get(lc.ENGINE_ID)
        self.assertEqual(spec.label, 'Laplace waves')
        self.assertEqual(spec.choices['method'], ('Filter', 'Wave bank'))
        self.assertEqual(spec.choices['waveform'], ('Sine', 'Saw', 'Square'))
        self.assertEqual(registry.spectrum_keys('laplacian', lc.ENGINE_ID),
                         tuple(p[0] for p in registry.get('laplacian').params))
        off = lc.inactive(dict(spec.defaults(), method=lc.METHOD_BANK))
        self.assertIn('filter_width_oct', off)
        self.assertIn('filter_depth_db', off)
        on = lc.inactive(dict(spec.defaults(), method=lc.METHOD_FILTER))
        self.assertNotIn('filter_width_oct', on)

    def test_scenes_carry_the_req_conditions(self):
        seen = {}
        for case in CASES:
            scene = load_scene(os.path.join(ROOT, 'demos', case['id'] + '.json'))
            self.assertEqual((scene.rows, scene.cols), (32, 32))
            self.assertEqual(scene.rate_hz, RATE_HZ)
            self.assertEqual(scene.f0_hz, F0_HZ)
            self.assertEqual(scene.cells, [(r, c) for r, c in jam_cells()])
            for side, name in (('A', case['a']), ('B', case['b'])):
                eid, params = scene.variants[side]
                want_id, want_p = variant(name)
                self.assertEqual(eid, want_id)
                self.assertEqual(params, want_p)
                self.assertEqual(scene.side_gain[side], SIDE_GAIN[name])
                # one variant -> one engine/params/gain everywhere it appears
                seen.setdefault(name, (eid, params, scene.side_gain[side]))
                self.assertEqual(seen[name], (eid, params, scene.side_gain[side]))

    def test_paused_field_keeps_sounding_and_stop_is_silent(self):
        scene = load_scene(os.path.join(ROOT, 'demos', 'lc_saw_baseline_bank.json'))
        runner = DemoRunner(scene)
        runner.post('start', at=0)
        for _ in range(60):
            runner.next_block()
        runner.post('pause', at=runner.out_samples, on=True)
        gen = runner.snapshot()['gen']
        peak = 0
        for _ in range(60):
            b = runner.next_block()
            peak = max(peak, int(np.abs(b.get('B')).max()))
        self.assertGreater(peak, 0)
        self.assertEqual(runner.snapshot()['gen'], gen)
        runner.post('stop', at=runner.out_samples)
        for _ in range(5):
            runner.next_block()
        for _ in range(10):
            b = runner.next_block()
            self.assertEqual(int(np.abs(b.get('B')).max()), 0)

    def test_painting_does_not_break_the_sound(self):
        scene = load_scene(os.path.join(ROOT, 'demos', 'lc_square_baseline_filter.json'))
        runner = DemoRunner(scene)
        runner.post('start', at=0)
        for i in range(200):
            if 40 <= i < 80:
                runner.post('set_cell', at=runner.out_samples, r=20 + (i % 3), c=5 + (i % 7), v=1)
            b = runner.next_block()
            self.assertTrue(np.all(np.isfinite(b.get('B').astype(float))))
        self.assertEqual(runner.snapshot()['clip_blocks'], dict(A=0, B=0))

    def test_levels_of_the_ready_tracks(self):
        scene = load_scene(os.path.join(ROOT, 'demos', 'lc_saw_baseline_bank.json'))
        pcm, runner = render_offline(scene, SECONDS, output=('A', 'B'))
        levels = {}
        for side in ('A', 'B'):
            y = pcm[side].astype(np.float64) / 32767.0
            levels[side] = db(rms(y))
            self.assertLess(float(np.abs(y).max()), 1.0)
        self.assertLessEqual(abs(levels['A'] - levels['B']), 1.0)
        self.assertEqual(runner.snapshot()['clip_blocks'], dict(A=0, B=0))

    def test_block_budget_of_the_six_scenes(self):
        import time
        budget_ms = BLOCK / SR * 1000.0
        for case in CASES:
            scene = load_scene(os.path.join(ROOT, 'demos', case['id'] + '.json'))
            runner = DemoRunner(scene)
            runner.post('start', at=0)
            times = []
            for _ in range(int(4.0 * SR / BLOCK)):
                t0 = time.perf_counter()
                runner.next_block()
                times.append((time.perf_counter() - t0) * 1000.0)
            warm = np.array(times[int(1.0 * SR / BLOCK):])
            p99 = float(np.percentile(warm, 99))
            self.assertLess(p99, budget_ms, f"{case['id']}: p99 {p99:.2f} ms")
            self.assertEqual(runner.snapshot()['clip_blocks'], dict(A=0, B=0))


if __name__ == '__main__':
    unittest.main()
