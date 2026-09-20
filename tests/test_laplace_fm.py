#!/usr/bin/env python3
"""Laplace FM gates (REQ memory/req-laplace-fm-2026-09-20.md section 6):

  1 input modes  -- the engine's (f_j, a_j) ARE the baseline analysis and equal the
    Researcher preflight on every phase of the Jam; each of the seven settings acts in
    its own area; a figure with no non-zero mode makes no carrier (empty field, one
    cell), and the bench gain never reaches the index
  2 the law      -- one modulator of a known index against the Bessel expansion; several
    non-integer frequencies with unequal weights against the phase formula at 32x;
    coincident modulators with equal and with different phases (indices add coherently);
    the stationary peak deviation sum beta_j*f_j; joint modulation of ONE carrier, never
    a sum of independently modulated carriers
  3 zero and DC  -- FM depth 0 is the carrier sine at the same scale through the same
    output filter; the fixed blocker kills the constant a zero combination frequency
    makes, its closed form equals the scalar recursion, and its DC gain is exactly 0
  4 band / time  -- the reference converges (16x vs 32x better than -80 dB) and the
    implementation is within -60 dB of it: stationary at depths 0/1/2/4, with unequal
    weights, at 1760 Hz, and over the Jam WITH its transitions, tails and ramps; the
    filter delay is the stated 39 samples; knobs reset no phase; a released tail keeps
    its indices and is not cut; a second change does not leave a ramp hanging
  5-7 records    -- the three scenes, one factor per variant, levels within 0.5 dB, no
    clip, R and Wave bank / Saw byte-identical to the records already listened to,
    Continue exact (also mid-transition and while paused), and the block budget

    python tests/test_laplace_fm.py

Stdlib runner (pytest is not installed).
"""
import hashlib
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

from casynth_config import SR, MASTER_GAIN, MAX_VOICES, MAX_MODES_PER_OBJ, CHUNK_S  # noqa: E402
from casynth_engine import step, events_field, analyse                        # noqa: E402
from casynth_lab import BLOCK, registry, DemoRunner, load_scene, scene_from_doc  # noqa: E402
from casynth_lab.engine_api import EngineContext, supports_snapshot           # noqa: E402
from casynth_lab import laplace_fm as lfm                                     # noqa: E402
from demos import laplace_fm_reference as ref                                 # noqa: E402
from demos.build_laplace_fm import (CASES, SIDE_GAIN, FM_DEPTH, FM_PREFLIGHT,  # noqa: E402
                                    REFERENCE_RECORD, scene_for, variant, jam_cells)
from demos.build_laplace_carriers import (SPECTRUM, F0_HZ, RATE_HZ, SECONDS,   # noqa: E402
                                          CATALOG_ROOT as CARRIERS_ROOT)

F0 = F0_HZ
CTX = EngineContext(SR, BLOCK, 2, F0, 1.0, RATE_HZ)
GAIN = MASTER_GAIN * 0.7
BLOCKS_PER_GEN = int(round(SR / RATE_HZ / BLOCK))
N_BLOCKS_12S = int(math.ceil(SECONDS * SR / BLOCK))


def preflight():
    with open(FM_PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def jam_grid():
    g = np.zeros((32, 32), np.uint8)
    for r, c in jam_cells():
        g[r, c] = 1
    return g


def make(depth=FM_DEPTH, spectrum=None, f0=F0, oversample=lfm.OVERSAMPLE):
    ctx = CTX if f0 == F0 else EngineContext(SR, BLOCK, 2, f0, 1.0, RATE_HZ)
    return lfm.LaplaceFMEngine(ctx, dict(fm_depth=depth, **(spectrum or SPECTRUM)),
                               oversample=oversample)


def run_blocks(engine, n, gain=GAIN):
    """n blocks on the frozen field -> (float mono, int16 PCM, max pre-clip peak, clips)."""
    mono, pcm, peak, clips = [], [], 0.0, 0
    for i in range(n):
        buf, pk, nc = engine.render(gain, i * BLOCK)
        mono.append(engine.mono.copy())
        pcm.append(buf)
        peak = max(peak, float(pk))
        clips += (1 if nc else 0)
    return np.concatenate(mono), np.concatenate(pcm, axis=0), peak, clips


def run_jam(engine, seconds, gain=GAIN, grid=None):
    """The Jam evolving at RATE_HZ under the engine, exactly as the bench steps it."""
    g = jam_grid() if grid is None else grid.copy()
    exc = None
    engine.init(g, exc, gain)
    n = int(math.ceil(seconds * SR / BLOCK))
    mono, ca, gen = [], 0, 0
    for i in range(n):
        if ca >= (gen + 1) * (SR / RATE_HZ):
            new = step(g)
            exc = events_field(g, new)
            g = new
            gen += 1
            engine.update_field(g, exc)
        engine.render(gain, i * BLOCK)
        mono.append(engine.mono.copy())
        ca += BLOCK
    return np.concatenate(mono), gen


def single(freqs, amps, scale=None):
    """A hand-made analysis for ONE channel (white-box control of the modulators)."""
    fr = np.asarray(freqs, dtype=float)
    am = np.asarray(amps, dtype=float)
    live = (fr > 0.0) & (am > 0.0)
    a = float(math.sqrt(float((am[live] * am[live]).sum()))) if scale is None else float(scale)
    return [(fr, am, a)] + [None] * (MAX_VOICES - 1)


def junction_d2(y, block=BLOCK):
    """(max |second difference| at the block seams, max inside the blocks) -- a corner in
    the envelope shows up at a seam and nowhere else (the click probe's own metric)."""
    d2 = np.abs(np.diff(np.asarray(y, dtype=float), 2))
    seam = np.zeros(len(d2), dtype=bool)
    for k in range(block - 3, len(d2), block):
        seam[max(0, k):k + 4] = True
    inside = ~seam
    return (float(d2[seam].max()) if seam.any() else 0.0,
            float(d2[inside].max()) if inside.any() else 0.0)


class SpectralBase(unittest.TestCase):
    """REQ 6.1: the same (f_j, a_j) as the baseline and the preflight."""

    def test_modes_equal_the_preflight_on_every_phase(self):
        doc = preflight()
        rows = {(r['generation'], r['object_index']): r for r in doc['rows']}
        self.assertTrue(doc['core_hash_matches'])
        grid = jam_grid()
        for gen in (0, 1, 2):
            e = make()
            e.init(grid, None, GAIN)
            seen = 0
            for i, item in enumerate(e._mods):
                if item is None:
                    continue
                row = rows.get((gen, i))
                self.assertIsNotNone(row, f"gen {gen} object {i} missing from the preflight")
                fr, am, _a = item
                live = (fr > 0.0) & (am > 0.0)
                got = (fr[live] / F0).tolist()
                self.assertEqual(len(got), len(row['ratios']))
                for a, b in zip(got, row['ratios']):
                    self.assertLess(abs(a - b), 1e-12)
                for a, b in zip(am[live].tolist(), row['amplitudes']):
                    self.assertLess(abs(a - b), 1e-12)
                seen += 1
            self.assertGreater(seen, 0)
            grid = step(grid)

    def test_the_engine_reads_the_baseline_analysis(self):
        grid = jam_grid()
        base = analyse(grid, F0, 'laplacian', dict(SPECTRUM), exc=None)[1]
        e = make()
        e.init(grid, None, GAIN)
        self.assertEqual(len(e.voices), len(base))
        for v, r in zip(e.voices, base):
            self.assertTrue(np.array_equal(np.asarray(v['freqs']), np.asarray(r['freqs'])))
            self.assertTrue(np.array_equal(np.asarray(v['amps']), np.asarray(r['amps'])))

    def test_every_spectrum_setting_acts(self):
        jam = step(jam_grid())
        jam_exc = events_field(jam_grid(), jam)
        big = np.zeros((32, 32), np.uint8)
        big[8:20, 8:20] = 1
        big[9, 9] = 0
        base = dict(n=4, spread=0.5, alpha=0.5, shape=0.5, harm=0.2, fullshape=1, dyn=0.0)
        changes = dict(n=6, spread=1.0, alpha=1.5, shape=1.0, harm=1.0, fullshape=0, dyn=1.0)

        def mods(sp, grid, exc):
            e = make(spectrum=sp)
            e.init(grid, exc, GAIN)
            return [None if m is None else (np.asarray(m[0]).copy(), np.asarray(m[1]).copy())
                    for m in e._mods]

        for key, value in changes.items():
            grid, exc = (big, None) if key == 'fullshape' else (jam, jam_exc)
            a = mods(base, grid, exc)
            b = mods(dict(base, **{key: value}), grid, exc)
            moved = any((x is None) != (y is None)
                        or (x is not None and (not np.array_equal(x[0], y[0])
                                               or not np.array_equal(x[1], y[1])))
                        for x, y in zip(a, b))
            self.assertTrue(moved, f"setting {key} changed nothing")

    def test_no_non_zero_mode_no_carrier(self):
        """Empty field and a single cell: silent at depth 0 AND at depth 1."""
        empty = np.zeros((32, 32), np.uint8)
        one = np.zeros((32, 32), np.uint8)
        one[5, 5] = 1
        for grid in (empty, one):
            for depth in (0.0, 1.0, 4.0):
                e = make(depth=depth)
                e.init(grid, None, GAIN)
                _mono, pcm, peak, clips = run_blocks(e, 12)
                self.assertEqual(int(np.abs(pcm).max()), 0)
                self.assertEqual(peak, 0.0)
                self.assertEqual(clips, 0)
        one_voices = make().__class__(CTX, dict(fm_depth=1.0, **SPECTRUM))
        one_voices.init(one, None, GAIN)
        self.assertEqual(len(one_voices.voices), 1)        # a voice exists...
        self.assertTrue(all(m is None for m in one_voices._mods))   # ...with no mode

    def test_gain_never_reaches_the_index(self):
        """The same field at two master gains: the FM sum before the gain is identical."""
        grid = jam_grid()
        outs = []
        for g in (GAIN, GAIN * 0.25):
            e = make()
            e.init(grid, None, g)
            outs.append(run_blocks(e, 40, gain=g)[0])
        self.assertTrue(np.array_equal(outs[0], outs[1]))


class Law(unittest.TestCase):
    """REQ 6.2: the phase law itself."""

    def test_one_modulator_matches_the_bessel_expansion(self):
        """sin(theta_c + beta*sin(theta_m)) line by line: J_n(beta) at f0 + n*f_m, the
        reflected negative lines included."""
        f_m, beta = 190.5256, 1.7
        lines = ref.expansion_lines(F0, [(f_m, beta, 0.0)], order=14, floor=1e-13)
        coeffs = ref.bessel_coeffs(beta, 14)
        for n in range(-6, 7):
            freq = F0 + n * f_m
            want = coeffs[n + 14]
            if freq < 0.0:
                freq, want = -freq, -want
            got = lines.get(round(freq, 6), (0.0, 0.0))
            self.assertAlmostEqual(got[0], want, places=9)
            self.assertAlmostEqual(got[1], 0.0, places=9)
        # and the time series built from those lines is the phase formula itself
        direct = ref.raw([(1.0, 0.0, [(f_m, beta, 0.0)])], F0, 4096, 1)
        series = ref.expansion_signal(1.0, F0, 0.0, [(f_m, beta, 0.0)], 4096, order=14)
        self.assertLess(float(np.max(np.abs(direct - series))), 1e-9)

    def test_preflight_lines_of_the_researcher(self):
        """The strongest lines the Researcher published for the Jam at I=1, from an
        independent expansion here."""
        doc = preflight()
        for row in doc['rows']:
            if row['index'] != 1.0 or row['generation'] != 0 or row['object_index'] != 0:
                continue
            mods = [(F0 * r, 1.0 * a, 0.0) for r, a in zip(row['ratios'], row['amplitudes'])]
            lines = ref.expansion_lines(F0, mods, order=10, floor=1e-12)
            for line in row['strongest_lines'][:8]:
                got = lines.get(round(float(line['hz']), 6))
                self.assertIsNotNone(got, f"line {line['hz']} Hz missing")
                self.assertLess(abs(got[0] - float(line['sine_amplitude'])), 1e-6)
                self.assertLess(abs(got[1]), 1e-9)

    def test_several_non_integer_frequencies_and_unequal_weights(self):
        """The engine's own sum against the phase formula written out at 32x."""
        freqs = [137.31, 411.93, 733.077, 1021.4]
        amps = [1.0, 0.62, 0.31, 0.145]
        e = make(depth=2.5)
        e.init(jam_grid(), None, GAIN)
        e._mods = single(freqs, amps)
        mono, _pcm, _pk, _cl = run_blocks(e, 200)
        scale = float(math.sqrt(sum(a * a for a in amps)))
        want = ref.stationary([(scale, 0.0, [(f, 2.5 * a, 0.0) for f, a in zip(freqs, amps)])],
                              F0, len(mono), 32)
        self.assertLess(ref.rel_db(mono[SR:], want[SR:]), -100.0)

    def test_coincident_modulators_add_their_indices(self):
        """Two modulators on ONE frequency: with the same phase the indices simply add,
        with different phases they add as vectors -- never as powers."""
        f = 311.0
        e = make(depth=1.0)
        e.init(jam_grid(), None, GAIN)
        e._mods = single([f, f], [0.8, 0.5], scale=1.0)
        mono, _p, _k, _c = run_blocks(e, 160)
        same = ref.stationary([(1.0, 0.0, [(f, 1.3, 0.0)])], F0, len(mono), 32)
        self.assertLess(ref.rel_db(mono[SR:], same[SR:]), -100.0)
        # different phases: 0.8*sin(t) + 0.5*sin(t + phi) == B*sin(t + PHI), one modulator
        phi = 1.1
        e2 = make(depth=1.0)
        e2.init(jam_grid(), None, GAIN)
        e2._mods = single([f, f], [0.8, 0.5], scale=1.0)
        e2.render(GAIN, 0)                       # the fresh source zeroed its phases
        e2.src.th_mod[0, 1] = e2.src.th_mod[0, 0] + phi      # ... now they differ by phi
        th_c0, th_m0 = float(e2.src.th_c[0]), float(e2.src.th_mod[0, 0])
        mono2 = []
        for i in range(1, 200):
            e2.render(GAIN, i * BLOCK)
            mono2.append(e2.mono.copy())
        mono2 = np.concatenate(mono2)
        x = 0.8 + 0.5 * math.cos(phi)
        yv = 0.5 * math.sin(phi)
        B, PHI = math.hypot(x, yv), math.atan2(yv, x)
        want = ref.stationary([(1.0, th_c0, [(f, B, th_m0 + PHI)])], F0, len(mono2), 32)
        self.assertLess(ref.rel_db(mono2[SR:], want[SR:]), -95.0)
        self.assertLess(abs(B - 1.15), 0.12)     # the vector sum, not 1.3 and not 0.94
        # the two phase cases are genuinely different sounds
        self.assertGreater(ref.rel_db(mono2[SR:], mono[SR:len(mono2)]), -20.0)

    def test_stationary_peak_deviation(self):
        """One modulator: the instantaneous frequency swings f0 +- beta*f_m."""
        f0, f_m, beta = 4000.0, 50.0, 10.0
        e = make(depth=beta, f0=f0)
        e.init(jam_grid(), None, GAIN)
        e._mods = single([f_m], [1.0])
        mono, _p, _k, _c = run_blocks(e, 260)
        period = int(round(SR / f_m))                     # 882 samples, exactly
        self.assertEqual(period * f_m, SR)
        seg = mono[SR:SR + 40 * period]
        self.assertEqual(len(seg), 40 * period)           # whole periods, nothing truncated
        inst = ref.instantaneous_freq(seg, SR)
        inst = inst[4 * period:36 * period]                # whole periods, off the edges
        self.assertLess(abs(float(inst.max()) - (f0 + beta * f_m)), 3.0)
        self.assertLess(abs(float(inst.min()) - (f0 - beta * f_m)), 3.0)
        self.assertLess(abs(float(inst.mean()) - f0), 1.0)

    def test_one_carrier_jointly_modulated_not_a_sum_of_carriers(self):
        freqs, amps, depth = [201.0, 337.0], [1.0, 0.7], 1.5
        e = make(depth=depth)
        e.init(jam_grid(), None, GAIN)
        e._mods = single(freqs, amps)
        mono, _p, _k, _c = run_blocks(e, 160)
        scale = float(math.sqrt(sum(a * a for a in amps)))
        joint = ref.stationary([(scale, 0.0, [(f, depth * a, 0.0) for f, a in zip(freqs, amps)])],
                               F0, len(mono), 32)
        apart = ref.stationary([(scale / 2.0, 0.0, [(freqs[0], depth * amps[0], 0.0)]),
                                (scale / 2.0, 0.0, [(freqs[1], depth * amps[1], 0.0)])],
                               F0, len(mono), 32)
        self.assertLess(ref.rel_db(mono[SR:], joint[SR:]), -100.0)
        self.assertGreater(ref.rel_db(mono[SR:], apart[SR:]), -10.0)

    def test_index_is_proportional_to_the_weight_and_to_the_depth(self):
        """beta = I * a exactly: no division by the number of modes, the weight sum or an
        RMS, and no second multiplication by A_object."""
        freqs, amps = [190.5256, 417.3], [1.0, 0.4]
        for depth in (0.5, 1.0, 3.0):
            e = make(depth=depth)
            e.init(jam_grid(), None, GAIN)
            e._mods = single(freqs, amps)
            e.render(GAIN, 0)
            got = e.src.beta_cur[0, :2].tolist()
            self.assertAlmostEqual(got[0], depth * amps[0], places=12)
            self.assertAlmostEqual(got[1], depth * amps[1], places=12)
            # the scale of the source is A_object -- the index is not multiplied by it
            self.assertAlmostEqual(float(e.src.amp_tgt[0]),
                                   math.sqrt(sum(a * a for a in amps)), places=12)


class ZeroAndDC(unittest.TestCase):
    """REQ 6.3."""

    def test_depth_zero_is_the_carrier_sine(self):
        grid = step(step(jam_grid()))              # one figure, three modes
        e = make(depth=0.0)
        e.init(grid, None, GAIN)
        mono, _p, _k, _c = run_blocks(e, 200)
        scale = e._mods[0][2]
        want = ref.stationary([(scale, 0.0, [])], F0, len(mono), 32)
        self.assertLess(ref.rel_db(mono[SR:], want[SR:]), -120.0)
        self.assertGreater(scale, 1.0)             # its own scale, not 1
        # and it is NOT the Laplacian sum of sines of side A
        base = registry.create('laplacian', CTX, dict(SPECTRUM))
        base.init(grid, None, GAIN)
        b = np.concatenate([base.render(GAIN, i * BLOCK)[0][:, 0].astype(np.float64)
                            for i in range(200)])
        self.assertGreater(ref.rel_db(mono * GAIN / math.sqrt(2.0) * 32767.0, b), -6.0)

    def test_dc_blocker_removes_a_zero_combination_frequency(self):
        """A modulator at exactly f0 whose phase has run away from the carrier's puts a
        constant into the sum; the fixed blocker takes it out."""
        e = make(depth=2.0)
        e.init(jam_grid(), None, GAIN)
        e._mods = single([F0], [1.0])
        e.render(GAIN, 0)
        e.src.th_mod[0, 0] = math.pi / 2.0
        mono = []
        for i in range(1, 200):
            e.render(GAIN, i * BLOCK)
            mono.append(e.mono.copy())
        mono = np.concatenate(mono)
        # the same configuration WITHOUT the blocker really does carry a constant
        raw = ref.raw([(1.0, 0.0, [(F0, 2.0, math.pi / 2.0)])], F0, 4 * SR, 8)
        dc = ref.mean_dc(raw)
        self.assertGreater(abs(dc), 0.4)                   # the sum really has a constant
        self.assertLess(abs(ref.mean_dc(mono[SR:])), abs(dc) * 1e-4)

    def test_dc_blocker_closed_form_equals_the_recursion(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(BLOCK) * 0.3 + 0.9
        r = math.exp(-2.0 * math.pi * lfm.DC_HZ / SR)
        got, xl, yl = lfm.dc_block(x, r, 0.25, -0.4)
        y = np.empty(len(x))
        xp, yp = 0.25, -0.4
        for i, v in enumerate(x):
            yp = v - xp + r * yp
            xp = v
            y[i] = yp
        self.assertLess(float(np.max(np.abs(got - y))), 1e-14)
        self.assertAlmostEqual(xl, float(x[-1]), places=15)
        self.assertAlmostEqual(yl, float(y[-1]), places=15)

    def test_dc_blocker_converges_and_has_zero_dc_gain(self):
        r = math.exp(-2.0 * math.pi * lfm.DC_HZ / SR)
        x = np.ones(BLOCK)
        y, xl, yl = lfm.dc_block(x, r, 0.0, 0.0)
        self.assertAlmostEqual(float(y[0]), 1.0, places=12)
        for _ in range(60):                         # ~0.5 s of a held constant
            y, xl, yl = lfm.dc_block(x, r, xl, yl)
        self.assertLess(abs(float(y[-1])), 1e-3)    # the constant is gone
        self.assertLess(abs(r ** 1 - r), 1e-15)
        # DC gain of (1 - z^-1)/(1 - r z^-1) at z = 1 is exactly 0
        self.assertEqual((1.0 - 1.0) / (1.0 - r), 0.0)


class BandAndTime(unittest.TestCase):
    """REQ 6.4: the output band, the reference, and the time behaviour."""

    def test_output_filter_shape_and_delay(self):
        for oversample in (4, 8, 16, 32):
            h = lfm.fir_kernel(oversample, SR)
            self.assertEqual(len(h), lfm.fir_taps(oversample))
            self.assertEqual((len(h) - 1) // (2 * oversample), lfm.FIR_DELAY_OUT)
            self.assertAlmostEqual(float(h.sum()), 1.0, places=12)
            self.assertTrue(np.allclose(h, h[::-1], atol=1e-15))     # linear phase

            def H(f):
                x = f / (oversample * SR)
                return abs(complex(np.sum(h * np.exp(-2j * np.pi * x * np.arange(len(h))))))
            self.assertLess(abs(ref.db(H(0.0))), 1e-6)
            self.assertLess(abs(ref.db(H(lfm.BAND_PASS * SR))), 0.01)
            self.assertLess(ref.db(H(lfm.BAND_STOP * SR)), -100.0)
            self.assertLess(ref.db(H(0.7 * SR)), -100.0)

    def test_measured_delay_is_39_output_samples(self):
        """The block-by-block chain equals the same filter run in one pass, and what
        comes out is the input delayed by exactly FIR_DELAY_OUT output samples."""
        e = make(depth=1.0)
        e.init(step(step(jam_grid())), None, GAIN)
        mono, _p, _k, _c = run_blocks(e, 200)
        want = ref.stationary(ref.sources_of(e), F0, len(mono), 8)
        self.assertLess(ref.rel_db(mono[SR:], want[SR:]), -120.0)
        # the unfiltered signal, sampled at sr, correlates best at exactly the delay
        raws = ref.raw(ref.sources_of(e), F0, len(mono), 8)[::8]
        corr = [float(np.dot(mono[SR:SR + 8000], raws[SR - d:SR + 8000 - d]))
                for d in range(0, 80)]
        self.assertEqual(int(np.argmax(corr)), lfm.FIR_DELAY_OUT)

    def test_reference_converges_and_the_engine_matches_it(self):
        """Stationary: 16x vs 32x better than -80 dB, the engine within -60 dB of 32x."""
        grids = {0: jam_grid(), 1: step(jam_grid()), 2: step(step(jam_grid()))}
        cases = [(0, 110.0, 0.0), (2, 110.0, 1.0), (2, 110.0, 2.0), (1, 110.0, 4.0),
                 (2, 1760.0, 1.0), (2, 1760.0, 4.0), (1, 110.0, 1.0)]
        for gen, f0, depth in cases:
            e = make(depth=depth, f0=f0)
            e.init(grids[gen], None, GAIN)
            mono, _p, _k, _c = run_blocks(e, 170)
            srcs = ref.sources_of(e, depth)
            r16 = ref.stationary(srcs, f0, len(mono), 16)
            r32 = ref.stationary(srcs, f0, len(mono), 32)
            conv = ref.rel_db(r16[SR:], r32[SR:])
            err = ref.rel_db(mono[SR:], r32[SR:])
            self.assertLess(conv, -80.0, f"reference not converged at gen {gen} f0 {f0} I {depth}")
            self.assertLess(err, -60.0, f"engine off the reference at gen {gen} f0 {f0} I {depth}")

    def test_unequal_weights_follow_the_reference(self):
        sp = dict(SPECTRUM, n=5, alpha=1.2, shape=0.6)
        e = make(depth=2.0, spectrum=sp)
        e.init(step(jam_grid()), None, GAIN)
        amps = [float(a) for a in e._mods[0][1] if a > 0]
        self.assertGreater(max(amps) - min(amps), 0.1)     # genuinely unequal
        mono, _p, _k, _c = run_blocks(e, 170)
        srcs = ref.sources_of(e, 2.0)
        r16 = ref.stationary(srcs, F0, len(mono), 16)
        r32 = ref.stationary(srcs, F0, len(mono), 32)
        self.assertLess(ref.rel_db(r16[SR:], r32[SR:]), -80.0)
        self.assertLess(ref.rel_db(mono[SR:], r32[SR:]), -60.0)

    def test_the_whole_jam_with_transitions_matches_a_higher_rate(self):
        """Not only a still figure: the full evolution with its tails, index ramps and
        appearing / disappearing figures, rendered by the SAME engine at 8x / 16x / 32x."""
        seconds = 4.0
        y8, gens = run_jam(make(), seconds)
        y16, _ = run_jam(make(oversample=16), seconds)
        y32, _ = run_jam(make(oversample=32), seconds)
        self.assertGreaterEqual(gens, 15)
        self.assertLess(ref.rel_db(y16, y32), -80.0)
        self.assertLess(ref.rel_db(y8, y32), -60.0)

    def test_knobs_do_not_reset_phases_or_the_field(self):
        e = make()
        e.init(jam_grid(), None, GAIN)
        run_blocks(e, 20)
        th_c = e.src.th_c.copy()
        th_m = e.src.th_mod.copy()
        f_prev = e.src.f_prev.copy()
        e.set_params(dict(e.params, fm_depth=3.0))
        self.assertTrue(np.array_equal(th_c, e.src.th_c))
        self.assertTrue(np.array_equal(th_m, e.src.th_mod))
        self.assertTrue(np.array_equal(f_prev, e.src.f_prev))
        mono, _p, _k, _c = run_blocks(e, 6)
        seam, inside = junction_d2(mono)
        self.assertLess(seam, 4.0 * inside)

    def test_a_second_change_does_not_leave_a_ramp_hanging(self):
        """Two depth changes in consecutive blocks: the sum stays continuous and ends on
        the last target (no half-finished ramp, no step)."""
        e = make(depth=1.0)
        e.init(jam_grid(), None, GAIN)
        run_blocks(e, 10)
        e.set_params(dict(e.params, fm_depth=3.5))
        e.render(GAIN, 0)
        e.set_params(dict(e.params, fm_depth=0.4))
        mono, _p, _k, _c = run_blocks(e, 8)
        seam, inside = junction_d2(mono)
        self.assertLess(seam, 4.0 * inside)
        live = e.src.f_mod[0, :MAX_MODES_PER_OBJ] > 0
        want = 0.4 * np.asarray(e._mods[0][1])[:MAX_MODES_PER_OBJ][live[:len(e._mods[0][1])]]
        got = e.src.beta_cur[0, :MAX_MODES_PER_OBJ][live]
        self.assertTrue(np.allclose(got[:len(want)], want, atol=1e-12))

    def test_a_lost_figure_keeps_its_modulators_in_the_tail(self):
        """The released carrier is not turned into a bare sine, and a further field change
        does not cut the tail."""
        grid = jam_grid()
        e = make()
        e.init(grid, None, GAIN)
        run_blocks(e, 8)
        alive_before = int(np.count_nonzero(e.src.alive))
        betas = e.src.beta_cur[:MAX_VOICES].copy()
        freqs = e.src.f_mod[:MAX_VOICES].copy()
        amps = e.src.amp_cur[:MAX_VOICES].copy()
        e.update_field(step(step(grid)), None)     # 3 figures -> 1
        mono, _p, _k, _c = run_blocks(e, 1)
        tails = np.nonzero(e.src.rel_cnt[MAX_VOICES:] > 0)[0] + MAX_VOICES
        self.assertGreaterEqual(len(tails), alive_before - 1)
        kept = 0
        for t in tails:
            row = e.src.beta_cur[t][:MAX_MODES_PER_OBJ]
            self.assertGreater(float(np.abs(row).max()), 0.0, "a tail was emptied of its indices")
            for v in range(MAX_VOICES):
                if (np.allclose(e.src.f_mod[t][:MAX_MODES_PER_OBJ], freqs[v][:MAX_MODES_PER_OBJ])
                        and amps[v] > 0):
                    self.assertTrue(np.allclose(row, betas[v][:MAX_MODES_PER_OBJ]))
                    kept += 1
                    break
        self.assertGreater(kept, 0)
        # a further change while the tail rings does not zero its countdown
        cnt = e.src.rel_cnt[tails].copy()
        e.update_field(step(step(step(grid))), None)
        run_blocks(e, 1)
        self.assertTrue(np.all(e.src.rel_cnt[tails] >= np.maximum(cnt - 1, 0)))
        seam, inside = junction_d2(mono)
        self.assertLess(seam, 6.0 * inside)

    def test_pan_is_centred_and_the_channels_are_identical(self):
        e = make()
        e.init(jam_grid(), None, GAIN)
        _mono, pcm, _pk, _cl = run_blocks(e, 40)
        self.assertTrue(np.array_equal(pcm[:, 0], pcm[:, 1]))
        self.assertGreater(int(np.abs(pcm).max()), 0)

    def test_a_figure_that_comes_back_starts_clean(self):
        """The Jam loses and regains figures every generation: a returning one gets a fresh
        source (zero phases, no leftover index) and the seam carries no corner."""
        grid = jam_grid()
        e = make()
        e.init(grid, None, GAIN)
        run_blocks(e, 8)
        e.update_field(step(step(grid)), None)          # 3 figures -> 1
        run_blocks(e, 8)
        e.update_field(grid, None)                      # ... and back to 3
        mono, _p, _k, _c = run_blocks(e, 8)
        self.assertEqual(int(np.count_nonzero(e.src.alive)), 3)
        for v in range(3):
            live = e.src.f_mod[v, :MAX_MODES_PER_OBJ] > 0
            self.assertTrue(live.any())
            self.assertEqual(int(np.count_nonzero(e.src.m_rel_cnt[v] > 0)), 0)
        seam, inside = junction_d2(mono)
        self.assertLess(seam, 6.0 * inside)

    def test_release_and_block_are_the_baseline_scales(self):
        e = make()
        self.assertEqual(e._attack_chunks, 1)
        self.assertEqual(e._decay_chunks, 1)
        self.assertEqual(e._sustain, 1.0)
        self.assertEqual(e._release_chunks, 3)
        self.assertAlmostEqual(BLOCK / SR * 1000.0, 7.9819, places=3)
        self.assertAlmostEqual(e._release_chunks * BLOCK / SR * 1000.0, 23.9456, places=3)
        self.assertAlmostEqual(CHUNK_S, 0.008, places=12)

    def test_a_paused_field_keeps_sounding_and_stop_is_silent(self):
        scene = load_scene(os.path.join(ROOT, 'demos', 'lfm_baseline.json'))
        runner = DemoRunner(scene)
        runner.post('start', at=0)
        for _ in range(30):
            runner.next_block()
        runner.post('pause', at=runner.out_samples, on=True)
        blocks = [runner.next_block().B for _ in range(40)]
        self.assertGreater(int(np.abs(np.concatenate(blocks, axis=0)).max()), 0)
        runner.post('stop', at=runner.out_samples)
        for _ in range(4):
            runner.next_block()
        after = np.concatenate([runner.next_block().B for _ in range(10)], axis=0)
        self.assertEqual(int(np.abs(after).max()), 0)


class Records(unittest.TestCase):
    """REQ 6.5-6.7: the scenes, the levels, reproducibility and the budget."""

    def test_scenes_hold_the_variants_and_their_factors(self):
        for case in CASES:
            doc = scene_for(case)
            scene = scene_from_doc(doc)
            self.assertEqual(scene.rate_hz, RATE_HZ)
            self.assertEqual(scene.f0_hz, F0)
            for side, name in (('A', case['a']), ('B', case['b'])):
                eid, params = variant(name)
                self.assertEqual(scene.variants[side][0], eid)
                self.assertEqual(scene.variants[side][1], params)
                self.assertAlmostEqual(scene.side_gain[side], SIDE_GAIN[name], places=12)
            on_disk = load_scene(os.path.join(ROOT, 'demos', case['id'] + '.json'))
            self.assertEqual(on_disk.doc, doc)
        self.assertEqual([c['b'] for c in CASES], ['FM', 'FM'])
        self.assertEqual(variant('FM')[1]['fm_depth'], FM_DEPTH)

    def _track(self, case, side):
        runner = DemoRunner(scene_from_doc(scene_for(case)))
        runner.post('start', at=0)
        blocks, peak = [], 0.0
        for _ in range(N_BLOCKS_12S):
            b = runner.next_block()
            blocks.append(getattr(b, side))
            peak = max(peak, float(runner.sides[side].peak))
        pcm = np.concatenate(blocks, axis=0)
        return pcm, peak, int(runner.sides[side].clip_blocks), runner

    def test_the_old_tracks_are_byte_identical(self):
        """R and Wave bank / Saw must be exactly the records the user already heard."""
        with open(os.path.join(CARRIERS_ROOT, REFERENCE_RECORD, 'record.json'),
                  encoding='utf-8') as f:
            old = json.load(f)
        self.assertEqual(old['scene']['id'], 'lc_saw_baseline_bank')
        want = {'R': old['audio']['A']['sha256'], 'W_saw': old['audio']['B']['sha256']}
        for case in CASES:
            pcm, _peak, clips, _r = self._track(case, 'A')
            self.assertEqual(len(pcm), old['audio']['A']['samples'])
            got = hashlib.sha256(np.ascontiguousarray(pcm, np.int16).tobytes()).hexdigest()
            self.assertEqual(got, want[case['a']],
                             f"side A of {case['id']} is not the pinned {case['a']} track")
            self.assertEqual(clips, 0)

    def test_the_fm_track_is_one_variant_at_one_level(self):
        """The same FM in both records, byte-identical, within 0.5 dB of the other two."""
        tracks, levels = {}, {}
        for case in CASES:
            b, peak, clips, _r = self._track(case, 'B')
            tracks[case['id']] = b
            levels[case['id']] = ref.db(ref.rms(b.astype(np.float64) / 32768.0))
            self.assertEqual(clips, 0)
            self.assertLess(peak, 1.0)
            a, _pk, _cl, _rr = self._track(case, 'A')
            self.assertLessEqual(abs(levels[case['id']]
                                     - ref.db(ref.rms(a.astype(np.float64) / 32768.0))), 0.5)
        ids = [c['id'] for c in CASES]
        self.assertTrue(np.array_equal(tracks[ids[0]], tracks[ids[1]]))

    def test_continue_is_exact_including_a_transition_and_a_pause(self):
        scene = load_scene(os.path.join(ROOT, 'demos', 'lfm_baseline.json'))
        for cut, pause in ((BLOCKS_PER_GEN + 1, False), (BLOCKS_PER_GEN * 2 + 2, False),
                           (40, True)):
            runner = DemoRunner(scene)
            runner.post('start', at=0)
            if pause:
                runner.post('pause', at=20 * BLOCK, on=True)
            for _ in range(cut):
                runner.next_block()
            self.assertTrue(supports_snapshot(runner.sides['B'].engine))
            state = runner.export_state()
            straight = np.concatenate([runner.next_block().B for _ in range(24)], axis=0)
            other = DemoRunner.from_state(state)
            resumed = np.concatenate([other.next_block().B for _ in range(24)], axis=0)
            self.assertTrue(np.array_equal(straight, resumed),
                            f"continue diverged (cut {cut}, paused {pause})")

    def test_a_state_from_another_oversampling_is_refused(self):
        e = make()
        e.init(jam_grid(), None, GAIN)
        run_blocks(e, 3)
        st = e.export_state()
        other = make(oversample=16)
        with self.assertRaises(ValueError):
            other.restore_state(jam_grid(), None, st)

    def test_block_budget_of_the_two_new_scenes(self):
        import time
        budget_ms = BLOCK / SR * 1000.0
        for case in CASES:
            runner = DemoRunner(scene_from_doc(scene_for(case)))
            runner.post('start', at=0)
            for _ in range(20):
                runner.next_block()                     # warm-up (caches, first analysis)
            times = []
            for _ in range(400):
                t0 = time.perf_counter()
                runner.next_block()
                times.append((time.perf_counter() - t0) * 1000.0)
            p99 = float(np.percentile(times, 99))
            self.assertLess(p99, budget_ms, f"{case['id']}: p99 {p99:.2f} ms over {budget_ms:.2f}")


class BenchIntegration(unittest.TestCase):

    def test_registry_entry(self):
        spec = registry.get(lfm.ENGINE_ID)
        self.assertEqual(spec.label, 'Laplace FM')
        self.assertEqual(spec.params[0][0], 'fm_depth')
        self.assertEqual((spec.params[0][2], spec.params[0][3], spec.params[0][5]),
                         (0.0, 4.0, 1.0))
        self.assertEqual(tuple(p[0] for p in spec.params[1:]), lfm.SPECTRUM_KEYS)
        self.assertNotIn('fm_depth', spec.inactive(dict(fm_depth=0.0, shape=0.0)))
        self.assertIn('dyn', spec.inactive(dict(fm_depth=1.0, shape=0.0)))
        self.assertIn('Laplace FM', spec.overlay(dict(fm_depth=2.0), 32, 32)['text'])
        # the seven settings are shared with every other Laplace engine ("spectrum A to B")
        self.assertEqual(registry.spectrum_keys(lfm.ENGINE_ID, 'laplacian'), lfm.SPECTRUM_KEYS)
        self.assertEqual(registry.spectrum_keys(lfm.ENGINE_ID, 'laplace_carriers'),
                         lfm.SPECTRUM_KEYS)

    def test_headless_draw_panel_and_display(self):
        import time
        import pygame
        import demo_bench as db
        from casynth_lab.audio_out import LiveEngine
        scene = load_scene(os.path.join(ROOT, 'demos', 'lfm_baseline.json'))
        runner = DemoRunner(scene)
        eng = LiveEngine(runner, sink=lambda m, b: None)
        pygame.init()
        app = db.BenchApp(scene, eng)
        self.assertIn(lfm.ENGINE_ID, app.engine_btns)
        screen = pygame.Surface((app.width, app.height))
        font = pygame.font.SysFont(db.FONT_NAMES, 17)
        small = pygame.font.SysFont(db.FONT_NAMES, 14)
        eng.start()
        try:
            eng.post('start')
            t0 = time.time()
            while time.time() - t0 < 5 and eng.snapshot()['gen'] < 2:
                time.sleep(0.02)
            eng.post('select', side='B')
            t0 = time.time()
            while time.time() - t0 < 5 and eng.snapshot()['selected'] != 'B':
                time.sleep(0.02)
            app.draw(screen, font, small)
            d = eng.snapshot()['display']['B']
            self.assertTrue(d.get('fm'))
            self.assertEqual(int(d['oversample']), lfm.OVERSAMPLE)
            self.assertEqual(int(d['taps']), lfm.fir_taps(lfm.OVERSAMPLE))
            self.assertEqual(int(d['delay_samples']), lfm.FIR_DELAY_OUT)
            self.assertAlmostEqual(float(d['depth']), FM_DEPTH, places=12)
            self.assertGreaterEqual(int(d['sounding']), 1)
        finally:
            eng.stop()


if __name__ == '__main__':
    unittest.main(verbosity=2)
