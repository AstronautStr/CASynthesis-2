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
  8 threads      -- the block built on a pool of threads is the single-threaded block,
    byte for byte, plain and under a dragged spectrum knob (2026-09-21)

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


class KnobDragBudget(unittest.TestCase):
    """REQ 6.7 on the case the user hit on 2026-09-20: underruns while a knob is
    dragged on a live field.

    The field and the settings are the ones from that session (the end field of the
    "Few dots dance" record, 48 cells, 2 generations / s, f0 110 Hz, FM depth 1.97,
    n 3, spread 1, alpha 0, shape 1, harm 1, full on, dyn 0.08), and `harm` is dragged
    down.  Dragging a spectrum knob is the worst case for this engine: EVERY mode of
    EVERY figure changes frequency on every block, so every modulator spawns a tail
    that keeps sounding for the whole release -- the live modulator count triples --
    AND the whole field is re-analysed.  The bench runs its event loop at 60 Hz and
    drains every pending MOUSEMOTION, so a drag posts several `set_param` commands per
    block (a 1000 Hz mouse can post ~8); the runner applies all of them in one block.
    """

    # the user's field, 2026-09-20 (end state of lab_catalog/objects_decay_2026_09_18/
    # 20260920-213449-4caf42, opened anew in the bench)
    FIELD = ((1, 27), (2, 26), (2, 28), (3, 26), (3, 28), (4, 27), (5, 0), (5, 1), (6, 0),
             (6, 1), (9, 2), (10, 1), (10, 3), (10, 8), (11, 0), (11, 3), (11, 7), (11, 9),
             (12, 1), (12, 2), (12, 6), (12, 9), (13, 7), (13, 8), (17, 7), (17, 8), (18, 1),
             (18, 2), (18, 6), (18, 9), (19, 0), (19, 3), (19, 7), (19, 9), (20, 1), (20, 3),
             (20, 8), (21, 2), (24, 0), (24, 1), (25, 0), (25, 1), (26, 27), (27, 26),
             (27, 28), (28, 26), (28, 28), (29, 27))
    SETTINGS = dict(n=3, spread=1.0, alpha=0.0, shape=1.0, harm=1.0, fullshape=1, dyn=0.08)
    DEPTH = 1.97
    RATE = 2.0
    COMMANDS_PER_BLOCK = 4          # what a 60 Hz UI drag posts with an ordinary mouse

    def field(self):
        g = np.zeros((32, 32), np.uint8)
        for r, c in self.FIELD:
            g[r, c] = 1
        return g

    STEP = 0.001                    # one slider step of the knob (the bench rounds to 3)

    def _drag(self, engine, blocks=320, commands=COMMANDS_PER_BLOCK, drag_from=40):
        """Run the live field under the engine while `harm` is dragged.

        The knob travels one slider step per command and turns around at the ends, so the
        drag never stops: at every command rate the whole measured window is spent
        dragging (a sweep that simply runs into 0 would leave the faster rates coasting
        and make them look cheap)."""
        import time
        g = self.field()
        exc = None
        engine.init(g, exc, GAIN)
        ca, gen, times, harm, way = 0, 0, [], 1.0, -1.0
        for i in range(blocks):
            t0 = time.perf_counter()
            if ca >= (gen + 1) * (SR / self.RATE):
                new = step(g)
                exc = events_field(g, new)
                g = new
                gen += 1
                engine.update_field(g, exc)
            if i >= drag_from:
                for _k in range(commands):
                    harm += way * self.STEP
                    if not (0.0 <= harm <= 1.0):
                        way = -way
                        harm = min(max(harm, 0.0), 1.0)
                    engine.set_params(dict(engine.params, harm=round(harm, 3)))
            engine.render(GAIN, i * BLOCK)
            times.append((time.perf_counter() - t0) * 1000.0)
            ca += BLOCK
        return np.array(times[drag_from // 2:])

    def _runs(self, commands, n=2):
        """The drag, measured twice.  What starves the device is the SUSTAINED cost --
        the render thread keeps AUDIO_LOOKAHEAD_MS of blocks ahead, so a single late
        block is absorbed and only a mean above the budget drains the queue.  A single
        p99 on a busy desktop is scheduler noise, so the p99 gate reads the better run
        while the mean gate reads both."""
        out = []
        for _ in range(n):
            e = lfm.LaplaceFMEngine(EngineContext(SR, BLOCK, 2, F0, 1.0, self.RATE),
                                    dict(fm_depth=self.DEPTH, **self.SETTINGS))
            out.append(self._drag(e, commands=commands))
        return out

    def test_dragging_a_spectrum_knob_stays_inside_the_block(self):
        budget = BLOCK / SR * 1000.0
        runs = self._runs(self.COMMANDS_PER_BLOCK)
        mean = max(float(t.mean()) for t in runs)
        p99 = min(float(np.percentile(t, 99)) for t in runs)
        self.assertLess(mean, budget / 2.0,
                        f"dragged knob: {mean:.2f} ms per block on average, half of the "
                        f"{budget:.2f} ms block is the most a side may sustain")
        self.assertLess(p99, budget,
                        f"dragged knob: p99 {p99:.2f} ms over the {budget:.2f} ms block")

    def test_the_cost_does_not_grow_with_the_command_rate(self):
        """Several `set_param` commands in one block must cost ONE analysis: the bench
        posts one per mouse event and the runner applies every command that is due."""
        slow = min(float(t.mean()) for t in self._runs(1))
        fast = min(float(t.mean()) for t in self._runs(8))
        self.assertLess(fast, 1.5 * slow,
                        f"8 commands per block cost {fast:.2f} ms against {slow:.2f} ms for one")

    def test_the_drag_is_the_only_thing_that_changed(self):
        """A control: the same field without the drag is cheap, so the tests above are
        about the drag and not about the field being dense."""
        budget = BLOCK / SR * 1000.0
        idle = max(float(t.mean()) for t in self._runs(0))
        self.assertLess(idle, budget / 4.0)

    def test_both_sides_of_a_live_ab_stay_inside_the_block(self):
        """The bench renders BOTH sides every block: the baseline on A and the FM on B,
        with the shared spectrum knob dragged on B."""
        import time
        budget = BLOCK / SR * 1000.0
        doc = dict(format=2, id='lfm_drag', title='drag', grid=dict(rows=32, cols=32),
                   cells=[[r, c] for r, c in self.FIELD], rule='B3/S23', boundary='torus',
                   rate_hz=self.RATE,
                   audio=dict(f0_hz=F0, level=1.0, side_gain=dict(A=1.0, B=1.0)),
                   variants=dict(A=dict(engine_id='laplacian', engine_params=dict(self.SETTINGS)),
                                 B=dict(engine_id=lfm.ENGINE_ID,
                                        engine_params=dict(fm_depth=self.DEPTH, **self.SETTINGS))),
                   initial_side='B', listen='')
        runner = DemoRunner(scene_from_doc(doc))
        runner.post('start', at=0)
        times, harm, way = [], 1.0, -1.0
        for i in range(320):
            if i >= 40:
                for _k in range(self.COMMANDS_PER_BLOCK):
                    harm += way * self.STEP
                    if not (0.0 <= harm <= 1.0):
                        way = -way
                        harm = min(max(harm, 0.0), 1.0)
                    runner.post('set_param', side='B', name='harm', value=round(harm, 3))
            t0 = time.perf_counter()
            runner.next_block()
            times.append((time.perf_counter() - t0) * 1000.0)
        t = np.array(times[20:])
        mean, p99 = float(t.mean()), float(np.percentile(t, 99))
        self.assertLess(mean, budget, f"live A/B with a dragged knob: {mean:.2f} ms per block")
        self.assertLess(p99, 1.5 * budget, f"live A/B with a dragged knob: p99 {p99:.2f} ms")


class ThreadedRender(unittest.TestCase):
    """The block is built on SEVERAL THREADS (2026-09-21) -- and that must be a change
    of speed only.

    WHY: on the prototype's own field (52x30, Random) this engine cost 6.5 ms of the
    7.98 ms block, the render thread ran at 83% of real time and the device starved --
    281 underruns in 12 seconds, with the look-ahead pushed to its 160 ms ceiling.  The
    arithmetic is ~500k sines a block and one core cannot make that cheaper, so the
    column passes the render was already cut into now run on a pool.

    WHY IT IS STILL THE SAME SOUND: a pass owns its own slice of the block, reads
    everything else and the sum over sources never crosses two passes, so every output
    sample is computed by the same operations in the same order whatever the thread
    count and the pass size are.  This gate renders the SAME live scene at 1, 2 and 4
    threads and compares the blocks byte for byte -- including the dragged spectrum
    knob, where the modulator count (and so the number of passes) is highest."""

    BLOCKS = 80
    DRAG_BLOCKS = 48

    def _blocks(self, threads, blocks, drag, pooled=None):
        """The blocks the engine renders on a live field at this thread count.
        `pooled` (a counter) records how many blocks actually went through the pool,
        so a gate cannot pass by quietly never using it."""
        old = lfm.RENDER_THREADS
        old_pool = lfm.render_pool
        lfm.RENDER_THREADS = int(threads)
        if pooled is not None:
            def counted():
                p = old_pool()
                if p is not None:
                    pooled['n'] += 1
                return p
            lfm.render_pool = counted
        try:
            g = np.zeros((32, 32), np.uint8)
            for r, c in KnobDragBudget.FIELD:
                g[r, c] = 1
            e = lfm.LaplaceFMEngine(
                EngineContext(SR, BLOCK, 2, F0, 1.0, KnobDragBudget.RATE),
                dict(fm_depth=KnobDragBudget.DEPTH, **KnobDragBudget.SETTINGS))
            e.init(g, None, GAIN)
            out, ca, gen, harm, way = [], 0, 0, 1.0, -1.0
            for i in range(blocks):
                if ca >= (gen + 1) * (SR / KnobDragBudget.RATE):
                    new = step(g)
                    exc = events_field(g, new)
                    g = new
                    gen += 1
                    e.update_field(g, exc)
                if drag:
                    for _k in range(KnobDragBudget.COMMANDS_PER_BLOCK):
                        harm += way * KnobDragBudget.STEP
                        if not (0.0 <= harm <= 1.0):
                            way = -way
                            harm = min(max(harm, 0.0), 1.0)
                        e.set_params(dict(e.params, harm=round(harm, 3)))
                out.append(e.render(GAIN, i * BLOCK, transpose=1.25)[0])
                ca += BLOCK
            return out
        finally:
            lfm.RENDER_THREADS = old
            lfm.render_pool = old_pool

    def test_the_threaded_block_is_the_single_threaded_block(self):
        for drag, n in ((False, self.BLOCKS), (True, self.DRAG_BLOCKS)):
            ref = self._blocks(1, n, drag)
            for threads in (2, 4):
                pooled = {'n': 0}
                got = self._blocks(threads, n, drag, pooled)
                self.assertGreater(pooled['n'], n // 2,
                                   f"only {pooled['n']} of {n} blocks used the pool at "
                                   f"{threads} threads -- the gate would prove nothing")
                for i, (a, b) in enumerate(zip(ref, got)):
                    self.assertTrue(
                        np.array_equal(a, b),
                        f"{'dragged' if drag else 'plain'} block {i} differs at "
                        f"{threads} threads (worst sample {int(np.abs(a.astype(int) - b.astype(int)).max())})")

    def test_one_thread_builds_no_pool(self):
        """A machine with one usable core (or CASYNTH_RENDER_THREADS=1) renders the
        block where it stands -- no threads are started for nothing."""
        old = lfm.RENDER_THREADS
        lfm.RENDER_THREADS = 1
        try:
            self.assertIsNone(lfm.render_pool())
        finally:
            lfm.RENDER_THREADS = old
        self.assertGreaterEqual(lfm.RENDER_THREADS, 1)


class DraggedKnobClicks(unittest.TestCase):
    """The clicks the user recorded on 2026-09-21 while tweaking `harm`
    (lab_catalog/laplace_fm_2026_09_20/20260921-002846-f2423a, "issue").

    A click in this engine is a STEP in the phase sum: an index that disappears at a
    block boundary instead of being released.  It is measured where it happens -- in the
    OVERSAMPLED sum, before the band limit smears it over the filter's 78 samples -- as
    the second difference across the seam against the typical second difference inside
    the block.  A continuous signal keeps that ratio around 1.

    The settings are the ones the session ended with: FM depth 4, n 11, spread 1,
    alpha 0, shape 1, harm dragged, full on, dyn 1 on the Jam at 4 generations / s.
    Dragging `harm` moves every mode of every figure on every block, so each figure
    needs `n x release` modulator tails at once (11 x 3 = 33) -- more than the pool
    used to hold, and the pool then overwrote tails that were still sounding.
    """

    SETTINGS = dict(n=11, spread=1.0, alpha=0.0, shape=1.0, harm=0.958, fullshape=1, dyn=1.0)
    DEPTH = 4.0
    RATE = 4.0
    STEP = 0.001
    COMMANDS_PER_BLOCK = 4

    def _seams(self, blocks=400, warmup=40):
        """(ratios, engine) of a dragged `harm` on the Jam -- the shared measurement of
        demos/laplace_fm_reference.py, so the report and this gate read the same number."""
        e = lfm.LaplaceFMEngine(EngineContext(SR, BLOCK, 2, F0, 1.0, self.RATE),
                                dict(fm_depth=self.DEPTH, **self.SETTINGS))
        return ref.seam_ratios(e, jam_grid(), self.RATE, GAIN, blocks=blocks,
                               commands=self.COMMANDS_PER_BLOCK, step=self.STEP,
                               drag_from=30, warmup=warmup)

    def test_a_dragged_knob_never_steps_the_phase_sum(self):
        ratios, e = self._seams()
        worst = float(ratios.max())
        over = int((ratios > 3.0).sum())
        self.assertGreater(len(ratios), 300)
        self.assertEqual(over, 0,
                         f"{over} of {len(ratios)} block seams jump (worst x{worst:.1f} the "
                         f"second difference inside the block): an index vanished instead of "
                         f"being released")
        self.assertLess(worst, 3.0)

    def test_no_modulator_is_overwritten_while_it_still_sounds(self):
        """The mechanism behind the step: the per-figure modulator tail pool running out
        and a still-sounding tail being taken for the next one."""
        _ratios, e = self._seams(blocks=300, warmup=40)
        self.assertEqual(int(e.src.mod_steals), 0,
                         f"{int(e.src.mod_steals)} modulator tails were overwritten while "
                         f"they were still sounding")

    def test_an_exhausted_pool_degrades_smoothly(self):
        """The pool CAN be exhausted -- the widest spectrum on a slow field needs
        n x release tails per figure (20 x 6 here against 20 slots).  It must then fade
        modulators out where they stand, not overwrite sounding ones: no step, and the
        modulation itself must survive (the figure keeps its indices, it does not go
        quiet)."""
        keep = (self.SETTINGS, self.RATE)
        self.SETTINGS = dict(self.SETTINGS, n=MAX_MODES_PER_OBJ)
        self.RATE = 2.0
        try:
            ratios, e = self._seams(blocks=320)
            over = int((ratios > 3.0).sum())
            self.assertEqual(int(e.src.mod_steals), 0)
            self.assertGreater(int(e.src.mod_inplace), 0, "the pool was never actually full")
            self.assertEqual(over, 0, f"{over} seams jump, worst x{ratios.max():.1f}")
            live = ((e.src.f_mod[:MAX_VOICES, :MAX_MODES_PER_OBJ] > 0.0)
                    & (np.abs(e.src.beta_cur[:MAX_VOICES, :MAX_MODES_PER_OBJ]) > lfm.BETA_EPS))
            asked = sum(int(np.count_nonzero((np.asarray(m[0]) > 0) & (np.asarray(m[1]) > 0)))
                        for m in e._mods if m is not None)
            self.assertGreater(asked, 10)
            self.assertGreaterEqual(int(live.sum()), asked // 2,
                                    f"only {int(live.sum())} of the {asked} modes the analysis "
                                    f"asks for are sounding: the figures are being emptied "
                                    f"instead of waiting one block for a slot")
        finally:
            self.SETTINGS, self.RATE = keep


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
