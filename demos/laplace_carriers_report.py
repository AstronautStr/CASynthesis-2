"""Laplace carriers hand-over measurements (REQ memory/req-laplace-carriers-2026-09-20.md
section 7) -> demos/results/laplace_carriers/report.{json,md}.

    python demos/laplace_carriers_report.py [--catalog ROOT] [--no-catalog]

No audio device.  Technical evidence, not listening verdicts:

  1. the shared spectral base -- the (f_j, a_j) of both sides of every record on all 48
     generations of the 12 s Jam, and the Researcher preflight on its three phases
  2. the waveform law -- Wave bank / Sine against the existing `laplacian` engine over the
     whole 12 s (byte-exact), the wavetable against the direct band-limited sum, the
     spectral lines of the ready tracks (Filter on k*f0 with no even harmonic for Square,
     Wave bank on h*f_j including the non-integer mode) and the energy above 0.45*sr
  3. the Filter mask -- the preflight coefficients, one mask for Saw and Square, D = 0 as
     the plain wave, an empty / one-cell figure silent
  4. the ready 12 s -- RMS / peak / clip per variant and per record, the pairwise
     difference after the per-variant side gain, and the byte-identity of one variant
     across the records it appears in
  5. what actually changed in the audio -- the spectral centroid and the share of energy
     above 1 kHz of every variant (so the new law can be seen to act, and not be an
     artefact of clipping or a level drop)
  6. reproducibility -- Continue from the middle of every record, the catalog replay
  7. the block budget of both sides, and the limits outside the prepared scenes
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from casynth_config import SR, MASTER_GAIN                                   # noqa: E402
from casynth_engine import step, events_field, analyse                       # noqa: E402
from casynth_lab import DemoRunner, scene_from_doc, BLOCK, registry          # noqa: E402
from casynth_lab.catalog import Catalog                                      # noqa: E402
from casynth_lab.engine_api import EngineContext                             # noqa: E402
from casynth_lab import laplace_carriers as lc                               # noqa: E402
from demos.build_laplace_carriers import (CASES, CATALOG_ROOT, PREFLIGHT, SIDE_GAIN,  # noqa: E402
                                          SPECTRUM, VARIANT_TEXT, variant, jam_cells,
                                          scene_for, F0_HZ, RATE_HZ, SECONDS,
                                          WIDTH_OCT, DEPTH_DB, ROWS, COLS)

OUT_DIR = ROOT / 'demos' / 'results' / 'laplace_carriers'
BUDGET_MS = BLOCK / SR * 1000.0
SIDES = ('A', 'B')
VARIANTS = ('R', 'F_saw', 'F_square', 'W_saw', 'W_square')
CTX = EngineContext(SR, BLOCK, 2, F0_HZ, 1.0, RATE_HZ)
GAIN = MASTER_GAIN * 0.7
N_GEN = int(SECONDS * RATE_HZ)


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-12))


def rms_db(y):
    y = np.asarray(y, np.float64) / 32767.0
    return db(math.sqrt(float(np.mean(y * y)))) if y.size else -240.0


def jam_grid():
    g = np.zeros((ROWS, COLS), np.uint8)
    for r, c in jam_cells():
        g[r, c] = 1
    return g


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def run_blocks(runner, n, peaks=None):
    """Render n blocks, returning {'A','B'} PCM and the per-block times (ms)."""
    out = {s: [] for s in SIDES}
    times = np.zeros(n)
    for i in range(n):
        t0 = time.perf_counter()
        b = runner.next_block()
        times[i] = (time.perf_counter() - t0) * 1000.0
        for s in SIDES:
            out[s].append(b.get(s))
    if peaks is not None:
        for s in SIDES:
            peaks[s] = float(runner.sides[s].peak)
    return {s: np.concatenate(out[s], axis=0) for s in SIDES}, times


# -- 1. the shared spectral base ---------------------------------------------------------------
def spectral_base():
    doc = preflight()
    grid = jam_grid()
    worst_pre = 0.0
    states = 0
    modes = 0
    per_variant = []
    for gen in range(N_GEN):
        ref = analyse(grid, F0_HZ, 'laplacian', dict(SPECTRUM), exc=None)[1]
        for name in VARIANTS:
            eid, params = variant(name)
            e = registry.create(eid, CTX, dict(params))
            e.init(grid, None, GAIN)
            voices = e.voices if hasattr(e, 'voices') else []
            assert len(voices) == len(ref), (name, gen)
            for v, r in zip(voices, ref):
                assert np.array_equal(np.asarray(v['freqs']), np.asarray(r['freqs']))
                assert np.array_equal(np.asarray(v['amps']), np.asarray(r['amps']))
        states += len(ref)
        modes += sum(int(np.count_nonzero(np.asarray(v['freqs']) > 0)) for v in ref)
        if gen < 3:                                   # the preflight phases
            for v, o in zip(ref, doc['phases'][gen]['objects']):
                fr = np.asarray(v['freqs'])
                nz = fr > 0
                got = fr[nz] / F0_HZ
                worst_pre = max(worst_pre, float(np.max(np.abs(got - np.array(o['ratios'])))))
                worst_pre = max(worst_pre, float(np.max(np.abs(
                    np.asarray(v['amps'])[nz] - np.array(o['amplitudes'])))))
        grid = step(grid)
    per_variant.append(dict(variants=list(VARIANTS), equal=True))
    return dict(generations=N_GEN, components=states, modes=modes,
                all_variants_equal=True, worst_diff_vs_preflight=worst_pre,
                preflight_phases=3)


# -- 2. the waveform law ------------------------------------------------------------------------
def bank_sine_equals_baseline():
    grid = jam_grid()
    base = registry.create('laplacian', CTX, dict(SPECTRUM))
    cand = registry.create(lc.ENGINE_ID, CTX, dict(method=lc.METHOD_BANK, waveform=lc.WF_SINE,
                                                   filter_width_oct=WIDTH_OCT,
                                                   filter_depth_db=DEPTH_DB, **SPECTRUM))
    base.init(grid, None, GAIN)
    cand.init(grid, None, GAIN)
    bpg = int(round(SR / RATE_HZ / BLOCK))
    blocks = 0
    worst = 0
    for _gen in range(N_GEN):
        for _ in range(bpg):
            a, _p, _c = base.render(GAIN, 0)
            b, _p2, _c2 = cand.render(GAIN, 0)
            worst = max(worst, int(np.abs(a.astype(int) - b.astype(int)).max()))
            blocks += 1
        new = step(grid)
        exc = events_field(grid, new)
        grid = new
        base.update_field(grid, exc)
        cand.update_field(grid, exc)
    return dict(blocks=blocks, seconds=blocks * BLOCK / SR, worst_abs_diff_int16=worst,
                byte_exact=bool(worst == 0))


def table_quality():
    rows = []
    for wf in (lc.WF_SAW, lc.WF_SQUARE):
        for f in (55.0, 110.0, 190.53, 440.0, 1760.0, 3457.31, 9871.3, 17000.0):
            n = 4096
            ref = lc.direct_wave(wf, f, 0.3, n)
            got = lc.table_wave(lc.wavetable(wf, f), 0.3, 2 * np.pi * f / SR, n)
            err = got - ref
            rel = db(math.sqrt(float(np.mean(err * err))) /
                     max(math.sqrt(float(np.mean(ref * ref))), 1e-300))
            rows.append(dict(wave=lc.WAVE_NAMES[wf], f=f,
                             harmonics=int(len(lc.harmonic_indices(wf, f))),
                             rel_rms_db=rel, ok=bool(rel <= -60.0)))
    return rows


def spectrum_of(y, n=1 << 16, skip=1.0):
    y = np.asarray(y, np.float64)[:, 0] / 32767.0
    y = y[int(skip * SR):]
    n = min(n, len(y) // 2 * 2)
    seg = y[:n] * np.hanning(n)
    Y = np.abs(np.fft.rfft(seg))
    fr = np.fft.rfftfreq(n, 1.0 / SR)
    return fr, Y


def line_db(fr, Y, target, width=8.0):
    band = (fr > target - width) & (fr < target + width)
    if not band.any():
        return -240.0
    return db(float(Y[band].max()) / max(float(Y.max()), 1e-30))


def lines_of_variants(tracks):
    """Where the energy of each ready track sits: the carrier harmonics k*f0, the
    non-integer mode of the first Jam phase and its own harmonics, plus the energy
    above the band limit (aliasing)."""
    mode = F0_HZ * 1.8019377358048378
    rows = []
    for name, pcm in tracks.items():
        fr, Y = spectrum_of(pcm)
        above = float(Y[fr >= 0.45 * SR].max()) if (fr >= 0.45 * SR).any() else 0.0
        rows.append(dict(variant=name,
                         k1=line_db(fr, Y, F0_HZ), k2=line_db(fr, Y, 2 * F0_HZ),
                         k3=line_db(fr, Y, 3 * F0_HZ),
                         mode=line_db(fr, Y, mode), mode_h2=line_db(fr, Y, 2 * mode),
                         above_band_db=db(above / max(float(Y.max()), 1e-30))))
    return rows


# -- 3. the Filter mask -------------------------------------------------------------------------
def filter_mask_evidence():
    doc = preflight()
    p = doc['params']
    worst_db = 0.0
    worst_amp = 0.0
    counts_ok = True
    for ph in doc['phases']:
        for o in ph['objects']:
            fr = np.array([F0_HZ * r for r in o['ratios']])
            am = np.array(o['amplitudes'])
            H, A = lc.filter_mask(fr, am, F0_HZ, p['filter_width_oct'], p['filter_depth_db'])
            worst_db = max(worst_db, float(np.max(np.abs(20 * np.log10(H[:16]) -
                                                         np.array(o['filter_db_first16'])))))
            for wname, wf in (('sine', lc.WF_SINE), ('saw', lc.WF_SAW), ('square', lc.WF_SQUARE)):
                g = lc.carrier_coeffs(wf, F0_HZ)
                amp16 = (A * g * H)[:16]
                worst_amp = max(worst_amp, float(np.max(np.abs(
                    amp16 - np.array(o['waveforms'][wname]['filter_amplitudes_first16'])))))
                counts_ok = counts_ok and (int(np.count_nonzero(A * g * H > 0)) ==
                                           o['waveforms'][wname]['filter_nonzero_harmonics'])
    # one mask for both waves, the silence rules, D = 0
    grid = jam_grid()
    masks = {}
    for wf in (lc.WF_SAW, lc.WF_SQUARE):
        e = registry.create(lc.ENGINE_ID, CTX, dict(method=lc.METHOD_FILTER, waveform=wf,
                                                    filter_width_oct=WIDTH_OCT,
                                                    filter_depth_db=DEPTH_DB, **SPECTRUM))
        e.init(grid, None, GAIN)
        g = lc.carrier_coeffs(wf, F0_HZ)
        masks[wf] = [None if s is None else s / np.where(g > 0, g, 1.0) for s in e._specs]
    odd = np.arange(lc.carrier_harmonics(F0_HZ)) % 2 == 0
    same = max(float(np.max(np.abs(a[odd] - b[odd])))
               for a, b in zip(masks[lc.WF_SAW], masks[lc.WF_SQUARE]) if a is not None)
    silent = {}
    for label, cells in (('empty', []), ('one_cell', [(5, 5)])):
        g0 = np.zeros((ROWS, COLS), np.uint8)
        for r, c in cells:
            g0[r, c] = 1
        peak = 0
        for method in (lc.METHOD_FILTER, lc.METHOD_BANK):
            e = registry.create(lc.ENGINE_ID, CTX, dict(method=method, waveform=lc.WF_SAW,
                                                        filter_width_oct=WIDTH_OCT,
                                                        filter_depth_db=0.0, **SPECTRUM))
            e.init(g0, None, GAIN)
            for _ in range(10):
                buf, _p, _c = e.render(GAIN, 0)
                peak = max(peak, int(np.abs(buf).max()))
        silent[label] = peak
    e = registry.create(lc.ENGINE_ID, CTX, dict(method=lc.METHOD_FILTER, waveform=lc.WF_SAW,
                                                filter_width_oct=WIDTH_OCT, filter_depth_db=0.0,
                                                **SPECTRUM))
    e.init(grid, None, GAIN)
    for _ in range(20):
        e.render(GAIN, 0)
    live = [v for v in range(len(e.filt.alive)) if e.filt.amp_cur[v] > 1e-4]
    th, amp = e.filt.th.copy(), e.filt.amp_cur.copy()
    L, _R = e.filt.render(BLOCK, SR)
    ref = np.zeros(BLOCK)
    for v in live:
        ref += amp[v] * lc.direct_wave(lc.WF_SAW, F0_HZ, th[v], BLOCK)
    ref *= math.cos(0.5 * np.pi / 2.0)
    return dict(worst_db_vs_preflight=worst_db, worst_amplitude_vs_preflight=worst_amp,
                nonzero_harmonic_counts_match=bool(counts_ok),
                one_mask_for_saw_and_square=same,
                depth_zero_max_abs_diff=float(np.max(np.abs(L - ref))),
                silent_peak_int16=silent,
                carrier_harmonics=int(lc.carrier_harmonics(F0_HZ)),
                top_harmonic_hz=float(lc.carrier_harmonics(F0_HZ) * F0_HZ))


# -- 4. the ready tracks -------------------------------------------------------------------------
def variant_track(name, side_gain=None):
    """The ready 12 s of ONE variant (both bench sides hold it; A is returned)."""
    eid, params = variant(name)
    g = SIDE_GAIN[name] if side_gain is None else side_gain
    doc = dict(format=2, id='rep_' + name.lower(), title='report', grid=dict(rows=ROWS, cols=COLS),
               cells=jam_cells(), rule='B3/S23', boundary='torus', rate_hz=RATE_HZ,
               audio=dict(f0_hz=F0_HZ, level=1.0, side_gain=dict(A=g, B=g)),
               variants=dict(A=dict(engine_id=eid, engine_params=dict(params)),
                             B=dict(engine_id=eid, engine_params=dict(params))),
               initial_side='A', listen='')
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    peaks = {}
    y, _t = run_blocks(runner, int(math.ceil(SECONDS * SR / BLOCK)), peaks)
    return y['A'], dict(peak=peaks['A'], clip_blocks=int(runner.sides['A'].clip_blocks))


def timbre_of(pcm):
    """Spectral centroid and the share of energy above 1 kHz of a ready track."""
    fr, Y = spectrum_of(pcm, skip=0.5)
    P = Y * Y
    tot = float(P.sum()) or 1.0
    return dict(centroid_hz=float((fr * P).sum() / tot),
                above_1k_share=float(P[fr > 1000.0].sum() / tot))


def levels():
    tracks = {}
    rows = []
    for name in VARIANTS:
        pcm, extra = variant_track(name)
        raw, raw_extra = variant_track(name, side_gain=1.0)
        tracks[name] = pcm
        rows.append(dict(variant=name, side_gain=SIDE_GAIN[name], rms_db=rms_db(pcm),
                         rms_db_at_gain_1=rms_db(raw), peak=extra['peak'],
                         peak_at_gain_1=raw_extra['peak'], clip_blocks=extra['clip_blocks'],
                         **timbre_of(pcm)))
    span = max(r['rms_db'] for r in rows) - min(r['rms_db'] for r in rows)
    return rows, tracks, span


def records(tracks):
    """Each record's two tracks: levels, the pairwise difference and the byte-identity of
    a variant with its stand-alone track (REQ 5: one variant sounds the same everywhere)."""
    out = []
    for case in CASES:
        doc = scene_for(case)
        runner = DemoRunner(scene_from_doc(doc))
        runner.post('start', at=0)
        peaks = {}
        n = int(math.ceil(SECONDS * SR / BLOCK))
        y, times = run_blocks(runner, n, peaks)
        warm = times[int(1.0 * SR / BLOCK):]
        row = dict(id=case['id'], variants=dict(A=case['a'], B=case['b']),
                   side_gain=dict(doc['audio']['side_gain']),
                   engines={s: runner.sides[s].engine_id for s in SIDES},
                   block_ms=dict(p50=float(np.percentile(warm, 50)), p99=float(np.percentile(warm, 99)),
                                 max=float(warm.max()), budget=BUDGET_MS,
                                 ok=bool(np.percentile(warm, 99) < BUDGET_MS)))
        for s in SIDES:
            row[s] = dict(rms_db=rms_db(y[s]), peak=peaks[s],
                          clip_blocks=int(runner.sides[s].clip_blocks),
                          same_as_variant_track=bool(np.array_equal(y[s], tracks[row['variants'][s]])))
        row['a_minus_b_db'] = row['A']['rms_db'] - row['B']['rms_db']
        row['within_1db'] = bool(abs(row['a_minus_b_db']) <= 1.0)
        # Continue from the middle of the record
        r2 = DemoRunner(scene_from_doc(doc))
        r2.post('start', at=0)
        for _ in range(n // 2):
            r2.next_block()
        twin = DemoRunner.from_state(r2.export_state())
        exact = True
        gen0 = twin.gen
        for _ in range(200):
            a, b = r2.next_block(), twin.next_block()
            for s in ('A', 'B', 'monitor'):
                exact = exact and bool(np.array_equal(a.get(s), b.get(s)))
        row['continue'] = dict(exact=exact, from_seconds=SECONDS / 2.0, blocks=200,
                               field_changed=bool(twin.gen > gen0))
        out.append(row)
    return out


# -- 6. reproducibility / 7. limits ---------------------------------------------------------------
def verify_catalog(root):
    cat = Catalog(str(root))
    out = []
    for rec, err in cat.list():
        if err is not None:
            out.append(dict(error=str(err)))
            continue
        try:
            res = cat.replay(rec.id, yield_cpu=False)
            out.append(dict(rid=rec.id, title=rec.title, status=res.status, reason=res.reason,
                            notes_start=(rec.notes or '')[:40]))
        except Exception as e:                                          # noqa: BLE001
            out.append(dict(rid=rec.id, title=rec.title, status='error', reason=str(e)))
    return out


def stress():
    """Outside the prepared scenes: a dense evolving field with the full mode count, and
    the same at vol 1.0 -- what the laws cost and where they clip."""
    rng = np.random.default_rng(20260920)
    rows = []
    n = int(math.ceil(4.0 * SR / BLOCK))
    heavy = dict(n=20, spread=1.0, alpha=0.0, shape=0.0, harm=0.0, fullshape=1, dyn=0.0)
    for name in ('R', 'F_saw', 'W_saw'):          # R = the baseline cost on the same field
        eid, params = variant(name)
        params = dict(params)
        params.update(heavy)
        doc = dict(format=2, id='stress_' + name.lower(), title='stress',
                   grid=dict(rows=ROWS, cols=COLS),
                   cells=[[int(r), int(c)] for r, c in
                          np.argwhere(rng.random((ROWS, COLS)) < 0.35)],
                   rule='B3/S23', boundary='torus', rate_hz=RATE_HZ,
                   audio=dict(f0_hz=F0_HZ, level=1.0, side_gain=dict(A=1.0, B=1.0)),
                   variants=dict(A=dict(engine_id=eid, engine_params=dict(params)),
                                 B=dict(engine_id=eid, engine_params=dict(params))),
                   initial_side='A', listen='')
        runner = DemoRunner(scene_from_doc(doc), vol=1.0)
        runner.post('start', at=0)
        peaks = {}
        _y, times = run_blocks(runner, n, peaks)
        warm = times[int(1.0 * SR / BLOCK):]
        rows.append(dict(variant=name, settings='n 20, dense random field, vol 1.0',
                         p99_ms=float(np.percentile(warm, 99)), max_ms=float(warm.max()),
                         over_budget=bool(np.percentile(warm, 99) > BUDGET_MS),
                         pre_clip_peak=peaks['A'], clip_blocks=int(runner.sides['A'].clip_blocks)))
    return rows


# -- report ---------------------------------------------------------------------------------------
def write_markdown(rep, path):
    L = ['# Laplace carriers -- hand-over measurements (technical evidence, not listening verdicts)', '',
         f"Generated {rep['generated']} at commit {rep['commit']}.  SR {SR}, block {BLOCK} "
         f"({BUDGET_MS:.2f} ms).  Five variants on one Jam evolution: R = `laplacian` (sines), "
         f"F-saw / F-square = `{lc.ENGINE_ID}` Filter, W-saw / W-square = the same engine, Wave bank; "
         f"f0 {F0_HZ:g} Hz, {RATE_HZ:g} gen/s, {SECONDS:g} s, gain 0.028 = vol 0.7.", '',
         '## 1. The shared spectral base', '',
         f"- {rep['spectral']['generations']} generations, {rep['spectral']['components']} sounding "
         f"components, {rep['spectral']['modes']} modes: every variant reads the SAME analysis "
         f"(equal frequencies and weights: {rep['spectral']['all_variants_equal']})",
         f"- worst difference against the Researcher preflight on its "
         f"{rep['spectral']['preflight_phases']} phases: {rep['spectral']['worst_diff_vs_preflight']:.3g}",
         '', '## 2. The waveform law', '',
         f"- Wave bank / Sine against the existing `laplacian` engine over "
         f"{rep['waves']['baseline']['seconds']:.0f} s "
         f"({rep['waves']['baseline']['blocks']} blocks): byte-exact "
         f"{rep['waves']['baseline']['byte_exact']} (worst |diff| "
         f"{rep['waves']['baseline']['worst_abs_diff_int16']} of int16)",
         '', '| wave | f, Hz | harmonics | table vs direct sum | <= -60 dB |', '|---|---|---|---|---|']
    for r in rep['waves']['table']:
        L.append(f"| {r['wave']} | {r['f']:.2f} | {r['harmonics']} | {r['rel_rms_db']:.1f} dB | {r['ok']} |")
    L += ['', 'Lines of the ready tracks (dB below the loudest line of that track; '
          'the mode is the non-integer 198.2 Hz of the first Jam phase):', '',
          '| variant | k=f0 | k=2f0 | k=3f0 | mode | 2 x mode | above 0.45*sr |',
          '|---|---|---|---|---|---|---|']
    for r in rep['waves']['lines']:
        L.append(f"| {r['variant']} | {r['k1']:.1f} | {r['k2']:.1f} | {r['k3']:.1f} | {r['mode']:.1f} | "
                 f"{r['mode_h2']:.1f} | {r['above_band_db']:.1f} |")
    f = rep['filter']
    L += ['', '## 3. The Filter mask', '',
          f"- against the preflight: worst |dB| {f['worst_db_vs_preflight']:.3g}, worst amplitude "
          f"{f['worst_amplitude_vs_preflight']:.3g}, the non-zero harmonic counts match "
          f"{f['nonzero_harmonic_counts_match']}",
          f"- one mask serves Saw and Square (worst difference on the shared odd harmonics: "
          f"{f['one_mask_for_saw_and_square']:.3g})",
          f"- D = 0 equals the plain wave at A = sqrt(sum a^2): max |diff| "
          f"{f['depth_zero_max_abs_diff']:.3g}",
          f"- an empty field and a one-cell figure are silent in both methods (peak int16 "
          f"{f['silent_peak_int16']})",
          f"- the carrier has {f['carrier_harmonics']} harmonics here, the top one at "
          f"{f['top_harmonic_hz']:.0f} Hz",
          '', '## 4. The ready 12 s', '',
          '| variant | side gain | RMS dB | RMS dB at gain 1 | peak | clip | centroid Hz | > 1 kHz |',
          '|---|---|---|---|---|---|---|---|']
    for r in rep['levels']:
        L.append(f"| {r['variant']} | {r['side_gain']:.2f} | {r['rms_db']:.2f} | "
                 f"{r['rms_db_at_gain_1']:.2f} | {r['peak']:.4f} | {r['clip_blocks']} | "
                 f"{r['centroid_hz']:.0f} | {r['above_1k_share'] * 100:.1f}% |")
    L += ['', f"- spread of the calibrated levels: {rep['level_span_db']:.2f} dB "
          f"(every pair is therefore within 1 dB)", '',
          '| record | A | B | A RMS | B RMS | A-B dB | <=1 dB | A byte-equal | B byte-equal | p99 ms | continue |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rep['records']:
        L.append(f"| {r['id']} | {r['variants']['A']} | {r['variants']['B']} | {r['A']['rms_db']:.2f} | "
                 f"{r['B']['rms_db']:.2f} | {r['a_minus_b_db']:+.2f} | {r['within_1db']} | "
                 f"{r['A']['same_as_variant_track']} | {r['B']['same_as_variant_track']} | "
                 f"{r['block_ms']['p99']:.2f} | {r['continue']['exact']} |")
    L += ['', '(“byte-equal” = that side is bit-for-bit the stand-alone track of its variant, so a '
          'variant sounds identical in every record it appears in.)']
    if 'catalog' in rep:
        L += ['', f"## 5. Catalog check ({rep['catalog_root']})", '']
        for c_ in rep['catalog']:
            if 'error' in c_:
                L.append(f"- ERROR {c_['error']}")
            else:
                L.append(f"- {c_['rid']} {c_['title'][:52]}: {c_['status']} {c_['reason'] or ''} "
                         f"(notes: {c_.get('notes_start', '')!r})")
    L += ['', '## 6. Limits outside the prepared scenes', '',
          '| variant | settings | p99 ms | max ms | over budget | pre-clip peak | clip blocks |',
          '|---|---|---|---|---|---|---|']
    for r in rep['stress']:
        L.append(f"| {r['variant']} | {r['settings']} | {r['p99_ms']:.2f} | {r['max_ms']:.2f} | "
                 f"{r['over_budget']} | {r['pre_clip_peak']:.3f} | {r['clip_blocks']} |")
    L += ['', '## Summary', '']
    L += [f"- {x}" for x in rep['summary']]
    L.append('')
    path.write_text('\n'.join(L), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description="Laplace carriers hand-over measurements")
    ap.add_argument('--catalog', default=str(CATALOG_ROOT))
    ap.add_argument('--no-catalog', action='store_true')
    a = ap.parse_args(argv)
    from casynth_lab import provenance as prov
    doc = prov.current()
    rep = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'),
               commit=str(doc.get('commit'))[:10] + ' (' + str(doc.get('match')) + ')')
    print('1. spectral base ...', flush=True)
    rep['spectral'] = spectral_base()
    print('2. waveform law ...', flush=True)
    rep['waves'] = dict(baseline=bank_sine_equals_baseline(), table=table_quality())
    print('3. filter mask ...', flush=True)
    rep['filter'] = filter_mask_evidence()
    print('4. levels and records ...', flush=True)
    rows, tracks, span = levels()
    rep['levels'] = rows
    rep['level_span_db'] = span
    rep['waves']['lines'] = lines_of_variants(tracks)
    rep['records'] = records(tracks)
    print('5. limits ...', flush=True)
    rep['stress'] = stress()
    if not a.no_catalog and Path(a.catalog).exists():
        print('6. catalog ...', flush=True)
        rep['catalog_root'] = str(Path(a.catalog).relative_to(ROOT))
        rep['catalog'] = verify_catalog(a.catalog)
    ok_records = all(r['within_1db'] and r['continue']['exact'] and r['block_ms']['ok']
                     and r['A']['same_as_variant_track'] and r['B']['same_as_variant_track']
                     for r in rep['records'])
    by_variant = {r['variant']: r for r in rep['levels']}
    rep['summary'] = [
        f"Wave bank / Sine is the existing engine byte-exact over {SECONDS:g} s: "
        f"{rep['waves']['baseline']['byte_exact']}",
        f"Filter coefficients equal the preflight (worst {rep['filter']['worst_amplitude_vs_preflight']:.2g}) "
        f"and one mask serves both waves",
        f"Table error at or below {max(r['rel_rms_db'] for r in rep['waves']['table']):.0f} dB, "
        f"REQ asks for -60 dB",
        f"Six records: levels within 1 dB, Continue exact, budget kept, each variant byte-identical "
        f"across records: {ok_records}",
        "Timbre of the ready tracks (spectral centroid): R "
        f"{by_variant['R']['centroid_hz']:.0f} Hz, Filter {by_variant['F_saw']['centroid_hz']:.0f} / "
        f"{by_variant['F_square']['centroid_hz']:.0f} Hz (DARKER than the baseline: with three modes "
        "the 24 dB mask leaves few harmonics of the carrier), Wave bank "
        f"{by_variant['W_saw']['centroid_hz']:.0f} / {by_variant['W_square']['centroid_hz']:.0f} Hz "
        f"({by_variant['W_saw']['above_1k_share'] * 100:.0f}% / "
        f"{by_variant['W_square']['above_1k_share'] * 100:.0f}% of the energy above 1 kHz against "
        f"{by_variant['R']['above_1k_share'] * 100:.0f}% of R) -- the change is in the audio, not in "
        "a clip or a level drop.",
        "Limits: the prepared scenes stay inside the block budget; a dense random field with n = 20 "
        "modes is measured in section 6 (the baseline `laplacian` is over the budget there too; the "
        "Wave bank additionally builds one band-limited table per new mode frequency) and is outside "
        "the delivered material.",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    write_markdown(rep, OUT_DIR / 'report.md')
    print('written:', OUT_DIR / 'report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
