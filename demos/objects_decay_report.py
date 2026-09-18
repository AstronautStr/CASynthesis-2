"""Measurements for the Objects "Decay law" delivery (REQ
memory/req-objects-decay-2026-09-18.md, sections 4 and 5), before anything is played
to the listener:

  1. targets: q / T / frequencies of the engine at every transition sample of the
     preflight (D1, D2 with its pause boundaries, D3 with split / merge), Common and
     Modal, against the preflight
  2. D1: the control constant (the mean target AND applied gamma of the Common side
     over 5..20 s against ln(1000) / 0.947926); the phase sequence of the APPLIED
     gamma in the mature cycles; the decay rate measured in the rendered PCM between
     the steps, per phase, for A (Fixed) and B (Common age)
  3. D2: the modal envelopes of the REAL engine in both pauses (Common / Modal) against
     the independent two-mode probe of the preflight, and the upper / lower ratio
     measured in the rendered PCM (DFT at 110 / 216.17 Hz)
  4. D3: split / merge, tails and their loss states, the spread of T inside the banks
  5. isolation: per block the two sides of D1 / D2 / D3 share ids, cells, frequencies,
     weights, packets, a and Attack; only the losses differ
  6. levels: A / B RMS of the REQ windows at unit gains and with the delivered side
     gains, peaks, clip, finiteness
  7. Continue: from inside a D2 pause, during a gamma transition (D1, a law switch),
     after split / merge (D3)
  8. timing: the block time p99 / max of both sides on the three scenes after 1 s of
     warm-up, and the limits of large fields (a random soup, a change every block, one
     large figure of N cells)
  9. the catalog: every record replays exactly (when built)

    python demos/objects_decay_report.py [--root DIR] [--quick]

Writes demos/results/objects_decay/report.json + report.md.  Stdout ASCII.
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
from casynth_engine import step, events_field                                # noqa: E402
from casynth_lab import BLOCK, registry, DemoRunner, scene_from_doc         # noqa: E402
from casynth_lab.engine_api import EngineContext                             # noqa: E402
from casynth_lab import object_resonators as orz                            # noqa: E402
from demos.build_objects_decay import (CASES, CATALOG_ROOT, SIDE_GAIN, STABLE_DECAY_S, D2_PAUSES,   # noqa: E402
                                       scene_for, side_params, preflight, preflight_case, window_mask,
                                       calibrate)
from demos.build_objects_event_source import rms_db                         # noqa: E402
from demos.objects_event_source_report import verify_catalog               # noqa: E402

OUT_DIR = ROOT / 'demos' / 'results' / 'objects_decay'
BUDGET_MS = BLOCK / SR * 1000.0
SIDES = ('A', 'B')
GAIN = MASTER_GAIN * 0.7
EID = orz.ENGINE_ID
ROWS = COLS = 32
FIXED, COMMON, MODAL = orz.LAW_FIXED, orz.LAW_COMMON, orz.LAW_MODAL
LN1000 = orz.LN1000


def ctx(rate):
    return EngineContext(SR, BLOCK, 2, 110.0, 1.0, rate)


def grid(cells):
    g = np.zeros((ROWS, COLS), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def drive_case(case_id, law, on_block=None, decay=STABLE_DECAY_S):
    """One side of a preflight case on the REQ's sample clock (the preflight's own
    step rule and pause schedule); on_block(e, out_samples, gen) after every block."""
    case = preflight_case(case_id)
    e = registry.create(EID, ctx(case['rate_hz']), side_params(law, decay))
    e.init(grid(case['cells']), None, GAIN)
    g = grid(case['cells'])
    gen = ca = out = 0
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
        e.render_float(GAIN)
        if on_block is not None:
            on_block(e, out, gen)
        out += BLOCK
        if not paused:
            ca += BLOCK
    return e


# -- 1. targets against the preflight ----------------------------------------------------------------
def target_checks():
    out = {}
    for case in preflight()['cases']:
        tr = {t['t_samples']: t for t in case['transitions']}
        pb = {p['t_samples']: p for p in case.get('pause_boundaries', [])}
        for law in (COMMON, MODAL):
            acc = dict(banks=0, worst_T=0.0, worst_f=0.0, worst_q_T=0.0)

            def check(e, out, gen, tr=tr, pb=pb, law=law, acc=acc):
                ref = pb.get(out) or tr.get(out)
                if ref is None:
                    return
                banks = {tuple(map(tuple, b['cells'])): b for b in ref['banks']}
                for f in e.figures.values():
                    if f.slot < 0:
                        continue
                    b = banks[tuple(map(tuple, f.cells.tolist()))]
                    nd = int(e.ndrive[f.slot])
                    T = LN1000 / e.gam_t[f.slot, :nd]
                    want = np.asarray(b['local_T60_s']) if law == MODAL else np.full(nd, b['common_T60_s'])
                    acc['banks'] += 1
                    acc['worst_T'] = max(acc['worst_T'], float(np.max(np.abs(T - want))))
                    acc['worst_f'] = max(acc['worst_f'], float(np.max(np.abs(e.ffreq[f.slot, :nd] - np.asarray(b['freqs'])))))
            drive_case(case['id'], law, on_block=check)
            out[f"{case['id']}_{orz.LAW_NAMES[law]}"] = acc
            print(f"  {case['id']} {orz.LAW_NAMES[law]}: {acc['banks']} banks, worst |T| {acc['worst_T']:.1e} "
                  f"|f| {acc['worst_f']:.1e}", flush=True)
    return out


# -- helpers on the runner --------------------------------------------------------------------------
def run_scene(case, seconds=None, side_gain=None, on_block=None):
    """The delivered scene through the runner (start at 0, the scene's script);
    on_block(runner, out_before) after every block.  -> (pcm {A, B}, block ms, runner)."""
    runner = DemoRunner(scene_from_doc(scene_for(case, side_gain)))
    runner.post('start', at=0)
    pcm = {s: [] for s in SIDES}
    times = []
    for _ in range(int(math.ceil((seconds or case['seconds']) * SR / BLOCK))):
        before = runner.out_samples
        t0 = time.perf_counter()
        blk = runner.next_block()
        times.append((time.perf_counter() - t0) * 1000.0)
        for s in SIDES:
            pcm[s].append(blk.get(s))
        if on_block is not None:
            on_block(runner, before)
    return {s: np.concatenate(pcm[s]) for s in SIDES}, np.asarray(times), runner


def step_samples(rate, n_gen):
    """The output sample of every step g = 1..n_gen of a scene started at 0 without
    pauses (the runner steps at the first block with ca >= g SR / rate)."""
    return [int(math.ceil(g * SR / rate / BLOCK)) * BLOCK for g in range(1, n_gen + 1)]


def mono(y):
    return (np.asarray(y, np.float64)[:, 0] + np.asarray(y, np.float64)[:, 1]) / 32767.0


T60_FLOOR = 30.0 / 32767.0          # 10 ms windows below this RMS (30 LSB) are not used for a slope


def decay_t60(x, a, b, win=441, min_windows=8):
    """T60 (s) from the slope of 10 log10(mean square) over the 10 ms windows in
    [a, b) -- only the leading windows whose RMS stays above T60_FLOOR (the int16
    quantisation bends the slope below it); nan with fewer than `min_windows`."""
    n = (b - a) // win
    if n < min_windows:
        return float('nan')
    seg = x[a:a + n * win].reshape(n, win)
    ms = np.maximum((seg * seg).mean(axis=1), 1e-30)
    low = np.nonzero(ms < T60_FLOOR ** 2)[0]
    n = int(low[0]) if low.size else n
    if n < min_windows:
        return float('nan')
    ms = ms[:n]
    db = 10.0 * np.log10(ms)
    t = (np.arange(n) + 0.5) * win / SR
    slope = float(np.polyfit(t, db, 1)[0])
    return -60.0 / slope if slope < 0 else float('inf')


# -- 2. D1 ----------------------------------------------------------------------------------------
def d1_checks():
    case = CASES[0]
    gt_mean, ga_mean, per_block = [], [], []

    def probe(runner, before):
        e = runner.sides['B'].engine
        f = e.figures.get(1)
        if f is None or f.slot < 0:
            return
        nd = int(e.ndrive[f.slot])
        g_t = float(e.gam_t[f.slot, :nd].mean())
        g_a = float(e.gam_a[f.slot, :nd].mean())
        per_block.append((before, runner.gen, g_t, g_a))
        if 5.0 * SR <= before < 20.0 * SR:
            gt_mean.append(g_t)
            ga_mean.append(g_a)
    y, times, runner = run_scene(case, on_block=probe)
    control = LN1000 / case['decay']['A']
    res = dict(control_T=case['decay']['A'], control_gamma=control,
               mean_target_gamma=float(np.mean(gt_mean)), mean_applied_gamma=float(np.mean(ga_mean)))
    res['target_rel_error'] = abs(res['mean_target_gamma'] - control) / control
    res['applied_rel_error'] = abs(res['mean_applied_gamma'] - control) / control
    res['within_1pct'] = bool(res['target_rel_error'] <= 0.01 and res['applied_rel_error'] <= 0.01)
    # the phases of the mature cycles: per step interval the mean applied T and the target at
    # the step, and the T60 measured in the PCM of both sides (after 60 ms, 10 ms windows)
    steps = step_samples(case['rate_hz'], 40)
    xa, xb = mono(y['A']), mono(y['B'])
    rows = []
    for g in range(10, 39):                                   # 5..19.5 s: cycles 3..8
        a, b = steps[g - 1], steps[g]
        inside = [r for r in per_block if a <= r[0] < b]
        rows.append(dict(gen=g, phase=g % 5, t_s=a / SR,
                         target_T_at_step=LN1000 / inside[0][2],
                         applied_T_mean=LN1000 / float(np.mean([r[3] for r in inside])),
                         applied_T_range=[LN1000 / max(r[3] for r in inside), LN1000 / min(r[3] for r in inside)],
                         pcm_T60_A=decay_t60(xa, a + int(0.06 * SR), b - int(0.01 * SR)),
                         pcm_T60_B=decay_t60(xb, a + int(0.06 * SR), b - int(0.01 * SR))))
    def med(vals):
        v = [x for x in vals if math.isfinite(x)]
        return float(np.median(v)) if v else float('nan')
    phases = {}
    for p in range(5):
        sel = [r for r in rows if r['phase'] == p]
        fb = [r['pcm_T60_B'] for r in sel if math.isfinite(r['pcm_T60_B'])]
        phases[p] = dict(n=len(sel), target_T=float(np.mean([r['target_T_at_step'] for r in sel])),
                         applied_T=float(np.mean([r['applied_T_mean'] for r in sel])),
                         pcm_T60_A=med([r['pcm_T60_A'] for r in sel]),
                         pcm_T60_B=med([r['pcm_T60_B'] for r in sel]),
                         pcm_measured=len(fb),
                         pcm_T60_B_spread=([float(min(fb)), float(max(fb))] if fb else [float('nan'), float('nan')]))
    res['phases'] = phases
    res['rows'] = rows
    ok = [v for v in phases.values() if math.isfinite(v['pcm_T60_B']) and math.isfinite(v['pcm_T60_A'])]
    ap = [v['applied_T'] for v in phases.values()]
    pb = [v['pcm_T60_B'] for v in ok]
    pa = [v['pcm_T60_A'] for v in ok]
    res['pcm_phases_measured'] = len(ok)
    res['applied_T_phase_ratio_max_min'] = max(ap) / min(ap)
    res['pcm_T60_B_phase_ratio_max_min'] = max(pb) / min(pb) if pb else float('nan')
    res['pcm_T60_A_phase_ratio_max_min'] = max(pa) / min(pa) if pa else float('nan')
    res['pcm_B_vs_applied_max_rel'] = max(abs(v['pcm_T60_B'] - v['applied_T']) / v['applied_T'] for v in ok) if ok else float('nan')
    res['pcm_A_vs_control_max_rel'] = max(abs(v['pcm_T60_A'] - case['decay']['A']) / case['decay']['A'] for v in ok) if ok else float('nan')
    # the mechanism in the rendered audio: the applied T differs by phase, the PCM decay of B
    # follows it where measurable, A stays at its constant
    res['mechanism_in_audio'] = bool(res['applied_T_phase_ratio_max_min'] > 1.3 and len(ok) >= 3
                                     and res['pcm_T60_B_phase_ratio_max_min'] > 1.2
                                     and res['pcm_B_vs_applied_max_rel'] < 0.1 and res['pcm_A_vs_control_max_rel'] < 0.05)
    return res


# -- 3. D2 ----------------------------------------------------------------------------------------
PCM_WIN = 2048                     # 46 ms Hann: 110 / 216 Hz stay apart (bin 21.5 Hz)
PCM_FLOOR = 10.0 / 32767.0 * PCM_WIN / 4.0     # the DFT amplitude of a 10-LSB sine (mono L + R)


def dft_amp(x, centre, f, n=PCM_WIN):
    a = max(0, centre - n // 2)
    seg = x[a:a + n]
    w = np.hanning(len(seg))
    k = np.arange(len(seg))
    return float(abs(np.sum(seg * w * np.exp(-2j * math.pi * f * k / SR))))


def d2_checks():
    case = CASES[1]
    probe_pf = preflight()['d2_independent_modal_probe']
    freqs = preflight_case('D2')['transitions'][1]['banks'][0]['freqs']
    mags = {}

    def probe(runner, before):
        for s in SIDES:
            e = runner.sides[s].engine
            f = e.figures.get(1)
            if f is None or f.slot < 0:
                continue
            m = np.hypot(e.zre[f.slot, :2], e.zim[f.slot, :2])
            mags.setdefault(s, {})[before + BLOCK] = (float(m[0]), float(m[1]))   # the state at the block END
    y, times, runner = run_scene(case, on_block=probe)
    offs = [0, 13, 31, 63]                                     # blocks after the first measurement (preflight rows)
    pauses = []
    for p_on, p_off in ((D2_PAUSES[0][0], D2_PAUSES[1][0]), (D2_PAUSES[2][0], D2_PAUSES[3][0])):
        first = p_on + BLOCK
        rows = []
        for k in offs:
            t = first + k * BLOCK
            row = dict(elapsed_s=k * BLOCK / SR)
            for s, law in (('A', 'Common'), ('B', 'Modal')):
                lo0, hi0 = mags[s][first]
                lo, hi = mags[s][t]
                row[law] = dict(relative_mode_db=[20 * math.log10(lo / lo0), 20 * math.log10(hi / hi0)],
                                upper_over_lower_db=20 * math.log10(hi / lo))
            rows.append(row)
        # the same ratio in the rendered int16 PCM (DFT at the two frequencies, 46 ms Hann
        # windows centred after the first measurement; a component below 10 LSB is not
        # measured -- the Common tail reaches the quantisation floor within ~0.3 s)
        pcm_rows = []
        for t_after in (0.03, 0.06, 0.10, 0.15, 0.20, 0.30):
            c = first + int(t_after * SR)
            r = dict(t_after_s=t_after)
            for s, law in (('A', 'Common'), ('B', 'Modal')):
                x = mono(y[s])
                lo, hi = dft_amp(x, c, freqs[0]), dft_amp(x, c, freqs[1])
                r[law] = 20 * math.log10(hi / lo) if min(lo, hi) >= PCM_FLOOR else None
            pcm_rows.append(r)
        pauses.append(dict(pause_on=p_on, pause_off=p_off, first_measurement=first, engine=rows, pcm=pcm_rows))
    # comparison with the independent probe of the preflight (first pause)
    cmp = []
    for pr, er in zip(probe_pf['rows'], pauses[0]['engine']):
        cmp.append(dict(elapsed_s=er['elapsed_s'], probe_elapsed_s=pr['elapsed_from_first_pause_block_end_s'],
                        common_engine=er['Common']['relative_mode_db'], common_probe=pr['relative_mode_db_common'],
                        modal_engine=er['Modal']['relative_mode_db'], modal_probe=pr['relative_mode_db_modal'],
                        ratio_common_engine=er['Common']['upper_over_lower_db'], ratio_common_probe=pr['upper_over_lower_db_common'],
                        ratio_modal_engine=er['Modal']['upper_over_lower_db'], ratio_modal_probe=pr['upper_over_lower_db_modal']))
    worst = max(max(abs(a - b) for a, b in zip(c['common_engine'] + c['modal_engine'], c['common_probe'] + c['modal_probe']))
                for c in cmp)
    growth = []
    for p in pauses:
        e0, e3 = p['engine'][0], p['engine'][-1]
        growth.append(dict(modal_db=e3['Modal']['upper_over_lower_db'] - e0['Modal']['upper_over_lower_db'],
                           common_db=e3['Common']['upper_over_lower_db'] - e0['Common']['upper_over_lower_db']))
    return dict(frequencies_hz=freqs, pauses=pauses, probe_comparison=cmp, worst_probe_diff_db=worst, growth=growth,
                journal_pauses=[(j[0], j[3]['on'], bool(j[3].get('script'))) for j in runner.journal if j[2] == 'pause'],
                pcm_growth=[pcm_growth(p['pcm']) for p in pauses],
                mechanism_in_audio=bool(all(g['modal_db'] > g['common_db'] + 20.0 for g in growth)
                                        and all(pg['modal_db'] > pg['common_db'] + 10.0
                                                for pg in (pcm_growth(p['pcm']) for p in pauses))))


def pcm_growth(rows):
    """Growth of upper - lower from the first to the last row where BOTH sides are
    above the floor."""
    ok = [r for r in rows if r['Common'] is not None and r['Modal'] is not None]
    if len(ok) < 2:
        return dict(from_s=None, to_s=None, modal_db=float('nan'), common_db=float('nan'))
    return dict(from_s=ok[0]['t_after_s'], to_s=ok[-1]['t_after_s'], modal_db=ok[-1]['Modal'] - ok[0]['Modal'],
                common_db=ok[-1]['Common'] - ok[0]['Common'])


# -- 4. D3 ----------------------------------------------------------------------------------------
def d3_checks():
    case = CASES[2]
    stats = dict(max_figures=0, max_tails=0, max_tails_adaptive=0, spread=[], changes=0)

    def probe(runner, before):
        e = runner.sides['B'].engine
        d = e.display()
        stats['max_figures'] = max(stats['max_figures'], d['n_figures'])
        stats['max_tails'] = max(stats['max_tails'], d['n_tails'])
        stats['max_tails_adaptive'] = max(stats['max_tails_adaptive'],
                                          int(np.count_nonzero((e.role == orz.ROLE_TAIL) & (e.slaw == 1))))
        for f in d['figures']:
            if f['slot'] >= 0 and f['adaptive'] and f['modes'] > 1:
                stats['spread'].append(f['t_hi'] / f['t_lo'])
        stats['changes'] = d['changes']
    y, times, runner = run_scene(case, on_block=probe)
    ea, eb = runner.sides['A'].engine, runner.sides['B'].engine
    sp = np.asarray(stats['spread'])
    return dict(figures_max=stats['max_figures'], tails_max=stats['max_tails'], adaptive_tails_max=stats['max_tails_adaptive'],
                changes=stats['changes'], gen=runner.gen,
                t_spread_in_bank=dict(median=float(np.median(sp)), p90=float(np.percentile(sp, 90)), max=float(sp.max())),
                ids_A=sorted(ea.figures), ids_B=sorted(eb.figures), next_id=[ea.next_id, eb.next_id],
                drops=[ea.display()['drops'], eb.display()['drops']],
                inplace=[ea.display()['inplace_fades'], eb.display()['inplace_fades']])


# -- 5. isolation ----------------------------------------------------------------------------------
def isolation():
    out = {}
    for case in CASES:
        acc = dict(blocks=0, equal=True, loss_blocks=0, first=None)

        def probe(runner, before, acc=acc):
            a, b = runner.sides['A'].engine, runner.sides['B'].engine
            acc['blocks'] += 1
            ok = sorted(a.figures) == sorted(b.figures) and np.array_equal(a.qq, b.qq)
            diff = False
            for fid, fa in a.figures.items():
                fb = b.figures.get(fid)
                if fb is None or fa.slot != fb.slot or not np.array_equal(fa.cells, fb.cells):
                    ok = False
                    continue
                s = fa.slot
                if s < 0:
                    continue
                ok &= bool(np.array_equal(a.ffreq[s], b.ffreq[s]) and np.array_equal(a.wtgt[s], b.wtgt[s])
                           and float(a.last_e[s]) == float(b.last_e[s]) and float(a.last_a[s]) == float(b.last_a[s])
                           and np.array_equal(a.last_b[s], b.last_b[s]) and int(a.ndrive[s]) == int(b.ndrive[s]))
                nd = int(a.ndrive[s])
                ra = np.exp(-a.gam_a[s, :nd] / SR) if a.slaw[s] else np.full(nd, a.rr[orz.R_CUR])
                rb = np.exp(-b.gam_a[s, :nd] / SR) if b.slaw[s] else np.full(nd, b.rr[orz.R_CUR])
                diff |= bool(np.abs(ra - rb).max() > 0.0)
            if not ok and acc['first'] is None:
                acc['first'] = before
            acc['equal'] &= bool(ok)
            acc['loss_blocks'] += int(diff)
        run_scene(case, on_block=probe)
        out[case['id']] = acc
        print(f"  {case['id']}: {acc['blocks']} blocks, equal {acc['equal']}, losses differ in {acc['loss_blocks']}", flush=True)
    return out


# -- 6. levels ---------------------------------------------------------------------------------------
def levels():
    unit = calibrate()
    delivered = {}
    for case in CASES:
        peaks = {s: 0.0 for s in SIDES}

        def probe(runner, before, peaks=peaks):
            for s in SIDES:
                peaks[s] = max(peaks[s], runner.sides[s].peak)
        y, times, runner = run_scene(case, on_block=probe)
        m = window_mask(case, y['A'].shape[0])
        d = dict(side_gain=SIDE_GAIN[case['id']], window=list(case['window'][1:]))
        for s in SIDES:
            d[s] = dict(rms_db=rms_db(y[s]), window_rms_db=rms_db(y[s][m]), peak=peaks[s],
                        clip_blocks=int(runner.sides[s].clip_blocks), finite=bool(np.all(np.isfinite(y[s]))))
        d['window_a_minus_b_db'] = d['A']['window_rms_db'] - d['B']['window_rms_db']
        d['within_1db'] = bool(abs(d['window_a_minus_b_db']) <= 1.0)
        warm = times[int(1.0 * SR / BLOCK):]
        d['block_ms'] = dict(p50=float(np.percentile(warm, 50)), p99=float(np.percentile(warm, 99)), max=float(warm.max()),
                             budget=BUDGET_MS, ok=bool(np.percentile(warm, 99) < BUDGET_MS))
        delivered[case['id']] = d
    return dict(unit=unit, delivered=delivered)


# -- 7. Continue ------------------------------------------------------------------------------------
def continue_checks():
    out = {}

    def compare(runner, n):
        twin = DemoRunner.from_state(runner.export_state())
        ok = True
        for _ in range(n):
            a, b = runner.next_block(), twin.next_block()
            for o in ('A', 'B', 'monitor'):
                ok &= bool(np.array_equal(a.get(o), b.get(o)))
        return ok
    r = DemoRunner(scene_from_doc(scene_for(CASES[1])))
    r.post('start', at=0)
    while r.out_samples < D2_PAUSES[0][0] + 100 * BLOCK:
        r.next_block()
    out['d2_inside_first_pause'] = dict(paused=bool(r.paused), pending=len(r._pending),
                                        exact=compare(r, int(4.0 * SR / BLOCK)), resumed=not r.paused)
    r = DemoRunner(scene_from_doc(scene_for(CASES[0])))
    r.post('start', at=0)
    for _ in range(int(6.0 * SR / BLOCK)):
        r.next_block()
    r.post('set_param', side='A', name='decay_law', value=MODAL)
    for _ in range(3):
        r.next_block()
    e = r.sides['A'].engine
    s = e.figures[1].slot
    gap = float(np.abs(e.gam_a[s, :3] - e.gam_t[s, :3]).max())
    out['d1_mid_gamma_transition'] = dict(gamma_gap=gap, exact=compare(r, 300))
    r = DemoRunner(scene_from_doc(scene_for(CASES[2])))
    r.post('start', at=0)
    for _ in range(int(9.0 * SR / BLOCK)):
        r.next_block()
    eb = r.sides['B'].engine
    out['d3_after_split_merge'] = dict(adaptive_tails=int(np.count_nonzero((eb.role == orz.ROLE_TAIL) & (eb.slaw == 1))),
                                       next_id=eb.next_id, exact=compare(r, int(3.0 * SR / BLOCK)))
    return out


# -- 8. timing and the limits of large fields -------------------------------------------------------------
def stress(label, g0, rate, seconds, law, change_every_block=False, seed=0):
    """Both sides at `law` on a field (rate gen/s, or a random change of 20 % of the
    cells every block): block ms p99 / max after 0.5 s, peaks, clip, finiteness."""
    doc = scene_for(CASES[2])
    doc['id'] = 'od_stress'
    doc['cells'] = [[int(r), int(c)] for r, c in np.argwhere(g0)]
    doc['rate_hz'] = float(rate)
    doc['audio']['side_gain'] = dict(A=1.0, B=1.0)
    for s in SIDES:
        doc['variants'][s]['engine_params'] = side_params(law, STABLE_DECAY_S)
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    rng = np.random.default_rng(seed)
    times, peak, fin = [], 0.0, True
    n = int(seconds * SR / BLOCK)
    for k in range(n):
        if change_every_block and k > 0:
            flips = rng.random((ROWS, COLS)) < 0.2
            cells = [[int(r), int(c), int(1 - runner.grid[r, c])] for r, c in np.argwhere(flips)]
            runner.post('set_cells', cells=cells)
        t0 = time.perf_counter()
        blk = runner.next_block()
        times.append((time.perf_counter() - t0) * 1000.0)
        peak = max(peak, runner.sides['A'].peak, runner.sides['B'].peak)
        fin &= bool(np.all(np.isfinite(blk.get('A'))) and np.all(np.isfinite(blk.get('B'))))
    warm = np.asarray(times[int(0.5 * SR / BLOCK):])
    d = runner.snapshot()['display']['B']
    return dict(label=label, law=orz.LAW_NAMES[law], cells=int(g0.sum()), p50=float(np.percentile(warm, 50)),
                p99=float(np.percentile(warm, 99)), max=float(warm.max()), ok=bool(np.percentile(warm, 99) < BUDGET_MS),
                peak=peak, clip=dict(runner.snapshot()['clip_blocks']), figures=d['n_figures'], finite=fin)


def big_figure_boundary_ms(n_side, law, repeats=6):
    """One figure of n_side x n_side cells whose geometry changes at every boundary:
    (new, again) = the median ms of one engine block of ONE side when every
    boundary brings a shape not seen before (one different cell of the bottom row
    missing each time -- the worst case), and when the same shapes come back (a
    periodic figure: the participation comes from the shape cache)."""
    rows = cols = max(32, n_side + 4)
    g = np.zeros((rows, cols), np.uint8)
    g[2:2 + n_side, 2:2 + n_side] = 1
    e = registry.create(EID, ctx(6.0), side_params(law, STABLE_DECAY_S))
    e.init(g, None, GAIN)
    e.render_float(GAIN)
    out = []
    for _pass in range(2):
        ts = []
        for k in range(repeats):
            for missing in (k, None):              # the shape, then the full square again
                g2 = g.copy()
                if missing is not None:
                    g2[2 + n_side - 1, 2 + missing] = 0
                e.update_field(g2, None)
                t0 = time.perf_counter()
                e.render_float(GAIN)
                if missing is not None:
                    ts.append((time.perf_counter() - t0) * 1000.0)
        out.append(float(np.median(ts)))
    return out[0], out[1]


def timing_limits():
    rng = np.random.default_rng(11)
    soup = (rng.random((ROWS, COLS)) < 0.35).astype(np.uint8)
    rows = []
    for law in (FIXED, MODAL):
        rows.append(stress('random soup 35 %, 6 gen/s', soup, 6.0, 6.0, law))
        rows.append(stress('random soup 35 %, 20 % of the cells flipped every block', soup, 6.0, 3.0, law,
                           change_every_block=True))
        print(f"  stress {orz.LAW_NAMES[law]}: p99 {rows[-2]['p99']:.2f} / {rows[-1]['p99']:.2f} ms", flush=True)
    big = []
    for n in (6, 10, 14, 18, 22):
        fx, _fx2 = big_figure_boundary_ms(n, FIXED)
        md, md2 = big_figure_boundary_ms(n, MODAL)
        big.append(dict(side=n, cells=n * n, fixed_ms=fx, modal_ms=md, modal_again_ms=md2))
        print(f"  big figure {n}x{n}: Fixed {fx:.2f} ms, Modal new {md:.2f} / again {md2:.2f} ms", flush=True)
    return dict(stress=rows, big_figure=big, budget_ms=BUDGET_MS)


# -- the report ---------------------------------------------------------------------------------------
def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_markdown(rep, path):
    L = []
    L.append('# Objects — Decay law (losses from the history of the cells): measurements')
    L.append('')
    L.append(f"Generated {rep['generated']} on commit `{rep['commit']}`; numba {rep['numba']}.  REQ "
             f"`memory/req-objects-decay-2026-09-18.md`, preflight `memory/research/objects-decay-preflight-2026-09-18.json`.  "
             f"No listening result here.")
    L.append('')
    L.append('## 1. Targets against the preflight')
    L.append('')
    L.append('| case / law | banks checked | worst abs(T − preflight) s | worst abs(f − preflight) Hz |')
    L.append('|---|---|---|---|')
    for k, v in rep['targets'].items():
        L.append(f"| {k} | {v['banks']} | {v['worst_T']:.1e} | {v['worst_f']:.1e} |")
    L.append('')
    d1 = rep['d1']
    L.append('## 2. D1 — Fixed 0.947926 s / Common age')
    L.append('')
    L.append(f"Control: ln(1000) / {d1['control_T']} = {d1['control_gamma']:.4f} 1/s.  Side B over 5…20 s: mean target gamma "
             f"{d1['mean_target_gamma']:.4f} ({100 * d1['target_rel_error']:.2f} %), mean applied gamma "
             f"{d1['mean_applied_gamma']:.4f} ({100 * d1['applied_rel_error']:.2f} %); within 1 %: {d1['within_1pct']}.")
    L.append('')
    L.append('Phases of the mature cycles (generation mod 5, 5…19.5 s): the target T at the step, the mean APPLIED T over the '
             'interval, and the T60 measured in the rendered PCM between the steps (10 ms windows from +60 ms; median over the '
             'cycles):')
    L.append('')
    L.append('| phase | intervals | target T at step s | applied T mean s | PCM T60 A s | PCM T60 B s (min…max, measured) |')
    L.append('|---|---|---|---|---|---|')
    for p, v in d1['phases'].items():
        L.append(f"| {p} | {v['n']} | {v['target_T']:.3f} | {v['applied_T']:.3f} | {v['pcm_T60_A']:.3f} | "
                 f"{v['pcm_T60_B']:.3f} ({v['pcm_T60_B_spread'][0]:.3f}…{v['pcm_T60_B_spread'][1]:.3f}, "
                 f"{v['pcm_measured']}) |")
    L.append('')
    L.append(f"Max / min over the phases: applied T {d1['applied_T_phase_ratio_max_min']:.2f}; over the "
             f"{d1['pcm_phases_measured']} phases measurable in the PCM: T60 of B {d1['pcm_T60_B_phase_ratio_max_min']:.2f}, "
             f"of A {d1['pcm_T60_A_phase_ratio_max_min']:.2f}; PCM B vs applied T within "
             f"{100 * d1['pcm_B_vs_applied_max_rel']:.1f} %, PCM A vs the control within "
             f"{100 * d1['pcm_A_vs_control_max_rel']:.1f} %.  The phase without births (no new strike) continues the tail "
             f"of the previous interval, which is near the int16 floor by then: \"nan\" = fewer than 8 windows above 30 LSB.  "
             f"The phase sequence differs in the rendered audio of B and not of A: {d1['mechanism_in_audio']}.")
    L.append('')
    d2 = rep['d2']
    L.append('## 3. D2 — Common age / Modal age, the two scripted pauses')
    L.append('')
    L.append(f"Frequencies {[round(f, 3) for f in d2['frequencies_hz']]} Hz; pauses in the journal (sample, on, scripted): "
             f"{d2['journal_pauses']}.")
    L.append('')
    for k, p in enumerate(d2['pauses']):
        L.append(f"Pause {k + 1} ({p['pause_on']}…{p['pause_off']}), resonator magnitudes of the REAL engine from the first "
                 f"block end ({p['first_measurement']}):")
        L.append('')
        L.append('| elapsed s | Common lower / upper dB | Common upper − lower dB | Modal lower / upper dB | Modal upper − lower dB |')
        L.append('|---|---|---|---|---|')
        for r in p['engine']:
            L.append(f"| {r['elapsed_s']:.3f} | {r['Common']['relative_mode_db'][0]:.1f} / {r['Common']['relative_mode_db'][1]:.1f} | "
                     f"{r['Common']['upper_over_lower_db']:.1f} | {r['Modal']['relative_mode_db'][0]:.1f} / "
                     f"{r['Modal']['relative_mode_db'][1]:.1f} | {r['Modal']['upper_over_lower_db']:.1f} |")
        L.append('')
        L.append('Upper − lower in the rendered int16 PCM (DFT, 46 ms Hann windows; "-" = a component below 10 LSB):')
        L.append('')
        L.append('| after s | Common dB | Modal dB |')
        L.append('|---|---|---|')
        for r in p['pcm']:
            cm = '-' if r['Common'] is None else f"{r['Common']:.1f}"
            md = '-' if r['Modal'] is None else f"{r['Modal']:.1f}"
            L.append(f"| {r['t_after_s']:.2f} | {cm} | {md} |")
        pg = d2['pcm_growth'][k]
        L.append('')
        L.append(f"PCM growth of upper − lower from {pg['from_s']} to {pg['to_s']} s: Modal {pg['modal_db']:+.1f} dB, "
                 f"Common {pg['common_db']:+.1f} dB.")
        L.append('')
    L.append('Against the independent two-mode probe of the preflight (first pause):')
    L.append('')
    L.append('| elapsed s | Common engine / probe dB | Modal engine / probe dB | ratio Modal engine / probe |')
    L.append('|---|---|---|---|')
    for c in d2['probe_comparison']:
        L.append(f"| {c['elapsed_s']:.3f} | {[round(x, 2) for x in c['common_engine']]} / {[round(x, 2) for x in c['common_probe']]} | "
                 f"{[round(x, 2) for x in c['modal_engine']]} / {[round(x, 2) for x in c['modal_probe']]} | "
                 f"{c['ratio_modal_engine']:.2f} / {c['ratio_modal_probe']:.2f} |")
    L.append('')
    L.append(f"Worst difference engine − probe {d2['worst_probe_diff_db']:.3f} dB.  Growth of upper − lower over 0.5 s: "
             + '; '.join(f"pause {k + 1} Modal {g['modal_db']:+.1f} dB, Common {g['common_db']:+.1f} dB"
                         for k, g in enumerate(d2['growth'])) + f".  The mechanism shows in the real audio: {d2['mechanism_in_audio']}.")
    L.append('')
    d3 = rep['d3']
    L.append('## 4. D3 — Fixed 1.39 s / Modal age')
    L.append('')
    L.append(f"{d3['gen']} generations, {d3['changes']} accepted transitions; up to {d3['figures_max']} figures and "
             f"{d3['tails_max']} tails on side B ({d3['adaptive_tails_max']} tails carrying their gamma); identities A {d3['ids_A']} / "
             f"B {d3['ids_B']}, next id {d3['next_id']}; drops {d3['drops']}, in place {d3['inplace']}.  T max / min inside a "
             f"bank of side B: median {d3['t_spread_in_bank']['median']:.2f}, p90 {d3['t_spread_in_bank']['p90']:.2f}, "
             f"max {d3['t_spread_in_bank']['max']:.2f}.")
    L.append('')
    L.append('## 5. Isolation of the sides')
    L.append('')
    L.append('| scene | blocks | ids, cells, frequencies, weights, packets, a, Attack equal | blocks where the losses differ |')
    L.append('|---|---|---|---|')
    for k, v in rep['isolation'].items():
        L.append(f"| {k} | {v['blocks']} | {v['equal']} | {v['loss_blocks']} |")
    L.append('')
    lv = rep['levels']
    L.append('## 6. Levels (one constant factor of side B per experiment)')
    L.append('')
    L.append('| scene | window s | A − B unit gains (whole) dB | side gain B | A / B window RMS dB | A − B delivered dB | peak A / B | clip | finite | p99 ms (budget 7.98) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for case in CASES:
        u, d = lv['unit'][case['id']], lv['delivered'][case['id']]
        L.append(f"| {case['id']} | {d['window'][0]:g}…{d['window'][1]:g} | {u['window_a_minus_b_db']:+.2f} ({u['a_minus_b_db']:+.2f}) | "
                 f"{d['side_gain']['B']} | {d['A']['window_rms_db']:.2f} / {d['B']['window_rms_db']:.2f} | "
                 f"{d['window_a_minus_b_db']:+.2f} | {d['A']['peak']:.3f} / {d['B']['peak']:.3f} | "
                 f"{d['A']['clip_blocks']}/{d['B']['clip_blocks']} | {d['A']['finite'] and d['B']['finite']} | "
                 f"{d['block_ms']['p99']:.2f} (max {d['block_ms']['max']:.2f}) |")
    L.append('')
    L.append('D2 window: the active parts after 2 s, both pauses excluded.  The side gain scales both D2 components alike: '
             'the upper / lower ratio above does not depend on it.')
    L.append('')
    c = rep['continue']
    L.append('## 7. Continue')
    L.append('')
    L.append(f"- D2 from inside the first pause (paused {c['d2_inside_first_pause']['paused']}, "
             f"{c['d2_inside_first_pause']['pending']} scripted commands pending): 4 s exact "
             f"{c['d2_inside_first_pause']['exact']}, resumed {c['d2_inside_first_pause']['resumed']}.")
    L.append(f"- D1 during a gamma transition (A switched to Modal age 3 blocks earlier, gap "
             f"{c['d1_mid_gamma_transition']['gamma_gap']:.2f} 1/s): 300 blocks exact {c['d1_mid_gamma_transition']['exact']}.")
    L.append(f"- D3 after split / merge ({c['d3_after_split_merge']['adaptive_tails']} tails with their gamma, next id "
             f"{c['d3_after_split_merge']['next_id']}): 3 s exact {c['d3_after_split_merge']['exact']}.")
    L.append('')
    t = rep['timing']
    L.append('## 8. Timing and the limits of large fields')
    L.append('')
    L.append('| run | law | cells | figures (B, end) | p50 ms | p99 ms | max ms | within budget | peak | clip | finite |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in t['stress']:
        L.append(f"| {r['label']} | {r['law']} | {r['cells']} | {r['figures']} | {r['p50']:.2f} | {r['p99']:.2f} | "
                 f"{r['max']:.2f} | {r['ok']} | {r['peak']:.3f} | {r['clip']} | {r['finite']} |")
    L.append('')
    L.append('One figure whose geometry changes at every boundary (median ms of one block of ONE side; "new" = a '
             'shape never seen before at every boundary, the worst case; "again" = the same shapes return, as in a '
             'periodic figure):')
    L.append('')
    L.append('| figure | cells | Fixed ms | Modal new ms | Modal again ms |')
    L.append('|---|---|---|---|---|')
    for b in t['big_figure']:
        L.append(f"| {b['side']} x {b['side']} | {b['cells']} | {b['fixed_ms']:.2f} | {b['modal_ms']:.2f} | "
                 f"{b['modal_again_ms']:.2f} |")
    L.append('')
    L.append(f"Budget {t['budget_ms']:.2f} ms for BOTH sides.  The eigen-decompositions run only when the geometry or the "
             f"mode set changes, never per sample.")
    L.append('')
    L.append('## 9. Catalog')
    L.append('')
    cat = rep['catalog']
    if not cat.get('present'):
        L.append(f"Catalog {cat['root']} not checked (not built or --quick).")
    else:
        for rid, x in cat['records'].items():
            if 'error' in x:
                L.append(f"- {rid}: error {x['error']}")
            else:
                L.append(f"- {rid} {x['title'][:70]}: replay {x['status']} {x['reason'] or ''}, {x['pinned']} {x['commit']}, "
                         f"notes {x['has_notes']}, side gain {x['side_gain']}")
    L.append('')
    Path(path).write_text('\n'.join(L), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(CATALOG_ROOT))
    ap.add_argument('--quick', action='store_true', help='skip the catalog replay and the stress runs')
    a = ap.parse_args(argv)
    from casynth_lab import provenance as prov
    rep = dict(generated=time.strftime('%Y-%m-%d %H:%M'), commit=prov.current().get('commit'), numba=orz.HAVE_NUMBA)
    print('1. targets', flush=True)
    rep['targets'] = target_checks()
    print('2. D1', flush=True)
    rep['d1'] = d1_checks()
    print('3. D2', flush=True)
    rep['d2'] = d2_checks()
    print('4. D3', flush=True)
    rep['d3'] = d3_checks()
    print('5. isolation', flush=True)
    rep['isolation'] = isolation()
    print('6. levels', flush=True)
    rep['levels'] = levels()
    print('7. continue', flush=True)
    rep['continue'] = continue_checks()
    print('8. timing', flush=True)
    rep['timing'] = dict(stress=[], big_figure=[], budget_ms=BUDGET_MS) if a.quick else timing_limits()
    print('9. catalog', flush=True)
    rep['catalog'] = dict(root=a.root, present=False) if a.quick else verify_catalog(a.root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float) + '\n',
                                         encoding='utf-8')
    write_markdown(rep, OUT_DIR / 'report.md')
    print(f"report: {OUT_DIR / 'report.md'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
