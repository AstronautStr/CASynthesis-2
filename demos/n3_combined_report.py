"""N3 hand-over measurements (REQ memory/req-network-combined-n3-2026-09-16.md, "Проверки до
передачи", items 2-4) -> demos/results/network_n3/report.{json,md}.

    python demos/n3_combined_report.py [--catalog ROOT] [--no-catalog]

No audio device.  The numbers are technical evidence (levels, peaks, clips, timing,
reactions, the two control probes, stress probes), not listening verdicts.  Item 1
(the chain against the scalar reference, the network states against N2, the frequency
law) and the snapshot contract are the gates of tests/test_n3_tuned_events.py; this
script drives the delivered scenes, the stress probes and checks that every record of
the delivered catalog replays exactly.  The summary keeps the ready scenes and the
stress probes apart and lists every limitation found (N2 review P2).
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
from casynth_engine import step                                              # noqa: E402
from casynth_lab import DemoRunner, scene_from_doc, BLOCK, registry          # noqa: E402
from casynth_lab.catalog import Catalog                                      # noqa: E402
from casynth_lab.engine_api import EngineContext                             # noqa: E402
from casynth_lab import tuned_events as te                                   # noqa: E402
from casynth_lab import periodic_readout as pr                               # noqa: E402
from demos.build_n3_combined import (CASES, CATALOG_ROOT, scene_for, PARAMS_A, PARAMS_B,   # noqa: E402
                                     tumbler, SWAP_AT_S)
from demos.network_reference_n0.analysis import compare                      # noqa: E402

OUT_DIR = ROOT / 'demos' / 'results' / 'network_n3'
PREFLIGHT = ROOT / 'memory' / 'research' / 'network-n3-preflight-2026-09-16.json'
BUDGET_MS = BLOCK / SR * 1000.0
SIDES = ('A', 'B')
PF_NAME = dict(n3_travel='Glider', n3_growth='R-pentomino')     # n3_cycles: two windows, see below
CTX = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 6.0)


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-30))


def rms_db(y):
    """Stereo RMS (all samples of both channels), as the Researcher's preflight."""
    return db(np.sqrt(np.mean(np.asarray(y, float) ** 2)))


def run_blocks(runner, n_blocks, on_block=None):
    out = {s: [] for s in SIDES}
    times = []
    for b in range(n_blocks):
        t0 = time.perf_counter()
        blk = runner.next_block()
        times.append((time.perf_counter() - t0) * 1000.0)
        for s in SIDES:
            out[s].append(blk.get(s))
        if on_block is not None:
            on_block(b, runner)
    y = {s: np.concatenate(out[s]).astype(float) / 32767.0 for s in SIDES}
    return y, times


def window_stats(y, start_s, end_s):
    seg = y[int(start_s * SR):int(end_s * SR)]
    return dict(rms_db=rms_db(seg), rms_mono_db=rms_db(seg.mean(1)), peak=float(np.abs(seg).max()),
                peak_db=db(np.abs(seg).max()))


def field_checks(case):
    g = np.zeros((32, 32), np.uint8)
    for r, c in case['cells']:
        g[r, c] = 1
    K = pr.weights(32, 32)
    pops, events, per_node = [int(g.sum())], [], []
    seen = {g.tobytes(): 0}
    period = None
    cur = g
    for gen in range(1, 200):
        nxt = step(cur)
        E = nxt != cur
        events.append(int(E.sum()))
        per_node.append([round(float(v), 4) for v in pr.weighted_counts(K, E)])
        pops.append(int(nxt.sum()))
        key = nxt.tobytes()
        if period is None and key in seen:
            period = gen - seen[key]
        seen.setdefault(key, gen)
        cur = nxt
    info = dict(period=period, population=pops[:80], events_per_step=events[:80],
                events_range=[min(events[:72]), max(events[:72])], weighted_events_per_node=per_node[:72])
    if case['id'] == 'n3_cycles':
        t = np.zeros((32, 32), np.uint8)
        for r, c in tumbler():
            t[r, c] = 1
        seen_t, cur, p_t = {t.tobytes(): 0}, t, None
        for gen in range(1, 60):
            nxt = step(cur)
            if nxt.tobytes() in seen_t:
                p_t = gen - seen_t[nxt.tobytes()]
                break
            seen_t[nxt.tobytes()] = gen
            cur = nxt
        info['tumbler_period'] = p_t
    return info


def windows_of(case):
    if case['id'] == 'n3_cycles':
        return [('octagon', 2.0, 12.0, 'Octagon II'), ('swap', 12.0, 14.0, None),
                ('tumbler', 14.0, 24.0, 'Tumbler')]
    return [('main', 2.0, case['seconds'], PF_NAME[case['id']])]


def scene_measurements(case, preflight):
    doc = scene_for(case)
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    for kind, at, args in case['commands']:
        runner.post(kind, at=at, **args)
    n = int(math.ceil(case['seconds'] * SR / BLOCK))
    traj = []
    y, times = run_blocks(runner, n, lambda b, r: traj.append(
        (int(r.gen), int(r.grid.sum()), [round(float(v), 4) for v in r.sides['B'].engine.last_a],
         [round(float(v), 4) for v in r.sides['B'].engine.ratio])))
    info = dict(id=case['id'], seconds=case['seconds'], gen_end=int(runner.gen), block_ms=dict(
        p50=float(np.percentile(times[100:], 50)), p99=float(np.percentile(times[100:], 99)),
        max=float(np.max(times[100:])), budget=BUDGET_MS, ok=bool(np.percentile(times[100:], 99) < BUDGET_MS)))
    for s in SIDES:
        st = dict(clip_blocks=int(runner.sides[s].clip_blocks), finite=bool(np.all(np.isfinite(y[s]))),
                  peak=float(np.abs(y[s]).max()), windows={})
        for name, a, b, pf_case in windows_of(case):
            w = window_stats(y[s], a, b)
            if pf_case:
                pf = preflight.get((pf_case, s == 'B'))
                if pf:
                    w['preflight_rms_db'] = pf['rms_dbfs']
                    w['preflight_peak'] = pf['peak']
                    w['rms_diff_db'] = w['rms_db'] - pf['rms_dbfs']
            st['windows'][name] = w
        info[s] = st
    # the swap of N3.1: one journalled set_cells at the first boundary >= 12 s
    swaps = [(o, a) for o, q, k, a in runner.journal if k == 'set_cells']
    if swaps:
        o, a = swaps[0]
        blk = o // BLOCK
        info['swap'] = dict(out_sample=int(o), seconds=o / SR, cells=len(a['cells']),
                            gen_at_block=traj[blk][0], packet_max=max(traj[blk][2]),
                            count=len(swaps),
                            first_boundary_ok=bool(o == -(-int(round(SWAP_AT_S * SR)) // BLOCK) * BLOCK))
    main = windows_of(case)[0]
    info['ab'], *_ = compare(y['A'][int(main[1] * SR):int(main[2] * SR)].mean(1),
                             y['B'][int(main[1] * SR):int(main[2] * SR)].mean(1))
    info['ab_rms_diff_db'] = info['A']['windows'][main[0]]['rms_db'] - info['B']['windows'][main[0]]['rms_db']
    # tail: CA paused after the scene, 3 s without events, RMS of each second (float output)
    tail_runner = DemoRunner.from_state(runner.export_state())
    tail_runner.post('pause', on=True)
    tail_runner.next_block()
    gain = tail_runner._gain()
    for s in SIDES:
        eng = tail_runner.sides[s].engine
        yt = np.concatenate([eng.render_float(gain)[0] for _ in range(int(math.ceil(3.0 * SR / BLOCK)))])
        info[s]['tail_rms_db'] = [rms_db(yt[k * SR:(k + 1) * SR]) for k in range(3)]
        info[s]['tail_events'] = float(sum(eng.last_a))
    # continuation from the end state: the CA keeps evolving and both sides keep sounding
    cont = DemoRunner.from_state(runner.export_state())
    g0, gen0 = cont.grid.copy(), int(cont.gen)
    yc, _ = run_blocks(cont, int(math.ceil(4.0 * SR / BLOCK)))
    info['continue'] = dict(gen_advance=int(cont.gen) - gen0, field_changed=bool(not np.array_equal(g0, cont.grid)),
                            A_rms_db=rms_db(yc['A']), B_rms_db=rms_db(yc['B']))
    # the same saved state: one edit against no edit -> both sides react
    state = runner.export_state()
    edited, control = DemoRunner.from_state(state), DemoRunner.from_state(state)
    for r_ in (edited, control):
        r_.post('pause', on=True)
    moved = np.roll(runner.grid, 16, axis=0)
    cells = [[r, c, int(moved[r, c])] for r in range(32) for c in range(32) if moved[r, c] != runner.grid[r, c]]
    edited.post('set_cells', cells=cells)
    ye, _ = run_blocks(edited, int(math.ceil(4.0 * SR / BLOCK)))
    yc2, _ = run_blocks(control, int(math.ceil(4.0 * SR / BLOCK)))
    info['late_edit'] = dict(cells_changed=len(cells))
    for s in SIDES:
        d, *_ = compare(ye[s][:3 * SR].mean(1), yc2[s][:3 * SR].mean(1))
        info['late_edit'][s] = dict(differs=bool(not np.array_equal(ye[s], yc2[s])),
                                    edited_rms_db=rms_db(ye[s]), control_rms_db=rms_db(yc2[s]), **d)
    info['trajectory'] = dict(columns=['gen', 'population', 'B_packet_a_per_node', 'B_ratio_per_node'],
                              per_block=traj)
    return info


def control_probes():
    """REQ item 2: hold the frequencies while the impulses change (fixed mode, two fields),
    hold the impulse history while the frequencies change (one field, both modes); the
    silent retune and the excited tail across a retune."""
    def make(g, tuning, gain=0.028):
        e = registry.create(te.ENGINE_ID, CTX, dict(field_tuning=tuning, decay_s=0.8))
        e.init(g, None, gain)
        return e

    def render(e, seconds=2.0):
        return np.concatenate([e.raw_block() for _ in range(int(math.ceil(seconds * SR / BLOCK)))])

    def distances(ya, yb):
        ma, mb = ya.mean(1), yb.mean(1)
        d, *_ = compare(ma, mb)
        ea = np.sqrt((ma.reshape(-1, BLOCK) ** 2).mean(1))
        eb = np.sqrt((mb.reshape(-1, BLOCK) ** 2).mean(1))
        return dict(max_abs_diff=float(np.abs(ya - yb).max()), envelope_corr=float(np.corrcoef(ea, eb)[0, 1]), **d)

    g0 = np.zeros((32, 32), np.uint8)
    for r, c in ((9, 11), (9, 12), (9, 13), (11, 9), (12, 9), (13, 9), (25, 20), (0, 0), (31, 31)):
        g0[r, c] = 1
    g1 = np.roll(g0, 9, axis=1)
    out = {}
    a0, a1 = make(g0, 0), make(g1, 0)
    ya, yb = render(a0), render(a1)
    out['hold_frequencies_change_impulses'] = dict(frequencies_equal=bool(np.array_equal(a0.ffreq, a1.ffreq)),
                                                   packets=[[round(float(v), 3) for v in a0.last_a],
                                                            [round(float(v), 3) for v in a1.last_a]],
                                                   **distances(ya, yb))
    b0, b1 = make(g0, 0), make(g0, 1)
    za, zb = render(b0), render(b1)
    n0, n1 = b0.network_state(), b1.network_state()
    out['hold_impulses_change_frequencies'] = dict(
        network_states_equal=bool(all(np.array_equal(n0[k], n1[k]) for k in ('ring', 'lfilt', 'zf', 'zs'))),
        frequencies_equal=bool(np.array_equal(b0.ffreq, b1.ffreq)),
        ratio_B=[round(float(v), 3) for v in b1.ratio], **distances(za, zb))
    # silent retune: an empty field, the switch toggled -> the output stays exactly zero
    e = make(np.zeros((32, 32), np.uint8), 1)
    peak = 0.0
    for b in range(40):
        if b % 7 == 3:
            e.set_params(dict(field_tuning=b % 2, decay_s=0.8))
        peak = max(peak, float(np.abs(e.render_float(0.028)[0]).max()))
    out['silent_retune_peak'] = peak
    # excited tail: a packet, half a second, then the retune by the switch; per-second RMS after it
    e.inject(np.full(8, 0.5))
    for _ in range(int(0.5 * SR / BLOCK)):
        e.render_float(0.028)
    zre = e.zre.copy()
    e.set_params(dict(field_tuning=1, decay_s=0.8))
    kept = bool(np.array_equal(zre, e.zre))
    rms = []
    for sec in range(5):
        acc = np.concatenate([e.render_float(0.028)[0] for _ in range(int(SR / BLOCK))])
        rms.append(rms_db(acc))
    out['excited_tail_retune'] = dict(states_kept=kept, rms_db_per_second=rms, events_after=float(sum(e.last_a)))
    return out


def limits(case_id='n3_cycles'):
    """Dense / random fields, every-block toggles, repeated manual edits, both ends of
    Decay and moving knobs at the maximum standard gain (vol 1, level 1 -> 0.04), both
    sides, 6 s each.  The model has no reset guard (linear banks, bounded tanh network)."""
    rng = np.random.default_rng(20260916)
    doc = scene_for(next(c for c in CASES if c['id'] == case_id))
    results = []
    n = int(math.ceil(6.0 * SR / BLOCK))
    for decay in (0.20, 1.50):
        for mode in ('dense_random_evolving', 'random_every_block', 'full_toggle_every_block',
                     'repeated_edits_paused', 'knobs_moving'):
            runner = DemoRunner(scene_from_doc(doc), vol=1.0)
            for s in SIDES:
                runner.post('set_param', side=s, name='decay_s', value=decay)
            if mode == 'dense_random_evolving':
                g = (rng.random((32, 32)) < 0.5).astype(np.uint8)
                runner.post('set_cells', cells=[[r, c, int(g[r, c])] for r in range(32) for c in range(32)])
            runner.post('start', at=0)
            if mode == 'repeated_edits_paused':
                runner.post('pause', at=BLOCK, on=True)

            def on_block(b, r, mode=mode, decay=decay):
                if mode == 'random_every_block':
                    g = (rng.random((32, 32)) < 0.5).astype(np.uint8)
                    r.post('set_cells', cells=[[rr, c, int(g[rr, c])] for rr in range(32) for c in range(32)])
                elif mode == 'full_toggle_every_block':
                    r.post('set_cells', cells=[[rr, c, int(1 - r.grid[rr, c])] for rr in range(32) for c in range(32)])
                elif mode in ('repeated_edits_paused', 'knobs_moving') and b % 12 == 0:
                    r0, c0 = int(rng.integers(0, 27)), int(rng.integers(0, 27))
                    r.post('set_cells', cells=[[r0 + i, c0 + j, int(1 - r.grid[r0 + i, c0 + j])]
                                               for i in range(5) for j in range(5)])
                    if mode == 'knobs_moving':
                        k = b // 12
                        for s in SIDES:
                            r.post('set_param', side=s, name='decay_s', value=(0.2 if k % 2 else 1.5))
                            r.post('set_param', side=s, name='field_tuning', value=(k // 2) % 2)
            y, times = run_blocks(runner, n, on_block)
            row = dict(decay_s=decay, mode=mode, gain=runner._gain(),
                       block_ms_p99=float(np.percentile(times[50:], 99)))
            for s in SIDES:
                st = window_stats(y[s], 0.5, 6.0)
                st['clip_blocks'] = int(runner.sides[s].clip_blocks)
                st['finite'] = bool(np.all(np.isfinite(y[s])))
                row[s] = st
            results.append(row)
    return results


def timing(case_id='n3_travel', seconds=8.0):
    doc = scene_for(next(c for c in CASES if c['id'] == case_id))
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    _, times = run_blocks(runner, int(seconds * SR / BLOCK))
    t = np.array(times[100:])
    return dict(p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)),
                p99=float(np.percentile(t, 99)), max=float(t.max()), budget=BUDGET_MS,
                ok=bool(np.percentile(t, 99) < BUDGET_MS), numba=te.HAVE_NUMBA)


def verify_catalog(root):
    catalog = Catalog(str(root))
    checked = []
    for record, error in catalog.list():
        if error:
            checked.append(dict(error=str(error)))
            continue
        result = catalog.replay(record.id, yield_cpu=False)
        checked.append(dict(rid=record.id, title=record.title, status=result.status,
                            reason=result.reason, notes_start=(record.notes or '')[:40]))
    return checked


def write_markdown(rep, path):
    L = ['# N3 tuned events -- hand-over measurements (technical evidence, not listening verdicts)', '',
         f"Generated {rep['generated']} at commit {rep['commit']}.  SR {SR}, block {BLOCK} "
         f"({BUDGET_MS:.2f} ms), one engine `{te.ENGINE_ID}` on both sides (A field_tuning 0, "
         f"B field_tuning 1, decay {PARAMS_A['decay_s']} s), rho {te.RHO}, output x{te.OUT_SCALE:g}, "
         f"gain 0.028 (vol 0.7).  RMS = stereo RMS as in the preflight; (pre) = preflight value.", '',
         '## Scenes (from a fresh start; windows in seconds)', '',
         '| scene | window | A rms dB (pre) | B rms dB (pre) | A peak (pre) | B peak (pre) | A-B dB |',
         '|---|---|---|---|---|---|---|']
    for s in rep['scenes']:
        for name in s['A']['windows']:
            wa, wb = s['A']['windows'][name], s['B']['windows'][name]
            pa = f" ({wa['preflight_rms_db']:.1f})" if 'preflight_rms_db' in wa else ''
            pb = f" ({wb['preflight_rms_db']:.1f})" if 'preflight_rms_db' in wb else ''
            ka = f" ({wa['preflight_peak']:.3f})" if 'preflight_peak' in wa else ''
            kb = f" ({wb['preflight_peak']:.3f})" if 'preflight_peak' in wb else ''
            L.append(f"| {s['id']} | {name} | {wa['rms_db']:.1f}{pa} | {wb['rms_db']:.1f}{pb} | "
                     f"{wa['peak']:.3f}{ka} | {wb['peak']:.3f}{kb} | {wa['rms_db'] - wb['rms_db']:.2f} |")
    L += ['', '| scene | period | events/step (72 gens) | gens | clip A/B | finite | A tail 1/2/3 s dB | B tail 1/2/3 s dB | spectral dist A-B dB | p99 ms |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for s in rep['scenes']:
        f = s['field']
        per = f"{f['period']}" + (f" / {f['tumbler_period']}" if 'tumbler_period' in f else '')
        ta = ' / '.join(f"{v:.0f}" for v in s['A']['tail_rms_db'])
        tb = ' / '.join(f"{v:.0f}" for v in s['B']['tail_rms_db'])
        L.append(f"| {s['id']} | {per} | {f['events_range'][0]}-{f['events_range'][1]} | {s['gen_end']} | "
                 f"{s['A']['clip_blocks']}/{s['B']['clip_blocks']} | {s['A']['finite'] and s['B']['finite']} | "
                 f"{ta} | {tb} | {s['ab'].get('log_spectral_distance_db', float('nan')):.2f} | {s['block_ms']['p99']:.2f} |")
    sw = next((s['swap'] for s in rep['scenes'] if 'swap' in s), None)
    if sw:
        L += ['', f"N3.1 swap: {sw['count']} `set_cells` command of {sw['cells']} cells at sample "
              f"{sw['out_sample']} ({sw['seconds']:.4f} s; first block boundary at or after 12 s: "
              f"{sw['first_boundary_ok']}), generation {sw['gen_at_block']} at that block, "
              f"largest packet a of the swap block {sw['packet_max']:.3f}."]
    L += ['', '## Continuation and a late edit (same saved end state)', '',
          '| scene | gens in 4 s | field moved | edit cells | A differs / spectral dist dB | B differs / spectral dist dB |',
          '|---|---|---|---|---|---|']
    for s in rep['scenes']:
        le, co = s['late_edit'], s['continue']
        L.append(f"| {s['id']} | {co['gen_advance']} | {co['field_changed']} | {le['cells_changed']} | "
                 f"{le['A']['differs']} / {le['A'].get('log_spectral_distance_db', float('nan')):.2f} | "
                 f"{le['B']['differs']} / {le['B'].get('log_spectral_distance_db', float('nan')):.2f} |")
    c = rep['controls']
    h1, h2 = c['hold_frequencies_change_impulses'], c['hold_impulses_change_frequencies']
    L += ['', '## Control probes (engine level, 2 s, raw output)', '',
          f"- Frequencies held (fixed mode), impulses from two fields: frequencies equal {h1['frequencies_equal']}, "
          f"max |diff| {h1['max_abs_diff']:.3f}, envelope corr {h1['envelope_corr']:.3f}, spectral dist "
          f"{h1.get('log_spectral_distance_db', float('nan')):.2f} dB.",
          f"- Impulse history held (one field), fixed vs field tuning: network states equal {h2['network_states_equal']}, "
          f"frequencies equal {h2['frequencies_equal']}, max |diff| {h2['max_abs_diff']:.3f}, envelope corr "
          f"{h2['envelope_corr']:.3f}, spectral dist {h2.get('log_spectral_distance_db', float('nan')):.2f} dB.",
          f"- Silent retune (empty field, switch toggled): output peak {c['silent_retune_peak']:.1e}.",
          f"- Excited tail across a retune: states kept {c['excited_tail_retune']['states_kept']}, RMS per second "
          f"after it {', '.join(f'{v:.0f}' for v in c['excited_tail_retune']['rms_db_per_second'])} dB, "
          f"events after {c['excited_tail_retune']['events_after']:.0f}."]
    L += ['', '## Stress probes at the maximum standard gain (0.04), 6 s each, both sides', '',
          '| decay s | mode | A rms dB | A peak | B rms dB | B peak | clip A/B | finite | p99 ms |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in rep['limits']:
        L.append(f"| {r['decay_s']:.2f} | {r['mode']} | {r['A']['rms_db']:.1f} | {r['A']['peak']:.3f} | "
                 f"{r['B']['rms_db']:.1f} | {r['B']['peak']:.3f} | {r['A']['clip_blocks']}/{r['B']['clip_blocks']} | "
                 f"{r['A']['finite'] and r['B']['finite']} | {r['block_ms_p99']:.2f} |")
    t = rep['timing']
    L += ['', f"Timing (n3_travel, 8 s, both sides per block): p50 {t['p50']:.2f} / p95 {t['p95']:.2f} / "
          f"p99 {t['p99']:.2f} / max {t['max']:.2f} ms of {t['budget']:.2f} ms (numba {t['numba']})."]
    if rep.get('catalog'):
        L += ['', '## Delivered catalog', '', f"Root: `{rep['catalog_root']}`", '']
        for c_ in rep['catalog']:
            L.append(f"- {c_.get('rid')}  {c_.get('title')}  -> {c_.get('status')} {c_.get('reason') or ''}  "
                     f"(notes: {c_.get('notes_start', '')!r})")
    L += ['', '## Summary', '', f"- Scenes: {rep['summary']['scenes']}", f"- Stress probes: {rep['summary']['stress']}",
          '- Limitations:']
    L += [f"  - {x}" for x in rep['summary']['limitations']]
    L.append('')
    path.write_text('\n'.join(L), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalog', default=str(CATALOG_ROOT))
    ap.add_argument('--no-catalog', action='store_true')
    a = ap.parse_args(argv)
    preflight = {}
    if PREFLIGHT.is_file():
        pf = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
        preflight = {(m['case'], bool(m['tune'])): m for m in pf['measurements'] if m['mode'] == 'scene'}
    try:
        import subprocess
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip() or 'unknown'
    except Exception:                                                   # pragma: no cover
        commit = 'unknown'
    rep = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), commit=commit, scenes=[])
    for case in CASES:
        print('scene', case['id'], flush=True)
        info = scene_measurements(case, preflight)
        info['field'] = field_checks(case)
        rep['scenes'].append(info)
    print('controls', flush=True)
    rep['controls'] = control_probes()
    print('limits', flush=True)
    rep['limits'] = limits()
    print('timing', flush=True)
    rep['timing'] = timing()
    root = Path(a.catalog)
    if not a.no_catalog and root.is_dir():
        print('catalog', flush=True)
        rep['catalog_root'] = str(root)
        rep['catalog'] = verify_catalog(root)
    scene_problems, stress_problems, limitations = [], [], []
    for s in rep['scenes']:
        for side in SIDES:
            if s[side]['clip_blocks'] or not s[side]['finite']:
                scene_problems.append(f"{s['id']} {side}: clip / non-finite")
            if not s['late_edit'][side]['differs']:
                scene_problems.append(f"{s['id']} {side}: a late edit does not change the sound")
            if s[side]['tail_rms_db'][2] >= s[side]['tail_rms_db'][0]:
                scene_problems.append(f"{s['id']} {side}: the tail does not decay")
            for name, w in s[side]['windows'].items():
                if 'rms_diff_db' in w and abs(w['rms_diff_db']) > 1.0:
                    limitations.append(f"{s['id']} {side} window {name}: RMS {w['rms_diff_db']:+.1f} dB "
                                       f"against the preflight")
        if not s['block_ms']['ok']:
            scene_problems.append(f"{s['id']}: p99 {s['block_ms']['p99']:.2f} ms over budget")
        if not s['continue']['field_changed']:
            scene_problems.append(f"{s['id']}: the field does not evolve after Continue")
        if s['id'] == 'n3_cycles' and (not s.get('swap') or s['swap']['count'] != 1 or not s['swap']['first_boundary_ok']):
            scene_problems.append("n3_cycles: the swap is not one command at the first boundary >= 12 s")
    c = rep['controls']
    if not (c['hold_frequencies_change_impulses']['frequencies_equal'] and
            c['hold_frequencies_change_impulses']['max_abs_diff'] > 1e-3):
        scene_problems.append("control probe: held frequencies / changed impulses did not give different responses")
    if not (c['hold_impulses_change_frequencies']['network_states_equal'] and
            not c['hold_impulses_change_frequencies']['frequencies_equal'] and
            c['hold_impulses_change_frequencies']['max_abs_diff'] > 1e-3):
        scene_problems.append("control probe: held impulses / changed frequencies did not give different responses")
    if c['silent_retune_peak'] != 0.0 or not c['excited_tail_retune']['states_kept']:
        scene_problems.append("control probe: a silent retune made sound or a retune touched the bank states")
    for r in rep['limits']:
        for side in SIDES:
            if r[side]['clip_blocks'] or not r[side]['finite']:
                stress_problems.append(f"limits {r['mode']} decay {r['decay_s']} {side}: clip / non-finite")
    worst = max(rep['limits'], key=lambda r: max(r['A']['peak'], r['B']['peak']))
    worst_peak = max(worst['A']['peak'], worst['B']['peak'])
    if not rep['timing']['ok']:
        scene_problems.append(f"timing: p99 {rep['timing']['p99']:.2f} ms over budget")
    for c_ in rep.get('catalog', []):
        if c_.get('status') != 'match':
            scene_problems.append(f"catalog {c_.get('rid')}: {c_.get('status')} {c_.get('reason')}")
    limitations += [
        f"stress peak {worst_peak:.3f} at gain 0.04 ({worst['mode']}, decay {worst['decay_s']} s): the headroom is "
        f"measured for these finite probes, not guaranteed for any playing",
        "the model has no reset guard (linear banks, bounded tanh network): 'resets' are not a quantity here",
        "preflight Tumbler values come from a fresh 12 s start; here the Tumbler window follows the swap "
        "with the Octagon tail still ringing",
        "tails are measured on the float output (int16 quantises below -90 dBFS to zero)",
        "no listening in this report: hearing the difference is the user's verdict",
    ]
    rep['problems'] = dict(scenes=scene_problems, stress=stress_problems)
    rep['summary'] = dict(
        scenes=('no clipping, no NaN, tails decay, late edits change the sound, control probes as expected, '
                'all records replay exactly' if not scene_problems else '; '.join(scene_problems)),
        stress=(f"no clipping, no NaN in all {len(rep['limits'])} probes; worst peak {worst_peak:.3f}"
                if not stress_problems else '; '.join(stress_problems)),
        limitations=limitations)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    write_markdown(rep, OUT_DIR / 'report.md')
    print('summary scenes:', rep['summary']['scenes'])
    print('summary stress:', rep['summary']['stress'])
    print('written:', OUT_DIR / 'report.json', OUT_DIR / 'report.md')
    return 1 if (scene_problems or stress_problems) else 0


if __name__ == '__main__':
    raise SystemExit(main())
