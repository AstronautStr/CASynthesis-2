"""N2 hand-over measurements (REQ memory/req-network-events-n2-2026-09-16.md, "Проверки до
передачи", items 2 and 3) -> demos/results/network_n2/report.{json,md}.

    python demos/n2_events_report.py [--catalog ROOT] [--no-catalog]

No audio device.  The numbers are technical evidence (levels, peaks, clips, resets,
timing, reactions), not listening verdicts.  Item 1 (math / causality) and item 4
(bench contract) are the gates of tests/test_n2_events.py; this script reads the
delivered catalog and checks every record replays exactly when --catalog exists.
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
from casynth_lab import event_network as en                                  # noqa: E402
from casynth_lab import periodic_readout as pr                               # noqa: E402
from demos.build_n2_events import CASES, CATALOG_ROOT, scene_for, PARAMS_A, PARAMS_B   # noqa: E402
from demos.network_reference_n0.analysis import compare                      # noqa: E402

OUT_DIR = ROOT / 'demos' / 'results' / 'network_n2'
PREFLIGHT = ROOT / 'memory' / 'research' / 'network-n2-preflight-2026-09-16.json'
BUDGET_MS = BLOCK / SR * 1000.0
SIDES = ('A', 'B')


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-30))


def rms_db(y):
    return db(np.sqrt(np.mean(np.asarray(y, float) ** 2)))


def run_blocks(runner, n_blocks, on_block=None):
    """Drive the runner; returns float stereo per side (-1..1) and per-block ms."""
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


def side_stats(y, start_s=2.0, end_s=12.0):
    seg = y[int(start_s * SR):int(end_s * SR)]
    return dict(rms_db=rms_db(seg.mean(1)), peak=float(np.abs(y).max()),
                peak_db=db(np.abs(y).max()), finite=bool(np.all(np.isfinite(y))))


def field_checks(case):
    """Generations of the scene: population / events per step, period, the REQ numbers."""
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
    return dict(period=period, population=pops, events_per_step=events,
                pop_at_72=pops[72], events_after_72=events[72],
                events_range=[min(events[:72]), max(events[:72])],
                weighted_events_per_node=per_node[:72])


def scene_measurements(case, preflight):
    doc = scene_for(case)
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    n = int(math.ceil(12.0 * SR / BLOCK))
    traj = []
    y, times = run_blocks(runner, n, lambda b, r: traj.append(
        (int(r.gen), int(r.grid.sum()), [round(float(v), 4) for v in r.sides['B'].engine.last_a])))
    info = dict(id=case['id'], gen_at_12s=int(runner.gen), block_ms=dict(
        p50=float(np.percentile(times[100:], 50)), p99=float(np.percentile(times[100:], 99)),
        max=float(np.max(times[100:])), budget=BUDGET_MS, ok=bool(np.percentile(times[100:], 99) < BUDGET_MS)))
    for s in SIDES:
        st = side_stats(y[s])
        st['clip_blocks'] = int(runner.sides[s].clip_blocks)
        st['resets'] = int(runner.sides[s].engine.resets.sum()) if s == 'A' else 0
        info[s] = st
    pf = preflight.get(case['id'])
    if pf:
        info['preflight'] = dict(A_rms_db=pf['A_calibrated_rms_db'], B_rms_db=pf['B_calibrated_rms_db'],
                                 B_peak=pf['B_calibrated_stereo_peak'],
                                 A_diff_db=info['A']['rms_db'] - pf['A_calibrated_rms_db'],
                                 B_diff_db=info['B']['rms_db'] - pf['B_calibrated_rms_db'])
    info['ab'], *_ = compare(y['A'][2 * SR:12 * SR].mean(1), y['B'][2 * SR:12 * SR].mean(1))
    # tail of B: CA paused after 12 s, 2 s without events, RMS of the last second --
    # measured on the float output (the int16 PCM quantises below -90 dBFS to zero)
    tail_runner = DemoRunner.from_state(runner.export_state())
    tail_runner.post('pause', on=True)
    tail_runner.next_block()                                       # applies the pause
    gain = tail_runner._gain()
    yt = {s: np.concatenate([tail_runner.sides[s].engine.render_float(gain)[0]
                             for _ in range(int(math.ceil(2.0 * SR / BLOCK)))]) for s in SIDES}
    info['B']['tail_last_second_rms_db'] = rms_db(yt['B'][SR:].mean(1))
    info['B']['tail_first_second_rms_db'] = rms_db(yt['B'][:SR].mean(1))
    info['A']['paused_last_second_rms_db'] = rms_db(yt['A'][SR:].mean(1))
    # continuation from the end state: the CA keeps evolving and both sides keep sounding
    cont = DemoRunner.from_state(runner.export_state())
    g0, gen0 = cont.grid.copy(), int(cont.gen)
    yc, _ = run_blocks(cont, int(math.ceil(4.0 * SR / BLOCK)))
    info['continue'] = dict(gen_advance=int(cont.gen) - gen0, field_changed=bool(not np.array_equal(g0, cont.grid)),
                            A_rms_db=rms_db(yc['A'].mean(1)), B_rms_db=rms_db(yc['B'].mean(1)))
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
        d, *_ = compare(ye[s][:3 * SR].mean(1), yc2[s][:3 * SR].mean(1))   # from the edit on (B decays fast)
        info['late_edit'][s] = dict(differs=bool(not np.array_equal(ye[s], yc2[s])),
                                    edited_rms_db=rms_db(ye[s].mean(1)), control_rms_db=rms_db(yc2[s].mean(1)), **d)
    info['trajectory'] = dict(columns=['gen', 'population', 'B_packet_a_per_node'], per_block=traj)
    return info


def transfer_between_nodes():
    """The same event delivered to different nodes (B alone): energy-equalised envelope /
    spectrum distances -- at engine level (a unit packet) and at field level (one cell
    toggled at a node centre)."""
    ctx = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 6.0)
    n = int(SR / BLOCK)

    def render_inject(src):
        e = registry.create(en.ENGINE_ID, ctx, dict(PARAMS_B))
        e.init(np.zeros((32, 32), np.uint8), None, 0.028)
        e.raw_block()
        a = np.zeros(8); a[src] = 0.5
        e.inject(a)
        return np.concatenate([e.raw_block(events=False) for _ in range(n)])

    def render_cell(src):
        e = registry.create(en.ENGINE_ID, ctx, dict(PARAMS_B))
        e.init(np.zeros((32, 32), np.uint8), None, 0.028)
        e.raw_block()
        cx, cy = pr.centre_cells(32, 32)[src]
        g = np.zeros((32, 32), np.uint8)
        g[int(round(cy)), int(round(cx))] = 1
        e.update_field(g, None)
        y = np.concatenate([e.raw_block() for _ in range(n)])
        return y, [round(float(v), 4) for v in e.last_a]

    def distances(ya, yb):
        ma, mb = ya.mean(1), yb.mean(1)
        ma, mb = ma / np.sqrt(np.mean(ma ** 2)), mb / np.sqrt(np.mean(mb ** 2))
        ea = np.sqrt((ma.reshape(-1, BLOCK) ** 2).mean(1))
        eb = np.sqrt((mb.reshape(-1, BLOCK) ** 2).mean(1))
        sa = 20 * np.log10(np.abs(np.fft.rfft(ma)) + 1e-12)
        sb = 20 * np.log10(np.abs(np.fft.rfft(mb)) + 1e-12)
        d, *_ = compare(ma, mb)
        return dict(envelope_corr=float(np.corrcoef(ea, eb)[0, 1]),
                    log_spectrum_rms_db=float(np.sqrt(np.mean((sa - sb) ** 2))),
                    left_right_ratio=[float(np.abs(ya[:, 0]).mean() / max(np.abs(ya[:, 1]).mean(), 1e-30)),
                                      float(np.abs(yb[:, 0]).mean() / max(np.abs(yb[:, 1]).mean(), 1e-30))],
                    **d)

    out = dict(unit_packet={}, one_cell={})
    for a_, b_ in ((0, 5), (1, 2), (3, 7)):
        out['unit_packet'][f'{a_}_vs_{b_}'] = distances(render_inject(a_), render_inject(b_))
        (ya, aa), (yb, ab) = render_cell(a_), render_cell(b_)
        out['one_cell'][f'{a_}_vs_{b_}'] = dict(packets=[aa, ab], **distances(ya, yb))
    return out


def limits(case_id='n2_rhythm'):
    """Dense / random fields, repeated manual edits and both ends of Response at the
    maximum standard gain (vol 1, level 1 -> 0.04) on BOTH sides, 6 s each."""
    rng = np.random.default_rng(20260916)
    doc = scene_for(next(c for c in CASES if c['id'] == case_id))
    results = []
    n = int(math.ceil(6.0 * SR / BLOCK))
    for rho in (0.60, 0.95):
        for mode in ('dense_random_evolving', 'random_every_block', 'full_toggle_every_block',
                     'repeated_edits_paused'):
            runner = DemoRunner(scene_from_doc(doc), vol=1.0)
            runner.post('set_param', side='B', name='rho', value=rho)
            if mode == 'dense_random_evolving':
                g = (rng.random((32, 32)) < 0.5).astype(np.uint8)
                runner.post('set_cells', cells=[[r, c, int(g[r, c])] for r in range(32) for c in range(32)])
            runner.post('start', at=0)
            if mode == 'repeated_edits_paused':
                runner.post('pause', at=BLOCK, on=True)

            def on_block(b, r, mode=mode):
                if mode == 'random_every_block':
                    g = (rng.random((32, 32)) < 0.5).astype(np.uint8)
                    r.post('set_cells', cells=[[rr, c, int(g[rr, c])] for rr in range(32) for c in range(32)])
                elif mode == 'full_toggle_every_block':
                    r.post('set_cells', cells=[[rr, c, int(1 - r.grid[rr, c])] for rr in range(32) for c in range(32)])
                elif mode == 'repeated_edits_paused' and b % 12 == 0:      # a 5 x 5 block ~every 96 ms
                    r0, c0 = int(rng.integers(0, 27)), int(rng.integers(0, 27))
                    r.post('set_cells', cells=[[r0 + i, c0 + j, int(1 - r.grid[r0 + i, c0 + j])]
                                               for i in range(5) for j in range(5)])
            y, times = run_blocks(runner, n, on_block)
            row = dict(rho=rho, mode=mode, gain=runner._gain(),
                       block_ms_p99=float(np.percentile(times[50:], 99)))
            for s in SIDES:
                st = side_stats(y[s], 0.5, 6.0)
                st['clip_blocks'] = int(runner.sides[s].clip_blocks)
                st['resets'] = int(runner.sides[s].engine.resets.sum()) if s == 'A' else 0
                row[s] = st
            if mode == 'full_toggle_every_block' and rho == 0.60:
                # the same pathological stimulus on the plain N1 gutter_field: the node
                # resets belong to the Gutter model (its NaN guard), not to the readout
                row['A']['n1_gutter_field_resets'] = n1_resets_under_full_toggle(1)
                row['A']['n1_gutter_field_resets_every_4_blocks'] = n1_resets_under_full_toggle(4)
            results.append(row)
    return results


def n1_resets_under_full_toggle(period_blocks, seconds=6.0):
    ctx = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 6.0)
    e = registry.create('gutter_field', ctx, dict(PARAMS_A))
    g = np.zeros((32, 32), np.uint8)
    e.init(g, None, MASTER_GAIN)
    for b in range(int(seconds * SR / BLOCK)):
        if b % period_blocks == 0:
            g = 1 - g
            e.update_field(g, None)
        e.render(MASTER_GAIN, 0)
    return int(e.resets.sum())


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
    L = ['# N2 events -- hand-over measurements (technical evidence, not listening verdicts)', '',
         f"Generated {rep['generated']} at commit {rep['commit']}.  SR {SR}, block {BLOCK} "
         f"({BUDGET_MS:.2f} ms), calibration A x20 x 10^(-16.5/20), B x96, gain 0.028 (vol 0.7).", '',
         '## Scenes (12 s from a fresh start, mono RMS over seconds 2-12)', '',
         '| scene | period | events/step | pop@72 / ev after | A rms dB (pre) | B rms dB (pre) | A peak | B peak (pre) | clip A/B | resets A | B tail +1 s / +2 s dB | p99 ms |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for s in rep['scenes']:
        f = s['field']
        pf = s.get('preflight', {})
        L.append(f"| {s['id']} | {f['period']} | {f['events_range'][0]}-{f['events_range'][1]} | "
                 f"{f['pop_at_72']} / {f['events_after_72']} | {s['A']['rms_db']:.1f} ({pf.get('A_rms_db', float('nan')):.1f}) | "
                 f"{s['B']['rms_db']:.1f} ({pf.get('B_rms_db', float('nan')):.1f}) | {s['A']['peak']:.3f} | "
                 f"{s['B']['peak']:.3f} ({pf.get('B_peak', float('nan')):.3f}) | {s['A']['clip_blocks']}/{s['B']['clip_blocks']} | "
                 f"{s['A']['resets']} | {s['B']['tail_first_second_rms_db']:.1f} / {s['B']['tail_last_second_rms_db']:.1f} | {s['block_ms']['p99']:.2f} |")
    L += ['', '## Continuation and a late edit (same saved end state)', '',
          '| scene | gens in 4 s | field moved | edit cells | A differs / spectral dist dB | B differs / spectral dist dB |',
          '|---|---|---|---|---|---|']
    for s in rep['scenes']:
        le, co = s['late_edit'], s['continue']
        L.append(f"| {s['id']} | {co['gen_advance']} | {co['field_changed']} | {le['cells_changed']} | "
                 f"{le['A']['differs']} / {le['A'].get('log_spectral_distance_db', float('nan')):.2f} | "
                 f"{le['B']['differs']} / {le['B'].get('log_spectral_distance_db', float('nan')):.2f} |")
    L += ['', '## B: the same event on different nodes (energy equalised, 1 s)', '',
          '| pair | kind | envelope corr | log-spectrum rms dB | spectral dist dB | L/R ratio a, b |',
          '|---|---|---|---|---|---|']
    for kind in ('unit_packet', 'one_cell'):
        for pair, d in rep['transfer'][kind].items():
            L.append(f"| {pair} | {kind} | {d['envelope_corr']:.3f} | {d['log_spectrum_rms_db']:.2f} | "
                     f"{d.get('log_spectral_distance_db', float('nan')):.2f} | {d['left_right_ratio'][0]:.2f}, {d['left_right_ratio'][1]:.2f} |")
    L += ['', '## Limits at the maximum standard gain (0.04), 6 s each', '',
          '| rho | mode | A rms dB | A peak | B rms dB | B peak | clip A/B | resets A | finite | p99 ms |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for r in rep['limits']:
        L.append(f"| {r['rho']:.2f} | {r['mode']} | {r['A']['rms_db']:.1f} | {r['A']['peak']:.3f} | "
                 f"{r['B']['rms_db']:.1f} | {r['B']['peak']:.3f} | {r['A']['clip_blocks']}/{r['B']['clip_blocks']} | "
                 f"{r['A']['resets']} | {r['A']['finite'] and r['B']['finite']} | {r['block_ms_p99']:.2f} |")
    if rep.get('catalog'):
        L += ['', '## Delivered catalog', '', f"Root: `{rep['catalog_root']}`", '']
        for c in rep['catalog']:
            L.append(f"- {c.get('rid')}  {c.get('title')}  -> {c.get('status')} {c.get('reason') or ''}  "
                     f"(notes: {c.get('notes_start', '')!r})")
    if rep.get('observations'):
        L += ['', 'Observations (not hand-over failures): ' + '; '.join(rep['observations']) +
              '.  The plain N1 gutter_field under the same every-block full toggle: '
              f"{rep['limits'][2]['A'].get('n1_gutter_field_resets')} resets; toggled every 4 blocks: "
              f"{rep['limits'][2]['A'].get('n1_gutter_field_resets_every_4_blocks')} resets."]
    L += ['', f"Summary: {rep['summary']}", '']
    path.write_text('\n'.join(L), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalog', default=str(CATALOG_ROOT))
    ap.add_argument('--no-catalog', action='store_true')
    a = ap.parse_args(argv)
    preflight = {}
    if PREFLIGHT.is_file():
        pf = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
        names = dict(pulse='n2_rhythm', travel='n2_travel', growth='n2_growth')
        preflight = {names[m['case']]: m for m in pf['measurements']}
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
    print('transfer', flush=True)
    rep['transfer'] = transfer_between_nodes()
    print('limits', flush=True)
    rep['limits'] = limits()
    root = Path(a.catalog)
    if not a.no_catalog and root.is_dir():
        print('catalog', flush=True)
        rep['catalog_root'] = str(root)
        rep['catalog'] = verify_catalog(root)
    problems = []
    for s in rep['scenes']:
        for side in SIDES:
            if s[side]['clip_blocks'] or s[side]['resets'] or not s[side]['finite']:
                problems.append(f"{s['id']} {side}: clip/reset/non-finite")
            if not s['late_edit'][side]['differs']:
                problems.append(f"{s['id']} {side}: a late edit does not change the sound")
        if not s['block_ms']['ok']:
            problems.append(f"{s['id']}: p99 {s['block_ms']['p99']:.2f} ms over budget")
        if not s['continue']['field_changed']:
            problems.append(f"{s['id']}: the field does not evolve after Continue")
    observations = []
    for r in rep['limits']:
        for side in SIDES:
            if r[side]['clip_blocks'] or not r[side]['finite']:
                problems.append(f"limits {r['mode']} rho {r['rho']} {side}: clip/non-finite")
            if r[side]['resets']:
                # node resets of the Gutter side (A) -- reachable only by the every-block
                # full-field toggle (a B stress probe, 125 Hz retune of every bank); the
                # plain N1 gutter_field resets identically there, hand / CA edits do not
                (problems if r['mode'] != 'full_toggle_every_block' else observations).append(
                    f"limits {r['mode']} rho {r['rho']} {side}: {r[side]['resets']} node resets")
    rep['observations'] = observations
    for c in rep.get('catalog', []):
        if c.get('status') != 'match':
            problems.append(f"catalog {c.get('rid')}: {c.get('status')} {c.get('reason')}")
    rep['problems'] = problems
    rep['summary'] = 'no clipping, no resets, no NaN, all records replay exactly' if not problems else \
        '; '.join(problems)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    write_markdown(rep, OUT_DIR / 'report.md')
    print('summary:', rep['summary'])
    print('written:', OUT_DIR / 'report.json', OUT_DIR / 'report.md')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
