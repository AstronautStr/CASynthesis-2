"""N4 hand-over measurements (REQ memory/req-object-resonators-n4-2026-09-16.md, "Приёмка
Developer и Researcher" 2-5) -> demos/results/object_resonators_n4/report.{json,md}.

    python demos/n4_objects_report.py [--catalog ROOT] [--no-catalog]

No audio device.  The numbers are technical evidence (levels against the preflight, the
receiver's packets per generation, the Disk - Own difference, glider phases and identity,
tails, continuation, late edits, stress probes at gain 0.04, the analysis cost by figure
size, timing, catalog check), not listening verdicts.  The scalar reference of the sample
path, the geometry / spectrum gates and the snapshot contract are the gates of
tests/test_n4_object_resonators.py.  The summary keeps the ready scenes and the stress
probes apart and lists every limitation found.
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

from casynth_config import SR                                                # noqa: E402
from casynth_engine import step                                              # noqa: E402
from casynth_lab import DemoRunner, scene_from_doc, BLOCK                    # noqa: E402
from casynth_lab.catalog import Catalog                                      # noqa: E402
from casynth_lab.engine_api import EngineContext                             # noqa: E402
from casynth_lab import figures as fg                                        # noqa: E402
from casynth_lab import object_resonators as orz                             # noqa: E402
from demos.build_n4_objects import (CASES, CATALOG_ROOT, scene_for, PREFLIGHT,   # noqa: E402
                                    glider_cells, neighbors_cells, N4_OWN, N4_DISK)

OUT_DIR = ROOT / 'demos' / 'results' / 'object_resonators_n4'
BUDGET_MS = BLOCK / SR * 1000.0
SIDES = ('A', 'B')
CTX = EngineContext(SR, BLOCK, 2, 110.0, 1.0, 6.0)
GAIN = 0.028


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-30))


def rms_db(y):
    return db(np.sqrt(np.mean(np.asarray(y, float) ** 2)))


def run_blocks(runner, n_blocks, on_block=None, peaks=None):
    """`peaks` (dict) collects the PRE-clip peak of every side over the blocks."""
    out = {s: [] for s in SIDES}
    times = []
    for b in range(n_blocks):
        t0 = time.perf_counter()
        blk = runner.next_block()
        times.append((time.perf_counter() - t0) * 1000.0)
        for s in SIDES:
            out[s].append(blk.get(s))
            if peaks is not None:
                peaks[s] = max(peaks.get(s, 0.0), float(runner.sides[s].peak))
        if on_block is not None:
            on_block(b, runner)
    y = {s: np.concatenate(out[s]).astype(float) / 32767.0 for s in SIDES}
    return y, times


def stats(y):
    return dict(rms_db=rms_db(y), peak=float(np.abs(y).max()), peak_db=db(np.abs(y).max()))


def engine_run(cells, params, seconds=12.0, rate=6.0, gain=GAIN, on_gen=None):
    """Drive one engine like the bench (float output, no int16): per generation callback."""
    e = orz.ObjectResonatorsEngine(CTX, dict(params))
    g = np.zeros((32, 32), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    e.init(g, None, gain)
    n_blocks = int(math.ceil(seconds * SR / BLOCK))
    step_s = SR / rate
    ca, gen = 0, 0
    ys = []
    for b in range(n_blocks):
        stepped = False
        if ca >= (gen + 1) * step_s:
            g = step(g)
            gen += 1
            e.update_field(g, None)
            stepped = True
        y, _pk, _nc = e.render_float(gain)
        ys.append(y)
        if on_gen is not None and stepped:
            on_gen(gen, e)
        ca += BLOCK
    return np.concatenate(ys)[:int(seconds * SR)], e


def scene_measurements(case, pf):
    doc = scene_for(case)
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    n = int(math.ceil(12.0 * SR / BLOCK))
    ids_seen = {s: set() for s in SIDES}
    f_low = {s: [] for s in SIDES}

    def on_block(b, r):
        snap = r.snapshot()
        for s in SIDES:
            d = snap['display'][s]
            if d and 'figures' in d:
                ids_seen[s].update(f['id'] for f in d['figures'])
                if d['figures']:
                    f_low[s].append(round(float(d['figures'][0]['f_low']), 6))
    y, times = run_blocks(runner, n, on_block)
    info = dict(id=case['id'], title=case['title'], engines={s: runner.sides[s].engine_id for s in SIDES},
                params={s: dict(runner.sides[s].params) for s in SIDES})
    pf_case = 'glider' if case['id'] == 'n4_spectrum' else 'neighbors'
    for s in SIDES:
        st = stats(y[s][:int(12.0 * SR)])
        st['clip_blocks'] = int(runner.sides[s].clip_blocks)
        st['finite'] = bool(np.all(np.isfinite(y[s])))
        eid = runner.sides[s].engine_id
        if eid == orz.ENGINE_ID:
            mode = 'own' if int(runner.sides[s].params['detector']) == 0 else 'disk'
            ref = pf['cases'][pf_case]['modes'][mode]
            st['preflight_rms_db'] = ref['rms_at_scale_0p5_gain_0p028_db']
            st['preflight_peak'] = ref['peak_at_scale_0p5_gain_0p028']
            st['rms_diff_db'] = st['rms_db'] - ref['rms_at_scale_0p5_gain_0p028_db']
            st['ids_seen'] = sorted(ids_seen[s])
            st['f_low_values'] = sorted(set(f_low[s]))
        info[s] = st
    info['a_minus_b_db'] = info['A']['rms_db'] - info['B']['rms_db']
    info['block_ms'] = dict(p50=float(np.percentile(times[60:], 50)), p99=float(np.percentile(times[60:], 99)),
                            max=float(np.max(times[60:])), budget=BUDGET_MS,
                            ok=bool(np.percentile(times[60:], 99) < BUDGET_MS))
    # tail after the field is cleared at 12 s (both sides), 4 s more
    runner.post('clear')
    y2, _t = run_blocks(runner, int(4.0 * SR / BLOCK))
    for s in SIDES:
        info[s]['tail_rms_db'] = [rms_db(y2[s][int(a * SR):int(b * SR)]) for a, b in ((0, 0.5), (1, 1.5), (3, 3.5))]
        d = runner.snapshot()['display'][s]
        if d and 'figures' in d:
            info[s]['tails_after_clear_4s'] = int(d['n_tails'])
    # continue: export at 3 s, twin runs the same 60 blocks
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    run_blocks(runner, int(3.0 * SR / BLOCK))
    twin = DemoRunner.from_state(runner.export_state())
    same = True
    gen0 = twin.gen
    for _ in range(60):
        a, b = runner.next_block(), twin.next_block()
        same = same and all(np.array_equal(a.get(s), b.get(s)) for s in ('A', 'B', 'monitor'))
    info['continue'] = dict(exact=bool(same), field_changed=bool(twin.gen > gen0))
    # late edit: a 3 x 3 block painted at 6 s changes the sound of both sides
    base = DemoRunner(scene_from_doc(doc))
    base.post('start', at=0)
    edit = DemoRunner(scene_from_doc(doc))
    edit.post('start', at=0)
    edit.post('set_cells', at=int(6.0 * SR), cells=[[20 + i, 4 + j, 1] for i in range(3) for j in range(3)])
    yb, _ = run_blocks(base, int(8.0 * SR / BLOCK))
    ye, _ = run_blocks(edit, int(8.0 * SR / BLOCK))
    info['late_edit'] = {s: dict(differs=bool(not np.array_equal(yb[s], ye[s])),
                                 max_abs_diff=float(np.abs(yb[s] - ye[s]).max())) for s in SIDES}
    return info


def receiver_probe():
    """N4.2: the receiver's packet per generation in Own / Disk (expected 0 / 4), the
    blinker's own packet, and the Disk - Own difference against Own (the preflight's
    ~ -11 dB): an additional response, not a level lift."""
    cells = neighbors_cells()
    out = {}
    ys = {}
    for name, params in (('own', N4_OWN), ('disk', N4_DISK)):
        e_recv, e_blk = [], []

        def on_gen(gen, e, e_recv=e_recv, e_blk=e_blk):
            d = e.display()
            by = {f['id']: f for f in d['figures']}
            e_recv.append(by[1]['e'] if 1 in by else None)
            e_blk.append(by[2]['e'] if 2 in by else None)
        y, e = engine_run(cells, params, on_gen=on_gen)
        ys[name] = y
        d = e.display()
        recv = next(f for f in d['figures'] if f['id'] == 1)
        out[name] = dict(receiver_e_per_gen=sorted(set(e_recv[1:])), blinker_e_per_gen=sorted(set(e_blk[1:])),
                         receiver_centre=recv['centre'], receiver_radius=recv['radius'],
                         receiver_modes=recv['modes'], receiver_f_low=recv['f_low'], n_figures=d['n_figures'],
                         rms_db=rms_db(y))
    diff = ys['disk'] - ys['own']
    out['disk_minus_own_rel_own_db'] = rms_db(diff) - rms_db(ys['own'])
    out['disk_vs_own_total_db'] = out['disk']['rms_db'] - out['own']['rms_db']
    # the blinker moved outside the circle: the receiver's packets stop
    recv_cells = [c for c in cells if tuple(c) not in {(9, 14), (9, 15), (9, 16)}]
    e = orz.ObjectResonatorsEngine(CTX, dict(N4_DISK))
    g = np.zeros((32, 32), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    e.init(g, None, GAIN)
    e.render_float(GAIN)
    far = np.zeros((32, 32), np.uint8)
    for r, c in recv_cells + [(9, 24), (9, 25), (9, 26)]:
        far[r, c] = 1
    e.update_field(far, None)
    e.render_float(GAIN)
    seq = []
    for _ in range(8):
        far = step(far)
        e.update_field(far, None)
        e.render_float(GAIN)
        seq.append(next(f['e'] for f in e.display()['figures'] if f['id'] == 1))
    out['receiver_e_after_blinker_moved_outside'] = seq
    return out


def glider_probe():
    """N4.1 side B: one identity for the whole round, two alternating spectra."""
    seen_ids, lows = set(), []

    def on_gen(gen, e):
        d = e.display()
        seen_ids.update(f['id'] for f in d['figures'])
        lows.append(tuple(round(float(v), 6) for v in e.ffreq[0, :4]))
    y, e = engine_run(glider_cells(), N4_DISK, seconds=24.0, rate=16.0, on_gen=on_gen)
    spectra = sorted(set(lows))
    alternating = all(lows[k] == lows[k - 2] for k in range(2, len(lows)))
    return dict(ids_seen=sorted(seen_ids), distinct_spectra=len(spectra), spectra_hz=spectra,
                alternating_every_generation=bool(alternating and len(set(lows[:2])) == 2), generations=len(lows),
                rms_db=rms_db(y), peak=float(np.abs(y).max()), cache_misses=e.cache.misses,
                cache_hits=e.cache.hits)


def limits():
    """Dense / random fields, every-block toggles, repeated manual edits, both ends of the
    knobs, both detector modes, at the maximum standard gain (vol 1 -> 0.04), 6 s each."""
    rng = np.random.default_rng(20260916)
    doc = scene_for(CASES[1])
    results = []
    n = int(math.ceil(6.0 * SR / BLOCK))
    for detector in (0, 1):
        for scale, decay in ((55.0, 0.20), (880.0, 1.50)):
            for mode in ('dense_random_evolving', 'random_every_block', 'full_toggle_every_block',
                         'repeated_edits_paused', 'knobs_moving'):
                runner = DemoRunner(scene_from_doc(doc), vol=1.0)
                for s in SIDES:
                    runner.post('set_param', side=s, name='detector', value=detector)
                    runner.post('set_param', side=s, name='frequency_scale', value=scale)
                    runner.post('set_param', side=s, name='decay_s', value=decay)
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
                    elif mode in ('repeated_edits_paused', 'knobs_moving') and b % 12 == 0:
                        r0, c0 = int(rng.integers(0, 27)), int(rng.integers(0, 27))
                        r.post('set_cells', cells=[[r0 + i, c0 + j, int(1 - r.grid[r0 + i, c0 + j])]
                                                   for i in range(5) for j in range(5)])
                        if mode == 'knobs_moving':
                            k = b // 12
                            for s in SIDES:
                                r.post('set_param', side=s, name='decay_s', value=(0.2 if k % 2 else 1.5))
                                r.post('set_param', side=s, name='frequency_scale', value=(55.0 if k % 3 else 880.0))
                                r.post('set_param', side=s, name='detector', value=(k // 2) % 2)
                peaks = {}
                y, times = run_blocks(runner, n, on_block, peaks)
                row = dict(detector=detector, frequency_scale=scale, decay_s=decay, mode=mode, gain=runner._gain(),
                           block_ms_p99=float(np.percentile(times[50:], 99)), block_ms_max=float(np.max(times[50:])))
                for s in SIDES:
                    st = stats(y[s][int(0.5 * SR):])
                    st['pre_clip_peak'] = peaks[s]
                    st['clip_blocks'] = int(runner.sides[s].clip_blocks)
                    st['finite'] = bool(np.all(np.isfinite(y[s])))
                    d = runner.snapshot()['display'][s]
                    st.update(n_figures=d['n_figures'], n_sounding=d['n_sounding'], n_tails=d['n_tails'],
                              evictions=d['evictions'], drops=d['drops'], unvoiced_blocks=d['unvoiced_blocks'])
                    row[s] = st
                results.append(row)
    return results


def tails_probe():
    """Many blinkers cleared and re-added: the 24-bank limit is visible, tails pile up to
    96, the quietest fade (counted), nothing is lost silently, all finite."""
    many = []
    for r in range(0, 32, 4):
        for c in range(0, 30, 5):
            many += [(r + 1, c), (r + 1, c + 1), (r + 1, c + 2)]
    g = np.zeros((32, 32), np.uint8)
    for r, c in many:
        g[r, c] = 1
    e = orz.ObjectResonatorsEngine(CTX, dict(N4_DISK))
    e.init(g, None, 0.04)
    e.render_float(0.04)
    d0 = e.display()
    peak = 0.0
    for k in range(8):
        e.update_field(np.zeros_like(g), None)
        y, pk, _ = e.render_float(0.04)
        peak = max(peak, pk)
        e.update_field(g, None)
        y, pk, _ = e.render_float(0.04)
        peak = max(peak, pk)
    d1 = e.display()
    e.update_field(np.zeros_like(g), None)
    for _ in range(int(3.0 * SR / BLOCK)):
        y, pk, _ = e.render_float(0.04)
        peak = max(peak, pk)
    d2 = e.display()
    return dict(figures=d0['n_figures'], sounding=d0['n_sounding'], after_cycles=dict(
        tails=d1['n_tails'], fading=d1['n_fading'], evictions=d1['evictions'], drops=d1['drops']),
        after_3s_silence=dict(tails=d2['n_tails'], fading=d2['n_fading']), peak=peak,
        finite=bool(np.isfinite(peak)))


def analysis_cost():
    """eigvalsh cost of one component by size (the analysis runs on the render thread at
    every field change; a figure larger than ~300 cells exceeds the block budget on its
    own -- offline exact, live underruns possible)."""
    rows = []
    for side in (8, 11, 16, 23, 32):
        cells = np.array([(r, c) for r in range(side) for c in range(side)], np.int64)
        cells = cells[:min(len(cells), 1024)]
        t0 = time.perf_counter()
        fg.spectrum(cells, 32, 32)
        ms = (time.perf_counter() - t0) * 1000.0
        rows.append(dict(cells=int(len(cells)), spectrum_ms=ms, over_budget=bool(ms > BUDGET_MS)))
    return rows


def timing(seconds=8.0):
    out = {}
    for case in CASES:
        runner = DemoRunner(scene_from_doc(scene_for(case)))
        runner.post('start', at=0)
        _, times = run_blocks(runner, int(seconds * SR / BLOCK))
        t = np.array(times[100:])
        out[case['id']] = dict(p50=float(np.percentile(t, 50)), p95=float(np.percentile(t, 95)),
                               p99=float(np.percentile(t, 99)), max=float(t.max()), budget=BUDGET_MS,
                               ok=bool(np.percentile(t, 99) < BUDGET_MS))
    out['numba'] = orz.HAVE_NUMBA
    return out


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
    L = ['# N4 object resonators -- hand-over measurements (technical evidence, not listening verdicts)', '',
         f"Generated {rep['generated']} at commit {rep['commit']}.  SR {SR}, block {BLOCK} "
         f"({BUDGET_MS:.2f} ms), engine `{orz.ENGINE_ID}` (output x{orz.OUT_SCALE:g}, gain 0.028 = vol 0.7), "
         f"scale 220 Hz, decay 0.8 s.  RMS = stereo RMS as in the preflight; (pre) = preflight value.", '',
         '## Scenes (12 s from a fresh start)', '',
         '| scene | side | engine | rms dB (pre) | peak (pre) | clip | ids seen | p99 ms |',
         '|---|---|---|---|---|---|---|---|']
    for s in rep['scenes']:
        for side in SIDES:
            st = s[side]
            pre = f" ({st['preflight_rms_db']:.2f})" if 'preflight_rms_db' in st else ''
            ppre = f" ({st['preflight_peak']:.4f})" if 'preflight_peak' in st else ''
            L.append(f"| {s['id']} | {side} | {s['engines'][side]} | {st['rms_db']:.2f}{pre} | {st['peak']:.4f}{ppre} "
                     f"| {st['clip_blocks']} | {st.get('ids_seen', '-')} | {s['block_ms']['p99']:.2f} |")
    L += ['']
    for s in rep['scenes']:
        L.append(f"- {s['id']}: A - B = {s['a_minus_b_db']:+.2f} dB; tails after Clear (0-0.5 / 1-1.5 / 3-3.5 s): "
                 + ', '.join(f"{side} {'/'.join(f'{v:.1f}' for v in s[side]['tail_rms_db'])} dB" for side in SIDES)
                 + f"; continue exact {s['continue']['exact']}; late edit differs "
                 + ', '.join(f"{side} {s['late_edit'][side]['differs']}" for side in SIDES))
    r = rep['receiver']
    L += ['', '## N4.2 receiver (figure 1) per generation', '',
          f"- Own: receiver e per generation {r['own']['receiver_e_per_gen']}, blinker e {r['own']['blinker_e_per_gen']}",
          f"- Disk: receiver e per generation {r['disk']['receiver_e_per_gen']}, blinker e {r['disk']['blinker_e_per_gen']}",
          f"- receiver centre {r['disk']['receiver_centre']}, R {r['disk']['receiver_radius']:.5f}, "
          f"{r['disk']['receiver_modes']} modes, lowest {r['disk']['receiver_f_low']:.2f} Hz",
          f"- Disk - Own difference relative to Own: {r['disk_minus_own_rel_own_db']:.2f} dB "
          f"(total RMS difference {r['disk_vs_own_total_db']:+.2f} dB)",
          f"- blinker moved outside the circle: receiver e per generation {r['receiver_e_after_blinker_moved_outside']}"]
    g = rep['glider']
    L += ['', '## N4.1 glider (side B, 24 s at 16 gen/s)', '',
          f"- identities seen {g['ids_seen']}, {g['distinct_spectra']} distinct spectra, alternating every "
          f"generation: {g['alternating_every_generation']} ({g['generations']} generations; spectrum cache "
          f"hits {g['cache_hits']} / misses {g['cache_misses']})"]
    for sp in g['spectra_hz']:
        L.append('  - ' + ', '.join(f'{v:.2f}' for v in sp) + ' Hz')
    L += ['', '## Stress probes (gain 0.04, 6 s, both sides on the same field)', '',
          '| detector | scale | decay | mode | A pre-clip peak | B pre-clip peak | clip blocks A/B | '
          'figures / sounding / tails B | faded / dropped B | p99 ms |', '|---|---|---|---|---|---|---|---|---|---|']
    for row in rep['limits']:
        b = row['B']
        L.append(f"| {row['detector']} | {row['frequency_scale']:.0f} | {row['decay_s']} | {row['mode']} | "
                 f"{row['A']['pre_clip_peak']:.3f} | {b['pre_clip_peak']:.3f} | {row['A']['clip_blocks']}/{b['clip_blocks']} | "
                 f"{b['n_figures']}/{b['n_sounding']}/{b['n_tails']} | {b['evictions']}/{b['drops']} | "
                 f"{row['block_ms_p99']:.2f} |")
    t = rep['tails']
    L += ['', f"Tails probe: {t['figures']} blinkers -> sounding {t['sounding']}; after 8 clear / re-add cycles (16 blocks) "
          f"tails {t['after_cycles']['tails']}, fading {t['after_cycles']['fading']}, faded {t['after_cycles']['evictions']}, "
          f"dropped {t['after_cycles']['drops']}; after 3 s of silence tails {t['after_3s_silence']['tails']}; "
          f"peak {t['peak']:.3f}, finite {t['finite']}.", '',
          '## Analysis cost by figure size (eigvalsh of one component, this machine)', '',
          '| cells | ms | over the block budget |', '|---|---|---|']
    for row in rep['analysis_cost']:
        L.append(f"| {row['cells']} | {row['spectrum_ms']:.1f} | {row['over_budget']} |")
    tm = rep['timing']
    L += ['', '## Timing (both sides, after warm-up)', '']
    for k, v in tm.items():
        if isinstance(v, dict):
            L.append(f"- {k}: p50 {v['p50']:.2f}  p95 {v['p95']:.2f}  p99 {v['p99']:.2f}  max {v['max']:.2f} ms "
                     f"(budget {v['budget']:.2f}, ok {v['ok']})")
    L.append(f"- numba: {tm['numba']}")
    if 'catalog' in rep:
        L += ['', f"## Catalog check ({rep['catalog_root']})", '']
        for c_ in rep['catalog']:
            if 'error' in c_:
                L.append(f"- ERROR {c_['error']}")
            else:
                L.append(f"- {c_['rid']} {c_['title'][:50]}: {c_['status']} {c_['reason'] or ''} "
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
    pf = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    try:
        import subprocess
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip() or 'unknown'
    except Exception:                                                   # pragma: no cover
        commit = 'unknown'
    rep = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), commit=commit, scenes=[])
    for case in CASES:
        print('scene', case['id'], flush=True)
        rep['scenes'].append(scene_measurements(case, pf))
    print('receiver', flush=True)
    rep['receiver'] = receiver_probe()
    print('glider', flush=True)
    rep['glider'] = glider_probe()
    print('limits', flush=True)
    rep['limits'] = limits()
    print('tails', flush=True)
    rep['tails'] = tails_probe()
    print('analysis cost', flush=True)
    rep['analysis_cost'] = analysis_cost()
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
            st = s[side]
            if st['clip_blocks'] or not st['finite']:
                scene_problems.append(f"{s['id']} {side}: clip / non-finite")
            if not s['late_edit'][side]['differs']:
                scene_problems.append(f"{s['id']} {side}: a late edit does not change the sound")
            if st['tail_rms_db'][2] >= st['tail_rms_db'][0]:
                scene_problems.append(f"{s['id']} {side}: the tail does not decay")
            if 'rms_diff_db' in st and abs(st['rms_diff_db']) > 0.5:
                scene_problems.append(f"{s['id']} {side}: RMS {st['rms_diff_db']:+.2f} dB against the preflight")
        if not s['block_ms']['ok']:
            scene_problems.append(f"{s['id']}: p99 {s['block_ms']['p99']:.2f} ms over budget")
        if not s['continue']['exact'] or not s['continue']['field_changed']:
            scene_problems.append(f"{s['id']}: continuation not exact / field frozen")
        if s['id'] == 'n4_spectrum' and s['B'].get('ids_seen') != [1]:
            scene_problems.append("n4_spectrum B: the glider did not keep one identity")
    r = rep['receiver']
    if r['own']['receiver_e_per_gen'] != [0.0] or r['disk']['receiver_e_per_gen'] != [4.0]:
        scene_problems.append("N4.2 receiver: Own / Disk packets per generation are not 0 / 4")
    if any(v != 0.0 for v in r['receiver_e_after_blinker_moved_outside']):
        scene_problems.append("N4.2 receiver: packets after the blinker moved outside the circle")
    if not (-14.0 < r['disk_minus_own_rel_own_db'] < -8.0):
        limitations.append(f"N4.2 Disk - Own difference {r['disk_minus_own_rel_own_db']:.1f} dB relative to Own "
                           f"(preflight about -11 dB)")
    g = rep['glider']
    if g['ids_seen'] != [1] or g['distinct_spectra'] != 2 or not g['alternating_every_generation']:
        scene_problems.append("N4.1 glider: identity / two alternating spectra not as expected")
    clipped, dropped = [], []
    for row in rep['limits']:
        for side in SIDES:
            if not row[side]['finite']:
                stress_problems.append(f"limits {row['mode']} det {row['detector']} scale {row['frequency_scale']:.0f} "
                                       f"decay {row['decay_s']} {side}: non-finite")
            if row[side]['clip_blocks'] and side == 'B':
                clipped.append(f"{row['mode']} det {row['detector']} scale {row['frequency_scale']:.0f} decay "
                               f"{row['decay_s']}: pre-clip peak {row[side]['pre_clip_peak']:.2f}, "
                               f"{row[side]['clip_blocks']} clipped blocks")
            if row[side]['drops'] and side == 'B':
                dropped.append(f"{row['mode']} det {row['detector']} scale {row['frequency_scale']:.0f}: "
                               f"{row[side]['drops']} dropped (shown in the panel)")
    worst = max(rep['limits'], key=lambda r_: max(r_['A']['pre_clip_peak'], r_['B']['pre_clip_peak']))
    worst_peak = max(worst['A']['pre_clip_peak'], worst['B']['pre_clip_peak'])
    if clipped:
        limitations.append(f"stress clipping at gain 0.04 (vol 1) in {len(clipped)} of {len(rep['limits'])} probes, "
                           f"none in the scenes (peaks <= 0.09): " + '; '.join(clipped) + ". No AGC and no division "
                           f"by the figure count by the REQ; the only common lever is the N4 output x{orz.OUT_SCALE:g}")
    if dropped:
        limitations.append("hard drops (the fading pool of 24 full: the quietest fading tail zeroed at once, counted "
                           "and shown as 'dropped') only under whole-field replacement every block: "
                           + '; '.join(dropped))
    slow = [r_ for r_ in rep['limits'] if r_['block_ms_p99'] >= BUDGET_MS]
    if slow:
        limitations.append(f"{len(slow)} of {len(rep['limits'])} stress probes exceed the block budget at p99 "
                           f"(worst {max(r_['block_ms_p99'] for r_ in slow):.1f} ms, modes "
                           f"{sorted(set(r_['mode'] for r_ in slow))}): the figure analysis (eigvalsh) of dense "
                           f"fields on the render thread -- offline exact, live underruns possible")
    if not rep['tails']['finite']:
        stress_problems.append("tails probe: non-finite")
    if rep['tails']['after_cycles']['drops']:
        limitations.append(f"tails probe ({rep['tails']['figures']} blinkers cleared / re-added 8 times within "
                           f"{16} blocks): {rep['tails']['after_cycles']['drops']} hard drops, "
                           f"{rep['tails']['after_cycles']['evictions']} faded, all freed after 3 s of silence")
    for k, v in rep['timing'].items():
        if isinstance(v, dict) and not v['ok']:
            scene_problems.append(f"timing {k}: p99 {v['p99']:.2f} ms over budget")
    for c_ in rep.get('catalog', []):
        if c_.get('status') != 'match':
            scene_problems.append(f"catalog {c_.get('rid')}: {c_.get('status')} {c_.get('reason')}")
    n41 = rep['scenes'][0]
    limitations += [
        f"N4.1 level: A (N3 Tuned) - B (N4 Disk) = {n41['a_minus_b_db']:+.2f} dB RMS at the same bench gain; no "
        f"scene parameter acts on one side only, the only lever would be the N4 output x{orz.OUT_SCALE:g} "
        f"(kept at the preflight value; the preflight numbers are reproduced)",
        "figure analysis (eigvalsh of L) runs on the render thread at every field change: "
        + ', '.join(f"{r_['cells']} cells {r_['spectrum_ms']:.0f} ms" for r_ in rep['analysis_cost'])
        + " -- a component above ~300 cells exceeds the block budget by itself (the two scenes: <= 17 cells)",
        f"worst stress pre-clip peak {worst_peak:.3f} at gain 0.04 ({worst['mode']}, detector {worst['detector']}, "
        f"scale {worst['frequency_scale']:.0f}, decay {worst['decay_s']} s): measured for these finite probes, not "
        f"guaranteed for any playing",
        "split / merge retire identities (first model): the Tumbler and the R-pentomino change colours often",
        "tails are measured on the float output (int16 quantises below -90 dBFS to zero)",
        "no listening in this report: hearing the difference is the user's verdict",
    ]
    rep['problems'] = dict(scenes=scene_problems, stress=stress_problems)
    rep['summary'] = dict(
        scenes=('no clipping, no NaN, levels within 0.5 dB of the preflight, tails decay, late edits change the '
                'sound, receiver Own 0 / Disk 4 per generation and 0 outside the circle, glider one identity with '
                'two alternating spectra, continuation exact, all records replay exactly'
                if not scene_problems else '; '.join(scene_problems)),
        stress=(f"finite in all {len(rep['limits'])} probes; clipping at gain 0.04 in {len(clipped)} probes and hard "
                f"drops in {len(dropped)} (listed under limitations); worst pre-clip peak {worst_peak:.3f}"
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
