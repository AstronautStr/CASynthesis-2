"""Measurements for the Objects "Birth strength" delivery (REQ
memory/req-objects-event-source-modal-2026-09-17.md section 5, "Приёмка дополнения"):

  1. the law: the profiles of every M1 phase bank and synthetic case at s = 0 / 0.5 /
     1 / 2 / 4 against the independent preflight of 2026-09-18, the REQ control
     example (16 cells), the norm, zeros, equal coefficients, one mode
  2. five engines on the M1 field side by side (s = 0 / 0.5 / 1 / 2 / 4, 72
     transitions): frequencies, output weights, packet moments, a and the tracker
     equal; the per-packet coefficients of every phase against the preflight
  3. the PCM path: Birth position at s = 0 equals Uniform bit for bit (whole scene
     render); Uniform at s = 4 equals Uniform at s = 1; the stored WAVs of the
     pinned 2026-09-17 records (E1, M1) equal the current code with the default s
     (s = 1 is the delivered Birth position); s changed on a still field: no packet,
     the tail unchanged (through 0 and 1); a sequence of s across transitions
     (per-mode history kept at s = 0), a runner journal with knob changes continued
     exactly from a snapshot; a v4 snapshot imported as s = 1
  4. levels: RMS / peaks / clip / finiteness of side B at the five s with unit gain
     and with the delivered side gain; the delivered pair 1 / 4: RMS after 2 s within
     1 dB, Continue from 6 s exact, the block time budget of both sides at every s
     and with the knob moved during the run
  5. the catalog: every record replays exactly (when built)

    python demos/objects_birth_strength_report.py [--root DIR] [--quick]

Writes demos/results/objects_birth_strength/report.json + report.md.  Stdout ASCII.
"""
import argparse
import copy
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
from demos.build_objects_event_source import (preflight_params, case_cells, rms_db, F0_HZ, RATE_HZ, SECONDS,   # noqa: E402
                                              STEADY_FROM_S, CATALOG_ROOT as OES_ROOT)
from demos.build_objects_birth_strength import (CASES, CATALOG_ROOT, PREFLIGHT_STRENGTH, SIDE_GAIN, STRENGTHS,   # noqa: E402
                                                scene_for, levels)
from demos.objects_event_source_report import verify_catalog, run_blocks     # noqa: E402

OUT_DIR = ROOT / 'demos' / 'results' / 'objects_birth_strength'
BUDGET_MS = BLOCK / SR * 1000.0
SIDES = ('A', 'B')
CTX = EngineContext(SR, BLOCK, 2, F0_HZ, 1.0, RATE_HZ)
GAIN = MASTER_GAIN * 0.7
EID = orz.ENGINE_ID
ROWS = COLS = 32
BIRTHS = orz.EV_BIRTHS
UNIFORM, POSITION = orz.EXC_UNIFORM, orz.EXC_POSITION


def preflight_strength():
    with open(PREFLIGHT_STRENGTH, encoding='utf-8') as f:
        return json.load(f)


def params(strength=1.0, ex=POSITION, **over):
    p = preflight_params(BIRTHS, ex)
    p['birth_strength'] = float(strength)
    p.update(over)
    return p


def grid(cells):
    g = np.zeros((ROWS, COLS), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def make_engine(cells, p):
    e = registry.create(EID, CTX, p)
    e.init(grid(cells), None, GAIN)
    return e


def run(e, n):
    return np.concatenate([e.render_float(GAIN)[0] for _ in range(n)])


def gens_of(cells, n):
    g = grid(cells)
    out = [g]
    for _ in range(n):
        g = step(g)
        out.append(g)
    return out


# -- 1. the law ----------------------------------------------------------------------------------
def law_checks():
    pf = preflight_strength()
    out = dict(law=pf['law'], profiles=[], max_diff=0.0, max_norm_error=0.0)
    for case in pf['cases'] + pf['synthetic']:
        b = np.asarray(case.get('weights_spatial', case.get('b')), np.float64)
        label = (f"gen{case['generation']}_{case['cells']}c_{case['births']}b" if 'generation' in case else case['id'])
        row = dict(id=label, b=[round(float(x), 6) for x in b], strengths={})
        for prof in case['profiles']:
            s = float(prof['strength'])
            if s == 0.0:
                c = np.ones(len(b)) if int(case.get('births', 1)) > 0 else np.zeros(len(b))   # the Uniform path
            elif not b.any():
                c = np.zeros(len(b))                                                            # no packet
            else:
                c = orz.birth_strength_profile(b, s)
            d = float(np.max(np.abs(c - np.asarray(prof['b'])))) if len(b) else 0.0
            out['max_diff'] = max(out['max_diff'], d)
            if c.any():
                out['max_norm_error'] = max(out['max_norm_error'], abs(float((c * c).sum()) - len(b)))
            row['strengths'][f"{s:g}"] = dict(c=[round(float(x), 6) for x in c], max_diff=d)
        out['profiles'].append(row)
    b16 = np.asarray([0.7814996203212149, 0.7037460155649663, 1.3762266851846143])
    out['control_16_cells'] = {f"{s:g}": [round(float(x), 6) for x in
                                           (np.ones(3) if s == 0 else orz.birth_strength_profile(b16, s))]
                               for s in STRENGTHS}
    zero = np.asarray([0.0, 1.0, math.sqrt(2.0)])
    out['exact_zero'] = {f"{s:g}": float(orz.birth_strength_profile(zero, s)[0]) for s in (0.5, 1.0, 2.0, 4.0)}
    out['equal_stay_equal'] = all(bool(np.allclose(orz.birth_strength_profile(np.ones(3), s), 1.0, atol=1e-15))
                                  for s in (0.5, 2.0, 4.0))
    out['one_mode'] = all(float(orz.birth_strength_profile(np.ones(1), s)[0]) == 1.0 for s in (0.5, 2.0, 4.0))
    out['s1_is_b_itself'] = bool(orz.birth_strength_profile(b16, 1.0) is b16)
    out['all_nonnegative'] = all(bool((orz.birth_strength_profile(b16, s) >= 0).all()) for s in (0.5, 2.0, 4.0))
    return out


# -- 2. five engines on M1 ---------------------------------------------------------------------------
def m1_five_strengths(transitions=72):
    cells = case_cells('M1')
    pf = preflight_strength()
    engines = {s: make_engine(cells, params(s)) for s in STRENGTHS}
    ref = engines[1.0]
    equal = dict(frequencies=True, weights=True, packet_moments=True, a=True, tracker=True)
    seen = {s: {} for s in STRENGTHS}
    packets = 0
    max_norm = 0.0
    g = grid(cells)
    per_gen = int(round(SR / RATE_HZ / BLOCK))
    for t in range(transitions + 1):
        if t > 0:
            prev, g = g, step(g)
            for e in engines.values():
                e.update_field(g, events_field(prev, g))
        for _ in range(per_gen):
            for e in engines.values():
                e.render_float(GAIN)
            for e in engines.values():
                if sorted(e.figures) != sorted(ref.figures):
                    equal['tracker'] = False
                    continue
                for fid, fr in ref.figures.items():
                    fe = e.figures[fid]
                    if fr.slot != fe.slot:
                        equal['tracker'] = False
                    if fr.slot < 0 or fe.slot < 0:
                        continue
                    equal['frequencies'] &= bool(np.array_equal(ref.ffreq[fr.slot], e.ffreq[fe.slot]))
                    equal['weights'] &= bool(np.array_equal(ref.wtgt[fr.slot], e.wtgt[fe.slot]))
                    equal['packet_moments'] &= float(ref.last_e[fr.slot]) == float(e.last_e[fe.slot])
                    equal['a'] &= float(ref.last_a[fr.slot]) == float(e.last_a[fe.slot])
        for s, e in engines.items():
            for f in e.display()['figures']:
                if f['slot'] < 0 or f['e'] <= 0.0:
                    continue
                c = np.asarray(f['b'])
                max_norm = max(max_norm, abs(float((c * c).sum()) - f['modes']))
                if s == 1.0:
                    packets += 1
                seen[s].setdefault(((t - 1) % 3 + 1, f['n'], int(f['e'])), c)
    phases = []
    for case in pf['cases']:
        if int(case['births']) <= 0:
            continue
        key = (int(case['generation']), int(case['cells']), int(case['births']))
        row = dict(phase=f"gen{key[0]}_{key[1]}c_{key[2]}b", frequencies_hz=[round(x, 1) for x in case['frequencies_hz']],
                   strengths={})
        for prof in case['profiles']:
            s = float(prof['strength'])
            got = seen[s].get(key)
            row['strengths'][f"{s:g}"] = dict(measured=(None if got is None else [round(float(x), 4) for x in got]),
                                             preflight=[round(float(x), 4) for x in prof['b']],
                                             max_diff=(None if got is None else float(np.max(np.abs(got - np.asarray(prof['b']))))))
        phases.append(row)
    return dict(transitions=transitions, equal=equal, packets_with_events=packets, max_norm_error=max_norm,
                phases=phases,
                states={f"{s:g}": dict(per_mode_max=float(np.abs(e.zfm).max()), uniform_max=float(np.abs(e.zf).max()),
                                       zero_participation=int(e.display()['zero_participation']),
                                       refused=int(e.display()['unsupported_packets']))
                        for s, e in engines.items()})


# -- 3. the PCM path ------------------------------------------------------------------------------------
def render_doc(doc, seconds=SECONDS):
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    y, _t = run_blocks(runner, int(math.ceil(seconds * SR / BLOCK)))
    return y


def rendered_like(rec):
    """The record's embedded scene (no birth_strength key -> the default 1 added in
    every parameter set of the document) rendered with the CURRENT code against the
    stored WAVs, bit for bit per output."""
    doc = copy.deepcopy(rec.meta['scene'])
    for group in ('variants', 'factory_variants'):
        for _side, var in (doc.get(group) or {}).items():
            if var.get('engine_id') == EID:
                var['engine_params'].setdefault('birth_strength', 1.0)
    for _side, per_engine in (doc.get('param_memory') or {}).items():
        if EID in per_engine:
            per_engine[EID].setdefault('birth_strength', 1.0)
    cmds = [j for j in rec.meta['journal'] if j['kind'] != 'step']
    if [(j['kind'], j['out_sample']) for j in cmds] != [('start', 0)]:
        return dict(error='journal is not a plain start')
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    ref = {o: rec.pcm(o) for o in ('A', 'B', 'monitor')}
    out = {o: [] for o in ref}
    for _ in range(ref['A'].shape[0] // BLOCK):
        blk = runner.next_block()
        for o in ref:
            out[o].append(np.array(blk.get(o), copy=True))
    return {o: bool(np.array_equal(np.concatenate(out[o]), ref[o])) for o in ref}


def pcm_paths():
    case = CASES[0]
    out = {}
    unit = dict(A=1.0, B=1.0)
    # s = 0 (B) against Uniform (A) on the whole scene; Uniform at any stored s
    doc = scene_for(case, side_gain=unit, strength=dict(A=1.0, B=0.0))
    doc['variants']['A']['engine_params']['excitation'] = UNIFORM
    y = render_doc(doc)
    out['s0_equals_uniform_whole_scene'] = bool(np.array_equal(y['A'], y['B']))
    doc = scene_for(case, side_gain=unit, strength=dict(A=1.0, B=4.0))
    for s in SIDES:
        doc['variants'][s]['engine_params']['excitation'] = UNIFORM
    y = render_doc(doc)
    out['uniform_same_at_s1_and_s4'] = bool(np.array_equal(y['A'], y['B']))
    y = render_doc(scene_for(case, side_gain=unit, strength=dict(A=1.0, B=4.0)))
    out['s1_vs_s4_max_abs_diff'] = int(np.abs(y['A'].astype(np.int32) - y['B'].astype(np.int32)).max())
    # the pinned 2026-09-17 records against the current code (by the user's rule the
    # records themselves are never rebuilt with substituted values: this is a
    # side-by-side comparison of their WAVs, not a catalog replay)
    from casynth_lab.catalog import Catalog
    recs = {}
    if Path(OES_ROOT).is_dir():
        cat = Catalog(str(OES_ROOT))
        for rec, err in cat.list():
            if err is None:
                recs[rec.id] = dict(title=rec.title, commit=rec.commit, outputs=rendered_like(rec))
    out['records_2026_09_17_equal_current_code'] = recs
    # s changed on a still field: no packet, tail unchanged, through 0 and 1
    cells = case_cells('M1')
    gens = gens_of(cells, 2)
    e = make_engine(cells, params(1.0))
    twin = make_engine(cells, params(1.0))
    for x in (e, twin):
        run(x, 3)
        x.update_field(gens[1], events_field(gens[0], gens[1]))
        run(x, 2)
    still = True
    for s in (0.0, 1.0, 4.0, 0.5, 1.0, 0.0, 2.0):
        e.set_params(dict(e.params, birth_strength=s))
        still &= bool(np.array_equal(e.render_float(GAIN)[0], twin.render_float(GAIN)[0]))
    still &= bool(np.array_equal(e.zfm, twin.zfm)) and bool(np.array_equal(e.zf, twin.zf))
    out['still_field_change_no_packet_tail_equal'] = still
    out['still_field_changes_counter_equal'] = e.display()['changes'] == twin.display()['changes']
    # a sequence of s across transitions on E1 (one bank the whole period)
    cells = case_cells('E1')
    gens = gens_of(cells, 6)
    seq = [4.0, 0.0, 0.5, 1.0, 2.0, 0.0]
    e = make_engine(cells, params(seq[0]))
    run(e, 3)
    rows = []
    for k in range(1, len(gens)):
        e.set_params(dict(e.params, birth_strength=seq[k - 1]))
        zfm_before = float(np.abs(e.zfm).max())
        zf_before = float(np.abs(e.zf).max())
        e.update_field(gens[k], events_field(gens[k - 1], gens[k]))
        run(e, 1)
        f = e.display()['figures'][0]
        rows.append(dict(transition=k, s=seq[k - 1], e=float(f['e']), c=[round(float(x), 4) for x in f['b']],
                         per_mode_before=zfm_before, per_mode_after=float(np.abs(e.zfm).max()),
                         uniform_before=zf_before, uniform_after=float(np.abs(e.zf).max())))
    out['sequence_e1'] = rows
    # a runner journal with knob changes: Continue from a snapshot after them is exact
    runner = DemoRunner(scene_from_doc(scene_for(case)))
    runner.post('start', at=0)
    changes = [(1.5, 2.0), (2.5, 0.0), (3.5, 4.0), (4.5, 0.5), (5.5, 1.0)]
    for t, s in changes:
        runner.post('set_param', at=int(t * SR), side='B', name='birth_strength', value=s)
    for _ in range(int(6.0 * SR / BLOCK)):
        runner.next_block()
    st = runner.export_state()
    twin = DemoRunner.from_state(st)
    exact = True
    for _ in range(200):
        a, b = runner.next_block(), twin.next_block()
        for s in ('A', 'B', 'monitor'):
            exact &= bool(np.array_equal(a.get(s), b.get(s)))
    out['journal_knob_changes_continue_exact'] = dict(changes=changes, exact=exact,
                                                      b_strength_at_snapshot=float(twin.snapshot()['sides']['B'][1]['birth_strength']))
    # a v4 snapshot (no key) imports as s = 1: the same sound
    cells = case_cells('M1')
    gens = gens_of(cells, 4)
    e1 = make_engine(cells, params(1.0))
    run(e1, 2)
    e1.update_field(gens[1], events_field(gens[0], gens[1]))
    run(e1, 2)
    st4 = e1.export_state()
    st4['version'] = 4
    st4['model_version'] = orz.COMPATIBLE_STATES[4]
    st4['params'] = {k: v for k, v in st4['params'].items() if k != 'birth_strength'}
    old = registry.create(EID, CTX, dict(e1.params))
    old.init(np.zeros((ROWS, COLS), np.uint8), None, 0.0)
    old.restore_state(e1._grid, e1._exc, st4)
    same = old._birth_strength() == 1.0
    for k in range(2, 4):
        for x in (e1, old):
            x.update_field(gens[k], events_field(gens[k - 1], gens[k]))
        for _ in range(20):
            same &= bool(np.array_equal(e1.render_float(GAIN)[0], old.render_float(GAIN)[0]))
    out['v4_snapshot_imports_as_s1_same_sound'] = bool(same)
    return out


# -- 4. levels and timing ------------------------------------------------------------------------------
def scene_measurements(case, seconds=SECONDS):
    doc = scene_for(case)
    n = int(math.ceil(seconds * SR / BLOCK))
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    peaks = {}
    y, times = run_blocks(runner, n, peaks)
    warm = times[int(1.0 * SR / BLOCK):]
    k0 = int(STEADY_FROM_S * SR)
    res = dict(id=case['id'], side_gain=dict(doc['audio']['side_gain']), strength=dict(case['strength']), seconds=seconds,
               block_ms=dict(p50=float(np.percentile(warm, 50)), p99=float(np.percentile(warm, 99)),
                             max=float(warm.max()), budget=BUDGET_MS, ok=bool(np.percentile(warm, 99) < BUDGET_MS)))
    for s in SIDES:
        res[s] = dict(rms_db=rms_db(y[s]), steady_rms_db=rms_db(y[s][k0:]), peak=peaks[s],
                      clip_blocks=int(runner.sides[s].clip_blocks), finite=bool(np.all(np.isfinite(y[s]))))
    res['a_minus_b_db'] = res['A']['rms_db'] - res['B']['rms_db']
    res['steady_a_minus_b_db'] = res['A']['steady_rms_db'] - res['B']['steady_rms_db']
    res['within_1db'] = bool(abs(res['steady_a_minus_b_db']) <= 1.0)
    doc1 = json.loads(json.dumps(doc))
    doc1['audio']['side_gain'] = dict(A=1.0, B=1.0)
    r1 = DemoRunner(scene_from_doc(doc1))
    r1.post('start', at=0)
    y1, _t = run_blocks(r1, n)
    res['steady_a_minus_b_db_at_unit_side_gain'] = rms_db(y1['A'][k0:]) - rms_db(y1['B'][k0:])
    r2 = DemoRunner(scene_from_doc(doc))
    r2.post('start', at=0)
    for _ in range(n // 2):
        r2.next_block()
    st = r2.export_state()
    twin = DemoRunner.from_state(st)
    exact = True
    gen0 = twin.gen
    for _ in range(200):
        a, b = r2.next_block(), twin.next_block()
        for s in ('A', 'B', 'monitor'):
            exact = exact and bool(np.array_equal(a.get(s), b.get(s)))
    res['continue'] = dict(exact=bool(exact), field_changed=bool(twin.gen > gen0), from_seconds=seconds / 2, blocks=200)
    for s in SIDES:
        d = runner.snapshot()['display'][s]
        res[s].update(figures=d['n_figures'], sounding=d['n_sounding'], tails=d['n_tails'], evictions=d['evictions'],
                      inplace_fades=d['inplace_fades'], drops=d['drops'], excitation=d['excitation_name'],
                      birth_strength=d['birth_strength'], position_supported=d['position_supported'],
                      unsupported=d['unsupported_packets'], zero_participation=d['zero_participation'])
    return res


def timing_at_strengths(case, seconds=6.0):
    """p99 block time of both sides at every s (side B at s, A at 1) and with the knob
    moved every second during the run (the live window's case)."""
    n = int(math.ceil(seconds * SR / BLOCK))
    out = {}
    for s in STRENGTHS:
        runner = DemoRunner(scene_from_doc(scene_for(case, strength=dict(A=1.0, B=s))))
        runner.post('start', at=0)
        y, times = run_blocks(runner, n)
        warm = times[int(1.0 * SR / BLOCK):]
        out[f"{s:g}"] = dict(p99_ms=float(np.percentile(warm, 99)), max_ms=float(warm.max()),
                             finite=bool(np.all(np.isfinite(y['B']))), peak_b=float(runner.sides['B'].peak),
                             clip_b=int(runner.sides['B'].clip_blocks))
    runner = DemoRunner(scene_from_doc(scene_for(case)))
    runner.post('start', at=0)
    for k, s in enumerate((0.0, 2.0, 4.0, 0.5, 1.0)):
        runner.post('set_param', at=int((k + 1) * SR), side='B', name='birth_strength', value=s)
    y, times = run_blocks(runner, n)
    warm = times[int(1.0 * SR / BLOCK):]
    out['knob_moved'] = dict(p99_ms=float(np.percentile(warm, 99)), max_ms=float(warm.max()),
                             finite=bool(np.all(np.isfinite(y['B']))), peak_b=float(runner.sides['B'].peak),
                             clip_b=int(runner.sides['B'].clip_blocks),
                             final_strength=float(runner.snapshot()['sides']['B'][1]['birth_strength']))
    out['budget_ms'] = BUDGET_MS
    out['ok'] = all(v['p99_ms'] < BUDGET_MS for k, v in out.items() if isinstance(v, dict))
    return out


# -- the report -----------------------------------------------------------------------------------------
def write_markdown(rep, path):
    L = []
    L.append('# Objects — Birth strength (M2): measurements')
    L.append('')
    L.append(f"Generated {rep['generated']} on commit `{rep['commit']}`; numba {rep['numba']}.")
    L.append('')
    L.append('## 1. The law against the independent preflight')
    L.append('')
    law = rep['law']
    L.append(f"Law: {law['law']}")
    L.append('')
    L.append(f"{len(law['profiles'])} cases x 5 strengths: max |c − preflight| {law['max_diff']:.1e}, max |Σc² − m| "
             f"{law['max_norm_error']:.1e}; s = 1 returns b itself {law['s1_is_b_itself']}; equal b stay equal "
             f"{law['equal_stay_equal']}; one mode c = 1 {law['one_mode']}; all c ≥ 0 {law['all_nonnegative']}; an exact "
             f"zero at s = 0.5 / 1 / 2 / 4: {law['exact_zero']}.")
    L.append('')
    L.append('Control example (16 cells, three births, 110 / 545 / 660 Hz):')
    L.append('')
    L.append('| s | c1 | c2 | c3 |')
    L.append('|---|---|---|---|')
    for s, c in law['control_16_cells'].items():
        L.append(f"| {s} | {c[0]:.6f} | {c[1]:.6f} | {c[2]:.6f} |")
    L.append('')
    L.append('| case | b | s 0 | s 0.5 | s 1 | s 2 | s 4 | max diff |')
    L.append('|---|---|---|---|---|---|---|---|')
    for row in law['profiles']:
        cs = row['strengths']
        md = max(v['max_diff'] for v in cs.values())
        L.append(f"| {row['id']} | {row['b']} | " + ' | '.join(str([round(x, 3) for x in cs[k]['c']]) for k in ('0', '0.5', '1', '2', '4'))
                 + f" | {md:.1e} |")
    L.append('')
    L.append('## 2. Five engines on the M1 field')
    L.append('')
    m = rep['m1']
    L.append(f"{m['transitions']} transitions side by side (s = 0 / 0.5 / 1 / 2 / 4): equal {m['equal']}; packets with events "
             f"{m['packets_with_events']}; max |Σc² − m| {m['max_norm_error']:.1e}.")
    L.append('')
    L.append('| phase bank | frequencies Hz | s | measured c | preflight c | max diff |')
    L.append('|---|---|---|---|---|---|')
    for ph in m['phases']:
        for s, v in ph['strengths'].items():
            md = '-' if v['max_diff'] is None else f"{v['max_diff']:.1e}"
            L.append(f"| {ph['phase']} | {ph['frequencies_hz']} | {s} | {v['measured']} | {v['preflight']} | {md} |")
    L.append('')
    L.append('States at the end: ' + '; '.join(f"s {s}: per-mode max {v['per_mode_max']:.3g}, uniform max {v['uniform_max']:.3g}, "
                                             f"zero {v['zero_participation']}, refused {v['refused']}" for s, v in m['states'].items()) + '.')
    L.append('')
    L.append('## 3. The PCM path')
    L.append('')
    p = rep['pcm']
    L.append(f"Birth position at s = 0 equals Uniform on the whole M2 scene render: {p['s0_equals_uniform_whole_scene']}.  "
             f"Uniform at s = 4 equals Uniform at s = 1: {p['uniform_same_at_s1_and_s4']}.  s = 1 vs s = 4 max |ΔPCM| "
             f"{p['s1_vs_s4_max_abs_diff']}.")
    L.append('')
    L.append('The pinned records of 2026-09-17 (scene without the key, rendered with the current code and the default s = 1) '
             'against their stored WAVs:')
    L.append('')
    for rid, r in p['records_2026_09_17_equal_current_code'].items():
        L.append(f"- {rid} {r['title'][:60]}: {r['outputs']} (pinned {r['commit']})")
    L.append('')
    L.append(f"s changed on a still field through 0 and 1: no packet, tail and states equal to a twin "
             f"{p['still_field_change_no_packet_tail_equal']}, change counter equal {p['still_field_changes_counter_equal']}.")
    L.append('')
    L.append('A sequence of s across the E1 transitions (one bank; the per-mode history of s > 0 is kept at s = 0, the uniform path adds):')
    L.append('')
    L.append('| transition | s | e | c | per-mode max before / after | uniform max before / after |')
    L.append('|---|---|---|---|---|---|')
    for r in p['sequence_e1']:
        L.append(f"| {r['transition']} | {r['s']:g} | {r['e']:g} | {r['c']} | {r['per_mode_before']:.3g} / {r['per_mode_after']:.3g} | "
                 f"{r['uniform_before']:.3g} / {r['uniform_after']:.3g} |")
    L.append('')
    j = p['journal_knob_changes_continue_exact']
    L.append(f"Runner journal with knob changes {j['changes']}: Continue from a 6 s snapshot exact {j['exact']} "
             f"(B strength at the snapshot {j['b_strength_at_snapshot']}).  A v4 snapshot imports as s = 1 with the same sound: "
             f"{p['v4_snapshot_imports_as_s1_same_sound']}.")
    L.append('')
    L.append('## 4. Levels and timing')
    L.append('')
    L.append('Side B at every s (A at 1), whole take and after 2 s:')
    L.append('')
    L.append('| gains | s | side gain B | RMS dB | after 2 s dB | peak | clip | finite |')
    L.append('|---|---|---|---|---|---|---|---|')
    for label, rows in rep['levels'].items():
        for s, r in rows.items():
            L.append(f"| {label} | {s} | {r['side_gain']:.2f} | {r['rms_db']:.2f} | {r['steady_rms_db']:.2f} | {r['peak']:.3f} | "
                     f"{r['clip_blocks']} | {r['finite']} |")
    L.append('')
    L.append('| scene | strength A/B | side gain B | A RMS dB (after 2 s) | B RMS dB (after 2 s) | A−B after 2 s (unit) | peak A/B | clip | continue | p99 ms (budget 7.98) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for x in rep['scenes']:
        L.append(f"| {x['id']} | {x['strength']['A']:g}/{x['strength']['B']:g} | {x['side_gain']['B']} | "
                 f"{x['A']['rms_db']:.2f} ({x['A']['steady_rms_db']:.2f}) | {x['B']['rms_db']:.2f} ({x['B']['steady_rms_db']:.2f}) | "
                 f"{x['steady_a_minus_b_db']:+.2f} ({x['steady_a_minus_b_db_at_unit_side_gain']:+.2f}) | "
                 f"{x['A']['peak']:.3f}/{x['B']['peak']:.3f} | {x['A']['clip_blocks']}/{x['B']['clip_blocks']} | "
                 f"{x['continue']['exact']} | {x['block_ms']['p99']:.2f} |")
        for s in SIDES:
            L.append(f"{x['id']} {s}: {x[s]['excitation']} s {x[s]['birth_strength']:g} (supported {x[s]['position_supported']}), "
                     f"figures {x[s]['figures']} sounding {x[s]['sounding']} tails {x[s]['tails']} faded {x[s]['evictions']} "
                     f"in place {x[s]['inplace_fades']} dropped {x[s]['drops']} refused {x[s]['unsupported']} zero {x[s]['zero_participation']}")
    L.append('')
    t = rep['timing']
    L.append('Block time p99 / max (ms) of the pair at every s of side B, and with the knob moved every second (6 s runs):')
    L.append('')
    L.append('| run | p99 ms | max ms | peak B | clip B | finite |')
    L.append('|---|---|---|---|---|---|')
    for k, v in t.items():
        if isinstance(v, dict):
            L.append(f"| {k} | {v['p99_ms']:.2f} | {v['max_ms']:.2f} | {v['peak_b']:.3f} | {v['clip_b']} | {v['finite']} |")
    L.append('')
    L.append(f"Budget {t['budget_ms']:.2f} ms; all p99 within the budget: {t['ok']}.")
    L.append('')
    L.append('## 5. Catalog')
    L.append('')
    c = rep['catalog']
    if not c.get('present'):
        L.append(f"Catalog {c['root']} not checked (not built or --quick).")
    else:
        for rid, x in c['records'].items():
            if 'error' in x:
                L.append(f"- {rid}: error {x['error']}")
            else:
                L.append(f"- {rid} {x['title'][:60]}: replay {x['status']} {x['reason'] or ''}, {x['pinned']} {x['commit']}, "
                         f"notes {x['has_notes']}, side gain {x['side_gain']}")
    L.append('')
    Path(path).write_text('\n'.join(L), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(CATALOG_ROOT))
    ap.add_argument('--quick', action='store_true', help='skip the catalog replay')
    a = ap.parse_args(argv)
    from casynth_lab import provenance as prov
    rep = dict(generated=time.strftime('%Y-%m-%d %H:%M'), commit=prov.current().get('commit'), numba=orz.HAVE_NUMBA)
    print('1. law', flush=True)
    rep['law'] = law_checks()
    print('2. M1 at five strengths', flush=True)
    rep['m1'] = m1_five_strengths()
    print('3. PCM path', flush=True)
    rep['pcm'] = pcm_paths()
    print('4. levels / timing', flush=True)
    rep['levels'] = levels()
    rep['scenes'] = [scene_measurements(c) for c in CASES]
    rep['timing'] = timing_at_strengths(CASES[0])
    print('5. catalog', flush=True)
    rep['catalog'] = dict(root=a.root, present=False) if a.quick else verify_catalog(a.root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    write_markdown(rep, OUT_DIR / 'report.md')
    print(f"report: {OUT_DIR / 'report.md'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
