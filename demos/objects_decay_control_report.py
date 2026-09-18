"""Measurements for the Objects decay CONTROL delivery D4 (REQ
memory/req-objects-decay-control-2026-09-18.md, section 6), before anything is played
to the listener:

  1. the mean: an independent per-sample recurrence of the applied gamma of the Modal
     side (the 20 ms smoother run sample by sample in numpy from the block-start
     states and the block targets, its block-end states compared with the engine's)
     over the driven modes of the active banks, window [132300, 793800); the pairs,
     the mean, T, the matched Fixed's gamma (also as the engine applies it) and the
     relative error; the preflight's numbers beside
  2. identity: the raw int16 PCM hashes of the delivered records against the preflight
     and the D3 record; A of D4.1 == A of D4.2, B of D4.1 == B of D3, B of D4.2 == A of
     D3; the journals; per block, the three conditions side by side (D4.1, D4.2 and D3
     run in lockstep) share the field, figures, frequencies, output weights, packets,
     Attack and panning of every active bank -- only the losses and the constant side
     gains differ; the two matched Fixed sides are equal in every engine array
  3. levels: RMS of both channels over 3..18 s of each condition from the records, the
     spread, pre-clip and int16 peaks, clip, finiteness; the unit-gain calibration
  4. the records: replay of A / B / monitor, provenance / pin, Continue from the saved
     end snapshot against a continuous render (125 blocks), the D3 source record still
     replays; the block time of each scene rendered alone

    python demos/objects_decay_control_report.py [--root DIR]

Writes demos/results/objects_decay_control/report.json + report.md.  Stdout ASCII.
"""
import argparse
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from casynth_config import SR                                                # noqa: E402
from casynth_lab import BLOCK, DemoRunner, scene_from_doc                   # noqa: E402
from casynth_lab.catalog import Catalog, read_wav                           # noqa: E402
from casynth_lab import object_resonators as orz                            # noqa: E402
from demos import build_objects_decay_control as d4                         # noqa: E402
from demos.build_objects_decay import CASES as D3_CASES, scene_for as d3_scene_for   # noqa: E402
from demos.build_objects_event_source import rms_db                         # noqa: E402

OUT_DIR = ROOT / 'demos' / 'results' / 'objects_decay_control'
BUDGET_MS = BLOCK / SR * 1000.0
OUTPUTS = ('A', 'B', 'monitor')
N_BLOCKS = d4.N_SAMPLES // BLOCK
CONTINUE_BLOCKS = 125
# per active slot: what the conditions must share (the frequencies, the output weights and
# their ramps, the packets / pulse and Attack states, the panning); zre / zim, rr, gam_*,
# slaw, level (the resonator states and the losses) may differ
SHARED_SLOT = ('slot_id', 'smode', 'ndrive', 'npulse', 'nlive', 'sqrtlam', 'ffreq', 'cth', 'sth', 'wcur', 'winc',
               'wtgt', 'wleft', 'zf', 'zs', 'zu', 'zfm', 'zsm', 'zum', 'pan', 'pleft', 'last_e', 'last_a', 'last_b')
ALL_ARRAYS = orz.ObjectResonatorsEngine._ARRAYS + ('cth', 'sth', 'age')


def sha_pcm(pcm):
    return hashlib.sha256(np.ascontiguousarray(pcm, np.int16).tobytes()).hexdigest()


def pairs_by_id():
    return {p['id']: p for p in d4.PAIRS}


# -- 1. the mean applied gamma of the Modal side (per-sample recurrence) --------------------------
def mean_check():
    """Run D4.1 through the runner; for side B (Modal) wrap the kernel: from the
    block-start applied gamma g and the block target gt of every driven mode of every
    active bank, run g <- k g + (1 - k) gt sample by sample (k = exp(-1 / (44100 *
    0.020)) computed here), sum g of the samples in the window, compare the block-end
    g with the engine's."""
    pf = d4.preflight()
    w0, w1 = d4.WINDOW
    k = math.exp(-1.0 / (44100.0 * 0.020))
    control_gamma = orz.LN1000 / d4.CONDITIONS['matched_fixed']['decay']
    runner = DemoRunner(scene_from_doc(d4.scene_for(pairs_by_id()['od_d4_control'])))
    e = runner.sides['B'].engine
    ea = runner.sides['A'].engine
    acc = dict(sum=0.0, sum_target=0.0, n=0, slower=0, faster=0, gmin=math.inf, gmax=0.0, end_err=0.0,
               not_adaptive=0, clock=0, blocks=0, max_active=0, max_modes=0)
    kernel = e._kernel

    def recurrence_kernel(n, out):
        t0 = acc['clock']
        act = [int(s) for s in np.flatnonzero(e.role == orz.ROLE_ACTIVE)]
        nds = [int(e.ndrive[s]) for s in act]
        acc['not_adaptive'] += sum(int(e.slaw[s] != 1) for s in act)
        g = np.concatenate([e.gam_a[s, :nd] for s, nd in zip(act, nds)]) if act else np.zeros(0)
        gt = np.concatenate([e.gam_t[s, :nd] for s, nd in zip(act, nds)]) if act else np.zeros(0)
        kernel(n, out)
        acc['max_active'] = max(acc['max_active'], len(act))
        acc['max_modes'] = max(acc['max_modes'], g.size)
        for i in range(n):
            g = k * g + (1.0 - k) * gt
            if w0 <= t0 + i < w1 and g.size:
                acc['sum'] += float(g.sum())
                acc['sum_target'] += float(gt.sum())
                acc['n'] += g.size
                acc['slower'] += int(np.count_nonzero(g < control_gamma))
                acc['faster'] += int(np.count_nonzero(g > control_gamma))
                acc['gmin'] = min(acc['gmin'], float(g.min()))
                acc['gmax'] = max(acc['gmax'], float(g.max()))
        if g.size:
            end = np.concatenate([e.gam_a[s, :nd] for s, nd in zip(act, nds)])
            acc['end_err'] = max(acc['end_err'], float(np.max(np.abs(end - g))))
        acc['clock'] = t0 + n
        acc['blocks'] += 1
    e._kernel = recurrence_kernel
    runner.post('start', at=0)
    fixed_gamma = []
    for _ in range(N_BLOCKS):
        runner.next_block()
        if ea.ints[orz.I_R_LEFT] == 0:
            fixed_gamma.append(orz.gamma_of_r(ea.rr[orz.R_CUR]))
    mean = acc['sum'] / acc['n']
    av = pf['averaging']
    out = dict(method='per-sample recurrence g <- k g + (1 - k) gt from the block-start states (numpy, float64); '
                      'J(s) = modes j < ndrive of the slots with role ACTIVE; every mode-sample weighted equally',
               window_samples=[w0, w1], window_seconds=[w0 / SR, w1 / SR], k=k, k_engine=e.ksm, k_equal=bool(k == e.ksm),
               mode_samples=acc['n'], mean_gamma_applied=mean, T_ref=orz.LN1000 / mean,
               mean_gamma_target=acc['sum_target'] / acc['n'], sum_applied=acc['sum'],
               slower_than_control=acc['slower'], faster_than_control=acc['faster'],
               min_gamma=acc['gmin'], max_gamma=acc['gmax'], block_end_max_error=acc['end_err'],
               active_not_adaptive=acc['not_adaptive'], blocks=acc['blocks'], max_active_banks=acc['max_active'],
               max_driven_modes=acc['max_modes'],
               control_decay_s=d4.CONDITIONS['matched_fixed']['decay'], control_gamma=control_gamma,
               control_relative_error=control_gamma / mean - 1.0,
               engine_fixed_gamma=dict(min=min(fixed_gamma), max=max(fixed_gamma), blocks=len(fixed_gamma)),
               engine_fixed_relative_error=max(abs(g / mean - 1.0) for g in fixed_gamma),
               preflight=dict(mode_samples=av['mode_samples'], mean_gamma_applied=av['mean_gamma_applied'],
                              mean_gamma_target=av['mean_gamma_target_diagnostic'],
                              slower=pf['independent_per_sample_average_check']['mode_samples_slower_than_control'],
                              faster=pf['independent_per_sample_average_check']['mode_samples_faster_than_control'],
                              min_gamma=av['min_gamma'], max_gamma=av['max_gamma']))
    out['vs_preflight'] = dict(mode_samples_equal=bool(acc['n'] == av['mode_samples']),
                               mean_relative=mean / av['mean_gamma_applied'] - 1.0,
                               target_relative=out['mean_gamma_target'] / av['mean_gamma_target_diagnostic'] - 1.0)
    out['ok'] = bool(abs(out['control_relative_error']) <= 1e-4 and out['vs_preflight']['mode_samples_equal']
                     and acc['not_adaptive'] == 0)
    print(f"  mean: {acc['n']} mode-samples, applied gamma {mean:.12f} 1/s, T {orz.LN1000 / mean:.10f} s, "
          f"control {control_gamma:.12f} ({out['control_relative_error'] * 100:+.6f} %), block-end error "
          f"{acc['end_err']:.1e}, vs preflight {out['vs_preflight']['mean_relative']:+.1e}", flush=True)
    return out


# -- 2 / 3. the three conditions side by side (D4.1, D4.2, D3 in lockstep) ------------------------------
def lockstep():
    """D4.1, D4.2 and D3 from their scenes, block by block: PCM of every side, the
    per-block equalities, peaks, clip, block times."""
    pb = pairs_by_id()
    docs = dict(control=d4.scene_for(pb['od_d4_control']), anchor=d4.scene_for(pb['od_d4_anchor']),
                d3=d3_scene_for(D3_CASES[2]))
    runners = {k: DemoRunner(scene_from_doc(doc)) for k, doc in docs.items()}
    for r in runners.values():
        r.post('start', at=0)
    cond = dict(matched_fixed=('control', 'A'), modal=('control', 'B'), long_fixed=('anchor', 'B'))
    pcm = {k: {o: [] for o in OUTPUTS} for k in runners}
    times = {k: [] for k in runners}
    acc = dict(field_equal=True, figures_equal=True, shared_equal=True, matched_twins_equal=True, attack_equal=True,
               first_bad=None, loss_differs=dict(control=0, anchor=0), bad_fields=set(), max_active=0, max_tails=0,
               fixed_r_ok=True, finite=True)
    eng = {c: runners[k].sides[s].engine for c, (k, s) in cond.items()}
    twin = runners['anchor'].sides['A'].engine
    r_matched = orz.decay_r(d4.CONDITIONS['matched_fixed']['decay'])
    r_long = orz.decay_r(d4.CONDITIONS['long_fixed']['decay'])
    g_matched = orz.gamma_of_r(r_matched)
    for b in range(N_BLOCKS):
        before = runners['control'].out_samples
        for k, r in runners.items():
            t0 = time.perf_counter()
            blk = r.next_block()
            times[k].append((time.perf_counter() - t0) * 1000.0)
            for o in OUTPUTS:
                pcm[k][o].append(blk.get(o))
        grids = [r.grid for r in runners.values()]
        ok_field = all(np.array_equal(grids[0], g) for g in grids[1:]) and len({r.gen for r in runners.values()}) == 1
        acc['field_equal'] &= ok_field
        ref = eng['matched_fixed']
        for c in ('modal', 'long_fixed'):
            e = eng[c]
            ok_fig = sorted(ref.figures) == sorted(e.figures)
            for fid, fa in ref.figures.items():
                fb = e.figures.get(fid)
                ok_fig &= bool(fb is not None and fa.slot == fb.slot and np.array_equal(fa.cells, fb.cells)
                               and fa.centre == fb.centre)
            acc['figures_equal'] &= ok_fig
            act = np.flatnonzero(ref.role == orz.ROLE_ACTIVE)
            ok_act = np.array_equal(act, np.flatnonzero(e.role == orz.ROLE_ACTIVE))
            for name in SHARED_SLOT:
                if not np.array_equal(getattr(ref, name)[act], getattr(e, name)[act]):
                    ok_act = False
                    acc['bad_fields'].add(name)
            acc['shared_equal'] &= bool(ok_act)
            acc['attack_equal'] &= bool(np.array_equal(ref.qq, e.qq) and ref.ints[orz.I_Q_LEFT] == e.ints[orz.I_Q_LEFT])
            if not (ok_fig and ok_act) and acc['first_bad'] is None:
                acc['first_bad'] = before
        acc['max_active'] = max(acc['max_active'], int(np.count_nonzero(ref.role == orz.ROLE_ACTIVE)))
        acc['max_tails'] = max(acc['max_tails'], int(np.count_nonzero(eng['modal'].role == orz.ROLE_TAIL)))
        # the losses: matched / long Fixed on the global r (after the start ramp), Modal on its own gamma
        if ref.ints[orz.I_R_LEFT] == 0:
            acc['fixed_r_ok'] &= bool(ref.rr[orz.R_CUR] == r_matched and eng['long_fixed'].rr[orz.R_CUR] == r_long
                                      and not ref.slaw.any() and not eng['long_fixed'].slaw.any())
        act = np.flatnonzero(ref.role == orz.ROLE_ACTIVE)
        em = eng['modal']
        acc['loss_differs']['control'] += int(any(bool(em.slaw[s]) and bool(np.any(em.gam_a[s, :int(em.ndrive[s])]
                                                                                   != g_matched)) for s in act))
        acc['loss_differs']['anchor'] += int(bool(act.size) and ref.rr[orz.R_CUR] != eng['long_fixed'].rr[orz.R_CUR])
        # the matched Fixed of D4.1 and of D4.2: every engine array equal
        for name in ALL_ARRAYS:
            if not np.array_equal(getattr(ref, name), getattr(twin, name)):
                acc['matched_twins_equal'] = False
                acc['bad_fields'].add('twin:' + name)
        acc['finite'] &= bool(all(np.isfinite(runners[k].sides[s].peak) for k in runners for s in ('A', 'B'))
                              and all(np.isfinite(e.zre).all() and np.isfinite(e.zim).all() and np.isfinite(e.gam_a).all()
                                      for e in eng.values()))
    y = {k: {o: np.concatenate(pcm[k][o]) for o in OUTPUTS} for k in runners}
    return y, runners, times, acc


def identity_and_levels(y, runners, acc, records):
    pf = d4.preflight()
    pfl = pf['levels_3_to_18_s']
    d3meta = d4.source_meta()
    d3 = {s: read_wav(str(d4.SOURCE_RECORD / f'{s}.wav')) for s in ('A', 'B')}
    cond_pcm = dict(matched_fixed=y['control']['A'], modal=y['control']['B'], long_fixed=y['anchor']['B'])
    out = dict(lengths={k: {o: int(y[k][o].shape[0]) for o in OUTPUTS} for k in y})
    out['rendered'] = dict(
        control_A_equals_anchor_A=bool(np.array_equal(y['control']['A'], y['anchor']['A'])),
        control_B_equals_d3_B=bool(np.array_equal(y['control']['B'], y['d3']['B'])),
        anchor_B_equals_d3_A=bool(np.array_equal(y['anchor']['B'], y['d3']['A'])),
        d3_rendered_equals_d3_record=bool(np.array_equal(y['d3']['A'], d3['A']) and np.array_equal(y['d3']['B'], d3['B'])),
        monitor_equals_A=bool(all(np.array_equal(y[k]['monitor'], y[k]['A']) for k in y)))
    out['hashes'] = {c: dict(rendered=sha_pcm(p), preflight=pfl[c]['pcm_sha256'], equal=sha_pcm(p) == pfl[c]['pcm_sha256'])
                     for c, p in cond_pcm.items()}
    out['d3_record_hashes'] = {s: d3meta['audio'][s]['sha256'] for s in ('A', 'B')}
    # the delivered records: their stored hashes, the WAVs, the journals
    recs = {}
    for pid, rec in records.items():
        meta = rec.meta
        wav = {o: read_wav(rec.wav_path(o)) for o in OUTPUTS}
        steps = [(j['out_sample'], j['args'].get('gen')) for j in meta['journal'] if j['kind'] == 'step']
        d3_steps = [(j['out_sample'], j['args'].get('gen')) for j in d3meta['journal'] if j['kind'] == 'step']
        recs[pid] = dict(id=rec.id, samples={o: int(wav[o].shape[0]) for o in OUTPUTS},
                         sha256={o: meta['audio'][o]['sha256'] for o in OUTPUTS},
                         sha256_of_wav={o: sha_pcm(wav[o]) for o in OUTPUTS},
                         wav_equals_rendered={o: bool(np.array_equal(wav[o], y['control' if pid == 'od_d4_control'
                                                                                  else 'anchor'][o])) for o in OUTPUTS},
                         journal_kinds=sorted({j['kind'] for j in meta['journal']}),
                         steps=len(steps), steps_equal_d3=steps == d3_steps,
                         end_sample=meta['end_sample'], clip_blocks=meta['diagnostics']['clip_blocks'],
                         side_gain=meta['scene']['audio']['side_gain'],
                         vol_initial=meta['runner']['vol_initial'], status=rec.status, commit=rec.commit)
    out['records'] = recs
    a1, a2 = records['od_d4_control'], records['od_d4_anchor']
    out['records_identity'] = dict(
        A_equal=recs['od_d4_control']['sha256']['A'] == recs['od_d4_anchor']['sha256']['A']
        and bool(np.array_equal(read_wav(a1.wav_path('A')), read_wav(a2.wav_path('A')))),
        A_is_preflight_matched=recs['od_d4_control']['sha256']['A'] == pfl['matched_fixed']['pcm_sha256'],
        B_d41_equals_d3_B=recs['od_d4_control']['sha256']['B'] == d3meta['audio']['B']['sha256'] == pfl['modal']['pcm_sha256'],
        B_d42_equals_d3_A=recs['od_d4_anchor']['sha256']['B'] == d3meta['audio']['A']['sha256'] == pfl['long_fixed']['pcm_sha256'])
    # per block
    out['per_block'] = dict(blocks=N_BLOCKS, field_equal=acc['field_equal'], figures_equal=acc['figures_equal'],
                            active_banks_shared_equal=acc['shared_equal'], attack_equal=acc['attack_equal'],
                            matched_twins_every_array_equal=acc['matched_twins_equal'], first_bad_sample=acc['first_bad'],
                            differing_fields=sorted(acc['bad_fields']), fixed_r_exact=acc['fixed_r_ok'],
                            blocks_losses_differ=acc['loss_differs'], max_active_banks=acc['max_active'],
                            max_tails_modal=acc['max_tails'], shared_fields=list(SHARED_SLOT), finite=acc['finite'])
    # levels from the records (RMS of all samples of both channels over [3, 18) s)
    wavs = dict(matched_fixed=read_wav(a1.wav_path('A')), modal=read_wav(a1.wav_path('B')),
                long_fixed=read_wav(a2.wav_path('B')))
    lv = {}
    for c, p in wavs.items():
        k, s = dict(matched_fixed=('control', 'A'), modal=('control', 'B'), long_fixed=('anchor', 'B'))[c]
        lv[c] = dict(rms_dbfs=d4.window_rms_db(p), preflight_rms_dbfs=pfl[c]['rms_dbfs'],
                     int16_peak=float(np.abs(p.astype(np.int32)).max()) / 32767.0,
                     window_int16_peak=float(np.abs(p[d4.WINDOW[0]:d4.WINDOW[1]].astype(np.int32)).max()) / 32767.0,
                     preflight_pcm_peak=pfl[c]['pcm_peak'], preclip_peak_take=None,
                     side_gain=d4.CONDITIONS[c]['side_gain'], clip_blocks=int(runners[k].sides[s].clip_blocks),
                     at_full_scale=int(np.count_nonzero(np.abs(p.astype(np.int32)) >= 32767)))
        lv[c]['vs_preflight_db'] = lv[c]['rms_dbfs'] - lv[c]['preflight_rms_dbfs']
    vals = [v['rms_dbfs'] for v in lv.values()]
    out['levels'] = dict(conditions=lv, spread_db=max(vals) - min(vals), within_0_1_db=bool(max(vals) - min(vals) <= 0.1),
                         preflight_preclip_peaks_main_pair=pf['preclip_peaks_main_pair'])
    return out


def preclip_peaks():
    """The pre-clip peak (max |float| of a block) of every condition over the take."""
    pb = pairs_by_id()
    out = {}
    for pid, conds in (('od_d4_control', ('matched_fixed', 'modal')), ('od_d4_anchor', (None, 'long_fixed'))):
        r = DemoRunner(scene_from_doc(d4.scene_for(pb[pid])))
        r.post('start', at=0)
        pk = dict(A=0.0, B=0.0)
        for _ in range(N_BLOCKS):
            r.next_block()
            for s in ('A', 'B'):
                pk[s] = max(pk[s], r.sides[s].peak)
        for s, c in zip(('A', 'B'), conds):
            if c:
                out[c] = pk[s]
    return out


# -- 4. the records: replay, pin, Continue ----------------------------------------------------------
def records_check(root):
    cat = Catalog(str(root))
    recs = {}
    for rec, err in cat.list():
        if err is not None:
            raise SystemExit(f'broken record: {err}')
        recs[rec.meta['demo_scene']['id']] = rec
    assert sorted(recs) == sorted(p['id'] for p in d4.PAIRS), sorted(recs)
    out = {}
    for pid, rec in recs.items():
        res = cat.replay(rec.id, yield_cpu=False)
        # Continue: the saved end snapshot, driven on, against one continuous render
        runner, _state, _r = cat.continue_runner(rec.id)
        cont = DemoRunner(scene_from_doc(d4.scene_for(pairs_by_id()[pid])))
        cont.post('start', at=0)
        while cont.out_samples < rec.meta['end_sample']:
            cont.next_block()
        exact = dict(A=True, B=True, monitor=True)
        clock_equal = runner.out_samples == cont.out_samples and runner.gen == cont.gen
        for _ in range(CONTINUE_BLOCKS):
            a, b = runner.next_block(), cont.next_block()
            for o in OUTPUTS:
                exact[o] &= bool(np.array_equal(a.get(o), b.get(o)))
        out[pid] = dict(id=rec.id, title=rec.title, replay=res.status, replay_outputs=res.outputs, status=rec.status,
                        commit=rec.commit, pin_ref=(rec.meta.get('pin') or {}).get('ref'), has_notes=rec.has_notes,
                        notes_start=rec.notes[:40], continue_blocks=CONTINUE_BLOCKS, continue_clock_equal=clock_equal,
                        continue_exact=exact)
        print(f"  {pid} {rec.id}: replay {res.status} {res.outputs}, {rec.status} {rec.commit[:7]}, Continue "
              f"{CONTINUE_BLOCKS} blocks {exact}", flush=True)
    src = Catalog(str(d4.SOURCE_RECORD.parent))
    r3 = src.replay(d4.SOURCE_RECORD.name, yield_cpu=False)
    out['d3_source'] = dict(id=d4.SOURCE_RECORD.name, replay=r3.status, replay_outputs=r3.outputs)
    print(f"  D3 source record replay: {r3.status} {r3.outputs}", flush=True)
    return out, recs


def timing():
    """Block time of each delivered scene rendered ALONE (both sides, no inspection in
    the loop), after 1 s of warm-up."""
    warm = int(1.0 * SR / BLOCK)
    out = {}
    for pair in d4.PAIRS:
        r = DemoRunner(scene_from_doc(d4.scene_for(pair)))
        r.post('start', at=0)
        t = []
        for _ in range(N_BLOCKS):
            t0 = time.perf_counter()
            r.next_block()
            t.append((time.perf_counter() - t0) * 1000.0)
        k = pair['id']
        t = np.asarray(t[warm:])
        out[k] = dict(p50=float(np.percentile(t, 50)), p99=float(np.percentile(t, 99)), max=float(t.max()),
                      budget=BUDGET_MS, ok=bool(np.percentile(t, 99) < BUDGET_MS))
    return out


# -- the report ---------------------------------------------------------------------------------------
def write_markdown(rep, path):
    m, idn, lv = rep['mean'], rep['identity'], rep['identity']['levels']
    L = ['# Objects — decay control D4: measurements', '',
         f"Generated {rep['generated']} on commit `{rep['commit']}`. REQ `memory/req-objects-decay-control-2026-09-18.md`, "
         f"preflight `memory/research/objects-decay-control-preflight-2026-09-18.json` (on `255af20`). "
         f"Catalog `{rep['catalog']}`.", '',
         '## 1. The mean applied gamma of the Modal side', '',
         f"Method: {m['method']}. k = exp(-1 / (44100 * 0.020)) = {m['k']!r} computed here, equal to the engine's: "
         f"{m['k_equal']}. Window [{m['window_samples'][0]}, {m['window_samples'][1]}) samples = "
         f"[{m['window_seconds'][0]:g}, {m['window_seconds'][1]:g}) s, the partial blocks at both ends counted sample by sample.",
         '',
         '| | this report | preflight |', '|---|---|---|',
         f"| mode-samples | {m['mode_samples']} | {m['preflight']['mode_samples']} |",
         f"| mean applied gamma, 1/s | {m['mean_gamma_applied']:.12f} | {m['preflight']['mean_gamma_applied']:.12f} |",
         f"| T_ref = ln 1000 / gamma, s | {m['T_ref']:.10f} | {d4.preflight()['averaging']['matched_fixed_T60_full_precision']:.10f} |",
         f"| mean target gamma (diagnostic, not the control), 1/s | {m['mean_gamma_target']:.12f} | {m['preflight']['mean_gamma_target']:.12f} |",
         f"| mode-samples below / above the control gamma | {m['slower_than_control']} / {m['faster_than_control']} | "
         f"{m['preflight']['slower']} / {m['preflight']['faster']} |",
         f"| min / max applied gamma, 1/s | {m['min_gamma']:.6f} / {m['max_gamma']:.6f} | {m['preflight']['min_gamma']:.6f} / "
         f"{m['preflight']['max_gamma']:.6f} |", '',
         f"Relative difference of the mean from the preflight: {m['vs_preflight']['mean_relative']:+.2e}; block-end applied "
         f"gamma of the recurrence against the engine: max |diff| {m['block_end_max_error']:.1e} over {m['blocks']} blocks "
         f"(up to {m['max_active_banks']} active banks, {m['max_driven_modes']} driven modes); active banks not on their own "
         f"gamma: {m['active_not_adaptive']}.", '',
         f"**Control:** Decay {m['control_decay_s']} s -> gamma ln 1000 / {m['control_decay_s']} = {m['control_gamma']:.12f} 1/s, "
         f"**{m['control_relative_error'] * 100:+.6f} %** from the mean (limit 0.01 %). The engine's applied Fixed gamma "
         f"(-SR ln r, r = 10^(-3 / (SR T))): {m['engine_fixed_gamma']['min']:.12f}…{m['engine_fixed_gamma']['max']:.12f} 1/s, "
         f"max {m['engine_fixed_relative_error'] * 100:+.6f} % from the mean.", '',
         '## 2. Identity', '',
         'Raw int16 interleaved PCM, whole take (794112 samples):', '',
         '| condition | rendered now | preflight | equal |', '|---|---|---|---|']
    for c, h in idn['hashes'].items():
        L.append(f"| {c} | `{h['rendered'][:16]}…` | `{h['preflight'][:16]}…` | {h['equal']} |")
    L += ['', '| record | id | samples A/B/monitor | SHA-256 A | SHA-256 B | WAV == stored hash | journal | steps == D3 | '
          'side gain | pin |', '|---|---|---|---|---|---|---|---|---|---|']
    for pid, r in idn['records'].items():
        L.append(f"| {pid} | `{r['id']}` | {r['samples']['A']}/{r['samples']['B']}/{r['samples']['monitor']} | "
                 f"`{r['sha256']['A'][:12]}…` | `{r['sha256']['B'][:12]}…` | "
                 f"{all(r['sha256'][o] == r['sha256_of_wav'][o] for o in OUTPUTS)} | {', '.join(r['journal_kinds'])} "
                 f"({r['steps']} steps) | {r['steps_equal_d3']} | A {r['side_gain']['A']} / B {r['side_gain']['B']} | "
                 f"{r['status']} `{(r['commit'] or '')[:7]}` |")
    ri, rr = idn['records_identity'], idn['rendered']
    pb = idn['per_block']
    L += ['', f"- A of D4.1 == A of D4.2 (bytes of the WAVs and hashes): **{ri['A_equal']}**; == the preflight's matched Fixed: "
          f"{ri['A_is_preflight_matched']}",
          f"- B of D4.1 == B of the D3 record == the preflight's Modal: **{ri['B_d41_equals_d3_B']}**",
          f"- B of D4.2 == A of the D3 record == the preflight's long Fixed: **{ri['B_d42_equals_d3_A']}**",
          f"- rendered side by side now: A(D4.1) == A(D4.2) {rr['control_A_equals_anchor_A']}, B(D4.1) == B(D3) "
          f"{rr['control_B_equals_d3_B']}, B(D4.2) == A(D3) {rr['anchor_B_equals_d3_A']}; the D3 scene renders the D3 "
          f"record's WAVs {rr['d3_rendered_equals_d3_record']}; monitor == A (side A listened) {rr['monitor_equals_A']}",
          '', f"Per block ({pb['blocks']} blocks, D4.1, D4.2 and D3 in lockstep): field and generation equal "
          f"{pb['field_equal']}; figures (ids, slots, cells, centres) equal {pb['figures_equal']}; every active bank "
          f"of the three conditions shares {', '.join(pb['shared_fields'])}: {pb['active_banks_shared_equal']}; Attack "
          f"state equal {pb['attack_equal']}; the two matched Fixed sides equal in every engine array: "
          f"{pb['matched_twins_every_array_equal']}; differing fields: {pb['differing_fields'] or 'none'}. "
          f"Losses: matched / long Fixed on exactly r(0.671529) / r(1.39), no own-gamma slot: {pb['fixed_r_exact']}; "
          f"blocks where the losses of the active banks differ from the matched Fixed — D4.1 (Modal applied gamma != "
          f"the matched gamma) {pb['blocks_losses_differ']['control']}, D4.2 (r) "
          f"{pb['blocks_losses_differ']['anchor']}. Up to {pb['max_active_banks']} active banks, {pb['max_tails_modal']} "
          f"tails in Modal. What differs by design: the resonator states, the losses and the constant side gain.", '',
          '## 3. Levels', '',
          'RMS of all samples of both channels over [3, 18) s of the delivered records; one constant side gain per condition.',
          '', '| condition | side gain | RMS dBFS | preflight | diff dB | int16 peak | pre-clip peak | clip blocks | full-scale samples |',
          '|---|---|---|---|---|---|---|---|---|']
    for c, v in lv['conditions'].items():
        L.append(f"| {c} | {v['side_gain']} | {v['rms_dbfs']:.4f} | {v['preflight_rms_dbfs']:.4f} | {v['vs_preflight_db']:+.1e} | "
                 f"{v['int16_peak']:.4f} | {v['preclip_peak_take']:.4f} | {v['clip_blocks']} | {v['at_full_scale']} |")
    cal = rep['calibration']
    L += ['', f"Spread of the three RMS: **{lv['spread_db']:.4f} dB** (limit 0.1). All values finite: {pb['finite']}.", '',
          'Unit-gain calibration (`--calibrate`): ' + '; '.join(
              f"{c} {v['rms_db']:.4f} dBFS -> gain to the long Fixed {v['gain_to_long_fixed']:.6f} (delivered {v['delivered_gain']})"
              for c, v in cal.items()) + '.', '',
          '## 4. Records', '',
          '| record | id | replay A / B / monitor | pin | Notes | Continue (125 blocks) |', '|---|---|---|---|---|---|']
    for pid in ('od_d4_control', 'od_d4_anchor'):
        r = rep['records'][pid]
        L.append(f"| {pid} | `{r['id']}` | {r['replay']} {r['replay_outputs']} | {r['status']} `{r['commit'][:7]}` | "
                 f"{r['has_notes']} | clock {r['continue_clock_equal']}, {r['continue_exact']} |")
    s3 = rep['records']['d3_source']
    L += ['', f"The D3 source record `{s3['id']}` still replays: {s3['replay']} {s3['replay_outputs']}.", '',
          'Block time, each scene rendered alone (both sides), after 1 s:', '',
          '| scene | p50 ms | p99 ms | max ms | budget |', '|---|---|---|---|---|']
    for k, t in rep['timing'].items():
        L.append(f"| {k} | {t['p50']:.2f} | {t['p99']:.2f} | {t['max']:.2f} | {t['budget']:.2f} |")
    L += ['', 'The engine and its code are those of D3 (no sound code changed): single slow blocks are desktop noise -- '
          'the D3 scene itself, measured twice in a row the same way on 2026-09-18, gave max 4.98 ms and 11.72 ms '
          '(p99 4.01 / 5.67 ms).', '', f"**Verdict:** {rep['verdict']}", '']
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description="measurements of the Objects decay control delivery D4")
    ap.add_argument('--root', default=str(d4.CATALOG_ROOT))
    a = ap.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    print('1. mean', flush=True)
    mean = mean_check()
    print('4. records', flush=True)
    recs_out, recs = records_check(a.root)
    print('2/3. lockstep', flush=True)
    y, runners, _times, acc = lockstep()
    ident = identity_and_levels(y, runners, acc, recs)
    for c, pk in preclip_peaks().items():
        ident['levels']['conditions'][c]['preclip_peak_take'] = pk
    print('   calibration', flush=True)
    cal = d4.calibrate()
    ri, pb, lv = ident['records_identity'], ident['per_block'], ident['levels']
    checks = dict(mean=mean['ok'], identity=bool(all(ri.values()) and all(ident['rendered'].values())),
                  per_block=bool(pb['field_equal'] and pb['figures_equal'] and pb['active_banks_shared_equal']
                                 and pb['attack_equal'] and pb['matched_twins_every_array_equal'] and pb['fixed_r_exact']),
                  levels=bool(lv['within_0_1_db'] and pb['finite']
                              and all(v['clip_blocks'] == 0 for v in lv['conditions'].values())),
                  records=bool(all(recs_out[p]['replay'] == 'match' and all(recs_out[p]['continue_exact'].values())
                                   and recs_out[p]['status'] == 'pinned' for p in ('od_d4_control', 'od_d4_anchor'))
                               and recs_out['d3_source']['replay'] == 'match'))
    verdict = ('all checks pass' if all(checks.values()) else 'FAILED: ' + ', '.join(k for k, v in checks.items() if not v))
    rep = dict(generated=_dt.datetime.now().isoformat(timespec='seconds'), commit=commit,
               catalog=str(Path(a.root).relative_to(ROOT)).replace('\\', '/'), mean=mean, identity=ident,
               calibration=cal, records=recs_out, timing=timing(), checks=checks, verdict=verdict)
    (OUT_DIR / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float) + '\n',
                                         encoding='utf-8')
    write_markdown(rep, OUT_DIR / 'report.md')
    print(f"checks {checks}\n{verdict}\nreport: {OUT_DIR / 'report.md'}", flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
