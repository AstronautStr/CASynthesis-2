"""Measurements for the Objects "event source / modal excitation" delivery (REQ
memory/req-objects-event-source-modal-2026-09-17.md, "Сверка перед передачей"):

  1. E1 (Octagon II p5): the packets of the one bank per generation for Births /
     Deaths / Both against INDEPENDENT masks (births inside the figure's current
     cells, deaths inside its previous cells) over the 72 transitions of 12 s, the
     per-period sequences, the bank id; a manual birth and a manual death on a
     still figure (which side strikes), the last figure vanishing in Deaths (the
     tail's single packet: how many blocks it feeds, never repeated)
  2. M1 (Jam p3): A Uniform / B Birth position side by side -- frequencies, output
     weights, packet moments and a equal block by block; the b of every packet
     (sum b^2 = m) and the preflight coefficients per phase; the single-birth
     transfer on the fixed 11-cell geometry, zero participation, the degenerate
     groups (sign / basis rotation invariance), translation and rotation
  3. the excitation path: responses of two packets add (a strike never changes the
     output weights); Attack 0 / 4 ms on a per-mode packet (first sample, RMS)
  4. the two scenes: RMS of both sides with the side gains (whole take and after
     2 s, <= 1 dB), peaks / clip, Continue from 6 s exact, the block time budget of
     both sides after warm-up
  5. the catalog: every record replays exactly (when built)

    python demos/objects_event_source_report.py [--root DIR] [--quick]

Writes demos/results/objects_event_source/report.json + report.md.  Stdout ASCII.
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
from casynth_lab import figures as fg                                       # noqa: E402
from casynth_lab import object_resonators as orz                            # noqa: E402
from demos.build_objects_event_source import (CASES, CATALOG_ROOT, PREFLIGHT, F0_HZ, RATE_HZ, SECONDS,   # noqa: E402
                                              STEADY_FROM_S, scene_for, preflight_params, case_cells, preflight)

OUT_DIR = ROOT / 'demos' / 'results' / 'objects_event_source'
BUDGET_MS = BLOCK / SR * 1000.0
SIDES = ('A', 'B')
CTX = EngineContext(SR, BLOCK, 2, F0_HZ, 1.0, RATE_HZ)
GAIN = MASTER_GAIN * 0.7
EID = orz.ENGINE_ID
ROWS = COLS = 32
BOTH, BIRTHS, DEATHS = orz.EV_BOTH, orz.EV_BIRTHS, orz.EV_DEATHS
UNIFORM, POSITION = orz.EXC_UNIFORM, orz.EXC_POSITION
EV_NAME = {BOTH: 'Both', BIRTHS: 'Births', DEATHS: 'Deaths'}


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-12))


def rms_db(y):
    y = np.asarray(y, np.float64)
    if y.dtype.kind == 'i' or (y.size and np.abs(y).max() > 4.0):
        y = y / 32767.0
    return db(math.sqrt(float(np.mean(y * y)))) if y.size else -240.0


def grid(cells, rows=ROWS, cols=COLS):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def own(cells):
    return fg.own_mask(np.asarray(cells), ROWS, COLS)


def params(events, excitation, **over):
    p = preflight_params(events, excitation)
    p.update(over)
    return p


def make_engine(cells, p):
    e = registry.create(EID, CTX, p)
    e.init(grid(cells), None, GAIN)
    return e


def generations(e, g, transitions, blocks_per_gen):
    out = [(g.copy(), e.display())]
    for _ in range(transitions):
        prev, g = g, step(g)
        e.update_field(g, events_field(prev, g))
        for _b in range(blocks_per_gen):
            e.render_float(GAIN)
        out.append((g.copy(), e.display()))
    return out


# -- 1. E1 ----------------------------------------------------------------------------------------
def e1_packets(transitions=72):
    cells = case_cells('E1')
    per_gen_blocks = int(round(SR / RATE_HZ / BLOCK))
    out = dict(transitions=transitions, period=preflight()['cases']['E1']['period'], runs={})
    for events in (BIRTHS, DEATHS, BOTH):
        e = make_engine(cells, params(events, UNIFORM))
        e.render_float(GAIN)
        start = float(e.display()['figures'][0]['e'])
        gens = generations(e, grid(cells), transitions, per_gen_blocks)
        ids = sorted({f['id'] for _g, d in gens for f in d['figures']})
        seq, expect, mismatches = [], [], 0
        for t in range(1, len(gens)):
            g_prev, d_prev = gens[t - 1]
            g_cur, d_cur = gens[t]
            births = (g_cur != 0) & (g_prev == 0)
            deaths = (g_prev != 0) & (g_cur == 0)
            prev = {f['id']: f for f in d_prev['figures']}
            for f in d_cur['figures']:
                e_b = int(np.count_nonzero(births & own(f['cells'])))
                p = prev.get(f['id'])
                e_d = int(np.count_nonzero(deaths & own(p['cells']))) if p else 0
                want = (e_b + e_d) if events == BOTH else e_b if events == BIRTHS else (e_d if p else 0)
                seq.append(float(f['e']))
                expect.append(want)
                mismatches += int(float(f['e']) != float(want))
        period = out['period']
        d = e.display()
        out['runs'][EV_NAME[events]] = dict(
            start_packet=start, bank_ids=ids, one_bank=(len(ids) == 1), packet_mismatches=mismatches,
            first_two_periods=seq[:2 * period], period_sequence=seq[period:2 * period],
            periodic=all(seq[k] == seq[k % period + period] for k in range(period, len(seq))),
            packets_total=sum(1 for v in seq if v > 0), tails=d['n_tails'], tails_fed=d['n_tails_fed'],
            unsupported=d['unsupported_packets'], per_mode_states_nonzero=int(np.count_nonzero(e.zfm)))
    b, dd, o = (out['runs'][k]['first_two_periods'] for k in ('Births', 'Deaths', 'Both'))
    out['both_equals_births_plus_deaths'] = all(x + y == z for x, y, z in zip(b, dd, o))
    return out


def manual_isolation():
    """A still 2 x 2 block: a manual birth (a cell added), then its death, then the
    whole figure erased -- what each event source does, and the tail of Deaths."""
    block = [(5, 5), (5, 6), (6, 5), (6, 6)]
    fields = [grid(block), grid(block + [(4, 4)]), grid(block), np.zeros((ROWS, COLS), np.uint8)]
    out = {}
    for events in (BIRTHS, DEATHS, BOTH):
        e = make_engine(block, params(events, UNIFORM))
        e.render_float(GAIN)
        row = dict(start_packet=float(e.display()['figures'][0]['e']))
        seq = []
        for k in range(1, 3):
            e.update_field(fields[k], events_field(fields[k - 1], fields[k]))
            e.render_float(GAIN)
            seq.append(float(e.display()['figures'][0]['e']))
        row['manual_birth_then_death_packets'] = seq
        e.update_field(fields[3], events_field(fields[2], fields[3]))
        e._boundary()
        t = int(np.nonzero(e.role == orz.ROLE_TAIL)[0][0])
        row['vanished_tail'] = dict(zf=float(e.zf[t]), npulse=int(e.npulse[t]), fed=e.display()['n_tails_fed'])
        fed_blocks, peaks, zf_trace = 0, [], []
        for k in range(30):
            y, _pk, _nc = e.render_float(GAIN)
            peaks.append(float(np.abs(y).max()))
            zf_trace.append(float(e.zf[t]))
            fed_blocks += int(e.display()['n_tails_fed'] > 0)
        row['tail_feeds_blocks'] = fed_blocks
        row['tail_pulse_monotone'] = all(b <= a for a, b in zip(zf_trace, zf_trace[1:]))
        row['tail_peak_first_block'] = peaks[0]
        row['tail_peak_block_30'] = peaks[-1]
        out[EV_NAME[events]] = row
    return out


# -- 2. M1 -----------------------------------------------------------------------------------------
def m1_side_by_side(transitions=72):
    cells = case_cells('M1')
    pf = preflight()['cases']['M1']
    per_gen_blocks = int(round(SR / RATE_HZ / BLOCK))
    ea = make_engine(cells, params(BIRTHS, UNIFORM))
    eb = make_engine(cells, params(BIRTHS, POSITION))
    g = grid(cells)
    equal = dict(frequencies=True, weights=True, packet_moments=True, a=True, tracker=True)
    norm_err = 0.0
    seen = {}
    packets = 0
    for t in range(transitions + 1):
        if t > 0:
            prev, g = g, step(g)
            for e in (ea, eb):
                e.update_field(g, events_field(prev, g))
        for _ in range(per_gen_blocks):
            ea.render_float(GAIN)
            eb.render_float(GAIN)
            if sorted(ea.figures) != sorted(eb.figures):
                equal['tracker'] = False
                continue
            for fid, fa in ea.figures.items():
                sa, sb = fa.slot, eb.figures[fid].slot
                if sa != sb:
                    equal['tracker'] = False
                if sa < 0 or sb < 0:
                    continue
                equal['frequencies'] &= bool(np.array_equal(ea.ffreq[sa], eb.ffreq[sb]))
                equal['weights'] &= bool(np.array_equal(ea.wtgt[sa], eb.wtgt[sb]) and np.array_equal(ea.wcur[sa], eb.wcur[sb]))
                equal['packet_moments'] &= float(ea.last_e[sa]) == float(eb.last_e[sb])
                equal['a'] &= float(ea.last_a[sa]) == float(eb.last_a[sb])
        da, dbb = ea.display(), eb.display()
        for fa, fb in zip(da['figures'], dbb['figures']):
            if fa['slot'] < 0 or fa['e'] <= 0.0:
                continue
            packets += 1
            b = np.asarray(fb['b'])
            norm_err = max(norm_err, abs(float((b * b).sum()) - fa['modes']))
            key = f"gen{(t - 1) % pf['period'] + 1 if t else 0}_{fa['n']}c_{int(fa['e'])}b"
            seen.setdefault(key, dict(b=b.tolist(), a=float(fa['a']), freqs=[float(x) for x in ea.ffreq[fa['slot'], :fa['modes']]]))
    pre = {}
    for ph in pf['phases']:
        for bank in ph['banks']:
            if bank['births'] > 0:
                key = f"gen{ph['generation']}_{len(bank['cells'])}c_{bank['births']}b"
                got = seen.get(key)
                pre[key] = dict(preflight=bank['weights_spatial'], measured=(got or {}).get('b'),
                                frequencies_preflight=bank['frequencies_hz'], frequencies=(got or {}).get('freqs'),
                                max_abs_diff=(float(np.abs(np.asarray(got['b']) - np.asarray(bank['weights_spatial'])).max())
                                              if got else None))
    dbb = eb.display()
    return dict(transitions=transitions, equal=equal, packets_with_events=packets, max_norm_error=norm_err,
                preflight_phases=pre, zero_participation=dbb['zero_participation'],
                unsupported=dbb['unsupported_packets'], b_sample=seen)


def law_checks():
    pf = preflight()
    chk = pf['independent_checks']['single_birth_same_11_cell_geometry']
    cells11 = [tuple(c) for c in pf['cases']['M1']['phases'][0]['banks'][0]['cells']]
    settings = orz.laplace_settings(params(BIRTHS, POSITION))

    def graph(cells):
        return orz.laplace_modes_of(np.asarray(cells), ROWS, COLS, F0_HZ, settings, None, with_graph=True)

    def born(cells, born_cells, gr):
        cells = np.asarray(cells)
        m = np.zeros((ROWS, COLS), bool)
        for r, c in born_cells:
            m[r, c] = True
        return m[cells[gr['order'], 0], cells[gr['order'], 1]]
    freqs, _a, g11 = graph(cells11)
    single = []
    for cell in chk['cells']:
        b, total = orz.birth_position_weights(g11['L'], g11['idx'], born(cells11, [tuple(cell)], g11))
        single.append(dict(cell=list(cell), b=b.tolist(), total=total))
    dist = float(np.linalg.norm(np.asarray(single[0]['b']) - np.asarray(single[1]['b'])))
    out = dict(single_birth=dict(cells=chk['cells'], measured=[s['b'] for s in single], preflight=chk['input_weights'],
                                 profile_distance=dist, preflight_distance=chk['max_profile_distance'],
                                 max_abs_diff=float(np.abs(np.asarray([s['b'] for s in single]) - np.asarray(chk['input_weights'])).max())))
    b0, t0 = orz.birth_position_weights(g11['L'], g11['idx'], np.zeros(11, bool))
    out['zero_participation'] = dict(b=b0.tolist(), total=t0, packet=bool(t0 > orz.ZERO_PART))
    ball, _t = orz.birth_position_weights(g11['L'], g11['idx'], np.ones(11, bool))
    out['whole_figure_birth'] = dict(b=ball.tolist(), max_abs_diff_from_one=float(np.abs(ball - 1.0).max()))
    # degenerate groups (E1 gen 1: ranks [2, 8, 1]): sign and basis rotation invariance
    cells24 = [tuple(c) for c in pf['cases']['E1']['phases'][0]['banks'][0]['cells']]
    _f, _a, g24 = graph(cells24)
    lam, V = np.linalg.eigh(g24['L'])
    groups = orz.degenerate_groups(lam, g24['idx'])
    bmask = born(cells24, cells24[:5], g24)
    b_ref, _t = orz.birth_position_weights(g24['L'], g24['idx'], bmask)
    rng = np.random.default_rng(11)

    def b_of(Vx):
        part = (Vx[bmask, :] ** 2).sum(axis=0)
        p = np.array([part[q].sum() / len(q) for q in groups])
        return np.sqrt(len(p) * p / p.sum())
    Vs = V * rng.choice([-1.0, 1.0], size=V.shape[1])[None, :]
    Vr = V.copy()
    for q in groups:
        if len(q) > 1:
            Q, _r = np.linalg.qr(rng.normal(size=(len(q), len(q))))
            Vr[:, q] = V[:, q] @ Q
    out['degenerate'] = dict(ranks=[len(q) for q in groups], preflight_ranks=pf['cases']['E1']['phases'][0]['banks'][0]['degenerate_ranks'],
                             sign_flip_max_error=float(np.abs(b_of(Vs) - b_ref).max()),
                             basis_rotation_max_error=float(np.abs(b_of(Vr) - b_ref).max()),
                             preflight_sign=pf['independent_checks']['sign_invariance_max_error'],
                             preflight_rotation=pf['independent_checks']['degenerate_basis_rotation_max_error'])
    # translation (across the seam) and rotation of the 11-cell figure with its birth
    ref = np.asarray(single[0]['b'])
    moves = {}
    for dr, dc in ((20, 25), (-9, 13), (28, 0)):
        moved = [((r + dr) % ROWS, (c + dc) % COLS) for r, c in cells11]
        _f, _a, gm = graph(moved)
        bm, _t = orz.birth_position_weights(gm['L'], gm['idx'], born(moved, [((8 + dr) % ROWS, (9 + dc) % COLS)], gm))
        moves[f"{dr},{dc}"] = float(np.abs(bm - ref).max())
    rot = [(c, ROWS - 1 - r) for r, c in cells11]
    _f, _a, gr = graph(rot)
    br, _t = orz.birth_position_weights(gr['L'], gr['idx'], born(rot, [(9, ROWS - 1 - 8)], gr))
    out['translation_max_error'] = moves
    out['rotation_max_error'] = float(np.abs(br - ref).max())
    return out


# -- 3. the excitation path ---------------------------------------------------------------------------
def setup_slot(e, s, freqs, w, pan_p=0.0):
    n = len(freqs)
    e.role[s] = orz.ROLE_ACTIVE
    e.slot_id[s] = 1000
    e.ndrive[s] = n
    e.nlive[s] = n
    e.ffreq[s, :n] = freqs
    c, sn = orz.trig_of(np.asarray(freqs), e.sr)
    e.cth[s, :n] = c
    e.sth[s, :n] = sn
    e.wcur[s, :n] = w
    e.wtgt[s, :n] = w
    L, R = orz.pan_of(pan_p * (COLS - 1), COLS)
    e.pan[s] = (L, 0.0, L, R, 0.0, R)


def excitation_path():
    freqs = [110.0, 434.66, 551.01]
    w = [0.7, 0.35, 0.2]

    def fresh(attack_ms=4.0):
        e = registry.create(EID, CTX, dict(detector=0, frequency_scale=220.0, decay_s=1.39, attack_ms=attack_ms,
                                           events=1, excitation=1))
        e.init(np.zeros((ROWS, COLS), np.uint8), None, GAIN)
        setup_slot(e, 0, freqs, w)
        return e

    def run(e, packets, n):
        out = []
        for k in range(n):
            if k in packets:
                e.inject_modes(0, packets[k])
            out.append(e.render_float(GAIN)[0])
        return np.concatenate(out)
    d1, d2 = [0.6 * 1.289, 0.0, 0.6 * 1.157], [0.4 * 0.78, 0.4 * 0.70, 0.4 * 1.38]
    y1, y2 = run(fresh(), {0: d1}, 40), run(fresh(), {6: d2}, 40)
    e12 = fresh()
    y12 = run(e12, {0: d1, 6: d2}, 40)
    out = dict(superposition_max_error=float(np.abs(y12 - (y1 + y2)).max()),
               weights_after_strikes=e12.wcur[0, :3].tolist(), weights_set=w)
    for ms in (0.0, 4.0):
        e = fresh(ms)
        e.inject_modes(0, d1)
        y = np.concatenate([e.render_float(GAIN)[0] for _ in range(int(SR / BLOCK))])[:, 0] / (orz.OUT_SCALE * GAIN)
        out[f'attack_{ms:g}'] = dict(first_sample=abs(float(y[0])), response_rms=float(np.sqrt(np.mean(y * y))),
                                     peak=float(np.abs(y).max()))
    out['attack_4_vs_0_rms_db'] = db(out['attack_4']['response_rms'] / out['attack_0']['response_rms'])
    return out


# -- 4. the scenes -----------------------------------------------------------------------------------
def run_blocks(runner, n_blocks, peaks=None):
    out = {s: [] for s in SIDES}
    times = []
    for _b in range(n_blocks):
        t0 = time.perf_counter()
        blk = runner.next_block()
        times.append((time.perf_counter() - t0) * 1000.0)
        for s in SIDES:
            out[s].append(blk.get(s))
        if peaks is not None:
            for s in SIDES:
                peaks[s] = max(peaks.get(s, 0.0), runner.sides[s].peak)
    return {s: np.concatenate(out[s]) for s in SIDES}, np.array(times)


def scene_measurements(case, seconds=SECONDS):
    doc = scene_for(case)
    n = int(math.ceil(seconds * SR / BLOCK))
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    peaks = {}
    y, times = run_blocks(runner, n, peaks)
    warm = times[int(1.0 * SR / BLOCK):]
    k0 = int(STEADY_FROM_S * SR)
    res = dict(id=case['id'], side_gain=dict(doc['audio']['side_gain']), events=dict(case['events']),
               excitation=dict(case['excitation']), seconds=seconds,
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
                      inplace_fades=d['inplace_fades'], drops=d['drops'], events=d['events_name'],
                      excitation=d['excitation_name'], position_supported=d['position_supported'],
                      unsupported=d['unsupported_packets'], zero_participation=d['zero_participation'])
    return res


# -- 5. the catalog --------------------------------------------------------------------------------
def verify_catalog(root):
    from casynth_lab.catalog import Catalog
    root = Path(root)
    if not root.is_dir():
        return dict(root=str(root), present=False)
    cat = Catalog(str(root))
    out = dict(root=str(root), present=True, records={})
    for rec, err in cat.list():
        if err is not None:
            out['records'][str(rec)] = dict(error=str(err))
            continue
        res = cat.replay(rec.id, yield_cpu=False)
        out['records'][rec.id] = dict(title=rec.title, status=res.status, reason=res.reason, pinned=rec.status,
                                      commit=rec.commit, has_notes=rec.has_notes,
                                      side_gain=(rec.meta.get('scene') or {}).get('audio', {}).get('side_gain'))
    return out


# -- the report ---------------------------------------------------------------------------------------
def write_markdown(rep, path):
    L = []
    L.append('# Objects — event source / modal excitation: measurements')
    L.append('')
    L.append(f"Generated {rep['generated']} on commit `{rep['commit']}`; numba {rep['numba']}.")
    L.append('')
    L.append('## 1. E1 — Births / Deaths against independent masks')
    L.append('')
    e1 = rep['e1']
    L.append(f"{e1['transitions']} transitions, period {e1['period']}; Both = Births + Deaths generation by generation: "
             f"{e1['both_equals_births_plus_deaths']}.")
    L.append('')
    L.append('| Events | start packet | one bank | mismatches | period sequence | periodic | packets | tails / fed | per-mode states |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for name, r in e1['runs'].items():
        L.append(f"| {name} | {r['start_packet']:g} | {r['one_bank']} | {r['packet_mismatches']} | {r['period_sequence']} | "
                 f"{r['periodic']} | {r['packets_total']} | {r['tails']} / {r['tails_fed']} | {r['per_mode_states_nonzero']} |")
    L.append('')
    m = rep['manual']
    L.append('Manual events on a still 2 x 2 block (a cell added, then removed, then the figure erased):')
    L.append('')
    L.append('| Events | start | birth, death packets | vanished: tail zf / fed modes | feeds blocks | pulse monotone | tail peak first / block 30 |')
    L.append('|---|---|---|---|---|---|---|')
    for name, r in m.items():
        v = r['vanished_tail']
        L.append(f"| {name} | {r['start_packet']:g} | {r['manual_birth_then_death_packets']} | {v['zf']:.4f} / {v['npulse']} | "
                 f"{r['tail_feeds_blocks']} | {r['tail_pulse_monotone']} | {r['tail_peak_first_block']:.4f} / {r['tail_peak_block_30']:.4f} |")
    L.append('')
    L.append('## 2. M1 — Uniform / Birth position')
    L.append('')
    m1 = rep['m1']
    L.append(f"{m1['transitions']} transitions side by side: equal {m1['equal']}; packets with events {m1['packets_with_events']}, "
             f"max |sum b^2 - m| {m1['max_norm_error']:.2e}; zero participation {m1['zero_participation']}, refused {m1['unsupported']}.")
    L.append('')
    L.append('| phase bank | measured b | preflight b | max diff | frequencies Hz |')
    L.append('|---|---|---|---|---|')
    for key, x in m1['preflight_phases'].items():
        L.append(f"| {key} | {[round(v, 4) for v in (x['measured'] or [])]} | {[round(v, 4) for v in x['preflight']]} | "
                 f"{x['max_abs_diff'] if x['max_abs_diff'] is None else f'{x['max_abs_diff']:.1e}'} | "
                 f"{[round(v, 1) for v in (x['frequencies'] or [])]} |")
    L.append('')
    lw = rep['law']
    sb = lw['single_birth']
    L.append(f"Single birth on the fixed 11-cell geometry: cells {sb['cells']} -> b {[[round(v, 4) for v in b] for b in sb['measured']]} "
             f"(preflight {[[round(v, 4) for v in b] for b in sb['preflight']]}, max diff {sb['max_abs_diff']:.1e}); "
             f"profile distance {sb['profile_distance']:.4f} (preflight {sb['preflight_distance']:.4f}).")
    L.append(f"Zero participation (no born node): b {lw['zero_participation']['b']}, total {lw['zero_participation']['total']}, "
             f"packet {lw['zero_participation']['packet']}.  Whole-figure birth: max |b - 1| {lw['whole_figure_birth']['max_abs_diff_from_one']:.1e}.")
    dg = lw['degenerate']
    L.append(f"Degenerate groups of E1 gen 1: ranks {dg['ranks']} (preflight {dg['preflight_ranks']}); sign flip max error "
             f"{dg['sign_flip_max_error']:.1e} (preflight {dg['preflight_sign']:.1e}), basis rotation {dg['basis_rotation_max_error']:.1e} "
             f"(preflight {dg['preflight_rotation']:.1e}).")
    L.append(f"Translation max error {lw['translation_max_error']}; rotation 90 deg max error {lw['rotation_max_error']:.1e}.")
    L.append('')
    L.append('## 3. The excitation path')
    L.append('')
    ex = rep['excitation']
    L.append(f"Two per-mode packets: superposition max error {ex['superposition_max_error']:.1e}; weights after the strikes "
             f"{ex['weights_after_strikes']} (set {ex['weights_set']}).  Attack 0 / 4 ms on a per-mode packet: first sample "
             f"{ex['attack_0']['first_sample']:.4f} / {ex['attack_4']['first_sample']:.4f}, response RMS change {ex['attack_4_vs_0_rms_db']:+.2f} dB.")
    L.append('')
    L.append('## 4. Scenes')
    L.append('')
    L.append('| scene | events A/B | excitation A/B | side gain B | A RMS dB (after 2 s) | B RMS dB (after 2 s) | A−B after 2 s (unit) | peak A/B | clip | continue | p99 ms (budget 7.98) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for x in rep['scenes']:
        L.append(f"| {x['id']} | {x['events']['A']}/{x['events']['B']} | {x['excitation']['A']}/{x['excitation']['B']} | {x['side_gain']['B']} | "
                 f"{x['A']['rms_db']:.2f} ({x['A']['steady_rms_db']:.2f}) | {x['B']['rms_db']:.2f} ({x['B']['steady_rms_db']:.2f}) | "
                 f"{x['steady_a_minus_b_db']:+.2f} ({x['steady_a_minus_b_db_at_unit_side_gain']:+.2f}) | "
                 f"{x['A']['peak']:.3f}/{x['B']['peak']:.3f} | {x['A']['clip_blocks']}/{x['B']['clip_blocks']} | "
                 f"{x['continue']['exact']} | {x['block_ms']['p99']:.2f} |")
    L.append('')
    for x in rep['scenes']:
        for s in SIDES:
            L.append(f"{x['id']} {s}: {x[s]['events']} / {x[s]['excitation']} (supported {x[s]['position_supported']}), figures {x[s]['figures']} "
                     f"sounding {x[s]['sounding']} tails {x[s]['tails']} faded {x[s]['evictions']} in place {x[s]['inplace_fades']} "
                     f"dropped {x[s]['drops']} refused {x[s]['unsupported']} zero {x[s]['zero_participation']}")
    L.append('')
    L.append('## 5. Catalog')
    L.append('')
    c = rep['catalog']
    if not c.get('present'):
        L.append(f"No catalog at {c['root']}.")
    else:
        for rid, x in c['records'].items():
            if 'error' in x:
                L.append(f"- {rid}: error {x['error']}")
            else:
                L.append(f"- {rid} {x['title'][:60]}: replay {x['status']} {x['reason']}, {x['pinned']} {x['commit']}, "
                         f"notes {x['has_notes']}, side gain {x['side_gain']}")
    L.append('')
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(CATALOG_ROOT))
    ap.add_argument('--quick', action='store_true', help='skip the catalog replay')
    a = ap.parse_args(argv)
    from casynth_lab import provenance as prov
    rep = dict(generated=time.strftime('%Y-%m-%d %H:%M'), commit=prov.current().get('commit'), numba=orz.HAVE_NUMBA)
    print('1. E1', flush=True)
    rep['e1'] = e1_packets()
    rep['manual'] = manual_isolation()
    print('2. M1', flush=True)
    rep['m1'] = m1_side_by_side()
    rep['law'] = law_checks()
    print('3. excitation path', flush=True)
    rep['excitation'] = excitation_path()
    print('4. scenes', flush=True)
    rep['scenes'] = [scene_measurements(c) for c in CASES]
    print('5. catalog', flush=True)
    rep['catalog'] = dict(root=a.root, present=False) if a.quick else verify_catalog(a.root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    write_markdown(rep, OUT_DIR / 'report.md')
    print(f"report: {OUT_DIR / 'report.md'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
