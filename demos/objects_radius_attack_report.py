"""Measurements for the Objects radius / attack delivery (REQ
memory/req-objects-radius-attack-2026-09-17.md, "Сдача"):

  1. the two radius pairs on the preflight fields: every packet of every bank against
     INDEPENDENT masks (births in the figure's current circle, deaths in the previous
     one), the foreign part (cells of the other family) per bank / generation, the
     distribution of packet sizes, the totals against the preflight; the joint
     evolution equals the union of the two independent evolutions; the blinker of R1
     shifted out of the x2 circle stops contributing (and x3 reaches it again)
  2. Attack: the smoothed pulse and the engine's response on the Researcher's probe
     (three resonators 110 / 440 / 880 Hz, weights 1 / 0.4 / 0.1, Decay 1.39) at
     0 / 1 / 4 / 10 / 20 ms -- first sample, peak, energy above 3 kHz, RMS change,
     max adjacent step at equal RMS -- against the preflight probe; the same on the
     A1 scene (Attack 0 / 4 ms, Radius 32)
  3. the four scenes: RMS of both sides with the side gains (<= 1 dB), peaks / clip,
     Continue from 6 s exact, the block time budget of both sides after warm-up
  4. the catalog: every record replays exactly (when built)

    python demos/objects_radius_attack_report.py [--root DIR] [--quick]

Writes demos/results/objects_radius_attack/report.json + report.md.  Stdout ASCII.
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
from demos.build_objects_radius_attack import (CASES, CATALOG_ROOT, PREFLIGHT, F0_HZ, RATE_HZ, SECONDS,   # noqa: E402
                                               scene_for, screenshot_params, preflight)

OUT_DIR = ROOT / 'demos' / 'results' / 'objects_radius_attack'
BUDGET_MS = BLOCK / SR * 1000.0
SIDES = ('A', 'B')
CTX = EngineContext(SR, BLOCK, 2, F0_HZ, 1.0, RATE_HZ)
GAIN = MASTER_GAIN * 0.7
EID = orz.ENGINE_ID
PROBE_FREQS = [110.0, 440.0, 880.0]
PROBE_WEIGHTS = [1.0, 0.4, 0.1]


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-12))


def rms_db(y):
    y = np.asarray(y, np.float64)
    if y.dtype.kind == 'i' or np.abs(y).max() > 4.0:
        y = y / 32767.0
    return db(math.sqrt(float(np.mean(y * y)))) if y.size else -240.0


def grid(cells, rows=32, cols=32):
    g = np.zeros((rows, cols), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def torus_disk(centre, radius, rows=32, cols=32):
    """An independent disk mask (cell by cell, shortest torus distance <= radius + 1e-9)."""
    m = np.zeros((rows, cols), bool)
    cy, cx = float(centre[0]), float(centre[1])
    for r in range(rows):
        dr = abs(r - cy)
        dr = min(dr, rows - dr)
        for c in range(cols):
            dc = abs(c - cx)
            dc = min(dc, cols - dc)
            m[r, c] = math.hypot(dr, dc) <= radius + 1e-9
    return m


def evolve(cells, transitions):
    g = grid(cells)
    out = [g]
    for _ in range(transitions):
        g = step(g)
        out.append(g)
    return out


def engine_generations(cells, params, transitions):
    e = registry.create(EID, CTX, params)
    g = grid(cells)
    e.init(g, None, GAIN)
    e.render_float(GAIN)
    out = [(g.copy(), e.display())]
    for _ in range(transitions):
        prev, g = g, step(g)
        e.update_field(g, events_field(prev, g))
        e.render_float(GAIN)
        out.append((g.copy(), e.display()))
    return e, out


# -- 1. the radius pairs against independent masks --------------------------------------------
def radius_case(case, muls, transitions=71):
    groups = [[(int(r), int(c)) for r, c in g] for g in preflight()['cases'][case]['groups']]
    fam = [evolve(g, transitions) for g in groups]
    joint = evolve(sum(groups, []), transitions)
    union_ok = all(not (fam[0][t] & fam[1][t]).any() and np.array_equal(joint[t], fam[0][t] | fam[1][t])
                   for t in range(transitions + 1))
    out = dict(union_of_independent_evolutions=union_ok, transitions=transitions, runs={})
    for mul in muls:
        _e, gens = engine_generations(sum(groups, []), screenshot_params(mul, 0.0), transitions)
        mismatches = 0
        foreign_sum = [0, 0]
        packets = [0, 0]
        own_sum = [0, 0]
        dist = [{}, {}]
        per_gen = []
        for t in range(1, len(gens)):
            g_prev, d_prev = gens[t - 1]
            g_cur, d_cur = gens[t]
            births = (g_cur != 0) & (g_prev == 0)
            deaths = (g_prev != 0) & (g_cur == 0)
            prev_by_id = {f['id']: f for f in d_prev['figures']}
            row = []
            for f in d_cur['figures']:
                cells = np.asarray(f['cells'])
                k = [i for i, fields in enumerate(fam) if fields[t][cells[:, 0], cells[:, 1]].all()][0]
                k_cur = torus_disk(f['centre'], f['radius'])
                own_cur, own_prev = fam[k][t] != 0, fam[k][t - 1] != 0
                e = int(np.count_nonzero(births & k_cur))
                foreign = int(np.count_nonzero(births & k_cur & ~own_cur))
                p = prev_by_id.get(f['id'])
                if p is not None:
                    k_prev = torus_disk(p['centre'], p['radius'])
                    e += int(np.count_nonzero(deaths & k_prev))
                    foreign += int(np.count_nonzero(deaths & k_prev & ~own_prev))
                if f['slot'] >= 0:
                    if float(f['e']) != float(e):
                        mismatches += 1
                    foreign_sum[k] += foreign
                    own_sum[k] += e - foreign
                    packets[k] += 1 if e > 0 else 0
                    dist[k][e] = dist[k].get(e, 0) + 1
                row.append(dict(id=f['id'], family=k, n=f['n'], e=float(f['e']), expected=e, foreign=foreign,
                                covers_all=bool(f.get('covers_all'))))
            if t <= 16:
                per_gen.append(dict(gen=t, figures=row))
        pf = preflight()['cases'][case]['runs'].get(f'{mul}', {})
        out['runs'][f'{mul}'] = dict(
            packet_mismatches=mismatches, foreign_events_sum=foreign_sum, own_events_sum=own_sum, packets=packets,
            packet_size_distribution=[{str(k): v for k, v in sorted(d.items())} for d in dist],
            preflight=dict(foreign_events_sum=pf.get('foreign_events_sum'), packets=pf.get('packets')),
            matches_preflight=(pf.get('foreign_events_sum') == foreign_sum and pf.get('packets') == packets),
            first_generations=per_gen)
    return out


def r1_shift():
    """The blinker of R1 moved 8 columns to the right: outside the x2 circle of the
    receiver (13.3 .. 14.1 cells from its centre vs R_eff 10.66), inside x3 (16.0)."""
    recv, blink = [[(int(r), int(c)) for r, c in g] for g in preflight()['cases']['receiver']['groups']]
    far = [(r, c + 8) for r, c in blink]
    out = {}
    for mul in (2.0, 3.0):
        _e, gens = engine_generations(recv + far, screenshot_params(mul, 0.0), 12)
        recv_e = sorted({float(f['e']) for _g, d in gens[1:] for f in d['figures'] if f['n'] == 17})
        blink_e = sorted({float(f['e']) for _g, d in gens[1:] for f in d['figures'] if f['n'] == 3})
        d = gens[1][1]
        f17 = [f for f in d['figures'] if f['n'] == 17][0]
        out[f'{mul}'] = dict(receiver_e_values=recv_e, blinker_e_values=blink_e, receiver_R_eff=f17['radius'],
                             receiver_centre=f17['centre'])
    return out


# -- 2. Attack ----------------------------------------------------------------------------------
def pulse_and_smoothing(attack_ms, a=0.6, seconds=0.05):
    """The N2 pulse of one packet `a` and its smoothed version u (direct formulas)."""
    n = int(seconds * SR)
    qf, qs = math.exp(-1.0 / (SR * 0.00025)), math.exp(-1.0 / (SR * 0.002))
    zf = zs = a
    q = orz.attack_q(attack_ms, SR)
    p = np.zeros(n)
    u = np.zeros(n)
    uk = 0.0
    for k in range(n):
        p[k] = 0.75 * ((1.0 - qf) * zf - (1.0 - qs) * zs) / (qs - qf)
        zf *= qf
        zs *= qs
        uk = q * uk + (1.0 - q) * p[k]
        u[k] = uk
    return p, u


def hf_energy(y, f_lo=3000.0):
    spec = np.abs(np.fft.rfft(np.asarray(y, np.float64))) ** 2
    f = np.fft.rfftfreq(len(y), 1.0 / SR)
    return float(spec[f > f_lo].sum())


def probe_engine(attack_ms, decay_s=1.39):
    e = registry.create(EID, CTX, dict(detector=1, frequency_scale=220.0, decay_s=decay_s, attack_ms=attack_ms))
    e.init(np.zeros((32, 32), np.uint8), None, GAIN)
    s = 0
    n = len(PROBE_FREQS)
    e.role[s] = orz.ROLE_ACTIVE
    e.slot_id[s] = 1000
    e.ndrive[s] = n
    e.nlive[s] = n
    e.ffreq[s, :n] = PROBE_FREQS
    c, sn = orz.trig_of(np.asarray(PROBE_FREQS), e.sr)
    e.cth[s, :n] = c
    e.sth[s, :n] = sn
    e.wcur[s, :n] = PROBE_WEIGHTS
    e.wtgt[s, :n] = PROBE_WEIGHTS
    e.pan[s] = (1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    return e


def attack_probe():
    pf = preflight()['attack_proposal_probe']['results']
    out = dict(law='u = q u + (1 - q) p, q = exp(-ln 9 / (SR attack_ms / 1000)), 0 at attack 0',
               frequencies_hz=PROBE_FREQS, weights=PROBE_WEIGHTS, decay_s=1.39, results={})
    base = None
    for ms in (0.0, 1.0, 4.0, 10.0, 20.0):
        p, u = pulse_and_smoothing(ms)
        e = probe_engine(ms)
        e.inject(0, 0.6)
        blocks = [e.render_float(GAIN)[0][:, 0] for _ in range(int(1.0 * SR / BLOCK))]
        y = np.concatenate(blocks) / (orz.OUT_SCALE * GAIN)
        row = dict(q=orz.attack_q(ms, SR), pulse_first_sample=float(u[0]), pulse_peak=float(u.max()),
                   pulse_hf_energy_above_3khz=hf_energy(u), response_rms=float(np.sqrt(np.mean(y * y))),
                   response_peak=float(np.abs(y).max()), response_max_adjacent_step=float(np.abs(np.diff(y)).max()))
        if base is None:
            base = row
        row['hf_energy_change_db'] = 10.0 * math.log10(row['pulse_hf_energy_above_3khz'] / base['pulse_hf_energy_above_3khz'])
        row['rms_change_db'] = db(row['response_rms'] / base['response_rms'])
        row['adjacent_step_at_equal_rms_vs_zero_db'] = db((row['response_max_adjacent_step'] / row['response_rms'])
                                                         / (base['response_max_adjacent_step'] / base['response_rms']))
        ref = pf.get(f'{ms}', {})
        row['preflight'] = {k: ref.get(k) for k in ('q', 'hf_energy_change_db', 'rms_change_db',
                                                    'adjacent_step_at_equal_rms_vs_old_db')}
        out['results'][f'{ms}'] = row
    return out


def a1_scene_attack():
    """The A1 field at Attack 0 / 4 ms (unit side gains): RMS, energy above 3 kHz,
    max adjacent step (raw and at equal RMS), the packet moments identical."""
    case = [c for c in CASES if c['id'] == 'ora_a1'][0]
    doc = scene_for(case, side_gain=dict(A=1.0, B=1.0))
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    n = int(math.ceil(SECONDS * SR / BLOCK))
    y = {s: [] for s in SIDES}
    same_packets = True
    for _ in range(n):
        blk = runner.next_block()
        for s in SIDES:
            y[s].append(blk.get(s))
        ea, eb = runner.sides['A'].engine, runner.sides['B'].engine
        same_packets = same_packets and np.array_equal(ea.last_e, eb.last_e) and np.array_equal(ea.zf, eb.zf)
    y = {s: np.concatenate(y[s])[:, 0].astype(np.float64) / 32767.0 for s in SIDES}
    out = dict(packet_moments_identical=bool(same_packets))
    for s in SIDES:
        out[s] = dict(rms_db=rms_db(y[s]), hf_energy_above_3khz=hf_energy(y[s]),
                      max_adjacent_step=float(np.abs(np.diff(y[s])).max()), peak=float(np.abs(y[s]).max()))
    out['b_minus_a_rms_db'] = out['B']['rms_db'] - out['A']['rms_db']
    out['b_minus_a_hf_db'] = 10.0 * math.log10(out['B']['hf_energy_above_3khz'] / out['A']['hf_energy_above_3khz'])
    out['b_vs_a_adjacent_step_at_equal_rms_db'] = db((out['B']['max_adjacent_step'] / 10 ** (out['B']['rms_db'] / 20))
                                                     / (out['A']['max_adjacent_step'] / 10 ** (out['A']['rms_db'] / 20)))
    return out


# -- 3. the scenes ------------------------------------------------------------------------------
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
    res = dict(id=case['id'], side_gain=dict(doc['audio']['side_gain']), radius=dict(case['radius']),
               attack=dict(case['attack']), range=list(case['range']),
               ranges_in_scene=doc['param_ranges'], seconds=seconds,
               block_ms=dict(p50=float(np.percentile(warm, 50)), p99=float(np.percentile(warm, 99)),
                             max=float(warm.max()), budget=BUDGET_MS, ok=bool(np.percentile(warm, 99) < BUDGET_MS)))
    for s in SIDES:
        res[s] = dict(rms_db=rms_db(y[s]), peak=peaks[s], clip_blocks=int(runner.sides[s].clip_blocks),
                      finite=bool(np.all(np.isfinite(y[s]))))
    res['a_minus_b_db'] = res['A']['rms_db'] - res['B']['rms_db']
    res['within_1db'] = bool(abs(res['a_minus_b_db']) <= 1.0)
    doc1 = json.loads(json.dumps(doc))
    doc1['audio']['side_gain'] = dict(A=1.0, B=1.0)
    r1 = DemoRunner(scene_from_doc(doc1))
    r1.post('start', at=0)
    y1, _t = run_blocks(r1, n)
    res['a_minus_b_db_at_unit_side_gain'] = rms_db(y1['A']) - rms_db(y1['B'])
    r2 = DemoRunner(scene_from_doc(doc))
    r2.post('start', at=0)
    for _ in range(n // 2):
        r2.next_block()
    st = r2.export_state()
    twin = DemoRunner.from_state(st)
    exact = twin.side_ranges() == r2.side_ranges()
    gen0 = twin.gen
    for _ in range(200):
        a, b = r2.next_block(), twin.next_block()
        for s in ('A', 'B', 'monitor'):
            exact = exact and bool(np.array_equal(a.get(s), b.get(s)))
    res['continue'] = dict(exact=bool(exact), field_changed=bool(twin.gen > gen0), from_seconds=seconds / 2, blocks=200)
    for s in SIDES:
        d = runner.snapshot()['display'][s]
        res[s].update(figures=d['n_figures'], sounding=d['n_sounding'], tails=d['n_tails'], evictions=d['evictions'],
                      inplace_fades=d['inplace_fades'], drops=d['drops'], covers_all=sum(1 for f in d['figures'] if f['covers_all']),
                      attack_ms=d['attack_ms'])
    return res


# -- 4. the catalog -----------------------------------------------------------------------------
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
                                      side_gain=(rec.meta.get('scene') or {}).get('audio', {}).get('side_gain'),
                                      param_ranges=(rec.meta.get('scene') or {}).get('param_ranges'))
    return out


# -- the report ---------------------------------------------------------------------------------
def write_markdown(rep, path):
    L = []
    L.append('# Objects — radius range / attack: measurements')
    L.append('')
    L.append(f"Generated {rep['generated']} on commit `{rep['commit']}`; numba {rep['numba']}.")
    L.append('')
    L.append('## 1. Radius pairs against independent masks')
    L.append('')
    for case, r in rep['radius'].items():
        L.append(f"**{case}** — joint evolution = union of the independent ones: {r['union_of_independent_evolutions']}, "
                 f"{r['transitions']} transitions.")
        L.append('')
        L.append('| Radius x | packet mismatches | foreign events (fam 0 / 1) | own events | packets | preflight foreign / packets | match |')
        L.append('|---|---|---|---|---|---|---|')
        for mul, x in r['runs'].items():
            L.append(f"| {mul} | {x['packet_mismatches']} | {x['foreign_events_sum']} | {x['own_events_sum']} | {x['packets']} | "
                     f"{x['preflight']['foreign_events_sum']} / {x['preflight']['packets']} | {x['matches_preflight']} |")
        L.append('')
        for mul, x in r['runs'].items():
            L.append(f"Packet sizes at x{mul}: family 0 {x['packet_size_distribution'][0]}, family 1 {x['packet_size_distribution'][1]}")
        L.append('')
    s = rep['r1_shift']
    L.append(f"R1 blinker shifted 8 columns right: at x2 receiver e values {s['2.0']['receiver_e_values']} "
             f"(R_eff {s['2.0']['receiver_R_eff']:.2f}), at x3 {s['3.0']['receiver_e_values']} (R_eff {s['3.0']['receiver_R_eff']:.2f}); "
             f"the blinker's own e {s['2.0']['blinker_e_values']} / {s['3.0']['blinker_e_values']}.")
    L.append('')
    L.append('## 2. Attack')
    L.append('')
    L.append('| Attack ms | q | pulse first sample | pulse peak | HF > 3 kHz change dB | response RMS change dB | adjacent step at equal RMS dB | preflight HF / RMS / step dB |')
    L.append('|---|---|---|---|---|---|---|---|')
    for ms, x in rep['attack_probe']['results'].items():
        p = x['preflight']
        L.append(f"| {ms} | {x['q']:.6f} | {x['pulse_first_sample']:.4f} | {x['pulse_peak']:.4f} | {x['hf_energy_change_db']:+.2f} | "
                 f"{x['rms_change_db']:+.2f} | {x['adjacent_step_at_equal_rms_vs_zero_db']:+.2f} | "
                 f"{p.get('hf_energy_change_db', float('nan')):+.2f} / {p.get('rms_change_db', float('nan')):+.2f} / "
                 f"{p.get('adjacent_step_at_equal_rms_vs_old_db', float('nan')):+.2f} |")
    L.append('')
    a = rep['a1_scene']
    L.append(f"A1 field (Radius 32, unit side gains): packet moments identical {a['packet_moments_identical']}; "
             f"B − A RMS {a['b_minus_a_rms_db']:+.2f} dB, energy above 3 kHz {a['b_minus_a_hf_db']:+.2f} dB, "
             f"max adjacent step at equal RMS {a['b_vs_a_adjacent_step_at_equal_rms_db']:+.2f} dB; "
             f"peaks A {a['A']['peak']:.3f} B {a['B']['peak']:.3f}.")
    L.append('')
    L.append('## 3. Scenes')
    L.append('')
    L.append('| scene | radius A/B | attack A/B | range | side gain B | A RMS dB | B RMS dB | A−B dB (unit) | peak A/B | clip | continue | p99 ms (budget 7.98) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for x in rep['scenes']:
        L.append(f"| {x['id']} | {x['radius']['A']}/{x['radius']['B']} | {x['attack']['A']}/{x['attack']['B']} | {x['range']} | "
                 f"{x['side_gain']['B']} | {x['A']['rms_db']:.2f} | {x['B']['rms_db']:.2f} | {x['a_minus_b_db']:+.2f} "
                 f"({x['a_minus_b_db_at_unit_side_gain']:+.2f}) | {x['A']['peak']:.3f}/{x['B']['peak']:.3f} | "
                 f"{x['A']['clip_blocks']}/{x['B']['clip_blocks']} | {x['continue']['exact']} | {x['block_ms']['p99']:.2f} |")
    L.append('')
    for x in rep['scenes']:
        L.append(f"{x['id']}: B figures {x['B']['figures']} sounding {x['B']['sounding']} tails {x['B']['tails']} "
                 f"covers_all {x['B']['covers_all']} faded {x['B']['evictions']} in place {x['B']['inplace_fades']} dropped {x['B']['drops']}")
    L.append('')
    L.append('## 4. Catalog')
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
                         f"notes {x['has_notes']}, side gain {x['side_gain']}, ranges {x['param_ranges']}")
    L.append('')
    path.write_text('\n'.join(L) + '\n', encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(CATALOG_ROOT))
    ap.add_argument('--quick', action='store_true', help='skip the catalog replay')
    a = ap.parse_args(argv)
    from casynth_lab import provenance as prov
    rep = dict(generated=time.strftime('%Y-%m-%d %H:%M'), commit=prov.current().get('commit'), numba=orz.HAVE_NUMBA)
    print('1. radius pairs', flush=True)
    rep['radius'] = dict(receiver=radius_case('receiver', (1.0, 2.0)), p3_p5=radius_case('p3_p5', (1.0, 8.0, 32.0)))
    rep['r1_shift'] = r1_shift()
    print('2. attack', flush=True)
    rep['attack_probe'] = attack_probe()
    rep['a1_scene'] = a1_scene_attack()
    print('3. scenes', flush=True)
    rep['scenes'] = [scene_measurements(c) for c in CASES]
    print('4. catalog', flush=True)
    rep['catalog'] = dict(root=a.root, present=False) if a.quick else verify_catalog(a.root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    write_markdown(rep, OUT_DIR / 'report.md')
    print(f"report: {OUT_DIR / 'report.md'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
