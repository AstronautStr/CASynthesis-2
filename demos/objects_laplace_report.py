"""Objects / Laplace hand-over measurements (REQ memory/req-objects-laplace-comparison-2026-09-17.md,
"Уровни и сдача") -> demos/results/objects_laplace/report.{json,md}.

    python demos/objects_laplace_report.py [--catalog ROOT] [--no-catalog]

No audio device.  Technical evidence, not listening verdicts: the equality of the
spectral data of A (the old Laplace) and B (Objects / Laplace) on every component of
every generation of the three scenes for the seven settings; the effect of each setting
on the galaxy; the L3 receiver events at Radius x 1 / 1.5 and the live lever; no packet
from any setting or from Restore; the life of the L2 voices (4 -> 2 -> 4 rule, tails,
no drops); the levels of the ready 12 s (A / B RMS after the scene's side gain, peaks,
clip); Replay / Continue exactness; the block budget of both sides after warm-up; the
catalog check; stress probes and limits kept apart.  The scalar reference of the sample
path and the snapshot contract are the gates of tests/test_n4_object_resonators.py and
tests/test_objects_laplace.py.
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
from casynth_core import map_laplacian, extract, PATCH_SIZE                  # noqa: E402
from casynth_engine import step, events_field, _crop_like_extract            # noqa: E402
from casynth_lab import DemoRunner, scene_from_doc, BLOCK, registry          # noqa: E402
from casynth_lab.catalog import Catalog                                      # noqa: E402
from casynth_lab.engine_api import EngineContext                             # noqa: E402
from casynth_lab import figures as fg                                        # noqa: E402
from casynth_lab import object_resonators as orz                             # noqa: E402
from demos.build_objects_laplace import (CASES, CATALOG_ROOT, scene_for, PREFLIGHT,   # noqa: E402
                                         SPECTRUM, objects_side, F0_HZ)

OUT_DIR = ROOT / 'demos' / 'results' / 'objects_laplace'
BUDGET_MS = BLOCK / SR * 1000.0
SIDES = ('A', 'B')
CTX = EngineContext(SR, BLOCK, 2, F0_HZ, 1.0, 6.0)
GAIN = MASTER_GAIN * 0.7
SETTING_SETS = [dict(SPECTRUM),
                dict(n=4, spread=1.0, alpha=2.0, shape=1.0, harm=1.0, fullshape=0, dyn=1.0),
                dict(n=20, spread=0.5, alpha=0.5, shape=0.6, harm=0.3, fullshape=1, dyn=0.7)]


def db(x):
    return 20.0 * math.log10(max(float(x), 1e-12))


def rms_db(y):
    y = np.asarray(y, np.float64) / 32767.0
    return db(math.sqrt(float(np.mean(y * y)))) if y.size else -240.0


def grid(cells):
    g = np.zeros((32, 32), np.uint8)
    for r, c in cells:
        g[r, c] = 1
    return g


def run_blocks(runner, n_blocks, on_block=None, peaks=None):
    """Drive the runner; returns ({side: pcm}, block times ms)."""
    out = {s: [] for s in SIDES}
    times = []
    for b in range(n_blocks):
        if on_block is not None:
            on_block(b, runner)
        t0 = time.perf_counter()
        blk = runner.next_block()
        times.append((time.perf_counter() - t0) * 1000.0)
        for s in SIDES:
            out[s].append(blk.get(s))
        if peaks is not None:
            for s in SIDES:
                peaks[s] = max(peaks.get(s, 0.0), runner.sides[s].peak)
    return {s: np.concatenate(out[s]) for s in SIDES}, np.array(times)


def old_laplace_on_bbox(cells, sets, exc):
    cells = np.asarray(cells)
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), np.uint8)
    sub[cells[:, 0] - r0, cells[:, 1] - c0] = 1
    if sets['fullshape']:
        patch = sub
        ex = None if exc is None else exc[r0:r1 + 1, c0:c1 + 1] * sub
    else:
        patch = extract(sub, PATCH_SIZE)
        ex = None if exc is None else _crop_like_extract(exc[r0:r1 + 1, c0:c1 + 1] * sub, sub, PATCH_SIZE)
    fr, am = map_laplacian(patch, F0_HZ, sets['n'], sets['spread'], sets['alpha'], sets['shape'], sets['harm'],
                           bool(sets['fullshape']), sets['dyn'], ex)
    num = int(np.count_nonzero(fr))
    return fr[:num], am[:num]


# -- 1. the spectral data of A and B ------------------------------------------------------
def spectral_equality():
    """Every component of every one of the 73 states of the three scenes, three setting
    sets: Objects / Laplace (laplace_modes_of) against map_laplacian on the object's
    bounding box (the old analyse() call)."""
    out = []
    for case in CASES:
        g = grid(case['cells']())
        exc = None
        worst = 0.0
        n_comp = 0
        n_modes = 0
        for gen in range(73):
            for sets in SETTING_SETS:
                for cells in fg.components(g):
                    f, a = orz.laplace_modes_of(cells, 32, 32, F0_HZ, sets, exc)
                    fr, am = old_laplace_on_bbox(cells, sets, exc)
                    assert len(f) == len(fr) and len(a) == len(am)
                    if len(f):
                        worst = max(worst, float(np.abs(f - fr).max()), float(np.abs(a - am).max()))
                    n_comp += 1
                    n_modes += len(f)
            prev, g = g, step(g)
            exc = events_field(prev, g)
        out.append(dict(id=case['id'], states=73, setting_sets=len(SETTING_SETS), components_compared=n_comp,
                        modes_compared=n_modes, worst_abs_diff=worst, equal_within_1e10=bool(worst <= 1e-10)))
    return out


def settings_effect():
    """Each of the seven settings against the REQ table on the galaxy's period (0..7):
    the generations in which the spectral data of at least one component differ."""
    pf = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    g = grid(CASES[1]['cells']())
    fields = [(g, None)]
    for _ in range(7):
        prev, g = g, step(g)
        fields.append((g, events_field(prev, g)))
    base = dict(SPECTRUM)
    out = {}
    for k in orz.SPECTRUM_KEYS:
        v = pf['galaxy_controls_over_period'][k]['value']
        alt = dict(base, **{k: v})
        ref = base
        if k == 'dyn':
            alt['shape'] = 1.0
            ref = dict(base, shape=1.0)
        gens = []
        for gen, (gg, exc) in enumerate(fields):
            differs = False
            for cells in fg.components(gg):
                f0_, a0 = orz.laplace_modes_of(cells, 32, 32, F0_HZ, ref, exc)
                f1, a1 = orz.laplace_modes_of(cells, 32, 32, F0_HZ, alt, exc)
                if len(f0_) != len(f1) or not np.array_equal(f0_, f1) or not np.array_equal(a0, a1):
                    differs = True
            if differs:
                gens.append(gen)
        out[k] = dict(value=v, changed_generations=gens,
                      preflight_generations=pf['galaxy_controls_over_period'][k]['changed_generations'])
    return out


# -- 2. the engine: packets, L3 events, tails -----------------------------------------------
def engine_run(cells, params, seconds, on_gen=None, gain=GAIN, exc0=None):
    e = registry.create(orz.ENGINE_ID, CTX, params)
    g = grid(cells)
    e.init(g, exc0, gain)
    n_blocks = int(math.ceil(seconds * SR / BLOCK))
    step_s = SR / 6.0
    ca, gen = 0, 0
    peak = 0.0
    for b in range(n_blocks):
        if ca >= (gen + 1) * step_s:
            prev, g = g, step(g)
            gen += 1
            e.update_field(g, events_field(prev, g))
        y, pk, _nc = e.render_float(gain)
        peak = max(peak, pk)
        if on_gen is not None:
            on_gen(b, gen, e)
        ca += BLOCK
    return e, peak


def no_packet_from_settings_and_restore():
    """Every setting of side B (the seven, spectrum, detector, radius, decay) changed on a
    running field: the pulse states do not move; export / restore: the next blocks equal."""
    cells = CASES[2]['cells']()
    e, _ = engine_run(cells, objects_side(1.5)[1], 1.0)
    rows = []
    for name, value in (('n', 5), ('spread', 0.8), ('alpha', 1.7), ('shape', 0.6), ('harm', 0.9),
                        ('fullshape', 0), ('dyn', 0.5), ('spectrum', 0), ('spectrum', 1), ('detector', 0),
                        ('radius_mul', 1.0), ('decay_s', 1.2)):
        zf, zs = e.zf.copy(), e.zs.copy()
        ff = e.ffreq.copy()
        e.set_params(dict(e.params, **{name: value}))
        moved = bool(not np.array_equal(zf, e.zf) or not np.array_equal(zs, e.zs))
        retuned = bool(not np.array_equal(ff, e.ffreq))
        ramp = int(e.wleft[:orz.N_ACTIVE].max())
        y, _pk, _nc = e.render_float(GAIN)
        rows.append(dict(setting=name, value=value, pulse_moved=moved, frequencies_changed=retuned,
                         weight_ramp_samples=ramp, finite=bool(np.all(np.isfinite(y)))))
    st = e.export_state()
    other = registry.create(orz.ENGINE_ID, CTX, dict(e.params))
    other.init(np.zeros((32, 32), np.uint8), None, 0.0)
    other.restore_state(e._grid, e._exc, st)
    same = True
    for _ in range(40):
        a = e.render_float(GAIN)[0]
        b = other.render_float(GAIN)[0]
        same = same and bool(np.array_equal(a, b))
    return dict(settings=rows, restore_exact=same)


def l3_events():
    """The receiver's packets per generation at Radius x 1 and 1.5, the blinker's, and the
    live lever 1.5 -> 1.0 (a 17-cell figure + a blinker three columns right)."""
    cells = CASES[2]['cells']()
    out = {}
    for mul in (1.0, 1.5):
        per_gen = {}

        def on_gen(b, gen, e, per_gen=per_gen):
            d = {f['n']: f for f in e.display()['figures']}
            per_gen.setdefault(gen, {17: d[17]['e'], 3: d[3]['e']})
        e, _ = engine_run(cells, objects_side(mul)[1], 12.0, on_gen)
        gens = sorted(per_gen)[1:]
        d = {f['n']: f for f in e.display()['figures']}
        out[f'radius_{mul}'] = dict(generations=len(gens),
                                    receiver_e=sorted(set(per_gen[g][17] for g in gens)),
                                    blinker_e=sorted(set(per_gen[g][3] for g in gens)),
                                    receiver_centre=d[17]['centre'], receiver_R=d[17]['radius_geom'],
                                    receiver_R_eff=d[17]['radius'], ids=sorted(f['id'] for f in e.display()['figures']),
                                    receiver_modes=d[17]['modes'], receiver_f_low=d[17]['f_low'],
                                    blinker_modes=d[3]['modes'], blinker_f_low=d[3]['f_low'])
    # the live lever (the next generation of the ENGINE's own field)
    e, _ = engine_run(cells, objects_side(1.5)[1], 1.0)
    g = e._grid.copy()
    zf = e.zf.copy()
    e.set_params(dict(e.params, radius_mul=1.0))
    lever_no_packet = bool(np.array_equal(zf, e.zf))
    e.update_field(step(g), None)
    e.render_float(GAIN)
    d = {f['n']: f for f in e.display()['figures']}
    out['lever_1p5_to_1'] = dict(no_packet_from_the_knob=lever_no_packet, receiver_e_next=d[17]['e'],
                                 blinker_e_next=d[3]['e'])
    return out


def l2_voices():
    """The galaxy on side B for 12 s: identities, sounding figures per generation, tails,
    counters; the 4 -> 2 -> 4 review case on the Objects / Laplace engine (n as the lever)."""
    cells = CASES[1]['cells']()
    per_gen = {}

    def on_gen(b, gen, e):
        d = e.display()
        per_gen.setdefault(gen, dict(figures=d['n_figures'], sounding=d['n_sounding'], tails=d['n_tails'],
                                     sizes=sorted(f['n'] for f in d['figures'])))
    e, peak = engine_run(cells, objects_side(1.0)[1], 12.0, on_gen)
    d = e.display()
    max_tails = max(v['tails'] for v in per_gen.values())
    # the review's case on this engine: a 5-cell figure 4 -> 2 -> 4 modes (Figure law)
    # and 12 -> 6 -> 12 (Laplace law, n): the returning modes start from zero
    five = [(9, 14), (9, 15), (9, 16), (10, 15), (10, 16)]
    cases = {}
    for law, lever in (('Figure', None), ('Laplace', 'n')):
        params = dict(objects_side(1.0)[1], spectrum=(orz.SPEC_FIGURE if law == 'Figure' else orz.SPEC_LAPLACE))
        ee = registry.create(orz.ENGINE_ID, CTX, params)
        ee.init(grid(five), None, GAIN)
        for _ in range(12):
            ee.render_float(GAIN)
        s = ee.display()['figures'][0]['slot']
        n0 = int(ee.ndrive[s])
        if lever is None:
            ee.update_field(grid(five[:3]), None)
        else:
            ee.set_params(dict(ee.params, n=2))
        ee.render_float(GAIN)
        n1 = int(ee.ndrive[s])
        tails = [t for t in range(orz.N_SLOTS) if ee.role[t] == orz.ROLE_TAIL]
        if lever is None:
            ee.update_field(grid(five), None)
        else:
            ee.set_params(dict(ee.params, n=n0))
        ee._boundary()
        n2 = int(ee.ndrive[s])
        cases[law] = dict(modes=[n0, n1, n2], returning_modes_max_abs_state=float(np.abs(ee.zre[s, n1:n2]).max()),
                          tail_slots=len(tails), tail_undriven=bool(all(ee.ndrive[t] == 0 for t in tails)),
                          tail_pulse=float(max((abs(ee.zf[t]) for t in tails), default=0.0)),
                          id_kept=ee.display()['figures'][0]['id'] == 1)
    return dict(generations=len(per_gen), ids_seen=int(e.next_id - 1), max_figures=max(v['figures'] for v in per_gen.values()),
                max_sounding=max(v['sounding'] for v in per_gen.values()), max_tails=max_tails,
                evictions=d['evictions'], inplace_fades=d['inplace_fades'], drops=d['drops'],
                unvoiced_blocks=d['unvoiced_blocks'], peak_pre_gain_scale=peak, mode_return=cases,
                per_generation={str(k): v for k, v in sorted(per_gen.items())[:9]})


# -- 3. the scenes: levels, continuation, timing --------------------------------------------
def scene_measurements(case):
    doc = scene_for(case)
    n = int(math.ceil(12.0 * SR / BLOCK))
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    peaks = {}
    y, times = run_blocks(runner, n, None, peaks)
    res = dict(id=case['id'], side_gain=dict(doc['audio']['side_gain']),
               engines={s: runner.sides[s].engine_id for s in SIDES},
               block_ms=dict(p50=float(np.percentile(times[125:], 50)), p99=float(np.percentile(times[125:], 99)),
                             max=float(times[125:].max()), budget=BUDGET_MS,
                             ok=bool(np.percentile(times[125:], 99) < BUDGET_MS)))
    for s in SIDES:
        res[s] = dict(rms_db=rms_db(y[s]), peak=peaks[s], clip_blocks=int(runner.sides[s].clip_blocks),
                      finite=bool(np.all(np.isfinite(y[s]))))
    res['a_minus_b_db'] = res['A']['rms_db'] - res['B']['rms_db']
    res['within_1db'] = bool(abs(res['a_minus_b_db']) <= 1.0)
    # the same scene without the side gain: what the calibration corrected
    doc1 = json.loads(json.dumps(doc))
    doc1['audio']['side_gain'] = dict(A=1.0, B=1.0)
    r1 = DemoRunner(scene_from_doc(doc1))
    r1.post('start', at=0)
    y1, _t = run_blocks(r1, n)
    res['a_minus_b_db_at_unit_side_gain'] = rms_db(y1['A']) - rms_db(y1['B'])
    # continuation from the middle: export at 6 s, the next 200 blocks equal
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
    res['continue'] = dict(exact=exact, field_changed=bool(twin.gen > gen0), from_seconds=6.0, blocks=200)
    d = runner.snapshot()['display']['B']
    res['B'].update(spectrum=d['spectrum_name'], f0=d['f0'], figures=d['n_figures'], tails=d['n_tails'],
                    evictions=d['evictions'], inplace_fades=d['inplace_fades'], drops=d['drops'])
    return res


def timing(seconds=8.0):
    """p50 / p95 / p99 / max block time of both sides on each scene after 1 s of warm-up."""
    out = {}
    for case in CASES:
        runner = DemoRunner(scene_from_doc(scene_for(case)))
        runner.post('start', at=0)
        _y, times = run_blocks(runner, int(seconds * SR / BLOCK))
        warm = times[int(1.0 * SR / BLOCK):]
        out[case['id']] = dict(p50=float(np.percentile(warm, 50)), p95=float(np.percentile(warm, 95)),
                               p99=float(np.percentile(warm, 99)), max=float(warm.max()), budget=BUDGET_MS,
                               ok=bool(np.percentile(warm, 99) < BUDGET_MS))
    out['numba'] = orz.HAVE_NUMBA
    return out


# -- 4. stress (apart from the scenes) --------------------------------------------------------
def stress():
    rng = np.random.default_rng(20260917)
    rows = []
    n = int(math.ceil(6.0 * SR / BLOCK))
    for mode in ('dense_random_evolving', 'random_every_block', 'full_toggle_every_block', 'knobs_moving'):
        for sets in (dict(SPECTRUM), dict(n=20, spread=1.0, alpha=0.0, shape=1.0, harm=0.0, fullshape=1, dyn=1.0)):
            doc = scene_for(CASES[1])
            runner = DemoRunner(scene_from_doc(doc), vol=1.0)
            for k, v in sets.items():
                for s in SIDES:
                    runner.post('set_param', side=s, name=k, value=v)
            if mode == 'dense_random_evolving':
                g = (rng.random((32, 32)) < 0.5).astype(np.uint8)
                runner.post('set_cells', cells=[[r, c, int(g[r, c])] for r in range(32) for c in range(32)])
            runner.post('start', at=0)

            def on_block(b, r, mode=mode):
                if mode == 'random_every_block':
                    g = (rng.random((32, 32)) < 0.5).astype(np.uint8)
                    r.post('set_cells', cells=[[rr, c, int(g[rr, c])] for rr in range(32) for c in range(32)])
                elif mode == 'full_toggle_every_block':
                    r.post('set_cells', cells=[[rr, c, int(1 - r.grid[rr, c])] for rr in range(32) for c in range(32)])
                elif mode == 'knobs_moving' and b % 12 == 0:
                    k = b // 12
                    for s in SIDES:
                        r.post('set_param', side=s, name='n', value=(3 if k % 2 else 20))
                        r.post('set_param', side=s, name='spread', value=(1.0 if k % 3 else 0.0))
                        r.post('set_param', side=s, name='shape', value=(1.0 if k % 2 else 0.0))
                        r.post('set_param', side=s, name='harm', value=(1.0 if k % 4 else 0.0))
                        r.post('set_param', side=s, name='fullshape', value=(k // 2) % 2)
                    r.post('set_param', side='B', name='decay_s', value=(0.2 if k % 2 else 1.5))
                    r.post('set_param', side='B', name='radius_mul', value=(4.0 if k % 2 else 0.25))
            peaks = {}
            y, times = run_blocks(runner, n, on_block, peaks)
            row = dict(mode=mode, settings=sets, gain=runner._gain(),
                       block_ms_p99=float(np.percentile(times[50:], 99)), block_ms_max=float(times[50:].max()))
            for s in SIDES:
                st = dict(pre_clip_peak=peaks[s], clip_blocks=int(runner.sides[s].clip_blocks),
                          finite=bool(np.all(np.isfinite(y[s]))), rms_db=rms_db(y[s][int(0.5 * SR):]))
                if s == 'B':
                    d = runner.snapshot()['display'][s]
                    st.update(n_figures=d['n_figures'], n_sounding=d['n_sounding'], n_tails=d['n_tails'],
                              evictions=d['evictions'], inplace_fades=d['inplace_fades'], drops=d['drops'],
                              unvoiced_blocks=d['unvoiced_blocks'])
                row[s] = st
            rows.append(row)
    return rows


def analysis_cost():
    """Cost of the Laplace law on one component by size (shape 0: eigvalsh; shape 1: eigh),
    the full torus graph without the old node ceiling (no decimation)."""
    out = []
    for side in (8, 16, 24, 32):
        cells = np.array([(r, c) for r in range(side) for c in range(side)])
        for shape in (0.0, 1.0):
            sets = dict(SPECTRUM, shape=shape)
            t0 = time.perf_counter()
            orz.laplace_modes_of(cells, 32, 32, F0_HZ, sets)
            ms = (time.perf_counter() - t0) * 1000.0
            out.append(dict(cells=int(side * side), shape=shape, ms=ms, over_budget=bool(ms > BUDGET_MS)))
    return out


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


# -- report -------------------------------------------------------------------------------------
def write_markdown(rep, path):
    L = ['# Objects / Laplace -- hand-over measurements (technical evidence, not listening verdicts)', '',
         f"Generated {rep['generated']} at commit {rep['commit']}.  SR {SR}, block {BLOCK} ({BUDGET_MS:.2f} ms). "
         f"A = `laplacian`, B = `{orz.ENGINE_ID}` (Spectrum Laplace, LAPLACE_GAIN {orz.LAPLACE_GAIN:g}, output "
         f"x{orz.OUT_SCALE:g}); f0 {F0_HZ:g} Hz, 6 gen/s, gain 0.028 = vol 0.7.  RMS = stereo RMS of the ready 12 s.",
         '', '## Spectral data A / B (73 states, 3 setting sets, every component)', '',
         '| scene | components | modes | worst abs diff | equal within 1e-10 |', '|---|---|---|---|---|']
    for s in rep['spectral']:
        L.append(f"| {s['id']} | {s['components_compared']} | {s['modes_compared']} | {s['worst_abs_diff']:.3g} | "
                 f"{s['equal_within_1e10']} |")
    L += ['', '## The seven settings on the galaxy (generations 0..7 in which the data differ)', '',
          '| setting | value | generations | preflight |', '|---|---|---|---|']
    for k, v in rep['settings'].items():
        L.append(f"| {k} | {v['value']} | {v['changed_generations']} | {v['preflight_generations']} |")
    L += ['', '## Scenes (12 s from a fresh start, the scene side gain applied)', '',
          '| scene | side | engine | rms dB | peak | clip | p99 ms |', '|---|---|---|---|---|---|---|']
    for s in rep['scenes']:
        for side in SIDES:
            st = s[side]
            L.append(f"| {s['id']} | {side} | {s['engines'][side]} | {st['rms_db']:.2f} | {st['peak']:.4f} | "
                     f"{st['clip_blocks']} | {s['block_ms']['p99']:.2f} |")
    L += ['']
    for s in rep['scenes']:
        L.append(f"- {s['id']}: side gain {s['side_gain']}; A - B = {s['a_minus_b_db']:+.2f} dB (at unit side gain "
                 f"{s['a_minus_b_db_at_unit_side_gain']:+.2f} dB); within 1 dB {s['within_1db']}; continue from 6 s exact "
                 f"{s['continue']['exact']} ({s['continue']['blocks']} blocks); B figures at the end {s['B']['figures']}, "
                 f"tails {s['B']['tails']}, faded {s['B']['evictions']}, in place {s['B']['inplace_fades']}, "
                 f"dropped {s['B']['drops']}")
    p = rep['packets']
    L += ['', '## No packet from any setting or from Restore (side B on the L3 field)', '',
          '| setting | value | pulse moved | frequencies changed | weight ramp (samples) |', '|---|---|---|---|---|']
    for r in p['settings']:
        L.append(f"| {r['setting']} | {r['value']} | {r['pulse_moved']} | {r['frequencies_changed']} | "
                 f"{r['weight_ramp_samples']} |")
    L.append(f"\nRestore (export / restore, the next 40 blocks): exact {p['restore_exact']}.")
    e3 = rep['l3']
    L += ['', '## L3 receiver events per generation', '']
    for k in ('radius_1.0', 'radius_1.5'):
        v = e3[k]
        L.append(f"- {k}: receiver e {v['receiver_e']} ({v['generations']} generations), blinker e {v['blinker_e']}; "
                 f"receiver centre {v['receiver_centre']}, R {v['receiver_R']:.5f} -> R_eff {v['receiver_R_eff']:.5f}; "
                 f"ids {v['ids']}; receiver {v['receiver_modes']} modes from {v['receiver_f_low']:.2f} Hz, blinker "
                 f"{v['blinker_modes']} modes from {v['blinker_f_low']:.2f} Hz")
    lv = e3['lever_1p5_to_1']
    L.append(f"- live lever 1.5 -> 1.0: no packet from the knob {lv['no_packet_from_the_knob']}; next generation "
             f"receiver e {lv['receiver_e_next']}, blinker e {lv['blinker_e_next']}")
    v = rep['l2']
    L += ['', '## L2 voices (side B, 12 s)', '',
          f"- {v['generations']} generations, identities seen {v['ids_seen']}, max figures {v['max_figures']}, max "
          f"sounding {v['max_sounding']}, max tails {v['max_tails']}; faded {v['evictions']}, in place "
          f"{v['inplace_fades']}, dropped {v['drops']}, unvoiced blocks {v['unvoiced_blocks']}"]
    for law, c in v['mode_return'].items():
        L.append(f"- mode return ({law} law): modes {c['modes']}, returning modes' state before their first sample "
                 f"{c['returning_modes_max_abs_state']:g}, tail slots {c['tail_slots']} (undriven {c['tail_undriven']}, "
                 f"pulse {c['tail_pulse']:g}), id kept {c['id_kept']}")
    L += ['', '## Timing (both sides, after 1 s of warm-up)', '']
    for k, t in rep['timing'].items():
        if isinstance(t, dict):
            L.append(f"- {k}: p50 {t['p50']:.2f}  p95 {t['p95']:.2f}  p99 {t['p99']:.2f}  max {t['max']:.2f} ms "
                     f"(budget {t['budget']:.2f}, ok {t['ok']})")
    L.append(f"- numba: {rep['timing']['numba']}")
    L += ['', '## Stress probes (gain 0.04, 6 s, apart from the scenes)', '',
          '| mode | settings | A peak / clip | B peak / clip | figures / sounding / tails B | faded / in place / dropped B | p99 ms |',
          '|---|---|---|---|---|---|---|']
    for r in rep['stress']:
        b = r['B']
        sets = 'REQ table' if r['settings'] == SPECTRUM else 'n20 spread1 alpha0 shape1 full1 dyn1'
        L.append(f"| {r['mode']} | {sets} | {r['A']['pre_clip_peak']:.3f} / {r['A']['clip_blocks']} | "
                 f"{b['pre_clip_peak']:.3f} / {b['clip_blocks']} | {b['n_figures']}/{b['n_sounding']}/{b['n_tails']} | "
                 f"{b['evictions']}/{b['inplace_fades']}/{b['drops']} | {r['block_ms_p99']:.2f} |")
    L += ['', '## Cost of the Laplace law by component size (full torus graph, no decimation)', '',
          '| cells | shape | ms | over the block budget |', '|---|---|---|---|']
    for r in rep['analysis_cost']:
        L.append(f"| {r['cells']} | {r['shape']:g} | {r['ms']:.1f} | {r['over_budget']} |")
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
    try:
        import subprocess
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip() or 'unknown'
    except Exception:                                                   # pragma: no cover
        commit = 'unknown'
    rep = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), commit=commit)
    print('spectral equality', flush=True)
    rep['spectral'] = spectral_equality()
    print('settings', flush=True)
    rep['settings'] = settings_effect()
    rep['scenes'] = []
    for case in CASES:
        print('scene', case['id'], flush=True)
        rep['scenes'].append(scene_measurements(case))
    print('packets', flush=True)
    rep['packets'] = no_packet_from_settings_and_restore()
    print('l3', flush=True)
    rep['l3'] = l3_events()
    print('l2', flush=True)
    rep['l2'] = l2_voices()
    print('timing', flush=True)
    rep['timing'] = timing()
    print('stress', flush=True)
    rep['stress'] = stress()
    print('analysis cost', flush=True)
    rep['analysis_cost'] = analysis_cost()
    root = Path(a.catalog)
    if not a.no_catalog and root.is_dir():
        print('catalog', flush=True)
        rep['catalog_root'] = str(root)
        rep['catalog'] = verify_catalog(root)
    problems, stress_problems, limitations = [], [], []
    for s in rep['spectral']:
        if not s['equal_within_1e10']:
            problems.append(f"{s['id']}: spectral data differ ({s['worst_abs_diff']:.3g})")
    for k, v in rep['settings'].items():
        if not v['changed_generations']:
            problems.append(f"setting {k} changes nothing on the galaxy")
    if rep['settings']['spread']['changed_generations'] != [2, 4, 5] or rep['settings']['fullshape']['changed_generations'] != [2, 4, 5]:
        problems.append("spread / fullshape do not act in the generations 2, 4, 5 of the preflight")
    for s in rep['scenes']:
        for side in SIDES:
            if s[side]['clip_blocks'] or not s[side]['finite']:
                problems.append(f"{s['id']} {side}: clip / non-finite")
        if not s['within_1db']:
            problems.append(f"{s['id']}: A - B = {s['a_minus_b_db']:+.2f} dB beyond 1 dB")
        if not s['block_ms']['ok']:
            problems.append(f"{s['id']}: p99 {s['block_ms']['p99']:.2f} ms over budget")
        if not s['continue']['exact'] or not s['continue']['field_changed']:
            problems.append(f"{s['id']}: continuation not exact / field frozen")
        if s['B']['drops']:
            problems.append(f"{s['id']} B: hard drops")
    for r in rep['packets']['settings']:
        if r['pulse_moved']:
            problems.append(f"setting {r['setting']} moved the pulse states")
    if not rep['packets']['restore_exact']:
        problems.append("Restore is not exact")
    e3 = rep['l3']
    if e3['radius_1.0']['receiver_e'] != [0.0] or e3['radius_1.5']['receiver_e'] != [4.0]:
        problems.append("L3 receiver events at x1 / x1.5 are not 0 / 4 per generation")
    if e3['radius_1.0']['blinker_e'] != [4.0] or e3['radius_1.5']['blinker_e'] != [4.0]:
        problems.append("L3 blinker does not receive its four changes")
    if e3['lever_1p5_to_1']['receiver_e_next'] != 0.0 or e3['lever_1p5_to_1']['blinker_e_next'] != 4.0:
        problems.append("L3 live lever 1.5 -> 1.0 does not stop the receiver's packets")
    v = rep['l2']
    if v['drops']:
        problems.append("L2: hard drops")
    for law, c in v['mode_return'].items():
        if c['returning_modes_max_abs_state'] != 0.0 or not c['tail_undriven'] or c['tail_pulse'] != 0.0 or not c['id_kept']:
            problems.append(f"mode return ({law}): the 4 -> 2 -> 4 rule is violated")
    for k, t in rep['timing'].items():
        if isinstance(t, dict) and not t['ok']:
            problems.append(f"timing {k}: p99 {t['p99']:.2f} ms over budget")
    for c_ in rep.get('catalog', []):
        if c_.get('status') != 'match':
            problems.append(f"catalog {c_.get('rid')}: {c_.get('status')} {c_.get('reason')}")
    clipped = [r for r in rep['stress'] if r['A']['clip_blocks'] or r['B']['clip_blocks']]
    slow = [r for r in rep['stress'] if r['block_ms_p99'] >= BUDGET_MS]
    for r in rep['stress']:
        for side in SIDES:
            if not r[side]['finite']:
                stress_problems.append(f"stress {r['mode']} {side}: non-finite")
    worst = max(max(r['A']['pre_clip_peak'], r['B']['pre_clip_peak']) for r in rep['stress'])
    n_setting = rep['settings']['n']['changed_generations']
    limitations += [
        f"the n setting changes the galaxy's data in generations {n_setting} only: in generations 3 and 7 every "
        f"component has 4 cells (3 modes), n = 4 and n = 12 select the same three (the preflight counted the "
        f"zero-padded array of the old synth)",
        "on a figure across the seam the old Laplace (bbox of the unwrapped labels) and Objects (torus component) "
        "may segment differently; none of the 73 states of the three scenes touches the seam",
        "the old Laplace's node ceiling (256, lattice decimation) is not carried into the Objects graph: a component "
        "above ~300 cells costs more than the block budget on the render thread (see the cost table)",
        f"the level match is one constant factor per scene on side B ({', '.join(s['id'] + ' ' + str(s['side_gain']['B']) for s in rep['scenes'])}), "
        f"chosen for <= 1 dB of integral RMS, not equal loudness; the old Laplace algorithm is untouched",
        "the two N4 experiments keep the Figure law; their records (v1 engine, pinned to the commit of their "
        "sound files) are replayed and continued by a separate bench of that version, never by this code",
    ]
    if clipped:
        limitations.append(f"stress clipping at gain 0.04 (vol 1) in {len(clipped)} of {len(rep['stress'])} probes: "
                           + '; '.join(f"{r['mode']} A {r['A']['clip_blocks']} / B {r['B']['clip_blocks']} blocks "
                                       f"(peaks {r['A']['pre_clip_peak']:.2f} / {r['B']['pre_clip_peak']:.2f})" for r in clipped))
    if slow:
        limitations.append(f"{len(slow)} of {len(rep['stress'])} stress probes exceed the block budget at p99 "
                           f"(worst {max(r['block_ms_p99'] for r in slow):.1f} ms): the figure analysis of dense fields "
                           f"on the render thread -- offline exact, live underruns possible")
    inplace = [r for r in rep['stress'] if r['B']['inplace_fades']]
    if inplace:
        limitations.append("in-place fades under whole-field replacement every block: "
                           + '; '.join(f"{r['mode']}: {r['B']['inplace_fades']}" for r in inplace))
    dropped = [r for r in rep['stress'] if r['B']['drops']]
    if dropped:
        limitations.append("hard drops (in-place fading modes cut by a regrowth within 20 ms with every pool busy): "
                           + '; '.join(f"{r['mode']}: {r['B']['drops']}" for r in dropped))
    limitations.append(f"worst stress pre-clip peak {worst:.3f} at gain 0.04 -- measured for these finite probes, not "
                       f"guaranteed for any playing")
    limitations.append("no listening in this report: hearing the difference is the user's verdict")
    rep['problems'] = dict(scenes=problems, stress=stress_problems)
    rep['summary'] = dict(
        scenes=('spectral data of A and B equal on every component of the 73 states (3 setting sets), all seven '
                'settings act (spread / full in generations 2, 4, 5), no packet from any setting or Restore, L3 '
                'receiver 0 / 4 per generation at x1 / x1.5 and the live lever stops it, L2 voices without drops '
                'and the 4 -> 2 -> 4 rule holds, A - B within 1 dB after the scene side gain, no clipping, '
                'continuation exact, p99 within the budget, records replay exactly'
                if not problems else '; '.join(problems)),
        stress=(f"finite in all {len(rep['stress'])} probes; clipping at gain 0.04 in {len(clipped)}, over budget "
                f"in {len(slow)}, in-place fades in {len(inplace)}, hard drops in {len(dropped)}; worst pre-clip peak {worst:.3f}"
                if not stress_problems else '; '.join(stress_problems)),
        limitations=limitations)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    write_markdown(rep, OUT_DIR / 'report.md')
    print('summary scenes:', rep['summary']['scenes'])
    print('summary stress:', rep['summary']['stress'])
    print('written:', OUT_DIR / 'report.json', OUT_DIR / 'report.md')
    return 1 if (problems or stress_problems) else 0


if __name__ == '__main__':
    raise SystemExit(main())
