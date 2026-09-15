#!/usr/bin/env python3
"""N1 hand-over measurements (REQ memory/req-network-ca-n1-2026-09-15.md, "Checks before
hand-over" 2 and 3) -- numbers only, nothing perceptual is claimed:

    python demos/gutter_field_n1_report.py            # offline measurements + 60 s live device run
    python demos/gutter_field_n1_report.py --no-live  # skip the device run

  fixtures    u / ratio / bank frequencies of the three fixture fields vs the fixtures file
  invariance  depth = 0 and Freeze CA: a run with field edits == the run without (PCM equal)
  ab          the prepared scene, 12 s: side A (follows the field) vs B (holds the initial
              control) -- level, centroid, band energy, spectral flux per 2 s window
  edits       side A, 20 s, the fixtures' edit schedule (vertical / moved every 4 s, no
              restart, both sides rendered) -- the same per-window measurements
  timing      offline: 60 s of both sides with paint / evolution / parameter moves, per-block
              render time p50 / p95 / p99 / max vs the block length 352 / 44100 s
  live        60 s on the real audio device through the bench's LiveEngine with the same kind of
              interventions: device underruns, block-render times (p95 / p99), clip blocks
  jvm         the node-port verification against the original gutterOsc.class, if
              artifacts/_n1/verification_node.json is present (verify_node.py)

Writes demos/results/network_n1/report.json + report.md (versioned, small); WAVs of the
measured runs go to artifacts/_n1/ (gitignored).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from casynth_config import SR, MASTER_GAIN, VOL_DEFAULT                          # noqa: E402
from casynth_lab import DemoRunner, BLOCK, load_scene, write_wav                # noqa: E402
from casynth_lab import gutter_field as GF                                       # noqa: E402
from demos.network_reference_n0 import analysis as an                            # noqa: E402

OUT = os.path.join(ROOT, 'demos', 'results', 'network_n1')
ART = os.path.join(ROOT, 'artifacts', '_n1')
FIXTURES = os.path.join(ROOT, 'memory', 'research', 'network-n1-fixtures-2026-09-15.json')
SCENE = os.path.join(ROOT, 'demos', 'network_n1_blinkers.json')
BUDGET_MS = BLOCK / SR * 1e3
WINDOW_S = 2.0


def _fx():
    with open(FIXTURES, encoding='utf-8') as f:
        return json.load(f)


def _cells_cmds(cells, at, value):
    return [('set_cell', at, dict(r=int(r), c=int(c), v=value)) for r, c in cells]


def _render(scene, seconds, commands, outputs=('A', 'B')):
    r = DemoRunner(scene)
    for kind, at, args in commands:
        r.post(kind, at=at, **args)
    n = int(round(seconds * SR))
    got = {o: [] for o in outputs}
    while len(got[outputs[0]]) * BLOCK < n:
        b = r.next_block()
        for o in outputs:
            got[o].append(b.get(o))
    return {o: np.concatenate(got[o])[:n] for o in outputs}, r


def _windows(pcm, seconds=WINDOW_S):
    """Per-window measurements of int16 stereo PCM: rms dB, centroid, band fractions, flux."""
    x = an.mono(pcm.astype(np.float64) / 32767.0)
    n = int(seconds * SR)
    out = []
    for k in range(0, len(x) - n + 1, n):
        w = x[k:k + n]
        f, p = an.spectrum(w)
        band = (f >= 20) & (f < 16000)
        tot = max(float(p[band].sum()), 1e-30)
        fr = [float(p[(f >= lo) & (f < hi)].sum() / tot)
              for lo, hi in ((20, 100), (100, 500), (500, 2000), (2000, 16000))]
        feats = an.spectral_features(f, p)
        temp = an.temporal_features(w)
        out.append(dict(t0=k / SR, rms_db=float(an.db(np.sqrt(np.mean(w * w)))),
                        centroid_hz=feats['centroid_hz'], flatness_db=feats['flatness_db'],
                        band_fraction=fr, spectral_flux=temp['spectral_flux'],
                        env_std_db_250ms=temp['env_std_db_250ms']))
    return out


def section_fixtures(fx):
    base = np.array(GF.N1_CONFIG['filters_hz'])
    res = {}
    for name, fld in fx['fields'].items():
        g = np.zeros((32, 32), np.uint8)
        for r, c in fld['cells']:
            g[r, c] = 1
        counts = GF.region_counts(g)
        u = GF.field_u(counts)
        ratio = GF.ratio_of(u, 1.0, 1.0)
        freqs = GF.bank_freqs(base, ratio)
        res[name] = dict(counts=counts.tolist(), u=u.tolist(), ratio=ratio.tolist(),
                         counts_match=counts.tolist() == fld['counts'], u_match=u.tolist() == fld['u'],
                         ratio_max_abs_err=float(np.abs(ratio - np.array(fld['ratio'])).max()),
                         freq_min_hz=float(freqs.min()), freq_max_hz=float(freqs.max()),
                         freqs_are_float32=bool(np.array_equal(freqs, freqs.astype(np.float32))))
    return res


def section_invariance(scene, fx):
    """depth = 0 / Freeze CA on side B: edits of the field must not change B's PCM."""
    res = {}
    moved = fx['fields']['moved']['cells']
    vertical = fx['fields']['vertical']['cells']
    for label, param in (('depth_0', ('depth', 0.0)), ('freeze_ca', ('freeze_ca', 1))):
        base_cmds = [('start', 0, {}), ('pause', 0, dict(on=True)),
                     ('set_param', 0, dict(side='B', name=param[0], value=param[1]))]
        edit_cmds = list(base_cmds)
        edit_cmds += _cells_cmds(vertical, int(1.0 * SR), 0) + _cells_cmds(moved, int(1.0 * SR), 1)
        edit_cmds += _cells_cmds(moved, int(2.0 * SR), 0) + _cells_cmds(vertical, int(2.0 * SR), 1)
        a, _ = _render(scene, 3.0, base_cmds)
        b, rb = _render(scene, 3.0, edit_cmds)
        res[label] = dict(B_pcm_identical_with_edits=bool(np.array_equal(a['B'], b['B'])),
                          A_pcm_identical_with_edits=bool(np.array_equal(a['A'], b['A'])),
                          edits_applied=int(sum(1 for j in rb.journal if j[2] == 'set_cell')),
                          B_peak=float(np.abs(b['B']).max() / 32767.0))
    return res


def section_ab(scene):
    pcm, r = _render(scene, 12.0, [('start', 0, {})])
    write_wav(os.path.join(ART, 'ab_A_follows.wav'), pcm['A'])
    write_wav(os.path.join(ART, 'ab_B_frozen.wav'), pcm['B'])
    first_step = 63 * BLOCK
    disp = r.snapshot()['display']
    return dict(generations=int(r.gen), clip_blocks=r.snapshot()['clip_blocks'],
                identical_until_first_step=bool(np.array_equal(pcm['A'][:first_step], pcm['B'][:first_step])),
                identical_whole=bool(np.array_equal(pcm['A'], pcm['B'])),
                peak={o: float(np.abs(pcm[o]).max() / 32767.0) for o in pcm},
                resets={o: disp[o]['resets'] for o in disp},
                windows={o: _windows(pcm[o]) for o in pcm})


def section_edits(scene, fx):
    cmds = [('start', 0, {}), ('pause', 0, dict(on=True))]        # CA paused: only the edits act
    prev = None
    for ev in fx['edit_schedule']:
        at = int(ev['seconds'] * SR)
        cells = fx['fields'][ev['field']]['cells']
        if prev is not None and prev != ev['field']:
            cmds += _cells_cmds(fx['fields'][prev]['cells'], at, 0)
        if prev != ev['field']:
            cmds += _cells_cmds(cells, at, 1)
        prev = ev['field']
    pcm, r = _render(scene, 20.0, cmds)
    write_wav(os.path.join(ART, 'edits_A.wav'), pcm['A'])
    applied = sorted({j[0] for j in r.journal if j[2] == 'set_cell'})
    disp = r.snapshot()['display']
    return dict(schedule=fx['edit_schedule'], edit_block_starts_s=[a / SR for a in applied],
                clip_blocks=r.snapshot()['clip_blocks'], resets={o: disp[o]['resets'] for o in disp},
                peak={o: float(np.abs(pcm[o]).max() / 32767.0) for o in pcm},
                final_u_A=disp['A']['u'], final_u_B=disp['B']['u'],
                windows={o: _windows(pcm[o]) for o in pcm})


def _interventions(r, i, base_at=None):
    """Paint / erase, parameter moves and the transport on a block index i; `r` is the
    runner (offline, single thread) or the LiveEngine (its queue feeds the render thread)."""
    at = base_at
    if i % 20 == 0:
        r.post('set_cell', at=at, r=(i // 20) % 32, c=(i // 7) % 32, v=1)
    if i % 45 == 0:
        r.post('set_cell', at=at, r=(i // 45 + 14) % 32, c=(i // 9) % 32, v=0)
    if i % 500 == 250:
        r.post('set_param', at=at, side='A', name='interaction', value=int(60 + (i // 500) * 40) % 257)
    if i % 700 == 350:
        r.post('set_param', at=at, side='B', name='scale', value=float(0.5 + ((i // 700) % 4) * 0.5))
    if i % 900 == 450:
        r.post('set_param', at=at, side='A', name='freeze_ca', value=int((i // 900) % 2))
    if i % 1100 == 550:
        r.post('set_param', at=at, side='A', name='depth', value=float(((i // 1100) % 3) * 0.5))
    if i % 1300 == 650:
        r.post('select', at=at, side='B' if (i // 1300) % 2 == 0 else 'A')
    if i % 1700 == 850:
        r.post('pause', at=at, on=bool((i // 1700) % 2 == 0))


def section_timing(scene, seconds=60.0):
    r = DemoRunner(scene)
    r.post('start', at=0)
    for _ in range(30):
        r.next_block()
    n = int(seconds * SR / BLOCK)
    ts = np.empty(n)
    for i in range(n):
        _interventions(r, i)
        t0 = time.perf_counter()
        r.next_block()
        ts[i] = time.perf_counter() - t0
    ms = ts * 1e3
    return dict(blocks=n, seconds=seconds, budget_ms=BUDGET_MS, sides=2,
                p50_ms=float(np.percentile(ms, 50)), p95_ms=float(np.percentile(ms, 95)),
                p99_ms=float(np.percentile(ms, 99)), max_ms=float(ms.max()), mean_ms=float(ms.mean()),
                blocks_over_budget=int(np.count_nonzero(ms > BUDGET_MS)),
                us_per_sample_per_side=float(ms.mean() * 1e3 / BLOCK / 2),
                generations=int(r.gen), clip_blocks=r.snapshot()['clip_blocks'],
                commands=len([j for j in r.journal if j[2] != 'step']))


def section_live(scene, seconds=60.0, vol=0.05):
    """The bench's own audio path: LiveEngine + sounddevice, real time, quiet volume."""
    from casynth_lab.audio_out import LiveEngine
    r = DemoRunner(scene, vol=vol)
    times = []
    orig = r.next_block

    def timed():
        t0 = time.perf_counter()
        b = orig()
        times.append(time.perf_counter() - t0)
        return b
    r.next_block = timed
    eng = LiveEngine(r)
    eng.start()
    if not eng.device_ok:
        eng.stop()
        return dict(device_ok=False, device_error=eng.device_error)
    eng.post('pause', on=False)
    t_start = time.perf_counter()
    i = 0
    try:
        while time.perf_counter() - t_start < seconds:
            _interventions(eng, i)             # through the engine's queue (render thread applies)
            i += 1
            time.sleep(BLOCK / SR)
    finally:
        eng.post('vol', value=0.0)
        time.sleep(0.2)
        diag = eng.diagnostics()
        eng.stop()
    ms = np.array(times[20:]) * 1e3
    return dict(device_ok=True, seconds=seconds, vol=vol, underruns=int(diag['underruns']),
                block_errors=int(diag['block_errors']), clip_blocks=diag['clip_blocks'],
                blocks_rendered=len(times), p50_ms=float(np.percentile(ms, 50)),
                p95_ms=float(np.percentile(ms, 95)), p99_ms=float(np.percentile(ms, 99)),
                max_ms=float(ms.max()), blocks_over_budget=int(np.count_nonzero(ms > BUDGET_MS)),
                budget_ms=BUDGET_MS, generations=int(r.gen), interventions=i)


def section_jvm():
    path = os.path.join(ART, 'verification_node.json')
    if not os.path.isfile(path):
        return dict(present=False)
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return dict(present=True, all_pass=d.get('all_pass'), source_commit=d.get('source_commit'),
                source_hashes_match=d.get('source_hashes_match'), samples=d.get('samples'),
                cases={k: v.get('pass') for k, v in d.get('cases', {}).items()})


def _md(rep):
    L = ["# N1 `gutter_field` -- hand-over measurements", "",
         f"Generated {rep['generated']} by demos/gutter_field_n1_report.py; model "
         f"`{rep['model_version']}`, numba {rep['numba']}.  Numbers only; nothing about hearing.", ""]
    L += ["## Fixtures (u / ratio / bank frequencies)", "",
          "| field | counts | u match | ratio max err | f min..max Hz |", "|---|---|---|---|---|"]
    for k, v in rep['fixtures'].items():
        L.append(f"| {k} | {v['counts']} | {v['counts_match'] and v['u_match']} | {v['ratio_max_abs_err']:.1e} "
                 f"| {v['freq_min_hz']:.1f}..{v['freq_max_hz']:.1f} |")
    L += ["", "## Invariance (edits while depth = 0 / Freeze CA)", ""]
    for k, v in rep['invariance'].items():
        L.append(f"- {k}: B PCM identical with {v['edits_applied']} cell edits = **{v['B_pcm_identical_with_edits']}**"
                 f" (A identical: {v['A_pcm_identical_with_edits']})")

    def table(title, w):
        L.extend(["", f"### {title}", "",
                  "| t0 s | rms dB | centroid Hz | flat dB | 20-100 | 100-500 | 500-2k | 2k-16k | flux | env std dB |",
                  "|---|---|---|---|---|---|---|---|---|---|"])
        for x in w:
            b = x['band_fraction']
            L.append(f"| {x['t0']:.0f} | {x['rms_db']:.1f} | {x['centroid_hz']:.0f} | {x['flatness_db']:.1f} "
                     f"| {b[0]:.2f} | {b[1]:.2f} | {b[2]:.2f} | {b[3]:.2f} | {x['spectral_flux']:.4f} "
                     f"| {x['env_std_db_250ms']:.2f} |")
    ab = rep['ab']
    L += ["", "## A / B on the prepared scene (12 s, 2 gen/s)", "",
          f"generations {ab['generations']}, clip blocks {ab['clip_blocks']}, resets {ab['resets']}, "
          f"peak {ab['peak']}, identical until the first step: {ab['identical_until_first_step']}, "
          f"identical whole: {ab['identical_whole']}"]
    table("A -- follows the field", ab['windows']['A'])
    table("B -- Freeze CA", ab['windows']['B'])
    ed = rep['edits']
    L += ["", "## Edit schedule (20 s, CA paused, vertical / moved every 4 s, no restart)", "",
          f"edits applied at block starts (s): {[round(t, 3) for t in ed['edit_block_starts_s']]}; "
          f"clip blocks {ed['clip_blocks']}, resets {ed['resets']}, peak {ed['peak']}"]
    table("A -- follows the edits", ed['windows']['A'])
    table("B -- Freeze CA (control)", ed['windows']['B'])
    t = rep['timing']
    L += ["", "## Timing, offline (both sides, 60 s with interventions)", "",
          f"| blocks | p50 ms | p95 ms | p99 ms | max ms | budget ms | over budget | us/sample/side |",
          "|---|---|---|---|---|---|---|---|",
          f"| {t['blocks']} | {t['p50_ms']:.3f} | {t['p95_ms']:.3f} | {t['p99_ms']:.3f} | {t['max_ms']:.3f} "
          f"| {t['budget_ms']:.3f} | {t['blocks_over_budget']} | {t['us_per_sample_per_side']:.2f} |"]
    lv = rep.get('live')
    L += ["", "## Live device run (bench LiveEngine, 60 s with interventions)", ""]
    if lv is None:
        L.append("skipped (--no-live)")
    elif not lv.get('device_ok'):
        L.append(f"no audio device: {lv.get('device_error')}")
    else:
        L += [f"| underruns | block errors | blocks | p50 ms | p95 ms | p99 ms | max ms | over budget | clip blocks |",
              "|---|---|---|---|---|---|---|---|---|",
              f"| {lv['underruns']} | {lv['block_errors']} | {lv['blocks_rendered']} | {lv['p50_ms']:.3f} "
              f"| {lv['p95_ms']:.3f} | {lv['p99_ms']:.3f} | {lv['max_ms']:.3f} | {lv['blocks_over_budget']} "
              f"| {lv['clip_blocks']} |"]
    j = rep['jvm']
    L += ["", "## Node port vs the original gutterOsc.class (JVM)", ""]
    if j['present']:
        L.append(f"all pass: **{j['all_pass']}**, source commit {j['source_commit']}, hashes match "
                 f"{j['source_hashes_match']}, {len(j['cases'])} cases: "
                 + ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in j['cases'].items()))
    else:
        L.append("verification_node.json not found in artifacts/_n1 (run verify_node.py)")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="N1 hand-over measurements")
    ap.add_argument('--no-live', action='store_true', help="skip the 60 s audio-device run")
    ap.add_argument('--live-seconds', type=float, default=60.0)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(ART, exist_ok=True)
    fx = _fx()
    scene = load_scene(SCENE)
    rep = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), model_version=GF.MODEL_VERSION,
               numba=(__import__('numba').__version__ if GF.HAVE_NUMBA else None),
               gain=dict(fixed_scale=GF.FIXED_SCALE, master_gain=MASTER_GAIN, vol_default=VOL_DEFAULT,
                         effective_at_defaults=GF.FIXED_SCALE * MASTER_GAIN * VOL_DEFAULT))
    for name, fn in (('fixtures', lambda: section_fixtures(fx)), ('invariance', lambda: section_invariance(scene, fx)),
                     ('ab', lambda: section_ab(scene)), ('edits', lambda: section_edits(scene, fx)),
                     ('timing', lambda: section_timing(scene))):
        t0 = time.time()
        rep[name] = fn()
        print(f"[{name}] done in {time.time() - t0:.1f} s", flush=True)
    rep['live'] = None if a.no_live else section_live(scene, a.live_seconds)
    if rep['live'] is not None:
        print(f"[live] {rep['live']}", flush=True)
    rep['jvm'] = section_jvm()
    with open(os.path.join(OUT, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(OUT, 'report.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(_md(rep))
    print(f"report: {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
