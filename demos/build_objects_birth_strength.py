"""Build the Objects "Birth strength" catalog: ONE live A/B experiment M2 on the M1
field, both sides Objects / Laplace with the Researcher's preflight settings and
Excitation = Birth position, A Birth strength 1 (the original law) / B 4 (stronger
selection) -- REQ memory/req-objects-event-source-modal-2026-09-17.md section 5.

    python demos/build_objects_birth_strength.py [--root DIR] [--scenes-only] [--calibrate] [--levels]

The scene demos/obs_m2.json is (re)written; the catalog is built only into a directory
that holds no records yet (records and Notes the user added are never overwritten).
`run_objects_birth_strength.bat` opens the bench on the catalog screen of the default
root.  The field: the exact `cases.M1.cells` of the 2026-09-17 preflight (Jam, period
3), Conway B3/S23 on a 32 x 32 torus, 6 generations / s, 12 s, f0 = 110 Hz; both
sides on the same field; the hypothesis of REQ 5.3 is written into Notes verbatim.

Both sides: `ca_object_resonators`, the preflight `settings` of M1: Own, Spectrum
Laplace, full 1, part 3, spread 1, harm 0.87, Decay 1.39 s, Attack 4 ms, shape 0,
alpha 0, dyn 0, Events Births, Excitation Birth position.  The sides differ ONLY in
`birth_strength`: A 1.0, B 4.0.  In the live window the knob moves freely 0..4 (0 =
the uniform strike, 2 = the middle value of the REQ); nothing changes it in the record.

Levels: SIDE_GAIN is ONE constant factor of side B (scene audio.side_gain, applied
before the PCM clip, part of the record / snapshot, so Replay / Continue use it)
measured for the delivered pair 1 / 4 from the integral RMS of the part AFTER the
first 2 s (the start fills the field, which is not an ordinary transition) so that A
and B differ by <= 1 dB there (--calibrate prints the measurement at unit gains; the
old 0.93 of M1 is NOT carried over).  The factor does not follow a manual change of
the knob: a manual change of the strength may change the loudness (said in the
record's description); no AGC, no per-strike levelling.  --levels prints RMS and peaks
of one side at s = 0 / 0.5 / 1 / 2 / 4 with unit gain and with the delivered gains
(demos/objects_birth_strength_report.py writes them into the report).
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

from casynth_config import SR                                             # noqa: E402
from casynth_lab import scene_from_doc, DemoRunner, BLOCK                # noqa: E402
from casynth_lab.catalog import Catalog                                  # noqa: E402
from casynth_lab.object_resonators import ENGINE_ID, EV_BIRTHS, EXC_POSITION   # noqa: E402
from demos.build_n1_demos import record_offline                          # noqa: E402
from demos.build_objects_laplace import require_pinnable                 # noqa: E402
from demos.build_objects_event_source import (preflight_params, case_cells, rms_db,   # noqa: E402
                                              ROWS, COLS, RATE_HZ, SECONDS, F0_HZ, STEADY_FROM_S,
                                              SETTINGS_TEXT)

CATALOG_ROOT = ROOT / 'lab_catalog' / 'objects_birth_strength_2026_09_18'
PREFLIGHT_STRENGTH = ROOT / 'memory' / 'research' / 'objects-birth-strength-preflight-2026-09-18.json'
STRENGTHS = (0.0, 0.5, 1.0, 2.0, 4.0)    # the REQ's five values of the level / coefficient checks
# ONE constant level factor of side B for the delivered pair 1 / 4 (measured with
# --calibrate at side gain 1, A - B integral RMS of 2..12 s -> B = 10 ** (diff / 20),
# 2 decimals); the numbers: memory/log/2026-09-18-objects-birth-strength.md
# measured 2026-09-18 at side gain 1: A-B after 2 s -0.73 dB (whole take -0.69), peaks 0.134 / 0.134, no clip
SIDE_GAIN = {'obs_m2': dict(A=1.0, B=0.92)}


def side_params(strength):
    """The M1 preflight settings + Births + Birth position + the strength."""
    p = preflight_params(EV_BIRTHS, EXC_POSITION)
    p['birth_strength'] = float(strength)
    return p


CASES = [
    dict(id='obs_m2', title='M2 - Сила места рождения: Birth strength 1 / 4 (Jam p3)',
         case='M1', strength=dict(A=1.0, B=4.0),
         side_a='А — Objects / Laplace, Births, Birth position, Birth strength 1',
         side_b='Б — Objects / Laplace, Births, Birth position, Birth strength 4',
         note='Jam (период 3) из preflight M1; 12 с, 6 поколений/с, f0 = 110 Hz; обе стороны — ' + SETTINGS_TEXT +
         ', Events Births, Excitation Birth position. Поле, моменты ударов, общая сила a, частоты и выходные веса '
         'совпадают; различается только сила влияния места рождения: в А исходное распределение пакета по модам '
         '(s = 1), в Б его отношения усилены (s = 4: c ~ b^4, та же норма пакета). У 16-клеточной компоненты '
         'c ≈ [0.78, 0.70, 1.38] в А и [0.18, 0.12, 1.72] в Б. Ручка Birth strength в живом окне двигается 0…4 '
         '(0 — общий удар, 1 — прежняя сила, 2…4 — усиленная). Калибровка уровня — один постоянный коэффициент '
         'стороны Б для пары 1 / 4 (по RMS после первых 2 с); при ручном изменении силы коэффициент не меняется, '
         'поэтому громкость может меняться.',
         hypothesis='Гипотеза - В А место рождения влияет на звук с прежней силой, в Б это влияние усилено. '
         'Ожидаем, что в Б отдельные удары будут заметнее различаться по слышимым тонам: на одних шагах '
         'выделяется один тон, на других другой. Проверяем, появляется ли интересное чередование звучаний или '
         'только сильнее выделяется один голос в прежней фразе. Моменты ударов на обеих сторонах одинаковы. '
         'Ручку Birth strength можно двигать: 0 — общий удар, 1 — прежняя сила, 2…4 — усиленная.'),
]


def listen_text(case):
    return ('1 / 2: ' + case['side_a'] + ' / ' + case['side_b'] + '. Notes: hypothesis and your listening '
            'result. Birth strength is the knob under Excitation on the Objects panel (0 = uniform strike, '
            '1 = original, >1 = stronger selection); a manual change of it may change the loudness.')


def scene_for(case, side_gain=None, strength=None):
    """The scene document; `strength` {A, B} overrides the case's pair (level checks)."""
    st = dict(case['strength']) if strength is None else dict(strength)
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=ROWS, cols=COLS),
                cells=case_cells(case['case']), rule='B3/S23', boundary='torus', rate_hz=RATE_HZ,
                audio=dict(f0_hz=F0_HZ, level=1.0,
                           side_gain=dict(side_gain if side_gain is not None else SIDE_GAIN[case['id']])),
                variants=dict(A=dict(engine_id=ENGINE_ID, engine_params=side_params(st['A'])),
                              B=dict(engine_id=ENGINE_ID, engine_params=side_params(st['B']))),
                initial_side='A', listen=listen_text(case))


def write_scenes():
    paths = []
    for case in CASES:
        doc = scene_for(case)
        scene_from_doc(doc)                       # strict validation before writing
        path = ROOT / 'demos' / (case['id'] + '.json')
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        paths.append(path)
    return paths


def render_sides(doc, seconds=SECONDS):
    """PCM of both sides of a scene document driven offline; (pcm {A, B}, peaks, clip blocks)."""
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    pcm = {s: [] for s in ('A', 'B')}
    peak = {s: 0.0 for s in ('A', 'B')}
    for _ in range(int(math.ceil(seconds * SR / BLOCK))):
        blk = runner.next_block()
        for s in ('A', 'B'):
            pcm[s].append(blk.get(s))
            peak[s] = max(peak[s], runner.sides[s].peak)
    return {s: np.concatenate(pcm[s]) for s in ('A', 'B')}, peak, dict(runner.snapshot()['clip_blocks'])


def calibrate(seconds=SECONDS, steady_from=STEADY_FROM_S):
    """RMS of both sides of the delivered pair at side gain 1 -- the whole take and the
    steady part after `steady_from` -- and the B factor that would equalise the steady
    part (printed; the constant above is set by hand from it)."""
    out = {}
    for case in CASES:
        y, peak, clip = render_sides(scene_for(case, side_gain=dict(A=1.0, B=1.0)), seconds)
        k0 = int(steady_from * SR)
        a, b = rms_db(y['A']), rms_db(y['B'])
        a2, b2 = rms_db(y['A'][k0:]), rms_db(y['B'][k0:])
        out[case['id']] = dict(rms_a_db=a, rms_b_db=b, a_minus_b_db=a - b,
                               steady_rms_a_db=a2, steady_rms_b_db=b2, steady_a_minus_b_db=a2 - b2, peak=peak,
                               b_gain_for_equal_steady_rms=round(10.0 ** ((a2 - b2) / 20.0), 2), clip_blocks=clip)
        print(f"  {case['id']}: whole A {a:.2f} B {b:.2f} dB (A-B {a - b:+.2f});  after {steady_from:g} s "
              f"A {a2:.2f} B {b2:.2f} dB (A-B {a2 - b2:+.2f}) -> B gain {out[case['id']]['b_gain_for_equal_steady_rms']}"
              f"  peaks A {peak['A']:.3f} B {peak['B']:.3f}  clip {clip}", flush=True)
    return out


def levels(case=None, seconds=SECONDS, steady_from=STEADY_FROM_S, strengths=STRENGTHS):
    """RMS (whole / after `steady_from`), peak, clip and finiteness of ONE side at every
    strength of `strengths`, with unit gain and with the delivered side gains (the
    knob on side B, so its constant factor applies): the REQ's five-value check."""
    case = CASES[0] if case is None else case
    k0 = int(steady_from * SR)
    out = {}
    for label, gain in (('unit', dict(A=1.0, B=1.0)), ('delivered', SIDE_GAIN[case['id']])):
        rows = {}
        for s in strengths:
            y, peak, clip = render_sides(scene_for(case, side_gain=gain, strength=dict(A=case['strength']['A'], B=s)),
                                         seconds)
            yb = y['B']
            rows[f"{s:g}"] = dict(strength=float(s), side_gain=float(gain['B']), rms_db=rms_db(yb),
                                  steady_rms_db=rms_db(yb[k0:]), peak=float(peak['B']), clip_blocks=int(clip['B']),
                                  finite=bool(np.all(np.isfinite(yb))))
            r = rows[f"{s:g}"]
            print(f"  {label:9s} s {s:<4g} gain {gain['B']:.2f}: RMS {r['rms_db']:.2f} dB (after {steady_from:g} s "
                  f"{r['steady_rms_db']:.2f})  peak {r['peak']:.3f}  clip {r['clip_blocks']}  finite {r['finite']}",
                  flush=True)
        out[label] = rows
    return out


def build(root=CATALOG_ROOT):
    root = Path(root)
    require_pinnable()
    if root.exists() and any(not p.name.startswith('.') for p in root.iterdir()):
        raise SystemExit('error: target catalog already holds records: ' + str(root) +
                         '\n(rebuild into a new directory with --root; records and Notes the '
                         'user added are never overwritten)')
    (root / '.tmp').mkdir(parents=True, exist_ok=True)
    cat = Catalog(str(root))
    results = []
    for case in reversed(CASES):
        started = time.monotonic()
        note = case['side_a'] + ' / ' + case['side_b'] + '. ' + case['note']
        rid, snap = record_offline(cat, scene_for(case), SECONDS, [], case['title'], note)
        cat.write_notes(rid, case['hypothesis'] + '\n\n')
        results.append((rid, case['id'], dict(snap['clip_blocks']), snap['gen']))
        print(f"  {rid}  {case['id']}  clip={snap['clip_blocks']} gen={snap['gen']}", flush=True)
        time.sleep(max(0.0, 1.05 - (time.monotonic() - started)))   # ids carry a 1 s stamp
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the Objects Birth strength scene and catalog (M2)")
    ap.add_argument('--root', default=str(CATALOG_ROOT))
    ap.add_argument('--scenes-only', action='store_true')
    ap.add_argument('--calibrate', action='store_true', help="measure the RMS of both sides at side gain 1")
    ap.add_argument('--levels', action='store_true', help="RMS / peaks of side B at s = 0 / 0.5 / 1 / 2 / 4")
    a = ap.parse_args(argv)
    if a.calibrate:
        calibrate()
        return 0
    if a.levels:
        levels()
        return 0
    paths = write_scenes()
    print(f"scenes: {len(paths)} files in {ROOT / 'demos'}")
    if a.scenes_only:
        return 0
    res = build(a.root)
    print(f"catalog: {len(res)} records in {a.root}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
