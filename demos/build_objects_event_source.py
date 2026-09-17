"""Build the Objects "event source / modal excitation" catalog: two live A/B
experiments, both Objects / Laplace with the Researcher's preflight settings on both
sides (REQ memory/req-objects-event-source-modal-2026-09-17.md).

    python demos/build_objects_event_source.py [--root DIR] [--scenes-only] [--calibrate]

Scenes demos/oes_*.json are (re)written; the catalog is built only into a directory
that holds no records yet (records and Notes the user added are never overwritten).
`run_objects_event_source_modal.bat` opens the bench on the catalog screen of the
default root.  Both experiments: Conway B3/S23 on a 32 x 32 torus, 6 generations / s,
12 s, f0 = 110 Hz, both sides on the same field; the cells are the Researcher's
preflight fixtures (memory/research/objects-event-source-modal-preflight-2026-09-17.json,
`cases.<name>.cells`); the hypotheses of the REQ are written into Notes verbatim.

Both sides: `ca_object_resonators`, the preflight `settings`: Own, Spectrum Laplace,
full 1, part 3, spread 1, harm 0.87, Decay 1.39 s, Attack 4 ms, shape 0, alpha 0,
dyn 0 (Radius x is stored but does not act with Own; Freq scale is inactive in the
Laplace law).  Attack 4 and the equal output weights are the conditions of this
research probe, not new user defaults.  The sides differ ONLY in the experiment's
selector:

  E1  Births / deaths          Octagon II (period 5); A Events Births, B Events Deaths;
                               Excitation Uniform on both
  M1  Uniform / birth position Jam (period 3); Events Births on both; A Excitation
                               Uniform, B Excitation Birth position

Levels: SIDE_GAIN is ONE constant factor per side per experiment (scene audio.side_gain,
applied before the PCM clip, part of the record / snapshot, so Replay / Continue use
it) chosen from the measured integral RMS of the steady cycles -- the part AFTER the
first 2 s (the start fills the field, which is not an ordinary transition) -- so that
A and B differ by <= 1 dB there (--calibrate prints the measurement at unit gains,
demos/objects_event_source_report.py measures the result); not equal loudness, no
AGC, no per-strike levelling.
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
from casynth_lab.object_resonators import (ENGINE_ID, OPTIONAL_PARAMS, PARAMS, EV_BIRTHS, EV_DEATHS,   # noqa: E402
                                           EXC_UNIFORM, EXC_POSITION)
from demos.build_n1_demos import record_offline                          # noqa: E402
from demos.build_objects_laplace import require_pinnable                 # noqa: E402

CATALOG_ROOT = ROOT / 'lab_catalog' / 'objects_event_source_modal_2026_09_17'
PREFLIGHT = ROOT / 'memory' / 'research' / 'objects-event-source-modal-preflight-2026-09-17.json'
ROWS = COLS = 32
RATE_HZ = 6.0
SECONDS = 12.0
F0_HZ = 110.0
STEADY_FROM_S = 2.0                   # the level calibration window starts here (after the start burst)
# one constant level factor per side per experiment (measured with --calibrate at side
# gain 1, A - B integral RMS of 2..12 s -> B = 10 ** (diff / 20), 2 decimals); the
# numbers: memory/log/2026-09-17-objects-event-source-modal.md
# measured 2026-09-17 at side gain 1: E1 A-B after 2 s -0.59 dB (whole take -0.39), M1 -0.60 dB (whole -0.56)
SIDE_GAIN = {'oes_e1': dict(A=1.0, B=0.93),
             'oes_m1': dict(A=1.0, B=0.93)}


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def preflight_params(events, excitation):
    """The preflight `settings` + the experiment's Events / Excitation; every registry
    parameter present, in registry order."""
    pf = preflight()['settings']
    det = 0 if str(pf['detector']).lower() == 'own' else 1
    spec = 1 if str(pf['spectrum']).lower() == 'laplace' else 0
    p = dict(detector=det, radius_mul=1.0, spectrum=spec, events=int(events), excitation=int(excitation),
             frequency_scale=220.0, decay_s=float(pf['decay_s']), attack_ms=float(pf['attack_ms']),
             n=int(pf['n']), spread=float(pf['spread']), alpha=float(pf['alpha']), shape=float(pf['shape']),
             harm=float(pf['harm']), fullshape=int(pf['fullshape']), dyn=float(pf['dyn']))
    for k, v in OPTIONAL_PARAMS.items():
        p.setdefault(k, v)
    return {k: p[k] for k in (q[0] for q in PARAMS)}


def case_cells(case):
    return [[int(r), int(c)] for r, c in preflight()['cases'][case]['cells']]


SETTINGS_TEXT = ('Own, Laplace, full 1, part 3, spread 1, harm 0.87, Decay 1.39 s, Attack 4 ms, shape 0, '
                 'alpha 0, dyn 0')

CASES = [
    dict(id='oes_e1', title='E1 - Рождения / смерти: Events Births / Deaths (Octagon II p5)',
         case='E1', events=dict(A=EV_BIRTHS, B=EV_DEATHS), excitation=dict(A=EXC_UNIFORM, B=EXC_UNIFORM),
         side_a='А — Objects / Laplace, Events Births, Excitation Uniform',
         side_b='Б — Objects / Laplace, Events Deaths, Excitation Uniform',
         note='Octagon II (период 5) из preflight; 12 с, 6 поколений/с, f0 = 110 Hz; обе стороны — ' + SETTINGS_TEXT +
         '. Один банк на весь период. За пять переходов рождения [8,16,0,8,8], смерти [0,16,8,0,16]: в фазе без '
         'нового импульса хвост продолжает звучать. Стартовое заполнение поля даёт в А один пакет из всех клеток, '
         'в Б — ни одного (первый удар Б — на первом переходе со смертями). Различается только тип возбуждающего '
         'события и постоянная калибровка уровня стороны (по RMS после первых 2 с).',
         hypothesis='Гипотеза - В А фигуру возбуждают только рождения, в Б только смерти. У этого осциллятора они '
         'происходят в разных фазах, поэтому ожидаем разные места акцентов и пауз в повторяющейся фразе при '
         'одинаковых резонансах. Проверяем, слышно ли отличие рисунка и какой вариант интереснее при игре.'),
    dict(id='oes_m1', title='M1 - Общий удар / место рождения: Excitation Uniform / Birth position (Jam p3)',
         case='M1', events=dict(A=EV_BIRTHS, B=EV_BIRTHS), excitation=dict(A=EXC_UNIFORM, B=EXC_POSITION),
         side_a='А — Objects / Laplace, Events Births, Excitation Uniform',
         side_b='Б — Objects / Laplace, Events Births, Excitation Birth position',
         note='Jam (период 3) из preflight; 12 с, 6 поколений/с, f0 = 110 Hz; обе стороны — ' + SETTINGS_TEXT +
         '. Поле, моменты событий, общая сила a, частоты и выходные веса совпадают; фигура штатно распадается на '
         'компоненты и соединяется (одинаковый трекер). В Б каждый пакет делится между модами по участию '
         'родившихся клеток (b по квадратам собственных векторов, sum b^2 = m): у 11-клеточной компоненты '
         'b ≈ [1.29, 0, 1.16], у 16-клеточной [0.78, 0.70, 1.38]. Различается только распределение импульса и '
         'постоянная калибровка уровня стороны (по RMS после первых 2 с).',
         hypothesis='Гипотеза - В А каждое возбуждение поступает одинаково во все выбранные моды фигуры. В Б его '
         'спектральный состав зависит от того, в каких клетках произошли рождения. Ожидаем, что отдельные удары в '
         'повторяющемся рисунке будут отличаться окраской и выраженностью высот сильнее, чем в А. Проверяем, '
         'слышится ли связь места события с характером удара и даёт ли она интересный рисунок при игре.'),
]


def objects_side(events, excitation):
    return (ENGINE_ID, preflight_params(events, excitation))


def listen_text(case):
    return ('1 / 2: ' + case['side_a'] + ' / ' + case['side_b'] + '. Notes: hypothesis and your listening '
            'result. Events / Excitation are the buttons of the Objects panel.')


def scene_for(case, side_gain=None):
    (ea, pa) = objects_side(case['events']['A'], case['excitation']['A'])
    (eb, pb) = objects_side(case['events']['B'], case['excitation']['B'])
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=ROWS, cols=COLS),
                cells=case_cells(case['case']), rule='B3/S23', boundary='torus', rate_hz=RATE_HZ,
                audio=dict(f0_hz=F0_HZ, level=1.0,
                           side_gain=dict(side_gain if side_gain is not None else SIDE_GAIN[case['id']])),
                variants=dict(A=dict(engine_id=ea, engine_params=dict(pa)),
                              B=dict(engine_id=eb, engine_params=dict(pb))),
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


def rms_db(pcm):
    y = np.asarray(pcm, np.float64) / 32767.0
    return 20.0 * math.log10(max(math.sqrt(float(np.mean(y * y))), 1e-12)) if y.size else -240.0


def calibrate(seconds=SECONDS, steady_from=STEADY_FROM_S):
    """RMS of both sides of every experiment at side gain 1 -- the whole take and the
    steady part after `steady_from` -- and the B factor that would equalise the steady
    part (printed; the constants above are set by hand from it)."""
    out = {}
    for case in CASES:
        doc = scene_for(case, side_gain=dict(A=1.0, B=1.0))
        runner = DemoRunner(scene_from_doc(doc))
        runner.post('start', at=0)
        pcm = {s: [] for s in ('A', 'B')}
        peak = {s: 0.0 for s in ('A', 'B')}
        for _ in range(int(math.ceil(seconds * SR / BLOCK))):
            blk = runner.next_block()
            for s in ('A', 'B'):
                pcm[s].append(blk.get(s))
                peak[s] = max(peak[s], runner.sides[s].peak)
        y = {s: np.concatenate(pcm[s]) for s in ('A', 'B')}
        k0 = int(steady_from * SR)
        a, b = rms_db(y['A']), rms_db(y['B'])
        a2, b2 = rms_db(y['A'][k0:]), rms_db(y['B'][k0:])
        out[case['id']] = dict(rms_a_db=a, rms_b_db=b, a_minus_b_db=a - b,
                               steady_rms_a_db=a2, steady_rms_b_db=b2, steady_a_minus_b_db=a2 - b2, peak=peak,
                               b_gain_for_equal_steady_rms=round(10.0 ** ((a2 - b2) / 20.0), 2),
                               clip_blocks=dict(runner.snapshot()['clip_blocks']))
        print(f"  {case['id']}: whole A {a:.2f} B {b:.2f} dB (A-B {a - b:+.2f});  after {steady_from:g} s "
              f"A {a2:.2f} B {b2:.2f} dB (A-B {a2 - b2:+.2f}) -> B gain {out[case['id']]['b_gain_for_equal_steady_rms']}"
              f"  peaks A {peak['A']:.3f} B {peak['B']:.3f}  clip {out[case['id']]['clip_blocks']}", flush=True)
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
    for case in reversed(CASES):                  # newest first on screen = REQ order
        started = time.monotonic()
        note = case['side_a'] + ' / ' + case['side_b'] + '. ' + case['note']
        rid, snap = record_offline(cat, scene_for(case), SECONDS, [], case['title'], note)
        cat.write_notes(rid, case['hypothesis'] + '\n\n')
        results.append((rid, case['id'], dict(snap['clip_blocks']), snap['gen']))
        print(f"  {rid}  {case['id']}  clip={snap['clip_blocks']} gen={snap['gen']}", flush=True)
        time.sleep(max(0.0, 1.05 - (time.monotonic() - started)))   # ids carry a 1 s stamp
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the Objects event-source / modal-excitation scenes and catalog")
    ap.add_argument('--root', default=str(CATALOG_ROOT))
    ap.add_argument('--scenes-only', action='store_true')
    ap.add_argument('--calibrate', action='store_true', help="measure the RMS of both sides at side gain 1")
    a = ap.parse_args(argv)
    if a.calibrate:
        calibrate()
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
