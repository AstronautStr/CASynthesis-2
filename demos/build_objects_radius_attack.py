"""Build the Objects radius / attack catalog: three A/B experiments (two radius pairs,
one attack pair) and the user's Radius 0.34 preset, all Objects / Laplace on both
sides (REQ memory/req-objects-radius-attack-2026-09-17.md).

    python demos/build_objects_radius_attack.py [--root DIR] [--scenes-only] [--calibrate]

Scenes demos/ora_*.json are (re)written; the catalog is built only into a directory
that holds no records yet (records and Notes the user added are never overwritten).
`run_objects_radius_attack.bat` opens the bench on the catalog screen of the default
root.  Every experiment: Conway B3/S23 on a 32 x 32 torus, 6 generations / s, 12 s,
f0 = 110 Hz, both sides on the same field; the cells are the Researcher's preflight
fixtures (memory/research/objects-radius-attack-preflight-2026-09-17.json,
`cases.<name>.groups`); the hypotheses of the REQ are written into Notes verbatim.

Both sides: `ca_object_resonators`, the settings of the user's screenshot (preflight
`settings`): Disk, Spectrum Laplace, Decay 1.39, part 3, spread 1, alpha 1.82,
shape 0.95, harm 0.87, full 1, dyn 0.87 (Freq scale inactive in this mode).
Radius x / Attack and the slider range of Radius x are what the experiments set:

  R1  Neighbour excites a still figure   the still 17-cell figure + the blinker at
                                         (9,17)-(9,19); A Radius 1, B Radius 2; Attack 0
  R2  Two periods: own and shared        Jam p3 from (5,3) + Octagon II p5 from (17,19);
                                         A Radius 1, B Radius 32; Attack 0; slider 0.25..32
  A1  Sharpness of the excitation        the same field, Radius 32 both; A Attack 0,
                                         B Attack 4 ms; slider 0.25..32
  U0  User preset Radius 0.34            the same field, Radius 0.34 on both sides,
                                         Attack 0, slider 0.25..1 (steps of 0.01 are
                                         6 px apart); not a radius A/B

Levels: SIDE_GAIN is ONE constant factor per side per experiment (scene audio.side_gain,
applied before the PCM clip, part of the record / snapshot, so Replay / Continue use
it) chosen from the measured integral RMS of the ready 12 s so that A and B differ by
<= 1 dB (--calibrate prints the measurement at unit gains, demos/
objects_radius_attack_report.py measures the result); not equal loudness.
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
from casynth_lab.object_resonators import ENGINE_ID, OPTIONAL_PARAMS     # noqa: E402
from demos.build_n1_demos import record_offline                          # noqa: E402
from demos.build_objects_laplace import require_pinnable                 # noqa: E402

CATALOG_ROOT = ROOT / 'lab_catalog' / 'objects_radius_attack_2026_09_17'
PREFLIGHT = ROOT / 'memory' / 'research' / 'objects-radius-attack-preflight-2026-09-17.json'
ROWS = COLS = 32
RATE_HZ = 6.0
SECONDS = 12.0
F0_HZ = 110.0
USER_RADIUS = 0.34                    # the user's setting of the screenshot (a preset, not an A/B)
# one constant level factor per side per experiment (measured with --calibrate at side
# gain 1, A - B integral RMS of the ready 12 s -> B = 10 ** (diff / 20), 2 decimals):
# see the session log memory/log/2026-09-17-objects-radius-attack.md for the numbers
# measured 2026-09-17 at side gain 1 (commit 8a8fbfb): R1 A-B -3.80 dB, R2 -3.26 dB, A1 +6.25 dB
# (Attack 4 ms is quieter, no compensation inside the knob), U0 0.00 dB (both sides equal)
SIDE_GAIN = {'ora_r1': dict(A=1.0, B=0.65),
             'ora_r2': dict(A=1.0, B=0.69),
             'ora_a1': dict(A=1.0, B=2.05),
             'ora_user034': dict(A=1.0, B=1.0)}


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def screenshot_params(radius_mul, attack_ms):
    """The settings of the user's screenshot (preflight `settings`) + the experiment's
    Radius x / Attack; every registry parameter present (older optional ones too)."""
    pf = preflight()['settings']
    p = dict(detector=int(pf['detector']), radius_mul=float(radius_mul), spectrum=int(pf['spectrum']),
             frequency_scale=float(pf['frequency_scale']), decay_s=float(pf['decay_s']),
             attack_ms=float(attack_ms), n=int(pf['n']), spread=float(pf['spread']), alpha=float(pf['alpha']),
             shape=float(pf['shape']), harm=float(pf['harm']), fullshape=int(pf['fullshape']),
             dyn=float(pf['dyn']))
    for k, v in OPTIONAL_PARAMS.items():
        p.setdefault(k, v)
    return p


def group_cells(case):
    """All cells of the case's families (the joint field)."""
    return [[int(r), int(c)] for g in preflight()['cases'][case]['groups'] for r, c in g]


SETTINGS_TEXT = 'Disk, Laplace, Decay 1.39, part 3, spread 1, alpha 1.82, shape 0.95, harm 0.87, full 1, dyn 0.87'

CASES = [
    dict(id='ora_r1', title='R1 - Сосед возбуждает неподвижную фигуру: Radius x 1 / 2 (17 клеток + blinker)',
         case='receiver', radius=dict(A=1.0, B=2.0), attack=dict(A=0.0, B=0.0), range=(0.25, 4.0),
         side_a='А — Objects / Laplace, Radius x 1, Attack 0',
         side_b='Б — Objects / Laplace, Radius x 2, Attack 0',
         note='Неподвижная 17-клеточная фигура и blinker (9,17)-(9,19) из preflight; 12 с, 6 поколений/с, '
         'f0 = 110 Hz; обе стороны — ' + SETTINGS_TEXT + '. Blinker звучит в А и Б. После стартового хвоста '
         'приёмник в А не получает пакетов, в Б принимает четыре чужих изменения за поколение (одна посылка '
         'на поколение). Различается только Radius x и постоянная калибровка уровня стороны.',
         hypothesis='Гипотеза - В А после стартового хвоста остаётся блинкер; в Б его изменения должны '
         'дополнительно возбуждать неподвижную соседнюю фигуру. Ожидаем дополнительный повторяющийся '
         'резонансный отклик при той же работе блинкера. Проверяем, слышно ли появление второго голоса при '
         'расширении круга.'),
    dict(id='ora_r2', title='R2 - Два периода: самостоятельные и общий приём: Radius x 1 / 32 (Jam p3 + Octagon II p5)',
         case='p3_p5', radius=dict(A=1.0, B=32.0), attack=dict(A=0.0, B=0.0), range=(0.25, 32.0),
         side_a='А — Objects / Laplace, Radius x 1, Attack 0',
         side_b='Б — Objects / Laplace, Radius x 32, Attack 0',
         note='Jam (период 3) от (5,3) и Octagon II (период 5) от (17,19) из preflight; 12 с, 6 поколений/с, '
         'f0 = 110 Hz; обе стороны — ' + SETTINGS_TEXT + '. Диапазон ползунка Radius x 0.25..32. При x1 между '
         'двумя семействами нет приёма событий; при x32 круг каждой звучащей компоненты покрывает всё поле '
         '(рамка в цвете фигуры), и все банки принимают изменения всего поля. x8 — проверенное промежуточное '
         'положение для игры. Jam в отдельных фазах состоит из нескольких компонент (банков).',
         hypothesis='Гипотеза - В А два осциллятора сохраняют свои области приёма. В Б события каждого должны '
         'возбуждать также банки второго, поэтому ожидаем больше общих акцентов и более плотное наложение '
         'звуков. Проверяем, слышно ли взаимное влияние и остаются ли различимы два ритмических рисунка; '
         'очень большой радиус может сделать их менее самостоятельными.'),
    dict(id='ora_a1', title='A1 - Резкость возбуждения: Attack 0 / 4 ms (Jam p3 + Octagon II p5, Radius x 32)',
         case='p3_p5', radius=dict(A=32.0, B=32.0), attack=dict(A=0.0, B=4.0), range=(0.25, 32.0),
         side_a='А — Objects / Laplace, Radius x 32, Attack 0',
         side_b='Б — Objects / Laplace, Radius x 32, Attack 4 ms',
         note='То же поле, что в R2; Radius x 32 с обеих сторон (полное покрытие тора), 12 с, 6 поколений/с, '
         'f0 = 110 Hz; обе стороны — ' + SETTINGS_TEXT + '. Все моменты возбуждения и спектральные настройки '
         'одинаковы; различается только сглаживание начала возбуждающего импульса (Attack = время 10-90 % '
         'ступенчатого отклика сглаживания) и постоянная калибровка уровня стороны. Компенсации громкости '
         'внутри ручки нет: сглаживание меняет яркость и уровень.',
         hypothesis='Гипотеза - При тех же моментах возбуждения в Б начало звуков станет мягче и менее '
         'щёлкающим, а в А останется нынешняя резкая атака. Ожидаем также изменение яркости. Проверяем, '
         'помогает ли атака сохранить интересный рисунок при плотном взаимном возбуждении и сделать его '
         'удобнее для игры.'),
    dict(id='ora_user034', title='U0 - Пресет пользователя: Radius x 0.34 (Jam p3 + Octagon II p5, обе стороны)',
         case='p3_p5', radius=dict(A=USER_RADIUS, B=USER_RADIUS), attack=dict(A=0.0, B=0.0), range=(0.25, 1.0),
         side_a='А — Objects / Laplace, Radius x 0.34, Attack 0',
         side_b='Б — Objects / Laplace, Radius x 0.34, Attack 0',
         note='Настройка со скриншота пользователя (2026-09-17): Radius x 0.34 и ' + SETTINGS_TEXT +
         '; поле R2/A1 (Jam p3 + Octagon II p5), 12 с, 6 поколений/с, f0 = 110 Hz. Обе стороны одинаковы: это '
         'доступный пресет для дальнейшей игры, не радиусный А/Б. Диапазон ползунка Radius x 0.25..1: шаги '
         '0.01 отстоят на 6 px, Min / Max можно сузить ещё.',
         hypothesis='Пресет - исходная точка пользователя, а не гипотеза: малый радиус отбирает события '
         '(прибавки 0.01-0.02 добавляют «ноты»). Здесь удобно проверять узкие диапазоны Min / Max и Attack '
         'на знакомом звуке; результат впишите ниже.'),
]


def objects_side(radius_mul, attack_ms):
    return (ENGINE_ID, screenshot_params(radius_mul, attack_ms))


def listen_text(case):
    return ('1 / 2: ' + case['side_a'] + ' / ' + case['side_b'] + '. Notes: hypothesis and your listening '
            'result. Radius x: Min / Max under the slider set its range for the selected side.')


def scene_for(case, side_gain=None):
    (ea, pa) = objects_side(case['radius']['A'], case['attack']['A'])
    (eb, pb) = objects_side(case['radius']['B'], case['attack']['B'])
    lo, hi = case['range']
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=ROWS, cols=COLS),
                cells=group_cells(case['case']), rule='B3/S23', boundary='torus', rate_hz=RATE_HZ,
                audio=dict(f0_hz=F0_HZ, level=1.0,
                           side_gain=dict(side_gain if side_gain is not None else SIDE_GAIN[case['id']])),
                variants=dict(A=dict(engine_id=ea, engine_params=dict(pa)),
                              B=dict(engine_id=eb, engine_params=dict(pb))),
                param_ranges={s: {ENGINE_ID: {'radius_mul': [float(lo), float(hi)]}} for s in ('A', 'B')},
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
    return 20.0 * math.log10(max(math.sqrt(float(np.mean(y * y))), 1e-12))


def calibrate(seconds=SECONDS):
    """RMS of both sides of every experiment at side gain 1 and the B factor that
    would equalise them (printed; the constants above are set by hand from it)."""
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
        a, b = rms_db(np.concatenate(pcm['A'])), rms_db(np.concatenate(pcm['B']))
        out[case['id']] = dict(rms_a_db=a, rms_b_db=b, a_minus_b_db=a - b, peak=peak,
                               b_gain_for_equal_rms=round(10.0 ** ((a - b) / 20.0), 2),
                               clip_blocks=dict(runner.snapshot()['clip_blocks']))
        print(f"  {case['id']}: A {a:.2f} dB  B {b:.2f} dB  A-B {a - b:+.2f} dB  -> B gain {out[case['id']]['b_gain_for_equal_rms']}"
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
    ap = argparse.ArgumentParser(description="build the Objects radius / attack scenes and catalog")
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
