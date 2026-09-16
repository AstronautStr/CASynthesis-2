"""Build the N2 listening catalog: field events exciting a delay network (B) against the
field retuning the Gutter resonators (A), both through the periodic readout
(REQ memory/req-network-events-n2-2026-09-16.md).

    python demos/build_n2_events.py [--root DIR] [--scenes-only]

Scenes demos/n2_*.json are (re)written; the catalog is built only into a directory
that holds no records yet (records and Notes the user added are never overwritten).
`run_network_n2.bat` opens the bench on the catalog screen of the default root.
Three experiments, 12 s each from a fresh start, Conway B3/S23 on a 32 x 32 torus,
evolution running; the hypotheses of the REQ are written into Notes verbatim.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from casynth_lab import scene_from_doc                                   # noqa: E402
from casynth_lab.catalog import Catalog                                  # noqa: E402
from casynth_lab.gutter_field_periodic import ENGINE_ID as A_ID          # noqa: E402
from casynth_lab.event_network import ENGINE_ID as B_ID                  # noqa: E402
from demos.build_n1_demos import record_offline                          # noqa: E402

CATALOG_ROOT = ROOT / 'lab_catalog' / 'network_n2_events_2026_09_16'
ROWS = COLS = 32
SECONDS = 12.0
PARAMS_A = dict(scale=1.0, depth=1.0, interaction=127, freeze_ca=0)
PARAMS_B = dict(rho=0.88)
SIDE_A = 'А — поле меняет частоты'
SIDE_B = 'Б — события возбуждают сеть'
LISTEN = ('1 / 2: ' + SIDE_A + ' / ' + SIDE_B + '. Notes: hypothesis and your listening '
          'result. Draw, pause and run the CA on either side.')


def pulsar(shift=9):
    a, b = (0, 5, 7, 12), (2, 3, 4, 8, 9, 10)
    cells = {(r, c) for r in a for c in b} | {(r, c) for r in b for c in a}
    return sorted([r + shift, c + shift] for r, c in cells)


def glider(shift=28):
    return [[r + shift, c + shift] for r, c in ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2))]


def r_pentomino(shift=14):
    return [[r + shift, c + shift] for r, c in ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1))]


CASES = [
    dict(id='n2_rhythm', title='N2.1 - Ритм: pulsar, 6 поколений/с', cells=pulsar(), rate=6.0,
         note=SIDE_A + ' / ' + SIDE_B + '. Pulsar (период 3 поколения, цикл 0.5 с; '
         '32-56 событий на шаг), 12 с от старта, эволюция включена.',
         hypothesis='Гипотеза - Периодическая фигура в А будет повторять изменения высоты и '
         'окраски, а в Б — давать повторяющийся рисунок атак со звенящими затуханиями между '
         'ними. Проверяем, слышен ли этот рисунок как самостоятельное поведение, а не только '
         'очередное ровное жужжание.'),
    dict(id='n2_travel', title='N2.2 - Движение: glider, 16 поколений/с', cells=glider(), rate=16.0,
         note=SIDE_A + ' / ' + SIDE_B + '. Glider (5 клеток, 4 события на шаг; полный обход '
         'тора за 128 поколений = 8 с; в первые секунды пересекает внешний край), 12 с от старта.',
         hypothesis='Гипотеза - При движении фигуры в Б должны меняться окраска и расположение '
         'звенящих откликов, хотя число событий остаётся одинаковым. В А ожидается движение '
         'высоты и окраски. Проверяем, даёт ли положение события слышимый характер и не '
         'выделяется ли край поля отдельным всплеском.'),
    dict(id='n2_growth', title='N2.3 - Рост: R-pentomino, 6 поколений/с', cells=r_pentomino(),
         rate=6.0,
         note=SIDE_A + ' / ' + SIDE_B + '. R-pentomino (нерегулярная эволюция; на поколении 72 '
         'остаётся 71 живая клетка, следующий шаг даёт 66 событий), 12 с от старта.',
         hypothesis='Гипотеза - При нерегулярном росте в Б отдельные атаки будут переходить в '
         'перекрывающуюся дробную, шероховатую фактуру, а в А — в изменения высоты и плотности '
         'непрерывного звука. Проверяем, даёт ли событийный способ разные типы поведения при '
         'редких и плотных изменениях, а не только прибавляет громкость.'),
]


def scene_for(case):
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=ROWS, cols=COLS),
                cells=[[int(r), int(c)] for r, c in case['cells']], rule='B3/S23',
                boundary='torus', rate_hz=float(case['rate']),
                audio=dict(f0_hz=110.0, level=1.0),
                variants=dict(A=dict(engine_id=A_ID, engine_params=dict(PARAMS_A)),
                              B=dict(engine_id=B_ID, engine_params=dict(PARAMS_B))),
                initial_side='A', listen=LISTEN)


def write_scenes():
    paths = []
    for case in CASES:
        doc = scene_for(case)
        scene_from_doc(doc)                       # strict validation before writing
        path = ROOT / 'demos' / (case['id'] + '.json')
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        paths.append(path)
    return paths


def build(root=CATALOG_ROOT):
    root = Path(root)
    if root.exists() and any(not p.name.startswith('.') for p in root.iterdir()):
        raise SystemExit('error: target catalog already holds records: ' + str(root) +
                         '\n(rebuild into a new directory with --root; records and Notes the '
                         'user added are never overwritten)')
    (root / '.tmp').mkdir(parents=True, exist_ok=True)
    cat = Catalog(str(root))
    results = []
    for case in reversed(CASES):                  # newest first on screen = REQ order
        started = time.monotonic()
        rid, snap = record_offline(cat, scene_for(case), SECONDS, [], case['title'], case['note'])
        cat.write_notes(rid, case['hypothesis'] + '\n\n')
        results.append((rid, case['id'], dict(snap['clip_blocks']), snap['gen']))
        print(f"  {rid}  {case['id']}  clip={snap['clip_blocks']} gen={snap['gen']}", flush=True)
        time.sleep(max(0.0, 1.05 - (time.monotonic() - started)))   # ids carry a 1 s stamp
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the N2 listening scenes and catalog")
    ap.add_argument('--root', default=str(CATALOG_ROOT))
    ap.add_argument('--scenes-only', action='store_true')
    a = ap.parse_args(argv)
    paths = write_scenes()
    print(f"scenes: {len(paths)} files in {ROOT / 'demos'}")
    if a.scenes_only:
        return 0
    res = build(a.root)
    print(f"catalog: {len(res)} records in {a.root}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
