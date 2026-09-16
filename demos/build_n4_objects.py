"""Build the N4 listening catalog: every figure a resonator bank of its own Laplacian,
struck by the changes inside its circle (REQ memory/req-object-resonators-n4-2026-09-16.md).

    python demos/build_n4_objects.py [--root DIR] [--scenes-only]

Scenes demos/n4_*.json are (re)written; the catalog is built only into a directory
that holds no records yet (records and Notes the user added are never overwritten).
`run_object_resonators_n4.bat` opens the bench on the catalog screen of the default
root.  Both experiments: Conway B3/S23 on a 32 x 32 torus, evolution running at
6 generations / s, 12 s, both sides always on the same field; the cells are the
Researcher's preflight fixtures (memory/research/object-resonators-n4-preflight-2026-09-16.json,
`cases`); the hypotheses of the REQ are written into Notes verbatim.

  N4.1  Spectrum of the figure itself   glider from (28, 28), crosses the seam.
        A = the N3 side the user found interesting (`ca_tuned_events`, Field tuning, decay 0.8)
        B = N4 Disk (`ca_object_resonators`, detector Disk, scale 220, decay 0.8)
  N4.2  A figure hears its neighbour      a still 17-cell figure + a blinker inside its circle
        (centre (11.7647, 11.7647), R = 5.32962).  A = N4 Own, B = N4 Disk.
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
from casynth_lab.object_resonators import ENGINE_ID, DET_OWN, DET_DISK   # noqa: E402
from casynth_lab.tuned_events import ENGINE_ID as N3_ENGINE_ID           # noqa: E402
from demos.build_n1_demos import record_offline                          # noqa: E402

CATALOG_ROOT = ROOT / 'lab_catalog' / 'object_resonators_n4_2026_09_16'
PREFLIGHT = ROOT / 'memory' / 'research' / 'object-resonators-n4-preflight-2026-09-16.json'
ROWS = COLS = 32
RATE_HZ = 6.0
SECONDS = 12.0
DECAY_S = 0.8
SCALE_HZ = 220.0
RADIUS_MUL = 1.0
N4_OWN = dict(detector=DET_OWN, radius_mul=RADIUS_MUL, frequency_scale=SCALE_HZ, decay_s=DECAY_S)
N4_DISK = dict(detector=DET_DISK, radius_mul=RADIUS_MUL, frequency_scale=SCALE_HZ, decay_s=DECAY_S)
N3_FIELD = dict(field_tuning=1, decay_s=DECAY_S)


def preflight_cells(case):
    with open(PREFLIGHT, encoding='utf-8') as f:
        doc = json.load(f)
    return [[int(r), int(c)] for r, c in doc['cases'][case]['cells']]


def glider_cells():
    return preflight_cells('glider')


def neighbors_cells():
    return preflight_cells('neighbors')


CASES = [
    dict(id='n4_spectrum', title='N4.1 - Спектр самой фигуры: glider, 6 поколений/с',
         cells=glider_cells,
         A=(N3_ENGINE_ID, N3_FIELD), B=(ENGINE_ID, N4_DISK),
         side_a='А — N3 Tuned (удары + настройка от поля)',
         side_b='Б — N4 Objects, детектор Disk',
         note='Glider с исходной позицией (28,28), проходит через шов тора; 12 с, 6 поколений/с. '
         'А — прежний интересный Б N3 (ca_tuned_events, field_tuning=1, decay 0.8); '
         'Б — N4: частоты от лапласиана самой фигуры, детектор — круг от центра масс (Disk). '
         'Сравнение двух способов сонификации в целом. Обе стороны получают одно поле.',
         hypothesis='Гипотеза - В Б спектр принадлежит глайдеру и движется вместе с ним: ожидаем '
         'повторяющееся чередование двух окрасок отклика при смене его фаз, сохраняющееся в разных '
         'местах поля. В А положение относительно прежних датчиков продолжает перестраивать звук. '
         'Проверяем, становится ли связь формы и звучания понятнее и интереснее; каждая из четырёх '
         'фаз не обязана давать отдельную высоту.'),
    dict(id='n4_neighbor', title='N4.2 - Фигура слышит соседа: неподвижная фигура + blinker, 6 поколений/с',
         cells=neighbors_cells,
         A=(ENGINE_ID, N4_OWN), B=(ENGINE_ID, N4_DISK),
         side_a='А — N4 Objects, детектор Own',
         side_b='Б — N4 Objects, детектор Disk',
         note='Связная неподвижная фигура из 17 клеток (центр (11.7647, 11.7647), R = 5.32962) и '
         'отдельный blinker (9,14)-(9,16) внутри её круга; 12 с, 6 поколений/с. А — детектор Own '
         '(только свои клетки), Б — детектор Disk (круг); остальные параметры одинаковы. Фигуры не '
         'сливаются и эволюционируют независимо: blinker даёт четыре изменения за поколение внутри '
         'круга приёмника, сам приёмник не меняется.',
         hypothesis='Гипотеза - В А после стартового хвоста остаются удары блинкера. В Б те же '
         'изменения должны поддерживать дополнительный резонансный отклик неподвижной соседней фигуры, '
         'чей круг охватывает блинкер. Ожидаем более многосоставный, звенящий отклик; проверяем, слышна '
         'ли такая связь фигур и интересна ли она в игре. При переносе блинкера за круг дополнительный '
         'отклик должен затухнуть.'),
]


def listen_text(case):
    return ('1 / 2: ' + case['side_a'] + ' / ' + case['side_b'] + '. Notes: hypothesis and your '
            'listening result. Draw, pause and run the CA on either side.')


def scene_for(case):
    (ea, pa), (eb, pb) = case['A'], case['B']
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=ROWS, cols=COLS),
                cells=[[int(r), int(c)] for r, c in case['cells']()], rule='B3/S23',
                boundary='torus', rate_hz=RATE_HZ,
                audio=dict(f0_hz=110.0, level=1.0),
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
        note = case['side_a'] + ' / ' + case['side_b'] + '. ' + case['note']
        rid, snap = record_offline(cat, scene_for(case), SECONDS, [], case['title'], note)
        cat.write_notes(rid, case['hypothesis'] + '\n\n')
        results.append((rid, case['id'], dict(snap['clip_blocks']), snap['gen']))
        print(f"  {rid}  {case['id']}  clip={snap['clip_blocks']} gen={snap['gen']}", flush=True)
        time.sleep(max(0.0, 1.05 - (time.monotonic() - started)))   # ids carry a 1 s stamp
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the N4 listening scenes and catalog")
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
