"""Build the N3 listening catalog: the field tunes the resonances and its events strike
them (REQ memory/req-network-combined-n3-2026-09-16.md).  One engine `ca_tuned_events`
on both sides; A = hits on FIXED resonances (field_tuning 0), B = the same hits plus
the field tuning the banks (field_tuning 1).

    python demos/build_n3_combined.py [--root DIR] [--scenes-only]

Scenes demos/n3_*.json are (re)written; the catalog is built only into a directory
that holds no records yet (records and Notes the user added are never overwritten).
`run_network_n3.bat` opens the bench on the catalog screen of the default root.
All experiments: Conway B3/S23 on a 32 x 32 torus, evolution running, both sides
always on the same field; the cells are the Researcher's preflight fixtures
(memory/research/network-n3-preflight-2026-09-16.json); the hypotheses of the REQ
are written into Notes verbatim.

  N3.1  Two cycles   24 s, 6 gen/s: Octagon II (period 5) for the first 12 s, then at the
                     first block boundary at or after 12 s the WHOLE field is replaced by
                     the Tumbler (period 14) with one atomic `set_cells` command (journalled;
                     audio states and the CA clock are kept), 12 more seconds.
  N3.2  Travel       12 s, 16 gen/s: glider from (28, 28), crosses the outer edge, one torus
                     round in 8 s.
  N3.3  Growth       12 s, 6 gen/s: R-pentomino from (14, 14).
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

from casynth_config import SR                                            # noqa: E402
from casynth_lab import scene_from_doc                                   # noqa: E402
from casynth_lab.catalog import Catalog                                  # noqa: E402
from casynth_lab.tuned_events import ENGINE_ID                           # noqa: E402
from demos.build_n1_demos import record_offline                          # noqa: E402

CATALOG_ROOT = ROOT / 'lab_catalog' / 'network_n3_combined_2026_09_16'
ROWS = COLS = 32
DECAY_S = 0.8
PARAMS_A = dict(field_tuning=0, decay_s=DECAY_S)
PARAMS_B = dict(field_tuning=1, decay_s=DECAY_S)
SIDE_A = 'А — удары, фиксированные резонансы'
SIDE_B = 'Б — удары + настройка от поля'
LISTEN = ('1 / 2: ' + SIDE_A + ' / ' + SIDE_B + '. Notes: hypothesis and your listening '
          'result. Draw, pause and run the CA on either side.')
SWAP_AT_S = 12.0


def octagon_ii(r0=12, c0=12):
    """Octagon II, period 5, bounding box 8 x 8 placed at (r0, c0)."""
    rel = ((0, 3), (0, 4), (1, 2), (1, 5), (2, 1), (2, 6), (3, 0), (3, 7),
           (4, 0), (4, 7), (5, 1), (5, 6), (6, 2), (6, 5), (7, 3), (7, 4))
    return sorted([r0 + r, c0 + c] for r, c in rel)


def tumbler(r0=13, c0=12):
    """Tumbler, period 14, bounding box 6 x 7 placed at (r0, c0)."""
    rel = ((0, 0), (0, 1), (0, 5), (0, 6), (1, 0), (1, 2), (1, 4), (1, 6),
           (2, 0), (2, 2), (2, 4), (2, 6), (3, 2), (3, 4),
           (4, 1), (4, 2), (4, 4), (4, 5), (5, 1), (5, 2), (5, 4), (5, 5))
    return sorted([r0 + r, c0 + c] for r, c in rel)


def glider(shift=28):
    return [[r + shift, c + shift] for r, c in ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2))]


def r_pentomino(shift=14):
    return [[r + shift, c + shift] for r, c in ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1))]


def field_command(cells, at_s, rows=ROWS, cols=COLS):
    """One atomic `set_cells` that makes the WHOLE field equal to `cells` (every cell
    listed, so the result does not depend on the field found at that moment),
    applied at the first block boundary at or after `at_s`."""
    live = {(int(r), int(c)) for r, c in cells}
    full = [[r, c, int((r, c) in live)] for r in range(rows) for c in range(cols)]
    return ('set_cells', int(round(at_s * SR)), dict(cells=full))


CASES = [
    dict(id='n3_cycles', title='N3.1 - Два цикла: Octagon II, затем Tumbler, 6 поколений/с',
         cells=octagon_ii(), rate=6.0, seconds=24.0,
         commands=[field_command(tumbler(), SWAP_AT_S)],
         note=SIDE_A + ' / ' + SIDE_B + '. Смена фигур: первые 12 с — Octagon II (период 5, '
         'размещение (12,12)); на первой границе блока не раньше 12 с всё поле заменено на Tumbler '
         '(период 14, размещение (13,12)) одной командой set_cells (в журнале), дальше ещё 12 с. '
         'Аудиосостояния и часы КА при замене сохранены; одноразовый переход при замене — не рисунок '
         'цикла. Обе стороны получают одно поле.',
         hypothesis='Гипотеза - В обоих вариантах удары идут по изменениям поколений. В Б настройка '
         'резонаторов должна добавлять повторяющийся рисунок окраски и высоты откликов: один у октагона '
         'в первой половине записи, другой у тумблера во второй. Проверяем, становятся ли эти циклы '
         'различимее, чем в А с фиксированными резонансами.'),
    dict(id='n3_travel', title='N3.2 - Движение: glider, 16 поколений/с', cells=glider(), rate=16.0,
         seconds=12.0, commands=[],
         note=SIDE_A + ' / ' + SIDE_B + '. Glider с исходной позицией (28,28): пять живых клеток и '
         'четыре события на каждый шаг; пересекает внешний край и проходит тор за 8 с (128 поколений). '
         '12 с от старта.',
         hypothesis='Гипотеза - В А движение меняет распределение ударов по готовым банкам. В Б к этому '
         'добавится движение самих резонансов: ожидаются перкуссионные отклики с поднимающейся и '
         'опускающейся окраской и меняющимися хвостами. Проверяем, слышно ли совместное действие '
         'настройки и ударов и интересно ли оно в игре.'),
    dict(id='n3_growth', title='N3.3 - Рост: R-pentomino, 6 поколений/с', cells=r_pentomino(),
         rate=6.0, seconds=12.0, commands=[],
         note=SIDE_A + ' / ' + SIDE_B + '. R-pentomino с исходной позицией (14,14), нерегулярный рост, '
         '12 с от старта.',
         hypothesis='Гипотеза - При нерегулярном росте Б будет менять характер перекрывающихся откликов '
         'вместе с состоянием поля, тогда как А меняет главным образом распределение возбуждения между '
         'фиксированными банками. Ожидаем более разнообразную последовательность звонких и гулких '
         'откликов; проверяем, не остаётся ли результат тем же однообразным стуком с переменной '
         'громкостью.'),
]


def scene_for(case):
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=ROWS, cols=COLS),
                cells=[[int(r), int(c)] for r, c in case['cells']], rule='B3/S23',
                boundary='torus', rate_hz=float(case['rate']),
                audio=dict(f0_hz=110.0, level=1.0),
                variants=dict(A=dict(engine_id=ENGINE_ID, engine_params=dict(PARAMS_A)),
                              B=dict(engine_id=ENGINE_ID, engine_params=dict(PARAMS_B))),
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
        rid, snap = record_offline(cat, scene_for(case), case['seconds'], list(case['commands']),
                                   case['title'], case['note'])
        cat.write_notes(rid, case['hypothesis'] + '\n\n')
        results.append((rid, case['id'], dict(snap['clip_blocks']), snap['gen']))
        print(f"  {rid}  {case['id']}  clip={snap['clip_blocks']} gen={snap['gen']}", flush=True)
        time.sleep(max(0.0, 1.05 - (time.monotonic() - started)))   # ids carry a 1 s stamp
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the N3 listening scenes and catalog")
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
