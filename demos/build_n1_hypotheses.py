"""Build the revised N1 listening catalog, with researcher hypotheses in Notes.

Run: python demos/build_n1_hypotheses.py [--root DIR] [--scenes-only]
Existing records, including the previous N1 catalog and its feedback, are kept.
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

from casynth_lab import scene_from_doc
from casynth_lab.catalog import Catalog
from casynth_lab.gutter_controls import DAMP_ID, LINKS_ID
from demos.build_n1_demos import record_offline

CATALOG_ROOT = ROOT / 'lab_catalog' / 'network_n1_hypotheses_2026_09_16_r2'
PARAMS = dict(scale=1.0, depth=1.0, interaction=127, freeze_ca=0)


def blinkers(shift=0):
    return [[(r + shift) % 32, c] for c in (4, 12, 20, 28) for r in (14, 15, 16)]


def gliders():
    return [[(r + dr) % 32, (c + dc) % 32]
            for r, c in ((3, 3), (3, 19), (19, 11), (19, 27))
            for dr, dc in ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2))]


CASES = [
    dict(id='n1h_rhythm', title='N1.1 - Ритм: влияние демпфирования',
         A='gutter_field', B=DAMP_ID, cells=blinkers(), rate=4.0, seconds=12.0,
         hypothesis='Гипотеза - периодические изменения распределения клеток можно слышать '
         'как ритм изменения тембра: в обеих сторонах поле перестраивает резонаторы, '
         'в Б дополнительно меняет демпфирование узлов. Ожидаю в A более шероховатое '
         'чередование спектральных полос, в Б более ровный и звонкий рисунок; '
         'рисование должно менять этот рисунок в обеих сторонах.'),
    dict(id='n1h_travel', title='N1.2 - Движение: влияние связей',
         A='gutter_field', B=LINKS_ID, cells=gliders(), rate=16.0, seconds=12.0,
         hypothesis='Гипотеза - перемещение фигур при неизменном числе клеток даст '
         'разную звуковую траекторию: в A поле настраивает резонаторы, в Б ещё и связи '
         'сети. Ожидаю в A последовательные смены звонких полос, а в Б более '
         'неровное движение фактуры и медленные волны низкого рокота: меняется влияние '
         'узлов друг на друга, и звук зависит от пройденного пути поля.'),
    dict(id='n1h_return', title='N1.3 - Возврат: память сети',
         A=DAMP_ID, B=LINKS_ID, cells=blinkers(), rate=4.0, seconds=20.0,
         hypothesis='Гипотеза - управление связями даст более длительное последействие '
         'перестановки поля, чем управление демпфированием. В сохранённой записи фигуры переносятся '
         'вниз и обратно каждые четыре секунды без сброса звука. В обеих сторонах поле '
         'также перестраивает резонаторы. В A ожидаю более быстрые смены звонкой '
         'фактуры, в Б затяжные волны и переходы к низкому рокоту, которые могут '
         'сохраняться после возвращения клеток.'),
]


def scene_for(case):
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=32, cols=32),
                cells=case['cells'], rule='B3/S23', boundary='torus', rate_hz=case['rate'],
                audio=dict(f0_hz=110.0, level=1.0),
                variants={s: dict(engine_id=case[s], engine_params=dict(PARAMS))
                          for s in ('A', 'B')}, initial_side='A',
                listen='1 / 2: A / B. Notes: hypothesis and your listening result. '
                       'Draw and run the CA on either side.')


def commands_for(case):
    if case['id'] != 'n1h_return':
        return []
    commands = [('pause', 0, dict(on=True))]
    for k in range(1, 5):
        before = blinkers(0 if k % 2 else 8)
        after = blinkers(8 if k % 2 else 0)
        at = k * 4 * 44100
        commands.extend(('set_cell', at, dict(r=r, c=c, v=0)) for r, c in before)
        commands.extend(('set_cell', at, dict(r=r, c=c, v=1)) for r, c in after)
    return commands


def write_scenes():
    for case in CASES:
        doc = scene_for(case)
        scene_from_doc(doc)
        (ROOT / 'demos' / (case['id'] + '.json')).write_text(
            json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def build(root=CATALOG_ROOT):
    root = Path(root)
    if root.exists() and any(not p.name.startswith('.') for p in root.iterdir()):
        raise SystemExit('Target already holds records: ' + str(root))
    (root / '.tmp').mkdir(parents=True, exist_ok=True)
    cat = Catalog(str(root))
    ids = []
    for case in reversed(CASES):
        started = time.monotonic()
        rid, snap = record_offline(cat, scene_for(case), case['seconds'], commands_for(case),
                                   case['title'], case['hypothesis'])
        cat.write_notes(rid, case['hypothesis'] + '\n\n')
        ids.append(rid)
        print(case['id'], rid, 'clips=' + str(snap['clip_blocks']), flush=True)
        time.sleep(max(0.0, 1.05 - (time.monotonic() - started)))
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(CATALOG_ROOT))
    parser.add_argument('--scenes-only', action='store_true')
    args = parser.parse_args(argv)
    write_scenes()
    if not args.scenes_only:
        build(args.root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
