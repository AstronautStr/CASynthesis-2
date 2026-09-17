"""Build the Objects / Laplace comparison catalog: the existing Laplace synth (A) against
the Objects engine in its Laplace spectrum mode (B) on the same field with the same
seven spectrum settings (REQ memory/req-objects-laplace-comparison-2026-09-17.md).

    python demos/build_objects_laplace.py [--root DIR] [--scenes-only]

Scenes demos/ol_*.json are (re)written; the catalog is built only into a directory
that holds no records yet (records and Notes the user added are never overwritten).
`run_objects_laplace.bat` opens the bench on the catalog screen of the default root.
All three experiments: one field, Conway B3/S23 on a 32 x 32 torus, 6 generations / s,
12 s, f0 = 110 Hz, both sides always on the same field; the cells are the Researcher's
preflight fixtures (memory/research/object-resonators-laplace-preflight-2026-09-17.json,
`cases.<name>.cells`); the hypotheses of the REQ are written into Notes verbatim.

  A = `laplacian` with the seven settings of the REQ table (n 12, spread 0, alpha 1,
      shape 0, harm 0, full 1, dyn 0)
  B = `ca_object_resonators`, Spectrum = Laplace with the SAME seven settings,
      detector Disk, Decay 0.8, Radius x as below (Freq scale inactive in this mode)

  L1  One figure            glider inside the field (no seam in 12 s), Radius x 1
  L2  Assembly and decay    Kok's galaxy at (11, 11), Radius x 1
  L3  Neighbour and radius  the still 17-cell figure + a blinker three columns to the
                            right of N4.2, Radius x 1.5 (its four changes are outside
                            the x1 circle and inside the x1.5 one)

Levels: SIDE_GAIN is ONE constant factor per side per experiment (scene audio.side_gain,
applied before the PCM clip, part of the record / snapshot, so Replay / Continue use it)
chosen from the measured integral RMS of the ready 12 s so that A and B differ by
<= 1 dB (demos/objects_laplace_report.py measures both); not equal loudness.
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
from casynth_lab.object_resonators import (ENGINE_ID, DET_DISK, SPEC_LAPLACE,   # noqa: E402
                                           laplace_settings)
from demos.build_n1_demos import record_offline                          # noqa: E402

CATALOG_ROOT = ROOT / 'lab_catalog' / 'objects_laplace_2026_09_17'
PREFLIGHT = ROOT / 'memory' / 'research' / 'object-resonators-laplace-preflight-2026-09-17.json'
LAPLACE_ID = 'laplacian'
ROWS = COLS = 32
RATE_HZ = 6.0
SECONDS = 12.0
F0_HZ = 110.0
DECAY_S = 0.8
SCALE_HZ = 220.0                      # inactive in the Laplace mode, kept at the default
SPECTRUM = dict(n=12, spread=0.0, alpha=1.0, shape=0.0, harm=0.0, fullshape=1, dyn=0.0)
assert SPECTRUM == laplace_settings({}), "the REQ table equals the Laplace registry defaults"
# one constant level factor per side per experiment (measured: see the module docstring)
# measured 2026-09-17 at side gain 1 (A - B integral RMS of the ready 12 s, LAPLACE_GAIN 0.7):
# glider +0.69 dB, galaxy -2.61 dB, neighbor +1.91 dB -> B = 10 ** (diff / 20), 2 decimals
SIDE_GAIN = {'ol_glider': dict(A=1.0, B=1.08),
             'ol_galaxy': dict(A=1.0, B=0.74),
             'ol_neighbor': dict(A=1.0, B=1.25)}


def laplace_side():
    return (LAPLACE_ID, dict(SPECTRUM))


def objects_side(radius_mul):
    return (ENGINE_ID, dict(detector=DET_DISK, radius_mul=float(radius_mul), spectrum=SPEC_LAPLACE,
                            frequency_scale=SCALE_HZ, decay_s=DECAY_S, attack_ms=0.0, **SPECTRUM))


def preflight_cells(case):
    with open(PREFLIGHT, encoding='utf-8') as f:
        doc = json.load(f)
    return [[int(r), int(c)] for r, c in doc['cases'][case]['cells']]


def glider_cells():
    return preflight_cells('glider')


def galaxy_cells():
    return preflight_cells('galaxy')


def neighbor_cells():
    return preflight_cells('neighbor')


CASES = [
    dict(id='ol_glider', title='L1 - Одна фигура: Laplace / Objects (glider, 6 поколений/с)',
         cells=glider_cells, radius_mul=1.0,
         side_a='А — существующий Laplace (n 12, spread 0, alpha 1, shape 0, harm 0, full 1, dyn 0)',
         side_b='Б — Objects / Laplace, те же семь настроек, Disk, Radius x 1, Decay 0.8',
         note='Glider внутри поля (за 12 с не пересекает шов); 12 с, 6 поколений/с, f0 = 110 Hz. '
         'Семь общих настроек спектра одинаковы на обеих сторонах. Сравниваем два способа озвучивания '
         'одной и той же последовательности спектров. Обе стороны получают одно поле.',
         hypothesis='Гипотеза - При одинаковой настройке спектра А даёт непрерывно поддерживаемый голос '
         'глайдера, а Б — последовательность затухающих откликов на изменения его клеток. Ожидаем в Б '
         'более отчётливое членение по событиям при узнаваемом повторении окраски. Проверяем, делает ли '
         'это движение фигуры слышнее и интереснее в игре.'),
    dict(id='ol_galaxy', title='L2 - Сборка и распад: Laplace / Objects (Kok\'s galaxy, 6 поколений/с)',
         cells=galaxy_cells, radius_mul=1.0,
         side_a='А — существующий Laplace (n 12, spread 0, alpha 1, shape 0, harm 0, full 1, dyn 0)',
         side_b='Б — Objects / Laplace, те же семь настроек, Disk, Radius x 1, Decay 0.8',
         note='Kok\'s galaxy из библиотеки, сдвиг (11, 11); 12 с, 6 поколений/с, f0 = 110 Hz. За период '
         'возникают компоненты от 1 до 64 клеток, максимум 12 компонент и 8 звучащих; шва нет. Все семь '
         'настроек меняют спектральные данные хотя бы в одной фазе (spread / full — в фазах 2, 4, 5; dyn при '
         'shape > 0). Опыт для работы ручками и жизни голосов.',
         hypothesis='Гипотеза - На одной и той же распадающейся и собирающейся фигуре А меняет состав '
         'поддерживаемых голосов, а Б добавляет к новым событиям затухающие отклики прежних фигур. Ожидаем '
         'различие в ритме и наложении звуков между фазами. Проверяем, помогает ли такой способ слышать '
         'развитие поля; настройки самого спектра доступны одинаковые на обеих сторонах.'),
    dict(id='ol_neighbor', title='L3 - Сосед и радиус: Laplace / Objects (17 клеток + blinker, 6 поколений/с)',
         cells=neighbor_cells, radius_mul=1.5,
         side_a='А — существующий Laplace (n 12, spread 0, alpha 1, shape 0, harm 0, full 1, dyn 0)',
         side_b='Б — Objects / Laplace, те же семь настроек, Disk, Radius x 1.5, Decay 0.8',
         note='Неподвижная 17-клеточная фигура из N4.2 и blinker, сдвинутый на три столбца вправо '
         '((9,17)-(9,19)); 12 с, 6 поколений/с, f0 = 110 Hz. Изменения блинкера вне круга x1 приёмника, все '
         'четыре — внутри x1.5. Обе фигуры существуют отдельно; сам блинкер принимает свои четыре изменения '
         'и при x1, и при x1.5.',
         hypothesis='Гипотеза - В А обе фигуры звучат через обычный Laplace. В Б изменения блинкера '
         'дополнительно возбуждают неподвижного соседа внутри увеличенного круга. Ожидаем составной '
         'ритмический отклик; при уменьшении Radius x с 1.5 до 1 отклик соседа должен затухнуть, а блинкер '
         'продолжить звучать. Проверяем, даёт ли радиус понятный способ управлять взаимодействием фигур.'),
]


def listen_text(case):
    return ('1 / 2: ' + case['side_a'] + ' / ' + case['side_b'] + '. Notes: hypothesis and your '
            'listening result. The seven spectrum settings are shared: "spectrum A to B" copies them.')


def scene_for(case):
    (ea, pa), (eb, pb) = laplace_side(), objects_side(case['radius_mul'])
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=ROWS, cols=COLS),
                cells=[[int(r), int(c)] for r, c in case['cells']()], rule='B3/S23',
                boundary='torus', rate_hz=RATE_HZ,
                audio=dict(f0_hz=F0_HZ, level=1.0, side_gain=dict(SIDE_GAIN[case['id']])),
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


def require_pinnable():
    """Records of a delivered catalog must be PINNED to a commit (rule 2026-09-17): the
    sound code that renders them has to be committed BEFORE the build, otherwise the
    records stay Local and cannot be run in their own version later."""
    from casynth_lab import provenance as prov
    doc = prov.current()
    if doc.get('match') != 'clean':
        raise SystemExit('error: the sound code is not a clean commit (' + str(doc.get('reason') or '?') +
                         ')\ncommit it first -- catalog records must be pinned to a commit '
                         '(developer rule 2026-09-17)')
    return doc['commit']


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
    ap = argparse.ArgumentParser(description="build the Objects / Laplace comparison scenes and catalog")
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
