"""Build the Laplace carriers catalog (REQ memory/req-laplace-carriers-2026-09-20.md):
each new law first against the existing Laplacian on sines, then the two laws against
each other -- six records on ONE Jam evolution.

    python demos/build_laplace_carriers.py [--root DIR] [--scenes-only]

Scenes demos/lc_*.json are (re)written; the catalog is built only into a directory that
holds no records yet (records and Notes the user added are never overwritten).
`run_laplace_carriers.bat` opens the bench on the catalog screen of the default root.

One field in all six: Conway B3/S23 on a 32 x 32 torus, the Jam of the REQ (the cells of
demos/oes_m1.json, duplicated in the preflight), 4 generations / s, 12 s, f0 = 110 Hz,
n 3 / spread 1 / alpha 0 / shape 0 / harm 0 / full 1 / dyn 0 on every side.  Five sound
variants, each with ONE constant level factor of its own, so that a variant is byte-exact
in every record it appears in (REQ section 5, "Единые варианты во всех парах"):

  R          the existing `laplacian` engine -- the sum of sines on the Laplacian modes
  F-saw      `laplace_carriers`, Filter, Saw     (width 0.35 oct, depth 24 dB)
  F-square   `laplace_carriers`, Filter, Square  (same mask settings)
  W-saw      `laplace_carriers`, Wave bank, Saw     (the mask settings are stored, inactive)
  W-square   `laplace_carriers`, Wave bank, Square  (same)

Levels: SIDE_GAIN below is that per-variant factor (scene audio.side_gain, applied before
the PCM clip, part of the record / snapshot, so Replay / Continue use it), measured with
demos/laplace_carriers_report.py from the integral RMS of the ready 12 s so that any two
sides differ by <= 1 dB and no track clips -- not equal subjective loudness.
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
from casynth_lab import laplace_carriers as lc                           # noqa: E402
from demos.build_n1_demos import record_offline                          # noqa: E402

CATALOG_ROOT = ROOT / 'lab_catalog' / 'laplace_carriers_2026_09_20'
PREFLIGHT = ROOT / 'memory' / 'research' / 'laplace-carriers-preflight-2026-09-20.json'
BASELINE_ID = 'laplacian'
ROWS = COLS = 32
RATE_HZ = 4.0
SECONDS = 12.0
F0_HZ = 110.0
SPECTRUM = dict(n=3, spread=1.0, alpha=0.0, shape=0.0, harm=0.0, fullshape=1, dyn=0.0)
WIDTH_OCT = 0.35
DEPTH_DB = 24.0

# one constant level factor per VARIANT (not per scene): a variant sounds byte-identical
# in every record it appears in.  Measured 2026-09-20 on the ready 12 s at gain 1
# (integral RMS / peak, demos/laplace_carriers_report.py section "levels"):
#   R -29.93 dBFS (peak 0.135), F-saw -28.90 (0.085), F-square -29.65 (0.064),
#   W-saw -27.78 (0.200), W-square -29.04 (0.131)
# -> factor = 10 ** ((target - x) / 20) with the target -29.93 dBFS of R, 2 decimals.
# The loudest track then peaks at 0.157 of full scale: headroom for every side, no clip.
SIDE_GAIN = {'R': 1.0, 'F_saw': 0.89, 'F_square': 0.97, 'W_saw': 0.78, 'W_square': 0.90}

VARIANT_TEXT = {
    'R': 'существующий Laplace на синусоидах (n 3, spread 1, alpha 0, shape 0, harm 0, full 1, dyn 0)',
    'F_saw': 'Laplace waves, Filter, пила (width 0.35 окт, depth 24 dB)',
    'F_square': 'Laplace waves, Filter, меандр (width 0.35 окт, depth 24 dB)',
    'W_saw': 'Laplace waves, Wave bank, пила на каждой моде',
    'W_square': 'Laplace waves, Wave bank, меандр на каждой моде',
}


def variant(name):
    """(engine_id, engine_params) of one of the five sound variants."""
    if name == 'R':
        return BASELINE_ID, dict(SPECTRUM)
    method = lc.METHOD_FILTER if name.startswith('F_') else lc.METHOD_BANK
    wave = lc.WF_SAW if name.endswith('_saw') else lc.WF_SQUARE
    return lc.ENGINE_ID, dict(method=method, waveform=wave, filter_width_oct=WIDTH_OCT,
                              filter_depth_db=DEPTH_DB, **SPECTRUM)


def jam_cells():
    with open(PREFLIGHT, encoding='utf-8') as f:
        doc = json.load(f)
    return [[int(r), int(c)] for r, c in doc['phases'][0]['cells']]


FIELD_NOTE = ('Поле Jam (13 клеток), 32 x 32, тор, B3/S23; 12 с, 4 поколения/с, f0 = 110 Hz. '
              'Период 3: фигуры 7+3+3 -> 11+3 -> 16 клеток. Семь настроек спектра одинаковы '
              'на обеих сторонах (n 3, spread 1, alpha 0, shape 0, harm 0, full 1, dyn 0); '
              'частоты и веса мод у всех вариантов одни и те же.')

CASES = [
    dict(id='lc_saw_baseline_filter', a='R', b='F_saw',
         title='1 - Пила через лапласиановский фильтр: Laplace / Filter (Jam, 4 поколения/с)',
         note=FIELD_NOTE + ' Б: одна пила на частоте ноты, её гармоники окрашены профилем мод.',
         hypothesis='Гипотеза - А — исходный Laplacian на синусоидах. В Б тот же Jam окрашивает '
         'пилообразную волну через лапласиановский фильтр. Ожидаем изменение характера и другой '
         'способ слышать три состояния фигуры. Проверяем, появляется ли полезный звук относительно '
         'А и остаются ли изменения фигуры различимыми.'),
    dict(id='lc_saw_baseline_bank', a='R', b='W_saw',
         title='2 - Пила на каждой моде: Laplace / Wave bank (Jam, 4 поколения/с)',
         note=FIELD_NOTE + ' Б: каждая синусоида моды заменена пилой на той же частоте и с тем же весом.',
         hypothesis='Гипотеза - А — исходный Laplacian на синусоидах. В Б каждая его синусоида '
         'заменена пилой при тех же основных частотах и весах. Ожидаем новый характер при сохранении '
         'рисунка исходных тонов. Проверяем, расширяет ли это звучание или дополнительные гармоники '
         'только перегружают и скрывают жизнь фигуры.'),
    dict(id='lc_square_baseline_filter', a='R', b='F_square',
         title='3 - Меандр через лапласиановский фильтр: Laplace / Filter (Jam, 4 поколения/с)',
         note=FIELD_NOTE + ' Б: одна меандровая волна на частоте ноты; чётных гармоник у неё нет.',
         hypothesis='Гипотеза - А — исходный Laplacian на синусоидах. В Б фигура управляет фильтром '
         'меандровой волны, у которой отсутствуют чётные гармоники. Часть лапласиановского рисунка '
         'может поэтому не попасть в звук. Проверяем, даёт ли Б полезный новый характер относительно '
         'А и слышны ли по-прежнему три состояния Jam.'),
    dict(id='lc_square_baseline_bank', a='R', b='W_square',
         title='4 - Меандр на каждой моде: Laplace / Wave bank (Jam, 4 поколения/с)',
         note=FIELD_NOTE + ' Б: каждая исходная частота звучит собственным меандром.',
         hypothesis='Гипотеза - А — исходный Laplacian на синусоидах. В Б каждая исходная частота '
         'звучит собственным меандром. Ожидаем изменение окраски с сохранением движения основных '
         'тонов. Проверяем, получается ли интересный характер для игры и не теряется ли реакция на '
         'фигуру в смеси этих голосов.'),
    dict(id='lc_saw_filter_bank', a='F_saw', b='W_saw',
         title='5 - Два способа с пилой: Filter / Wave bank (Jam, 4 поколения/с)',
         note=FIELD_NOTE + ' Обе стороны — Laplace waves с пилой: А фильтрует одну несущую, '
         'Б суммирует пилы на частотах мод. Те же дорожки, что в опытах 1 и 2.',
         hypothesis='Гипотеза - Теперь сравниваем два варианта пилы, уже сопоставленных с исходными '
         'синусоидами: А — лапласиановский фильтр, Б — пила на каждой частоте моды. Ожидаем разное '
         'поведение окраски и сочетания тонов на тех же состояниях Jam. Проверяем, дают ли способы '
         'разные полезные возможности или один из них предпочтительнее; различие само по себе не '
         'означает улучшения относительно baseline.'),
    dict(id='lc_square_filter_bank', a='F_square', b='W_square',
         title='6 - Два способа с меандром: Filter / Wave bank (Jam, 4 поколения/с)',
         note=FIELD_NOTE + ' Обе стороны — Laplace waves с меандром: А фильтрует одну несущую, '
         'Б суммирует меандры на частотах мод. Те же дорожки, что в опытах 3 и 4.',
         hypothesis='Гипотеза - Теперь сравниваем два варианта меандра, уже сопоставленных с '
         'исходными синусоидами: А — лапласиановский фильтр, Б — меандр на каждой частоте моды. '
         'Проверяем, какой способ лучше сохраняет интересный рисунок фигуры и нужны ли оба. Если оба '
         'уступают исходному звуку, их взаимное различие не считаем успехом.'),
]


def listen_text(case):
    return ('1 / 2: А — ' + VARIANT_TEXT[case['a']] + ' / Б — ' + VARIANT_TEXT[case['b']] +
            '. Notes: hypothesis and your listening result. Method / Wave are the buttons of the '
            'Laplace waves panel; the seven spectrum settings are shared ("spectrum A to B").')


def scene_for(case):
    (ea, pa), (eb, pb) = variant(case['a']), variant(case['b'])
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=ROWS, cols=COLS),
                cells=jam_cells(), rule='B3/S23', boundary='torus', rate_hz=RATE_HZ,
                audio=dict(f0_hz=F0_HZ, level=1.0,
                           side_gain=dict(A=SIDE_GAIN[case['a']], B=SIDE_GAIN[case['b']])),
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
    sound code that renders them has to be committed BEFORE the build."""
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
        note = ('А — ' + VARIANT_TEXT[case['a']] + ' / Б — ' + VARIANT_TEXT[case['b']] + '. '
                + case['note'])
        rid, snap = record_offline(cat, scene_for(case), SECONDS, [], case['title'], note)
        cat.write_notes(rid, case['hypothesis'] + '\n\n')
        results.append((rid, case['id'], dict(snap['clip_blocks']), snap['gen']))
        print(f"  {rid}  {case['id']}  clip={snap['clip_blocks']} gen={snap['gen']}", flush=True)
        time.sleep(max(0.0, 1.05 - (time.monotonic() - started)))   # ids carry a 1 s stamp
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the Laplace carriers scenes and catalog")
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
