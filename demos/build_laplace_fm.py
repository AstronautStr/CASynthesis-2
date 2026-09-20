"""Build the Laplace FM catalog (REQ memory/req-laplace-fm-2026-09-20.md section 5):
the new FM first against the existing Laplacian on sines, then against the Wave bank
with a saw that the user already found useful -- on ONE Jam evolution.

    python demos/build_laplace_fm.py [--root DIR] [--scenes-only]

Scenes demos/lfm_*.json are (re)written; the catalog is built only into a directory that
holds no records yet (records and Notes the user added are never overwritten).
`run_laplace_fm.bat` opens the bench on the catalog screen of the default root.

One field in all three: Conway B3/S23 on a 32 x 32 torus, the Jam of the REQ (the cells
of the preceding preflight), 4 generations / s, 12 s, f0 = 110 Hz, n 3 / spread 1 /
alpha 0 / shape 0 / harm 0 / full 1 / dyn 0 on every side.  Three sound variants:

  R          the existing `laplacian` engine -- the sum of sines on the Laplacian modes
  W_saw      `laplace_carriers`, Wave bank, Saw -- a saw per mode (already listened to)
  FM         `laplace_fm`, FM depth 1 -- the modes of a figure modulate its one carrier

Order (REQ section 5).  Record 2 is the ALREADY LISTENED comparison R / W-saw: the
untouched record of the carriers catalog, copied here with its own provenance, pin, title
and the user's Notes (never re-rendered, never re-levelled).  Its id carries the time it
was made, which is earlier than the two new ones, so the catalog screen -- newest first --
shows it below them; the titles are numbered 1 / 2 / 3 in the REQ's listening order.

Levels: SIDE_GAIN is one constant factor per VARIANT (scene audio.side_gain, applied
before the PCM clip, part of the record / snapshot, so Replay / Continue use it).  R and
W-saw keep the factors of the carriers catalog, so their tracks stay byte-identical to the
records the user already heard; only FM gets a new one, measured from the integral RMS of
the ready 12 s (demos/laplace_fm_report.py, section "levels") so that it lands within
0.5 dB of both -- not equal subjective loudness.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from casynth_lab import scene_from_doc                                   # noqa: E402
from casynth_lab.catalog import Catalog                                  # noqa: E402
from casynth_lab import laplace_carriers as lc                           # noqa: E402
from casynth_lab import laplace_fm as lfm                                # noqa: E402
from demos.build_n1_demos import record_offline                          # noqa: E402
from demos.build_laplace_carriers import (CATALOG_ROOT as CARRIERS_ROOT,  # noqa: E402
                                          PREFLIGHT, BASELINE_ID, ROWS, COLS, RATE_HZ,
                                          SECONDS, F0_HZ, SPECTRUM, WIDTH_OCT, DEPTH_DB,
                                          jam_cells, require_pinnable)

CATALOG_ROOT = ROOT / 'lab_catalog' / 'laplace_fm_2026_09_20'
FM_PREFLIGHT = ROOT / 'memory' / 'research' / 'laplace-fm-preflight-2026-09-20.json'
FM_DEPTH = 1.0

# the already listened R / W-saw record of the carriers catalog (REQ section 5, pair 2)
REFERENCE_RECORD = '20260920-164914-f2dedf'
REFERENCE_SCENE = 'lc_saw_baseline_bank'

# one constant level factor per VARIANT.  R and W_saw are the carriers catalog's own
# factors (their tracks must stay byte-identical to the records already listened to);
# FM measured 2026-09-20 on the ready 12 s at gain 1: R -29.93 dBFS (peak 0.135),
# W-saw -29.94 (0.156 at its 0.78), FM -29.24 (0.095)
# -> factor = 10 ** ((-29.93 - (-29.24)) / 20), 2 decimals.  FM then sits 0.03 dB under R
# and 0.02 dB under W-saw, and peaks at 0.087 of full scale: no clip on any track.
SIDE_GAIN = {'R': 1.0, 'W_saw': 0.78, 'FM': 0.92}

VARIANT_TEXT = {
    'R': 'существующий Laplace на синусоидах (n 3, spread 1, alpha 0, shape 0, harm 0, full 1, dyn 0)',
    'W_saw': 'Laplace waves, Wave bank, пила на каждой моде',
    'FM': 'Laplace FM, FM depth 1: моды фигуры модулируют её синусоидальную несущую',
}


def variant(name):
    """(engine_id, engine_params) of one of the three sound variants."""
    if name == 'R':
        return BASELINE_ID, dict(SPECTRUM)
    if name == 'FM':
        return lfm.ENGINE_ID, dict(fm_depth=FM_DEPTH, **SPECTRUM)
    return lc.ENGINE_ID, dict(method=lc.METHOD_BANK, waveform=lc.WF_SAW,
                              filter_width_oct=WIDTH_OCT, filter_depth_db=DEPTH_DB, **SPECTRUM)


FIELD_NOTE = ('Поле Jam (13 клеток), 32 x 32, тор, B3/S23; 12 с, 4 поколения/с, f0 = 110 Hz. '
              'Период 3: фигуры 7+3+3 -> 11+3 -> 16 клеток. Семь настроек спектра одинаковы '
              'на обеих сторонах (n 3, spread 1, alpha 0, shape 0, harm 0, full 1, dyn 0); '
              'частоты и веса мод у всех вариантов одни и те же.')

FM_NOTE = ('FM: у каждой фигуры одна синусоидальная несущая на частоте ноты, моды этой фигуры '
           'задают частоты модуляторов, их веса — индексы (beta = FM depth * a). Ручка FM depth '
           '0...4; при 0 остаётся синус несущей, а не исходный Laplacian.')

CASES = [
    dict(id='lfm_baseline', a='R', b='FM',
         title='1 - Моды модулируют несущую: Laplace / Laplace FM (Jam, 4 поколения/с)',
         note=FIELD_NOTE + ' ' + FM_NOTE,
         hypothesis='Гипотеза - А — исходный Laplacian на синусоидах. В Б моды каждой фигуры '
         'модулируют одну синусоидальную несущую: частоты мод задают частоты модуляторов, веса — '
         'их глубины. Ожидаем повторяющуюся смену FM-окраски при трёх состояниях Jam. Проверяем, '
         'полезен ли этот характер относительно А и различима ли жизнь фигуры. FM depth=0 '
         'оставляет синус несущей, а не возвращает к А.'),
    dict(id='lfm_saw_bank', a='W_saw', b='FM',
         title='3 - Два кандидата: Wave bank / Laplace FM (Jam, 4 поколения/с)',
         note=FIELD_NOTE + ' А — та же дорожка пил, что в опыте 2. ' + FM_NOTE,
         hypothesis='Гипотеза - А — уже понравившийся Wave bank с пилой, Б — тот же FM из '
         'сравнения с исходным Laplacian. Ожидаем разный характер смены окраски: ряды пил на '
         'частотах мод против их совместной модуляции несущей. Проверяем, даёт ли FM отдельную '
         'полезную роль или предпочтение остаётся за массивом пил; взаимное различие само по себе '
         'не означает пользу FM. При FM depth=0 в Б остаётся синус несущей.'),
]


def listen_text(case):
    return ('1 / 2: А — ' + VARIANT_TEXT[case['a']] + ' / Б — ' + VARIANT_TEXT[case['b']] +
            '. Notes: hypothesis and your listening result. FM depth is the first knob of the '
            'Laplace FM panel (0 = the carrier sine alone, not the Laplacian sum); the seven '
            'spectrum settings are shared ("spectrum A to B").')


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


def copy_reference(root):
    """Pair 2: the untouched R / W-saw record of the carriers catalog, copied with its id,
    provenance, pin, title and the user's Notes.  Nothing inside is rewritten and the
    source record is left alone (REQ section 5)."""
    src = Path(CARRIERS_ROOT) / REFERENCE_RECORD
    if not (src / 'record.json').exists():
        raise SystemExit(f"error: the reference record is missing: {src}")
    dst = Path(root) / REFERENCE_RECORD
    if dst.exists():
        raise SystemExit(f"error: the reference record is already in the catalog: {dst}")
    shutil.copytree(src, dst)
    doc = json.loads((dst / 'record.json').read_text(encoding='utf-8'))
    if doc.get('id') != REFERENCE_RECORD or doc['scene']['id'] != REFERENCE_SCENE:
        raise SystemExit('error: the copied reference record is not the expected one')
    return REFERENCE_RECORD


def build(root=CATALOG_ROOT):
    root = Path(root)
    require_pinnable()
    if root.exists() and any(not p.name.startswith('.') for p in root.iterdir()):
        raise SystemExit('error: target catalog already holds records: ' + str(root) +
                         '\n(rebuild into a new directory with --root; records and Notes the '
                         'user added are never overwritten)')
    (root / '.tmp').mkdir(parents=True, exist_ok=True)
    cat = Catalog(str(root))
    rid = copy_reference(root)
    print(f"  {rid}  {REFERENCE_SCENE}  (copied unchanged from {CARRIERS_ROOT.name})", flush=True)
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
    ap = argparse.ArgumentParser(description="build the Laplace FM scenes and catalog")
    ap.add_argument('--root', default=str(CATALOG_ROOT))
    ap.add_argument('--scenes-only', action='store_true')
    a = ap.parse_args(argv)
    paths = write_scenes()
    print(f"scenes: {len(paths)} files in {ROOT / 'demos'}")
    if a.scenes_only:
        return 0
    res = build(a.root)
    print(f"catalog: {len(res) + 1} records in {a.root}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
