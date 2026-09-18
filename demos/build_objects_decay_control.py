"""Build the Objects decay CONTROL catalog D4: two A/B records on the exact field and
scene of the delivered D3 record, three conditions -- REQ
memory/req-objects-decay-control-2026-09-18.md, preflight
memory/research/objects-decay-control-preflight-2026-09-18.json.

    python demos/build_objects_decay_control.py [--root DIR] [--scenes-only] [--calibrate]

Scenes demos/od_d4_*.json are (re)written; the catalog is built only into a directory
that holds no records yet (records and Notes the user added are never overwritten).
`run_objects_decay_control.bat` opens the bench on the CATALOG screen of the default root.

The scenes are DERIVED from the scene embedded in the D3 record
(lab_catalog/objects_decay_2026_09_18/20260918-204155-4ff923, `demo_scene`), not from
the current defaults: the field is the preflight's `cells` (the same 29 cells), and a
condition changes only the Decay law, Decay and the constant side gain of the side it
starts from.  Checked against the preflight at every call.

  condition        from D3 side   Decay law   Decay / Stable decay   side gain
  matched_fixed    A              Fixed       0.671529 s             1.220467
  modal            B              Modal age   1.39 s                 1.21      (= D3 B)
  long_fixed       A              Fixed       1.39 s                 1.0       (= D3 A)

  D4.1  od_d4_control   A matched_fixed / B modal
  D4.2  od_d4_anchor    A matched_fixed / B long_fixed

0.671529 s = ln(1000) / the mean APPLIED gamma (after the 20 ms smoothing) of every
driven mode of the active banks of the D3 Modal side over the samples [132300, 793800),
each mode-sample weighted equally (preflight; reproduced independently by
demos/objects_decay_control_report.py) -- never tuned by ear or by level.  The side
gains are ONE constant factor per condition from the RMS of all samples of both
channels over 3..18 s (the old conditions keep 1.0 / 1.21; the matched Fixed is levelled
to the long Fixed); --calibrate measures them at unit gains.  18 s at 6 gen/s = 794112
samples, the length of D3; start from the original field with a new history, no
scripted or manual intervention.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from casynth_config import SR                                             # noqa: E402
from casynth_lab import scene_from_doc, BLOCK                            # noqa: E402
from casynth_lab.catalog import Catalog                                  # noqa: E402
from casynth_lab.object_resonators import ENGINE_ID, LAW_FIXED, LAW_MODAL   # noqa: E402
from demos.build_n1_demos import record_offline                          # noqa: E402
from demos.build_objects_laplace import require_pinnable                 # noqa: E402
from demos.build_objects_event_source import rms_db                      # noqa: E402
from demos.build_objects_decay import SETTINGS_TEXT, render_sides        # noqa: E402

CATALOG_ROOT = ROOT / 'lab_catalog' / 'objects_decay_control_2026_09_18'
PREFLIGHT = ROOT / 'memory' / 'research' / 'objects-decay-control-preflight-2026-09-18.json'
SOURCE_RECORD = ROOT / 'lab_catalog' / 'objects_decay_2026_09_18' / '20260918-204155-4ff923'
SECONDS = 18.0
N_SAMPLES = 794112                               # = ceil(18 s / block) blocks, the length of D3
WINDOW = (132300, 793800)                        # [3, 18) s: the level and the averaging window

# the REQ's table (checked against the preflight's `conditions`)
CONDITIONS = {
    'matched_fixed': dict(base='A', law=LAW_FIXED, decay=0.671529, side_gain=1.220467),
    'modal': dict(base='B', law=LAW_MODAL, decay=1.39, side_gain=1.21),
    'long_fixed': dict(base='A', law=LAW_FIXED, decay=1.39, side_gain=1.0),
}

SIDE_TEXT = {
    'matched_fixed': 'подобранный Fixed: Objects / Laplace, Decay law Fixed, Decay 0.671529 s, уровень стороны x1.220467',
    'modal': 'прежний Modal: Objects / Laplace, Decay law Modal age, Stable decay 1.39 s, уровень стороны x1.21',
    'long_fixed': 'прежний длинный Fixed: Objects / Laplace, Decay law Fixed, Decay 1.39 s, уровень стороны x1.0',
}

COMMON_NOTE = ('Поле и сцена исходного D3: Jam (период 3) + Octagon II (период 5), 29 клеток, старт с исходного поля '
               'с новой историей; 18 с, 6 поколений/с, f0 = 110 Hz; обе стороны — ' + SETTINGS_TEXT + '. '
               'Decay 0.671529 s подобранного Fixed — ln 1000 / средняя ПРИМЕНЁННАЯ (после сглаживания 20 мс) скорость '
               'потерь gamma = 10.2866 1/s всех возбуждаемых мод активных банков прежнего Modal за 3…18 с, каждая пара '
               '«мода–отсчёт» с равным весом (5 293 928 пар); не подбор на слух и не по громкости. Частоты, удары, веса, '
               'Attack, клетки и панорама сторон одинаковы; различаются только потери и постоянный уровень стороны '
               '(одна калибровка на условие по RMS обоих каналов за 3…18 с). Среднее и уровни рассчитаны для этой '
               'исходной сцены и при ручной игре не пересчитываются.')

PAIRS = [
    dict(id='od_d4_control', label='D4.1', A='matched_fixed', B='modal',
         title='D4.1 - Fixed по средней скорости потерь / Modal (Jam p3 + Octagon II p5)',
         extra='Б побитно равна Б записи D3.',
         hypothesis='Гипотеза - В А у всех частот один постоянный хвост, подобранный по средней скорости затухания Б. '
         'В Б — тот же Modal, который мы уже слушали: разные составляющие гаснут по-разному и меняют затухание вместе с '
         'историей клеток. Проверяем, остаётся ли у Б интересный характер отдельных звуков и их наложений по сравнению с '
         'подобранным Fixed. После первых трёх секунд: что именно отличается, в каком месте и какой вариант хочется '
         'оставить? Если слышна только общая сухость, длина или потеря составляющей, так и напиши. А в следующей записи '
         'будет точно такой же.'),
    dict(id='od_d4_anchor', label='D4.2', A='matched_fixed', B='long_fixed',
         title='D4.2 - Тот же Fixed / прежний длинный Fixed (Jam p3 + Octagon II p5)',
         extra='А побитно равна А записи D4.1, Б — А записи D3. Новый закон затухания в этой паре не используется.',
         hypothesis='Гипотеза - А здесь точно такой же, как в D4.1. Б — прежний длинный Fixed из D3, в котором тебе '
         'нравилась дополнительная составляющая. Проверяем, приводит ли уже обычное укорачивание всех хвостов в А к её '
         'исчезновению или потере интереса. Что возвращает длинный Fixed: только общую протяжённость или конкретные '
         'голоса и наложения? Какой вариант интереснее? Эта пара нужна как опора для оценки Modal в D4.1; она не '
         'использует новый закон затухания.'),
]


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def source_meta():
    with open(SOURCE_RECORD / 'record.json', encoding='utf-8') as f:
        return json.load(f)


def source_scene():
    """The scene document embedded in the D3 record (as the builder of D3 wrote it)."""
    meta = source_meta()
    doc = copy.deepcopy(meta['demo_scene'])
    assert doc['id'] == 'od_d3' and meta['end_sample'] == N_SAMPLES and not doc.get('script'), 'not the D3 record'
    return doc


def condition_params(name, src=None):
    """Engine parameters of a condition: the D3 side it starts from with its Decay law
    and Decay replaced (checked against the preflight)."""
    cond = CONDITIONS[name]
    src = src or source_scene()
    p = dict(src['variants'][cond['base']]['engine_params'])
    p['decay_law'] = int(cond['law'])
    p['decay_s'] = float(cond['decay'])
    return p


def check_against_preflight():
    """The REQ's constants, the field and the shared parameters == the preflight; the
    conditions differ from their D3 side only where the REQ says."""
    pf = preflight()
    src = source_scene()
    assert sorted(map(tuple, pf['cells'])) == sorted(map(tuple, src['cells'])), 'field != D3'
    assert pf['clock'] == dict(sr=SR, block=BLOCK, record_samples=N_SAMPLES, window_samples=list(WINDOW),
                               window_seconds=[3, 18]), pf['clock']
    assert int(math.ceil(SECONDS * SR / BLOCK)) * BLOCK == N_SAMPLES
    for name, cond in CONDITIONS.items():
        want = pf['conditions'][name]
        assert (cond['law'], cond['decay'], cond['side_gain']) == (want['decay_law'], want['decay_s'], want['side_gain']), name
        p = condition_params(name, src)
        base = src['variants'][cond['base']]['engine_params']
        diff = {k for k in p if p[k] != base[k]}
        assert diff <= {'decay_s', 'decay_law'} and (diff == {'decay_s'}) == (name == 'matched_fixed'), (name, diff)
        for k, v in pf['shared_params'].items():
            assert p[k] == v, (name, k)
    old_gain = src['audio']['side_gain']
    assert (CONDITIONS['long_fixed']['side_gain'], CONDITIONS['modal']['side_gain']) == (old_gain['A'], old_gain['B'])
    assert [(q['id'], q['A'], q['B']) for q in pf['pairs']] == [(q['id'], q['A'], q['B']) for q in PAIRS]
    return pf


def listen_text(pair):
    return ('1 / 2: А — ' + SIDE_TEXT[pair['A']] + ' / Б — ' + SIDE_TEXT[pair['B']] + '. Notes: hypothesis and your '
            'listening result. The control Decay and the levels were computed for this scene from its start; they are '
            'not recomputed when you play.')


def record_note(pair):
    return ('А — ' + SIDE_TEXT[pair['A']] + ' / Б — ' + SIDE_TEXT[pair['B']] + '. ' + pair['extra'] + ' ' + COMMON_NOTE)


def scene_for(pair, side_gain=None):
    src = source_scene()
    doc = copy.deepcopy(src)
    doc['id'] = pair['id']
    doc['title'] = pair['title']
    doc['cells'] = [[int(r), int(c)] for r, c in preflight()['cells']]
    doc['variants'] = {s: dict(engine_id=ENGINE_ID, engine_params=condition_params(pair[s], src)) for s in ('A', 'B')}
    doc['audio'] = dict(src['audio'], side_gain=dict(side_gain if side_gain is not None else
                                                     {s: CONDITIONS[pair[s]]['side_gain'] for s in ('A', 'B')}))
    doc['initial_side'] = 'A'
    doc['listen'] = listen_text(pair)
    return doc


def write_scenes():
    check_against_preflight()
    paths = []
    for pair in PAIRS:
        doc = scene_for(pair)
        scene_from_doc(doc)                       # strict validation before writing
        path = ROOT / 'demos' / (pair['id'] + '.json')
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        paths.append(path)
    return paths


def window_rms_db(pcm):
    """RMS (dBFS) of all samples of both channels over [3, 18) s of an int16 (n, 2) take."""
    return rms_db(pcm[WINDOW[0]:WINDOW[1]])


def calibrate():
    """Window RMS of the three conditions at unit side gain and the constant factor that
    levels each to the long Fixed (printed; the constants above come from the REQ)."""
    unit = {}
    for pair in PAIRS:
        y, peak, clip, _r = render_sides(scene_for(pair, side_gain=dict(A=1.0, B=1.0)), SECONDS)
        for s in ('A', 'B'):
            unit.setdefault(pair[s], dict(rms_db=window_rms_db(y[s]), peak=peak[s], clip=clip[s]))
    ref = unit['long_fixed']['rms_db']
    out = {}
    for name, u in unit.items():
        g = 10.0 ** ((ref - u['rms_db']) / 20.0)
        out[name] = dict(u, gain_to_long_fixed=g, delivered_gain=CONDITIONS[name]['side_gain'],
                         delivered_rms_db=u['rms_db'] + 20.0 * math.log10(CONDITIONS[name]['side_gain']))
        print(f"  {name}: unit-gain RMS 3..18 s {u['rms_db']:.4f} dBFS, gain to the long Fixed {g:.6f} "
              f"(delivered {CONDITIONS[name]['side_gain']}), peak {u['peak']:.4f}, clip {u['clip']}", flush=True)
    return out


def build(root=CATALOG_ROOT):
    root = Path(root)
    require_pinnable()
    check_against_preflight()
    if root.exists() and any(not p.name.startswith('.') for p in root.iterdir()):
        raise SystemExit('error: target catalog already holds records: ' + str(root) +
                         '\n(rebuild into a new directory with --root; records and Notes the '
                         'user added are never overwritten)')
    (root / '.tmp').mkdir(parents=True, exist_ok=True)
    cat = Catalog(str(root))
    results = []
    for pair in reversed(PAIRS):                  # newest first on screen = REQ order
        started = time.monotonic()
        rid, snap = record_offline(cat, scene_for(pair), SECONDS, [], pair['title'], record_note(pair))
        cat.write_notes(rid, pair['hypothesis'] + '\n\n')
        results.append((rid, pair['id'], dict(snap['clip_blocks']), snap['gen']))
        print(f"  {rid}  {pair['id']}  clip={snap['clip_blocks']} gen={snap['gen']}", flush=True)
        time.sleep(max(0.0, 1.05 - (time.monotonic() - started)))   # ids carry a 1 s stamp
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the Objects decay control scenes and catalog (D4.1 / D4.2)")
    ap.add_argument('--root', default=str(CATALOG_ROOT))
    ap.add_argument('--scenes-only', action='store_true')
    ap.add_argument('--calibrate', action='store_true', help="measure the window RMS at unit side gains")
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
