"""Build the Objects "Decay law" catalog: three live A/B experiments D1 / D2 / D3 on the
Researcher's preflight fields, both sides Objects / Laplace with the preflight settings,
the sides differing only in the Decay law (and the constant level factor) -- REQ
memory/req-objects-decay-2026-09-18.md, preflight
memory/research/objects-decay-preflight-2026-09-18.json.

    python demos/build_objects_decay.py [--root DIR] [--scenes-only] [--calibrate]

Scenes demos/od_*.json are (re)written; the catalog is built only into a directory that
holds no records yet (records and Notes the user added are never overwritten).
`run_objects_decay.bat` opens the bench on the CATALOG screen of the default root.
Conway B3/S23 on a 32 x 32 torus, f0 = 110 Hz, both sides on the same field; the
cells are the exact `cases[*].cells` of the preflight; the hypotheses of the REQ are
written into Notes verbatim.

Both sides: `ca_object_resonators`, Own, Spectrum Laplace, full 1, part 3, spread 1,
harm 0.87, shape 0, alpha 0, dyn 0, Attack 4 ms, Events Births, Excitation Uniform,
Birth strength stored 1 (inactive), Radius x stored 1 (inactive with Own).

  D1  Octagon II p5, 2 gen/s, 20 s    A Fixed, Decay 0.947926 s (the control: the mean
                                      target loss rate of B over 5..20 s at the real
                                      block boundaries, from the preflight -- never
                                      tuned by ear) / B Common age, stable 1.39 s
  D2  Blinker p2, 2 gen/s, 16 s       A Common age / B Modal age, stable 1.39 s both;
                                      the CA pauses TWICE (pause on / off at the exact
                                      samples 132704 / 242880 / 375232 / 485408 of the
                                      scene clock -- one block after the last excitation)
                                      so the tail can be heard on its own; sound and
                                      history go on.  The pauses are the scene's `script`:
                                      the runner queues them at every start from the
                                      beginning (the record, Restart, a live start), they
                                      stand in the record's journal (marked `script`)
                                      and a Continue never repeats the past ones
  D3  Jam p3 + Octagon II p5, 6 gen/s, A Fixed, Decay 1.39 s / B Modal age, stable
      18 s                             1.39 s; no scripted intervention

Levels: SIDE_GAIN is ONE constant factor of side B per experiment (scene
audio.side_gain, applied before the PCM clip, part of the record / snapshot, so Replay
/ Continue use it) measured with --calibrate at unit gains from the integral RMS of the
REQ's window -- D1 5..20 s, D2 the active parts after 2 s EXCLUDING the pauses, D3
3..18 s -- so that A and B differ by <= 1 dB there.  The factor never follows the field,
the law or the knobs; no per-strike or per-pause levelling.
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
from casynth_lab.object_resonators import (ENGINE_ID, OPTIONAL_PARAMS, PARAMS, EV_BIRTHS, EXC_UNIFORM,   # noqa: E402
                                           LAW_FIXED, LAW_COMMON, LAW_MODAL, LAW_NAMES)
from demos.build_n1_demos import record_offline                          # noqa: E402
from demos.build_objects_laplace import require_pinnable                 # noqa: E402
from demos.build_objects_event_source import rms_db                      # noqa: E402

CATALOG_ROOT = ROOT / 'lab_catalog' / 'objects_decay_2026_09_18'
PREFLIGHT = ROOT / 'memory' / 'research' / 'objects-decay-preflight-2026-09-18.json'
ROWS = COLS = 32
F0_HZ = 110.0
STABLE_DECAY_S = 1.39
# one constant level factor of side B per experiment (measured with --calibrate at side
# gain 1, A - B integral RMS of the REQ window -> B = 10 ** (diff / 20), 2 decimals);
# the numbers: memory/log/2026-09-18-objects-decay.md
# measured 2026-09-18 at side gain 1 (A - B): D1 window 5..20 s +0.79 dB (whole +0.97), D2 active parts after
# 2 s without the pauses -1.77 (whole -1.75), D3 3..18 s +1.64 (whole +1.83); peaks A/B 0.046/0.045,
# 0.027/0.027, 0.163/0.097; no clip
SIDE_GAIN = {'od_d1': dict(A=1.0, B=1.09),
             'od_d2': dict(A=1.0, B=0.82),
             'od_d3': dict(A=1.0, B=1.21)}


def preflight():
    with open(PREFLIGHT, encoding='utf-8') as f:
        return json.load(f)


def preflight_case(case_id):
    for c in preflight()['cases']:
        if c['id'] == case_id:
            return c
    raise KeyError(case_id)


def case_cells(case_id):
    return [[int(r), int(c)] for r, c in preflight_case(case_id)['cells']]


def side_params(law, decay_s):
    """The REQ's common settings + the side's Decay law and Decay (stable T60)."""
    p = dict(detector=0, radius_mul=1.0, spectrum=1, events=int(EV_BIRTHS), excitation=int(EXC_UNIFORM),
             birth_strength=1.0, frequency_scale=220.0, decay_s=float(decay_s), decay_law=int(law),
             attack_ms=4.0, n=3, spread=1.0, alpha=0.0, shape=0.0, harm=0.87, fullshape=1, dyn=0.0)
    for k, v in OPTIONAL_PARAMS.items():
        p.setdefault(k, v)
    return {k: p[k] for k in (q[0] for q in PARAMS)}


SETTINGS_TEXT = ('Own, Laplace, full 1, part 3, spread 1, harm 0.87, shape 0, alpha 0, dyn 0, Attack 4 ms, '
                 'Events Births, Excitation Uniform')

# D2: the two CA pauses at exact output samples (block boundaries), from the REQ
D2_PAUSES = ((132704, True), (242880, False), (375232, True), (485408, False))


def script_of(case):
    """The scene `script` of a case: its scripted CA pauses (scene clock samples)."""
    return [dict(at=int(at), kind='pause', args=dict(on=bool(on))) for at, on in case.get('pauses', ())]


CASES = [
    dict(id='od_d1', title='D1 - Хвост меняется вместе с фазами фигуры: Fixed 0.948 s / Common age (Octagon II p5)',
         case='D1', rate_hz=2.0, seconds=20.0,
         law=dict(A=LAW_FIXED, B=LAW_COMMON), decay=dict(A=0.947926, B=STABLE_DECAY_S),
         window=('D1', 5.0, 20.0),
         side_a='А — Objects / Laplace, Decay law Fixed, Decay 0.947926 s',
         side_b='Б — Objects / Laplace, Decay law Common age, Stable decay 1.39 s',
         note='Octagon II (период 5) из preflight — одна связная фигура во всех фазах; 20 с, 2 поколения/с, f0 = 110 Hz; '
         'обе стороны — ' + SETTINGS_TEXT + '. В А одна постоянная скорость потерь (Decay 0.947926 s — средняя целевая '
         'скорость потерь Б на участке 5…20 с при реальных границах блока, не подбор на слух). В Б всем модам банка даётся '
         'одна скорость потерь, вычисленная из истории клеток (свежая клетка гасит быстрее, устойчивая сохраняет дольше; '
         'память клетки tau 0.5 с, T от 0.08 с до Stable decay); на переходах зрелого цикла Common T ≈ 0.651 / 0.415 / '
         '1.156 / 0.853 / 0.735 с, между поколениями меняется плавно (сглаживание 20 мс). Частоты, удары, веса, Attack '
         'и банк одинаковы; различаются только потери и постоянная калибровка уровня стороны (по RMS за 5…20 с). '
         'Ручки Decay law и Decay можно двигать вживую.',
         hypothesis='Гипотеза - В А у звуков одна постоянная скорость затухания. В Б она меняется вместе с жизнью '
         'клеток: после одних шагов хвост должен уходить быстрее, после других сохраняться дольше. Ожидаем изменение '
         'рисунка затуханий внутри повторяющейся фразы. Слышно ли это после первых пяти секунд и интересно ли звучит? '
         'Если Б просто целиком кажется суше или длиннее, без различимого изменения от шага к шагу, так и напиши.'),
    dict(id='od_d2', title='D2 - Две частоты расходятся по длине хвоста: Common age / Modal age (Blinker p2, две паузы)',
         case='D2', rate_hz=2.0, seconds=16.0,
         law=dict(A=LAW_COMMON, B=LAW_MODAL), decay=dict(A=STABLE_DECAY_S, B=STABLE_DECAY_S),
         pauses=D2_PAUSES, window=('D2', 2.0, 16.0),
         side_a='А — Objects / Laplace, Decay law Common age, Stable decay 1.39 s',
         side_b='Б — Objects / Laplace, Decay law Modal age, Stable decay 1.39 s',
         note='Blinker (период 2), клетки (10,9),(10,10),(10,11) из preflight; 16 с, 2 поколения/с, f0 = 110 Hz; обе '
         'стороны — ' + SETTINGS_TEXT + '. Две частоты 110 и ≈216.168 Hz с равными весами. Поле дважды специально '
         'останавливается (Pause CA в журнале записи: 3.009…5.507 с и 8.509…11.007 с, через один блок после последнего '
         'удара), чтобы отдельно услышать хвост; звук и история клеток во время пауз продолжаются. Центр блинкера живёт '
         'непрерывно, края сменяются каждый шаг: в Б в момент паузы целевые T ≈ 0.101 / 0.958 с (нижний тон гаснет '
         'быстрее, верхний остаётся), в А обе ≈ 0.182 с (средняя скорость потерь та же). Различаются только '
         'распределение потерь между частотами и постоянная калибровка уровня (по RMS активных участков после 2 с, '
         'без пауз). После записи Pause можно нажимать самому.',
         hypothesis='Гипотеза - Поле дважды остановится, чтобы можно было услышать, как заканчивается звук. В А две '
         'частоты теряют силу вместе. В Б нижний тон должен уходить быстрее, а верхний оставаться в хвосте: окраска '
         'звука меняется по мере затухания. Проверяем, слышно ли именно это расхождение, а не только разницу '
         'громкости или общей длины. После паузы блинкер продолжит двигаться.'),
    dict(id='od_d3', title='D3 - Разные хвосты в смеси двух периодов: Fixed 1.39 s / Modal age (Jam p3 + Octagon II p5)',
         case='D3', rate_hz=6.0, seconds=18.0,
         law=dict(A=LAW_FIXED, B=LAW_MODAL), decay=dict(A=STABLE_DECAY_S, B=STABLE_DECAY_S),
         window=('D3', 3.0, 18.0),
         side_a='А — Objects / Laplace, Decay law Fixed, Decay 1.39 s',
         side_b='Б — Objects / Laplace, Decay law Modal age, Stable decay 1.39 s',
         note='Jam (период 3, смещение (6,4)) + Octagon II (период 5, смещение (6,20)) из preflight, 29 исходных клеток, '
         'семейства не пересекаются (проверено 120 переходов); 18 с, 6 поколений/с, f0 = 110 Hz; обе стороны — ' +
         SETTINGS_TEXT + '. В А прежний звуковой механизм с общим Decay 1.39 s; в Б каждая мода каждой фигуры получает '
         'свою скорость потерь из истории её клеток (Stable decay 1.39 s — верхняя граница T). Средние потери сторон '
         'здесь намеренно не равны: проверяется весь механизм. Распады и слияния Jam, хвосты исчезнувших банков — '
         'штатные. Различаются только потери и постоянная калибровка уровня (по RMS за 3…18 с).',
         hypothesis='Гипотеза - В А знакомая смесь двух периодов звучит с одинаковым законом хвоста для всех частот. '
         'В Б история клеток определяет, какие составляющие быстро гаснут, а какие продолжают звучать. Ожидаем более '
         'различимый характер отдельных звуков и их наложений. Даёт ли это интересное развитие при прослушивании и '
         'игре или смесь просто становится короче, беднее либо менее приятной? Если разница слабая, это тоже '
         'результат.'),
]


def listen_text(case):
    return ('1 / 2: ' + case['side_a'] + ' / ' + case['side_b'] + '. Notes: hypothesis and your listening '
            'result. Decay law (Fixed / Common / Modal) and Decay are the knobs of the Objects panel; under Common / '
            'Modal age Decay is the stable (upper) T60 and each figure row shows its current target T range.')


def scene_for(case, side_gain=None):
    doc = _scene_doc(case, side_gain)
    if case.get('pauses'):
        doc['script'] = script_of(case)
    return doc


def _scene_doc(case, side_gain):
    return dict(format=2, id=case['id'], title=case['title'], grid=dict(rows=ROWS, cols=COLS),
                cells=case_cells(case['case']), rule='B3/S23', boundary='torus', rate_hz=float(case['rate_hz']),
                audio=dict(f0_hz=F0_HZ, level=1.0,
                           side_gain=dict(side_gain if side_gain is not None else SIDE_GAIN[case['id']])),
                variants=dict(A=dict(engine_id=ENGINE_ID, engine_params=side_params(case['law']['A'], case['decay']['A'])),
                              B=dict(engine_id=ENGINE_ID, engine_params=side_params(case['law']['B'], case['decay']['B']))),
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


def render_sides(doc, seconds, commands=()):
    """PCM of both sides of a scene document driven offline with 'start' at 0 plus
    `commands` [(kind, at_samples, args)]; (pcm {A, B}, peaks, clip blocks, runner)."""
    runner = DemoRunner(scene_from_doc(doc))
    runner.post('start', at=0)
    for kind, at, args in commands:
        runner.post(kind, at=at, **args)
    pcm = {s: [] for s in ('A', 'B')}
    peak = {s: 0.0 for s in ('A', 'B')}
    for _ in range(int(math.ceil(seconds * SR / BLOCK))):
        blk = runner.next_block()
        for s in ('A', 'B'):
            pcm[s].append(blk.get(s))
            peak[s] = max(peak[s], runner.sides[s].peak)
    return {s: np.concatenate(pcm[s]) for s in ('A', 'B')}, peak, dict(runner.snapshot()['clip_blocks']), runner


def window_mask(case, n_samples):
    """bool (n,): the samples of the REQ's level window of a case -- D1 5..20 s, D3
    3..18 s, D2 after 2 s with the two pauses (from the pause commands) excluded."""
    _id, t0, t1 = case['window']
    k = np.arange(n_samples)
    mask = (k >= int(round(t0 * SR))) & (k < int(round(t1 * SR)))
    pauses = list(case.get('pauses', ()))
    for a, b in zip(pauses[0::2], pauses[1::2]):
        assert a[1] is True and b[1] is False
        mask &= ~((k >= a[0]) & (k < b[0]))
    return mask


def calibrate(cases=None):
    """RMS of both sides at side gain 1 -- the whole take and the REQ window -- and the
    B factor that would equalise the window (printed; the constants above are set by
    hand from it)."""
    out = {}
    for case in (CASES if cases is None else cases):
        y, peak, clip, _r = render_sides(scene_for(case, side_gain=dict(A=1.0, B=1.0)), case['seconds'])
        m = window_mask(case, y['A'].shape[0])
        a, b = rms_db(y['A']), rms_db(y['B'])
        a2, b2 = rms_db(y['A'][m]), rms_db(y['B'][m])
        out[case['id']] = dict(rms_a_db=a, rms_b_db=b, a_minus_b_db=a - b, window=list(case['window'][1:]),
                               window_rms_a_db=a2, window_rms_b_db=b2, window_a_minus_b_db=a2 - b2, peak=peak,
                               b_gain_for_equal_window_rms=round(10.0 ** ((a2 - b2) / 20.0), 2), clip_blocks=clip)
        print(f"  {case['id']}: whole A {a:.2f} B {b:.2f} dB (A-B {a - b:+.2f});  window {case['window'][1]:g}.."
              f"{case['window'][2]:g} s A {a2:.2f} B {b2:.2f} dB (A-B {a2 - b2:+.2f}) -> B gain "
              f"{out[case['id']]['b_gain_for_equal_window_rms']}  peaks A {peak['A']:.3f} B {peak['B']:.3f}  clip {clip}",
              flush=True)
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
        rid, snap = record_offline(cat, scene_for(case), case['seconds'], [], case['title'], note)
        cat.write_notes(rid, case['hypothesis'] + '\n\n')
        results.append((rid, case['id'], dict(snap['clip_blocks']), snap['gen'], snap['paused']))
        print(f"  {rid}  {case['id']}  clip={snap['clip_blocks']} gen={snap['gen']} paused={snap['paused']}", flush=True)
        time.sleep(max(0.0, 1.05 - (time.monotonic() - started)))   # ids carry a 1 s stamp
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="build the Objects Decay law scenes and catalog (D1 / D2 / D3)")
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
