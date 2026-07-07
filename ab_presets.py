"""ab_presets.py — agent-authored A/B listening presets for ab_bench.py.

The workflow this file exists for: an agent proposes a fix hypothesis, writes the
discriminating A/B cases here, the human launches the bench and listens.  No
knob-twiddling by hand.

Format
------
PRESETS is a list of dicts.  Launch `python ab_bench.py`, press the preset's
number key (1..9) to load it, then click field A or B to hear (only one sounds).
`python ab_bench.py --preset 2` auto-loads on start.  `--selftest` validates that
every preset applies and renders headlessly.

Each preset:
    key      : str digit shown in the UI / pressed to load ('1'..'9').
    name     : short id.
    listen   : one line — WHAT to listen for and what each outcome proves.
    scene    : a CASES name (str, e.g. 'beacon') OR an inline case dict
               dict(name=, gw=, gh=, place=[(pattern_name, x, y), ...]).
    common   : dict(note=<midi>, speed=<ticks/s>, vol=<0..1>)  (all optional).
    A, B     : field specs.  Each may set:
        engine     : engine id ('laplacian', 'fft2d', 'walsh', 'random', 'granulo')
        params     : per-engine param overrides merged onto registry defaults
                     (laplacian: shape, spread, alpha, harm, dyn, fullshape, n).
        attack_ms, decay_ms, release_ms : per-mode ADSR lengths in ms.
        sustain    : 0..1 held level.
        tune       : 0..1 Sethares snap amount.

Note on ADSR units: the bench ADSR is in MILLISECONDS (the older per-mode model),
while the main synth's GEN ADSR is FRACTIONS OF A TICK.  To mirror "GEN A=1 tick"
at a given speed, use attack_ms = 1000 / speed  (e.g. speed 6 -> ~167 ms).

Current battery: the shape>0 "beep" investigation (2026-07-07).  Root cause found:
the GEN/per-mode attack re-triggers only on a mode's FREQUENCY change, never on an
amplitude change; shape=1 makes a stable-frequency mode (the 131 Hz fundamental of
the beacon) gate its amplitude 1<->0 every tick, which bypasses the attack -> a
hard on/off beep.  These four presets triangulate that on the `beacon` case.
"""

# one automaton tick at speed=6 is ~167 ms; that is the "A=1 / R=1 tick" reference.
_TICK_MS = 1000.0 / 6.0

PRESETS = [
    dict(
        key='1', name='beep-repro',
        scene='beacon',
        listen="A(shape 1) бипает: фундамент 131Гц гейтится вкл/выкл каждый тик; "
               "B(shape 0) держит его ровным дроном. Подтверждает, что баг есть и он от shape.",
        common=dict(note=48, speed=6.0),
        A=dict(engine='laplacian', params=dict(shape=1.0),
               attack_ms=1, release_ms=_TICK_MS, sustain=1.0),
        B=dict(engine='laplacian', params=dict(shape=0.0),
               attack_ms=1, release_ms=_TICK_MS, sustain=1.0),
    ),
    dict(
        key='2', name='attack-does-nothing',
        scene='beacon',
        listen="Оба shape 1; A attack=1мс, B attack=300мс. Низкий БИП (131Гц) одинаков в A и B "
               "-> атака до него не доходит (частота стабильна, ретригера нет). Может чуть мягче "
               "звенеть верхний шиммер 440Гц у B — на него смотреть НЕ надо.",
        common=dict(note=48, speed=6.0),
        A=dict(engine='laplacian', params=dict(shape=1.0),
               attack_ms=1, release_ms=_TICK_MS, sustain=1.0),
        B=dict(engine='laplacian', params=dict(shape=1.0),
               attack_ms=300, release_ms=_TICK_MS, sustain=1.0),
    ),
    dict(
        key='3', name='anchor-floor',
        scene='beacon',
        listen="A(shape 1) vs B(shape 0.4). При 0.4 rolloff держит фундамент на ~0.86 вместо 0 -> "
               "бип пропадает/смягчается. Проверяет фикс 'пол rolloff' (вариант B) и просто 'играй "
               "на промежуточном shape' (вариант 0).",
        common=dict(note=48, speed=6.0),
        A=dict(engine='laplacian', params=dict(shape=1.0),
               attack_ms=1, release_ms=_TICK_MS, sustain=1.0),
        B=dict(engine='laplacian', params=dict(shape=0.4),
               attack_ms=1, release_ms=_TICK_MS, sustain=1.0),
    ),
    dict(
        key='4', name='release-irrelevant',
        scene='beacon',
        listen="Оба shape 1; A release=20мс, B release=800мс. Разницы в бипе нет -> это амплитудный "
               "гейт на стабильной высоте, а не трель (релиз срабатывает только на смене частоты). "
               "Подтверждает подслучай B.",
        common=dict(note=48, speed=6.0),
        A=dict(engine='laplacian', params=dict(shape=1.0),
               attack_ms=1, release_ms=20, sustain=1.0),
        B=dict(engine='laplacian', params=dict(shape=1.0),
               attack_ms=1, release_ms=800, sustain=1.0),
    ),
]
