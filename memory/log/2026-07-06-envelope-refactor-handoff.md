# 2026-07-06 — HANDOFF: рефактор огибающих (КА = осциллятор, голос = классический)

> **ИСПОЛНЕНО 2026-07-07** — все шаги 1–5 в коде (коммиты 983e1dc…e097cf2 + cleanup).
> Как именно + отклонения от плана (golden-master НЕ переблагословляли: примитивы движка не
> менялись) — `memory/log/2026-07-07-envelope-refactor-done.md`. Ниже — исходный план.

Документ для продолжения в новой сессии. Дизайн-решение (ЧТО/ПОЧЕМУ) — в
`memory/decisions.md`, запись **2026-07-06 — Целевая модель огибающих**. Здесь — план
реализации (КАК) с конкретными точками касания и гейтами проверки. **Рефактор НЕ начат.**

## Контекст в одном абзаце
Синт монофонический: несущая нота задаёт f0, тембр рождается из живого поля Game of Life.
Сейчас **смена несущей ошибочно трактуется как смена частоты моды** → в `SlotPool` срабатывает
per-mode ретриггер attack по всем модам + спавн tail-слотов на КАЖДУЮ ноту. Отсюда два
наблюдавшихся дефекта: (1) «зависание» длинных нот, (2) churn/underrun'ы tail-пула на плотном
MIDI (щелчки, тормоза). Цель — переразвести на классическую модель.

## Целевая модель (принята Пользователем 2026-07-06)
```
КА-поле (тикает по часам автомата, СВОБОДНО, независимо от нот)
  → analyse → моды {частота при reference f0, амплитуда из формы}
  → Generation ADSR на моду (attack на рождении, release на смерти)   ← часы АВТОМАТА, ноты НЕ ретриггерят
  ═ ОСЦИЛЛЯТОР (питч-нормированный живой спектр)
  × транспонирование к несущей f0                                      ← ПИТЧ ноты (фазо-непрерывно, БЕЗ ретриггера)
  → Σ мод/объектов (стерео pan)
  × Voice ADSR (VCA)                                                   ← note-on attack / note-off release (ретриггер на note-on)
  × master gain
```
Три часовых домена (автомат / питч ноты / note-on-off) больше не смешиваются.
Подтверждено Пользователем: осциллятор свободнобегущий; два раздельных набора ручек
(Voice в мс, Generation в долях интервала тика); транспоз мгновенный (портаменто — потом).

## Ключевая идея развязки (главное!)
`analyse` звать при **фиксированном reference f0 = `midi_to_freq(NOTE_DEFAULT)`** → пул живёт в
питч-нормированном пространстве. Несущую применять как **транспоз на рендере**
(`freq_слота × ratio`, где `ratio = (tuned_f0 при tune>0 иначе midi_to_freq(note)) / reference`).
Тогда `SlotPool._freq_prev`-сравнение видит только смену ФОРМЫ (тики), не ноту. На дефолтной
ноте `ratio=1` → бит-в-бит с golden-master.

## Точки касания в коде (проверены 2026-07-06)
- `gol_synth.py`:
  - `_live_voices` (≈513–544) — сейчас рескейлит freqs на ratio ПЕРЕД пулом; станет: отдаёт
    reference-голоса как есть (rescale уезжает в рендер).
  - публикация `render_spec` в `main()` (≈940–947): `base_f0 = f0()` → `analyse(...)` →
    `render_spec['cur'] = {'voices','base_f0'}`. Менять `base_f0` на reference; добавить в спек
    интервал тика (`_step_interval()`) для Generation-долей.
  - `_render_loop` (≈609–633): семплит note/gate; `voices_in = _live_voices(...) if gate else []`
    (Шаг 2b: убрать зависимость от gate — осциллятор всегда кормится); `pool.update(...)`;
    `render_chunk_laplacian(...)`. Здесь же вести Voice-VCA (advance по чанку, применить гейн).
  - `_snap_note_on` (≈551–607) — tune; ratio транспоза должен брать `tuned_f0` при tune>0.
  - `replay_controls` dict (≈965–978) — добавить новые контролы read+write.
- `casynth_engine.py`:
  - `render_chunk_laplacian` (≈201–253) — СЮДА добавить умножение частоты слота (active И tail)
    на `ratio` транспоза.
  - `SlotPool.update` (≈392–500) — `_freq_prev` (318), env-state (322–324), tail/release
    (`_acquire_tail` 337–352, `_enforce_tail_budget` 354–369, Phase-2 486–500).
  - `_advance_env` (371–389) — Шаг 4: длины в долях интервала тика.
- `casynth_config.py` — дефолты/диапазоны ADSR (`ATTACK_MS_*`, `DECAY_MS_*`, `SUSTAIN_*`,
  `RELEASE_MS_*`), `TOOLBAR_H` (растёт под новые ручки, как было для `tune`: +22px/строка).
- `casynth_ui.py` — ENV-блок (лейблы/раскладка ручек A/D/S/R; добавить Voice-строки, релейбл Gen).
- ENV-ручки строятся в `main()` ctrls (tune добавлялась как 5-я строка ENV — тем же паттерном).
- `casynth_session.py` — read-side `replay_session`/`replay_controls` (симметрия лога).
- Гейты: `artifacts/_golden_master.py` (+`_golden_master_ref.npz`), `CASYNTH_DUMPFRAME`
  (+`_ui_before.png`), `tests/test_casynth_core.py`.
- reference f0 = `midi_to_freq(NOTE_DEFAULT)`.

## Порядок шагов (каждый: прототип играет; один сдвиг поведения за шаг)
**Шаг 0 — сеть безопасности (без кода).** record→replay на сцене из 5 фигур; зелёные
golden-master + UI-dumpframe; A/B-проба `audio-artifact-probe`. Эталон «до».

**Шаг 1 — развязать транспонирование (питч-нормированный пул). ⟵ ЯДРО.**
analyse при reference f0; `_live_voices` отдаёт reference-голоса; `render_chunk_laplacian`
умножает частоту (active+tail) на `ratio`. Проверка: golden-master **БИТ-В-БИТ** (нота=дефолт);
на MIDI churn и «зависание» уходят, питч без щелчка (проба). *Devil's advocate:* проверить
фазо-непрерывность на больших скачках.

**Шаг 2a — Voice ADSR (VCA) как no-op.** Скалярный env в render-треде + гейн; UI-ручки (мс);
`replay_controls` read+write. Дефолты (A=0/D=0/S=1/R≈сейчас) → множитель ≡1. golden-master
**БИТ-В-БИТ**.

**Шаг 2b — note-off → VCA (RE-BLESS).** Пул кормится живыми голосами ВСЕГДА (gate не опустошает
`voices_in`); Voice ADSR: note-on→attack, note-off→release. Осциллятор свободнобегущий. Хвосты
теперь только от смертей мод. Проверка на слух. *Devil's advocate:* два релиза (VCA + Gen-хвосты)
— не «двойной шлейф»?

**Шаг 3 — ретриггер Voice ADSR на note-on (RE-BLESS).** Смена несущей при поднятом gate → сброс
Voice-фазы в attack. Легато НЕ делаем (позже сверху). Каждая нота артикулируется.

**Шаг 4 — Generation ADSR в долях интервала тика + релейбл (RE-BLESS).** Длины = доля ×
`interval_s/CHUNK_S` (интервал из спека, кламп чтобы влезть); диапазоны 0..1; лейблы «Gen A/D/S/R»;
`replay_controls`. Проверка: BPM меняет текстуру. *Devil's advocate:* поведение клампа на быстром
автомате (срез decay vs пропорциональное сжатие).

**Шаг 5 — уборка.** Переименования (`_gen_env`/`voice_env`), комментарий tail-пула → «Release
Generation-огибающей», тесты ядра, лог сессии, освежить `current.md`.

## Правила/оговорки
- **Правило лога:** новый контрол = сразу read+write в `replay_controls`/`replay_session`.
- **Golden-master:** бит-в-бит держим до Шага 2b; дальше RE-BLESS осознанно (звук меняется по
  замыслу) — сохранить старый ref как `_golden_master_ref_pre_envelope.npz` для сравнения.
- **Развилка MIDI-reduction (highest vs last-note)** — ОТДЕЛЬНАЯ тема (маппинг входа), вернуться
  ПОСЛЕ рефактора.

## Состояние репозитория на момент хэндоффа
- Ветка `master`, последний коммит `cdb6222` (spectrum strip). Рабочее дерево **грязное и много
  чего не закоммичено ещё до этой сессии** (dyn/tune/ab_bench/frontiers_explainer и логи).
- **Добавлено этой сессией (не закоммичено):** MIDI-file плеер — новый `casynth_midifile.py`,
  правки `gol_synth.py` (кнопка/плеер/last-note-редукция/Event.wait-планировщик) и `casynth_ui.py`
  (кнопка «♪ Play MIDI file»). Детали фичи — `memory/current.md` (запись «MIDI-file плеер»).
  Тест-файл `moonlight-sonata.mid`. 58/58 тестов, golden-master бит-в-бит на момент записи.
- Рефактор огибающих кода НЕ касался.
