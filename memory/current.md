# Current Implementation State

_Снапшот живого состояния (консолидирован /dream 2026-07-14). Полная история —
`memory/archive/current-history-2026-06-16.md` + `memory/log/` (нарративы фиксов);
решённые вопросы — `memory/archive/questions-resolved.md`._

## Карта файлов

| Файл | Что это |
|---|---|
| `gol_synth.py` | **Активный прототип** (мульти-движковый): event-loop, layout, аудио/MIDI-потоки |
| `casynth_config.py` | ВСЕ константы (геометрия/аудио/слоты/ADSR/палитра); pygame-free |
| `casynth_engine.py` | `step/analyse/events_field/SlotPool/render_chunk_laplacian/midi_to_freq` |
| `casynth_core.py` | Либа маппинга форма→(freqs,amps): 5 `map_*` + реестр `ENGINES` (чистые данные) |
| `casynth_session.py` | Session-рекордер: `_dump_session`/`replay_session`/CLI `replay <ts>` |
| `casynth_ui.py` | Весь рендеринг (`draw_frame`, пианино, спектр-полоска) |
| `casynth_midi.py` / `casynth_midifile.py` | MIDI-вход (rtmidi callback) / MIDI-file плеер |
| `casynth_tuning.py` | Сетхарес: `pair_dissonance/dissonance_curve/scale_minima/snap_ratio` |
| `gol_life_synth_laplacian.py` | Предшественник (одно-движковый Laplace), заморожен; движок разошёлся намеренно |
| `gol_life_synth.py` | Legacy v1 (size→harmonic, pygame.mixer) |
| `mapping_bench.py` / `ab_bench.py` | Discover-стенд (матрица приёмов) / A/B слуховой стенд (+`ab_presets.py`) |
| `gol_vocoder.py` | Offline GoL-вокодер (SA-ресинтез) — **SHELVED 2026-07-14** |
| `laplacian_explainer/`, `frontiers_explainer/` | Обучалки (JS дублирует ядро — известный trade-off) |
| `patterns.py` | 29 паттернов в 9 категориях |
| `demo_bench.py` + `casynth_lab/` + `demos/` | **Demo-стенд S1+S2** (2026-09-07/08): `DemoRunner` (единый live/offline исполнитель, общее поле + две стороны A/B, часы в сэмплах, очередь команд + журнал, crossfade 20 мс в мониторе), `LiveEngine` (render-поток + sounddevice), сцены JSON v1/v2; запуск `run_demo_bench.bat` (`laplace_ab.json`) |
| `check.py` | **Все гейты одной командой** (см. «Гейты») |
| `tests/` | `test_casynth_core.py` (59 тестов) + `test_demo_lab.py` (33 теста S1+S2) + `golden/` (эталоны golden-master и UI) |

## Гейты (обязательны после каждого изменения)

`python check.py` = 59 юнит-тестов + 33 теста demo-стенда S1+S2 + golden-master аудио (байт-в-байт,
sha256[:16]=cd3126907ad13b6c) + UI-кадр (пиксель-в-пиксель vs `tests/golden/ui_frame.png`)
+ import/init smoke. Эталоны ВЕРСИОНИРУЮТСЯ в `tests/golden/` (переехали из gitignored
`artifacts/` 2026-07-14). Легитимное изменение UI → `python check.py --bless-ui` в том же
изменении. Конвенция каждой новой ручки: **дефолт = бит-в-бит**.

## Ядро и движки (casynth_core)

- 5 движков в реестре `ENGINES`: FFT / Walsh / Random / **Laplacian** (активный) / Granulo.
- `map_laplacian(patch, f0, n, spread, alpha, shape, harm, fullshape, dyn, exc)`:
  граф-Лапласиан живых клеток, √λ → частоты, низшая взятая мода всегда = f0.
  - `spread/alpha` — отбор мод / rolloff-яркость (приняты 2026-06-17).
  - `shape` [0,1] — blend rolloff ↔ проекция краевого возбуждения `|⟨e,φ_k⟩|`, `e=deg`
    (принят на слух 2026-06-22). Частоты не трогает.
  - `harm` [0,1] — притяжение отношений частот к целым, ось металл→гармоника (принят на
    слух 2026-06-22; лечит регистровую «грязь»). Только частоты.
  - `fullshape` (bool) — граф по ВСЕЙ форме (тайтовая маска `sub`, потолок
    `MAX_LAPLACIAN_NODES=256` с решёточной децимацией). Дефолт ядра False (бит-в-бит);
    реестровый Laplacian = **True**. Реализован 2026-06-22.
  - `dyn` [0,1] — `e` из событий поколения (`events_field`: рождения 1.0 + живая кромка
    смертей `W_WOUND=0.7`; держится между шагами; per-object маскирование в analyse;
    геометрическое выравнивание exc↔patch на обоих путях). Слуховой гейт закрыт
    2026-07-06: **слабый рычаг**, остаётся «краской» (дефолт 0).
- `tune` [0,1] (synth-wide, `casynth_tuning.py`) — Сетхарес-магнит несущей к минимумам
  кривой диссонанса колонии на note-on; `state['tuned_f0']` float; октавная свёртка;
  якорь Т-тетромино 600.49¢. Принят по code-review 2026-07-06.

## Прототип gol_synth.py (инженерия, не под дизайн-ревью)

- **Модель огибающих (рефактор 2026-07-07, коммиты 983e1dc…6fb093a):** КА-поле =
  свободнобегущий осциллятор (`analyse` при фиксированном `REFERENCE_F0`, пул кормится
  всегда); несущая = **скалярный транспоз на рендере** (active+tail, фазо-непрерывно,
  ноты НЕ ретриггерят моды — чинит «зависание» и churn хвостов); **GEN ADSR** per-mode в
  долях интервала тика (масштабируется с BPM); **Voice ADSR (VCA)** — note-on/off скаляр
  над суммой (дефолты = no-op). GEN amp-slew тумблер против «beep» при shape>0 (6fb093a).
- **Аудио:** sounddevice callback-стрим, SlotPool active+tail слоты с кросс-фейдом хвостов,
  `MAX_VOICES=24`; mixer форсирован в stereo (`allowedchanges=0`).
- **UI:** поле (рисование/стирание), play/pause/step/random/clear, BPM+деление, сайдбар
  29 паттернов (drag-and-drop), динамическая панель ручек выбранного движка
  (`state['engine_params']` помнит значения по движку), блоки VOICE/GEN/tune, спектр-полоска
  (сырой выход движка до ADSR, лог-ось 20..20k), level/clip-метр, легенда, DPI-aware окно.
- **MIDI:** вход через rtmidi-callback (gate-семантика, селектор устройств); MIDI-file
  плеер (`♪`-кнопка + CLI `play <file.mid>`), poly→mono = last-note по онсету, плеер драйвит
  только высоту+гейт; пустое поле авто-сидится random.
- **Session-рекордер (контракт: КАЖДАЯ ручка — read+write в одном изменении):**
  `CASYNTH_RECORD=1` пишет пер-кадровый `replay_controls` (снапшот ВСЕХ контролов) +
  `steps`/`step_prevs` (реконструкция exc, serial-матчинг после clear); чтение
  `python gol_synth.py replay <ts>` + sample-fidelity vs записанный WAV.
- **Несущая:** экранное/клавиатурное пианино C3–C5 (latched) + MIDI.

## Статус ревью

- Researcher ревьюит ТОЛЬКО `casynth_core.py` (map_* + данные реестра); прототипы/стенды —
  инженерия Developer.
- **Приняты (код):** map_* + spread/alpha/release (2026-06-17), shape (2026-06-20 + слух
  06-22), harm (06-22 + слух), dyn (07-06 + слуховой гейт закрыт), tune (07-06),
  explainer REQ-1/REQ-2 (06-23).
- **Реализовано, ревью НЕ закрыто:** `fullshape` (дизайн принят 06-22; формального ревью
  реализации и прослушивания нет — тред в `questions.md`).
- **SHELVED (Пользователь, 2026-07-14):** `gol_vocoder.py` (T3) — не фокус; вернуться по
  явному запросу.

## Known issues

- Segmentation не wrap-aware: объект на шве тора временно распадается на два голоса.
- Нет трекинга идентичности объектов (нет birth/death/merge/split-событий на уровне голосов).
- Заведено ~3 из ~8 спроектированных перцептивных осей — диапазон тембра узок.
- Клиппинг при плотном ярком поле — ручной headroom (метр+громкость); Researcher 06-17:
  для фазы прослушивания достаточно.
- **Оффлайн-реплей не воспроизводит VCA-артикуляцию** (per-frame путь superseded рефактором
  огибающих; лог пишет `voice_*` для контекста). Контракт рекордера дырявый для
  VCA-зависимых багов — вернуть при возрождении оффлайн-аудио-реплея.
- PyInstaller-сборка (`CASynth.spec`, dist/) — разовый эксперимент Пользователя, НЕ канал
  дистрибуции; spec закоммичен исторически (решить судьбу при следующей уборке).

## Ожидает прослушивания (очередь слуховых гейтов; готовить кейсы — через ab_bench/record+replay)

1. **`tune`** — главный слуховой вопрос ветки: Laplace-металл, A/B tune 0↔1 — интервалы
   из минимумов консонанснее 12-TET? Контроль: гармонический движок + tune=1 (магнит к JI).
2. **`fullshape`** — A/B «центр+shape (8×8) vs вся форма»: даёт ли граница различимость
   СВЕРХ достигнутой shape. (+ формальный code-review реализации.)
3. **Explainer-ы** — слуховой/визуальный пласт (solo-моды, раскраски) — не блокер.

_(A/B стенд `ab_bench.py` с пресетами (клавиши 1–9, `ab_presets.py`) — построен ровно
для этой очереди; спектр-полоски под полями показывают разницу глазом.)_

## In-flight / следующее

- **Demo lab (ТЗ `memory/req-demo-lab.md`, спринты S1–S9):** S1 принят 2026-09-08
  (`memory/log/2026-09-07-demo-lab-s1.md`). **S2 (A/B + параметры из реестра, 5 движков)
  сдан 2026-09-08** (`memory/log/2026-09-08-demo-lab-s2.md`, ТЗ `req-demo-lab-s2.md`),
  ждёт ручной приёмки (3 пункта в начале ТЗ). S3 (новый движок с памятью) — по отдельному ТЗ.

- **`acc` (событийные акценты, громкостный канал ветки ДИНАМИКА)** — дизайн принят
  2026-07-06 (REQ с критериями A–G в `decisions.md`), ЖДЁТ РЕАЛИЗАЦИИ. Переиспользует
  `events_field`, синергичен с dyn. **Фичи приостановлены Пользователем 2026-07-14
  («некогда смотреть») — стартовать по его команде.**
- **Движок `scan` («волны по пластине», scanned synthesis)** — ПРЕДЛОЖЕНИЕ Researcher
  (`memory/research/research-scan-engine-design-2026-07-07.md`), ждёт акцепта Пользователя;
  в decisions.md НЕ внесено.
- **VST-инструмент — новая цель Пользователя (2026-07-14):** прототипировать дальше в
  Python, но играть в DAW. Развилка моста (external-instrument через виртуальный
  MIDI/audio-кабель / тонкий плагин с контрольным каналом к Python-мозгу / порт DSP)
  НЕ решена — нужна дизайн-сессия. Ядро уже pygame-free и отделено от UI; golden-master =
  готовый паритет-эталон для любого порта.

## Candidate next steps (приоритизация за Researcher — decisions.md)

- Остальные перцептивные оси: jaggedness→distortion, activity→tremolo,
  symmetry→consonance, order/chaos→noise, elongation→detune/vibrato.
- Wrap-aware сегментация (union-find через шов тора); трекинг идентичности объектов.
- Профилирование законсервированных FFT/Walsh/Random/Granulo на крупных/плотных полях
  (живой селектор в gol_synth есть; формат стенда под крупные режимы — отдельная задача).
- Ветка ОБЪЕКТ, разведка 2026-06-22: двери A (граничный Фурье) / B (сеть задержек),
  ось топологии — ничего не выбрано.
- MIDI-reduction highest vs last-note (маппинг входа) — можно вернуться после рефактора
  огибающих.
