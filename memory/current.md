# Current Implementation State

_Снапшот живого состояния (консолидирован /dream 2026-07-14; демо S/N добавлены 2026-09-14). Полная история —
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
| `demo_bench.py` + `casynth_lab/` + `demos/` | **Demo-стенд S1–S4** (2026-09-07/09): `DemoRunner` (единый live/offline исполнитель, общее поле + две стороны A/B, часы в сэмплах, очередь команд + журнал, crossfade 20 мс в мониторе), `LiveEngine` (render-поток + sounddevice), сцены JSON v1/v2; движки через `registry.py` (интерфейс `engine_api.SoundEngine`, адаптер `legacy_engine.py`); каталог опытов `recorder.py`/`catalog.py`; **снепшоты S5** (2026-09-12): `snapshot.py` (JSON+npz, без pickle), `export_state`/`restore_state` в интерфейсе движка и адаптере, `DemoRunner.export_state/from_state`, «Continue» / ветки с `parent_record_id`; **версии S6** (2026-09-13): `provenance.py` (отпечаток runtime-набора, git-соответствие, окружение), `versions.py` (кэш worktree + дочерний стенд по договору `--catalog/--record/--action`), статусы Pinned/Local, Pin, «Continue in version»; **проверка каталога S7** (2026-09-13): `verify.py` (`Catalog.recompute` — единый путь реплея; `compare_pcm`, `Verifier.run`/`run_in_subprocess`, отчёты `lab_catalog/local/.verify/<run>/`, отметки на слух, проверка пар по отпечаткам), плеер пары Saved/Recomputed на одном курсоре (`audio_out.play_pair`), экран отчёта в `demo_bench.py`; демо-каталог `tests/s7_demo_catalog.py`; запуск `run_demo_bench.bat` (`laplace_ab.json`) |
| `casynth_lab/scan_surface.py` + `casynth_lab/pm_network.py` | **Демо S/N (2026-09-14, ТЗ `memory/req-sonification-sn-demos-2026-09-14.md`)**: движки стенда `scan_surface` (Scan: поверхность Smooth/Distance + путь Ellipse/Lissajous/Raster, band-limited аддитивный рендер с непрерывной фазой, 20-мс переход коэффициентов) и `pm_network` (Network: 4 генератора 1..4·f0, 6 направленных PM-связей из шести гауссовых масок поля, 4× рендер + FIR-децматор + HPF 5 Гц, сглаживание W/beta 30 мс, gate 20 мс, frozen links); оба с полным snapshot (Continue байт-в-байт). Подсказки реестра `choices/inactive/overlay` + `display()` движка → панель со словесными режимами, `Full field`, оверлей пути/масок, полоски W (`demo_bench.py`, геометрия панели от числа движков, прокрутка каталога колёсиком). Материал: `demos/build_sn_demos.py` (поля F1–F5 + Kok, 15 сцен `demos/sn_*.json`, каталог `lab_catalog/sn_demos_2026_09_14/{scan,network}/`), входы `run_scan_demo.bat` / `run_network_demo.bat`, тесты `tests/test_demo_lab_sn.py` (15). Уровни: сырой запас ×8, trim по умолчанию Scan +6 / Network +3 dB (RMS = Laplace на F3). Отчёт `memory/log/2026-09-14-sn-demos.md`. **Запуск стенда (2026-09-14): `demo_bench.py --demo <scene> [--catalog ROOT]` открывается ЭКРАНОМ КАТАЛОГА (сцена за ним на паузе, Esc/Back → к ней; двойной клик по записи = Continue); `--live` открывает сразу звучащее поле — использовать в автотестах/скриптах, которым нужен живой экран; `--render`/`--record` не затронуты. **Notes (2026-09-14):** кнопка Notes в живом окне и в каталоге — свободный текст записи (`<record>/notes.md`, сохраняется при вводе; принадлежит продолженной/открытой записи или последней сохранённой в сессии); **фидбек Пользователя вытаскивать: `python -m casynth_lab.catalog notes lab_catalog/<root>`** (Markdown; `--all` — плюс описания Save). **Отпечаток происхождения (2026-09-14) = только ЗВУКОВОЙ набор** (`casynth_core/engine/config`, `casynth_lab/{runner,engine_api,registry,legacy_engine,scene,snapshot,движки}.py`); правки `demo_bench.py`, `audio_out/catalog/recorder/verify/versions/provenance`, `requirements.txt`, `demos/*.json` записи НЕ перепривязывают; старые записи (set 1) сравниваются по содержимому на их коммите. Диагностика: `python -m casynth_lab.provenance [why ROOT]`** |
| `demos/network_reference_n0/` + `demos/results/network_reference_n0/` | **N0 — референс Gutter Synthesis (2026-09-15, ТЗ `memory/req-network-reference-n0-2026-09-15.md`)**: `gutter_node.py` — скалярный порт исполняемого `gutterOsc.class` (НЕ `gutterOsc.java`), побитово сверен с оригиналом на JVM (`verify_node.py` + `javaref/`, JDK не в репо: `--jdk`, `--source <clone>`); `jmath.py` — fdlibm atan / Java-семантика f32, div, sin, exp; `gutter_network.py` — 8 узлов + Max-цепочка (`matrix~`→`delay~` 2000+64→×interaction→демпфирование `c`; `clip~`/`svf~`/`tanh~`/pan), слайдеры источника `set_slider(name, raw 0..256)`, `set_matrix`, `export_state/from_state`, ~52 мкс/отсчёт; `source_manifest.py` — commit/хеши/таблица параметров/банки `filters.txt`; `analysis.py` — метрики; `render_n0.py` — один запуск строит пакет (пробы 3 оси × 5 параллельно, `01_palette`/`02_control`/`03_links`/`03b_links_strong`, `manifest.json`, `figures/`, `raw/` gitignored). Отчёт `memory/log/2026-09-15-network-reference-n0.md`; допущения Max-обвязки — `questions.md` (OPEN). Не подключён к стенду/КА (это N1). |
| `check.py` | **Все гейты одной командой** (см. «Гейты») |
| `tests/` | `test_casynth_core.py` (59 тестов) + `test_demo_lab.py` (39 тестов S1–S3) + `test_demo_lab_s4.py` (9 тестов S4) + `test_demo_lab_s5.py` (6 тестов S5) + `test_demo_lab_s6.py` (6 тестов S6, изолированные git-репо в `artifacts/_s6/`) + `test_demo_lab_s7.py` (6 тестов S7, `artifacts/_s7/`; демо-каталог `s7_demo_catalog.py`) + `test_demo_lab_sn.py` (15 тестов демо S/N, `artifacts/_sn/`) + `golden/` (эталоны golden-master и UI) |

## Гейты (обязательны после каждого изменения)

`python check.py` = 59 юнит-тестов + 39 + 9 + 6 + 6 + 6 + 15 тестов demo-стенда S1–S7 и демо S/N + 9 тестов N0 (гейт 1h) + golden-master аудио (байт-в-байт,
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

- **N0 (2026-09-15) сдан технически** — пакет `demos/results/network_reference_n0/` ждёт
  технического ревью Researcher и короткого прослушивания Пользователем
  (`memory/req-listen-network-reference-n0-2026-09-15.md`). На слух никто не проверял.
  Следующий шаг N1 (КА → один канал управления) — отдельное ТЗ Researcher по итогам.

- **Demo lab (ТЗ `memory/req-demo-lab.md`, спринты S1–S9):** S1 принят 2026-09-08
  (`memory/log/2026-09-07-demo-lab-s1.md`). **S2 (A/B + параметры из реестра, 5 движков)
  сдан 2026-09-08 и принят Пользователем 2026-09-09** (`memory/log/2026-09-08-demo-lab-s2.md`;
  по приёмке добавлены: копирование настроек `<<`/`>>`, Factory A+B, звёздочка изменённой
  стороны, хоткеи R / Space / 1 / 2). **S3 (интерфейс звукового движка + реестр стенда,
  ТЗ `req-demo-lab-s3.md`) сдан 2026-09-09** (`memory/log/2026-09-09-demo-lab-s3.md`):
  `casynth_lab/engine_api.py` / `legacy_engine.py` / `registry.py`; пять методов —
  адаптер под прежними id, звук S2 закреплён `tests/golden/demo_lab_s2_ref.json`.
  Принят (пользователь запросил S4). **S4 (локальный каталог опытов + повтор с начала,
  ТЗ `req-demo-lab-s4.md`) сдан 2026-09-09** (`memory/log/2026-09-09-demo-lab-s4.md`):
  `casynth_lab/recorder.py` + `catalog.py`, записи в `lab_catalog/local/<id>/` (gitignored),
  повтор в подпроцессе. **Принят Пользователем 2026-09-10** после доработок: каталог —
  отдельный экран (КА на паузе, живой звук заглушён), Open in bench (конечное состояние),
  запись = кольцо последних 30 с с началом на Start/Restart, транспорт Stop (S) / Pause CA /
  Restart без кнопки Start (стенд открывается на паузе). **S5 (точное продолжение
  снепшота + ответвления, ТЗ `req-demo-lab-s5.md`) сдан и ПРИНЯТ Пользователем
  2026-09-12** (`memory/log/2026-09-12-demo-lab-s5.md`): Save пишет полный снепшот
  исполнителя на границе среза (`state_end.json/.npz`); в каталоге **Continue**
  (продолжение со следующего блока, байт-в-байт с непрерывным расчётом для 5 движков),
  **Open field anew** (бывший Open in bench), **Check reproducibility** (бывший Replay
  from start), ссылка **Derived from** на родителя; ветка = своя копия исходного
  снепшота + журнал + 3 WAV + конечный снепшот + `parent_record_id`; формат записи 2
  (S4 = формат 1 читается без миграции). Попутно закрыта дыра S4-реплея: кроссфейд,
  взведённый на остановленной сцене, больше не ждёт Start. Ручная приёмка — 3 шага из
  ТЗ («Исходная S5» harm B=0.25 → Continue → «Вариант S5» harm B=0.75 → по ссылке
  снова Continue исходной). Открытый UX-вопрос: нужна ли кнопка Open field anew для обычных
  записей (единственный её сценарий — холодный старт с найденного поля; решить после 2–3
  реальных демо, предложение — показывать только там, где Continue недоступен).
  Каталог автотестов: `tests/README.md`. **S6 (версии кода: закреплённые/локальные записи,
  запуск исходной версии, ТЗ `req-demo-lab-s6.md`) сдан и ПРИНЯТ Пользователем 2026-09-13**
  (`memory/log/2026-09-13-demo-lab-s6.md`): формат записи 3; происхождение = явный
  runtime-набор + отпечаток + git-соответствие (CRLF-безопасно) + окружение, захват один
  раз на процесс; pin-ref `refs/casynth/pins/<commit>`; «Pin to commit» для локальных;
  «Continue in version <sha>» = дочерний стенд из кэша worktree `lab_catalog/worktrees/`
  (только ревизии ≥ S6 знают договор запуска). Ручная приёмка — 3 шага из ТЗ на
  подготовленных демо-записях. Форма Save: автоповтор клавиш + Ctrl+Backspace (638d407); поле самодельное поверх
  pygame — переход на `pygame_gui` только если форм станет больше (обсуждено 2026-09-13).
  **S7 (проверка каталога после изменения кода, ТЗ `req-demo-lab-s7.md`) сдан и ПРИНЯТ Пользователем 2026-09-13** (`memory/log/2026-09-13-demo-lab-s7.md`): «Check catalog» =
  один worker-процесс (его происхождение = цель отчёта) пересчитывает все записи тем же
  путём, что одиночная проверка (`Catalog.recompute`), по дорожкам A / B / monitor →
  exact / differs (доля сэмплов, max |d| без переполнения, длина отдельно) / failed
  (причина) / unchecked; смена runtime-набора посреди прогона останавливает его без
  смешения версий; отмена сохраняет готовое; отчёт `.verify/<run>/` переживает закрытие
  («Last report», older/newer), пары проверяются по отпечаткам, одиночная проверка пару не
  трогает; прослушивание Saved/Recomputed на одном курсоре (Space), отметка «Can't hear /
  Hear it / Not rated» на пару и дорожку, без автоматического «не слышно». Демо-каталог для приёмки
  пересобирается `python tests/s7_demo_catalog.py` (каталог `lab_catalog/` вайпнут
  Пользователем 2026-09-13 как тестовый мусор). Следующий спринт S8 (пакетный прогон
  демо) — по ТЗ.
- **Демо S/N (ТЗ Researcher `memory/req-sonification-sn-demos-2026-09-14.md`, слуховые кейсы
  `memory/req-listen-sn-demos-2026-09-14.md`) сданы технически 2026-09-14**
  (`memory/log/2026-09-14-sn-demos.md`): движки `scan_surface` / `pm_network` в `casynth_lab/`,
  15 сцен + два каталога записей (сборка `python demos/build_sn_demos.py`), ярлыки
  `run_scan_demo.bat` / `run_network_demo.bat`. **Ждёт: ревью Researcher на соответствие ТЗ
  (вопрос в `questions.md` 2026-09-14 — пять инженерных выборов) и слуховую приёмку
  Пользователя по трём шагам.** Инженерные итоги: 4× vs 8× у Network −109 dB; удвоенная
  плотность у Scan < −50 dB; Pulsar/Kok's galaxy двигают W лишь на 0.017/0.025 (шкала
  0..1/3) — основной контраст N остаётся F5. Эллипс/Лиссажу радиуса 0.8 обходят
  центрированный Pulsar (F3 подготовлен только с Raster).

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
