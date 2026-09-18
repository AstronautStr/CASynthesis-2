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
| `casynth_lab/scan_surface.py` + `casynth_lab/pm_network.py` | **Демо S/N (2026-09-14, ТЗ `memory/req-sonification-sn-demos-2026-09-14.md`)**: движки стенда `scan_surface` (Scan: поверхность Smooth/Distance + путь Ellipse/Lissajous/Raster, band-limited аддитивный рендер с непрерывной фазой, 20-мс переход коэффициентов) и `pm_network` (Network: 4 генератора 1..4·f0, 6 направленных PM-связей из шести гауссовых масок поля, 4× рендер + FIR-децматор + HPF 5 Гц, сглаживание W/beta 30 мс, gate 20 мс, frozen links); оба с полным snapshot (Continue байт-в-байт). Подсказки реестра `choices/inactive/overlay` + `display()` движка → панель со словесными режимами, `Full field`, оверлей пути/масок, полоски W (`demo_bench.py`, геометрия панели от числа движков, прокрутка каталога колёсиком). Материал: `demos/build_sn_demos.py` (поля F1–F5 + Kok, 15 сцен `demos/sn_*.json`, каталог `lab_catalog/sn_demos_2026_09_14/{scan,network}/`), входы `run_scan_demo.bat` / `run_network_demo.bat`, тесты `tests/test_demo_lab_sn.py` (15). Уровни: сырой запас ×8, trim по умолчанию Scan +6 / Network +3 dB (RMS = Laplace на F3). Отчёт `memory/log/2026-09-14-sn-demos.md`. **Запуск стенда (2026-09-14): `demo_bench.py --demo <scene> [--catalog ROOT]` открывается ЭКРАНОМ КАТАЛОГА (сцена за ним на паузе, Esc/Back → к ней; двойной клик по записи = Continue); `--live` открывает сразу звучащее поле — использовать в автотестах/скриптах, которым нужен живой экран; `--render`/`--record` не затронуты. **Notes (2026-09-14):** кнопка Notes в живом окне и в каталоге — свободный текст записи (`<record>/notes.md`, сохраняется при вводе; принадлежит продолженной/открытой записи или последней сохранённой в сессии); **с 2026-09-17 — та же форма в отдельном окне ОС** (`casynth_lab/notes_window.py`: второе `pygame.Window`, отрисовка и `TextEdit` стенда без изменений; главный цикл маршрутизирует события по `ev.window` → `BenchApp.notes_event`, кадр — `draw_notes`, привязка к записи — `sync_notes`): заголовок/свернуть/закрыть/фокус от Windows, поле стенда не блокируется; окно принадлежит одной записи и закрывается при смене адресата заметок / Back / Continue-Save другой / закрытии ОС / выходе из стенда; режима `notes` у стенда больше нет; описание при Save теперь называется **Description** (ключ `note` в record.json не менялся); **фидбек Пользователя вытаскивать: `python -m casynth_lab.catalog notes lab_catalog/<root>`** (Markdown; `--all` — плюс описания Save). **Отпечаток происхождения (2026-09-14) = только ЗВУКОВОЙ набор** (`casynth_core/engine/config`, `casynth_lab/{runner,engine_api,registry,legacy_engine,scene,snapshot,движки}.py`); правки `demo_bench.py`, `audio_out/catalog/recorder/verify/versions/provenance`, `requirements.txt`, `demos/*.json` записи НЕ перепривязывают; старые записи (set 1) сравниваются по содержимому на их коммите. Диагностика: `python -m casynth_lab.provenance [why ROOT]`** **2026-09-16 (лог `memory/log/2026-09-16-bench-text-fields-clear-library.md`):** кнопка **Clear (C)** (команда раннера `clear`) и команда `set_cells` (пакетная правка поля одной записью журнала); **все текстовые поля** (Save Title/Note, Notes) — одна модель `casynth_lab/textedit.py` (`TextEdit`: каретка кликом/стрелками, выделение Shift/drag, Ctrl+A/X/C/V через `pygame.scrap`, Home/End/Up/Down, перенос по ширине, вид следует за кареткой; в стенде единый маршрут `_edit_key`/`_caret_from_pos`/`_draw_edit` — новое поведение добавляется один раз для всех полей); **колонка Patterns** (`patterns.py`, 29 форм) справа от панели — drag-n-drop формы на поле с призраком и торовым переносом, Esc/отпускание вне поля — отмена, колесо прокручивает колонку. |
| `demos/network_reference_n0/` + `demos/results/network_reference_n0/` | **N0 — референс Gutter Synthesis (2026-09-15, ТЗ `memory/req-network-reference-n0-2026-09-15.md`)**: `gutter_node.py` — скалярный порт исполняемого `gutterOsc.class` (НЕ `gutterOsc.java`), побитово сверен с оригиналом на JVM (`verify_node.py` + `javaref/`, JDK не в репо: `--jdk`, `--source <clone>`); `jmath.py` — fdlibm atan / Java-семантика f32, div, sin, exp; `gutter_network.py` — 8 узлов + Max-цепочка (`matrix~`→`delay~` 2000+64→×interaction→демпфирование `c`; `clip~`/`svf~`/`tanh~`/pan), слайдеры источника `set_slider(name, raw 0..256)`, `set_matrix`, `export_state/from_state`, ~52 мкс/отсчёт; `source_manifest.py` — commit/хеши/таблица параметров/банки `filters.txt`; `analysis.py` — метрики; `render_n0.py` — один запуск строит пакет (пробы 3 оси × 5 параллельно, `01_palette`/`02_control`/`03_links`/`03b_links_strong`, `manifest.json`, `figures/`, `raw/` gitignored). Отчёт `memory/log/2026-09-15-network-reference-n0.md`; допущения Max-обвязки — `questions.md` (OPEN). Не подключён к стенду/КА (это N1). |
| `casynth_lab/gutter_field.py` + `casynth_lab/gutter_field_n1_config.py` | **N1 — сеть Gutter, управляемая полем (2026-09-15, ТЗ `memory/req-network-ca-n1-2026-09-15.md`)**: движок стенда `gutter_field` («Gutter»): восемь узлов N0 под ФИКСИРОВАННОЙ конфигурацией из fixtures Researcher (явные 8×24 частоты, задержка 2064, SVF q=0.99, маршруты R1: у узла 6 нет правого выхода ни в мастер, ни в матрицу; `post_math=scalar`, `model_version=gutter_field_n1_v1`); numba-ядро `_render` — тот же порядок операций, что у скалярного порта узла и `GutterNetwork` в скалярном режиме (≈0.4 мкс/отсчёт/сторона, блок обеих сторон ≈0.5 мс при бюджете 7.98 мс; без numba — чистый Python, верно, но не real-time; `numba` в `requirements.txt`). Поле → частоты: 2×4 областей (`i=4r+c`), `u=n/(n+2)`, `ratio=2^clip(log2 scale + depth(2u−1), −1, 1)`, `f=min(19000, base·ratio)` как float32-сообщения setFreqN атомарно на границе блока, состояния сохраняются, накопления нет. Ручки `scale` (Resonators 0.5..2), `depth` (CA amount), `interaction` (Links raw 0..256, рампа 50 мс), `freeze_ca` (держит достигнутый вектор u; init — начальное поле; off — текущее поле). Оверлей: границы областей + номера узлов (`lines` — новый ключ оверлея стенда); панель: u удержанный (бар) / u поля (тик), счётчики, ratio, links, resets. Уровень: сырая сумма ×20 → gain стенда (0.56 при дефолтах). Снапшот полный (Continue байт-в-байт). **Актуальные опыты 2026-09-16:** `demos/build_n1_hypotheses.py` собирает три опыта с гипотезами в Notes в `lab_catalog/network_n1_hypotheses_2026_09_16_r2/`; `run_network_n1.bat` открывает этот каталог. `casynth_lab/gutter_controls.py` регистрирует дополнительные каналы поля для демпфирования и связей, сохраняя управление резонаторами. Первоначальные пять записей N1 сохранены как история с фидбеком; старый сборщик перенаправлен на новый набор. Измерения `demos/gutter_field_n1_report.py` → `demos/results/network_n1/report.{json,md}`. Модель N0 получила явные ключи `output_routes` / `post_math` (без них — прежнее поведение N0, пакет N0 воспроизводим). Тесты `tests/test_gutter_field_n1.py` (10, гейт 1i). Отчёт `memory/log/2026-09-15-network-n1.md`. |
| `casynth_lab/periodic_readout.py` + `casynth_lab/gutter_field_periodic.py` + `casynth_lab/event_network.py` | **N2 — события поля возбуждают сеть задержек (2026-09-16, ТЗ `memory/req-network-events-n2-2026-09-16.md`, лог `memory/log/2026-09-16-network-n2-events.md`)**: `periodic_readout.py` — общие для обеих сторон периодические веса `K_i` (8 перекрывающихся raised-cosine, сумма 1, непрерывны на торе; центры узлов, оверлей без границ). **А `gutter_field_periodic` («Gutter K»)** — модель/источник/маршруты/ручки/снапшот N1 без изменений, только `n_i = Σ K_i·G` вместо жёстких областей и фиксированный trim −16.5 dB после ×20 (`model_version=gutter_field_periodic_n2_v1`). **Б `ca_event_network` («Events»)** — новая модель (не порт Gutter): на границе блока `E=(G≠G_prev)`, `a_i=e_i/(e_i+2)` в пару импульсных состояний (0.25/2 мс, сила 0.75), 8 линий задержки D=149…1361, ортогональная матрица `0.25−δ`, фильтр потерь 6 кГц, tanh, `rho` («Response» 0.60…0.95, рампа 20 мс на семпл-пути), выход с фильтров cos/sin-панорама ×96 → gain стенда (косинусная рампа как в N1); прямого пути импульса нет, без событий выход ровно 0. numba-ядро (p99 обеих сторон ≈0.3–0.7 мс при бюджете 7.98 мс), float64; только SR 44100/блок 352/стерео (иначе ValueError). Снапшот полный (задержки+позиции, фильтры, импульсы, `grid_prev`/`grid_pending`, рампа, `gain_prev`, версии); restore не добавляет пакет и требует совпадения ожидающего поля со стендовым. Панель стенда: бары пакета `a_i`, тик уровня `max|l_i|`, rho и рампа; оверлей — центры узлов + кольцо полувеса. `display()` А отдаёт дробные `counts`. Реестр: 13 движков → 5 рядов кнопок. Материал: `demos/build_n2_events.py` (3 сцены `demos/n2_{rhythm,travel,growth}.json`: pulsar 6 gen/s, glider 16 gen/s, R-pentomino 6 gen/s; 12 с; гипотезы ТЗ в Notes) → `lab_catalog/network_n2_events_2026_09_16/`; `run_network_n2.bat` открывает каталог. Измерения `demos/n2_events_report.py` → `demos/results/network_n2/report.{json,md}` (уровни/пики/клип/resets, хвост, продолжение, поздняя правка, перенос события между узлами, лимиты при gain 0.04, тайминг, проверка каталога). Тесты `tests/test_n2_events.py` (19, гейт 1j): ядро Б == независимый скалярный референс побитово. |
| `casynth_lab/tuned_events.py` | **N3 — поле настраивает резонансы, события их возбуждают (2026-09-16, ТЗ `memory/req-network-combined-n3-2026-09-16.md`, лог `memory/log/2026-09-16-network-n3-tuned-events.md`)**: один новый движок `ca_tuned_events` («Tuned», `model_version=ca_tuned_events_n3_v1`), два режима ручки `field_tuning` (Fixed / Field). Сетевая часть = семпловая математика N2 Б без изменений при `rho=0.88` фиксированном (состояния сети побитово равны `ca_event_network`); значение фильтра потерь каждого узла ведёт банк из 24 затухающих комплексных резонаторов (`z'=r·e^{iθ}z+l_i`, `b_i=mean Re z'`, `r=10^(−3/(SR·T60))`) на точных частотах N1 `CONFIG['filters_hz']` × `ratio_i=2^(2u_i−1)` (закон А при scale 1/depth 1, float32-сообщения на границе блока, состояния при перестройке сохраняются; в Fixed множители 1). Выход: панорама N2 → HP 20 Гц на канал → ×4 → gain стенда. Ручки `field_tuning` (перестройка без reset и без события) и `decay_s` («Decay» 0.20…1.50, дефолт 0.80, рампа r 882 отсчёта). Только SR 44100/блок 352/стерео. Снапшот полный (сеть, Re/Im, `ffreq` сверяется с законом поля при restore, r+рампа, HP, гриды, gain, версии). Панель: бары `a_i`, тик `max|b_i|`/5, множитель, режим, decay. Реестр: 14 движков → 5 рядов. Материал: `demos/build_n3_combined.py` → `demos/n3_{cycles,travel,growth}.json` + `lab_catalog/network_n3_combined_2026_09_16/` (3 записи; N3.1 = Octagon II 12 с, затем одна команда `set_cells` всех клеток на 12 с → Tumbler, ещё 12 с; N3.2 glider 16 gen/s; N3.3 R-pentomino 6 gen/s; гипотезы ТЗ в Notes); `run_network_n3.bat`. Измерения `demos/n3_combined_report.py` → `demos/results/network_n3/report.{json,md}` (итог: сцены и стресс раздельно + ограничения). Тесты `tests/test_n3_tuned_events.py` (16, гейт 1k): вся цепочка против скалярного референса ≤ 1e-12 в обоих режимах. |
| `casynth_lab/figures.py` + `casynth_lab/object_resonators.py` | **N4 — резонаторы фигур и круговые детекторы (2026-09-16, ТЗ `memory/req-object-resonators-n4-2026-09-16.md`, лог `memory/log/2026-09-16-object-resonators-n4.md`)**: движок `ca_object_resonators` («Objects», `model_version=ca_object_resonators_n4_v2` с 2026-09-17 — см. строку «Objects / Laplace» ниже; v1 — исходная поставка). `figures.py` — компоненты 8-связности через шов тора, периодический центр (точный минимизатор Σd² по оси, детерминированные тай-брейки), R, маски Disk/Own, спектр `L=D−A` полного графа по модулю тора из канонического размещения (побитово одинаков для любого положения формы) с кэшем, сопоставление по графу перекрытий (распад/слияние → хвосты + новые id; без перекрытия — смещение центра ≤ 1.5). Движок: на границе блока при изменении поля `e = Σ births·K_cur + Σ deaths·K_prev` (новая фигура — births·K_cur), `a=e/(e+2)` в импульс N2 → банк до 24 резонаторов на `f=scale·√λ` (мода j→j, новые с нуля, исчезнувшие дозвучивают, веса 1/n рампа 20 мс), панорама cos/sin по центру (рампа 20 мс), HP 20 Гц, ×0.5, gain стенда (рампа 20 мс на семпл-пути); слоты 24 активных + 96 хвостов + 24 затухающих, тихие хвосты освобождаются, при переполнении самые тихие затухают за 20 мс, hard drops считаются и видны; фигура без слота отслеживается («звучат X из Y»). Ручки `detector` Own/Disk (default Disk), **`radius_mul` «Radius x» 0.25…4.0 (default 1.0, только Disk: `R_eff = mul·R`, не удар; добавлено по запросу Пользователя 2026-09-16)**, `frequency_scale` 55…880 (default 220), `decay_s` 0.20…1.50 (default 0.80). Только SR 44100/блок 352/стерео. Снапшот полный (слоты + трекер + поля + рампы), Continue байт-в-байт. Стенд: динамический оверлей фигур прослушиваемой стороны из `display()` (клетки/круг через шов/центр в цвете id; Own — маска с обводкой + пунктирный круг), панель фигур; реестр 15 движков → 5 рядов. Материал: `demos/build_n4_objects.py` → `demos/n4_spectrum.json` (А = N3 Tuned Field, Б = N4 Disk), `demos/n4_neighbor.json` (А = N4 Own, Б = N4 Disk), клетки из preflight, 6 gen/s, 12 с → `lab_catalog/object_resonators_n4_2026_09_16/`; `run_object_resonators_n4.bat`. Отчёт `demos/n4_objects_report.py` → `demos/results/object_resonators_n4/report.{json,md}`. Тесты `tests/test_n4_object_resonators.py` (27, гейт 1l): семпловый путь против скалярного референса ≤ 1e−12 через все рампы. |
| `demos/build_objects_radius_attack.py` + `demos/objects_radius_attack_report.py` | **Radius range / Attack (2026-09-17, ТЗ `memory/req-objects-radius-attack-2026-09-17.md`, лог `memory/log/2026-09-17-objects-radius-attack.md`)**: реестр `EngineSpec.ranges` + `validate_range/default_range` (`radius_mul` spec `0..inf`, диапазон слайдера по умолчанию 0.25..4); раннер `set_range` (журнал, per (сторона, движок), `export_state['ranges']`, `snapshot()['ranges']`, `range_of/side_ranges/param_ranges`); сцена `param_ranges`; `figures.full_cover_radius/covers_field`; движок v3 `attack_ms` (`attack_q`, `zu`, `qq`, `I_Q_LEFT`, `COMPATIBLE_STATES` v2→Attack 0, display `covers_all/attack_ms/q/attack_ramp_left`); стенд: строка «range» с полями Min/Max (`_range_rects`, `_focus_range/commit_range/cancel_range`, `_range_of` с pending), `RANGE_FIELD_W/RANGE_DECIMALS`, `ROW_H 24`, рамка полного покрытия, `_params_height`. Сцены `demos/ora_{r1,r2,a1,user034}.json`, каталог `lab_catalog/objects_radius_attack_2026_09_17/`, `run_objects_radius_attack.bat`, отчёт `demos/results/objects_radius_attack/`. Тесты `tests/test_objects_radius_attack.py` (19, гейт 1n). |
| `demos/build_objects_event_source.py` + `demos/objects_event_source_report.py` | **Events / Birth position (2026-09-17, ТЗ `memory/req-objects-event-source-modal-2026-09-17.md`, лог `memory/log/2026-09-17-objects-event-source-modal.md`)**: движок v4 (`ca_object_resonators_n4_v4`, `STATE_VERSION 4`; v3/v2 снимки = Both/Uniform побитово): ручка **Events** Both / Births / Deaths (источник новых пакетов; Both = прежний путь побитово; Deaths: новый банк без пакета, исчезнувший/split/merge банк уносит в хвост ОДИН последний удар смертей прежней маски — `npulse` хвоста кормит прежние моды до `TAIL_FLOOR`), ручка **Excitation** Uniform / Birth position (распределение пакета между модами: `b_j = sqrt(m·p_j/Σp)`, `p_j` = квадраты участия рождённых клеток, усреднённые по кратной группе `1e-8`; `Σb² = m`; per-mode импульсные состояния `zfm/zsm/zum`, Attack на каждой моде; определён для Births+Own+Laplace+full, иначе пакетов нет + счётчик `unsupported_packets` и красная строка стенда); `casynth_core.laplacian_modes(return_index=True)` отдаёт индексы выбранных мод; `laplace_modes_of(with_graph=True)`, `birth_position_weights`, `degenerate_groups`; `display()`: `events_name/excitation_name/position_supported/n_tails_fed/zero_participation/unsupported_packets`, у фигуры `b`. Стенд: `ROW_H 20`, кнопки выбора 92 px при длинных названиях, третья строка шапки Objects (`FIGURE_HEAD_H 48`). Сцены `demos/oes_{e1,m1}.json` (обе стороны — настройки preflight: Own, Laplace, full, part 3, spread 1, harm 0.87, Decay 1.39, Attack 4; E1 А Births / Б Deaths, M1 А Uniform / Б Birth position; side_gain Б 0.93), каталог `lab_catalog/objects_event_source_modal_2026_09_17/`, `run_objects_event_source_modal.bat`, отчёт `demos/results/objects_event_source/`. Тесты `tests/test_objects_event_source.py` (13, гейт 1o). |
| `demos/build_objects_birth_strength.py` + `demos/objects_birth_strength_report.py` | **Birth strength — сила влияния места рождения (2026-09-18, ТЗ `memory/req-objects-event-source-modal-2026-09-17.md` раздел 5, лог `memory/log/2026-09-18-objects-birth-strength.md`)**: движок v5 (`ca_object_resonators_n4_v5`, `STATE_VERSION 5`, массивы те же; v4-снимок без ключа = s 1 побитово): ручка **Birth strength** `birth_strength` 0…4 (float, default 1, отсутствие ключа = 1, валидируется в реестре / `set_params` / `restore_state`) сразу после Excitation; закон только на СЛЕДУЮЩИЕ пакеты Birth position: s=0 → прежний Uniform-тракт (`_strike(s, e)` без пространственного участия, условие Births/Own/Laplace/full по-прежнему обязательно), 0<s<1 → `v=(1−s)+s·b`, 1<s≤4 → `v=b^s`, `c=√m·v/‖v‖` (`birth_strength_profile`, при s=1 возвращает сам b без арифметики); ноль участия при s>0 — пакета нет; смена s — без пакета, состояния обоих трактов не трогаются. `inactive` при Uniform: `"1.00  Birth position only"`; `display`: `birth_strength`, `birth_strength_hint`. Стенд: `ROW_H 18` (`CHOICE_H 16`, `RANGE_FIELD_H 18`; 17 строк панели помещаются в 880 px), четвёртая строка шапки Objects (`FIGURE_HEAD_H 62`, `BIRTH_STRENGTH_LINE` «0 Uniform, 1 original, >1 stronger selection» / «Birth strength 1.00 stored (Birth position only)»), неактивный текст ручки с отступом 6 px. Все Objects-сцены (`n4_*`, `ol_*`, `ora_*`, `oes_*`) несут `birth_strength: 1.0`. Сцена `demos/obs_m2.json` (M2: поле M1, обе стороны Births + Birth position с настройками M1; А s=1 / Б s=4; side_gain Б 0.92), каталог `lab_catalog/objects_birth_strength_2026_09_18/`, `run_objects_birth_strength.bat`, отчёт `demos/results/objects_birth_strength/`. Тесты `tests/test_objects_birth_strength.py` (8, гейт 1p). |
| `demos/build_objects_laplace.py` + `demos/objects_laplace_report.py` | **Objects / Laplace — старый Laplace против Objects с его настройками (2026-09-17, ТЗ `memory/req-objects-laplace-comparison-2026-09-17.md`, лог `memory/log/2026-09-17-objects-laplace.md`)**: в `ca_object_resonators` (v2, `STATE_VERSION 2`) ручка **`spectrum` Figure / Laplace** (дефолт реестра Laplace; отсутствие ключа в параметрах/снимке = Figure) и семь настроек старого Laplace из реестра ядра (`n` part / spread / alpha / shape / harm / fullshape full / dyn — те же диапазоны/дефолты/названия). Laplace-закон = `casynth_core.laplacian_modes` (спектральная часть `map_laplacian`, вынесена 2026-09-17; старый путь побайтно прежний) на полном торовом графе собственной компоненты (full=1, без децимации) или `map_laplacian` на окне 8×8 (full=0); нижняя мода = `ctx.f0`; `dyn` берёт штатный `exc` стенда в клетках фигуры в порядке узлов графа (веса, не удар); веса = `LAPLACE_GAIN 0.7` × амплитуды, банк `Σ w_j Re z_j`; Freq scale неактивен в Laplace, семёрка — в Figure, dyn — при shape=0. Смена закона/настроек/нового `exc` на том же поле (когда dyn действует) перестраивает все звучащие фигуры сразу (состояния сохранены, веса 20 мс) без удара. На компактной компоненте без шва частоты и веса/0.7 побитово равны `map_laplacian` на bbox (73 состояния трёх сцен). **Правила хвостов v2 (три P2 ревью):** исчезнувшие моды уходят в отдельный хвост (возвращающиеся с нуля, старый хвост без удара); при занятых пулах — затухание на месте 20 мс (`in place`), обрыв только при возврате фигуры в эти 20 мс (`dropped`). Стенд: команда/кнопки **copy_spectrum** («<< spectrum B to A» / «spectrum A to B >>», только общие настройки, `registry.spectrum_keys`), `audio.side_gain` сцены (постоянный множитель стороны до клипа, default 1 = старые записи побайтно, в снимке/Continue), панель растёт под 12 настроек + строки фигур. Опыты: `demos/ol_{glider,galaxy,neighbor}.json` → `lab_catalog/objects_laplace_2026_09_17/` (А = `laplacian` с таблицей ТЗ, Б = Objects/Laplace те же семь, Disk, Decay 0.8, Radius x 1/1/1.5; side_gain Б 1.08/0.74/1.25 → А−Б ≤ 0.03 dB); вход `run_objects_laplace.bat`. Тесты `tests/test_objects_laplace.py` (13, гейт 1m). Отчёт `demos/results/objects_laplace/report.{json,md}`; отчёт N4 перегенерирован (пик/клип пробы хвостов в сводке). |
| `demos/build_objects_decay.py` + `demos/objects_decay_report.py` | **Decay law — затухание от истории клеток, D1–D3 (2026-09-18, ТЗ `memory/req-objects-decay-2026-09-18.md`, лог `memory/log/2026-09-18-objects-decay.md`)**: движок v6 (`ca_object_resonators_n4_v6`, `STATE_VERSION 6`; снимок v5 = Fixed, u = 1 на живых клетках, тот же звук): ручка **Decay law** `decay_law` Fixed / Common age / Modal age (кнопки Fixed / Common / Modal, `LAW_NAMES`; 0/1/2, default 0) сразу после Decay. Fixed = прежний тракт побайтно. Свежесть каждой позиции `age` (рождение 1, смерть 0, выжившие хранят, `×exp(−B/(SR·0.5))` после каждого блока, и при паузе КА; принятый переход = та же разность G_prev→G, что у возбуждения). Common/Modal (только Laplace + full; сочетание отвергают реестр — новый хук `EngineSpec.validate` / `registry.validate_params`, раннер при постановке с учётом очереди, сцена, снимок, конструктор; стенд блокирует обе стороны текстом): `P` = квадраты собственных векторов, усреднённые по кратной группе (кэш на фигуру, пересчёт только при смене геометрии/набора мод), `q = P·u`, `T = Decay − (Decay − 0.08)·q`, `gamma = ln1000/T`; Common = средняя СКОРОСТЬ. Ядро: у слота `slaw = 1` на каждый сэмпл `ga ← k·ga + (1−k)·gt` (20 мс) и `r_j = exp(−ga/SR)` вместо общего r; новые слоты/моды стартуют с цели; хвосты уносят `gam_a/gam_t` и держат цель; переход Fixed→адаптивный: активные банки с текущего r, хвосты эпохи Fixed держат Decay момента переключения; обратно — сходимость к глобальному r (`_settle_fixed`). `display`: закон, `t_lo/t_hi/t_applied` фигур; стенд: строка шапки подбирается по ширине (`fit_text`), четвёртая строка при адаптивном законе (`decay_law_line`), строка фигуры `T lo-hi s` (`figure_decay_text`), `ROW_H 17`, `RANGE_FIELD_H 17`, зона клика слайдера ±4 px. **Ключ сцены `script`** (команды `pause` по часам сцены): раннер ставит их при каждом старте сцены с начала (Start, отпускание паузы остановленной сцены, Restart), в журнале с пометкой `script`, реплей каталога их пропускает (`catalog._replayed`). Сцены `demos/od_d{1,2,3}.json`, каталог `lab_catalog/objects_decay_2026_09_18/`, `run_objects_decay.bat`, отчёт `demos/results/objects_decay/`. Все Objects-сцены несут `decay_law: 0`. Тесты `tests/test_objects_decay.py` (23, гейт 1q). |
| `check.py` | **Все гейты одной командой** (см. «Гейты») |
| `tests/` | `test_casynth_core.py` (59 тестов) + `test_demo_lab.py` (39 тестов S1–S3) + `test_demo_lab_s4.py` (9 тестов S4) + `test_demo_lab_s5.py` (6 тестов S5) + `test_demo_lab_s6.py` (6 тестов S6, изолированные git-репо в `artifacts/_s6/`) + `test_demo_lab_s7.py` (6 тестов S7, `artifacts/_s7/`; демо-каталог `s7_demo_catalog.py`) + `test_demo_lab_sn.py` (16 тестов демо S/N, `artifacts/_sn/`) + `test_n2_events.py` (19 тестов N2, `artifacts/_n2_tests/`) + `test_n3_tuned_events.py` (16 тестов N3, `artifacts/_n3_tests/`) + `test_n4_object_resonators.py` (28 тестов N4, `artifacts/_n4_tests/`) + `test_objects_laplace.py` (13 тестов Objects/Laplace, `artifacts/_ol_tests/`) + `test_objects_radius_attack.py` (19 тестов Radius range / Attack, `artifacts/_ora_tests/`) + `test_objects_event_source.py` (13 тестов Events / Birth position, `artifacts/_oes_tests/`) + `test_objects_birth_strength.py` (8 тестов Birth strength, `artifacts/_obs_tests/`) + `golden/` (эталоны golden-master и UI) |

## Гейты (обязательны после каждого изменения)

`python check.py` = 59 юнит-тестов + 42 + 9 + 6 + 6 + 6 + 15 тестов demo-стенда S1–S7 и демо S/N + 9 тестов N0 (гейт 1h) + 10 тестов N1 (гейт 1i) + 3 теста гипотез N1 + 19 тестов N2 (гейт 1j) + 16 тестов N3 (гейт 1k) + 28 тестов N4 (гейт 1l) + 13 тестов Objects/Laplace (гейт 1m) + 19 тестов Objects radius/attack (гейт 1n) + 13 тестов Objects events/modal (гейт 1o) + 8 тестов Objects birth strength (гейт 1p) + 23 теста Objects decay law (гейт 1q) + golden-master аудио (байт-в-байт,
sha256[:16]=cd3126907ad13b6c) + UI-кадр (пиксель-в-пиксель vs `tests/golden/ui_frame.png`)
+ import/init smoke. Эталоны ВЕРСИОНИРУЮТСЯ в `tests/golden/` (переехали из gitignored
`artifacts/` 2026-07-14). Легитимное изменение UI → `python check.py --bless-ui` в том же
изменении. Конвенция каждой новой ручки: **дефолт = бит-в-бит**.

## Ядро и движки (casynth_core)

- 5 движков в реестре `ENGINES`: FFT / Walsh / Random / **Laplacian** (активный) / Granulo.
- `map_laplacian(patch, f0, n, spread, alpha, shape, harm, fullshape, dyn, exc)`:
  граф-Лапласиан живых клеток, √λ → частоты, низшая взятая мода всегда = f0.
  С 2026-09-17 строит L и вызывает `laplacian_modes(L, f0, n, spread, alpha, shape, harm, dyn, e_ev)`
  (спектральная часть на готовой матрице; старый путь побайтно прежний, golden master цел) —
  её же использует Objects/Laplace стенда на полном торовом графе фигуры.
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

- **N0/N1 (2026-09-15): ревью Researcher завершено** —
  `memory/research/audit-network-n0-2026-09-15.md`. Узел и задержки проверены;
  найден лишний правый маршрут узла 6, точность Max-обвязки не установлена.
  Готово ТЗ `memory/req-network-ca-n1-2026-09-15.md`: исправление маршрута,
  КА → частоты резонаторов, живая демка. **N1 реализован 2026-09-15**
  (`memory/log/2026-09-15-network-n1.md`): движок `gutter_field`, каталог из 5 записей
  (`demos/build_n1_demos.py`), `run_network_n1.bat` открывает каталог, 10 тестов (гейт 1i),
  отчёт измерений `demos/results/network_n1/report.md`. **Исправлено 2026-09-16:** прежние
  пользовательские сравнения отозваны. Три новых опыта, гипотезы в Notes; все стороны
  управляются полем. `gutter_controls.py` добавляет поле → демпфирование / исходящие связи
  к исходному поле → резонаторы. `run_network_n1.bat` открывает новый каталог;
  заметки: `python -m casynth_lab.catalog notes lab_catalog/network_n1_hypotheses_2026_09_16_r2`.
  Модель, реакции, точное продолжение проверены. **Фидбек получен:** в N1.1 характеры
  различимы; в N1.2 A/Б очень похожи и слышны всплески на краях; в N1.3 доминирует
  общая «сирена», гипотеза о памяти не подтверждена. Метод не принят.
  Описание: `memory/research/network-n1-hypotheses-2026-09-16.md`.

- **N2 (2026-09-16): реализация проверена, слуховой фидбек получен** —
  ТЗ `memory/req-network-events-n2-2026-09-16.md`; лог `memory/log/2026-09-16-network-n2-events.md`.
  А `gutter_field_periodic` (Gutter с периодическим чтением поля + trim −16.5 dB);
  Б `ca_event_network` (события КА → затухающая сеть задержек, ручка Response). Обе через
  общие веса `periodic_readout.py`. Каталог `lab_catalog/network_n2_events_2026_09_16/`
  (3 записи, гипотезы в Notes), вход `run_network_n2.bat`; заметки:
  `python -m casynth_lab.catalog notes lab_catalog/network_n2_events_2026_09_16`.
  Ревью: `memory/research/network-n2-review-2026-09-16.md`; полная регрессия прошла,
  все три записи и Continue точны. Б: окраска от положения слышна, ожидаемое разнообразие
  ритма/фактур не получено — однообразный стук в такт поколений. А: пользователь отметил
  разные высотные рисунки фигур и интересное движение глайдера. Продолжение — N3 ниже.
  Ограничение А при полном переключении поля каждый блок и P2 к итогу отчёта — `questions.md`.

- **Objects / Laplace (2026-09-17): реализовано, получен положительный слуховой фидбек** — ТЗ
  `memory/req-objects-laplace-comparison-2026-09-17.md`; лог `memory/log/2026-09-17-objects-laplace.md`.
  В `ca_object_resonators` (v2) — `Spectrum: Figure / Laplace` и семь настроек старого Laplace (из реестра
  ядра; `casynth_core.laplacian_modes` на полном торовом графе фигуры); на компактной компоненте без шва
  частоты/веса побитово равны `map_laplacian` (73 состояния × 3 сцены). Три P2 ревью N4 закрыты
  (исчезнувшие моды — в отдельный хвост, возврат с нуля; при занятых пулах затухание на месте вместо обрыва;
  пик/клип пробы хвостов в сводке отчёта N4). Стенд: `copy_spectrum` (кнопки «<< spectrum B to A» /
  «spectrum A to B >>»), `audio.side_gain` сцены (default 1 — старые записи побайтно). Каталог
  `lab_catalog/objects_laplace_2026_09_17/` (L1 glider, L2 Kok's galaxy, L3 сосед+blinker Radius x 1.5;
  А = `laplacian`, Б = Objects/Laplace, те же семь настроек; side_gain Б 1.08/0.74/1.25 → А−Б ≤ 0.03 dB),
  вход `run_objects_laplace.bat`; заметки: `python -m casynth_lab.catalog notes lab_catalog/objects_laplace_2026_09_17`.
  Измерения `demos/results/objects_laplace/report.md`: семь ручек действуют на Galaxy (spread/full — фазы
  2, 4, 5; `n` — не в фазах 3/7: у 4-клеточных компонент 3 моды — расхождение с preflight, `questions.md`),
  L3 приёмник 0 / 4 на поколение при ×1 / ×1.5 и рычаг вживую, L2 без обрывов и правило 4→2→4, Continue
  точен, p99 обеих сторон 2.3 / 5.8 / 3.4 ms; стресс (gain 0.04) — клип и превышение бюджета анализом
  плотных полей (ограничения вне сцен, как в N4). **Открыто (`questions.md`, 2026-09-17):** способ
  выравнивания (LAPLACE_GAIN 0.7 + side_gain на сцену), артефакт preflight по `n`.
  Прежний N4-каталог (закон Figure), его сцены и Notes сохранены; записи N4 (v1) были Local (сборка
  при незакоммиченном Radius x) и запинены к коммиту их звуковых файлов (`6aee6a8`) — стенд открывает их
  отдельным стендом этой версии («Continue in version»), проверка версией — match; текущий код их сцену
  отвергает намеренно (значения параметров старой записи не выдумываются). **Правило 2026-09-17:** звуковой
  код коммитить ДО сборки каталога; сборщики отказывают при грязном наборе (`require_pinnable()`). Ревью N4 Researcher: `memory/research/object-resonators-n4-review-2026-09-16.md`
  (опыты соответствуют гипотезам; это прежняя сверка до нынешнего слухового результата).
  **Новый фидбек:** интересны арпеджиаторные рисунки и полиритмы, радиус <1 выбирает события;
  приём соседей пока не различён на слух, плотное возбуждение бывает шумным. Настройки скриншота
  и разбор — `memory/research/objects-feedback-radius-attack-2026-09-17.md`. Researcher независимо
  прошёл 13 целевых тестов; полного независимого ревью всей поставки этим не объявлял.
  **Radius range / Attack (2026-09-17): реализовано, каталог собран, ждёт сверки Researcher и слуха** — ТЗ
  `memory/req-objects-radius-attack-2026-09-17.md`; лог `memory/log/2026-09-17-objects-radius-attack.md`.
  Radius x без потолка (любое конечное ≥ 0), под слайдером поля Min/Max — диапазон слайдера выбранной стороны
  (реестр `EngineSpec.ranges`, команда раннера `set_range`, сцена `param_ranges`, в записи/Continue; старые =
  0.25/4; диапазон, исключающий значение, прижимает его один раз), круг ≥ `full_cover_radius` = полная маска и
  рамка в цвете фигуры. Движок v3 (`ca_object_resonators_n4_v3`, `STATE_VERSION 3`): ручка **Attack 0…20 ms**
  (однополюсное сглаживание импульса перед банком, `u` на слот, рампа `q` 20 мс; 0 = прежний путь побитово,
  снимок v2 = Attack 0; без компенсации громкости). Каталог `lab_catalog/objects_radius_attack_2026_09_17/`
  (R1 Radius 1/2 на приёмнике+blinker; R2 Radius 1/32 и A1 Attack 0/4 ms на Jam p3 + Octagon II p5; U0 пресет
  пользователя Radius 0.34; все — Objects/Laplace с настройками скриншота, side_gain Б 0.65/0.69/2.05/1.0 →
  А−Б ≤ 0.06 dB), вход `run_objects_radius_attack.bat`; заметки: `python -m casynth_lab.catalog notes
  lab_catalog/objects_radius_attack_2026_09_17`. Отчёт `demos/results/objects_radius_attack/report.md`: пакеты
  всех банков = независимые маски (0 расхождений, итоги preflight совпали), Attack-проба совпала с preflight
  до 0.2 dB, Continue точен, p99 ≤ 4.7 ms, живые сцены без underrun/клипа. Сцены прежних Objects-каталогов
  перегенерированы с `attack_ms: 0` (их записи — в своей версии). Открыто (`questions.md`): поле пресета U0,
  диапазоны при `copy_side`/`factory`.

- **Events / Birth position (2026-09-17): реализовано, прослушано пользователем 2026-09-18** (фидбек `memory/research/objects-event-source-modal-feedback-2026-09-18.md`: E1 — оба режима интересны как «два режима арпеджиатора», селектор оставить параметром синта; M1 — различие слабое; Birth position при Disk не работает — не ожидалось пользователем; независимая техническая сверка Researcher не объявлена) — ТЗ
  `memory/req-objects-event-source-modal-2026-09-17.md`; лог `memory/log/2026-09-17-objects-event-source-modal.md`.
  Objects v4: Events Both/Births/Deaths, Excitation Uniform/Birth position (см. карту файлов). Каталог
  `lab_catalog/objects_event_source_modal_2026_09_17/` (E1 Octagon II p5: Births / Deaths; M1 Jam p3: Uniform / Birth
  position; гипотезы ТЗ в Notes), вход `run_objects_event_source_modal.bat`; заметки: `python -m casynth_lab.catalog notes
  lab_catalog/objects_event_source_modal_2026_09_17`. Отчёт `demos/results/objects_event_source/report.md`: пакеты E1 =
  независимые маски (Births [8,16,0,8,8], Deaths [0,16,8,0,16]), коэффициенты M1 = preflight, стороны M1 равны по частотам/весам/
  моментам/a, Continue точен. Открыто (`questions.md`, 2026-09-17): стартовый пакет, пространственный пакет новым фигурам при
  split, хвост Deaths при занятых пулах, доступность кнопки Birth position вне сочетания.

- **Decay law / D1–D3 (2026-09-18): реализовано, каталог собран и запинен (`1fdf8d1`), ждёт сверки Researcher и слуха** —
  ТЗ `memory/req-objects-decay-2026-09-18.md`; лог `memory/log/2026-09-18-objects-decay.md`. Objects v6: Decay law Fixed /
  Common age / Modal age (см. карту файлов). Каталог `lab_catalog/objects_decay_2026_09_18/` (D1 Octagon II p5: Fixed
  0.947926 s / Common age; D2 Blinker с двумя паузами сценария: Common / Modal age; D3 Jam p3 + Octagon II p5: Fixed 1.39 s /
  Modal age; гипотезы ТЗ в Notes дословно), вход `run_objects_decay.bat`; заметки: `python -m casynth_lab.catalog notes
  lab_catalog/objects_decay_2026_09_18`. Отчёт `demos/results/objects_decay/report.md`: цели == preflight (≤ 2e−16 s), контроль D1
  в пределах 0.00 %, фазы D1 различаются и в PCM Б (0.71–1.06 s) при постоянной А (0.94 s), D2 верх − низ +46.7 dB за 0.5 s
  в обеих паузах (Common +0.04), совпадение с независимой пробой preflight до 1e−11 dB, стороны изолированы, А − Б ≤ 0.04 dB,
  Continue точен, p99 1.2–3.8 ms; большие меняющиеся поля вне бюджета (как и Fixed). Открыто (`questions.md`, 2026-09-18):
  инженерные выборы (ключ сцены `script`, хвосты эпохи Fixed, подписи кнопок, блокировки) и наблюдение по D1 (средняя T
  интервала длиннее цели на шаге).

- **Birth strength / M2 (2026-09-18): реализовано, прослушано пользователем 2026-09-18 — эффект слабый, гипотеза полезного усиления не подтвердилась (`memory/research/objects-birth-strength-feedback-2026-09-18.md`); полной независимой технической сверки Researcher нет** — ТЗ
  `memory/req-objects-event-source-modal-2026-09-17.md` раздел 5; лог `memory/log/2026-09-18-objects-birth-strength.md`.
  Objects v5: ручка Birth strength 0…4 (см. карту файлов). Каталог `lab_catalog/objects_birth_strength_2026_09_18/`
  (M2: поле M1 Jam p3, Births + Birth position обе стороны, А s=1 / Б s=4, гипотеза ТЗ 5.3 в Notes дословно; ручка
  свободно двигается в живом окне, коэффициент уровня постоянный), вход `run_objects_birth_strength.bat`; заметки:
  `python -m casynth_lab.catalog notes lab_catalog/objects_birth_strength_2026_09_18`. Отчёт
  `demos/results/objects_birth_strength/report.md`: закон = preflight на всех фазах при s 0/0.5/1/2/4, пять движков на M1
  равны по частотам/весам/моментам/a, s=0 == Uniform побитово на всей сцене, WAV записей E1/M1 от 2026-09-17 == текущий
  код побитово (s=1 = прежний Birth position), Continue точен, p99 в бюджете при всех s и при движении ручки.

- **N3 (2026-09-16): реализован, получен первый слуховой результат** —
  ТЗ `memory/req-network-combined-n3-2026-09-16.md`; лог `memory/log/2026-09-16-network-n3-tuned-events.md`.
  Движок `ca_tuned_events` (события N2 Б возбуждают 8 банков резонаторов на частотах N1, поле
  их настраивает; А = Fixed, Б = Field). Каталог `lab_catalog/network_n3_combined_2026_09_16/`
  (3 записи, гипотезы в Notes), вход `run_network_n3.bat`; заметки:
  `python -m casynth_lab.catalog notes lab_catalog/network_n3_combined_2026_09_16`.
  Б интересен пользователю, А слышится одинаковой долбёжкой; подробная заметка только
  к первому опыту (`memory/research/network-n3-feedback-2026-09-16.md`). Полное независимое
  техническое ревью N3 ещё не выполнено; следующий выбранный опыт — N4 выше.
  Измерения совпали с preflight Researcher до 0.1 dB, все записи и Continue точны.
  **Открыто:** клип в стресс-пробе «всё поле каждый блок» при Decay 1.5 с (пик 1.157 при gain
  0.04; дефолт 0.8 — без клипа) — вопрос о коэффициенте ×4 в `questions.md`.

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
- Wrap-aware сегментация и трекинг идентичности объектов — есть в `casynth_lab/figures.py` (N4, стенд); перенос в `casynth_engine.analyse` для `gol_synth` — отдельная задача.
- Профилирование законсервированных FFT/Walsh/Random/Granulo на крупных/плотных полях
  (живой селектор в gol_synth есть; формат стенда под крупные режимы — отдельная задача).
- Ветка ОБЪЕКТ, разведка 2026-06-22: двери A (граничный Фурье) / B (сеть задержек),
  ось топологии — ничего не выбрано.
- MIDI-reduction highest vs last-note (маппинг входа) — можно вернуться после рефактора
  огибающих.
