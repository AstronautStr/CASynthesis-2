# Current Implementation State

_Снапшот живого состояния (консолидирован /dream 2026-07-14; демо S/N добавлены 2026-09-14). Полная история —
`memory/archive/current-history-2026-06-16.md` + `memory/log/` (нарративы фиксов);
решённые вопросы — `memory/archive/questions-resolved.md`._

## Карта файлов

**Шов зашит 2026-09-21** (ТЗ `memory/req-seam-2026-09-21.md`, лог `memory/log/2026-09-21-seam.md`):
движки переехали из `casynth_lab/` в `casynth_engines/` — ниже по тексту старые пути
`casynth_lab/<движок>.py` означают `casynth_engines/<движок>.py` (в `casynth_lab/` на этих именах
лежат шимы, отдающие ТОТ ЖЕ объект модуля). Прототип больше не копирует звуковой код: он хостит
экземпляр `SoundEngine`.

| Файл | Что это |
|---|---|
| `gol_synth.py` | **Активный прототип**: event-loop, layout, MIDI, VCA; ХОСТИТ экземпляр `SoundEngine` из `casynth_engines` (вкладки — движки, умеющие играть ноту). **2026-09-22** (лог `memory/log/2026-09-22-instrument-panel-and-save.md`): открывается на `DEFAULT_ENGINE = laplace_unified` и на `START_SPECTRUM` (n 20, spread 1, alpha 0, shape 1, harm 1, full on, dyn 1 — накладывается на дефолты реестра у всех движков с полным лапласовым спектром; сами дефолты реестра не трогаются: они — нейтральная точка чужих гейтов и таблиц ТЗ); у каждого слайдера поля Min/Max (значение переехало внутрь трека), панель перестраивается, когда слайдер меняет набор неактивных рядов; кнопка **Save** превращает сыгранное в запись каталога; журнал сессии пишется всегда (аудио-чанки — только при `CASYNTH_RECORD=1`, `CASYNTH_NO_RECORD=1` выключает журнал), тестовый хук `CASYNTH_PANEL_LOG` |
| `casynth_engines/` | **Движки как продукт**: `engine_api` (контракт: `render(gain, t, *, gain_prev, transpose)`, `set_envelope`/`set_rate`, `EngineContext(pan=)`), `registry` (ленивая регистрация: `import casynth_engines` не тянет numba), `legacy_engine`, `figures` + 12 модулей движков |
| `casynth_host.py` | Устройство и кольцо блоков для ОБОИХ хостов: `BlockRing`, `AudioHost`, `open_output_stream` |
| `casynth_engines/render_pool.py` | **Общий пул потоков рендера (2026-09-21, лог `memory/log/2026-09-21-render-budget.md`)**: `THREADS` (половина ядер, максимум 4; `CASYNTH_RENDER_THREADS`, 1 = выключено и поток не создаётся), `pool()` (ленивый, демоны), `ranges(n, parts)`. Движку раздаётся диапазон СЕМПЛОВ блока — сумма по источникам никогда не пересекает границу воркера, поэтому звук побитово тот же при любом числе потоков (гейты `ThreadedRender`, `SlabRender`). Используют `laplace_fm` и `_BankVoices` из `laplace_carriers` |
| `casynth_panel.py` | Какой виджет у параметра (`panel_rows`) — одно решение для панели прототипа и стенда; с 2026-09-22 знает про диапазон ЛЮБОГО слайдера (`ranges`, `bound_lo/bound_hi`) и про то, как пишется его конец (`range_text`) |
| `casynth_textedit.py` | Модель текстового поля (каретка, выделение, Ctrl+A/C/V/X) — **общая для стенда и прототипа** (переехала из `casynth_lab/textedit.py` 2026-09-22, там шим); прототипу нельзя импортировать стенд |
| `casynth_lab/offline_record.py` | **Сцена → запись каталога оффлайн** (2026-09-22): `DemoRunner` + `Recorder` + `Catalog.save` — один путь для `demos/build_*` и для кнопки Save прототипа; в `SOUND_EXCLUDE` (отпечаток звука не трогает) |
| `casynth_config.py` | ВСЕ константы (геометрия/аудио/слоты/ADSR/палитра); pygame-free |
| `casynth_engine.py` | `step/analyse/events_field/SlotPool/render_chunk_laplacian/midi_to_freq` |
| `casynth_core.py` | Либа маппинга форма→(freqs,amps): 5 `map_*` + реестр `ENGINES` (чистые данные) |
| `casynth_session.py` | Session-рекордер `_dump_session`; **сессия → сцена** (`scene_from_session`, CLI `scene <ts>`) — оффлайн-сверка, неуязвимая к просадкам CPU (побайтово совпала с живым прогоном); `replay_session` знает только пять маппингов ядра и отсылает к сцене |
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
| `demos/build_objects_decay.py` + `demos/objects_decay_report.py` | **Decay law — затухание от истории клеток, D1–D3 (2026-09-18, ТЗ `memory/req-objects-decay-2026-09-18.md`, лог `memory/log/2026-09-18-objects-decay.md`)**: движок v6 (`ca_object_resonators_n4_v6`, `STATE_VERSION 6`; снимок v5 = Fixed, u = 1 на живых клетках, тот же звук): ручка **Decay law** `decay_law` Fixed / Common age / Modal age (кнопки Fixed / Common / Modal, `LAW_NAMES`; 0/1/2, default 0) сразу после Decay. Fixed = прежний тракт побайтно. Свежесть каждой позиции `age` (рождение 1, смерть 0, выжившие хранят, `×exp(−B/(SR·0.5))` после каждого блока, и при паузе КА; принятый переход = та же разность G_prev→G, что у возбуждения). Common/Modal (только Laplace + full; сочетание отвергают реестр — новый хук `EngineSpec.validate` / `registry.validate_params`, раннер при постановке с учётом очереди, сцена, снимок, конструктор; стенд блокирует обе стороны текстом): `P` = квадраты собственных векторов, усреднённые по кратной группе (кэш на фигуру, пересчёт только при смене геометрии/набора мод), `q = P·u`, `T = Decay − (Decay − 0.08)·q`, `gamma = ln1000/T`; Common = средняя СКОРОСТЬ. Ядро: у слота `slaw = 1` на каждый сэмпл `ga ← k·ga + (1−k)·gt` (20 мс) и `r_j = exp(−ga/SR)` вместо общего r; новые слоты/моды стартуют с цели; хвосты уносят `gam_a/gam_t` и держат цель; переход Fixed→адаптивный: активные банки с текущего r, хвосты эпохи Fixed держат Decay момента переключения; обратно — сходимость к глобальному r (`_settle_fixed`). `display`: закон, `t_lo/t_hi/t_applied` фигур; стенд: строка шапки подбирается по ширине (`fit_text`), четвёртая строка при адаптивном законе (`decay_law_line`), строка фигуры `T lo-hi s` (`figure_decay_text`), `ROW_H 17`, `RANGE_FIELD_H 17`, зона клика слайдера ±4 px. **Ключ сцены `script`** (команды `pause` по часам сцены): раннер ставит их при каждом старте сцены с начала (Start, отпускание паузы остановленной сцены, Restart), в журнале с пометкой `script`, реплей каталога их пропускает (`catalog._replayed`). Сцены `demos/od_d{1,2,3}.json`, каталог `lab_catalog/objects_decay_2026_09_18/`, `run_objects_decay.bat`, отчёт `demos/results/objects_decay/`. Все Objects-сцены несут `decay_law: 0`. Тесты `tests/test_objects_decay.py` (25 с D4, гейт 1q). |
| `demos/build_objects_decay_control.py` + `demos/objects_decay_control_report.py` | **Контроль затухания D4 (2026-09-18, ТЗ `memory/req-objects-decay-control-2026-09-18.md`, лог `memory/log/2026-09-18-objects-decay-control.md`)**: код движка не менялся. Сцены выводятся из встроенной сцены записи D3 (`demo_scene`, поле = `cells` preflight): условие меняет только Decay law / Decay / side gain своей стороны D3 — подобранный Fixed (А D3, Decay 0.671529 s, ×1.220467), прежний Modal (Б D3, ×1.21), прежний длинный Fixed (А D3, ×1.0); `check_against_preflight()` сверяет всё с preflight. `demos/od_d4_control.json` (D4.1: подобранный Fixed / Modal), `demos/od_d4_anchor.json` (D4.2: тот же Fixed / длинный Fixed), каталог `lab_catalog/objects_decay_control_2026_09_18/`, `run_objects_decay_control.bat`, отчёт `demos/results/objects_decay_control/` (независимое посэмпловое среднее gamma, хеши, изоляция по блокам, уровни, replay, Continue). Тесты — класс `D4ControlTests` в `tests/test_objects_decay.py` (гейт 1q, теперь 25). |
| `casynth_lab/laplace_carriers.py` + `demos/build_laplace_carriers.py` + `demos/laplace_carriers_report.py` | **Laplace carriers — фильтр несущей и банк волн (2026-09-20, ТЗ `memory/req-laplace-carriers-2026-09-20.md`, лог `memory/log/2026-09-20-laplace-carriers.md`)**: отдельный движок стенда `laplace_carriers` («Laplace waves», `STATE_VERSION 1`) — ТЕ ЖЕ частоты и веса `analyse(..., 'laplacian', семь настроек)`, два закона озвучивания. **Wave bank** (default): прежний `SlotPool` целиком (хвосты/ADSR/кроссфейды мод), заменена только волна слота `W(theta;f)=Σ_h c_h·b(h·f)·sin(h·theta)`; Sine идёт прямым `np.sin` → побитово равен движку `laplacian` (12 с Jam, 1488 блоков, diff 0). **Filter**: своя структура несущих (канал = позиция голоса, частота `f0`), маска `P_k=Σ_j a_j·exp(−0.5(log2(k/r_j)/σ)²)`, `E=P/max P`, `H=10^(−D(1−E)/20)`, масштаб `A=√Σa_j²`; K гармоник (180 при f0 110 Hz) через кэшированные базисы `cos/sin(k·ω·i)`; фигура без весов и плоское `P≤1e−12` молчат (голой несущей нет и при D=0). Полоса `b(ν)` (1 до 0.40·sr, косинус, 0 на 0.45·sr) — по частоте КАЖДОЙ волны; Saw/Square из полосно-ограниченной таблицы под точную частоту (`TABLE_N 8192`, кэш 256, ошибка ≤ −76 dB против прямой суммы). Ручки `method` Filter/Wave bank, `waveform` Sine/Saw/Square, `filter_width_oct` 0.10…1.00 (0.35), `filter_depth_db` 0…36 (24; обе неактивны в Wave bank) + семь настроек Laplace (`copy_spectrum` работает с `laplacian`). Переходы: спектр/волна — глайд за один блок по `_RAMP`, метод — кроссфейд за время отпускания голосов, повторное переключение разворачивает его. Снимок полный, Continue байт-в-байт (в т.ч. из середины кроссфейда). Сцены `demos/lc_*.json` (6), каталог `lab_catalog/laplace_carriers_2026_09_20/`, `run_laplace_carriers.bat`, отчёт `demos/results/laplace_carriers/`. Тесты `tests/test_laplace_carriers.py` (36, гейт 1r). **Скорость банка (2026-09-21, лог `memory/log/2026-09-21-render-budget.md`):** кэш таблиц стал LRU (полная очистка при переполнении стоила ~69 `irfft` на блок) и лежит строками одного буфера (`table_row` / `table_buffer`, `TABLE_CACHE_MAX` 512); блок считается слабами по 96 слотов, а диапазоны семплов раздаются `render_pool`. Порядок сложения сохранён тем, что текущие L/R едут первой строкой суммы слаба. Старый цикл жив как `render_reference` — эталон гейта `SlabRender` и путь того блока, где МЕНЯЕТСЯ форма волны. Живьём на поле прототипа: Saw 24.5 → 3.5 мс на блок, 871 → 1 андерран. |
| `check.py` | **Все гейты одной командой** (см. «Гейты») |
| `casynth_lab/laplace_fm.py` + `demos/build_laplace_fm.py` + `demos/laplace_fm_reference.py` + `demos/laplace_fm_report.py` | **Laplace FM — моды фигуры модулируют её несущую (2026-09-20, ТЗ `memory/req-laplace-fm-2026-09-20.md`, лог `memory/log/2026-09-20-laplace-fm.md`)**: движок стенда `laplace_fm` («Laplace FM», `STATE_VERSION 1`) — тот же `analyse(..., 'laplacian', семь настроек)`, но у каждой фигуры ОДНА синусоидальная несущая на `f0`, а её моды входят в ОДНУ фазу: `y = A_object·sin(theta_c + Σ_j beta_j·sin(theta_j))`, `beta_j = I·a_j`, `A_object = √Σa_j²`. Ручка **`FM depth` 0…4 (default 1)** + семь настроек Laplace (`copy_spectrum` работает с `laplacian`/`laplace_carriers`). Индекс не нормируется (ни на число мод, ни на Σa, ни на RMS) и берётся до общего Gain; совпадающие частоты складываются когерентно; **depth 0 = синус несущей**, а не исходный Laplacian; фигура без ненулевых мод молчит. Полоса: сумма строится на **8×sr** и опускается линейно-фазовым FIR Кайзера (ровно до 0.40·sr, −118 dB от 0.50·sr; длина `2·OS·39+1` → задержка **39 выходных отсчётов при любом oversampling**, поэтому эталон 16×/32× совпадает по времени), затем фиксированный DC-блокер 5 Гц (замкнутая форма, == скалярной рекурсии 6e−16). Время — шкалы baseline (release 3 блока = 23.95 мс): несущая держит фазу, ручки не сбрасывают фазы, смена моды уводит старый модулятор в ХВОСТ МОДУЛЯТОРА (фаза идёт, индекс спадает за release), исчезновение фигуры уносит весь источник в хвост несущей с замороженными индексами (немодулированного синуса на отпускании нет). Хвосты обоих пулов: свободный слот, иначе кража самого тихого со счётчиком. `engine.mono` — float-сумма блока до пана/гейна/int16, только для гейтов. Сцены `demos/lfm_{baseline,saw_bank}.json`, каталог `lab_catalog/laplace_fm_2026_09_20/`, `run_laplace_fm.bat`, отчёт `demos/results/laplace_fm/`. Тесты `tests/test_laplace_fm.py` (45, гейт 1s). **Скорость (2026-09-21, коммит `0cd38be`, лог `memory/log/2026-09-21-render-budget.md`):** колоночные порции блока раздаются `render_pool` (один hand-off на поток, воркер идёт по своему диапазону семплов кэш-порциями) — numpy отпускает GIL внутри `sin`, звук побитово тот же при любом числе потоков (гейт `ThreadedRender`). Живьём: 6.71 → 4.65 мс на блок, 281 → 0 андерранов; драг спектральной ручки 52 → 24 мс (дальше побитово не ускорить — упирается в ~500k `sin` на 8× сетке). |
| `casynth_engines/laplace_unified.py` + `casynth_engines/unified_events.py` | **Единый Laplace-движок — артикуляция x озвучивание (2026-09-21, ТЗ `memory/req-unified-laplace-2026-09-21.md`, лог `memory/log/2026-09-21-unified-laplace.md`)**: движок `laplace_unified` («Laplace+») — ДВЕ оси над одним лапласиановским спектром: `artic` Env / Events и `voice` Bank (Sine/Saw/Square) / FM, плюс семь спектральных настроек, действующих во ВСЕХ клетках, и ручки осей (`FM d`, `ev`, `rad`, `dec`, `atk`). Клетка не переписывает закон: Env+Bank — `_BankVoices` из `laplace_carriers`, Env+FM — `_FMSources` из `laplace_fm`, Events — сам `ObjectResonatorsEngine` (при Sine — его собственное ядро `_render`). Поэтому четыре якоря — это уже написанные гейты: Env+Bank+Sine == `laplacian`, Env+Bank+Saw/Square == `laplace_carriers` (Wave bank), Env+FM == `laplace_fm`, Events+Bank+Sine == `ca_object_resonators` — байт-в-байт, в том числе под нотой и под драгом громкости. **Новые клетки** (`Events x Saw/Square`, `Events x FM`): у моды берётся амплитуда `a_j = w_j·|z_j|` и её СОБСТВЕННАЯ фаза `theta_j = arg(z_j) + pi/2` (читается из состояния, не хранится рядом — поэтому h=1 у волны равен ровно `Re(z)`, а мода, уехавшая в хвост, уносит фазу со своим z). Волна — закон `req-laplace-carriers` §3 без изменений (`c_h`, полоса `b(h·f)` по частоте каждой волны, таблица на точную звучащую частоту); ФМ — закон `req-laplace-fm` §2-4 без изменений (`beta_j = I·a_j` в радианах, ни на что не делится, одна несущая на фигуру, одна фазовая сумма, 8x oversampling + FIR Кайзера, фиксированный DC-блокер 5 Гц), но индекс теперь ЗАТУХАЕТ вместе с модой — это новый закон, а не следствие старого. Переключение оси не даёт ни удара, ни обрыва: ушедшая клетка дозвучивает и гаснет за кроссфейд (>= 20 мс), пришедшая создаётся МОЛЧА (`prime_silent`), обратное переключение разворачивает кроссфейд. Смена волны внутри клетки — смешивание двух чтений за один блок. **Старые четыре id — алиасы**: их фабрики идут через `laplace_unified.create(...)`, который отдаёт клетку, закреплённую осями, то есть тот же класс, те же параметры, тот же снапшот и те же байты. В `object_resonators` добавлены три точки расширения без смены поведения: `begin_block`/`end_block`, `prime_silent`, `attach_slot_state`. Только SR 44100 / блок 352 / стерео. Тесты `tests/test_laplace_unified.py` (гейт 1v). |
| `tests/` | `test_casynth_core.py` (60 тестов) + `test_demo_lab.py` (39 тестов S1–S3) + `test_demo_lab_s4.py` (9 тестов S4) + `test_demo_lab_s5.py` (6 тестов S5) + `test_demo_lab_s6.py` (6 тестов S6, изолированные git-репо в `artifacts/_s6/`) + `test_demo_lab_s7.py` (6 тестов S7, `artifacts/_s7/`; демо-каталог `s7_demo_catalog.py`) + `test_demo_lab_sn.py` (16 тестов демо S/N, `artifacts/_sn/`) + `test_n2_events.py` (19 тестов N2, `artifacts/_n2_tests/`) + `test_n3_tuned_events.py` (16 тестов N3, `artifacts/_n3_tests/`) + `test_n4_object_resonators.py` (28 тестов N4, `artifacts/_n4_tests/`) + `test_objects_laplace.py` (13 тестов Objects/Laplace, `artifacts/_ol_tests/`) + `test_objects_radius_attack.py` (19 тестов Radius range / Attack, `artifacts/_ora_tests/`) + `test_objects_event_source.py` (13 тестов Events / Birth position, `artifacts/_oes_tests/`) + `test_objects_birth_strength.py` (8 тестов Birth strength, `artifacts/_obs_tests/`) + `test_objects_decay.py` (25 тестов Decay law) + `test_laplace_carriers.py` (34 теста Laplace carriers) + `test_laplace_fm.py` (43 теста Laplace FM) + `test_laplace_unified.py` (единый Laplace: оси, четыре байтовых якоря, алиасы, новые клетки) + `timing_gate.py` (метка `@timing_test`: тест, который МЕРЯЕТ время, и потому не делит машину) + пробы прототипа `ui_click_probe.py` / `ui_sound_probe.py` / `ui_articulation_probe.py` / `ui_panel_probe.py` / `ui_save_probe.py` + `golden/` (эталоны golden-master и UI) |

## Гейты (обязательны после каждого изменения)

_2026-09-21: добавлен гейт **3b** — `tests/ui_click_probe.py`: сценарные клики по КАЖДОЙ ветке
мышиной цепочки прототипа (кнопки, BPM, деление, громкость, все ряды ручек, вкладки, пианино,
рисование). Появился после того, как забытое поле `hit` у синтезаторных строк валило любой клик по
тулбару, а три полных `check.py` это пропустили: пиксельный эталон кадра НЕ посылает в окно событий
и событийный цикл прототипа не исполнялся ни одним тестом._

_2026-09-21: добавлен гейт **1t** — `tests/test_seam.py` (19 тестов шва: пакет движков без стенда и
без numba, контракт всех 17 движков, кольцо `casynth_host`, решение панели `casynth_panel`,
артикулированная сцена, `VoiceEnvelope` против прежней арифметики, и **путь через хост побайтово
равен прямому пути** `analyse → SlotPool → render_chunk_laplacian`). Тайминговые тесты бюджета
Laplace FM (гейт 1s) нагрузочно-чувствительны: на занятой машине падают и на НЕТРОНУТОМ коде.
(2026-09-21, после перевода ФМ на пул потоков `KnobDragBudget` проходит с запасом — но природа
теста та же, и на занятой машине он снова станет чувствительным.)
2026-09-21: то же верно для S4 (запись/реплей идут в реальном времени). Признак — набор падает
внутри `check.py` и проходит изолированно; лечится чистым прогоном, а не правкой кода._

_2026-09-21: три гейта на вопросы, которые НИ ОДИН гейт не задавал, пока инструмент стоял немой._
_**3c** `tests/ui_sound_probe.py` — «нажал Random и Play, уровень двигается?» на каждом движке._
_**1u** `tests/test_gen_envelope.py` (8 тестов) — живые ручки GEN и темп доходят до каждого движка,_
_который их заявляет, и ни до одного, который не заявляет; подсказка реестра совпадает с классом;_
_«хост ничего не сказал» == «сказал дефолты» байт-в-байт (это и держит каталог)._
_**3d** `tests/ui_articulation_probe.py` — слышна ли огибающая VOICE вообще: отпустил клавишу (при_
_HOLD выкл нота отпускается, при вкл держится) и проигрывание файла (дефолты → огибающая плоская,_
_медленная атака + низкий sustain → ходит, хотя гейт открыт все 100 % блоков)._

`python check.py` = 60 юнит-тестов + 42 + 9 + 6 + 6 + 6 + 15 тестов demo-стенда S1–S7 и демо S/N + 9 тестов N0 (гейт 1h) + 10 тестов N1 (гейт 1i) + 3 теста гипотез N1 + 19 тестов N2 (гейт 1j) + 16 тестов N3 (гейт 1k) + 28 тестов N4 (гейт 1l) + 13 тестов Objects/Laplace (гейт 1m) + 19 тестов Objects radius/attack (гейт 1n) + 13 тестов Objects events/modal (гейт 1o) + 8 тестов Objects birth strength (гейт 1p) + 25 тестов Objects decay law и контроля D4 (гейт 1q) + 34 теста Laplace carriers (гейт 1r) + 43 теста Laplace FM (гейт 1s) + 8 тестов живых ручек GEN (гейт 1u)
+ golden-master аудио (байт-в-байт,
sha256[:16]=cd3126907ad13b6c) + UI-кадр (пиксель-в-пиксель vs `tests/golden/ui_frame.png`)
+ import/init smoke. Эталоны ВЕРСИОНИРУЮТСЯ в `tests/golden/` (переехали из gitignored
`artifacts/` 2026-07-14). Легитимное изменение UI → `python check.py --bless-ui` в том же
изменении. Конвенция каждой новой ручки: **дефолт = бит-в-бит**.

**Как гоняется (2026-09-22, лог `memory/log/2026-09-22-instrument-panel-and-save.md`):**
две волны — сначала гейты, которые МЕРЯЮТ ВРЕМЯ (по одному, на неразогретой машине, с паузой
`COOLDOWN_S = 6` между ними), потом всё остальное пулом на 6 процессов с `CASYNTH_SKIP_TIMING=1`.
Тайминговый тест обязан нести `@timing_test` (`tests/timing_gate.py`), иначе в параллельной
волне он меряет планировщик. Полный прогон **231 с** (было 623), `python check.py --fast` —
**13 с** (юниты, golden master, шов, единый Laplace, пробы кликов и панели); `--jobs=1` —
прежний последовательный прогон, `-v` — вывод всех гейтов. Каждый гейт хронометрируется,
самые медленные печатаются в конце. Все гейты гоняют инструмент при `CASYNTH_VOLUME=0.01` —
прогон не должен играть музыку тому, кто сидит рядом.

**Новые гейты 2026-09-22:** **3e** `tests/ui_panel_probe.py` — ручка, решающая судьбу ДРУГОГО
ряда (`shape` над `dyn` в Laplace+), обязана этот ряд показать без других кликов; она же
проверяет поля Min/Max (ввод 8 → размах 1..8, 999 отвергнут, правый клик вернул 1..20). Читает
`CASYNTH_PANEL_LOG`. **3f** `tests/ui_save_probe.py` — сыграть, нажать Save и спросить у САМОГО
стенда: есть ли запись, реплеится ли байт-в-байт, можно ли продолжить.

**Известное красное:** `ui frame` расходится строкой тулбара с ИМЕНЕМ аудиовыхода (эталон снят
на другом устройстве). Решение пользователя 2026-09-22: оставить как есть, не блессить.

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
  **Живые ручки GEN на всех вкладках (2026-09-21, лог `memory/log/2026-09-21-gen-envelope-and-hold.md`):**
  `set_envelope`/`set_rate` контракта опциональны, и до этой правки их читал только
  `LegacySynthEngine` — на вкладках Laplace waves / Laplace FM блок GEN (и темп, и slew) не
  действовал вовсе. Общий миксин `legacy_engine.GenEnvelopeKnobs` подмешан в оба; slew у них
  объявлен НЕдействующим (`SUPPORTS_AMP_SLEW=False`: у Filter-несущих и FM-источников своя
  поканальная огибающая). Objects читает не GEN, а свой Decay law. Что действует — говорит
  реестр (`EngineSpec.gen_envelope` / `.gen_amp_slew`, сверяются с классом через
  `engine_api.reads_gen_envelope` / `supports_amp_slew`), панель прототипа рисует недействующий
  блок текстом `engine's own`. Движок, которому хост ничего не сказал, байт-в-байт прежний —
  каталог и golden не тронуты.
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
- **Несущая:** экранное/клавиатурное пианино C3–C5 + MIDI. **Note-off и HOLD (2026-09-21):**
  отпускание клавиши/мыши = note-off (до этого `KEYUP` не обрабатывался вовсе и VOICE release
  звучал только от MIDI). Гейт опускает `_release_gate()`, когда его отпустило ПОСЛЕДНЕЕ из
  четырёх: клавиатура, мышь на пиано, MIDI, файл-плеер. Тогл `hold` в заголовке VOICE (дефолт
  выкл) возвращает латч: нота может только смениться, но не деактивироваться — действует и на
  MIDI. Стартовый `gate=True` сохранён (поле звучит сразу).
  **Удар ноты переартикулирует (2026-09-21):** мелодия — это легато (в первой минуте
  `moonlight-sonata.mid` 199 нот закрывают моно-гейт 16 раз, каждая пауза 0.000 с), а
  `VoiceEnvelope` запускала атаку только по ФРОНТУ гейта — значит при игре файла из четырёх ручек
  VOICE действовала одна S. Теперь удар — событие: `state['onset']` инкрементируют все четыре
  пути note-on, рендер-тред зовёт `VoiceEnvelope.retrigger(attack_s)` (та же арифметика фронта;
  при дефолтах A=0, S=1 → ровно 1.0, no-op). Событие сцены `note` в `casynth_lab/runner.py`
  делает то же, а `midi_onsets` сессии теперь пишет и повторный удар той же ноты под удержанным
  гейтом. **GEN-огибающая мод НЕ ретриггерится** — «нота = транспоз» не тронуто.

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
- ~~Оффлайн-реплей не воспроизводит VCA-артикуляцию~~ — **закрыто 2026-09-21**: путь сверки теперь
  СЦЕНА (`python gol_synth.py scene <ts>`), она несёт ноту, гейт, громкость, обе огибающие, пан,
  правки поля и шаги автомата на их сэмплах. Живая сессия прототипа совпала с оффлайн-рендером
  стенда побайтово (353408/353408 и 343552/343552 после того, как прототип стал хостить движок).
  Старый `replay_session` остался только для пяти маппингов ядра и сам отсылает к сцене.
- PyInstaller-сборка (`CASynth.spec`, dist/) — разовый эксперимент Пользователя, НЕ канал
  дистрибуции; spec закоммичен исторически (решить судьбу при следующей уборке).
- **Тех. долг P2 очереди команд (2026-09-18, решение пользователя — сейчас не чинить):** прогноз
  `runner._params_when_applied` в ветках `copy_side` / `copy_spectrum` берёт параметры источника как есть
  сейчас, а не после команд перед копией в той же очереди. Итог — ложный отказ допустимой команды,
  поставленной в тот же блок (пример: A full=1 → copy_spectrum A→B → B alpha при B Modal). Звук и
  состояние не портятся, проверка при применении остаётся. Фикс: прогноз обеих сторон вместе в порядке
  очереди + тесты из ревью (`memory/research/objects-decay-feedback-review-2026-09-18.md` §5). Раннер
  в звуковом наборе — делать при следующей его правке.

## Ожидает прослушивания (очередь слуховых гейтов; готовить кейсы — через ab_bench/record+replay)

1. **`tune`** — главный слуховой вопрос ветки: Laplace-металл, A/B tune 0↔1 — интервалы
   из минимумов консонанснее 12-TET? Контроль: гармонический движок + tune=1 (магнит к JI).
2. **`fullshape`** — A/B «центр+shape (8×8) vs вся форма»: даёт ли граница различимость
   СВЕРХ достигнутой shape. (+ формальный code-review реализации.)
3. **Explainer-ы** — слуховой/визуальный пласт (solo-моды, раскраски) — не блокер.

_(A/B стенд `ab_bench.py` с пресетами (клавиши 1–9, `ab_presets.py`) — построен ровно
для этой очереди; спектр-полоски под полями показывают разницу глазом.)_

## In-flight / следующее

- **Единый Laplace-движок (2026-09-21): реализован, гейты зелёные, ДВЕ НОВЫЕ КЛЕТКИ ЖДУТ
  ПОСТАНОВКИ СЛУХОВОГО ОПЫТА (Researcher)** — ТЗ `memory/req-unified-laplace-2026-09-21.md`,
  лог `memory/log/2026-09-21-unified-laplace.md`, коммиты C1 `04ea53b`, C2 `81a94aa`, C3.
  Движок `laplace_unified` («Laplace+», вкладка прототипа и кнопка стенда): оси `artic`
  Env / Events и `voice` Bank (Sine/Saw/Square) / FM над одним лапласиановским спектром.
  Четыре старые клетки — байт-в-байт свои движки (гейт `1v`), старые id стали алиасами,
  сверка каталогов не дала ни одного нового расхождения.

  **К постановке опыта — то, чего раньше не существовало как кода:**
  - `artic = Events`, `voice = Bank`, `wave = Saw | Square` — фигура, которую ударил автомат,
    звучит не синусами мод, а пилой/меандром на частоте КАЖДОЙ моды. Первая гармоника волны
    равна ровно синусной линии, гармоники `c_h·b(h·f_j)` — надстройка над ней.
  - `artic = Events`, `voice = FM` — у каждой фигуры одна несущая на `f0`, её моды —
    модуляторы в одной фазе, `beta_j = I·a_j` в радианах, и индекс ЗАТУХАЕТ вместе с модой:
    удар даёт яркий приступ, гаснущий в чистую несущую. Классическая ФМ-перкуссия, но это
    новый закон, а не следствие старого. `FM d = 0` — затухающий синус на `f0` с масштабом
    фигуры, а НЕ возврат к сумме мод.

  Действующие в этих клетках ручки: семь спектральных, `ev` (Both/Births/Deaths), `rad`,
  `dec` 0.20…1.50 c, `atk` 0…20 мс и `wave` либо `FM d` 0…4. Кейсы Developer не придумывал —
  ТЗ §6 и [пайплайн](research/listening-experiment-pipeline.md): нужен нетривиальный вопрос
  про автомат и решение, которое ответ может изменить.

  **Два замечания к постановке.** (1) Клетки — АЛЬТЕРНАТИВЫ на одной оси, а не надстройки:
  сравнивать осмысленно `Events x Sine` против `Events x Saw` / `Events x FM` на одном поле
  и одних спектральных настройках (движок один, «скопировать спектр» не нужно — он общий).
  (2) Громкость осей не выровнена: `Events` заметно громче `Env` на одном гейне (разные
  исторические калибровки). Для записи каталога это `side_gain` сцены; как ручка внутри
  одного движка этого нет — вопрос в `questions.md`.

- **Laplace FM (2026-09-20): реализовано, каталог собран и запинен (`e418449`), ждёт сверки Researcher и слуха** —
  ТЗ `memory/req-laplace-fm-2026-09-20.md`; лог `memory/log/2026-09-20-laplace-fm.md`; измерения
  `demos/results/laplace_fm/report.md`. Движок `laplace_fm` («Laplace FM»): те же лапласиановские частоты/веса, но
  у каждой фигуры одна синусоидальная несущая на `f0`, а её моды — модуляторы в одной фазе, `beta_j = FM depth · a_j`
  (см. карту файлов). Каталог `lab_catalog/laplace_fm_2026_09_20/` — три записи; вход `run_laplace_fm.bat`; заметки:
  `python -m casynth_lab.catalog notes lab_catalog/laplace_fm_2026_09_20`. Поле — тот же Jam, что у carriers
  (32×32, тор, 4 поколения/с, 12 с, f0 110 Hz, n 3 / spread 1 / alpha 0 / shape 0 / harm 0 / full 1 / dyn 0):
  **1** `lfm_baseline` (А `laplacian` / Б FM depth 1), **2** — ПОБАЙТОВАЯ КОПИЯ уже прослушанной записи
  `lc_saw_baseline_bank` (А `laplacian` / Б Wave bank+Saw) со своим пином `cea5127`, заголовком и Notes
  пользователя (перепрослушивание не требуется; её id старше, поэтому на экране порядок читается 1, 3, 2 —
  замысел несут номера в заголовках), **3** `lfm_saw_bank` (А Wave bank+Saw / Б тот же FM). Калибровка — один
  множитель на вариант: R 1.00 и W-saw 0.78 сохранены из каталога carriers, их дорожки **побитово равны**
  прежней записи (sha256 совпал), FM — 0.92, одинаков в обеих записях. Уровни R/W-saw/FM −29.93/−29.94/−29.96 dBFS
  (разброс 0.03 dB), клипа нет, pre-clip пик по ВСЕМ блокам 0.135/0.156/0.087. Отчёт: моды == baseline и preflight
  (0), закон совпал с независимым разложением Бесселя (2e−13), совместная модуляция ОДНОЙ несущей (−158 dB против
  совместной формулы, −2 dB против суммы отдельных несущих), эталон 16×/32× ≤ −165 dB и реализация 8× ≤ −155 dB на
  стационарных случаях (ТЗ: −80 / −60), на полном Jam с переходами −98.5 / −88.9 dB (контроль при depth 0 даёт
  −94.5 dB → эти цифры меряют поблочную рампу при сравнении двух частот дискретизации, а не алиасинг), DC-блокер
  −123 dB, Continue точен из середины смены поколения и на паузе, replay каталога — match. Бюджет с анализом поля
  и oversampling: Jam p99 2.3 мс, нота 1760 Hz — 1.8 мс, Jam с n 20 — 2.0 мс (бюджет 7.98); плотное случайное поле
  выходит за бюджет (15.1 / 26.5 мс при n 3 / n 20 — на том же поле `laplacian` даёт 12.8 / 16.6 мс).
  Побочно: семнадцатый движок потребовал пятого ряда кнопок стенда — они стали 21 px с зазором 3 px
  (`5 × 24` == прежние `4 × 30`), высота окна не изменилась (872 px, рабочий стол пользователя 1536×960).
  **Андерраны при вращении ручек (2026-09-21, по фидбеку пользователя): найдены, закрыты гейтом, починены.**
  Драг спектральной ручки на живом поле — худший случай: меняется каждая частота каждой фигуры каждый блок
  (значит каждый модулятор уходит в хвост и звучит весь release), и стенд на 60 Гц шлёт НЕСКОЛЬКО `set_param`
  в один блок, каждый из которых заново анализировал поле. Было 14.6 мс на блок в среднем при бюджете 7.98
  (при 8 командах — 23.7), стало 3.5–3.8 мс и больше не зависит от частоты команд. Три побитово нейтральные
  правки: (1) `casynth_core.map_laplacian` строит `L = D − A` плотно вместо круга через scipy (~127 → ~6 мкс
  на фигуру; помогает всем лапласовым движкам) — ВНИМАНИЕ: `-A` оставляет `-0.0` в дырках, и LAPACK от этого
  возвращает собственные значения с разницей в последнем бите, которая перебрасывает одно через порог
  `> 1e-6` и меняет нижний мод (гейт Objects/Laplace поймал 0.34 Hz); правильная запись — `-1` в нулевую
  матрицу, закреплено тестом `test_laplacian_matrix_equals_the_sparse_reference_bit_for_bit`;
  (2) `laplace_fm` считает блок колоночными порциями (рабочий набор в кеше, те же операции в том же
  порядке — вдвое быстрее при том же бите); (3) `laplace_fm.set_params` откладывает анализ до `render`,
  поэтому на блок приходится один анализ, а не один на команду. Звук не изменился: replay обеих записей
  каталога FM, опорной записи carriers и 19-минутной живой сессии пользователя — `match`; golden master цел.
  Гейт: `tests/test_laplace_fm.py::KnobDragBudget` (4 теста, поле и настройки той сессии вшиты). Решающая
  величина — СРЕДНЕЕ по блокам (поток рендера держит 40 мс запаса); редкие выбросы ~8 мс — планировщик ОС.
  **Бюджет блока на ЖИВОМ поле прототипа (2026-09-21/22, логи `2026-09-21-render-budget.md` и
  `2026-09-22-instrument-panel-and-save.md`).** Мерилось headless-прогоном самого прототипа
  (Random + Play + удержанная нота, фальшивое устройство тянет кольцо в реальном времени).
  При прежних дефолтах (n 12): Laplace+ Saw 24.5 → **3.5 мс** (871 → 1 андерран), ФМ 6.71 → **4.65**
  (281 → 0), Laplace waves Saw 3.4, Laplace+ Sine 2.8; `laplacian` (legacy `render_chunk_laplacian`)
  НЕ оптимизирован — 5.0 мс. **При стартовых настройках игрока (n 20, shape/harm/dyn на максимуме)
  работа примерно удваивается:** Sine 5.25 (0 андерранов), Saw **8.34** (28), ФМ **9.58** (284) при
  бюджете 7.98 — то есть на Saw и ФМ с этими настройками дропы возвращаются. Драг спектральной
  ручки в ФМ — 24 мс на блок.
  Абсолютные миллисекундные пороги на этой машине пляшут до ×1.9 от нагрева — поэтому гейты,
  которые их меряют, `check.py` гоняет первыми, по одному и с паузой.
  **Три случая игрока закрыты 2026-09-22 (лог `memory/log/2026-09-22-live-budget.md`, коммиты
  `7738e24`, `5f89ea5`, `b9aa68f`; гейт `tests/test_live_budget.py` = `1w`, поля и ручки взяты
  из сессий пользователя).** Всё побитово, `golden master` зелёный.
  1) **`artic` → Events** давал 504 андеррана за 10 с: блок 9.98 мс, из них 6.3 — `atan2`+`sqrt`
  на КАЖДУЮ моду КАЖДОГО семпла (1040 мод × 352 = 366 тыс.). Блок теперь считается ПО СЛОТАМ
  (`_wave_ramps` / `_wave_slots` / `_wave_mix`, слоты делит `render_pool`; семплы делить нельзя —
  рекуррентность). Плюс `_quietest_tail` вместо 96 вызовов `_energy` на вытеснение и компилируемые
  `figures.laplacian_matrix` / `periodic_mean`. 9.16 → **3.84 мс** (секунда звука 1.14 → 0.48 с).
  2) **Драг `harm`** давал 1000 андерранов за 103 с (`rend` до 165 мс) при artic=Env: поле
  анализировалось ДВАЖДЫ (UI-поток для картинки + `set_params` в потоке рендера, те же аргументы),
  и волновые таблицы строились по одной. `casynth_engine.analyse` теперь помнит ответ по аргументам
  (4 записи, хит отдаёт КОПИЮ — Env-ячейка перезаписывает `pan`), `set_params` 278 → 6 мс;
  `lc.table_rows` берёт частоты блока разом (один `irfft` батчем через пул, полуспектры на общей
  сетке гармоник). Драг: 0.90 → **0.52** доли реального времени (Bank+Square), 0.66 → **0.46** (ФМ).
  3) **Краш без трейсбека** (0xc0000005, модуль «unknown»): `unified_events.WaveTables` был ВТОРЫМ
  пулом поверх `laplace_carriers`, и его `clear()` пересоздавал буфер на 16 строк — строки, уже
  записанные в `tab_a`/`tab_b` этого блока, оказывались за границей, а у ридаута нет проверки
  границ. Драг `harm` просит таблицы на ~1130 РАЗНЫХ частот при ёмкости 512 (в покое — 58).
  Теперь пул ОДИН, строка помечается блоком и не отдаётся второй частоте, буфер только растёт,
  а частота, которую пул не может обслужить, возвращает −1 = «мода читается как Re(z)»
  (Env-ветка тоже это понимает); обслуживаются НИЗКИЕ частоты первыми. Репра
  `tests/wave_pool_repro.py` на родителе падает на 6-м блоке, гейт `WavePoolSurvives` гоняет её
  подпроцессом. **Осталось решением Researcher:** ключ таблицы по точной частоте не может
  обслужить тысячу одновременно плывущих мод — REQ `req-laplace-carriers-2026-09-20.md` §9
  предлагает ключ по ЧИСЛУ ГАРМОНИК (brickwall вместо косинусного спада); при перегрузе сейчас
  ~98800 ридаутов за 240 блоков уходят в Re(z).
  Побитовый резерв по СЕМПЛАМ по-прежнему выбран: дальше либо n поменьше, либо размен бит
  (float32-sin даёт ×6.6 по замеру, рекуррентные синусы, OVERSAMPLE 8→4) с ре-блессом golden master.
  **Щелчки при твиках `harm` (2026-09-21, по записи пользователя `20260921-002846-f2423a` «issue»): причина
  найдена, гейт написан, починено.** Щелчок = СТУПЕНЬКА в фазовой сумме; меряется в передискретизованной сумме
  до полосового фильтра (вторая разность на стыке блока против типичной внутри блока). На записи ровно четыре
  таких блока (×4.0, ×8.4, ×30.2, ×33.1) — это блоки, где крался ещё звучащий хвост модулятора: пул был
  `MAX_MODES_PER_OBJ` = 20 на источник, а драг спектральной ручки требует `n × release` хвостов на фигуру
  (11 × 3 = 33). Правка: пул `MAX_MODES_PER_OBJ * 3` = 60, кражи нет вообще — при полном пуле старый модулятор
  **затухает на месте** (своя частота, индекс в 0 за один блок, новая частота следующим блоком; счётчик
  `mod_inplace` в панели как `+N`), `STATE_VERSION 2` со чтением снимков версии 1. На записи худший стык
  ×33.1 → ×1.24, рендеры различаются в 11 блоках из 3759 (ровно четыре бывших кражи); записи каталога FM,
  carriers и сессия «drone» реплеятся `match`, запись «issue» — `mismatch` (в ней и был баг).

- **Laplace carriers (2026-09-20): реализовано, каталог собран и запинен (`cea5127`), ждёт сверки Researcher и слуха** —
  ТЗ `memory/req-laplace-carriers-2026-09-20.md`; лог `memory/log/2026-09-20-laplace-carriers.md`. Движок
  `laplace_carriers` («Laplace waves»): те же лапласиановские частоты/веса, два закона — **Filter** (одна периодическая
  волна на частоте ноты, гармоники окрашены профилем мод) и **Wave bank** (своя волна на каждой моде; Sine = прежний
  `laplacian` побитово). Каталог `lab_catalog/laplace_carriers_2026_09_20/` — шесть записей в порядке ТЗ: сначала четыре
  сравнения с baseline (R / F-saw, R / W-saw, R / F-square, R / W-square), затем два сравнения методов между собой;
  вход `run_laplace_carriers.bat`; заметки: `python -m casynth_lab.catalog notes lab_catalog/laplace_carriers_2026_09_20`.
  Поле Jam из preflight, 32×32, тор, 4 поколения/с, 12 с, f0 110 Hz, n 3 / spread 1 / alpha 0 / shape 0 / harm 0 /
  full 1 / dyn 0. Калибровка — **один множитель на ВАРИАНТ** (R 1.00, F-saw 0.89, F-square 0.97, W-saw 0.78,
  W-square 0.90), поэтому каждый вариант побитово одинаков во всех своих записях (проверено). Отчёт
  `demos/results/laplace_carriers/report.md`: спектры 48 поколений == анализ baseline и == preflight (0), Wave bank /
  Sine == `laplacian` побитово на 12 с, коэффициенты Filter == preflight (5.6e−17), одна маска на Saw и Square,
  D = 0 == чистая волна × A, пустое/одноклеточное поле молчит, таблица ≤ −76 dB против прямой суммы, линии Filter на
  `k·f0` (у Square чётных нет) и Wave bank на `h·f_j`, энергия выше 0.45·sr ≤ −91 dB, уровни в пределах 0.05 dB без
  клипа, Continue и replay точны, p99 4.3–4.8 мс. **Наблюдение к оценке:** при трёх модах Filter звучит ТЕМНЕЕ
  baseline (центроид 141 / 120 Hz против 239), Wave bank — ярче (695 / 517 Hz; 15 % / 8 % энергии выше 1 кГц против
  0 %). Открыто (`questions.md`, 2026-09-20): длительности переходов, собственный пул несущих у Filter и его более
  мягкие переходы, `b(ν)` у Sine, пределы вне сцен.

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

- **Decay law / D1–D3 (2026-09-18): реализовано, запинено (`1fdf8d1`), прослушано и сверено Researcher (`memory/research/objects-decay-feedback-review-2026-09-18.md`: D1 почти одинаково, D2 долгий верхний тон слышен, D3 — Б теряет составляющую, А интереснее; ошибок реализации нет; P2 очереди команд — тех. долг, см. Known issues)** —
  ТЗ `memory/req-objects-decay-2026-09-18.md`; лог `memory/log/2026-09-18-objects-decay.md`. Objects v6: Decay law Fixed /
  Common age / Modal age (см. карту файлов). Каталог `lab_catalog/objects_decay_2026_09_18/` (D1 Octagon II p5: Fixed
  0.947926 s / Common age; D2 Blinker с двумя паузами сценария: Common / Modal age; D3 Jam p3 + Octagon II p5: Fixed 1.39 s /
  Modal age; гипотезы ТЗ в Notes дословно), вход `run_objects_decay.bat`; заметки: `python -m casynth_lab.catalog notes
  lab_catalog/objects_decay_2026_09_18`. Отчёт `demos/results/objects_decay/report.md`: цели == preflight (≤ 2e−16 s), контроль D1
  в пределах 0.00 %, фазы D1 различаются и в PCM Б (0.71–1.06 s) при постоянной А (0.94 s), D2 верх − низ +46.7 dB за 0.5 s
  в обеих паузах (Common +0.04), совпадение с независимой пробой preflight до 1e−11 dB, стороны изолированы, А − Б ≤ 0.04 dB,
  Continue точен, p99 1.2–3.8 ms; большие меняющиеся поля вне бюджета (как и Fixed). Инженерные выборы приняты Researcher;
  P2 очереди команд отложен пользователем как тех. долг (Known issues).

- **Контроль затухания D4 (2026-09-18): каталог собран и запинен (`4f7e072`), ждёт сверки Researcher и слуха** —
  ТЗ `memory/req-objects-decay-control-2026-09-18.md`; лог `memory/log/2026-09-18-objects-decay-control.md`. Каталог
  `lab_catalog/objects_decay_control_2026_09_18/` (D4.1: подобранный Fixed 0.671529 s / прежний Modal; D4.2: тот же Fixed /
  прежний длинный Fixed 1.39 s; поле и сцена D3; гипотезы ТЗ в Notes дословно), вход `run_objects_decay_control.bat`.
  Отчёт `demos/results/objects_decay_control/report.md`: среднее применённой gamma Modal 10.286612669487 1/s по 5 293 928
  парам (== preflight), контроль −0.000050 %; все три условия побитно == хешам preflight, А == А, Б D4.1 == Б D3, Б D4.2 ==
  А D3; уровни в пределах 0.021 dB; replay и Continue точны.

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
