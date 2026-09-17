# Каталог автотестов CASynth (снимок 2026-09-13)

Всё запускается одной командой `python check.py` (все гейты: юнит-тесты, стенд S1–S7, демо S/N, N0–N4, golden-master, UI-кадр). Ниже — каждый тест и
какой кейс он закрывает. Фикстуры ядра: blinker (P3), L-патч, плотный патч,
большой L-патч (> 8×8), 20×20 (> потолка узлов).

## 1. `tests/test_casynth_core.py` — ядро маппинга (59)

### Общий контракт map_*
- `test_contract_shapes_and_ranges` — каждый map_* возвращает (freqs, amps) длины n, amps в [0,1], freqs в [0, guard).
- `test_empty_patch_is_silent` — патч с < 2 живых клеток даёт нулевой спектр без падения.

### Laplacian: базовые инварианты
- `test_laplacian_lowest_mode_is_f0` — низшая ненулевая мода = f0 (якорь высоты).
- `test_laplacian_uses_sqrt_lambda_not_lambda` — частоты = √λ, а не λ (blinker: √3·f0, не 3·f0).
- `test_laplacian_invariant_to_rotation_reflection_translation` — граф изоморфен при повороте/отражении/сдвиге → тот же набор частот.
- `test_laplacian_amps_nonincreasing_rolloff` — амплитуды спадают как 1/i, не растут к верху.
- `test_laplacian_knob_defaults_backward_compatible` — явные spread=0, alpha=1 бит-в-бит равны историческому вызову.
- `test_laplacian_n_is_truncation_only` — меньший n только обрезает хвост: первые 8 частот при n=8 = первые 8 при n=20.
- `test_laplacian_spread_keeps_lowest_at_f0_and_reaches_higher` — spread не сдвигает низшую моду с f0, но достаёт более высокие резонансы.
- `test_laplacian_alpha_controls_brightness` — alpha=0 плоские амплитуды, больше alpha — круче спад; частоты не меняются.

### Laplacian: `shape`
- `test_laplacian_shape_does_not_affect_freqs` — shape никогда не трогает частоты (критерий B).
- `test_laplacian_shape_blinker_exact_amplitudes` — точные амплитуды на blinker: антисимметричная мода даёт проекцию 0, λ=3 даёт 2/√6 (критерий C, ловит перестановку столбцов).
- `test_laplacian_shape_amplitude_invariant_to_rotation` — при shape=1 сортированный вектор амплитуд инвариантен к повороту/отражению/сдвигу (критерий D).

### Laplacian: `harm`
- `test_laplacian_harm_zero_is_bitexact` — harm=0 бит-в-бит с базой.
- `test_laplacian_harm_one_integer_multiples` — harm=1: каждая частота — целое кратное f0.
- `test_laplacian_harm_lowest_mode_stays_f0` — harm при любом значении не сдвигает низшую моду.
- `test_laplacian_harm_does_not_affect_amps` — harm трогает только частоты.
- `test_laplacian_harm_blend_monotonic` — промежуточные harm монотонно тянут отношение к round(r).

### Laplacian: `fullshape`
- `test_laplacian_fullshape_false_is_bitexact` — fullshape=False бит-в-бит с базой (A).
- `test_laplacian_fullshape_small_matches_crop` — форма, влезающая в 8×8, даёт тот же спектр, что extract+crop (B: «перестали резать», не новый алгоритм).
- `test_laplacian_fullshape_large_differs_from_crop` — форма больше 8×8 даёт другой спектр, чем обрезка (C: периферия слышна).
- `test_laplacian_fullshape_invariant_to_rotation` — спектр инвариантен к повороту/отражению/zero-padding маски (D).
- `test_laplacian_fullshape_f0_anchor` — низшая мода = f0 при любом harm (E).
- `test_laplacian_fullshape_node_ceiling` — 20×20 (400 узлов > 256) не падает и детерминирован (F: потолок + децимация).

### Другие движки и реестр
- `test_fft_translation_invariant` — |FFT| инвариантен к циклическому сдвигу.
- `test_random_deterministic` — случайная проекция с фиксированным seed воспроизводима.
- `test_random_slice_consistency` — map_random(n=8) = первые 8 строк map_random(n=20) (матрица режется, не пересеивается).
- `test_engine_registry_integrity` — у каждого движка уникальный id, callable map_*, дефолты параметров внутри [lo, hi].
- `test_engine_registry_call_with_defaults` — вызов каждого движка с дефолтами даёт конечные (freqs, amps) длины n.
- `test_extract_size_and_centering` — extract возвращает size×size патч с массой формы в центре.

### `events_field` (ветка ДИНАМИКА)
- `test_events_field_blinker` — точные значения на переходе blinker: рождения 1.0, «рана» от смертей W_WOUND (критерий F).
- `test_events_field_empty` — нет событий → нули.
- `test_events_field_birth_only` — только рождения → клетки 1.0, раны нет.
- `test_events_field_wound_only` — только смерти → кромка = W_WOUND, сами мёртвые клетки 0.

### Laplacian: `dyn`
- `test_laplacian_dyn_zero_bitexact` — dyn=0 (с exc и без) бит-в-бит с базой (A).
- `test_laplacian_dyn_fallback_no_exc` — dyn=1 при exc=None = deg-путь (B).
- `test_laplacian_dyn_fallback_zero_exc` — dyn=1 при exc≡0 = deg-путь (B).
- `test_laplacian_dyn_scale_invariant` — exc и 5·exc дают те же амплитуды (D).
- `test_laplacian_dyn_does_not_affect_freqs` — dyn только амплитуды.
- `test_laplacian_dyn_symmetry_antisymmetric_mode` — возбуждение выбирает слышимые моды на blinker (C).
- `test_laplacian_dyn_alignment_rotation` — (маска, exc), повёрнутые вместе → те же амплитуды, оба пути fullshape (E).
- `test_laplacian_dyn_alignment_fullshape_large` — децимация применяется к патчу и exc одновременно (E, большая форма).
- `test_laplacian_dyn_analyse_no_leak` — exc объекта B внутри bbox объекта A не протекает в голос A (маскирование в analyse).
- `test_laplacian_dyn_symmetric_pair_real_projection` — симметричная пара на концах глушит антисимметричную моду реальной проекцией, без отката (FIX-C).
- `test_laplacian_dyn_uniform_exc_rollback` — равномерный exc ∝ DC-моде → все проекции 0 → откат к rolloff (FIX-C).
- `test_crop_like_extract_matches_extract` — `_crop_like_extract` использует тот же центроид/окно, что extract (FIX-D).
- `test_exc_for_frames_live_cycle_parity` — реконструкция exc из записи сессии (step_prevs) совпадает с живым расчётом, включая шаг кнопкой (FIX-B).
- `test_exc_for_frames_clear_scenario` — реконструкция переживает clear() (немонотонные gen, serial-матчинг) (FIX-F).
- `test_laplacian_dyn_registry_integrity` — spec параметра dyn валиден, дефолт в диапазоне.

### `tune` (Сетхарес)
- `test_tune_pair_dissonance_js_parity` — pair_dissonance совпадает с JS-эталоном (константы формулы) (B).
- `test_tune_dissonance_curve_harmonic_sanity` — минимумы кривой гармонического спектра в пределах 5¢ от октавы, квинты, кварты (C).
- `test_tune_dissonance_curve_t_tetromino_regression` — Т-тетромино: ровно один минимум у 600.49¢ ± 0.5 (регрессионный якорь).
- `test_tune_snap_ratio_exact` — tune=1 притягивает ровно к ближайшему минимуму (D).
- `test_tune_snap_ratio_midpoint` — tune=0.5 = геометрическая середина в центах (D).
- `test_tune_snap_ratio_monotone` — рост tune монотонно приближает к минимуму (D).
- `test_tune_snap_ratio_octave_fold` — октавная свёртка: 3.0 → 1.5 → квинта → обратно 3.0 (E).
- `test_tune_snap_ratio_transparent` — tune=0 или пустые минимумы — без изменений (F).
- `test_tune_zero_bitexact` — tune=0 возвращает r_raw бит-в-бит (A).

### Рендер
- `test_render_transpose_equivalence` — транспоз несущей в render_chunk_laplacian = чистое масштабирование частоты: (f, r) ≡ (f·r, 1).

## 2. `tests/test_demo_lab.py` — demo-стенд S1–S3 (39)

### Сцены
- `test_scene_loads_and_matches_spec` — laplace_basic v1 читается с ожидаемыми полем, rate, движком, f0.
- `test_scene_v2_loads` — laplace_ab v2: variants A/B, initial_side, listen.
- `test_scene_rejects_unknown_engine` — неизвестный engine_id → SceneError.
- `test_scene_rejects_unknown_and_missing_params` — лишний/отсутствующий параметр движка → ошибка.
- `test_scene_rejects_bad_cells_and_bad_json` — клетка вне поля, кривая пара, битый JSON → понятные ошибки.
- `test_scene_v2_rejects_bad_variants` — нет variants / нет B / лишняя сторона C → ошибка.

### Оффлайн-рендер и детерминизм
- `test_offline_repeat_and_reset_identical_8s` — два рендера 8 с и рендер после reset идентичны; без NaN, без клиппинга, 31 поколение.
- `test_v1_render_preserved_against_s1_reference` — sha256 8-секундного рендера laplace_basic равен эталону приёмки S1.
- `test_stereo_identical_channels_and_nonsilent` — стерео, каналы одинаковы, звук не тишина.
- `test_live_equals_offline_on_scenario` — живой поток (LiveEngine с sink) байт-в-байт равен оффлайну на сценарии.
- `test_ui_update_rate_does_not_change_result` — медленный UI (батчи + sleep) не меняет результат.
- `test_events_apply_at_first_block_boundary_not_earlier` — команда с `at` применяется на первой границе блока ≥ at, порядок по seq.
- `test_tempo_from_sample_count_within_one_block` — 31 шаг за 8 с при 4 Гц, каждый в пределах блока от номинала (темп от сэмплов).
- `test_cli_render_all_outputs_and_errors` — CLI `--render` пишет WAV для A/B/monitor нужной длины; ошибки аргументов.

### Транспорт и часы
- `test_pause_freezes_ca_and_does_not_catch_up` — Pause CA замораживает автомат, после снятия пропущенные шаги не догоняются.
- `test_start_is_silent_and_frozen_until_started` — до Start тишина и поле стоит.
- `test_reset_clears_tails_phases_and_queue` — Restart обнуляет хвосты, фазы, очередь будущих команд, поле → сцена, но running.
- `test_stop_returns_to_initial_silent_scene` — Stop = исходная сцена, тишина, пауза.
- `test_stop_and_restart_keep_settings_but_reset_both_sides` — Stop/Restart сохраняют движки, параметры, громкость, сторону; обе стороны перезапускаются.
- `test_empty_field_goes_silent_after_tails` — пустое поле → после хвостов тишина.
- `test_volume_change_is_smoothed` — смена громкости сглажена (без ступеньки).

### A/B и параметры
- `test_every_engine_runs_on_the_demo_scene` — каждый движок из реестра рендерит на demo-сцене.
- `test_param_memory_per_side_and_engine` — память параметров по (сторона, движок): возврат к движку восстанавливает значения.
- `test_bad_commands_are_rejected_before_any_state_change` — плохая команда (сторона, диапазон, тип) отклоняется в post() без изменения состояния.
- `test_copy_side_and_factory_reset_and_modified_flag` — `<<`/`>>`, Factory A+B, звёздочка «изменено».
- `test_ab_equal_when_settings_equal_and_switching_changes_nothing` — равные настройки → A = B; переключение стороны не меняет сырой звук.
- `test_editing_b_does_not_change_raw_a` — правки B не меняют сырой A.
- `test_demo_ab_sides_differ_and_are_nonsilent` — в laplace_ab стороны различаются и не молчат.
- `test_param_change_applies_at_block_boundary_without_reset` — параметр применяется на границе блока без перезапуска (фазы продолжаются).
- `test_monitor_crossfade_sums_to_one_and_keeps_block_count` — кроссфейд монитора 20 мс: коэффициенты в сумме 1, число блоков не меняется.
- `test_live_equals_offline_on_ab_scenario` — живой A/B сценарий (все три выхода) = оффлайн.

### Интерфейс движка S3
- `test_s2_audio_reference_preserved_through_engine_interface` — 5 методов × фиксированный журнал: хэши равны эталону S2 до рефактора (`golden/demo_lab_s2_ref.json`).
- `test_registered_alias_of_existing_adapter_works_everywhere` — адаптер под новым id работает в сцене, валидации, командах, рендере, UI без правок кода.
- `test_stub_engine_pcm_passes_through_untouched` — PCM стороннего движка доходит до выхода без изменений; порядок вызовов интерфейса.
- `test_malformed_engine_block_never_reaches_output` — кривой блок (форма/dtype) заменяется тишиной и считается, до устройства не доходит.
- `test_registry_rejects_bad_registration_and_duplicates` — дубликат id, не-EngineSpec, кривой spec → ошибка.

### UI и устройство
- `test_ui_headless_smoke_buttons_and_painting` — headless: кнопки транспорта, вкладки, слайдеры, рисование/стирание, хоткеи, кадр рисуется.
- `test_no_audio_device_state_is_visible` — нет устройства → статус «NO AUDIO DEVICE», демо продолжает идти на pacer.
- `test_device_stream_opened_stereo` — поток открывается стерео (channels=2).

## 3. `tests/test_demo_lab_s4.py` — каталог и реплей (9)

- `test_save_replay_fixed_experiment_in_fresh_process` — фиксированный опыт (настройки до Start, рисование, смена движка, громкость, A/B, пауза, Stop/Start, Restart, copy, factory) → Save с кириллицей → в свежем процессе реплей байт-в-байт; запись начинается на Restart; WAV = блоки sink; конечное состояние даёт сцену для Open; старая запись без state_at_end пересчитывается (в т.ч. в дочернем процессе).
- `test_cut_is_a_consistent_prefix_and_session_continues` — срез на границе блока, будущие команды не входят; сессия продолжает работать; второй Save — новая запись, первая неизменна.
- `test_waiting_before_start_does_not_lengthen_the_wav` — ожидание до Start не удлиняет WAV; до Start «нечего сохранять»; событие до Start сохраняется для реплея.
- `test_record_is_independent_of_scene_file_and_engine_availability` — удалён исходный JSON → реплей ок; движок снят с регистрации → запись открывается, WAV играет, реплей «unavailable» с причиной.
- `test_errors_no_false_success_no_lost_records` — прерванная запись, «диск полон», битый record.json, обрезанный WAV, подменённый WAV → без ложного успеха, без .partial, старые записи целы, «Replay differs» с именем дорожки.
- `test_window_keeps_last_seconds_and_replay_still_matches` — кольцо последних N секунд; команды до окна сохраняются в журнале; реплей совпадает.
- `test_restart_and_stop_start_begin_a_new_recording` — Restart и Start-после-Stop начинают новую запись с новыми условиями (нарисованное поле, сохранённые настройки).
- `test_replay_progress_and_cancel` — прогресс до 1.0, отмена, дочерний процесс: тот же вердикт, отмена, неизвестный id → unavailable.
- `test_ui_headless_save_catalog_player_replay` — headless UI: Save до Start, форма с кириллицей, экран каталога, Play A/B/as heard через тот же callback (живой звук заглушён), Stop, Check reproducibility → «Replay matched», Play replay, Esc, Open field anew (конечное поле, пауза).

## 4. `tests/test_demo_lab_s5.py` — снепшоты и ветки (6)

- `test_continuity_five_engines_in_process_and_fresh_process` — для 5 движков: снепшот посреди опыта, продолжение оригинала и восстановление из файлов (в процессе и в свежем процессе) при одинаковых последующих командах (рисование, параметры, смена движка + память, copy, select, громкость, пауза, factory, stop, start, restart) → A/B/monitor, поле, gen, часы, журнал совпадают; звук не тишина.
- `test_hard_boundaries_and_side_effect_free_export` — срез внутри A/B-кроссфейда, при звучащих хвостах, перед шагом КА, при Pause CA, при Stop, с разными движками A/B + сменой параметра + запланированной командой; экспорт дважды одинаков, экспорт не меняет живую сессию, восстановление дважды идентично; отложенные команды выполняются в своё время.
- `test_branch_parent_continue_change_save_reload_check_continue_again` — родитель → Continue (часы стоят до старта, записи не создаётся) → правки → Save ветки (свой исходный снепшот, журнал, 3 WAV, parent_record_id, аудио с точки снепшота) → реплей в процессе и в дочернем → родитель неизменен → папка родителя скрыта: реплей/список/продолжение живут → повторный Continue родителя даёт те же первые блоки → продолжение ветки делает её родителем.
- `test_stop_restart_after_continue_start_fresh_recordings_with_parent_link` — Restart / Stop в продолженной сессии начинают свежую запись (S4-путь) с сохранённой ссылкой на родителя; сохранённый Stop остаётся Stop; окно записи ограничено.
- `test_compatibility_s4_records_and_errors` — S4-запись (формат 1): читается, играет, Open field anew, реплей ок, Continue недоступен, снепшот не дописывается; обрезанный npz, чужая версия, неверные формы поля/массивов → CatalogError; движок без снепшота: стенд и Save работают, Continue честно недоступен; снятый с регистрации движок → причина; сбой сохранения без .partial и порчи; object-массивы отвергаются.
- `test_ui_headless_continue_branch_parent_link` — headless UI по ручной приёмке: «Исходная S5» harm B=0.25 → Continue (сессия заменена, запись от снепшота) → harm 0.75 → «Вариант S5» → ссылка Derived from ведёт к родителю → Check reproducibility = «Replay matched» → Continue родителя: harm снова 0.25 → Open field anew (пауза, ссылка сохранена) → S4-запись: Continue отказан, сессия и экран не тронуты.

## 4b. `tests/test_demo_lab_s6.py` — версии кода и закреплённые записи (6)

Все git-операции — в изолированных тестовых репозиториях под `artifacts/_s6/` (копии
runtime-набора); настоящий checkout проекта не трогается.

- `test_classification_clean_dirty_docs_crlf` — чистый runtime → pinned; незакоммиченная / staged правка DSP, новый runtime-файл → local с именем файла; правка документа → статус прежний; CRLF↔LF и LF-репозиторий дают тот же отпечаток; окружение описано отдельно и сравнивается точно.
- `test_process_origin_is_the_code_that_ran_not_head_at_save` — захват при инициализации исполнителя (X), затем файлы и HEAD уходят на Y, Save → запись приписана X, ref удержания создан; новый исполнитель пишет Y; дочерний worker сообщает собственное происхождение; грязный checkout → local без ref.
- `test_pin_local_record_after_matching_commit` — Pin до коммита отклонён; после коммита ровно того кода запись закрепляется, id/WAV/снепшоты не меняются, повторный Pin идемпотентен; коммит ищется в истории; другой код с тем же PCM и явный чужой коммит отклоняются; запись без отпечатка отклоняется; сбой Git → ложного pin нет; чистый Save с несозданным ref → local с причиной.
- `test_source_version_runs_the_original_code_and_survives_gc` — X (полгейна) на временной ветке, Y на master: план = worktree; headless-продолжение в X даёт байт-в-байт результат X-DSP на том же снепшоте и ≠ Y; ветка из дочернего стенда в общем каталоге, pinned X, с родителем, без каталога внутри кэша; проверка воспроизводимости в Y считает Y (mismatch), worker X — match; checkout/индекс/ветка не изменены; удаление ветки + кэша + reflog expire + gc → X удержан и запускается; повреждённый кэш чинится; потерянный коммит → конкретная причина, WAV читается.
- `test_catalog_compatibility_and_environment` — запись S5 (без происхождения) → local «code version not recorded», продолжение и реплей по правилам S5, Pin отклонён, к HEAD не привязывается; несовместимое окружение → причина, WAV доступен; мусор в блоке происхождения не ломает список и читается как local; capture() не меняет checkout проекта.
- `test_ui_headless_status_pin_and_continue_in_version` — headless UI: статусы Pinned/Local в карточке, Pin после коммита, Continue закреплённой записи другой версии запускает дочерний стенд (headless), остальные действия «busy», по выходу каталог обновлён и ветка закреплена за X; Continue записи той же версии — в процессе с подписью версии.

## 4c. `tests/test_demo_lab_s7.py` — проверка каталога после изменения кода (6)

Git-операции — в изолированных репозиториях под `artifacts/_s7/`; каталоги тестов там же.

- `test_track_classification_and_diagnostics` — `compare_pcm`: точное совпадение; отличие в 1 PCM-единицу (доля 1/n, max 1); тишина↔тишина / тишина↔сигнал; крайние int16 (−32768↔32767 → 65535 без переполнения); разная длина = отличие даже при равном общем префиксе, хвост не режется. На записях: только A / только monitor / всё точно — классификация по дорожкам, выбор дорожки по умолчанию (monitor → первая из A/B), отметки пусты, вердикта о слышимости нет; WAV отличающейся дорожки сохранён и равен «истине» сессии, отпечатки обеих сторон записаны.
- `test_batch_equals_single_replay_on_long_history_and_branch` — обычная запись с предысторией длиннее окна (кольцо 1 с, 3 с событий) и ветка от снепшота с вмешательствами и переключениями A/B; хранимые WAV инвертированы → пакетная проверка в worker-процессе, одиночная `replay()` и PCM живой сессии совпадают байт-в-байт по всем трём дорожкам; in-process прогон даёт те же отпечатки.
- `test_errors_have_reasons_and_sources_stay_untouched` — старая запись (формат 1, без происхождения) проверяется, origin «не записано»; закреплённая запись с отсутствующим коммитом проверяется текущим кодом; ветка с чужой версией снепшота / обрезанным npz, обрезанный B.wav (только B failed, A/monitor exact), незарегистрированный движок, битый record.json → «не удалось проверить» с причиной у каждой дорожки; каталог (без `replay/`) не изменён, пакет не создаёт `<record>/replay/`; WAV недоступной записи читается; пустой каталог → полный отчёт с причиной.
- `test_target_provenance_runtime_change_cancel_partial_and_persistence` — worker из изолированного checkout сообщает СВОЙ commit (≠ HEAD стенда) и он — цель отчёта; изменение runtime-файла во время расчёта 2-й записи → `stopped`, 1-я сохранена, 2-я и 3-я «не проверено» с причиной (и детерминированно in-process через подменённый отпечаток); отмена внутри длинной записи (worker terminate и in-process) → `cancelled`, завершённые сохранены, остальные явно не проверены; частичный сбой (обрезанный B.wav) не даёт полного успеха; одиночная проверка не трогает пару отчёта; отметка «слышу» переживает новый `Verifier` («перезапуск»), не переносится в новый отчёт, старый отчёт неизменен, отметка на совпавшую дорожку и неизвестная отметка отклоняются; подменённый оригинал / удалённый пересчитанный WAV → пара недействительна по отпечатку; отчёт со статусом `running` без процесса читается как `interrupted`, `finalize_run` закрывает его один раз.
- `test_pair_player_shared_cursor_switch_gain_no_live` — плеер пары через `_audio_cb`: живой синт запущен и громок, но не подмешивается; общий курсор; переключение на ожидаемом сэмпле, кроссфейд 441 сэмпл только в плеере, дальше точь-в-точь вторая версия; тишина после конца короткой версии, курсор идёт до конца длинной; обратное переключение за концом короткой; идентичные версии → бит-в-бит (единичный gain); плохие формы/имена версии отклоняются.
- `test_ui_headless_check_catalog_listen_pair_mark_reopen` — три ручных шага headless: «Check catalog» (остальные действия — busy) → отчёт открывается на группе Differs, выбрана «как слушал»; группы Failed/Exact; Play → Saved/Recomputed на одном курсоре (кнопки и Space), смена дорожки сохраняет позицию; отметки Can't hear / Hear it / Not rated; exact-запись играет оригинал для обеих версий, отметка не предлагается; недоступная запись: «Recomputed» нет, «Saved» играет; переоткрытие стенда: отчёт, пара и отметки без пересчёта, папка отчёта не изменена, исходная запись играет как прежде (Play A); второй прогон = новый отчёт без отметок, `< older` возвращает прежний с отметками.

## 4d. `tests/test_demo_lab_sn.py` — движки Scan / Network (15)

ТЗ `memory/req-sonification-sn-demos-2026-09-14.md` §6; поля F1–F5 берутся из `demos/build_sn_demos.py`; временные файлы в `artifacts/_sn/`.

- `test_s_constant_fields_are_silent_for_every_path_and_surface` — пустое/полное поле: все пути × обе поверхности дают ровно ноль (float и int16); Distance на пустом/полном = −1/+1 без inf/NaN; очистка поля во время игры гаснет за 20 мс перехода.
- `test_s_analytic_reading_z_equals_x_gives_cosine` — фикстура z=x: эллипс читает cx + rx·cos, после DC ровно одна гармоника с амплитудой rx и нулевой фазой; сдвинутая синусоида сохраняет комплексную фазу.
- `test_s_paths_raster_coverage_seam_closure_and_radii` — Raster посещает все центры ровно один раз, шов 1 (чётная H) / √2 (нечётная), полная длина, замыкание P(0)=P(1), 1×1 без деления на ноль; эллипс/Лиссажу замкнуты, экстенты = радиусам и оверлею; radius — `Full field` только для Raster; на F1 три пути читают три разные последовательности.
- `test_s_surfaces_periodic_borders_signs_units_and_not_a_copy` — Gaussian: периодичность через шов, симметрия, нормировка ядра; Distance: евклидовы расстояния в клетках на торе, знаки, tanh(1/width) в живой клетке; Distance ≠ перенормированный Smooth (корреляция < 0.99, порядок рангов иной).
- `test_s_density_quality_dc_band_phase_and_no_phase_reset` — удвоенная плотность таблицы (16384 vs 8192) меняет band-limited волну < −50 дБ на F1/F2 (все пути) и F3 (Raster); DC = 0; K = 180 при 110 Гц; комплексные фазы сохранены; блок рендера = точная аддитивная оценка; смена пути не сбрасывает фазу, переход 882 сэмпла; правка посреди перехода стартует с достигнутой смеси, без разрыва; эллипс на F3 читает только хвосты (задокументировано).
- `test_n_coupling_zero_is_the_plain_sum_through_the_same_output_path` — coupling=0 на двух разных непустых полях даёт байт-в-байт одинаковый выход, равный независимому эталону (сумма четырёх синусов /4 → gate → FIR → HPF) с точностью 1e-9.
- `test_n_isolated_edge_direction_and_periodicity` — каждое ребро в отдельности: модулируется только i чистым x_j того же сэмпла; цепочка 2>1>0 без задержки; периодичность x(θ)=x(θ+1) на нецелых θ; beta=0 → сумма независимых.
- `test_n_weights_masks_fields_frozen_links_and_gate` — маски на торе, центры; полное поле → 1/3, пустое → 0; F5 верх/низ ≈ 0.29↔0.04 с перестановкой рядов; F1 ≈ 0.11; F4 слева/справа разные W, копии равны; Pulsar меняет W по циклу; init берёт W сразу, правка сглаживается с tau 30 мс (проверен первый блок); frozen: правки не двигают W, off — возобновляет; начальный freeze_links=1; gate закрывается/открывается, нулевые связи не глушат звук; display — простые числа.
- `test_n_pause_keeps_phase_and_smoothing_running` — на паузе КА фаза идёт, W сглаживаются к цели нарисованного поля, поколение не растёт.
- `test_n_oversampling_4x_vs_8x_and_s_render_quality` — 4× vs 8× на F5T/F5B/F4L/F1/F3, beta 2 и 4, f0 110 и 440: остаток RMS < −50 дБ (факт ≈ −109 дБ); децматор: 2·R·32+1 отводов, симметричен, единичное усиление DC, полоса/стоп-полоса.
- `test_same_journal_same_output_and_continuation_byte_exact` — пять пар (Scan×2, Network×2, Scan+Laplace, Network+Laplace, Scan+Network): одинаковый журнал → одинаковые A/B/monitor; снепшот посреди перехода Scan (0 < xpos < 882) → продолжение с правками/сменой стороны/параметра/паузой байт-в-байт по всем трём дорожкам, in-process и в свежем процессе из файлов.
- `test_snapshot_hygiene_and_rejections` — экспорт не ссылается на живые массивы; неверная версия/форма/K/раскладка рендера/engine_id/скаляр/NaN отклоняются до замены сессии.
- `test_limits_110_440_extreme_params_and_prepared_scenes_do_not_clip` — 110/440 Гц × пустое/полное/редкое/F1 × крайние параметры: конечные числа, int16 (block, 2), два одинаковых канала, n_clip ⇔ peak > 1; все 14 готовых сцен: n_clip = 0, обе стороны не тихие, running/paused как задано.
- `test_gain_and_trim_changes_are_smooth` — скачок gain и смена trim = глайд по _RAMP на неизменной волне (Scan); у Network границы блоков не выделяются в стационаре.
- `test_registry_ui_hints_and_scene_validation` — choices/value_text/describe_difference словами; невалидные значения отклонены; файлы `demos/sn_*.json` равны документам сборщика; панель стенда: 9 движков → 3 ряда, параметры ниже последнего ряда; кнопки режимов в панели; `Full field`; куски шва: замыкание Raster — два отрезка внутри поля.

## 4e. `tests/test_gutter_field_n1.py` — N1: сеть Gutter, управляемая полем (10)

ТЗ `memory/req-network-ca-n1-2026-09-15.md`, «Проверки Developer до передачи»; fixtures `memory/research/network-n1-fixtures-2026-09-15.json`; временные файлы в `artifacts/_n1_tests/`. Гейт 1i в `check.py`.

- `test_config_equals_fixtures_and_mappings_equal_n0` — `casynth_lab/gutter_field_n1_config.py` равен fixtures (+ маршруты R1, `post_math=scalar`, версия модели); все семь маппингов слайдеров равны N0 на raw 0..256; границы областей 2×4 (32×32 и 7×10); для трёх полей fixtures counts/u точно, ratio ≤ 1e-12, частоты float32 и ≤ 19 кГц, повторное применение того же поля не накапливает сдвиг; depth 0 → ratio = scale; клип ±1 октава; потолок 19 кГц.
- `test_kernel_equals_slow_model_bit_exact_static_evolution_edits_params` — numba-ядро против `GutterNetwork` (скалярный режим) байт-в-байт: статика за пределами задержки 2064, две эволюции с перестройкой банков, рампа Links 127→200, разрезанная границей блока, перенос поля туда-обратно без перезапуска, Freeze + scale/depth на удержанном векторе (правка поля в заморозке ничего не меняет), снятие заморозки; счётчики сбросов равны 0; скомпилированное ядро == та же функция в чистом Python (нет FMA-контракции).
- `test_kernel_node_equals_scalar_java_port_in_lockstep` — по отсчётам 2600 сэмплов: ядро, медленная модель и скалярный порт узла (сверенный с оригинальным `gutterOsc.class` на JVM) имеют одинаковые duffX/duffY/t/finalY всех восьми узлов; кольцо задержки совпадает.
- `test_r6_route_off_in_master_and_matrix_and_link_arrival` — одиночная связь 6→4 (0.75) при задержке 3 и 2064 без рамп: правый мастер = 0 и запись в матрицу = 0.5·L₆·0.75 (в N0-маршрутах правый ≠ 0); первое изменение входа демпфирования приёмника ровно на отсчёте D, значение равно независимому расчёту; N1 и N0 маршруты дают одинаковый левый канал и разный правый.
- `test_depth_zero_and_freeze_make_edits_inert_empty_field_continues` — при depth 0 / Freeze CA правки поля (перенос, полное, пустое) не меняют ни коэффициенты, ни PCM; init с Freeze держит начальное поле; выключение берёт текущее поле; повторная заморозка держит достигнутое, scale действует на него; пустое поле → n=0, ratio 0.5, звук продолжается без сброса; `display()` — простые числа.
- `test_same_journal_same_output_and_continuation_byte_exact` — сцена N1: одинаковый журнал → одинаковые A/B/monitor (> 2 поколений); снепшот посреди 50-мс рампы Links и сразу после правки → продолжение с правками/сменой стороны/параметров/паузой/заморозкой байт-в-байт in-process и в свежем процессе из файлов; экспорт не ссылается на живые массивы; отклонения версии/версии модели/формы кольца/задержки/engine_id/NaN/скаляра/набора параметров до замены сессии.
- `test_limits_extremes_no_nan_clip_counter_and_prepared_scene` — 4 поля × 5 крайних наборов ручек: int16 (352, 2), конечные состояния, n_clip ⇔ peak > 1, ни одного сброса, пик < 1 при штатном gain; клип-счётчик при ×100 считает ровно клиппированные значения; при gain 0 второй блок — нули; готовая сцена: обе стороны звучат, клипа нет, A = B до первого шага КА и различаются после; 12 клеток сохраняются; bat-файл ссылается на сцену, `--live`, `--catalog`.
- `test_realtime_budget_two_sides` — обе стороны сцены с правками и сменой Links: среднее время блока < половины бюджета 7.98 мс, p95 < бюджета; numba доступен.
- `test_registry_overlay_display_and_headless_draw` — реестр (метки, диапазоны, отклонения значений), оверлей: 4 линии + 8 подписей в центрах областей, `describe_difference` = «freeze_ca: A=0, B=1»; headless BenchApp: 9 движков → 3 ряда, кадр рисуется, display стороны A = u из fixtures, B заморожена; клик по тумблеру Freeze CA доходит до движка.
- `test_listening_catalog_builder_scenes_records_and_routes_control` — `demos/build_n1_demos.py`: пять сцен на диске равны документам сборщика и валидны; поля сборщика = fixtures (vertical / moved); расписание правок на 4/8/12/16 с с паузой КА; контрольный движок `gutter_field_n0r` (маршруты N0): та же модель, свой `model_version`, левый канал равен N1, правый отличается, снапшот чужого id отклоняется; короткая офлайн-запись в scratch-каталоге сохраняется, `Catalog.replay` = match, Continue открывает поле в сдвинутом состоянии; сборка в каталог с записями отклоняется.

## 4f. `tests/test_n2_events.py` — N2: периодическое чтение поля и событийная сеть (19)

ТЗ `memory/req-network-events-n2-2026-09-16.md`; временные файлы в `artifacts/_n2_tests/`. Гейт 1j в `check.py`.

- `PeriodicReadoutTests` (3) — веса `K_i` против формул ТЗ по клеткам, разбиение единицы, периодичность на торе (сдвиг на шаг узла = соседний узел, непрерывность через внешний край), взвешенные счёты и центры узлов.
- `GutterPeriodicTests` (3) — та же модель, что `gutter_field`, отличаются только чтение и trim −16.5 dB; взвешенные счёты ведут частоты (клетка внутри области N1 меняет счёт); снапшот посреди рампы Links и реестр.
- `EventNetworkTests` (10) — ядро Б == независимый скалярный референс побитово (одиночный и перекрывающийся пакеты; сквозь рампу rho); семантика событий (рождение+смерть без взаимоуничтожения, правки внутри блока сливаются, параметры — не событие, reset = один пакет, `exc` игнорируется); причинность по задержкам по состояниям; нуль остаётся нулём и хвост затухает без сброса; снапшот без добавления пакета и точное продолжение; перенос события между узлами меняет рисунок и стереообраз; лимиты при gain 0.04; отказ при чужом контексте, подсказки реестра, `display()`; тайминг p99.
- `BenchTests` (1) — headless: оверлеи и панель обеих сторон, выбор Б.
- `CatalogTests` (2) — сцены/периоды/события по ТЗ и preflight; сборка каталога, точный реплей, Continue побитово, Notes с гипотезой, защита от перезаписи.

## 4g. `tests/test_n3_tuned_events.py` — N3: поле настраивает резонансы, события их возбуждают (16)

ТЗ `memory/req-network-combined-n3-2026-09-16.md`; fixtures `memory/research/network-n3-preflight-2026-09-16.json`; временные файлы в `artifacts/_n3_tests/`. Гейт 1k в `check.py`.

- `test_kernel_matches_scalar_reference_in_both_modes` — вся цепочка события → задержки → банки → HP → выход против независимого скалярного расчёта формул ТЗ (допуск 1e-12) в режимах Fixed и Field, с перекрывающимся пакетом и перестройкой на границе блока; состояния банков и частоты совпадают.
- `test_kernel_matches_scalar_reference_through_a_decay_ramp` — ручное изменение Decay: линейная рампа r за 882 отсчёта, разрезанная границами блоков, против референса с посэмпловым r; приземление точно на цель.
- `test_network_states_equal_n2_and_do_not_depend_on_the_tuning` — при одной истории поля состояния сети (кольца, фильтры, импульсы, счётчик) А и Б N3 равны между собой и `ca_event_network` при rho 0.88 побитово; пакеты одинаковы; выходы банков различаются.
- `test_frequencies_follow_the_a_law_and_fixed_mode_is_ratio_one` — частоты Б == `gutter_field_periodic` (scale 1, depth 1) на том же поле и == формулам ТЗ; в Fixed все множители 1 и частоты = float32(base); таблицы cos/sin = `math.cos/sin` углов; переключение режима меняет только частоты (состояния и сеть не тронуты) и не создаёт события.
- `test_event_semantics` — начальное поле сравнивается с нулём один раз; то же поле — нет пакета; рождение и смерть положительны и складываются; несколько правок внутри блока сливаются (возврат — без события); параметры — не событие; reset = нули и один пакет; `exc` игнорируется.
- `test_control_probes_hold_frequencies_or_hold_impulses` — (а) Fixed на двух полях: частоты равны, сети различны, отклики различны; (б) одно поле, Fixed против Field: сети равны, частоты различны, отклики различны; проба воспроизводима.
- `test_silent_retune_makes_no_sound_and_an_excited_tail_keeps_its_state` — на нулевом состоянии переключения режима дают ровно 0; возбуждённый хвост при перестройке сохраняет Re/Im и сеть, продолжает звучать и затухает без новых событий (< −100 dBFS через 6 с).
- `test_decay_values_and_ramp` — `r = 10^(−3/(SR·T60))`; init начинает сразу с настроенного r; рампа 882 отсчёта, значение до первого отсчёта не меняется, приземление точное.
- `test_snapshot_mid_tail_and_mid_ramp_restores_exactly_without_a_packet` — снапшот посреди хвоста и рампы через `snapshot.py`: продолжение побитово 12 блоков, restore не добавляет пакет; отказ при чужом поле, версии модели, частотах не по закону поля, чужих параметрах, ином rho, чужом engine_id.
- `test_limits_full_toggle_random_edits_and_decay_ends_at_max_gain` — при gain 0.04: всё поле каждый блок, случайное поле каждый блок, правки 5×5, движение ручек (Decay между концами, режим) при T60 0.2 / 1.5 — конечно, без клипа (движок без шагов КА).
- `test_rejects_other_contexts_registry_hints_and_display` — отказ при SR 48000 / блоке 256; спецификации параметров, слова режимов Fixed/Field, оверлей (8 подписей, 16 окружностей, без линий), ключи `display()`, константы rho 0.88 / ×4.
- `test_engine_timing_budget` — p99 блока движка < половины бюджета.
- `BenchTests` (1) — headless: 15 движков → 5 рядов, оверлей, слова режимов, `display` обеих сторон (`tuning` False/True), выбор Б, отрисовка панели N3.
- `CatalogTests` (3) — клетки сборщика == fixtures preflight, периоды 5 / 14 / 128, glider 4 события на шаг, сцены на диске == документам, параметры сторон, команда замены = один `set_cells` всех 1024 клеток на 12 с, bat без `--live`; замена — одна запись журнала на первой границе блока ≥ 12 с, поле = Tumbler после причитающихся шагов, звук и часы КА не сброшены; сборка каталога, точный реплей, Continue побитово, Notes с гипотезой, защита от перезаписи.

## 4h. `tests/test_n4_object_resonators.py` — N4: резонаторы фигур и круговые детекторы (28)

ТЗ `memory/req-object-resonators-n4-2026-09-16.md`; fixtures `memory/research/object-resonators-n4-preflight-2026-09-16.json` (`cases`); временные файлы в `artifacts/_n4_tests/`. Гейт 1l в `check.py`.

- `GeometryAndSpectrumTests` (7) — blinker λ={0,1,3} → √λ = [1, √3]; спектр и ключ формы инвариантны к переносу/повороту/отражению и переносу через шов (горизонталь/вертикаль/угол тора); у glider две спектральные фазы, повторяющиеся через поколение; непериодические отношения частот == `casynth_core.map_laplacian(harm=0, fullshape=True)` (glider, приёмник N4.2, блок 5×5); центр (11.7647, 11.7647) и R=5.32962 приёмника N4.2, круг содержит свои клетки и блинкер, 92 клетки; R=0 и отсутствие мод у одиночной клетки; центр glider на шве — развёрнутое среднее; неоднозначный центр кольца через тор (наименьшая координата / ближайший к предыдущему), `periodic_mean` на [0,16] и через шов; правила сопоставления (glider 140 поколений — один id; сдвиг одиночной клетки на 1 — продолжение, на 3 — нет; добавление клетки и появление блока 5×5 — старый id сохранён; слияние → два хвоста + новый; распад → хвост + два новых); спектр побитово одинаков для любого размещения одной фазы (канонические клетки), кэш не меняет результат.
- `ExcitationTests` (5) — из нуля без событий — ровно 0; стартовый пакет e=3 для blinker (a=3/5); то же поле / параметры / `update_field` тем же полем — без пакета, затем затухание < 1e−3 за 3 с; blinker: 2 рождения по новой маске + 2 смерти по старой = 4 в обоих режимах; поле очищено → фигура в хвост без пакета; N4.2: приёмник Own e=0, Disk e=4 каждое поколение, blinker e=4, размер/моды приёмника неизменны; blinker перенесён за круг → старый в хвост, новый id, у приёмника e=3 однократно (смерти в Disk) и далее 0; несколько правок в блоке с возвратом — не событие; переключение детектора — не удар; **Radius x**: при ×0.5 круг приёмника (R 5.33 → 2.66) не держит ни блинкер, ни все свои клетки (стартовый пакет = число своих клеток внутри), Disk-приёмник и blinker (R 1 → 0.5) дают 0; при ×3 приёмник по-прежнему слышит 4; Own игнорирует множитель; одиночная клетка — R 0 при любом множителе; смена множителя — не удар и не событие (счётчик изменений поля не растёт).
- `ScalarReferenceTests` (2) — весь семпловый путь (импульс N2, банки, веса, панорама, HP20, ×0.5, gain) против независимого скалярного расчёта ≤ 1e−12 через рампы decay / gain / весов (смена числа мод 5→6 и 3→2, недоведённые моды звучат) / панорамы, с инжектами в разных блоках; `decay_r`, рампа gain 882 отсчёта с точным приземлением, `pan_of` на краях и с клипом.
- `LifeCycleTests` (4) — glider 12 с при 16 gen/s: один id, всегда 1 звучащая фигура; дорисованные клетки сохраняют id (3 → 5 клеток, 4 моды, рампа весов, e=2); Tumbler 6 с — конечно, id уходят в хвосты, drops=0; очистка всего поля → 2 хвоста и 0 активных, быстрое повторное добавление → новые id и стартовый пакет, через 3 с тишины хвосты освобождены и состояния ровно 0; 48 blinker'ов → «звучат 24 из 48», `unvoiced_blocks` растёт, 6 циклов clear/re-add → 96 хвостов и `evictions` > 0, конечно; освободившийся слот уходит ждущей фигуре без пакета; плотное случайное поле с полным переключением каждый блок на концах ручек (55/0.2, 880/1.5, оба детектора) при gain 0.04 — конечно, без hard drops.
- `SnapshotTests` (2) — снапшот посреди хвоста, рампы r, рамп весов/панорамы, с ожидающим полем, через `snapshot.py`: 30 блоков побитово и равный `display()`; снапшот без `radius_mul` (до появления ручки) восстанавливается с 1.0; отказ при чужой модели, подправленном спектре, чужом ожидающем поле; `DemoRunner` обеих сцен с командами decay/scale: экспорт через файл, 60 блоков A/B/monitor побитово и равные `display`.
- `LevelsTimingAndContractTests` (4) — уровни обеих сцен == preflight (RMS в пределах 0.1 dB, пики 1e−3), пики < 0.1; p99 блока обеих сторон < 7.98 ms на обеих сценах (с шагами КА); отказ при SR 48000 / блоке 256 / моно; дефолты, слова Own/Disk, отказ scale 1000, оверлей — только подпись; ключи `display()` и фигуры (17 клеток, R, f_low = 220·√λ₁), JSON-сериализуемость; сохранённая запись N3 воспроизводится текущим кодом (skip без WAV).
- `BenchTests` (1) — headless обе сцены: 15 движков → 5 рядов; `display` A без фигур (N3) / с фигурами; клетка фигуры на экране окрашена цветом её id; пауза; отрисовка фигур и панели по синтетическому `display` (круг через угол тора, режим Own).
- `CatalogTests` (2) — сцены на диске == документам, клетки == preflight `cases`, 6 gen/s, 12 с, стороны (N3 Field / N4 Disk; N4 Own / N4 Disk), гипотезы; фигуры N4.2 72 поколения не сливаются и эволюционируют как размещённые отдельно; bat без `--live`; сборка каталога, точный реплей, Continue побитово, Notes с гипотезой, защита от перезаписи.

## 4i. `tests/test_objects_event_source.py` — Objects: источник событий и возбуждение по месту рождения (13)

ТЗ `memory/req-objects-event-source-modal-2026-09-17.md`; fixtures `memory/research/objects-event-source-modal-preflight-2026-09-17.json`; сцены `demos/oes_{e1,m1}.json` (сборщик `demos/build_objects_event_source.py`); временные файлы в `artifacts/_oes_tests/`. Гейт 1o в `check.py`.

- `RegistryTests` (1) — ручки `events` Both/Births/Deaths и `excitation` Uniform/Birth position, дефолты 0/0, набор без ключей = дефолты, все 11 сцен Objects (n4/ol/ora/oes) грузятся; подпись оверлея; `position_supported` только при Births+Own+Laplace+full.
- `EventSourceTests` (3) — E1 (Octagon II p5): один банк на все 15 переходов, пакеты каждого поколения == независимые маски (рождения в текущих клетках / смерти в предыдущих), последовательности Births [8,16,0,8,8] и Deaths [0,16,8,0,16] трижды, Both = сумма, стартовый пакет 16/16/0, per-mode состояния при Uniform ровно 0; ручное рождение/смерть на неподвижном блоке (Births [1,0], Deaths [0,1], Both [1,1]); стирание последней фигуры: в Deaths хвост уносит ровно один удар (zf = остаток + a, npulse = моды, кормит ~4 блока и монотонно затухает, потом npulse 0, а отклик звучит), в Births/Both хвост без возбуждения; при занятом пуле хвостов и слиянии Jam ударённый банк всё равно попадает в хвост (вытеснение самого тихого), новые банки в Deaths без пакета, продолжающая фигура — смерти своей прежней маски.
- `BirthPositionTests` (3) — M1 (Jam p3) 40 поколений бок о бок: у каждой фигуры равные частоты, веса, моменты пакетов и a на обеих сторонах, один трекер; у Б sum b² = m у каждого пакета, коэффициенты 11-/16-/3-клеточных банков == preflight (1e−9); звук до первого перехода равен (1e−12), после — различается. Закон: одиночное рождение (8,9)/(8,12) на 11-клеточной геометрии == preflight `input_weights`, расстояние профилей 2.3647; ноль участия → b=0, пакета нет; ранги кратных групп E1 [2,8,1], инвариантность к знаку и повороту базиса (1e−12); перенос (через шов) и поворот 90° дают тот же b; рождение всей фигуры = единичный пакет. Неподдерживаемые сочетания (Disk / Both / Deaths / Figure / full 0) — ни одного пакета, тишина, счётчик `unsupported_packets`, `position_supported` False.
- `KernelTests` (2) — ядро против независимого скалярного эталона v4 (пакеты по модам, общий пакет, Attack 0→4→0 с рампой, gain, последний удар хвоста с правилом порога 1e−7) ≤ 1e−12; отклики двух пакетов складываются (1e−12), веса не меняются ударом, нулевой коэффициент оставляет моду в нуле, Attack 4 ms смягчает первый семпл per-mode пакета в >10 раз.
- `StateTests` (2) — смена Events/Excitation на неподвижном поле — без пакета (блоки побитово равны двойнику); снапшот v4 посреди per-mode импульса и кормимого хвоста Deaths продолжает точно (через переходы с распадом), отказ при npulse у активного слота и при v3-версии с моделью v4; снапшот v3 (без новых массивов, счётчики 5) принимается как Both/Uniform побитово.
- `SceneTests` (2) — обе сцены: клетки == preflight, стороны отличаются только `events` (E1) / `excitation` (M1), настройки preflight, side_gain, диск == документ, Continue раннера с 6 с 200 блоков побитово, без клипа, `display` Б; headless стенд: 16 строк панели, окно ≤ 880 px при низком столе, кнопки Excitation 92 px / Events 60 px внутри панели, строка статуса неподдерживаемого сочетания красная.

## 5. `tests/golden/golden_master.py` — golden-master аудио gol_synth

Фиксированная детерминированная программа через реальные функции движка: все 5
движков, ADSR (pluck/pad), большие формы (> 8×8 и децимация), рождение/смерть мод с
кроссфейдом хвостов → сравнение байт-в-байт с `golden_master_ref.npz`
(sha256[:16]=cd3126907ad13b6c). Переблагословение только `--save` после согласованного
слышимого изменения.

## 6. UI-кадр + smoke (`check.py`)

`gol_synth.py` с `CASYNTH_DUMPFRAME` в dummy-видео/аудио: импорт, инициализация pygame и
аудио, ровно один кадр → сравнение пиксель-в-пиксель с `tests/golden/ui_frame.png`.
Легитимное изменение UI → `python check.py --bless-ui` в том же изменении.
