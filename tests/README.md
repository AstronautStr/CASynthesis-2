# Каталог автотестов CASynth (снимок 2026-09-13)

Всё запускается одной командой `python check.py` (8 гейтов). Ниже — каждый тест и
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
