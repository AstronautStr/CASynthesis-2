# 2026-07-05 — REQ Динамическое возбуждение (`dyn`)

## Цель
Реализовать ручку `dyn` для `map_laplacian`: амплитудный отпечаток `|<e,phi_k>|` пересчитывается
от маски изменившихся клеток поколения, открывая моды, структурно запертые у симметричных форм.
Дизайн: `decisions.md` 2026-07-05. Задача: все 7 «Требуется», критерии A–G.

## Что изменено

### `casynth_engine.py`
- Добавлен `W_WOUND = 0.7` (константа вверху).
- Добавлена `events_field(prev, new) -> float H×W`: рождения +1.0; живые 8-соседи смертей
  +W_WOUND (используется `ndimage.convolve` с 8-связным ядром `_S8`, результат > 0 & live_in_new);
  вклады суммируются (рождение + кромка = 1.7).
- Добавлена `_crop_like_extract(data, template, size)`: float-версия `extract()` — берёт окно
  `size×size` вокруг центроида живых клеток `template`, заполняя нулями за границей.
- Расширена `analyse(..., exc=None)`: при `exc is not None` и `'dyn' in kwargs`:
  - Per-object: `exc_bbox = exc[bbox] * sub` (маскирование чужих клеток).
  - fullshape-ветка: `exc_for_fn = exc_bbox`; extract-ветка: `_crop_like_extract(exc_bbox, sub)`.
  - `call_kwargs = {**kwargs, 'exc': exc_for_fn}` — не мутирует общий `kwargs`.
  - Гармонические движки (без 'dyn' в params) exc не получают.

### `casynth_core.py`
- `map_laplacian` — добавлены параметры `dyn=0.0, exc=None`:
  - Fullshape-децимация применяется к обоим: `patch = patch[::step,::step]`,
    `exc = exc[::step,::step]` (геометрическое выравнивание).
  - Shape-ветка: при `dyn > 0 and exc is not None`: `e_ev = exc[patch>0]`;
    юнит-нормировка `e_hat_deg = e/(||e||+1e-12)`, `e_hat_ev = e_ev/||e_ev||`;
    при `||e_ev|| < 1e-9` fallback на `e_hat_deg`; иначе `e = (1-dyn)*e_hat_deg + dyn*e_hat_ev`.
  - `dyn=0` или `exc=None` — блок пропускается (бит-в-бит).
- Реестр Laplacian: добавлен `('dyn', 'dyn', 0.0, 1.0, False, 0.0)`.

### `gol_synth.py`
- Импорт `events_field` из `casynth_engine`.
- Добавлена переменная `exc_field = None` рядом с `grid`.
- Оба пути step (auto-tick и кнопка): `prev = grid.copy(); grid = step(grid); exc_field = events_field(prev, grid)`.
- Вызов `analyse(grid, base_f0, state['engine'], _ep, exc=exc_field)`.

### `casynth_session.py`
- Импорт `events_field`.
- `replay_session`: `exc_field` восстанавливается per-frame из `step_grids_all/step_gens_all`
  через `np.searchsorted(step_gens_all, frame_gen, side='right') - 1`; между шагами держится.
  `events_field(prev_step_grid, last_step_grid)` — та же функция, что в живом цикле.
- Передаётся в `analyse(grid, f0, engine, params, exc=exc_field)`.

### `tests/test_casynth_core.py`
14 новых тестов (критерии A–G + доп. инварианты):
- `test_events_field_blinker` — точечный критерий F: (0,1)=1.7, (1,1)=0.7, (2,1)=1.7.
- `test_events_field_empty` / `_birth_only` / `_wound_only` — граничные случаи.
- `test_laplacian_dyn_zero_bitexact` — критерий A.
- `test_laplacian_dyn_fallback_no_exc` / `_zero_exc` — критерий B.
- `test_laplacian_dyn_scale_invariant` — критерий D.
- `test_laplacian_dyn_does_not_affect_freqs` — extra invariant.
- `test_laplacian_dyn_symmetry_antisymmetric_mode` — критерий C: dyn=0 силенсит lambda=1 мод
  blinker (proj=0); asymmetric exc открывает его (amps ordering flips).
- `test_laplacian_dyn_alignment_rotation` — критерий E: sorted amps invariant под rot/flip,
  fullshape=False и True (малые формы).
- `test_laplacian_dyn_alignment_fullshape_large` — критерий E fullshape: проверяет что
  auto-decimation == manual-decimation (implementation invariant; NOT rotation covariance
  of arbitrary exc, which can't hold for lattice sampling).
- `test_laplacian_dyn_analyse_no_leak` — критерий E уровень analyse: exc на чужом объекте
  в пересекающемся bbox не меняет первый голос.
- `test_laplacian_dyn_registry_integrity` — реестр: spec valid, дефолт dyn=0 бит-в-бит.

## Решения при реализации

**events_field wound formula.** Blinker test {(0,1):1.7, (1,1):0.7, (2,1):1.7} показывает, что
вклад W_WOUND = +0.7 per WOUND CELL (не per death). `ndimage.convolve(deaths, _S8) > 0 & new_b`
— двоичная дилатация: каждая живая клетка рядом с хотя бы одной смертью получает +W_WOUND раз.

**_crop_like_extract vs extract.** Для fullshape=False exc прокидывается как exc_bbox (bbox-окно *
маска sub), затем `_crop_like_extract` извлекает size×size окно по центроиду живых клеток sub.
Это зеркалит логику `extract()` в casynth_core, гарантируя что `exc_for_fn[patch>0]` после
`extract()` на patch даёт выровненные значения.

**has_dyn check.** `'dyn' in kwargs` (kwargs = dict(engine_params)) — True для Laplacian, False
для harmonic engines. Предотвращает передачу `exc` kwarg harmonic map_* (TypeError).

**Symmetry test.** Blinker с симметричным exc [1,1,1]: exc пропорционален DC-моде phi_0 =
[1,1,1]/sqrt(3), которая ортогональна всем звучащим lambda>0 модам. Проекция на все сounding
modes = 0 -> mx_proj=0 -> fallback на rolloff. Тест проверяет: dyn=0 -> lambda=1 мод silent
(proj=0), dyn=1+asymmetric exc -> lambda=1 мод открыт (amps ordering flip).

**Alignment fullshape large.** Rotation invariance для arbitrary random exc after [::step,::step]
decimation CANNOT hold (rot90(X)[::s,::s] != rot90(X[::s,::s])). Тест заменён на invariant
«manual pre-decimation == auto-decimation» (implementation correctness).

## Результаты верификации (первый раунд — до code-review)

| Проверка | Результат |
|---|---|
| `python tests/test_casynth_core.py` | 44/44 PASS |
| `python artifacts/_golden_master.py` | PASS byte-exact (sha256[:16]=cd3126907ad13b6c) |
| Smoke (SDL dummy) | OK (no errors) |
| Replay criterion G (P5 asymmetric exc) | dyn=0.5 vs dyn=0: max|diff|=1000 LSB |

## Доработки после code-review (2026-07-06, FIX-A..FIX-E)

### FIX-A: Запись истинного prev_grid на каждом step

Три кейса неточной реконструкции exc в replay:
1. Кнопочный step не записывался в `rec['steps']` вовсе.
2. Первый шаг: в replay prev=zeros, вживую prev=нарисованное поле.
3. Правки поля между шагами теряются (replay использовал предыдущий шаг как prev).

Решение: 4-элементный кортеж `(gen, grid_after, note, prev_grid)` в обоих местах записи.
- `gol_synth.py do()`: кнопочный step добавляет `rec['steps'].append(...)` с true prev.
- `gol_synth.py` авто-цикл: `prev_grid` добавляется 4-м элементом (уже скопирован).
- `_dump_session`: извлекает step_prevs из 4-го элемента; legacy 3-кортежи → пустой массив.
- В npz: `step_prevs` (K×H×W uint8, или 0×H×W для legacy).

### FIX-B: Тестируемый хелпер реконструкции exc

`_exc_for_frames(step_gens, step_grids, step_prevs, frame_gens)` в `casynth_session.py`:
- has_prevs: len(step_prevs)==len(step_grids) → использует step_prevs[k] как true prev.
- Legacy (step_prevs=None или len=0): step_grids[k-1] (k=0 → zeros).
- `replay_session` теперь вызывает хелпер вместо inline-логики.
- Тест `test_exc_for_frames_live_cycle_parity`: 3 шага с draw/edit/random;
  новый путь воспроизводит живой цикл точно; legacy-путь отличается для gen=1,2. 48/48 PASS.

### FIX-C: Тест симметричной пары (NOT fallback — реальная проекция)

`test_laplacian_dyn_symmetric_pair_real_projection`: exc={endpoint1, endpoint2} на blinker.
e_ev=[1,0,1]/√2; <e, φ_1>=0, <e, φ_2>=1/√3 → mx_proj>0 → НЕ rollback → amps[:2]=[0,1.0].

`test_laplacian_dyn_uniform_exc_rollback`: uniform [w,w,w] → DC-мода → все sounding proj=0 →
mx_proj=0 → rollback rolloff → amps=[1.0,0.5]. Отличается от dyn=0 ([0,1]) и симм.пары.

### FIX-D: Паритет _crop_like_extract↔extract

`test_crop_like_extract_matches_extract`: для форм в PATCH_SIZE-окне:
`_crop_like_extract(exc, sub)[extract(sub)>0] == exc[sub>0]` (row-major). PASS на blinker и ell.

### FIX-E: no-leak тест с реально пересекающимися bbox

Геометрия: A=8-клеточная Г-форма (bbox [5,10]×[5,7]); B=3-клеточная полоска (8,7)-(10,7)
ВНУТРИ bbox A, 8-несвязна (col gap=2>1). exc: только (8,7)=10 (асимметрично).
Assertions: A unchanged (atol 1e-9) ✓; B changed (diff>1e-4) ✓.

### FIX-F: Step-serial matching (non-monotonic gens after clear)

**Проблема.** `clear()` делает `state['gen']=0`. При записи step_gens становятся немонотонными
(пример: [1..30, 1..10]). `np.searchsorted` требует сортированный массив и на несортированном
тихо возвращает мусорные индексы → exc-реконструкция коррумпируется для обеих половин.

**Решение — serial-path.**
1. **Write-side (`gol_synth.py`):** 6-й столбец `rec['frames']` = `len(rec['steps'])` на момент
   записи кадра (`n_steps_done`). Комментарий-схема обновлён:
   `[dt_ms, gen, n_voices, n_rendered, underrun, n_steps_done]`.
   Пустышка `np.zeros((0,5))` → `(0,6)` в `_dump_session`.
2. **Read-side (`casynth_session._exc_for_frames`):** добавлен `frame_serials=None`.
   При наличии: прямая индексация `k = serial - 1` (безопасна при немонотонных gens).
   `serial=0` → `None` (до первого шага). `k >= K` → safety-clamp. Legacy-фоллбек (gen-путь с
   searchsorted) остаётся для 5-колоночных записей без clear.
3. **`replay_session`:** извлекает `frame_serials = frame_data[:,5].astype(int)` при
   `frame_data.shape[1] >= 6`; передаёт в `_exc_for_frames(frame_serials=frame_serials)`.

**Поведение на clear (live).** `exc_field` НЕ сбрасывается (держится). С serial-матчингом
кадры после clear до нового step получают старый serial → старый exc. Воспроизводит live-поведение.

**Тест `test_exc_for_frames_clear_scenario` (FIX-F):** step_gens=[1,2,1,2] (non-monotonic);
frame_gens=[2,2,2,2] (все один gen → gen-путь неоднозначен); frame_serials=[2,2,4,4].
Serial-путь воспроизводит exact exc для обеих половин; также верифицировано, что gen-путь
расходится (searchsorted([1,2,1,2], 2, side='right')-1 = 3 для всех кадров → pre-clear кадры
получают exc_post вместо exc_pre).

## Результаты верификации (после доработок)

| Проверка | Результат |
|---|---|
| `python tests/test_casynth_core.py` (FIX-A..E) | 48/48 PASS |
| `python tests/test_casynth_core.py` (FIX-F, итог) | **49/49 PASS** |
| `python artifacts/_golden_master.py` | PASS byte-exact (sha256[:16]=cd3126907ad13b6c) |
| Smoke (SDL dummy) | OK (no errors) |

## Что ещё нужно (за Пользователем)
- Слуховое A/B: shape=1, глайдер и R-пентамино, dyn 0↔1 — слышна ли «тембровая походка».
- Blinker как контроль (dyn не должен сильно менять звук симметричного осциллятора).
