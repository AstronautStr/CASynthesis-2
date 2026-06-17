# 2026-06-16 — Мульти-движковый прототип `gol_synth.py`

## Запрос (Пользователь)
Скопировать `gol_life_synth_laplacian.py` → `gol_synth.py`, добавить остальные движки
озвучивания со стенда (все 5: FFT/Walsh/Random/Laplace/Granulo), сложить движки в одну
библиотеку, дать в приложении селектор движка. Параметры part/spread/alpha — атрибуты
конкретного движка: при смене движка интерфейс показывает его атрибуты; выбранные
значения помнятся по каждому движку (вернулся — тот же звук). Release — ручка синта в
целом, не движка. UI селектора — **ряд вкладок-кнопок** (выбор Пользователя в плане).

## Что сделано
1. **`casynth_core.py` (+реестр движков, аддитивно — поведение `map_*` не тронуто):**
   - `ENGINES` — список dict `{id, label, fn, params}`; `params` = tuple-спеки
     `(arg, label, lo, hi, integer, default)`, где `arg` = имя kwarg для `map_*`.
     Laplace: n/spread/alpha; FFT/Walsh/Random/Granulo: только n. `ENGINE_BY_ID` — индекс.
   - Чистые данные, без pygame/UI-форматирования (библиотека остаётся dependency-light).

2. **`gol_synth.py` (копия лапласиан-прототипа + обобщение):**
   - `analyse(grid, f0, engine_id, params)` — берёт `fn` из `ENGINE_BY_ID`, зовёт
     `fn(patch, f0, **params)` с `n`, ограниченным `MAX_MODES_PER_OBJ`. Конвейер
     SlotPool/render не менялся.
   - Состояние: `state['engine']` + `state['engine_params']={engine_id:{arg:val}}`
     (инициализируется дефолтами реестра → память по движкам). `release_ms` остался
     synth-wide, вне `engine_params`.
   - Динамическая панель: `rebuild_ctrls()` собирает слайдеры из `params` активного
     движка + всегда release последним. `set_ctrl`/`ctrl_value` учитывают `scope`
     ('engine' → пишет/читает `engine_params[engine][arg]`; 'synth' → `state[release_ms]`).
   - Ряд вкладок движков в нижней полосе тулбара (TOOLBAR_H 96→124); клик по вкладке →
     `state['engine']=id; rebuild_ctrls()`. Кнопка «Lib» отделена от имени движка.
   - Запись сессии: `replay`-кортеж читает spread/alpha с дефолтами для движков без них;
     добавлен parallel-массив `replay_engines` (id движка по кадру) в npz.

3. **`tests/test_casynth_core.py`:** +2 теста (целостность реестра: уникальные id,
   callable fn, дефолт в [lo,hi], наличие 'n'; вызов каждого движка с дефолтами →
   конечные freqs/amps длины n).

## Проверка
- Юнит-тесты: 16/16 PASS (14 старых + 2 новых).
- Headless-smoke `gol_synth.py`: 3 c без выхода, без трейсбэков.
- Функциональный проб: `analyse` на случайном поле для всех 5 движков → по 24 голоса,
  freqs/amps конечны; laplacian n=12 (старт-дефолт), гармонические n=16; низшая = f0.
- Регрессия: старт=laplacian с part=12/spread=0/alpha=1 → `map_laplacian` зовётся
  идентично → бит-в-бит как `gol_life_synth_laplacian.py` (опирается на core-тест
  backward-compatible).

## Заметки / на ревью Researcher
- Это совпадает со стратегией «не отбрасывать ветки; профилировать» (decisions.md
  2026-06-15): живой селектор = способ прогнать законсервированные FFT/Walsh/Random/
  Granulo на крупных/плотных полях.
- Дефолты движков теперь живут в реестре `ENGINES`; старые константы `SPREAD_DEFAULT`/
  `ALPHA_DEFAULT`/`N_PARTIALS_MIN`/`ALPHA_MIN/MAX` в `gol_synth.py` стали неиспользуемыми
  (оставлены как документация диапазонов; источник истины — реестр).
- Faithful-replay для не-Laplace движков частичный (spread/alpha дефолтят в 7-колоночном
  `replay`; id движка пишется отдельно). Полная мульти-движковая репродукция — отдельная
  задача, если понадобится для проб артефактов.
