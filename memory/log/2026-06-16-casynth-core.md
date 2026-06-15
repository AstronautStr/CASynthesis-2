# 2026-06-16 — Вынос алгоритмов маппинга в `casynth_core.py`

## Задача
Отделить алгоритмы маппинга «форма автомата → звук» в отдельную либу, которую
может валидировать Researcher и которая тестируется отдельно. Реализация решения
`decisions.md` 2026-06-15 «Общая библиотека `casynth_core.py`».

## Сделано
- Создан `casynth_core.py` (numpy/scipy, без pygame): `extract`, `_norm`, пять
  `map_*` (fft2d/walsh/random/laplacian/granulo), предвычисленные структуры,
  константы `SR`/`PATCH_SIZE`/`N_PARTIALS_DEFAULT=20`.
- Устранена обнаруженная **дивергенция числа мод** (стенд `K_MAX=20` vs прототип
  `MAX_MODES_PER_OBJ=8`): число партиалов стало явным параметром
  `map_*(patch, f0, n=20)`. Алгоритм единый; стенд передаёт 20, прототип — 8.
  Расхождение значений вынесено вопросом Researcher'у (`questions.md`,
  [OPEN] 2026-06-16); default — сохранены текущие значения (чистый рефактор).
- `mapping_bench.py` и `gol_life_synth_laplacian.py` переведены на импорт из core;
  локальные копии `map_laplacian`/`_extract` удалены. Канон Laplace взят из
  прототипа (новейшая версия с фиксами Researcher'а); в снапшоте обе копии были
  алгоритмически идентичны (различие только в n).
- `_diag_vectors.py` обновлён (`mb._extract` → `mb.extract`).
- `tests/test_casynth_core.py` — 10 тестов инвариантов, без pytest-зависимости.

## Проверки (все PASS)
- Стенд: `_diag_vectors.py` до/после — diff пуст (бит-в-бит).
- Прототип: `_render_probe.py compare before after` — cosine=1.000000, 0.00%,
  peak/клики идентичны.
- Тесты: 10/10.
- Smoke обоих приложений headless + функциональный прогон путей (analyse +
  render_chunk; build fa-матрицы).
- Изоляция: `import casynth_core` без pygame.

## Границы (не тронуто, по decisions.md)
Движок (`render_chunk` / `render_chunk_laplacian` / `SlotPool`), UI, sounddevice,
size→k маппинг базового `gol_life_synth.py`. `SR`/`PATCH_SIZE` в прототипе оставлены
локально (config движка); алгоритм маппинга единый в core.

## Следующий шаг
Ревью Researcher: алгоритмы в `casynth_core.py` на соответствие замыслу
(вынос не изменил поведение `map_*`/`_extract`).
