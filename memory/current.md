# Current Implementation State

_Снапшот живого состояния. Полная история реализации — `memory/archive/current-history-2026-06-16.md` и `memory/log/`; решённые вопросы — `memory/archive/questions-resolved.md`._

## Implemented (снапшот — детали в log/ и archive/)

**Прототипы и ядро**
- **`gol_life_synth_laplacian.py`** — основной прототип: каждый связный объект → голос,
  спектр = граф-Лапласиан формы (√λ → частоты мод), негармонический «металлический» тон.
  sounddevice callback-стрим (gapless, развязан с 60 fps). Живые крутилки тембра
  **partials/spread/alpha/release** (экранные слайдеры, меняются на лету; partials =
  число партиалов на объект, 1–20, дефолт 12 — твик из laplacian_explainer.html,
  пишется в лог сессии). `MAX_MODES_PER_OBJ`=20 — потолок/ёмкость слотов. Аудио-движок:
  active-слот + пул tail-слотов с бесшовным переносом хвоста (overlap-кросс-фейд без
  щелчка, хвосты наслаиваются в пэд). Громкость пре-клип + level/clip-метр.
- `gol_life_synth.py` — **legacy v1**: size→harmonic, pygame.mixer.
- `casynth_core.py` — общая либа алгоритмов маппинга форма→(freqs,amps): пять `map_*`
  (fft2d/walsh/random/laplacian/granulo), numpy/scipy без pygame.
  `map_laplacian(patch, f0, n, spread=0, alpha=1)`; низшая взятая мода всегда = f0.
- `mapping_bench.py` — discover-стенд (матрица осцилляторы × приёмы).
- `patterns.py` — 29 паттернов в 9 категориях (верифицированы).
- `tests/test_casynth_core.py` — 14 тестов инвариантов (без pytest-зависимости).

**Перцептивные оси (заведены)**
- size (area) → harmonic number k (v1) / набор Laplace-мод (v2)
- density (bbox fill) → amplitude (1/k^0.7 rolloff)
- centroid X → stereo pan
- Voice cap `MAX_VOICES = 24`; несущая — экранное пианино C3–C5 (latched)

**UI:** рисование/стирание клеток, play/pause/step/random/clear/скорость, сайдбар
паттернов с drag-and-drop (ghost-превью), легенда гармоник, level/clip-метр.

**На ревью Researcher:** алгоритмы `casynth_core.py` на соответствие замыслу; крутилки
spread/alpha/release + флаги DEV-1 (клиппинг яркости) / DEV-2 (хвосты) — см. `questions.md`.

## Known issues
- Segmentation не wrap-aware: объект на шве тора временно распадается на два голоса.
- Нет трекинга идентичности объектов между поколениями (нет birth/death-событий, коллизий).
- Заведено ~3 из ~8 спроектированных перцептивных осей — диапазон тембра узок.
- Клиппинг при плотном поле + яркая настройка — ручной headroom (метр+громкость, DEV-1);
  финальное дизайн-решение (нормировка vs gain vs рабочая зона) за Researcher.

## In-flight
Живые крутилки spread/alpha/release реализованы (см. Implemented). Следующий шаг —
**прослушивание Пользователем**: пройти A (spread) и D (alpha) на одной форме, развести
тёмный↔яркий без сдвига высоты; оценить release как пэд-рычаг. По итогам — решение
Researcher по DEV-1 (клиппинг яркости: MASTER_GAIN vs нормировка vs рабочая зона) и,
при нехватке пэда, реквест на tail-pool (DEV-2). Шаг B (формозависимые амплитуды через
собств. векторы) — следующим реквестом после прослушивания A/D.
P0-A/P0-B/P2 кросс-фейда ранее зафиксированы.

## Mapping bench — discover завершён (2026-06-15)
- `mapping_bench.py` — ревизия завершена, все P0/P1 прошли researcher-ревью.
  Матрица 4×5: осцилляторы Blinker, Toad, Beacon, Clock (Clock_C период-2) × маппинги
  2D-FFT, Walsh-2D (sequency), Random, Laplacian (негармонический), Granulo.
- **Итог batch-прослушивания (Пользователь):** FFT/Walsh/Random/Granulo на мелком
  зоопарке — различие есть, но **невыразительное**; Granulo вырожден. **Laplace выделился**
  — органичный металлический тон, самостоятельная фишка → выбран для второго прототипа.
- Стенд остаётся discover-инструментом; выбракованные приёмы НЕ выброшены, а
  законсервированы (профили — в `decisions.md`).

## Open research — профилирование маппингов (параллельная ветка, не блокирует прототип)
Стратегия: **не выбрасывать тупиковые ветки**, набрать зоопарк приёмов и собрать
**профиль/характер каждого** на разных режимах автомата → карта «какой приём для чего
годится». Рамка и текущие профили 5 приёмов — в `memory/decisions.md`.
- Ближайшая измерительная задача: прогнать законсервированные приёмы (FFT/Walsh/Granulo)
  на **крупных/плотных полях** — проверить гипотезу «шумный на малом → структурный на
  большом». Формат стенда под крупные режимы не заточен (отдельная задача).
- Разные приёмы могут управлять РАЗНЫМИ параметрами синта (тембр/огибающая/модуляция/
  пространство) — карта это и должна развести.

## Candidate next steps
See `memory/decisions.md` (индекс вверху) for prioritisation by Researcher.
- Remaining perceptual axes: jaggedness → distortion/inharmonicity, activity → tremolo,
  symmetry → consonance, order/chaos → noise blend, elongation → detune/vibrato
- Wrap-aware segmentation (union-find across toroidal seam)
- Object identity tracking (voice allocation with birth / death / merge / split events)
