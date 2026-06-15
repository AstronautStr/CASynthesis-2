# Current Implementation State

_Last updated: 2026-06-15 (построен второй прототип gol_life_synth_laplacian.py)_

## Implemented
- Continuous phase-coherent additive synthesis (pygame-ce + numpy)
- GoL on toroidal field (`scipy.ndimage.convolve`, mode='wrap', 8-connectivity)
- Connected-component segmentation (`scipy.ndimage.label`)
- Perceptual axes wired:
  - **size (area) → harmonic number k** — big object = low harmonic near fundamental (K_MAX = 20)
  - **density (bbox fill) → amplitude** — with 1/k^0.7 rolloff to tame upper partials
  - **centroid X → stereo pan**
- Voice cap: MAX_VOICES = 24 (largest objects prioritised when over limit)
- Carrier note set via on-screen piano (C3–C5, latched)
- Interactive UI: draw/erase cells, play/pause [Space], step [S], random [R], clear [C], speed [↑↓]
- Visualisation: harmonic hue (red=low → blue=high), brightness = amplitude; legend shown

- Pattern library sidebar (collapsible, "Lib ▶ / ◀ Lib" button in toolbar):
  - 29 patterns in 9 categories (separate file: `patterns.py`):
    still lifes ×7, oscillators p2 ×4, p3 ×3 (Pulsar, Jam, Caterer),
    p4 ×2 (Mold, Mazing), p5 ×1 (Octagon II), p8+ ×4 (Figure-8, Kok's galaxy, Tumbler, Pentadecathlon),
    spaceships ×4, guns ×1, methuselahs ×3
  - All patterns verified (explicit cells or simulation-confirmed RLE period)
  - Drag & drop: grid-snapped ghost preview (SRCALPHA), OR-insert on drop
  - Mouse-wheel scrolling in sidebar; scroll thumb indicator
  - Sidebar toggled via `state['sidebar_open']`; window width changes with `set_mode`

- **`gol_life_synth_laplacian.py`** — второй полноценный прототип, Laplacian-маппинг:
  - Каждый связный объект → голос; спектр = граф-Лапласиан формы (√λ → частоты мод)
  - Двойной банк слотов (front/back): TOTAL_SLOTS = MAX_VOICES×MAX_MODES_PER_OBJ×2 = 384
  - **Перекрывающийся (overlapping) кросс-фейд** при смене топологии (P0-A fixed):
    при изменении частоты моды — старый front-слот уходит в release (fade-out за
    FADE_CHUNKS чанков), новая частота немедленно поднимается на back-слоте (attack
    из amp=0). Оба слота рендерятся параллельно → нет провала амплитуды (gap).
    Банки меняются ролями после инициации кросс-фейда.
  - **FADE_CHUNKS задействован** (P0-B fixed): длительность фейда определяется
    `FADE_MS=50` мс → `FADE_CHUNKS = max(1, round(FADE_MS/1000/CHUNK_S))`;
    счётчик release_countdown декрементируется в каждом update(), amp_tgt шагает
    линейно к 0 за FADE_CHUNKS шагов. Не привязано к темпу GOL.
  - Фаза непрерывна (phase reset не используется); новый слот стартует с amp_cur=0,
    поэтому начальное значение фазы не слышно.
  - `_extract` + `map_laplacian` скопированы из `mapping_bench.py`; patch строится
    только из клеток данного объекта (не raw bbox, чтобы не захватить соседей)
  - UI идентичен `gol_life_synth.py`: пианино, паттерны, сайдбар, drag-and-drop
  - **Дисторшен исправлен (2026-06-15):** при многих голосах сумма партиалов
    (до 24 объектов × 8 мод + удвоение на кросс-фейде) превышала 1.0 → hard-clip.
    Замер на 3 пентадекатлонах: истинный пик 2.99, 15.4% сэмплов в насыщении.
    Фикс — `MASTER_GAIN` 0.25→0.04 (фикс. headroom под плотное поле; пик ~5.6@0.25).
    Только общий уровень, форма спектра/маппинг не тронуты. После: 0 клиппинга.
  - **Клики исправлены (2026-06-15):** доминирующие клики = крэкл клиппинга
    (4827 резких транзиентов |d2|>100 → 0 после gain-фикса). Остаточные «цыки» на
    стыках чанков = излом наклона амплитудной огибающей (линейный `linspace`-рамп
    переаймится на новый target каждый чанк). Фикс — raised-cosine (smoothstep)
    рамп `_RAMP` для amp/pan с нулевым наклоном на концах → C1 на границах
    (излом стыка ↓804× на уровне огибающей). Стабильные слоты bit-unchanged,
    спектр shape cosine 0.999.
  - Регрессионная harness: `_render_probe.py`, `_click_test.py` + скилл
    `.claude/skills/audio-artifact-probe/` (оффлайн-рендер + объективные метрики
    клиппинга/кликов/сохранности спектра).
  - **Реалтайм-клики ДИАГНОСТИРОВАНЫ и ФИКС реализован (2026-06-15).**
    Запись живой сессии (`CASYNTH_RECORD=1`) показала: аудиопоток движка (WAV)
    ЧИСТ (0 клиппинга, 2 микро-транзиента), но **36 underrun'ов очереди микшера**
    за 37 с — ВСЕ на нормальных кадрах ~17 мс (не просадки CPU; >90мс был 1 кадр).
    Причина: подача звука была завязана на 60fps-цикл pygame (play+1 слот очереди,
    гейт через `get_queue()`), очередь периодически голодала → пауза → щелчок.
    Прошлые фиксы (headroom, smoothstep) были реальны, но лечили мелкий слой.
    **Фикс:** переход на **sounddevice (PortAudio) callback-стрим**. Синтез
    (главный поток) рендерит чанки вперёд в `audio_q` (look-ahead
    `AUDIO_LOOKAHEAD_CHUNKS=3` ≈270мс), callback (аудио-поток) тянет сэмплы на
    аппаратной частоте из кольцевого буфера. Просадка кадра лишь уменьшает буфер,
    не рвёт воспроизведение. Бонус: `pool.update` теперь раз на ЧАНК (а не на
    кадр) → FADE_CHUNKS снова считает чанки, рассинхрон кросс-фейда устранён.
    Громкость применяется в callback. pygame.mixer больше не используется в этом
    файле (только дисплей/события/шрифты). Новая зависимость: **sounddevice**.
    Запись сессии (instrumentation) сохранена. Проверено юнит-тестами callback'а
    (сборка чанков/underrun/vol) + реальным стримом; ждём перезапись Пользователя
    для подтверждения underruns→0 в живой игре.

- `patterns.py` — Traffic light координаты исправлены: правый вертикальный блинкер
  перенесён с col 4 на col 6, горизонтальные (top/bottom) с cols 1-3 на cols 2-4.
  Паттерн теперь стабилен: period=2, 4 компонента на каждом шаге, bbox 7x7.

## Known issues
- Segmentation not wrap-aware: objects crossing the toroidal seam momentarily split into two voices
- No object identity tracking across generations (no true birth / death events, no collision handling)
- Only 3 of ~8 designed perceptual axes are wired; timbre range is correspondingly narrow

## In-flight
Ничего. Следующий шаг — прослушивание `gol_life_synth_laplacian.py` на живом поле
(рождение/смерть/рост структур) и фиксация характера Laplace на реальной динамике.
P0-A и P0-B из ревью зафиксированы (кросс-фейд реализован); P2 (docstring) приведён
в соответствие. Прослушивание теперь не заблокировано артефактами схемы слотов.

## Mapping bench — discover завершён (2026-06-15)
- **`mapping_bench.py`** — ревизия завершена, все P0/P1 прошли researcher-ревью.
  Матрица 4×5: осцилляторы Blinker, Toad, Beacon, Clock (Clock_C период-2) × маппинги
  2D-FFT, Walsh-2D (sequency), Random, Laplacian (негармонический), Granulo.
- **Итог batch-прослушивания (Пользователь):** FFT/Walsh/Random/Granulo на мелком
  зоопарке — различие есть, но **невыразительное** (плохой результат); Granulo вырожден
  (см. `questions.md`). **Laplace выделился** — органичный металлический тон, самостоятельная
  фишка. → Laplace выбран для второго прототипа (см. In-flight).
- Стенд остаётся discover-инструментом; выбракованные приёмы НЕ выброшены, а
  законсервированы (профили — в `decisions.md`).

## Open research — профилирование маппингов (параллельная ветка, не блокирует прототип)
Стратегия: **не выбрасывать тупиковые ветки**, набрать зоопарк приёмов побольше и
собрать **профиль/характер каждого** на разных режимах автомата → карта «какой приём
для чего годится». Рамка и шаблон профиля + текущие профили 5 приёмов — в
`memory/decisions.md` (раздел «Концепция профилирования маппингов»).
- Ближайшая измерительная задача: прогнать законсервированные приёмы (FFT/Walsh/Granulo)
  на **крупных/плотных полях** — проверить гипотезу «шумный на малом → структурный на
  большом». Формат стенда под крупные режимы не заточен (отдельная задача — surface при
  необходимости).
- Разные приёмы могут управлять РАЗНЫМИ параметрами синта (тембр/огибающая/модуляция/
  пространство) — карта это и должна развести.

## Candidate next steps
See `memory/decisions.md` for prioritisation by Researcher.

- Remaining perceptual axes: jaggedness → distortion/inharmonicity, activity → tremolo,
  symmetry → consonance, order/chaos → noise blend, elongation → detune/vibrato
- Wrap-aware segmentation (union-find across toroidal seam)
- Object identity tracking (voice allocation with birth / death / merge / split events)
