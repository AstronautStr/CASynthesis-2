# 2026-06-23 — Модуляризация монолита `gol_synth.py`

## Задача
`gol_synth.py` дорос до 1876 строк («критическая масса»). Разнести на плоские
модули `casynth_*.py`, поведение **бит-в-бит**, прототип запускается на каждом шаге.
Решения с Пользователем: аудио-движок отдельно И UI отдельно; плоская раскладка;
кросс-файловую дедупликацию с замороженным предком `gol_life_synth_laplacian.py`
НЕ делаем (его копии `SlotPool`/`analyse` намеренно разошлись: ADSR + мульти-движок;
измерено diff 64/33 строки).

## Результат — карта модулей
| Файл | строк | содержимое |
|---|---|---|
| `gol_synth.py` | 1876 → **801** | entry: pygame-init, layout-construction, event-loop, аудио/MIDI-потоки, write-side рекордера, CLI `replay`; `_KB_PIANO` (input-map, pygame). |
| `casynth_config.py` | 261 | все константы (геометрия/аудио/слоты/ADSR/палитра/NOTE_DIVS/`_RAMP`); pygame-free; `__all__` экспортит и `_`-имена. |
| `casynth_engine.py` | 423 | `step/hsv/midi_to_freq/note_name/analyse/render_chunk_laplacian/SlotPool` (+`_NEIGH/_S8`). |
| `casynth_session.py` | 166 | `_dump_session/replay_session/_replay_cli` (read-side рекордера). |
| `casynth_midi.py` | 67 | `MidiInput` + опциональный `mido`/`MIDI_AVAILABLE`. |
| `casynth_ui.py` | 345 | `_make_piano/pattern_preview_surf` + `draw_frame(screen,fonts,state,lay,rt)` (весь рендеринг). |

Иерархия импортов без циклов: `config ← engine ← session/ui ← gol_synth`.

## Как делалось (бит-в-бит)
- **Phase 0 golden-master** (до единой правки): `artifacts/_golden_master.py` — детерм.
  сцены (5 движков + ADSR pluck/pad + крупная форма 20×20 децимация + смена сцены),
  рендер реальным путём → `artifacts/_golden_master_ref.wav/.npz`. sha `cd31269...`,
  17248 сэмплов. Старые `_session_*` отброшены как протухшие: `2c67317` (CHUNK_S
  0.09→0.02; ныне 0.008) и `250c108` (threaded render) меняют выход для ранних записей.
- **Этап 1** — чистый перенос верхнеуровневых символов скриптом по диапазонам строк
  (kept-сегменты дословны). `from casynth_config import *` сохраняет «голые» ссылки в
  `main()` без правок тел. Гейт: golden байт-в-байт + 30/30 + smoke.
- **Этап 2** — вынос рендера. Добавлен хук `CASYNTH_DUMPFRAME=path` (дамп кадра 0 → exit)
  → снят эталон `_ui_before.png` ДО выноса. `draw_frame` = тело draw-блока ДОСЛОВНО с
  прологом распаковки `lay`/`rt` в те же локальные имена (2 подстановки: `ctrl_value`→
  `_ctrl_value(state,c)`, `f0()`→`midi_to_freq(state['note'])`). Гейт: `_ui_after.png`
  == `_ui_before.png` бит-в-бит (sha `ad5f1b4...`) + golden + 30/30 + живой цикл 3с.
  Баг по пути: `PATTERNS` не был импортирован в `casynth_ui` — добавлен.

## Сознательные решения по объёму
- `main()` ОСТАВЛЕН владельцем layout-construction, event-loop и потоков: они завязаны
  на изменяемое состояние; разбор в Layout-класс инвазивен и почти не покрыт
  автотестами (клики/драги). `lay` = `SimpleNamespace` тех же объектов, что мутирует
  event-loop (`ctrls` ребилдится in-place — ссылка отслеживает) → нет дивергенции.
- Контракт рекордера цел: схема `replay_controls` не менялась; write-side в `main()`,
  read-side в `casynth_session`.

## Follow-up (НЕ сделано, требует согласования)
- Конвенция `CLAUDE.md`/`developer.md` «все константы — вверху файла прототипа» теперь
  фактически «вверху `casynth_config.py`». Обновить формулировку — отдельно.
- Полный вынос layout-construction в `Layout`-класс — если захочется «main() ещё тоньше».
- Артефакты в `artifacts/` (gitignored): `_golden_master.py/_ref.*`, `_ui_before/after.png`
  — переиспользуемые регрессионные гейты (golden-master аудио + dump-frame UI).
