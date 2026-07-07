# 2026-07-07 — Рефактор огибающих ИСПОЛНЕН (КА = осциллятор, голос = классический)

Исполнение плана `memory/log/2026-07-06-envelope-refactor-handoff.md`. Дизайн (ЧТО/ПОЧЕМУ) —
`decisions.md` 2026-07-06 «Целевая модель огибающих». Здесь — КАК сделали, по шагам, с гейтами.

## Итог
Целевая модель в коде. Смена несущей стала чистым транспозом (фазо-непрерывным, без per-mode
ретриггера), артикуляция ноты вынесена в отдельный Voice ADSR (VCA) над суммой, GEN-огибающая
переведена в доли интервала тика. Структурно вылечены оба дефекта: «зависание» длинных нот и
churn/underrun tail-пула на плотном MIDI. Прототип играет на каждом шаге.

## Ключевое открытие (упростило план)
Три рендер-пути независимы: (1) живой `_render_loop` в `gol_synth.main()`, (2) `artifacts/
_golden_master.py::render()` (свой, ms-based, зовёт `analyse@note` напрямую), (3)
`casynth_session.replay_session` (оффлайн, per-frame, `n_rendered=0` в тред-эре → тишина).
Рефактор менял ТОЛЬКО (1). Примитивы движка (`analyse`/`SlotPool.update`/`render_chunk_laplacian`)
по поведению не менялись — добавлен лишь `transpose=1.0` с дефолтом-no-op. Поэтому **golden-master
остался байт-в-байт всю дорогу** — переблагословлять НЕ пришлось (в отличие от опасений хэндоффа).
Новую модель валидировали оффлайн-симуляциями + UI-dumpframe + слухом.

## Шаги и коммиты
- **983e1dc — Шаг 1 (ЯДРО): развязка транспонирования.** `render_chunk_laplacian(..., transpose=1.0)`
  умножает `freq_slots[k]` (active+tail) на ratio. В `gol_synth`: `_live_voices`(рескейл частот) →
  `_transpose`(ratio); `analyse` при `REFERENCE_F0=midi_to_freq(NOTE_DEFAULT)`; рендер-тред отдаёт
  транспоз; UI-спектр транспонирует display-копию (сохранён дизайн-интент «бары едут с нотой»).
  Гейт: golden-master байт-в-байт, 58/58, UI-кадр байт-в-байт (дефолтная нота → ratio=1). Проверка
  сути: скачок несущей на октаву → **0 краж хвостов** (было бы N краж под стекингом).
- **ed2269c — Шаг 2a: Voice ADSR (VCA) как no-op.** Константы `VOICE_*` (ms) в config; `TOOLBAR_H`
  248→358 под второй ADSR-блок (VOICE над GEN + Tune; метр/громкость/MIDI-бар/вкладки уехали вниз).
  State `voice_*`; геометрия правой колонки; `rebuild_ctrls` строит 4 VOICE + 4 GEN + Tune; в
  `_render_loop` — `venv` (скаляр), множится в master gain (сид = held-состояние; A=0,S=1 → ≡1).
  `replay_controls` пишет `voice_*`. UI рисует заголовки VOICE/GEN. Гейт: golden байт-в-байт, 58/58,
  VCA≡1.0 на hold/off/on (симуляция), UI-эталон обновлён.
- **a84737e — Шаг 2b+3: свободнобегущий осциллятор + release/ретриггер.** Убран gate-gating
  (`voices_in = spec['voices'] if spec else []`); VCA-release на note-off edge; note-on edge
  ретриггерит attack. GEN-хвосты — только на топ-событиях автомата. Проверка end-to-end (hold C3 →
  скачок в G3 → note-off): 0 краж, звук держится сквозь скачок (RMS 0.035→0.036), release гаснет
  до нуля (peak 0). Гейт: golden байт-в-байт, 58/58.
- **e097cf2 — Шаг 4: GEN ADSR в долях тика.** `GEN_*_DEFAULT`+`GEN_FRAC_MIN/MAX`; state
  `gen_{attack,decay,sustain,release}` (доли [0,1]); в `_render_loop` `chunks = max(1, round(доля ×
  interval_s/CHUNK_S))`, `interval` из live BPM/деления. `replay_controls` пишет `bpm`+`div_idx`+
  `gen_*`; `replay_session` читает `gen_*` через интервал (фолбэк на легаси `*_ms`). Дефолты при
  120BPM 1/4 = старые chunk-counts точь-в-точь (release 6, attack 1). Гейт: golden байт-в-байт,
  58/58, UI обновлён (GEN дробью, VOICE в ms).
- **Шаг 5 (cleanup):** `test_render_transpose_equivalence` (инвариант `render(f,×2)==render(2f,×1)`
  + `transpose=1` no-op) → **59/59**; стейл-комменты/ссылки на старые ключи поправлены; `current.md`,
  этот лог.

## Клампы/оговорки (реализованные решения)
- Клэмп GEN: пропорциональное сжатие (`chunks = max(1, round(доля×interval/CHUNK_S))`), НЕ срез —
  быстрый автомат просто укорачивает огибающую. (Снимает devil's-advocate хэндоффа.)
- Диапазон GEN A/D/R = [0,1] доли (S остаётся 0..1 уровнем).
- Транспоз применяется и к tail-слотам → релиз-хвосты едут с несущей (в модели tail = Release
  Generation-огибающей = часть осциллятора; фазо-непрерывно, без щелчка).
- `_render_loop`: VCA-`venv` сидится в held-состояние (нота залатчена на старте) — стабильное поле
  звучит без ожидания фронта; при дефолтах A=0,S=1 сидинг даёт ≡1.

## Не сделано (осознанно, помечено)
- VCA-артикуляция в `replay_session` НЕ воспроизводится (per-frame путь superseded, `n_rendered=0`).
  Достаточно того, что лог её пишет (контекст). Вернуть при возрождении оффлайн-аудио-реплея.
- Развилка MIDI-reduction highest vs last-note (маппинг входа) — теперь можно вернуться.
- Комментарии в `gol_life_synth_laplacian.py`/`ab_bench.py`/`_golden_master.py` про `*_ms` —
  это самостоятельные ms-based инструменты (предок заморожен; стенд/harness со своей ADSR), не
  трогали намеренно.
