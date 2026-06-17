# Developer Questions Queue

Developer appends questions here when a design decision is needed.
User or Researcher answers in-line.

## Entry format
```
## [OPEN] YYYY-MM-DD — [Topic]
**Question:** ...
**Context:** what is being implemented and why this matters.
**Default choice:** what was done in the absence of an answer.
**Answer:** _(to be filled)_
```

Change `[OPEN]` to `[ANSWERED]` once resolved.

_Решённые треды (8 закрытых) вынесены в `archive/questions-resolved.md` — читать по требованию._

---

**Открытых вопросов нет (на 2026-06-17).**

Последний тред — «Живые крутилки тембра Laplace» — реализован Developer (2026-06-16) и
принят Researcher (2026-06-17): DEV-2 закрыт (tail-pool), DEV-1 в режиме ручного headroom
(к авто-нормировке вернуться, если плотные яркие сцены станут рабочим режимом). Перенесён
в архив. Текущие следующие шаги — исследовательские, ведутся в `decisions.md` (2026-06-17:
шаг B / формозависимые амплитуды, развилка extract 8×8 → граф по клеткам объекта) и в
`current.md` (In-flight). Алгоритмы и реестр `ENGINES` в `casynth_core.py` приняты Researcher 2026-06-17; прототип `gol_synth.py` — инженерия, не под дизайн-ревью (см. `current.md` «Статус ревью»).
