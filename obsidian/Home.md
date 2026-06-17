---
type: index
---
	
# CASynth — карта исследования

Синтезатор, чей тембр управляется живым полем Conway's Game of Life.
Каждый связный объект поля → голос; поле = эволюционирующий спектр одной карьерной ноты.

---

## Архитектура (load-bearing)

- [[Load-Bearing]] — 4 решения, которые нельзя молча менять
- [[Sonification-Unit]] — единица сонификации = перцептивный объект
- [[Frequency-Framing]] — гармонический vs негармонический спектр

---

## Маппинги `форма → (freqs, amps)`

| Маппинг | Статус | Характер |
|---|---|---|
| [[Laplacian]] | #active | органичный металлический тон |
| [[FFT-2D]] | #conserved | невыразителен на мелком зоопарке |
| [[Walsh-Hadamard]] | #conserved | невыразителен на мелком зоопарке |
| [[Random-Baseline]] | #baseline | контрольный, без фишки |
| [[Granulo]] | #conserved | вырожден на тонких формах (диагноз) |

---

## Тесты / прослушивания

- [[Small-Zoo]] — матрица 4×5 на мелком зоопарке (done, 2026-06-15)
- [[Large-Dense-Fields]] — #untested (гипотеза: законсервированные оживут)

---

## Проблемы

- [[Bip-Topology-Change]] — #solved (кросс-фейд наборов мод)
- [[Chunk-Clicks]] — #solved (sounddevice callback-стрим + smoothstep-рамп; P1-legacy не мигрирован)

---

## Прототипы

- [[P1-Main]] — `gol_life_synth.py` (legacy v1, pygame.mixer)
- [[P2-Laplacian]] — `gol_life_synth_laplacian.py` (#done — лапласиан в живом виде; крутилки приняты 2026-06-17)
- `gol_synth.py` — мульти-движковый наследник P2 (живой селектор 5 приёмов: FFT/Walsh/Random/Laplace/Granulo); активный стенд профилирования. _(Заметку P3 завести по запросу.)_

---

## Стратегия исследования

- [[Profiling-Strategy]] — как профилировать маппинги
- [[Untested-Branches]] — что ещё не проверено
