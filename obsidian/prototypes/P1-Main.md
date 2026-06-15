---
type: prototype
status: working
file: gol_life_synth.py
---

# P1 — gol_life_synth.py

#working

## Что реализовано
- Непрерывный фазо-когерентный аддитивный синтез (pygame-ce + numpy)
- GOL на торическом поле (scipy, 8-связность)
- Сегментация связных компонент
- Пианино C3–C5 (лечение несущей ноты)
- Стереопан (centroid X)
- Библиотека паттернов (29 шт., сайдбар)
- Перцептивные оси: size→harmonic, density→amplitude, centroid→pan

## Маппинг в P1
`size → k`, `freq = k * f0` — гармоническая решётка. Контракт расширен: `render_chunk` принимает `freq_cur[]` (сделано для [[P2-Laplacian]]).

## Проблемы
- [[Chunk-Clicks]] — клики на сшивке чанков
- Сегментация не wrap-aware
- Нет identity tracking

## Связано
[[Load-Bearing]] · [[Sonification-Unit]] · [[P2-Laplacian]]
