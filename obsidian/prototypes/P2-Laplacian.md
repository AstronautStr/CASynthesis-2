---
type: prototype
status: in-progress
file: gol_life_synth_laplacian.py
---

# P2 — gol_life_synth_laplacian.py

#in-progress

## Цель
Вывести [[Laplacian]]-маппинг со стенда в живой интерактивный контекст. «Сырой вид» — наиграть характер приёма на реальной динамике поля (рождение/смерть/рост структур, непрерывная смена топологии). Не финальный инструмент.

## База
`gol_life_synth.py` + `map_laplacian` из `mapping_bench.py`.

## Изменения по сравнению с P1
- Спектр голоса = Laplace-резонансы формы объекта (не size→k·f0)
- Контракт: `map_laplacian(patch, f0) → (freqs[], amps[])`
- Привязка высоты: низшая мода объекта ≈ f0 карьерной ноты (подтверждено)
- **Кросс-фейд наборов мод** при смене топологии (решение [[Bip-Topology-Change]])
- Стереопан оставить как есть

## Что НЕ меняется
GOL-симуляция, UI, пианино, паттерны, сайдбар.

## Проблемы к решению
- [[Bip-Topology-Change]] — реализовать кросс-фейд (блокер качества)
- [[Chunk-Clicks]] — гигиена сшивки чанков (separate)

## Статус
Developer реализует. Ожидается результат.

## Связано
[[Laplacian]] · [[P1-Main]] · [[Bip-Topology-Change]] · [[Chunk-Clicks]] · [[Frequency-Framing]]
