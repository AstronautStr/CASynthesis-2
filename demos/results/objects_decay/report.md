# Objects — Decay law (losses from the history of the cells): measurements

Generated 2026-09-18 20:42 on commit `1fdf8d14079390cf64a2097e78607e356bedf1f7`; numba True.  REQ `memory/req-objects-decay-2026-09-18.md`, preflight `memory/research/objects-decay-preflight-2026-09-18.json`.  No listening result here.

## 1. Targets against the preflight

| case / law | banks checked | worst abs(T − preflight) s | worst abs(f − preflight) Hz |
|---|---|---|---|
| D1_Common age | 40 | 0.0e+00 | 8.0e-13 |
| D1_Modal age | 40 | 1.1e-16 | 8.0e-13 |
| D2_Common age | 26 | 0.0e+00 | 0.0e+00 |
| D2_Modal age | 26 | 1.1e-16 | 0.0e+00 |
| D3_Common age | 324 | 0.0e+00 | 8.0e-13 |
| D3_Modal age | 324 | 2.2e-16 | 8.0e-13 |

## 2. D1 — Fixed 0.947926 s / Common age

Control: ln(1000) / 0.947926 = 7.2872 1/s.  Side B over 5…20 s: mean target gamma 7.2872 (0.00 %), mean applied gamma 7.2872 (0.00 %); within 1 %: True.

Phases of the mature cycles (generation mod 5, 5…19.5 s): the target T at the step, the mean APPLIED T over the interval, and the T60 measured in the rendered PCM between the steps (10 ms windows from +60 ms; median over the cycles):

| phase | intervals | target T at step s | applied T mean s | PCM T60 A s | PCM T60 B s (min…max, measured) |
|---|---|---|---|---|---|
| 0 | 6 | 0.735 | 0.959 | 0.942 | 0.973 (0.971…0.973, 6) |
| 1 | 6 | 0.650 | 0.905 | 0.941 | 0.922 (0.920…0.922, 6) |
| 2 | 6 | 0.415 | 0.732 | 0.941 | 0.710 (0.710…0.710, 6) |
| 3 | 6 | 1.154 | 1.232 | nan | nan (nan…nan, 0) |
| 4 | 5 | 0.852 | 1.054 | 0.941 | 1.063 (1.063…1.064, 5) |

Max / min over the phases: applied T 1.68; over the 4 phases measurable in the PCM: T60 of B 1.50, of A 1.00; PCM B vs applied T within 3.0 %, PCM A vs the control within 0.8 %.  The phase without births (no new strike) continues the tail of the previous interval, which is near the int16 floor by then: "nan" = fewer than 8 windows above 30 LSB.  The phase sequence differs in the rendered audio of B and not of A: True.

## 3. D2 — Common age / Modal age, the two scripted pauses

Frequencies [110.0, 216.168] Hz; pauses in the journal (sample, on, scripted): [(132704, True, True), (242880, False, True), (375232, True, True), (485408, False, True)].

Pause 1 (132704…242880), resonator magnitudes of the REAL engine from the first block end (133056):

| elapsed s | Common lower / upper dB | Common upper − lower dB | Modal lower / upper dB | Modal upper − lower dB |
|---|---|---|---|---|
| 0.000 | 0.0 / 0.0 | -3.6 | 0.0 / 0.0 | 0.1 |
| 0.104 | -19.7 / -19.6 | -3.5 | -33.1 / -6.2 | 27.1 |
| 0.247 | -33.7 / -33.7 | -3.5 | -53.2 / -14.2 | 39.2 |
| 0.503 | -50.6 / -50.5 | -3.5 | -73.9 / -27.2 | 46.8 |

Upper − lower in the rendered int16 PCM (DFT, 46 ms Hann windows; "-" = a component below 10 LSB):

| after s | Common dB | Modal dB |
|---|---|---|
| 0.03 | -3.4 | 10.1 |
| 0.06 | -3.4 | 18.8 |
| 0.10 | -3.4 | 26.5 |
| 0.15 | -3.3 | - |
| 0.20 | -3.3 | - |
| 0.30 | - | - |

PCM growth of upper − lower from 0.03 to 0.1 s: Modal +16.4 dB, Common +0.0 dB.

Pause 2 (375232…485408), resonator magnitudes of the REAL engine from the first block end (375584):

| elapsed s | Common lower / upper dB | Common upper − lower dB | Modal lower / upper dB | Modal upper − lower dB |
|---|---|---|---|---|
| 0.000 | 0.0 / 0.0 | -3.6 | 0.0 / 0.0 | 0.1 |
| 0.104 | -19.7 / -19.6 | -3.5 | -33.1 / -6.1 | 27.1 |
| 0.247 | -33.7 / -33.7 | -3.5 | -53.2 / -14.2 | 39.2 |
| 0.503 | -50.6 / -50.5 | -3.5 | -73.9 / -27.2 | 46.9 |

Upper − lower in the rendered int16 PCM (DFT, 46 ms Hann windows; "-" = a component below 10 LSB):

| after s | Common dB | Modal dB |
|---|---|---|
| 0.03 | -3.4 | 10.1 |
| 0.06 | -3.4 | 18.8 |
| 0.10 | -3.4 | 26.5 |
| 0.15 | -3.3 | - |
| 0.20 | -3.3 | - |
| 0.30 | - | - |

PCM growth of upper − lower from 0.03 to 0.1 s: Modal +16.4 dB, Common +0.0 dB.

Against the independent two-mode probe of the preflight (first pause):

| elapsed s | Common engine / probe dB | Modal engine / probe dB | ratio Modal engine / probe |
|---|---|---|---|
| 0.000 | [0.0, 0.0] / [0.0, 0.0] | [0.0, 0.0] / [0.0, 0.0] | 0.13 / 0.13 |
| 0.104 | [-19.66, -19.62] / [-19.66, -19.62] | [-33.13, -6.16] / [-33.13, -6.16] | 27.10 / 27.10 |
| 0.247 | [-33.71, -33.67] / [-33.71, -33.67] | [-53.21, -14.18] / [-53.21, -14.18] | 39.16 / 39.16 |
| 0.503 | [-50.58, -50.53] / [-50.58, -50.53] | [-73.9, -27.21] / [-73.9, -27.21] | 46.82 / 46.82 |

Worst difference engine − probe 0.000 dB.  Growth of upper − lower over 0.5 s: pause 1 Modal +46.7 dB, Common +0.0 dB; pause 2 Modal +46.7 dB, Common +0.0 dB.  The mechanism shows in the real audio: True.

## 4. D3 — Fixed 1.39 s / Modal age

107 generations, 108 accepted transitions; up to 4 figures and 21 tails on side B (21 tails carrying their gamma); identities A [2, 181] / B [2, 181], next id [182, 182]; drops [0, 0], in place [0, 0].  T max / min inside a bank of side B: median 1.61, p90 3.04, max 11.92.

## 5. Isolation of the sides

| scene | blocks | ids, cells, frequencies, weights, packets, a, Attack equal | blocks where the losses differ |
|---|---|---|---|
| od_d1 | 2506 | True | 2506 |
| od_d2 | 2005 | True | 1943 |
| od_d3 | 2256 | True | 2256 |

## 6. Levels (one constant factor of side B per experiment)

| scene | window s | A − B unit gains (whole) dB | side gain B | A / B window RMS dB | A − B delivered dB | peak A / B | clip | finite | p99 ms (budget 7.98) |
|---|---|---|---|---|---|---|---|---|---|
| od_d1 | 5…20 | +0.79 (+0.97) | 1.09 | -44.47 / -44.51 | +0.04 | 0.046 / 0.049 | 0/0 | True | 1.42 (max 2.00) |
| od_d2 | 2…16 | -1.77 (-1.75) | 0.82 | -52.12 / -52.09 | -0.04 | 0.027 / 0.022 | 0/0 | True | 1.22 (max 2.51) |
| od_d3 | 3…18 | +1.64 (+1.83) | 1.21 | -33.93 / -33.91 | -0.02 | 0.163 / 0.117 | 0/0 | True | 3.84 (max 5.99) |

D2 window: the active parts after 2 s, both pauses excluded.  The side gain scales both D2 components alike: the upper / lower ratio above does not depend on it.

## 7. Continue

- D2 from inside the first pause (paused True, 3 scripted commands pending): 4 s exact True, resumed True.
- D1 during a gamma transition (A switched to Modal age 3 blocks earlier, gap 9.15 1/s): 300 blocks exact True.
- D3 after split / merge (20 tails with their gamma, next id 92): 3 s exact True.

## 8. Timing and the limits of large fields

| run | law | cells | figures (B, end) | p50 ms | p99 ms | max ms | within budget | peak | clip | finite |
|---|---|---|---|---|---|---|---|---|---|---|
| random soup 35 %, 6 gen/s | Fixed | 372 | 17 | 1.99 | 27.15 | 31.41 | False | 0.548 | {'A': 0, 'B': 0} | True |
| random soup 35 %, 20 % of the cells flipped every block | Fixed | 372 | 6 | 64.19 | 75.91 | 80.20 | False | 0.742 | {'A': 0, 'B': 0} | True |
| random soup 35 %, 6 gen/s | Modal age | 372 | 17 | 3.59 | 32.76 | 39.39 | False | 0.538 | {'A': 0, 'B': 0} | True |
| random soup 35 %, 20 % of the cells flipped every block | Modal age | 372 | 6 | 252.02 | 341.53 | 373.55 | False | 0.592 | {'A': 0, 'B': 0} | True |

One figure whose geometry changes at every boundary (median ms of one block of ONE side; "new" = a shape never seen before at every boundary, the worst case; "again" = the same shapes return, as in a periodic figure):

| figure | cells | Fixed ms | Modal new ms | Modal again ms |
|---|---|---|---|---|
| 6 x 6 | 36 | 0.66 | 2.04 | 0.63 |
| 10 x 10 | 100 | 1.19 | 3.32 | 1.24 |
| 14 x 14 | 196 | 2.13 | 6.82 | 2.15 |
| 18 x 18 | 324 | 13.24 | 40.98 | 15.18 |
| 22 x 22 | 484 | 30.15 | 133.02 | 29.69 |

Budget 7.98 ms for BOTH sides.  The eigen-decompositions run only when the geometry or the mode set changes, never per sample.

## 9. Catalog

- 20260918-204157-918466 D1 - Хвост меняется вместе с фазами фигуры: Fixed 0.948 s / Common age: replay match , pinned 1fdf8d14079390cf64a2097e78607e356bedf1f7, notes True, side gain {'A': 1.0, 'B': 1.09}
- 20260918-204156-1374c7 D2 - Две частоты расходятся по длине хвоста: Common age / Modal age (B: replay match , pinned 1fdf8d14079390cf64a2097e78607e356bedf1f7, notes True, side gain {'A': 1.0, 'B': 0.82}
- 20260918-204155-4ff923 D3 - Разные хвосты в смеси двух периодов: Fixed 1.39 s / Modal age (Ja: replay match , pinned 1fdf8d14079390cf64a2097e78607e356bedf1f7, notes True, side gain {'A': 1.0, 'B': 1.21}
