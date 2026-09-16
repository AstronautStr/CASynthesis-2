# Objects / Laplace -- hand-over measurements (technical evidence, not listening verdicts)

Generated 2026-09-17 01:33:25 at commit 289fb6d.  SR 44100, block 352 (7.98 ms). A = `laplacian`, B = `ca_object_resonators` (Spectrum Laplace, LAPLACE_GAIN 0.7, output x0.5); f0 110 Hz, 6 gen/s, gain 0.028 = vol 0.7.  RMS = stereo RMS of the ready 12 s.

## Spectral data A / B (73 states, 3 setting sets, every component)

| scene | components | modes | worst abs diff | equal within 1e-10 |
|---|---|---|---|---|
| ol_glider | 219 | 876 | 0 | True |
| ol_galaxy | 1065 | 4784 | 0 | True |
| ol_neighbor | 438 | 2774 | 0 | True |

## The seven settings on the galaxy (generations 0..7 in which the data differ)

| setting | value | generations | preflight |
|---|---|---|---|
| n | 4 | [0, 1, 2, 4, 5, 6] | [0, 1, 2, 3, 4, 5, 6, 7] |
| spread | 1.0 | [2, 4, 5] | [2, 4, 5] |
| alpha | 2.0 | [0, 1, 2, 3, 4, 5, 6, 7] | [0, 1, 2, 3, 4, 5, 6, 7] |
| shape | 1.0 | [0, 1, 2, 3, 4, 5, 6, 7] | [0, 1, 2, 3, 4, 5, 6, 7] |
| harm | 1.0 | [0, 1, 2, 3, 4, 5, 6, 7] | [0, 1, 2, 3, 4, 5, 6, 7] |
| fullshape | 0 | [2, 4, 5] | [2, 4, 5] |
| dyn | 1.0 | [1, 2, 3, 4, 5, 6, 7] | [0, 1, 2, 3, 4, 5, 6, 7] |

## Scenes (12 s from a fresh start, the scene side gain applied)

| scene | side | engine | rms dB | peak | clip | p99 ms |
|---|---|---|---|---|---|---|
| ol_glider | A | laplacian | -35.50 | 0.0470 | 0 | 2.62 |
| ol_glider | B | ca_object_resonators | -35.52 | 0.0858 | 0 | 2.62 |
| ol_galaxy | A | laplacian | -26.38 | 0.2263 | 0 | 6.21 |
| ol_galaxy | B | ca_object_resonators | -26.39 | 0.2995 | 0 | 6.21 |
| ol_neighbor | A | laplacian | -30.26 | 0.0820 | 0 | 3.06 |
| ol_neighbor | B | ca_object_resonators | -30.23 | 0.1815 | 0 | 3.06 |

- ol_glider: side gain {'A': 1.0, 'B': 1.08}; A - B = +0.02 dB (at unit side gain +0.69 dB); within 1 dB True; continue from 6 s exact True (200 blocks); B figures at the end 1, tails 0, faded 0, in place 0, dropped 0
- ol_galaxy: side gain {'A': 1.0, 'B': 0.74}; A - B = +0.01 dB (at unit side gain -2.61 dB); within 1 dB True; continue from 6 s exact True (200 blocks); B figures at the end 8, tails 37, faded 0, in place 0, dropped 0
- ol_neighbor: side gain {'A': 1.0, 'B': 1.25}; A - B = -0.03 dB (at unit side gain +1.91 dB); within 1 dB True; continue from 6 s exact True (200 blocks); B figures at the end 2, tails 0, faded 0, in place 0, dropped 0

## No packet from any setting or from Restore (side B on the L3 field)

| setting | value | pulse moved | frequencies changed | weight ramp (samples) |
|---|---|---|---|---|
| n | 5 | False | True | 882 |
| spread | 0.8 | False | True | 882 |
| alpha | 1.7 | False | False | 882 |
| shape | 0.6 | False | True | 882 |
| harm | 0.9 | False | True | 882 |
| fullshape | 0 | False | True | 882 |
| dyn | 0.5 | False | False | 882 |
| spectrum | 0 | False | True | 882 |
| spectrum | 1 | False | True | 882 |
| detector | 0 | False | False | 530 |
| radius_mul | 1.0 | False | False | 178 |
| decay_s | 1.2 | False | False | 0 |

Restore (export / restore, the next 40 blocks): exact True.

## L3 receiver events per generation

- radius_1.0: receiver e [0.0] (71 generations), blinker e [4.0]; receiver centre [11.764705882352942, 11.764705882352942], R 5.32962 -> R_eff 5.32962; ids [1, 2]; receiver 12 modes from 110.00 Hz, blinker 2 modes from 110.00 Hz
- radius_1.5: receiver e [4.0] (71 generations), blinker e [4.0]; receiver centre [11.764705882352942, 11.764705882352942], R 5.32962 -> R_eff 7.99443; ids [1, 2]; receiver 12 modes from 110.00 Hz, blinker 2 modes from 110.00 Hz
- live lever 1.5 -> 1.0: no packet from the knob True; next generation receiver e 0.0, blinker e 4.0

## L2 voices (side B, 12 s)

- 72 generations, identities seen 315, max figures 12, max sounding 8, max tails 51; faded 0, in place 0, dropped 0, unvoiced blocks 0
- mode return (Figure law): modes [4, 2, 4], returning modes' state before their first sample 0, tail slots 1 (undriven True, pulse 0), id kept True
- mode return (Laplace law): modes [4, 2, 4], returning modes' state before their first sample 0, tail slots 1 (undriven True, pulse 0), id kept True

## Timing (both sides, after 1 s of warm-up)

- ol_glider: p50 1.17  p95 2.02  p99 2.32  max 3.94 ms (budget 7.98, ok True)
- ol_galaxy: p50 1.78  p95 3.78  p99 5.94  max 7.46 ms (budget 7.98, ok True)
- ol_neighbor: p50 1.32  p95 2.19  p99 3.06  max 3.86 ms (budget 7.98, ok True)
- numba: True

## Stress probes (gain 0.04, 6 s, apart from the scenes)

| mode | settings | A peak / clip | B peak / clip | figures / sounding / tails B | faded / in place / dropped B | p99 ms |
|---|---|---|---|---|---|---|
| dense_random_evolving | REQ table | 0.613 / 0 | 1.186 / 4 | 19/15/95 | 363/0/0 | 21.23 |
| dense_random_evolving | n20 spread1 alpha0 shape1 full1 dyn1 | 0.599 / 0 | 2.076 / 26 | 18/13/96 | 429/0/0 | 28.66 |
| random_every_block | REQ table | 0.504 / 0 | 1.497 / 25 | 3/1/96 | 2136/117/0 | 54.22 |
| random_every_block | n20 spread1 alpha0 shape1 full1 dyn1 | 0.501 / 0 | 2.096 / 32 | 6/1/96 | 2098/131/0 | 154.10 |
| full_toggle_every_block | REQ table | 0.228 / 0 | 0.728 / 0 | 0/0/96 | 349/0/0 | 136.46 |
| full_toggle_every_block | n20 spread1 alpha0 shape1 full1 dyn1 | 0.931 / 0 | 2.974 / 206 | 0/0/96 | 349/0/0 | 319.47 |
| knobs_moving | REQ table | 0.305 / 0 | 0.471 / 0 | 12/8/5 | 0/0/0 | 19.76 |
| knobs_moving | n20 spread1 alpha0 shape1 full1 dyn1 | 1.099 / 2 | 0.598 / 0 | 12/8/5 | 0/0/0 | 19.89 |

## Cost of the Laplace law by component size (full torus graph, no decimation)

| cells | shape | ms | over the block budget |
|---|---|---|---|
| 64 | 0 | 0.4 | False |
| 64 | 1 | 1.2 | False |
| 256 | 0 | 6.1 | False |
| 256 | 1 | 19.0 | True |
| 576 | 0 | 39.4 | True |
| 576 | 1 | 108.1 | True |
| 1024 | 0 | 119.5 | True |
| 1024 | 1 | 224.4 | True |

## Catalog check (C:\Users\Astro\Documents\Projects\CASynth-2\lab_catalog\objects_laplace_2026_09_17)

- 20260917-012137-af7136 L1 - Одна фигура: Laplace / Objects (glider, 6 пок: match  (notes: 'Гипотеза - При одинаковой настройке спек')
- 20260917-012134-38c688 L2 - Сборка и распад: Laplace / Objects (Kok's gal: match  (notes: 'Гипотеза - На одной и той же распадающей')
- 20260917-012131-c47bd9 L3 - Сосед и радиус: Laplace / Objects (17 клеток : match  (notes: 'Гипотеза - В А обе фигуры звучат через о')

## Summary

- Scenes: spectral data of A and B equal on every component of the 73 states (3 setting sets), all seven settings act (spread / full in generations 2, 4, 5), no packet from any setting or Restore, L3 receiver 0 / 4 per generation at x1 / x1.5 and the live lever stops it, L2 voices without drops and the 4 -> 2 -> 4 rule holds, A - B within 1 dB after the scene side gain, no clipping, continuation exact, p99 within the budget, records replay exactly
- Stress probes: finite in all 8 probes; clipping at gain 0.04 in 6, over budget in 8, in-place fades in 2, hard drops in 0; worst pre-clip peak 2.974
- Limitations:
  - the n setting changes the galaxy's data in generations [0, 1, 2, 4, 5, 6] only: in generations 3 and 7 every component has 4 cells (3 modes), n = 4 and n = 12 select the same three (the preflight counted the zero-padded array of the old synth)
  - on a figure across the seam the old Laplace (bbox of the unwrapped labels) and Objects (torus component) may segment differently; none of the 73 states of the three scenes touches the seam
  - the old Laplace's node ceiling (256, lattice decimation) is not carried into the Objects graph: a component above ~300 cells costs more than the block budget on the render thread (see the cost table)
  - the level match is one constant factor per scene on side B (ol_glider 1.08, ol_galaxy 0.74, ol_neighbor 1.25), chosen for <= 1 dB of integral RMS, not equal loudness; the old Laplace algorithm is untouched
  - the two N4 experiments keep the Figure law; their records were made by the v1 engine and open in their own version
  - stress clipping at gain 0.04 (vol 1) in 6 of 8 probes: dense_random_evolving A 0 / B 4 blocks (peaks 0.61 / 1.19); dense_random_evolving A 0 / B 26 blocks (peaks 0.60 / 2.08); random_every_block A 0 / B 25 blocks (peaks 0.50 / 1.50); random_every_block A 0 / B 32 blocks (peaks 0.50 / 2.10); full_toggle_every_block A 0 / B 206 blocks (peaks 0.93 / 2.97); knobs_moving A 2 / B 0 blocks (peaks 1.10 / 0.60)
  - 8 of 8 stress probes exceed the block budget at p99 (worst 319.5 ms): the figure analysis of dense fields on the render thread -- offline exact, live underruns possible
  - in-place fades under whole-field replacement every block: random_every_block: 117; random_every_block: 131
  - worst stress pre-clip peak 2.974 at gain 0.04 -- measured for these finite probes, not guaranteed for any playing
  - no listening in this report: hearing the difference is the user's verdict
