# N2 events -- hand-over measurements (technical evidence, not listening verdicts)

Generated 2026-09-16 12:57:20 at commit 9e40aba.  SR 44100, block 352 (7.98 ms), calibration A x20 x 10^(-16.5/20), B x96, gain 0.028 (vol 0.7).

## Scenes (12 s from a fresh start, mono RMS over seconds 2-12)

| scene | period | events/step | pop@72 / ev after | A rms dB (pre) | B rms dB (pre) | A peak | B peak (pre) | clip A/B | resets A | B tail +1 s / +2 s dB | p99 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n2_rhythm | 3 | 32-56 | 48 / 56 | -37.1 (-37.1) | -35.4 (-35.4) | 0.061 | 0.132 (0.132) | 0/0 | 0 | -57.5 / -117.7 | 0.61 |
| n2_travel | 128 | 4-4 | 5 / 4 | -36.9 (-36.9) | -39.7 (-39.7) | 0.069 | 0.129 (0.129) | 0/0 | 0 | -56.2 / -118.8 | 0.69 |
| n2_growth | None | 2-78 | 71 / 66 | -36.4 (-36.4) | -35.2 (-35.2) | 0.069 | 0.159 (0.159) | 0/0 | 0 | -56.1 / -119.5 | 0.56 |

## Continuation and a late edit (same saved end state)

| scene | gens in 4 s | field moved | edit cells | A differs / spectral dist dB | B differs / spectral dist dB |
|---|---|---|---|---|---|
| n2_rhythm | 25 | True | 144 | True / 10.13 | True / 8.19 |
| n2_travel | 65 | True | 10 | True / 5.27 | True / 7.41 |
| n2_growth | 25 | True | 130 | True / 7.72 | True / 8.80 |

## B: the same event on different nodes (energy equalised, 1 s)

| pair | kind | envelope corr | log-spectrum rms dB | spectral dist dB | L/R ratio a, b |
|---|---|---|---|---|---|
| 0_vs_5 | unit_packet | 0.605 | 5.44 | 8.50 | 0.97, 1.00 |
| 1_vs_2 | unit_packet | 0.983 | 5.19 | 6.99 | 0.97, 0.98 |
| 3_vs_7 | unit_packet | 0.657 | 5.20 | 7.59 | 0.95, 0.95 |
| 0_vs_5 | one_cell | 0.609 | 5.46 | 8.43 | 0.97, 1.00 |
| 1_vs_2 | one_cell | 0.984 | 5.26 | 6.97 | 0.97, 0.98 |
| 3_vs_7 | one_cell | 0.664 | 5.24 | 7.59 | 0.95, 0.95 |

## Limits at the maximum standard gain (0.04), 6 s each

| rho | mode | A rms dB | A peak | B rms dB | B peak | clip A/B | resets A | finite | p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| 0.60 | dense_random_evolving | -33.8 | 0.097 | -35.0 | 0.247 | 0/0 | 0 | True | 0.76 |
| 0.60 | random_every_block | -33.5 | 0.095 | -18.4 | 0.516 | 0/0 | 0 | True | 0.89 |
| 0.60 | full_toggle_every_block | -16.6 | 0.431 | -18.6 | 0.529 | 0/0 | 911784 | True | 0.78 |
| 0.60 | repeated_edits_paused | -33.8 | 0.090 | -38.4 | 0.229 | 0/0 | 0 | True | 0.63 |
| 0.95 | dense_random_evolving | -33.9 | 0.091 | -24.0 | 0.315 | 0/0 | 0 | True | 0.62 |
| 0.95 | random_every_block | -33.5 | 0.093 | -9.2 | 0.965 | 0/0 | 0 | True | 0.90 |
| 0.95 | full_toggle_every_block | -16.6 | 0.431 | -9.4 | 0.969 | 0/0 | 911784 | True | 0.84 |
| 0.95 | repeated_edits_paused | -32.8 | 0.097 | -30.5 | 0.261 | 0/0 | 0 | True | 0.68 |

## Delivered catalog

Root: `C:\Users\Astro\Documents\Projects\CASynth-2\lab_catalog\network_n2_events_2026_09_16`

- 20260916-125311-831b09  N2.1 - Ритм: pulsar, 6 поколений/с  -> match   (notes: 'Гипотеза - Периодическая фигура в А буде')
- 20260916-125310-0fc015  N2.2 - Движение: glider, 16 поколений/с  -> match   (notes: 'Гипотеза - При движении фигуры в Б должн')
- 20260916-125309-83b2de  N2.3 - Рост: R-pentomino, 6 поколений/с  -> match   (notes: 'Гипотеза - При нерегулярном росте в Б от')

Observations (not hand-over failures): limits full_toggle_every_block rho 0.6 A: 911784 node resets; limits full_toggle_every_block rho 0.95 A: 911784 node resets.  The plain N1 gutter_field under the same every-block full toggle: 1044547 resets; toggled every 4 blocks: 0 resets.

Summary: no clipping, no resets, no NaN, all records replay exactly
