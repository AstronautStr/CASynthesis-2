# N3 tuned events -- hand-over measurements (technical evidence, not listening verdicts)

Generated 2026-09-16 16:51:25 at commit 5aea4f4.  SR 44100, block 352 (7.98 ms), one engine `ca_tuned_events` on both sides (A field_tuning 0, B field_tuning 1, decay 0.8 s), rho 0.88, output x4, gain 0.028 (vol 0.7).  RMS = stereo RMS as in the preflight; (pre) = preflight value.

## Scenes (from a fresh start; windows in seconds)

| scene | window | A rms dB (pre) | B rms dB (pre) | A peak (pre) | B peak (pre) | A-B dB |
|---|---|---|---|---|---|---|
| n3_cycles | octagon | -36.9 (-36.9) | -36.5 (-36.5) | 0.060 (0.060) | 0.067 (0.067) | -0.45 |
| n3_cycles | swap | -37.5 | -37.1 | 0.059 | 0.069 | -0.40 |
| n3_cycles | tumbler | -38.1 (-38.1) | -37.5 (-37.5) | 0.054 (0.054) | 0.069 (0.069) | -0.54 |
| n3_travel | main | -37.6 (-37.6) | -37.7 (-37.7) | 0.057 (0.057) | 0.062 (0.062) | 0.11 |
| n3_growth | main | -34.9 (-34.9) | -33.9 (-33.9) | 0.087 (0.087) | 0.105 (0.105) | -0.98 |

| scene | period | events/step (72 gens) | gens | clip A/B | finite | A tail 1/2/3 s dB | B tail 1/2/3 s dB | spectral dist A-B dB | p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| n3_cycles | 5 / 14 | 8-32 | 143 | 0/0 | True | -54 / -112 / -161 | -54 / -111 / -155 | 4.19 | 0.60 |
| n3_travel | 128 | 4-4 | 191 | 0/0 | True | -52 / -113 / -166 | -50 / -116 / -163 | 4.30 | 0.83 |
| n3_growth | None | 2-78 | 71 | 0/0 | True | -51 / -109 / -161 | -50 / -113 / -161 | 4.90 | 0.60 |

N3.1 swap: 1 `set_cells` command of 1024 cells at sample 529408 (12.0047 s; first block boundary at or after 12 s: True), generation 72 at that block, largest packet a of the swap block 0.838.

## Continuation and a late edit (same saved end state)

| scene | gens in 4 s | field moved | edit cells | A differs / spectral dist dB | B differs / spectral dist dB |
|---|---|---|---|---|---|
| n3_cycles | 25 | True | 36 | True / 7.97 | True / 9.54 |
| n3_travel | 65 | True | 10 | True / 5.43 | True / 6.21 |
| n3_growth | 25 | True | 130 | True / 8.25 | True / 10.78 |

## Control probes (engine level, 2 s, raw output)

- Frequencies held (fixed mode), impulses from two fields: frequencies equal True, max |diff| 0.264, envelope corr 0.976, spectral dist 3.89 dB.
- Impulse history held (one field), fixed vs field tuning: network states equal True, frequencies equal False, max |diff| 0.340, envelope corr 0.941, spectral dist 4.68 dB.
- Silent retune (empty field, switch toggled): output peak 0.0e+00.
- Excited tail across a retune: states kept True, RMS per second after it -74, -129, -174, -215, -257 dB, events after 0.

## Stress probes at the maximum standard gain (0.04), 6 s each, both sides

| decay s | mode | A rms dB | A peak | B rms dB | B peak | clip A/B | finite | p99 ms |
|---|---|---|---|---|---|---|---|---|
| 0.20 | dense_random_evolving | -32.1 | 0.112 | -33.0 | 0.109 | 0/0 | True | 0.86 |
| 0.20 | random_every_block | -13.9 | 0.419 | -19.0 | 0.349 | 0/0 | True | 1.79 |
| 0.20 | full_toggle_every_block | -14.1 | 0.466 | -14.5 | 0.692 | 0/0 | True | 1.65 |
| 0.20 | repeated_edits_paused | -34.8 | 0.099 | -34.9 | 0.087 | 0/0 | True | 1.44 |
| 0.20 | knobs_moving | -27.8 | 0.210 | -27.8 | 0.210 | 0/0 | True | 1.22 |
| 1.50 | dense_random_evolving | -26.4 | 0.159 | -26.5 | 0.205 | 0/0 | True | 0.90 |
| 1.50 | random_every_block | -7.6 | 0.889 | -11.3 | 0.820 | 0/0 | True | 1.69 |
| 1.50 | full_toggle_every_block | -7.8 | 0.987 | -9.0 | 1.000 | 0/104 | True | 1.79 |
| 1.50 | repeated_edits_paused | -29.1 | 0.148 | -28.1 | 0.187 | 0/0 | True | 0.84 |
| 1.50 | knobs_moving | -27.8 | 0.193 | -27.8 | 0.193 | 0/0 | True | 1.06 |

Timing (n3_travel, 8 s, both sides per block): p50 0.42 / p95 1.20 / p99 1.67 / max 2.64 ms of 7.98 ms (numba True).

## Delivered catalog

Root: `C:\Users\Astro\Documents\Projects\CASynth-2\lab_catalog\network_n3_combined_2026_09_16`

- 20260916-163014-75fdd6  N3.1 - Два цикла: Octagon II, затем Tumbler, 6 поколений/с  -> match   (notes: 'Гипотеза - В обоих вариантах удары идут ')
- 20260916-163012-4d425b  N3.2 - Движение: glider, 16 поколений/с  -> match   (notes: 'Гипотеза - В А движение меняет распредел')
- 20260916-163011-25bd05  N3.3 - Рост: R-pentomino, 6 поколений/с  -> match   (notes: 'Гипотеза - При нерегулярном росте Б буде')

## Summary

- Scenes: no clipping, no NaN, tails decay, late edits change the sound, control probes as expected, all records replay exactly
- Stress probes: limits full_toggle_every_block decay 1.5 B: clip / non-finite
- Limitations:
  - stress peak 1.000 at gain 0.04 (full_toggle_every_block, decay 1.5 s): the headroom is measured for these finite probes, not guaranteed for any playing
  - the model has no reset guard (linear banks, bounded tanh network): 'resets' are not a quantity here
  - preflight Tumbler values come from a fresh 12 s start; here the Tumbler window follows the swap with the Octagon tail still ringing
  - tails are measured on the float output (int16 quantises below -90 dBFS to zero)
  - no listening in this report: hearing the difference is the user's verdict
