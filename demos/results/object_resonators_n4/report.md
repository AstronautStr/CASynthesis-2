# N4 object resonators -- hand-over measurements (technical evidence, not listening verdicts)

Generated 2026-09-16 20:54:23 at commit 27bf753.  SR 44100, block 352 (7.98 ms), engine `ca_object_resonators` (output x0.5, gain 0.028 = vol 0.7), scale 220 Hz, decay 0.8 s.  RMS = stereo RMS as in the preflight; (pre) = preflight value.

## Scenes (12 s from a fresh start)

| scene | side | engine | rms dB (pre) | peak (pre) | clip | ids seen | p99 ms |
|---|---|---|---|---|---|---|---|
| n4_spectrum | A | ca_tuned_events | -42.36 | 0.0413 | 0 | - | 0.72 |
| n4_spectrum | B | ca_object_resonators | -40.11 (-40.10) | 0.0561 (0.0561) | 0 | [1] | 0.72 |
| n4_neighbor | A | ca_object_resonators | -35.56 (-35.54) | 0.0817 (0.0817) | 0 | [1, 2] | 1.06 |
| n4_neighbor | B | ca_object_resonators | -35.20 (-35.18) | 0.0855 (0.0855) | 0 | [1, 2] | 1.06 |

- n4_spectrum: A - B = -2.25 dB; tails after Clear (0-0.5 / 1-1.5 / 3-3.5 s): A -45.7/-115.9/-600.0 dB, B -57.4/-600.0/-600.0 dB; continue exact True; late edit differs A True, B True
- n4_neighbor: A - B = -0.35 dB; tails after Clear (0-0.5 / 1-1.5 / 3-3.5 s): A -52.6/-600.0/-600.0 dB, B -52.3/-600.0/-600.0 dB; continue exact True; late edit differs A True, B True

## N4.2 receiver (figure 1) per generation

- Own: receiver e per generation [0.0], blinker e [4.0]
- Disk: receiver e per generation [4.0], blinker e [4.0]
- receiver centre [11.764705882352942, 11.764705882352942], R 5.32962, 16 modes, lowest 80.85 Hz
- Disk - Own difference relative to Own: -11.06 dB (total RMS difference +0.35 dB)
- blinker moved outside the circle: receiver e per generation [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

## N4.1 glider (side B, 24 s at 16 gen/s)

- identities seen [1], 2 distinct spectra, alternating every generation: True (383 generations; spectrum cache hits 380 / misses 4)
  - 183.70, 258.63, 418.46, 456.35 Hz
  - 220.00, 311.13, 440.00, 491.93 Hz

## Stress probes (gain 0.04, 6 s, both sides on the same field)

| detector | scale | decay | mode | A pre-clip peak | B pre-clip peak | clip blocks A/B | figures / sounding / tails B | faded / dropped B | p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 55 | 0.2 | dense_random_evolving | 1.077 | 1.077 | 2/2 | 13/12/20 | 0/0 | 10.49 |
| 0 | 55 | 0.2 | random_every_block | 1.219 | 1.219 | 18/18 | 5/3/96 | 2266/159 | 80.02 |
| 0 | 55 | 0.2 | full_toggle_every_block | 0.164 | 0.164 | 0/0 | 0/0/31 | 0/0 | 5.72 |
| 0 | 55 | 0.2 | repeated_edits_paused | 0.162 | 0.162 | 0/0 | 2/1/1 | 0/0 | 36.62 |
| 0 | 55 | 0.2 | knobs_moving | 1.210 | 1.210 | 4/4 | 18/16/75 | 3/0 | 12.74 |
| 0 | 880 | 1.5 | dense_random_evolving | 0.515 | 0.515 | 0/0 | 26/20/96 | 341/0 | 19.30 |
| 0 | 880 | 1.5 | random_every_block | 0.589 | 0.589 | 0/0 | 4/1/96 | 2267/164 | 78.85 |
| 0 | 880 | 1.5 | full_toggle_every_block | 0.220 | 0.220 | 0/0 | 0/0/96 | 286/0 | 6.35 |
| 0 | 880 | 1.5 | repeated_edits_paused | 0.096 | 0.096 | 0/0 | 4/4/15 | 0/0 | 44.96 |
| 0 | 880 | 1.5 | knobs_moving | 1.126 | 1.126 | 4/4 | 26/23/66 | 0/0 | 12.06 |
| 1 | 55 | 0.2 | dense_random_evolving | 1.072 | 1.072 | 2/2 | 19/19/24 | 0/0 | 10.71 |
| 1 | 55 | 0.2 | random_every_block | 1.173 | 1.173 | 15/15 | 8/6/96 | 2253/158 | 78.08 |
| 1 | 55 | 0.2 | full_toggle_every_block | 0.165 | 0.165 | 0/0 | 0/0/31 | 0/0 | 5.06 |
| 1 | 55 | 0.2 | repeated_edits_paused | 0.217 | 0.217 | 0/0 | 2/2/0 | 0/0 | 45.68 |
| 1 | 55 | 0.2 | knobs_moving | 1.131 | 1.131 | 7/7 | 22/16/76 | 0/0 | 12.31 |
| 1 | 880 | 1.5 | dense_random_evolving | 0.521 | 0.521 | 0/0 | 24/20/96 | 361/0 | 19.02 |
| 1 | 880 | 1.5 | random_every_block | 0.660 | 0.660 | 0/0 | 2/1/96 | 2253/169 | 79.71 |
| 1 | 880 | 1.5 | full_toggle_every_block | 0.221 | 0.221 | 0/0 | 0/0/96 | 286/0 | 6.46 |
| 1 | 880 | 1.5 | repeated_edits_paused | 0.112 | 0.112 | 0/0 | 3/2/12 | 0/0 | 43.50 |
| 1 | 880 | 1.5 | knobs_moving | 1.131 | 1.131 | 5/5 | 18/14/73 | 4/0 | 12.84 |

Tails probe: 48 blinkers -> sounding 24; after 8 clear / re-add cycles (16 blocks) tails 96, fading 24, faded 96, dropped 72; after 3 s of silence tails 0; peak 2.565, finite True.

## Analysis cost by figure size (eigvalsh of one component, this machine)

| cells | ms | over the block budget |
|---|---|---|
| 64 | 0.4 | False |
| 121 | 0.8 | False |
| 256 | 6.7 | False |
| 529 | 33.3 | True |
| 1024 | 92.3 | True |

## Timing (both sides, after warm-up)

- n4_spectrum: p50 0.21  p95 0.58  p99 0.70  max 1.35 ms (budget 7.98, ok True)
- n4_neighbor: p50 0.20  p95 0.56  p99 1.10  max 1.41 ms (budget 7.98, ok True)
- numba: True

## Catalog check (C:\Users\Astro\Documents\Projects\CASynth-2\lab_catalog\object_resonators_n4_2026_09_16)

- 20260916-203922-cbe78b N4.2 - Фигура слышит соседа: неподвижная фигура + : match  (notes: 'Гипотеза - В А после стартового хвоста о')
- 20260916-203922-b08ea2 N4.1 - Спектр самой фигуры: glider, 6 поколений/с: match  (notes: 'Гипотеза - В Б спектр принадлежит глайде')

## Summary

- Scenes: no clipping, no NaN, levels within 0.5 dB of the preflight, tails decay, late edits change the sound, receiver Own 0 / Disk 4 per generation and 0 outside the circle, glider one identity with two alternating spectra, continuation exact, all records replay exactly
- Stress probes: finite in all 20 probes; clipping at gain 0.04 in 8 probes and hard drops in 4 (listed under limitations); worst pre-clip peak 1.219
- Limitations:
  - stress clipping at gain 0.04 (vol 1) in 8 of 20 probes, none in the scenes (peaks <= 0.09): dense_random_evolving det 0 scale 55 decay 0.2: pre-clip peak 1.08, 2 clipped blocks; random_every_block det 0 scale 55 decay 0.2: pre-clip peak 1.22, 18 clipped blocks; knobs_moving det 0 scale 55 decay 0.2: pre-clip peak 1.21, 4 clipped blocks; knobs_moving det 0 scale 880 decay 1.5: pre-clip peak 1.13, 4 clipped blocks; dense_random_evolving det 1 scale 55 decay 0.2: pre-clip peak 1.07, 2 clipped blocks; random_every_block det 1 scale 55 decay 0.2: pre-clip peak 1.17, 15 clipped blocks; knobs_moving det 1 scale 55 decay 0.2: pre-clip peak 1.13, 7 clipped blocks; knobs_moving det 1 scale 880 decay 1.5: pre-clip peak 1.13, 5 clipped blocks. No AGC and no division by the figure count by the REQ; the only common lever is the N4 output x0.5
  - hard drops (the fading pool of 24 full: the quietest fading tail zeroed at once, counted and shown as 'dropped') only under whole-field replacement every block: random_every_block det 0 scale 55: 159 dropped (shown in the panel); random_every_block det 0 scale 880: 164 dropped (shown in the panel); random_every_block det 1 scale 55: 158 dropped (shown in the panel); random_every_block det 1 scale 880: 169 dropped (shown in the panel)
  - 16 of 20 stress probes exceed the block budget at p99 (worst 80.0 ms, modes ['dense_random_evolving', 'knobs_moving', 'random_every_block', 'repeated_edits_paused']): the figure analysis (eigvalsh) of dense fields on the render thread -- offline exact, live underruns possible
  - tails probe (48 blinkers cleared / re-added 8 times within 16 blocks): 72 hard drops, 96 faded, all freed after 3 s of silence
  - N4.1 level: A (N3 Tuned) - B (N4 Disk) = -2.25 dB RMS at the same bench gain; no scene parameter acts on one side only, the only lever would be the N4 output x0.5 (kept at the preflight value; the preflight numbers are reproduced)
  - figure analysis (eigvalsh of L) runs on the render thread at every field change: 64 cells 0 ms, 121 cells 1 ms, 256 cells 7 ms, 529 cells 33 ms, 1024 cells 92 ms -- a component above ~300 cells exceeds the block budget by itself (the two scenes: <= 17 cells)
  - worst stress pre-clip peak 1.219 at gain 0.04 (random_every_block, detector 0, scale 55, decay 0.2 s): measured for these finite probes, not guaranteed for any playing
  - split / merge retire identities (first model): the Tumbler and the R-pentomino change colours often
  - tails are measured on the float output (int16 quantises below -90 dBFS to zero)
  - no listening in this report: hearing the difference is the user's verdict
