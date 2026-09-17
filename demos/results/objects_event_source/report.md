# Objects — event source / modal excitation: measurements

Generated 2026-09-17 23:35 on commit `a3efc67cbff6bc7e1ec5fd4f619e32ce99931a87`; numba True.

## 1. E1 — Births / Deaths against independent masks

72 transitions, period 5; Both = Births + Deaths generation by generation: True.

| Events | start packet | one bank | mismatches | period sequence | periodic | packets | tails / fed | per-mode states |
|---|---|---|---|---|---|---|---|---|
| Births | 16 | True | 0 | [8.0, 16.0, 0.0, 8.0, 8.0] | True | 58 | 0 / 0 | 0 |
| Deaths | 0 | True | 0 | [0.0, 16.0, 8.0, 0.0, 16.0] | True | 43 | 0 / 0 | 0 |
| Both | 16 | True | 0 | [8.0, 32.0, 8.0, 8.0, 24.0] | True | 72 | 0 / 0 | 0 |

Manual events on a still 2 x 2 block (a cell added, then removed, then the figure erased):

| Events | start | birth, death packets | vanished: tail zf / fed modes | feeds blocks | pulse monotone | tail peak first / block 30 |
|---|---|---|---|---|---|---|
| Births | 4 | [1.0, 0.0] | 0.0000 / 0 | 0 | True | 0.0893 / 0.0285 |
| Deaths | 0 | [0.0, 1.0] | 0.6667 / 3 | 4 | True | 0.1006 / 0.0353 |
| Both | 4 | [1.0, 1.0] | 0.0000 / 0 | 0 | True | 0.0834 / 0.0266 |

## 2. M1 — Uniform / Birth position

72 transitions side by side: equal {'frequencies': True, 'weights': True, 'packet_moments': True, 'a': True, 'tracker': True}; packets with events 123, max |sum b^2 - m| 4.44e-16; zero participation 0, refused 0.

| phase bank | measured b | preflight b | max diff | frequencies Hz |
|---|---|---|---|---|
| gen1_11c_3b | [1.2889, 0.0, 1.157] | [1.2889, 0.0, 1.157] | 2.2e-16 | [110.0, 434.7, 551.0] |
| gen1_3c_2b | [1.0, 1.0] | [1.0, 1.0] | 0.0e+00 | [110.0, 110.0] |
| gen2_16c_3b | [0.7815, 0.7037, 1.3762] | [0.7815, 0.7037, 1.3762] | 2.2e-16 | [110.0, 545.0, 659.8] |
| gen3_3c_2b | [1.2247, 0.7071] | [1.2247, 0.7071] | 2.2e-16 | [110.0, 216.2] |
| gen3_3c_1b | [1.2247, 0.7071] | [1.2247, 0.7071] | 0.0e+00 | [110.0, 216.2] |

Single birth on the fixed 11-cell geometry: cells [[8, 9], [8, 12]] -> b [[0.0466, 0.0, 1.7314], [1.0647, 1.3632, 0.0893]] (preflight [[0.0466, 0.0, 1.7314], [1.0647, 1.3632, 0.0893]], max diff 0.0e+00); profile distance 2.3647 (preflight 2.3647).
Zero participation (no born node): b [0.0, 0.0, 0.0], total 0.0, packet False.  Whole-figure birth: max |b - 1| 2.2e-16.
Degenerate groups of E1 gen 1: ranks [2, 8, 1] (preflight [2, 8, 1]); sign flip max error 0.0e+00 (preflight 0.0e+00), basis rotation 2.2e-16 (preflight 5.6e-17).
Translation max error {'20,25': 0.0, '-9,13': 0.0, '28,0': 0.0}; rotation 90 deg max error 2.0e-15.

## 3. The excitation path

Two per-mode packets: superposition max error 2.6e-16; weights after the strikes [0.7, 0.35, 0.2] (set [0.7, 0.35, 0.2]).  Attack 0 / 4 ms on a per-mode packet: first sample 0.5087 / 0.0063, response RMS change -4.39 dB.

## 4. Scenes

| scene | events A/B | excitation A/B | side gain B | A RMS dB (after 2 s) | B RMS dB (after 2 s) | A−B after 2 s (unit) | peak A/B | clip | continue | p99 ms (budget 7.98) |
|---|---|---|---|---|---|---|---|---|---|---|
| oes_e1 | 1/2 | 0/0 | 0.93 | -39.73 (-39.78) | -39.98 (-39.82) | +0.04 (-0.59) | 0.051/0.051 | 0/0 | True | 2.43 |
| oes_m1 | 1/1 | 0/1 | 0.93 | -36.60 (-36.85) | -36.66 (-36.88) | +0.03 (-0.60) | 0.134/0.125 | 0/0 | True | 3.33 |

oes_e1 A: Births / Uniform (supported True), figures 1 sounding 1 tails 0 faded 0 in place 0 dropped 0 refused 0 zero 0
oes_e1 B: Deaths / Uniform (supported False), figures 1 sounding 1 tails 0 faded 0 in place 0 dropped 0 refused 0 zero 0
oes_m1 A: Births / Uniform (supported True), figures 1 sounding 1 tails 26 faded 0 in place 0 dropped 0 refused 0 zero 0
oes_m1 B: Births / Birth position (supported True), figures 1 sounding 1 tails 26 faded 0 in place 0 dropped 0 refused 0 zero 0

## 5. Catalog

- 20260917-233513-cab7d1 E1 - Рождения / смерти: Events Births / Deaths (Octagon II p: replay match , pinned a3efc67cbff6bc7e1ec5fd4f619e32ce99931a87, notes True, side gain {'A': 1.0, 'B': 0.93}
- 20260917-233512-1edfef M1 - Общий удар / место рождения: Excitation Uniform / Birth: replay match , pinned a3efc67cbff6bc7e1ec5fd4f619e32ce99931a87, notes True, side gain {'A': 1.0, 'B': 0.93}

