# Objects — Birth strength (M2): measurements

Generated 2026-09-18 01:30 on commit `ecb464bf3213d20dbe5ef7a93a2d940afa09108f`; numba True.

## 1. The law against the independent preflight

Law: s=0: Uniform. 0<s<1: normalize_L2((1-s)+s*b) to sqrt(m). s=1: return original b without renormalization. 1<s<=4: normalize_L2(b**s) to sqrt(m). No births: no packet at every s. Zero participation with births: Uniform at s=0, no packet at s>0.

10 cases x 5 strengths: max |c − preflight| 4.4e-16, max |Σc² − m| 8.9e-16; s = 1 returns b itself True; equal b stay equal True; one mode c = 1 True; all c ≥ 0 True; an exact zero at s = 0.5 / 1 / 2 / 4: {'0.5': 0.5263540128930309, '1': 0.0, '2': 0.0, '4': 0.0}.

Control example (16 cells, three births, 110 / 545 / 660 Hz):

| s | c1 | c2 | c3 |
|---|---|---|---|
| 0 | 1.000000 | 1.000000 | 1.000000 |
| 0.5 | 0.901214 | 0.861881 | 1.202071 |
| 1 | 0.781500 | 0.703746 | 1.376227 |
| 2 | 0.515832 | 0.418295 | 1.599671 |
| 4 | 0.178722 | 0.117524 | 1.718792 |

| case | b | s 0 | s 0.5 | s 1 | s 2 | s 4 | max diff |
|---|---|---|---|---|---|---|---|
| gen1_11c_3b | [1.28891, 0.0, 1.157027] | [1.0, 1.0, 1.0] | [1.201, 0.525, 1.132] | [1.289, 0.0, 1.157] | [1.349, 0.0, 1.087] | [1.453, 0.0, 0.943] | 2.2e-16 |
| gen1_3c_2b | [1.0, 1.0] | [1.0, 1.0] | [1.0, 1.0] | [1.0, 1.0] | [1.0, 1.0] | [1.0, 1.0] | 0.0e+00 |
| gen2_16c_3b | [0.7815, 0.703746, 1.376227] | [1.0, 1.0, 1.0] | [0.901, 0.862, 1.202] | [0.781, 0.704, 1.376] | [0.516, 0.418, 1.6] | [0.179, 0.118, 1.719] | 4.4e-16 |
| gen3_7c_0b | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | 0.0e+00 |
| gen3_3c_2b | [1.224745, 0.707107] | [1.0, 1.0] | [1.122, 0.861] | [1.225, 0.707] | [1.342, 0.447] | [1.406, 0.156] | 2.2e-16 |
| gen3_3c_1b | [1.224745, 0.707107] | [1.0, 1.0] | [1.122, 0.861] | [1.225, 0.707] | [1.342, 0.447] | [1.406, 0.156] | 2.2e-16 |
| equal_modes | [1.0, 1.0, 1.0] | [1.0, 1.0, 1.0] | [1.0, 1.0, 1.0] | [1.0, 1.0, 1.0] | [1.0, 1.0, 1.0] | [1.0, 1.0, 1.0] | 0.0e+00 |
| one_mode | [1.0] | [1.0] | [1.0] | [1.0] | [1.0] | [1.0] | 0.0e+00 |
| exact_zero_component | [0.0, 1.0, 1.414214] | [1.0, 1.0, 1.0] | [0.526, 1.053, 1.271] | [0.0, 1.0, 1.414] | [0.0, 0.775, 1.549] | [0.0, 0.42, 1.68] | 2.2e-16 |
| zero_participation_with_births | [0.0, 0.0, 0.0] | [1.0, 1.0, 1.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | 0.0e+00 |

## 2. Five engines on the M1 field

72 transitions side by side (s = 0 / 0.5 / 1 / 2 / 4): equal {'frequencies': True, 'weights': True, 'packet_moments': True, 'a': True, 'tracker': True}; packets with events 123; max |Σc² − m| 8.9e-16.

| phase bank | frequencies Hz | s | measured c | preflight c | max diff |
|---|---|---|---|---|---|
| gen1_11c_3b | [110, 434.7, 551.0] | 0 | [1.0, 1.0, 1.0] | [1.0, 1.0, 1.0] | 0.0e+00 |
| gen1_11c_3b | [110, 434.7, 551.0] | 0.5 | [1.2013, 0.5248, 1.132] | [1.2013, 0.5248, 1.132] | 2.2e-16 |
| gen1_11c_3b | [110, 434.7, 551.0] | 1 | [1.2889, 0.0, 1.157] | [1.2889, 0.0, 1.157] | 2.2e-16 |
| gen1_11c_3b | [110, 434.7, 551.0] | 2 | [1.3487, 0.0, 1.0868] | [1.3487, 0.0, 1.0868] | 2.2e-16 |
| gen1_11c_3b | [110, 434.7, 551.0] | 4 | [1.4527, 0.0, 0.9433] | [1.4527, 0.0, 0.9433] | 4.4e-16 |
| gen1_3c_2b | [110, 110] | 0 | [1.0, 1.0] | [1.0, 1.0] | 0.0e+00 |
| gen1_3c_2b | [110, 110] | 0.5 | [1.0, 1.0] | [1.0, 1.0] | 0.0e+00 |
| gen1_3c_2b | [110, 110] | 1 | [1.0, 1.0] | [1.0, 1.0] | 0.0e+00 |
| gen1_3c_2b | [110, 110] | 2 | [1.0, 1.0] | [1.0, 1.0] | 0.0e+00 |
| gen1_3c_2b | [110, 110] | 4 | [1.0, 1.0] | [1.0, 1.0] | 0.0e+00 |
| gen2_16c_3b | [110, 545.0, 659.8] | 0 | [1.0, 1.0, 1.0] | [1.0, 1.0, 1.0] | 0.0e+00 |
| gen2_16c_3b | [110, 545.0, 659.8] | 0.5 | [0.9012, 0.8619, 1.2021] | [0.9012, 0.8619, 1.2021] | 0.0e+00 |
| gen2_16c_3b | [110, 545.0, 659.8] | 1 | [0.7815, 0.7037, 1.3762] | [0.7815, 0.7037, 1.3762] | 2.2e-16 |
| gen2_16c_3b | [110, 545.0, 659.8] | 2 | [0.5158, 0.4183, 1.5997] | [0.5158, 0.4183, 1.5997] | 4.4e-16 |
| gen2_16c_3b | [110, 545.0, 659.8] | 4 | [0.1787, 0.1175, 1.7188] | [0.1787, 0.1175, 1.7188] | 2.2e-16 |
| gen3_3c_2b | [110, 216.2] | 0 | [1.0, 1.0] | [1.0, 1.0] | 0.0e+00 |
| gen3_3c_2b | [110, 216.2] | 0.5 | [1.122, 0.8609] | [1.122, 0.8609] | 2.2e-16 |
| gen3_3c_2b | [110, 216.2] | 1 | [1.2247, 0.7071] | [1.2247, 0.7071] | 2.2e-16 |
| gen3_3c_2b | [110, 216.2] | 2 | [1.3416, 0.4472] | [1.3416, 0.4472] | 5.6e-17 |
| gen3_3c_2b | [110, 216.2] | 4 | [1.4056, 0.1562] | [1.4056, 0.1562] | 2.2e-16 |
| gen3_3c_1b | [110, 216.2] | 0 | [1.0, 1.0] | [1.0, 1.0] | 0.0e+00 |
| gen3_3c_1b | [110, 216.2] | 0.5 | [1.122, 0.8609] | [1.122, 0.8609] | 2.2e-16 |
| gen3_3c_1b | [110, 216.2] | 1 | [1.2247, 0.7071] | [1.2247, 0.7071] | 0.0e+00 |
| gen3_3c_1b | [110, 216.2] | 2 | [1.3416, 0.4472] | [1.3416, 0.4472] | 2.2e-16 |
| gen3_3c_1b | [110, 216.2] | 4 | [1.4056, 0.1562] | [1.4056, 0.1562] | 0.0e+00 |

States at the end: s 0: per-mode max 0, uniform max 3.27e-292, zero 0, refused 0; s 0.5: per-mode max 3.67e-292, uniform max 0, zero 0, refused 0; s 1: per-mode max 4.01e-292, uniform max 0, zero 0, refused 0; s 2: per-mode max 4.39e-292, uniform max 0, zero 0, refused 0; s 4: per-mode max 4.6e-292, uniform max 0, zero 0, refused 0.

## 3. The PCM path

Birth position at s = 0 equals Uniform on the whole M2 scene render: True.  Uniform at s = 4 equals Uniform at s = 1: True.  s = 1 vs s = 4 max |ΔPCM| 795.

The pinned records of 2026-09-17 (scene without the key, rendered with the current code and the default s = 1) against their stored WAVs:

- 20260917-233513-cab7d1 E1 - Рождения / смерти: Events Births / Deaths (Octagon II p: {'A': True, 'B': True, 'monitor': True} (pinned a3efc67cbff6bc7e1ec5fd4f619e32ce99931a87)
- 20260917-233512-1edfef M1 - Общий удар / место рождения: Excitation Uniform / Birth: {'A': True, 'B': True, 'monitor': True} (pinned a3efc67cbff6bc7e1ec5fd4f619e32ce99931a87)

s changed on a still field through 0 and 1: no packet, tail and states equal to a twin True, change counter equal True.

A sequence of s across the E1 transitions (one bank; the per-mode history of s > 0 is kept at s = 0, the uniform path adds):

| transition | s | e | c | per-mode max before / after | uniform max before / after |
|---|---|---|---|---|---|
| 1 | 4 | 8 | [0.7319, 1.4345, 0.6376] | 2.24e-42 / 1.56e-14 | 0 / 0 |
| 2 | 0 | 16 | [1.0, 1.0, 1.0] | 1.56e-14 / 2.13e-28 | 0 / 1.21e-14 |
| 3 | 0.5 | 0 | [0.0, 0.0, 0.0] | 2.13e-28 / 2.9e-42 | 1.21e-14 / 1.65e-28 |
| 4 | 1 | 8 | [1.1061, 1.3085, 0.2538] | 2.9e-42 / 1.43e-14 | 1.65e-28 / 2.24e-42 |
| 5 | 2 | 8 | [1.0, 1.0, 1.0] | 1.43e-14 / 1.09e-14 | 2.24e-42 / 3.06e-56 |
| 6 | 0 | 8 | [1.0, 1.0, 1.0] | 1.09e-14 / 1.48e-28 | 3.06e-56 / 1.09e-14 |

Runner journal with knob changes [(1.5, 2.0), (2.5, 0.0), (3.5, 4.0), (4.5, 0.5), (5.5, 1.0)]: Continue from a 6 s snapshot exact True (B strength at the snapshot 1.0).  A v4 snapshot imports as s = 1 with the same sound: True.

## 4. Levels and timing

Side B at every s (A at 1), whole take and after 2 s:

| gains | s | side gain B | RMS dB | after 2 s dB | peak | clip | finite |
|---|---|---|---|---|---|---|---|
| unit | 0 | 1.00 | -36.60 | -36.85 | 0.134 | 0 | True |
| unit | 0.5 | 1.00 | -36.21 | -36.44 | 0.134 | 0 | True |
| unit | 1 | 1.00 | -36.03 | -36.25 | 0.134 | 0 | True |
| unit | 2 | 1.00 | -35.80 | -36.00 | 0.134 | 0 | True |
| unit | 4 | 1.00 | -35.34 | -35.52 | 0.134 | 0 | True |
| delivered | 0 | 0.92 | -37.32 | -37.57 | 0.123 | 0 | True |
| delivered | 0.5 | 0.92 | -36.94 | -37.16 | 0.123 | 0 | True |
| delivered | 1 | 0.92 | -36.76 | -36.97 | 0.123 | 0 | True |
| delivered | 2 | 0.92 | -36.52 | -36.72 | 0.123 | 0 | True |
| delivered | 4 | 0.92 | -36.07 | -36.24 | 0.123 | 0 | True |

| scene | strength A/B | side gain B | A RMS dB (after 2 s) | B RMS dB (after 2 s) | A−B after 2 s (unit) | peak A/B | clip | continue | p99 ms (budget 7.98) |
|---|---|---|---|---|---|---|---|---|---|
| obs_m2 | 1/4 | 0.92 | -36.03 (-36.25) | -36.07 (-36.24) | -0.01 (-0.73) | 0.134/0.123 | 0/0 | True | 3.59 |
obs_m2 A: Birth position s 1 (supported True), figures 1 sounding 1 tails 26 faded 0 in place 0 dropped 0 refused 0 zero 0
obs_m2 B: Birth position s 4 (supported True), figures 1 sounding 1 tails 26 faded 0 in place 0 dropped 0 refused 0 zero 0

Block time p99 / max (ms) of the pair at every s of side B, and with the knob moved every second (6 s runs):

| run | p99 ms | max ms | peak B | clip B | finite |
|---|---|---|---|---|---|
| 0 | 2.88 | 3.26 | 0.006 | 0 | True |
| 0.5 | 3.09 | 3.76 | 0.007 | 0 | True |
| 1 | 3.43 | 4.73 | 0.008 | 0 | True |
| 2 | 3.62 | 4.13 | 0.007 | 0 | True |
| 4 | 3.35 | 5.33 | 0.010 | 0 | True |
| knob_moved | 3.40 | 4.00 | 0.008 | 0 | True |

Budget 7.98 ms; all p99 within the budget: True.

## 5. Catalog

- 20260918-013020-88dc88 M2 - Сила места рождения: Birth strength 1 / 4 (Jam p3): replay match , pinned ecb464bf3213d20dbe5ef7a93a2d940afa09108f, notes True, side gain {'A': 1.0, 'B': 0.92}
