# Objects — radius range / attack: measurements

Generated 2026-09-17 17:36 on commit `8a8fbfbb897697857564133af908b7963c4f2b10`; numba True.

## 1. Radius pairs against independent masks

**receiver** — joint evolution = union of the independent ones: True, 71 transitions.

| Radius x | packet mismatches | foreign events (fam 0 / 1) | own events | packets | preflight foreign / packets | match |
|---|---|---|---|---|---|---|
| 1.0 | 0 | [0, 0] | [0, 284] | [0, 71] | [0, 0] / [0, 71] | True |
| 2.0 | 0 | [284, 0] | [0, 284] | [71, 71] | [284, 0] / [71, 71] | True |

Packet sizes at x1.0: family 0 {'0': 71}, family 1 {'4': 71}
Packet sizes at x2.0: family 0 {'4': 71}, family 1 {'4': 71}

**p3_p5** — joint evolution = union of the independent ones: True, 71 transitions.

| Radius x | packet mismatches | foreign events (fam 0 / 1) | own events | packets | preflight foreign / packets | match |
|---|---|---|---|---|---|---|
| 1.0 | 0 | [0, 0] | [357, 1128] | [118, 71] | [0, 0] / [118, 71] | True |
| 8.0 | 0 | [392, 519] | [615, 1128] | [141, 71] | [392, 519] / [141, 71] | True |
| 32.0 | 0 | [1312, 519] | [615, 1128] | [141, 71] | [1312, 519] / [141, 71] | True |

Packet sizes at x1.0: family 0 {'0': 23, '1': 23, '2': 23, '3': 24, '4': 24, '5': 24}, family 1 {'8': 43, '24': 14, '32': 14}
Packet sizes at x8.0: family 0 {'3': 74, '5': 4, '9': 24, '11': 14, '13': 15, '19': 5, '21': 5}, family 1 {'12': 14, '17': 29, '28': 5, '33': 9, '36': 5, '41': 9}
Packet sizes at x32.0: family 0 {'3': 20, '5': 4, '11': 56, '13': 15, '17': 14, '19': 17, '21': 5, '33': 5, '41': 5}, family 1 {'12': 14, '17': 29, '28': 5, '33': 9, '36': 5, '41': 9}

R1 blinker shifted 8 columns right: at x2 receiver e values [0.0] (R_eff 10.66), at x3 [4.0] (R_eff 15.99); the blinker's own e [4.0] / [4.0].

## 2. Attack

| Attack ms | q | pulse first sample | pulse peak | HF > 3 kHz change dB | response RMS change dB | adjacent step at equal RMS dB | preflight HF / RMS / step dB |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.000000 | 0.4500 | 0.4500 | +0.00 | +0.00 | +0.00 | +0.00 / +0.00 / +0.00 |
| 1.0 | 0.951397 | 0.0219 | 0.1126 | -23.10 | -0.82 | -11.08 | -23.08 / -0.81 / -10.94 |
| 4.0 | 0.987621 | 0.0056 | 0.0398 | -35.11 | -4.78 | -15.23 | -35.09 / -4.77 / -15.38 |
| 10.0 | 0.995030 | 0.0022 | 0.0177 | -43.07 | -11.05 | -16.30 | -43.04 / -11.03 / -16.58 |
| 20.0 | 0.997512 | 0.0011 | 0.0092 | -49.05 | -16.76 | -16.70 | -49.06 / -16.74 / -16.72 |

A1 field (Radius 32, unit side gains): packet moments identical True; B − A RMS -5.82 dB, energy above 3 kHz -28.71 dB, max adjacent step at equal RMS -14.75 dB; peaks A 0.263 B 0.125.

## 3. Scenes

| scene | radius A/B | attack A/B | range | side gain B | A RMS dB | B RMS dB | A−B dB (unit) | peak A/B | clip | continue | p99 ms (budget 7.98) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ora_r1 | 1.0/2.0 | 0.0/0.0 | [0.25, 4.0] | 0.65 | -35.38 | -35.32 | -0.06 (-3.80) | 0.092/0.060 | 0/0 | True | 2.16 |
| ora_r2 | 1.0/32.0 | 0.0/0.0 | [0.25, 32.0] | 0.69 | -28.85 | -28.81 | -0.04 (-3.26) | 0.175/0.182 | 0/0 | True | 3.61 |
| ora_a1 | 32.0/32.0 | 0.0/4.0 | [0.25, 32.0] | 2.05 | -25.59 | -25.60 | +0.01 (+6.25) | 0.263/0.257 | 0/0 | True | 4.55 |
| ora_user034 | 0.34/0.34 | 0.0/0.0 | [0.25, 1.0] | 1.0 | -45.58 | -45.58 | +0.00 (+0.00) | 0.049/0.049 | 0/0 | True | 4.68 |

ora_r1: B figures 2 sounding 2 tails 0 covers_all 0 faded 0 in place 0 dropped 0
ora_r2: B figures 2 sounding 2 tails 34 covers_all 2 faded 0 in place 0 dropped 0
ora_a1: B figures 2 sounding 2 tails 34 covers_all 2 faded 0 in place 0 dropped 0
ora_user034: B figures 2 sounding 2 tails 6 covers_all 0 faded 0 in place 0 dropped 0

## 4. Catalog

- 20260917-173619-a0ec24 R1 - Сосед возбуждает неподвижную фигуру: Radius x 1 / 2 (17: replay match , pinned 8a8fbfbb897697857564133af908b7963c4f2b10, notes True, side gain {'A': 1.0, 'B': 0.65}, ranges {'A': {'ca_object_resonators': {'radius_mul': [0.25, 4.0]}}, 'B': {'ca_object_resonators': {'radius_mul': [0.25, 4.0]}}}
- 20260917-173618-0d6b45 R2 - Два периода: самостоятельные и общий приём: Radius x 1 : replay match , pinned 8a8fbfbb897697857564133af908b7963c4f2b10, notes True, side gain {'A': 1.0, 'B': 0.69}, ranges {'A': {'ca_object_resonators': {'radius_mul': [0.25, 32.0]}}, 'B': {'ca_object_resonators': {'radius_mul': [0.25, 32.0]}}}
- 20260917-173617-6b6728 A1 - Резкость возбуждения: Attack 0 / 4 ms (Jam p3 + Octagon: replay match , pinned 8a8fbfbb897697857564133af908b7963c4f2b10, notes True, side gain {'A': 1.0, 'B': 2.05}, ranges {'A': {'ca_object_resonators': {'radius_mul': [0.25, 32.0]}}, 'B': {'ca_object_resonators': {'radius_mul': [0.25, 32.0]}}}
- 20260917-173616-2bb6ae U0 - Пресет пользователя: Radius x 0.34 (Jam p3 + Octagon II: replay match , pinned 8a8fbfbb897697857564133af908b7963c4f2b10, notes True, side gain {'A': 1.0, 'B': 1.0}, ranges {'A': {'ca_object_resonators': {'radius_mul': [0.25, 1.0]}}, 'B': {'ca_object_resonators': {'radius_mul': [0.25, 1.0]}}}

