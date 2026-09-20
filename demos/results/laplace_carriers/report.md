# Laplace carriers -- hand-over measurements (technical evidence, not listening verdicts)

Generated 2026-09-20 16:49:25 at commit cea5127109 (clean).  SR 44100, block 352 (7.98 ms).  Five variants on one Jam evolution: R = `laplacian` (sines), F-saw / F-square = `laplace_carriers` Filter, W-saw / W-square = the same engine, Wave bank; f0 110 Hz, 4 gen/s, 12 s, gain 0.028 = vol 0.7.

## 1. The shared spectral base

- 48 generations, 96 sounding components, 240 modes: every variant reads the SAME analysis (equal frequencies and weights: True)
- worst difference against the Researcher preflight on its 3 phases: 0

## 2. The waveform law

- Wave bank / Sine against the existing `laplacian` engine over 12 s (1488 blocks): byte-exact True (worst |diff| 0 of int16)

| wave | f, Hz | harmonics | table vs direct sum | <= -60 dB |
|---|---|---|---|---|
| Saw | 55.00 | 360 | -76.2 dB | True |
| Saw | 110.00 | 180 | -85.9 dB | True |
| Saw | 190.53 | 104 | -93.1 dB | True |
| Saw | 440.00 | 45 | -103.7 dB | True |
| Saw | 1760.00 | 11 | -121.4 dB | True |
| Saw | 3457.31 | 5 | -129.6 dB | True |
| Saw | 9871.30 | 2 | -145.4 dB | True |
| Saw | 17000.00 | 1 | -145.4 dB | True |
| Square | 55.00 | 180 | -77.8 dB | True |
| Square | 110.00 | 90 | -87.5 dB | True |
| Square | 190.53 | 52 | -94.6 dB | True |
| Square | 440.00 | 23 | -105.4 dB | True |
| Square | 1760.00 | 6 | -123.9 dB | True |
| Square | 3457.31 | 3 | -130.6 dB | True |
| Square | 9871.30 | 1 | -145.4 dB | True |
| Square | 17000.00 | 1 | -145.4 dB | True |

Lines of the ready tracks (dB below the loudest line of that track; the mode is the non-integer 198.2 Hz of the first Jam phase):

| variant | k=f0 | k=2f0 | k=3f0 | mode | 2 x mode | above 0.45*sr |
|---|---|---|---|---|---|---|
| R | 0.0 | -28.5 | -44.4 | -5.8 | -6.4 | -93.9 |
| F_saw | 0.0 | -9.4 | -23.3 | -33.9 | -55.5 | -98.9 |
| F_square | 0.0 | -55.5 | -23.4 | -51.8 | -68.9 | -95.8 |
| W_saw | 0.0 | -0.4 | -7.9 | -5.6 | -5.9 | -91.2 |
| W_square | 0.0 | -28.5 | -8.0 | -5.8 | -6.4 | -93.6 |

## 3. The Filter mask

- against the preflight: worst |dB| 0, worst amplitude 5.55e-17, the non-zero harmonic counts match True
- one mask serves Saw and Square (worst difference on the shared odd harmonics: 0)
- D = 0 equals the plain wave at A = sqrt(sum a^2): max |diff| 1.03e-13
- an empty field and a one-cell figure are silent in both methods (peak int16 {'empty': 0, 'one_cell': 0})
- the carrier has 180 harmonics here, the top one at 19800 Hz

## 4. The ready 12 s

| variant | side gain | RMS dB | RMS dB at gain 1 | peak | clip | centroid Hz | > 1 kHz |
|---|---|---|---|---|---|---|---|
| R | 1.00 | -29.93 | -29.93 | 0.0452 | 0 | 239 | 0.0% |
| F_saw | 0.89 | -29.91 | -28.90 | 0.0180 | 0 | 141 | 0.1% |
| F_square | 0.97 | -29.91 | -29.65 | 0.0148 | 0 | 120 | 0.0% |
| W_saw | 0.78 | -29.94 | -27.79 | 0.0728 | 0 | 695 | 14.7% |
| W_square | 0.90 | -29.96 | -29.04 | 0.0446 | 0 | 517 | 8.0% |

- spread of the calibrated levels: 0.05 dB (every pair is therefore within 1 dB)

| record | A | B | A RMS | B RMS | A-B dB | <=1 dB | A byte-equal | B byte-equal | p99 ms | continue |
|---|---|---|---|---|---|---|---|---|---|---|
| lc_saw_baseline_filter | R | F_saw | -29.93 | -29.91 | -0.01 | True | True | True | 4.44 | True |
| lc_saw_baseline_bank | R | W_saw | -29.93 | -29.94 | +0.02 | True | True | True | 4.62 | True |
| lc_square_baseline_filter | R | F_square | -29.93 | -29.91 | -0.01 | True | True | True | 4.39 | True |
| lc_square_baseline_bank | R | W_square | -29.93 | -29.96 | +0.03 | True | True | True | 4.42 | True |
| lc_saw_filter_bank | F_saw | W_saw | -29.91 | -29.94 | +0.03 | True | True | True | 4.66 | True |
| lc_square_filter_bank | F_square | W_square | -29.91 | -29.96 | +0.05 | True | True | True | 4.65 | True |

(“byte-equal” = that side is bit-for-bit the stand-alone track of its variant, so a variant sounds identical in every record it appears in.)

## 5. Catalog check (lab_catalog\laplace_carriers_2026_09_20)

- 20260920-164918-277956 1 - Пила через лапласиановский фильтр: Laplace / Fil: match  (notes: 'Гипотеза - А — исходный Laplacian на син')
- 20260920-164914-f2dedf 2 - Пила на каждой моде: Laplace / Wave bank (Jam, 4: match  (notes: 'Гипотеза - А — исходный Laplacian на син')
- 20260920-164910-795d5f 3 - Меандр через лапласиановский фильтр: Laplace / F: match  (notes: 'Гипотеза - А — исходный Laplacian на син')
- 20260920-164906-98c8aa 4 - Меандр на каждой моде: Laplace / Wave bank (Jam,: match  (notes: 'Гипотеза - А — исходный Laplacian на син')
- 20260920-164902-181f45 5 - Два способа с пилой: Filter / Wave bank (Jam, 4 : match  (notes: 'Гипотеза - Теперь сравниваем два вариант')
- 20260920-164859-4c4922 6 - Два способа с меандром: Filter / Wave bank (Jam,: match  (notes: 'Гипотеза - Теперь сравниваем два вариант')

## 6. Limits outside the prepared scenes

| variant | settings | p99 ms | max ms | over budget | pre-clip peak | clip blocks |
|---|---|---|---|---|---|---|
| R | n 20, dense random field, vol 1.0 | 27.99 | 29.50 | True | 0.698 | 315 |
| F_saw | n 20, dense random field, vol 1.0 | 22.65 | 24.50 | True | 0.739 | 0 |
| W_saw | n 20, dense random field, vol 1.0 | 69.26 | 84.87 | True | 1.064 | 429 |

## Summary

- Wave bank / Sine is the existing engine byte-exact over 12 s: True
- Filter coefficients equal the preflight (worst 5.6e-17) and one mask serves both waves
- Table error at or below -76 dB, REQ asks for -60 dB
- Six records: levels within 1 dB, Continue exact, budget kept, each variant byte-identical across records: True
- Timbre of the ready tracks (spectral centroid): R 239 Hz, Filter 141 / 120 Hz (DARKER than the baseline: with three modes the 24 dB mask leaves few harmonics of the carrier), Wave bank 695 / 517 Hz (15% / 8% of the energy above 1 kHz against 0% of R) -- the change is in the audio, not in a clip or a level drop.
- Limits: the prepared scenes stay inside the block budget; a dense random field with n = 20 modes is measured in section 6 (the baseline `laplacian` is over the budget there too; the Wave bank additionally builds one band-limited table per new mode frequency) and is outside the delivered material.
