# N1 `gutter_field` -- hand-over measurements

Generated 2026-09-15 20:23:55 by demos/gutter_field_n1_report.py; model `gutter_field_n1_v1`, numba 0.67.0.  Numbers only; nothing about hearing.

## Fixtures (u / ratio / bank frequencies)

| field | counts | u match | ratio max err | f min..max Hz |
|---|---|---|---|---|
| vertical | [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0] | True | 0.0e+00 | 39.7..5485.8 |
| horizontal | [3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0] | True | 0.0e+00 | 25.0..6301.5 |
| moved | [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0] | True | 0.0e+00 | 25.0..6334.6 |

## Invariance (edits while depth = 0 / Freeze CA)

- depth_0: B PCM identical with 48 cell edits = **True** (A identical: False)
- freeze_ca: B PCM identical with 48 cell edits = **True** (A identical: False)

## A / B on the prepared scene (12 s, 2 gen/s)

generations 23, clip blocks {'A': 0, 'B': 0}, resets {'A': 0, 'B': 0}, peak {'A': 0.4160283211767937, 'B': 0.4084597308267464}, identical until the first step: True, identical whole: False

### A -- follows the field

| t0 s | rms dB | centroid Hz | flat dB | 20-100 | 100-500 | 500-2k | 2k-16k | flux | env std dB |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -20.4 | 2263 | -43.3 | 0.01 | 0.02 | 0.30 | 0.68 | 0.0226 | 0.93 |
| 2 | -20.4 | 2242 | -43.2 | 0.01 | 0.03 | 0.29 | 0.67 | 0.0203 | 0.09 |
| 4 | -20.5 | 2035 | -44.4 | 0.01 | 0.10 | 0.31 | 0.59 | 0.0193 | 0.07 |
| 6 | -20.6 | 2010 | -44.6 | 0.00 | 0.10 | 0.32 | 0.58 | 0.0199 | 0.08 |
| 8 | -20.4 | 2040 | -44.7 | 0.00 | 0.10 | 0.31 | 0.59 | 0.0192 | 0.10 |
| 10 | -20.2 | 2118 | -43.1 | 0.01 | 0.09 | 0.28 | 0.62 | 0.0197 | 0.06 |

### B -- Freeze CA

| t0 s | rms dB | centroid Hz | flat dB | 20-100 | 100-500 | 500-2k | 2k-16k | flux | env std dB |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -20.4 | 2501 | -47.0 | 0.00 | 0.03 | 0.30 | 0.67 | 0.0166 | 0.95 |
| 2 | -20.3 | 2345 | -46.3 | 0.00 | 0.11 | 0.24 | 0.64 | 0.0122 | 0.06 |
| 4 | -20.5 | 1988 | -48.4 | 0.00 | 0.29 | 0.17 | 0.54 | 0.0080 | 0.08 |
| 6 | -20.6 | 1958 | -48.7 | 0.00 | 0.29 | 0.17 | 0.53 | 0.0079 | 0.01 |
| 8 | -20.4 | 2013 | -48.4 | 0.00 | 0.28 | 0.17 | 0.55 | 0.0080 | 0.11 |
| 10 | -20.1 | 2130 | -48.0 | 0.00 | 0.27 | 0.14 | 0.59 | 0.0086 | 0.06 |

## Edit schedule (20 s, CA paused, vertical / moved every 4 s, no restart)

edits applied at block starts (s): [0.0, 4.007, 8.006, 12.005, 16.004]; clip blocks {'A': 0, 'B': 0}, resets {'A': 0, 'B': 0}, peak {'A': 0.47111423078096865, 'B': 0.423963133640553}

### A -- follows the edits

| t0 s | rms dB | centroid Hz | flat dB | 20-100 | 100-500 | 500-2k | 2k-16k | flux | env std dB |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -20.4 | 2501 | -47.0 | 0.00 | 0.03 | 0.30 | 0.67 | 0.0166 | 0.95 |
| 2 | -20.3 | 2345 | -46.3 | 0.00 | 0.11 | 0.24 | 0.64 | 0.0122 | 0.06 |
| 4 | -20.4 | 1061 | -46.3 | 0.10 | 0.42 | 0.38 | 0.10 | 0.0233 | 0.44 |
| 6 | -20.1 | 481 | -56.1 | 0.22 | 0.42 | 0.36 | 0.00 | 0.0306 | 0.22 |
| 8 | -19.2 | 566 | -52.0 | 0.49 | 0.25 | 0.11 | 0.15 | 0.0308 | 1.60 |
| 10 | -18.6 | 358 | -52.1 | 0.66 | 0.18 | 0.07 | 0.09 | 0.0319 | 2.08 |
| 12 | -19.6 | 259 | -57.8 | 0.53 | 0.35 | 0.13 | 0.00 | 0.0379 | 1.24 |
| 14 | -20.2 | 247 | -58.0 | 0.56 | 0.31 | 0.13 | 0.00 | 0.0556 | 1.10 |
| 16 | -19.8 | 378 | -54.1 | 0.38 | 0.50 | 0.02 | 0.10 | 0.0428 | 0.23 |
| 18 | -19.9 | 361 | -55.8 | 0.36 | 0.53 | 0.02 | 0.09 | 0.0427 | 0.73 |

### B -- Freeze CA (control)

| t0 s | rms dB | centroid Hz | flat dB | 20-100 | 100-500 | 500-2k | 2k-16k | flux | env std dB |
|---|---|---|---|---|---|---|---|---|---|
| 0 | -20.4 | 2501 | -47.0 | 0.00 | 0.03 | 0.30 | 0.67 | 0.0166 | 0.95 |
| 2 | -20.3 | 2345 | -46.3 | 0.00 | 0.11 | 0.24 | 0.64 | 0.0122 | 0.06 |
| 4 | -20.5 | 1988 | -48.4 | 0.00 | 0.29 | 0.17 | 0.54 | 0.0080 | 0.08 |
| 6 | -20.6 | 1958 | -48.7 | 0.00 | 0.29 | 0.17 | 0.53 | 0.0079 | 0.01 |
| 8 | -20.4 | 2013 | -48.4 | 0.00 | 0.28 | 0.17 | 0.55 | 0.0080 | 0.11 |
| 10 | -20.1 | 2130 | -48.0 | 0.00 | 0.27 | 0.14 | 0.59 | 0.0086 | 0.06 |
| 12 | -20.0 | 2204 | -48.2 | 0.01 | 0.26 | 0.12 | 0.62 | 0.0129 | 0.02 |
| 14 | -20.1 | 2172 | -47.5 | 0.01 | 0.27 | 0.11 | 0.61 | 0.0128 | 0.10 |
| 16 | -20.5 | 2076 | -48.0 | 0.00 | 0.29 | 0.12 | 0.58 | 0.0094 | 0.09 |
| 18 | -20.7 | 2006 | -49.1 | 0.00 | 0.32 | 0.12 | 0.56 | 0.0079 | 0.03 |

## Timing, offline (both sides, 60 s with interventions)

| blocks | p50 ms | p95 ms | p99 ms | max ms | budget ms | over budget | us/sample/side |
|---|---|---|---|---|---|---|---|
| 7517 | 0.417 | 0.565 | 0.747 | 2.683 | 7.982 | 0 | 0.62 |

## Live device run (bench LiveEngine, 60 s with interventions)

| underruns | block errors | blocks | p50 ms | p95 ms | p99 ms | max ms | over budget | clip blocks |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 7539 | 0.944 | 1.687 | 2.070 | 2.955 | 0 | {'A': 0, 'B': 0} |

## Node port vs the original gutterOsc.class (JVM)

all pass: **True**, source commit efa4737af31febf09bd746a45bd9cd57c88f74b1, hashes match True, 13 cases: filters_on_atan=ok, ramps_at_load=ok, noisy_damping=ok, audio_input=ok, highpass_active=ok, filters_off=ok, filter_count_12=ok, live_messages=ok, dist_mode_0=ok, dist_mode_1=ok, dist_mode_3=ok, dist_mode_4=ok, dist_mode_5=ok
