# Objects — decay control D4: measurements

Generated 2026-09-18T22:16:27 on commit `4f7e072`. REQ `memory/req-objects-decay-control-2026-09-18.md`, preflight `memory/research/objects-decay-control-preflight-2026-09-18.json` (on `255af20`). Catalog `lab_catalog/objects_decay_control_2026_09_18`.

## 1. The mean applied gamma of the Modal side

Method: per-sample recurrence g <- k g + (1 - k) gt from the block-start states (numpy, float64); J(s) = modes j < ndrive of the slots with role ACTIVE; every mode-sample weighted equally. k = exp(-1 / (44100 * 0.020)) = 0.998866855645396 computed here, equal to the engine's: True. Window [132300, 793800) samples = [3, 18) s, the partial blocks at both ends counted sample by sample.

| | this report | preflight |
|---|---|---|
| mode-samples | 5293928 | 5293928 |
| mean applied gamma, 1/s | 10.286612669487 | 10.286612669487 |
| T_ref = ln 1000 / gamma, s | 0.6715286656 | 0.6715286656 |
| mean target gamma (diagnostic, not the control), 1/s | 9.818623460940 | 9.818623460937 |
| mode-samples below / above the control gamma | 3650429 / 1643499 | 3650429 / 1643499 |
| min / max applied gamma, 1/s | 4.969608 / 86.346941 | 4.969608 / 86.346941 |

Relative difference of the mean from the preflight: -5.38e-14; block-end applied gamma of the recurrence against the engine: max |diff| 0.0e+00 over 2256 blocks (up to 4 active banks, 10 driven modes); active banks not on their own gamma: 0.

**Control:** Decay 0.671529 s -> gamma ln 1000 / 0.671529 = 10.286607546334 1/s, **-0.000050 %** from the mean (limit 0.01 %). The engine's applied Fixed gamma (-SR ln r, r = 10^(-3 / (SR T))): 10.286607546335…10.286607546335 1/s, max +0.000050 % from the mean.

## 2. Identity

Raw int16 interleaved PCM, whole take (794112 samples):

| condition | rendered now | preflight | equal |
|---|---|---|---|
| matched_fixed | `e4e5f4de7669c55c…` | `e4e5f4de7669c55c…` | True |
| modal | `9090f1d00d25d7d8…` | `9090f1d00d25d7d8…` | True |
| long_fixed | `24d5d1d0ef8fb9e1…` | `24d5d1d0ef8fb9e1…` | True |

| record | id | samples A/B/monitor | SHA-256 A | SHA-256 B | WAV == stored hash | journal | steps == D3 | side gain | pin |
|---|---|---|---|---|---|---|---|---|---|
| od_d4_control | `20260918-221025-d81b66` | 794112/794112/794112 | `e4e5f4de7669…` | `9090f1d00d25…` | True | start, step (107 steps) | True | A 1.220467 / B 1.21 | pinned `4f7e072` |
| od_d4_anchor | `20260918-221023-9e3dba` | 794112/794112/794112 | `e4e5f4de7669…` | `24d5d1d0ef8f…` | True | start, step (107 steps) | True | A 1.220467 / B 1.0 | pinned `4f7e072` |

- A of D4.1 == A of D4.2 (bytes of the WAVs and hashes): **True**; == the preflight's matched Fixed: True
- B of D4.1 == B of the D3 record == the preflight's Modal: **True**
- B of D4.2 == A of the D3 record == the preflight's long Fixed: **True**
- rendered side by side now: A(D4.1) == A(D4.2) True, B(D4.1) == B(D3) True, B(D4.2) == A(D3) True; the D3 scene renders the D3 record's WAVs True; monitor == A (side A listened) True

Per block (2256 blocks, D4.1, D4.2 and D3 in lockstep): field and generation equal True; figures (ids, slots, cells, centres) equal True; every active bank of the three conditions shares slot_id, smode, ndrive, npulse, nlive, sqrtlam, ffreq, cth, sth, wcur, winc, wtgt, wleft, zf, zs, zu, zfm, zsm, zum, pan, pleft, last_e, last_a, last_b: True; Attack state equal True; the two matched Fixed sides equal in every engine array: True; differing fields: none. Losses: matched / long Fixed on exactly r(0.671529) / r(1.39), no own-gamma slot: True; blocks where the losses of the active banks differ from the matched Fixed — D4.1 (Modal applied gamma != the matched gamma) 2256, D4.2 (r) 2256. Up to 4 active banks, 21 tails in Modal. What differs by design: the resonator states, the losses and the constant side gain.

## 3. Levels

RMS of all samples of both channels over [3, 18) s of the delivered records; one constant side gain per condition.

| condition | side gain | RMS dBFS | preflight | diff dB | int16 peak | pre-clip peak | clip blocks | full-scale samples |
|---|---|---|---|---|---|---|---|---|
| matched_fixed | 1.220467 | -33.9258 | -33.9258 | +0.0e+00 | 0.1899 | 0.1899 | 0 | 0 |
| modal | 1.21 | -33.9061 | -33.9061 | +0.0e+00 | 0.1171 | 0.1171 | 0 | 0 |
| long_fixed | 1.0 | -33.9268 | -33.9268 | +0.0e+00 | 0.1626 | 0.1626 | 0 | 0 |

Spread of the three RMS: **0.0207 dB** (limit 0.1). All values finite: True.

Unit-gain calibration (`--calibrate`): matched_fixed -35.6573 dBFS -> gain to the long Fixed 1.220467 (delivered 1.220467); modal -35.5628 dBFS -> gain to the long Fixed 1.207262 (delivered 1.21); long_fixed -33.9268 dBFS -> gain to the long Fixed 1.000000 (delivered 1.0).

## 4. Records

| record | id | replay A / B / monitor | pin | Notes | Continue (125 blocks) |
|---|---|---|---|---|---|
| od_d4_control | `20260918-221025-d81b66` | match {'A': True, 'B': True, 'monitor': True} | pinned `4f7e072` | True | clock True, {'A': True, 'B': True, 'monitor': True} |
| od_d4_anchor | `20260918-221023-9e3dba` | match {'A': True, 'B': True, 'monitor': True} | pinned `4f7e072` | True | clock True, {'A': True, 'B': True, 'monitor': True} |

The D3 source record `20260918-204155-4ff923` still replays: match {'A': True, 'B': True, 'monitor': True}.

Block time, each scene rendered alone (both sides), after 1 s:

| scene | p50 ms | p99 ms | max ms | budget |
|---|---|---|---|---|
| od_d4_control | 0.76 | 4.57 | 10.88 | 7.98 |
| od_d4_anchor | 0.61 | 4.75 | 9.78 | 7.98 |

The engine and its code are those of D3 (no sound code changed): single slow blocks are desktop noise -- the D3 scene itself, measured twice in a row the same way on 2026-09-18, gave max 4.98 ms and 11.72 ms (p99 4.01 / 5.67 ms).

**Verdict:** all checks pass

