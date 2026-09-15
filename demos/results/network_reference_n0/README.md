# N0 — Gutter Synthesis reference reproduction (no CA yet)

Package built by **one command** from the repository root:

```
python -m demos.network_reference_n0.render_n0
```

(~5 min: 16 probe renders in parallel + the three examples; `--quick` skips the probes.)
Node-vs-original verification (needs a JDK and a clone of the source; not part of the render):

```
python -m demos.network_reference_n0.verify_node --source <clone of tommmmudd/guttersynthesis> --jdk <jdk>/bin
```

**Nobody has listened to these files.** Every statement below is a measurement or a reading of
the patch; the listening questions are in `memory/req-listen-network-reference-n0-2026-09-15.md`.

## What was reproduced

Source: https://github.com/tommmmudd/guttersynthesis, commit `efa4737af31febf09bd746a45bd9cd57c88f74b1`
(GPL-3.0, (c) Tom Mudd); file hashes in `manifest.json` → `source.file_sha256`.

* **Node** = the class the patch actually loads, `gutterOsc.class` (top level of the repo).
  It is **not** compiled from `gutterOsc.java` in the same repo (that source is an older
  two-bank / 8-inlet variant, compiled form `gutterOsc for Java1.6/gutterOsc.class`).  The
  executable class has 6 signal inlets (gamma, omega, c, dt, singleGain, audioInput), one bank
  of up to 24 band-pass biquads, a one-pole lowpass and a one-pole highpass, six distortion
  methods (default 2 = atan).  Decompiled with CFR 0.152 and ported statement-for-statement
  (`demos/network_reference_n0/gutter_node.py`).
* **Verification** (`verification_node.json`): the original class file executed on a JVM
  (through two 10-line stand-ins for the Max Java API) on the same input arrays as the port —
  14 cases: patch default, line~-style ramps, link-like noisy damping, audio-input forcing,
  highpass active, filters off, 12 filters, messages during DSP, all six distortion methods,
  block sizes 1/64/512.  Float32 outlets **bit-exact**, double state traces identical
  (3000 samples) and final states identical in 13/14 cases; distortion method 5 (sigmoid via
  `Math.exp`, a HotSpot intrinsic; not used by the patch) differs by ≤ 1 ulp.  Getting there
  required a port of fdlibm `atan` (= `java.lang.Math.atan`): the C runtime's `atan` differs
  in the last ulp for ~0.6 % of inputs, which is enough to desynchronise this chaotic node.
* **Network** (`Gutter Synth.maxpat` + `mxjGutterCompact.maxpat`, reconstructed from the JSON
  graphs, `demos/network_reference_n0/gutter_network.py`):

  ```
  node n:  gamma_n = mapMod(mod_n) [line~ 30 ms]      -> inlet 0
           omega   = 0.002          [line~ 50 ms]      -> inlet 1
           c_n     = clip~(mapDamp(damp_n) [line~ 15 ms] + link_n, 0.0001, 1) -> inlet 2   <== the links
           dt_n    = mapRate(rate_n) [line~ 30 ms]     -> inlet 3
           gain_n  = mapGain(gain_n) [line~ 30 ms]     -> inlet 4
           out0 -> clip~ -5 5 -> svf~ 20 (HP) -> svf~ 30 (HP) -> tanh~ -> equal-power pan -> L_n, R_n
  matrix:  in_i = 0.5 (L_i + R_i);  mo_j = sum_i G[i,j] in_i   (matrix~ 8 8, preset 1: all cells 1, diagonal 0)
           link_j(k) = interaction * mo_j(k - D),  D = 2000 (delay~) + 64 (one signal vector in the send~/receive~ loop)
  master:  (sum L_n, sum R_n) * MASTER_GAIN          (patch: *1.0 *0.4 -> dac~)
  ```

  The links enter each node's **damping** term, not its audio input.  The forcing
  `gamma*sin(omega*t)` with omega = 0.002 and dt = 0.0027/sample has a period of ≈ 26 s: it is
  a slow push, the tone itself comes from the resonator-bank ↔ Duffing feedback (from the
  zero state with mod = 0 the model stays silent; once running it keeps going with mod = 0).
* **Initial conditions** = the patch state after load: `autopattr` restore values saved in the
  patches (per node: gain 162, damp 138, mod 46, rate 39, Q 149, soften 103, 24 filters;
  interaction 127), `loadmess` values (setHighpass 5 → bypassed, omega 0.002, pitch_shift 1.0),
  matrix preset 1.  The patch **randomises** at every load: filter frequencies (formula of
  `p randomise_filters_mixed_root`, reproduced with seed 20260915), the link delay
  (2000 + random(2000) samples; fixed at 2000 here) and, if the main patch's restore fires,
  per-node mod/rate of the "SLOW movement" preset (not applied).  Full table with sources:
  `manifest.json` → `parameters`; the exact values used: `config.json`.

## Files

| file | content |
|---|---|
| `01_palette.wav` | 3 × 8 s, 0.5 s pauses, each from the zero state: **A** base config; **B** damp slider 40 (c = 0.024); **C** interaction slider 230 (link gain 4.04). |
| `02_control.wav` | 12 s, no restart: 0–4 s base; at 4 s **damp 138 → 40** (patch ramp 15 ms); at 8 s back to 138. |
| `02_control_reference.wav` | the same run without the intervention. |
| `03_links.wav` | 4 s warm-up with links OFF (matrix all zero) → state saved (`raw/links_saved_state.npz`) → 8 s continued with links OFF ‖ 0.5 s ‖ 8 s continued from the **same state** with links ON (preset 1, interaction 127 as in the patch; switched instantly, the patch would ramp cells over 2 s). |
| `03b_links_strong.wav` | extra, clearly labelled: the same OFF/ON pair with the interaction slider at 230 (inside the source range 0..256). |
| `raw/*.f32.wav` | float32 output **before** PCM conversion (16-bit = raw × 0.85, one fixed coefficient for all files; peaks and clipped-sample counts per file in `manifest.json`). |
| `raw/probes/` | 16 float32 renders of the probe sweep + `probes.json` / `probes.md`. |
| `figures/` | spectrogram + waveform of every example and the probe grid. |
| `manifest.json` | source, environment, config, schedules, measurements, sha256 of every file. |
| `config.json` | every parameter of the model actually used (also the four assumptions below). |
| `verification_node.json` | node-vs-original report. |

`raw/` is git-ignored (regenerated by the command above; hashes are in the manifest).

## Probe sweep (what the controls do, measured)

Schedule for every probe: identical init, 0–4 s base config, at 4 s one slider is moved (with
the patch's ramp), held to 12 s; control = the same run untouched.  Window 6–12 s; the
control's level settles within 3 dB after 0.25 s while its spectral centroid keeps drifting
slowly (per-second values in `probes.md`), so "steady" here means a slowly evolving texture,
not a fixed spectrum.  Full table: `raw/probes/probes.md`.

| axis (source slider, raw 0..256) | values | what changes vs control (window 6–12 s) |
|---|---|---|
| **mod** (forcing amplitude gamma; base 46 → 0.32) | 0, 23 | almost nothing: level +0.2 dB, log-spectral distance ≈ 4 dB — the running system does not need the forcing |
| | 92 (gamma 1.29) | level −5 dB, centroid ×0.32, more spectral flux, envelope std +3.5 dB |
| | 138, 184 (gamma 2.9, 5.2) | the system collapses: −52 / −58 dB — effectively silence |
| **damp** (c base; base 138 → 0.29) | 40, 90 (c 0.024, 0.12) | centroid ×0.07 (≈ 60 Hz): resonator lines give way to a sub-100 Hz growl; +2…+4 dB |
| | 190, 230, 256 (c 0.55…1.0) | same level and centroid, cleaner line spectrum (log-spectral distance 3–6 dB) |
| **interaction** (link gain; base 127 → 1.23) | 0, 64 | same level, same centroid, different spectral content (log-spectral distance 9 / 5 dB) |
| | 190, 230, 256 (2.8…5.0) | centroid ×0.12…0.34, +2…+3 dB, slow amplitude modulation appears (flux ×1.3, envelope std +1…1.5 dB) |

PCM difference probe − control is > 0 dB for **every** probe (trajectories diverge, as the
task brief anticipated) and is reported but not interpreted.  No probe produced NaN/Inf or a
protective reset of the source (`resets = 0` everywhere; the counters are in the manifest).

## Examples: what the measurements say

Filled from `manifest.json` (mono = (L+R)/2 of the raw sum, before master gain):

| example | segment | rms dB | centroid Hz | flatness dB | env. std 250 ms dB | reading |
|---|---|---|---|---|---|---|
| 01 A base | 0–8 s | −14.9 | 960 | −48 | 0.55 | line spectrum of the resonator banks over a low rumble; slow drift of the centroid (1286 → 830 → 950 Hz over 12 s in the control run) |
| 01 B damp 40 | 8.5–16.5 s | −13.0 | 64 | −61 | 2.64 | energy below 100 Hz, broadband above it, amplitude moves in ~0.3–1 s waves (waveform in `figures/01_palette.png`) |
| 01 C interaction 230 | 17–25 s | −15.2 | 714 | −44 | 1.14 | starts like A, the links pull the spectrum down and add slow amplitude beating within the 8 s |
| 02 control | 0–4 s base | −14.9 | 1073 | −47 | 0.77 | = reference run (identical samples) |
| | 4–8 s damp 40 | −13.3 | 88 | −61 | 1.33 | reference in the same window: −14.9 dB, 848 Hz — the change is spectral (×0.08 centroid) and temporal (env. std +0.6 dB), level +1.6 dB |
| | 8–12 s back to 138 | −13.3 | 65 | −66 | 1.32 | **does not return**: the growl persists after the slider is back (reference: 898 Hz). Hysteresis of the nonlinear system, not a bug: the same parameters from the zero state give texture A |
| 03 links OFF | 0–8 s | −14.9 | 1227 | −63 | 0.09 | static line spectrum, no envelope movement |
| 03 links ON (interaction 127, as in the patch) | 8.5–16.5 s | −14.7 | 1141 | −49 | 0.09 | level and envelope unchanged; spectral content differs: log-spectral distance 10.6 dB, flatness −63 → −49 dB (more components between the lines). Whether this is audible is exactly the listening question. |
| 03b links ON (interaction 230) | 8.5–16.5 s | −13.3 | 350 | −48 | 1.79 | centroid ×0.29, flux ×2.3, envelope std +1.7 dB: after ~4 s the links push the network into the low, breathing mode (`figures/03b_links_strong.png`) |

No NaN/Inf, no protective reset of the source (`resetDuff`) in any example or probe; no
clipped sample in any 16-bit file; peak after the fixed master gain 0.719 (−2.9 dBFS,
`02_control`).

**Reproducibility.** Two complete runs (16 and 8 probe workers) gave byte-identical float
files (`raw/03_links.f32.wav`, `raw/02_control_reference.f32.wav` and every probe: same
sha256).  Rendering the same sequence in different chunk sizes gives identical samples
(`tests/test_network_reference_n0.py`), as does continuing from an exported state.  The
model has no internal block; the only "vector" quantity is the assumed 64-sample feedback
delay, stored explicitly.

## Known limitations / assumptions (also in `memory/questions.md`)

1. Max itself was not run.  The Max objects around the node (`line~`, `clip~`, `svf~`,
   `tanh~`, `cycle~` pan, `matrix~`, `delay~`) are transcriptions; only the node is proven
   bit-exact.
2. `send~`/`receive~` inside the feedback loop add one signal vector; assumed 64 samples
   (`extra_feedback_samples` in `config.json`; 0 / 64 / 128 possible without running Max).
3. `delay~` link time fixed at 2000 samples (the patch draws 2000..3999 at load).
4. `svf~` 20/30 Hz HP: Chamberlin SVF with q = 1 − res, res 0.01; the exact Max coefficient
   mapping is not published.  Affects only the sub-bass band (which does matter in the
   low-damping modes).
5. Load order of the UI restore (node sliders vs. main-patch master sliders gain 150 /
   damp 159 / rate 23 and the SLOW movement preset) is not determinable offline; the node
   values were used.
6. Speed: ~2.3 × slower than real time (pure numpy per-sample loop); fine for offline renders,
   not a live engine.
