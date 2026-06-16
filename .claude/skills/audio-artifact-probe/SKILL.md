---
name: audio-artifact-probe
description: >
  Catch and validate fixes for audio artifacts (clicks/pops, distortion/clipping)
  in the CASynth synth engines (gol_life_synth*.py). Use when the user reports
  clicks, crackle, buzzing or distortion, or asks to verify that an audio fix
  removed an artifact WITHOUT changing the timbre/spectrum. Renders the engine
  offline to a WAV on a reproducible scene and measures artifacts objectively.
---

# Audio artifact probe (offline render + objective metrics)

Audio bugs in this project are not reliably caught by ear or by "it runs". Render
the synth **offline, deterministically** to a WAV and **measure** the artifact,
then **measure again after the fix**. Never declare a fix done without a
before/after number.

## PRIMARY pipeline: reproduce + verify on the USER'S session snapshot
Artifacts here are usually **state-dependent** (a specific field × knob
combination — e.g. a click that only appears with long *release* on a particular
shape). A synthetic scene may not hit it. So the canonical pipeline is:

1. **Record the user's session** with full inputs (`CASYNTH_RECORD=1`, below).
   The user reproduces the artifact and quits; this saves the exact field history
   AND every live knob (spread / alpha / release / volume) per rendered frame.
2. **Replay that snapshot offline** with the current (unfixed) engine:
   `python _render_probe.py session <ts>`. This re-renders the user's exact
   inputs and prints a **fidelity check** vs the recorded WAV — they should be
   IDENTICAL, proving the replay reproduces the live artifact (not a guess).
3. **Measure** the artifact on the snapshot (clipping / clicks, below).
4. **Fix**, then **replay the SAME snapshot again** and re-measure: artifact gone,
   spectrum preserved (`compare`).

Do NOT diagnose a state-dependent artifact on a synthetic scene when a user
snapshot is available — reproduce on the snapshot. Synthetic scenes are a
**fallback only** (see "Fallback scenes" below), for when no snapshot exists or
the snapshot doesn't trigger it.

Ready-to-run tools live in the repo root:
- `_render_probe.py` — `session <ts>` replays a recorded snapshot (PRIMARY);
  `<label>` renders the fallback pentadecathlon scene; `compare A B` compares two
  saved spectra. Prints distortion + click metrics and saves a normalised spectrum.
- `_click_test.py` — isolate the envelope-corner click from carrier curvature
  (a controlled micro-test + a linear-vs-smooth scene diff).

Adapt these per investigation; keep them — they are the regression harness.

## CRITICAL: offline-clean ≠ realtime-clean
An offline render calls `pool.update` once per chunk and renders chunks
back-to-back — it cannot reproduce artifacts born in the **live audio path**:
- `feed_audio()` runs every frame (~60 fps) but only renders a chunk when the
  mixer queue is empty; `pool.update` thus runs ~5-6× per chunk, mutating
  `amp_cur`/banks between renders.
- `analyse()` runs `eigvalsh` on every object every frame; a frame hitch starves
  the 2-deep mixer queue → playback gap → click that is **not in the rendered
  samples at all** (the engine output is gapless; the gap is in playback timing).

So if your offline WAV is clean but the user still hears clicks live, **do not
conclude it's fixed** — record a real session (below) and analyse THAT.

## Live session recording (env `CASYNTH_RECORD=1`)
`gol_life_synth_laplacian.py` records when run with the env var set:
```
# PowerShell:  $env:CASYNTH_RECORD=1; python gol_life_synth_laplacian.py
# bash:        CASYNTH_RECORD=1 python gol_life_synth_laplacian.py
```
Play until it clicks, then quit (Esc / close). On quit it writes:
- `_session_<ts>.wav` — the exact gapless audio the engine produced.
- `_session_<ts>.npz` — diagnosis + replay data:
  - `frames` `[dt_ms, gen, n_voices, n_rendered, underrun]` per frame (timing /
    underrun diagnosis).
  - `replay` `[n_rendered, note, spread, alpha, release_ms, vol]` and
    `replay_grids` `[F, GRID_H, GRID_W]` — per RENDERED frame, the FULL engine
    input (field + carrier note + every live knob + chunk count). This is what
    makes `_render_probe.py session <ts>` reproduce the output deterministically.
  - `grids`/`gens`/`notes` (legacy field history), plus meta
    (`sr/chunk_s/step_hz/master_gain/max_modes/patch_size/...`).

Reproduce + diagnose:
- `python _render_probe.py session <ts>` re-renders the snapshot and prints a
  fidelity check vs the recorded WAV (max|diff|). IDENTICAL ⇒ the replay faithfully
  reproduces the live engine output; now measure the artifact on it.
- **WAV has clicks** ⇒ render-content artifact (envelope/crossfade/`amp_cur`
  decoupling). The replay reproduces it; fix in the engine and re-replay.
- **WAV clean but `underruns>0` / `frames` with `dt_ms > CHUNK_S*1000`** ⇒
  playback-starvation clicks (the click is in playback timing, NOT the samples).
  Fix = decouple audio from the frame loop / lighten per-frame cost (don't
  `eigvalsh` every frame; cache voices between GOL steps; render ahead), not the
  synthesis math. NOTE the meter reports the PRE-clip peak: a click while the
  meter shows headroom (e.g. −15 dB, no clip) is NOT clipping — it's a
  render-content envelope/crossfade artifact, so go the WAV-has-clicks route.

## How the offline render works
Drive the real engine functions, no pygame window/audio:
```python
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import gol_life_synth_laplacian as L
# per chunk: advance GOL by CHUNK_S, analyse(grid,f0[,spread,alpha]) -> voices,
#            pool.update(voices, amp_cur, release_chunks),
#            render_chunk_laplacian(phase, amp_cur, pan_cur,
#                pool.amp_tgt, pool.pan_tgt, pool.freq_slots, channels=2, gain)
#            -> returns (buf, pre_clip_peak, n_clipped)
```
The sim is deterministic, so two renders of the same scene are identical — this
is what lets you cancel the carrier and isolate envelope artifacts (below).

## Fallback scenes (only when no user snapshot reproduces it)
Use these ONLY if there is no recorded session, or the session doesn't trigger
the artifact and you need to provoke it. The primary repro is always the user's
snapshot (above). Escalate breadth until it reproduces:

1. **3 pentadecathlons**, evenly spaced on the 52-wide torus so they never touch
   (gaps ~15 cells), vertically centred:
   ```python
   PENTA = [(0,1),(1,1),(2,0),(2,2),(3,1),(4,1),(5,1),(6,1),(7,0),(7,2),(8,1),(9,1)]
   COLS  = [8, 25, 42];  ROW0 = 10
   ```
   Each p15 internally fragments during its period (3 → up to 24 components) so it
   exercises MAX_VOICES, constant topology changes (cross-fades every GOL step),
   and heavy partial summation.
2. **Dense random fill** (`RANDOM_DENSITY`, or higher) — peaks higher (clipping /
   headroom worst case) and produces many simultaneous voices + rapid topology
   churn.
3. **A pile of mixed oscillators** (from `patterns.py`: blinkers, toads, beacons,
   pulsars, pentadecathlons…) packed in — more variety of shapes / mode counts /
   topology-change cadences than a single repeated oscillator.

If reproducing yourself, **also set the relevant knobs** (spread / alpha /
release / volume) to the regime the user reported — a release-tail or
brightness-dependent artifact won't show at default knobs. A few pentadecathlons
at default settings may simply not be enough; widen the scene and push the knob.

## Metrics

### Distortion = clipping (hard clip adds harmonics + crackle)
- **saturated samples**: count `|int16| >= 32767`; report % and longest run.
- **true pre-clip peak**: render at a tiny `MASTER_GAIN` (e.g. 0.01) and scale
  back: `peak = max|int16|/32767 * (real_gain/tiny_gain)`. `> 1.0` ⇒ it clips.
- Clean target: **0 saturated samples**, true peak `< ~0.9` (≈10% headroom).

### Clicks = slope discontinuities / sharp transients
- **sharp-transient count**: `|d2| > 100` where `d2 = |np.diff(mono, 2)|`.
  Clipping produces thousands (flat-top corners); a clean render has ~0.
- **boundary vs interior**: chunk boundaries at multiples of `n=int(CHUNK_S*SR)`;
  compare `|d2|` there vs the interior median. An envelope-corner click raises
  boundaries above interior.
- **isolate the envelope corner** (decisive, carrier-independent): build the
  envelope alone (no carrier) for a step-then-steady two-chunk case and take
  `|d2|` at the junction. Linear `linspace` ramp leaves a corner; a raised-cosine
  (smoothstep) ramp `(1-cos(pi*t))/2` kills it (~800x lower). See `_click_test.py`.

### Spectrum preservation (timbre must not change, only artifacts removed)
Compare **normalised** magnitude spectra (a gain change alters level, not shape):
- **cosine similarity** of the two spectra — want `> 0.999`.
- **per-partial relative diff** on bins above ~2% of peak.
- **broadband floor RMS** on the non-partial bins — the artifact proxy; should
  drop (cleaner) after the fix.

## Gotchas / order of operations (learned the hard way)
1. **Fix distortion first.** Clipping corners dominate `|d2|` everywhere and
   mask the click metrics. Remove clipping, then click numbers become readable.
2. **The envelope corner sits at the junction sample `n-1`, not `n`.** Chunk i
   ends at its target amplitude; chunk i+1 may be flat — the slope break is at
   the seam. Scan a small window around the boundary, don't probe one sample.
3. **Scene-level `|d2|` understates a real envelope fix** because carrier
   curvature dominates that sample. Prove envelope fixes at the *envelope* level
   (carrier removed), then show the modest scene-level improvement.
4. **Compare spectra normalised**, and remember a clipped baseline is the
   *dirty* reference — a large diff vs it is the distortion you removed, not a
   regression.

## Known good fixes (this engine)
- **Distortion**: lower `MASTER_GAIN` for fixed headroom (sized to the dense-field
  raw peak). Not a per-chunk limiter — that pumps and modulates the spectrum,
  violating the constant-amplitude design.
- **Clicks**: smoothstep (raised-cosine) per-chunk amplitude/pan ramp instead of
  linear `linspace`; zero slope at both ends ⇒ C1-continuous envelope across
  chunk boundaries. Steady slots (amp_cur==amp_tgt) are bit-unchanged.

## Output the user expects
A before/after table of the numbers (saturated %, peak, sharp transients, cosine
similarity) and the rendered WAVs for A/B listening (`SendUserFile`).
