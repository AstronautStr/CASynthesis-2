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

Ready-to-run tools live in the repo root:
- `_render_probe.py` — render the canonical scene, print distortion + click
  metrics, save a normalised spectrum (`python _render_probe.py <label>`), and
  compare two saved spectra (`python _render_probe.py compare A B`).
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
- `_session_<ts>.npz` — `frames` `[dt_ms, gen, n_voices, n_rendered, underrun]`
  per frame, plus `grids`/`gens`/`notes` (field history for replay), and meta.

Diagnose:
- **WAV has clicks** ⇒ render-content artifact (envelope/crossfade/`amp_cur`
  decoupling). Replay `grids`/`notes` to reproduce; fix in the engine.
- **WAV clean but `underruns>0` / `frames` with `dt_ms > CHUNK_S*1000`** ⇒
  playback-starvation clicks. Fix = decouple audio from the frame loop / lighten
  per-frame cost (don't `eigvalsh` every frame; cache voices between GOL steps;
  render audio ahead on its own cadence), not the synthesis math.

## How the offline render works
Drive the real engine functions, no pygame window/audio:
```python
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import gol_life_synth_laplacian as L
# per chunk: advance GOL by CHUNK_S, analyse(grid,f0) -> voices,
#            pool.update(voices, amp_cur),
#            render_chunk_laplacian(phase, amp_cur, pan_cur,
#                pool.amp_tgt, pool.pan_tgt, pool.freq_slots, channels=2)
```
The sim is deterministic, so two renders of the same scene are identical — this
is what lets you cancel the carrier and isolate envelope artifacts (below).

## Canonical repro scene
**3 pentadecathlons**, evenly spaced on the 52-wide torus so they never touch
(gaps ~15 cells), vertically centred:
```python
PENTA = [(0,1),(1,1),(2,0),(2,2),(3,1),(4,1),(5,1),(6,1),(7,0),(7,2),(8,1),(9,1)]
COLS  = [8, 25, 42];  ROW0 = 10
```
Why this scene: each pentadecathlon p15 internally fragments during its period
(3 → up to 24 connected components) so it exercises MAX_VOICES, constant
topology changes (cross-fades every GOL step), and heavy partial summation — the
exact conditions that trigger both clicks and clipping. For a clipping-headroom
worst case also test dense random fields (`RANDOM_DENSITY`), which peak higher.

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
