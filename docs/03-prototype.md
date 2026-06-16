# 03 — Current Prototype

## Files
- **`gol_life_synth.py`** — the interactive desktop app (main artifact).
- **`gol_life_synth_laplacian.py`** — second interactive prototype: voice spectrum =
  Laplacian resonant modes of each object's shape (inharmonic, metallic). Uses a
  sounddevice callback stream + overlapping mode cross-fade. Has **live timbre knobs**
  (on-screen sliders, no restart): *spread* (low modes → whole spectrum), *alpha*
  (1/i^α brightness), *release* (mode-tail length: percussive ↔ pad). The lowest mode
  is always pinned to the carrier f0. See `memory/current.md` for the full state.
- **`casynth_core.py`** — shared shape→spectrum mapping library (`map_*`, `extract`),
  single source of truth for bench + prototypes; numpy/scipy only, unit-tested
  (`tests/test_casynth_core.py`).
- **`mapping_bench.py`** — discover bench (oscillator × mapping matrix).
- `gol_synth.py` — an earlier offline renderer that writes a WAV of a fixed scene
  (block + blinker + toad + glider). Useful as a minimal reference and for
  non-interactive rendering.
- `gol_synth_demo.wav` — sample output of the offline renderer. Note: that earlier
  version used the *sequencer* mapping (separate notes) and per-generation pulses;
  the interactive app has since moved to the additive + continuous design.

## Stack & run
```
pip install pygame-ce numpy scipy
python gol_life_synth.py
```
- **pygame-ce**, not mainline pygame: mainline has no Python 3.14 wheel and fails to
  build from source (`distutils.msvccompiler` was removed in Python 3.12+).
  pygame-ce ships a 3.14 wheel and is API-compatible (`import pygame`).
- The mixer is opened with `allowedchanges=0` to **force stereo**; some devices
  (e.g. 7.1 surround, which report 8 channels) otherwise break stereo buffers
  (`Array depth must match number of mixer channels`). The render buffer also adapts
  to the actual channel count as a fallback.
- Headless checks: run with `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy` to
  validate logic / draw / synth without a display or audio device.

## Pipeline
Game of Life (toroidal, `scipy.ndimage.convolve` with `mode='wrap'`) →
connected-component segmentation (`scipy.ndimage.label`, 8-connectivity) →
per-object features → harmonics of the carrier note → continuous additive synthesis.

## Mapping currently implemented (the live "feature → parameter" table)
- **size (area) → harmonic number k** of the carrier (big object = low harmonic near
  the fundamental; small = high). `K_MAX = 20`.
- **density (bbox fill) → that harmonic's amplitude** (with a 1/k^0.7 rolloff to tame
  upper partials).
- **centroid x → that harmonic's stereo pan.**
- **carrier note** = whichever piano key is latched (range C3–C5).
- Voices capped at `MAX_VOICES = 24` (largest objects win).

> This is ~3 of the perceptual axes from `docs/02-design-space.md`. Others
> (jaggedness, symmetry, order/chaos, activity, elongation) are *designed but not yet
> wired*. See `docs/02-design-space.md` §G for the plan. All tunable constants live
> at the top of `gol_life_synth.py`.

## Audio engine
Continuous, phase-coherent additive synthesis:
- One phase accumulator per harmonic, carried unbroken across audio chunks → no clicks.
- Per-harmonic amplitudes **glide** linearly toward their targets each chunk →
  constant overall loudness and a smooth spectral morph (no per-generation pulsing).
- Targets are aggregated per harmonic from the current objects every frame; chunks
  (~0.09 s) are streamed **gaplessly** through `pygame.mixer.Channel.queue`.
- Sound is *always* continuous and reflects the current field: Play/Pause gates the
  *automaton's evolution*, not the audio. A paused field = a steady held timbre; an
  empty field = silence (no harmonics).

## Controls
- **Mouse on field:** left-drag draw cells, right-drag erase (any time, even while running).
- **Piano (bottom):** click a key to set the carrier note (latched, highlighted).
- **Volume slider:** top-right of the toolbar.
- **Play/Pause** [Space], **Step** [S], **Random** [R], **Clear** [C],
  speed **−/+** [Down/Up], quit [Esc].

## Visualization
- Live cells are coloured by **their harmonic**: hue low → high (red → blue),
  brightness = amplitude — so the picture shows the sonification and you can *see*
  which voice is which.
- A legend shows the size → harmonic colour ramp.
- Grey cells = objects beyond the voice cap (not sounded).

## Known limitations / sharp edges
- Only ~3 perceptual axes are audible (size → harmonic, density → amplitude,
  centroid → pan); the timbre's expressive range is correspondingly narrow until
  more axes are wired.
- **Segmentation is not wrap-aware:** an object straddling the toroidal seam
  momentarily splits into two voices (and its features glitch) for the few
  generations it is crossing.
- **No object identity tracking across generations** → no true attacks / decays on
  birth / death, and collisions are not yet events.
- Per-object character is limited to a single additive partial (amplitude + pan);
  richer per-object timbre (detune, noise, sub-structure) is future work.
- Loudness scales somewhat with pattern density (more harmonics = fuller / louder);
  this is intended, not compression, but extreme fields rely on the limiter / volume.
