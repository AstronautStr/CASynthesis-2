# 2026-06-15 — Laplacian prototype (gol_life_synth_laplacian.py)

## What was done
Built `gol_life_synth_laplacian.py` — second full CASynth prototype with
Laplacian mapping, as specified by `decisions.md` entry "Второй прототип".

## Key architecture decisions made during implementation

### Flat slot pool (SlotPool)
The original `gol_life_synth.py` uses K_MAX=20 shared harmonic slots where
slot k maps to harmonic k of the single carrier.  Laplacian breaks this: each
object has its own set of inharmonic mode frequencies, so slots can't be
shared across objects on a harmonic grid.

Solution: flat pool of TOTAL_SLOTS = MAX_VOICES × MAX_MODES_PER_OBJ = 192
slots.  Voice v, mode m → slot `1 + v*MAX_MODES_PER_OBJ + m`.  SlotPool
manages targets (freq_slots, amp_tgt, pan_tgt) and crossfade state.

### Crossfade implementation
Meets `decisions.md` / `questions.md` spec: amplitude-based, not frequency
glide.

- `_freq_prev[slot]` tracks previous slot frequency.
- On change: if `amp_cur[slot] >= 0.01` → set `amp_tgt[slot] = 0` (release).
  `freq_slots` keeps old freq so the render completes the decaying waveform
  coherently.  On the next `update()` call, once `amp_cur < 0.01`, new freq
  is applied and `amp_tgt` set to new amplitude (attack from 0).
- Phase reset occurs at the moment of freq application, under near-zero
  amplitude → inaudible.
- Mode frequencies are never glided; they switch discretely when the slot
  becomes silent.

### Object-specific patch extraction
`analyse()` builds the sub-grid from only the object's own cells
(`sub[ys - r0, xs - c0] = 1`) rather than slicing the raw grid, which would
include neighboring objects that share the bounding box.

### FADE_CHUNKS
`max(1, int(0.05 / CHUNK_S))` = 1 chunk at CHUNK_S=0.09s ≈ 90ms.  Slightly
over the 50ms target but within perceptual range; adjustable via CHUNK_S.

## Open question logged
None new.  The crossfade "one-frame silence gap" between release and attack
(~16ms at 60 FPS) is a known timing artefact of the per-frame update model;
not logged as it's within perceptual tolerance and matches decisions.md
intent.

## Files changed
- `gol_life_synth_laplacian.py` — created
- `memory/current.md` — updated (In-flight -> done, Implemented list extended)
- `memory/log/2026-06-15-laplacian-prototype.md` — this file
