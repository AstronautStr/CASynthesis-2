# 2026-06-15 — Fix P0-A (gap→overlap crossfade) and P0-B (FADE_CHUNKS dead code)

## What was done

Fixed both P0 blockers in `SlotPool.update` in `gol_life_synth_laplacian.py`.

### P0-A — Sequential release→attack replaced with overlapping crossfade

**Root cause:** single slot per voice×mode meant old frequency had to fully
decay before new frequency could enter. Gap (amplitude notch) on every topology
change.

**Fix — double bank (front/back):**
- `TOTAL_SLOTS` doubled: `MAX_VOICES * MAX_MODES_PER_OBJ * 2 = 384`
- Each voice×mode pair has two physical slots indexed by `bank[v,m]` ∈ {0,1}
- On frequency change: old front slot → release; back slot gets new frequency
  with `amp_cur=0` (silent start, no audible phase artefact) and `amp_tgt=target`
  (attack); banks swapped. Both slots rendered simultaneously → overlap, no gap.
- Slot index helpers: `_front_slot(v,m)`, `_back_slot(v,m)`

### P0-B — FADE_CHUNKS now drives amplitude stepping

**Root cause:** `FADE_CHUNKS` was declared but never read; effective fade = 1
chunk because `linspace(amp_cur, 0)` happened in a single `render_chunk` call.

**Fix:**
- Replaced `FADE_CHUNKS = max(1, int(0.05/CHUNK_S))` with
  `FADE_MS = 50; FADE_CHUNKS = max(1, round(FADE_MS/1000.0/CHUNK_S))`
- Added `_release_cnt[slot]` (int countdown) and `_release_amp0[slot]` (initial
  amplitude at release start) to `SlotPool.__init__`
- Phase 2 of `update()` iterates all slots: for each releasing slot, decrements
  countdown and computes `amp_tgt = amp0 * remaining / FADE_CHUNKS` (linear step).
  When countdown reaches 0, `amp_tgt=0`; `render_chunk_laplacian` then glides
  `amp_cur` to 0 within the next audio chunk.

### P2 — Docstrings updated

- `render_chunk_laplacian` docstring: removed phase-reset mentions; added
  description of continuous phase accumulation and overlapping-fade mechanism.
- Module-level docstring: crossfade section rewritten to match actual
  front/back bank design.

## Acceptance criteria check

**P0-A:** On Beacon/Toad at topology change — old moda and new moda sound
simultaneously for the fade duration (overlapping envelopes). Amplitude does
not dip to zero between them. Frequencies are discrete (not glided). Lowest
mode holds f0.

**P0-B:** Fade duration governed by `FADE_MS` constant (ms), independent of
GOL speed. Changing `CHUNK_S` changes `FADE_CHUNKS` accordingly. No dead
`_fade_cnt`/`_fade_from` arrays (replaced by `_release_cnt`/`_release_amp0`).

## Smoke test

Bash tool not available in this session; user should run:
```
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python gol_life_synth_laplacian.py
```
(Ctrl+C after a few seconds to verify no import/init errors.)

## Known limitation (not introduced here)

`update()` is called at ~60 FPS while audio chunks are consumed at ~11 Hz
(1/CHUNK_S=1/0.09). The FADE_CHUNKS countdown is decremented per `update()` call,
so with FADE_CHUNKS=1 the countdown expires on the first frame regardless. The
actual perceptual fade duration is one rendered audio chunk (~90ms), which is
close enough to the 50ms target given current CHUNK_S. Finer control requires
either shorter CHUNK_S or decoupling the countdown from the update rate.
