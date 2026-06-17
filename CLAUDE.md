# CLAUDE.md — agent operating notes

## What this project is
A synthesizer whose timbre is driven by a live Conway's Game of Life field. Each
connected object in the field is a **harmonic of a carrier note** played on an
on-screen piano; the field = the evolving spectrum of that note.

`docs/01-problem.md`, `docs/02-design-space.md`, and `docs/03-prototype.md` hold the
full context and reasoning — consult them when a decision touches design rationale
(reference, not mandatory each session).

## Load-bearing decisions (do not silently reverse)
1. **Hand-crafted, interpretable perceptual features are the mapping backbone —
   NOT a learned embedding.** To map a feature to a *perceptually correct* synth
   parameter (e.g. roundness → low-pass) you must know what the feature means;
   entangled/latent axes risk *active mismatch*. Cross-modal correspondence
   research (bouba/kiki, etc.) is reused as pre-collected human-association data.
   A learned embedding is at most a *secondary* layer for uniqueness / residual
   texture. (Full reasoning in `docs/02-design-space.md` §C.)
2. **The unit of sonification is the perceptual object** — not the cell, not the
   whole field. Each object → one voice / partial. The field is an auditory scene
   (Bregman ASA / Gestalt grouping).
3. **Synthesizer, not sequencer.** The size / "height" axis = harmonic number of
   one carrier note, not separate notes.
4. **Audio must stay continuous and constant-amplitude** (phase-coherent partials,
   glided amplitudes), never per-generation pulses.

To revisit any of these, surface it explicitly with rationale.

## How to run
```
pip install pygame-ce numpy scipy sounddevice
python gol_life_synth.py
```
- `gol_life_synth_laplacian.py` uses **sounddevice** (PortAudio callback stream)
  for gapless audio decoupled from the 60 fps loop; pygame's play/queue mixer
  starved on normal frames and clicked. `gol_life_synth.py` still uses the mixer.
- Use **pygame-ce** — mainline `pygame` has no Python 3.14 wheel and fails to
  build (`distutils.msvccompiler` removed in 3.12+). pygame-ce is API-compatible
  (`import pygame`).
- The mixer is forced to stereo (`allowedchanges=0`); some devices open as
  8-channel surround and would break stereo buffers. The buffer also adapts to the
  actual channel count as a fallback.
- Headless sanity checks: prefix with `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`
  to validate logic / draw / synth without a display or audio device (works in CI).

## Project memory (version-controlled, under `memory/`)
Read these at the start of every substantive session (lean — live state, not history):
- `memory/current.md` — what is implemented, in-flight, prioritised next (snapshot, not a changelog)
- `memory/decisions.md` — Researcher decision log; **read the index at the top**, full entry on demand
- `memory/questions.md` — live questions (OPEN + items awaiting review)
- `memory/decision-tree.md` — дерево развилок проекта (Mermaid): корень → ветки → текущая точка

Consult on demand, NOT at session start:
- `memory/archive/` — resolved questions and trimmed history (recall a settled item)
- `memory/log/` — per-session summaries (detailed record of how each fix was done)
- `docs/01-problem.md`, `docs/02-design-space.md`, `docs/03-prototype.md` — read-only reference; read when a decision touches design rationale (do not append)

Role definitions live in `.claude/agents/researcher.md` and `.claude/agents/developer.md`
(single source — full role + subagent frontmatter; valid for subagent and standalone sessions).

## Working conventions (collaborator's preference)
- Propose before changing; offer devil's-advocate critique; decide one thing at a time.
- Keep changes small and verifiable; the prototype should always run.
- Collaborator works primarily in Russian; English is fine too.
