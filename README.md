# Game of Life → Synthesizer

A research / prototyping project: build a **sound synthesizer driven by Conway's
Game of Life**. The goal is to turn the richness of cellular-automaton patterns
into a rich, controllable space of *timbres*, mapped so the sound matches human
perceptual intuition.

This repo holds a working interactive prototype plus the design knowledge
accumulated so far. It is meant to be the knowledge base for further research and
prototyping (including AI agents).

## Status at a glance
- Working interactive desktop app: paint a Game of Life field, play a note on an
  on-screen piano, and hear the automaton shape the timbre in real time
  (continuous additive synthesis).
- The pitch / "height" axis is realized as **harmonics of a carrier note**, so it
  is a *synthesizer* (shaping one tone), not a *sequencer* (separate notes).
- ~3 of the planned perceptual mapping axes are wired; several more are designed
  but not yet implemented (see `docs/02-design-space.md`).

## Docs
- `docs/01-problem.md` — what we're building and why (the task and its perceptual core).
- `docs/02-design-space.md` — every sub-problem explored, the options, the decisions
  + rationale, and the mapped-out roadmap. **The main reference.**
- `docs/03-prototype.md` — what the current prototype does, how it's built, and its limits.
- `CLAUDE.md` — operating notes for AI agents (load-bearing decisions, how to run, conventions).

## Project structure
```
docs/          reference docs (read-only): problem statement, design space, prototype notes
roles/         role instructions for Researcher and Developer agent sessions
memory/        version-controlled project memory: decisions, questions, current state, session log
```

## Prototype files
- `gol_life_synth.py` — the interactive app (main artifact).
- `gol_synth.py` — earlier offline renderer (writes a WAV of a fixed scene).
- `gol_synth_demo.wav` — sample output of the offline renderer.

## Run
```
pip install pygame-ce numpy scipy
python gol_life_synth.py
```
Use `pygame-ce`, not `pygame` (see `docs/03-prototype.md` for why).
