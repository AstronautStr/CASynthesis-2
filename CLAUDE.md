# CLAUDE.md — agent operating notes

## What this project is
A synthesizer whose timbre is driven by a Cellular Automata.

## Project memory (version-controlled, under `memory/`)
Read these at the start of every substantive session (lean — live state, not history):
- `memory/current.md` — what is implemented, in-flight, prioritised next (snapshot, not a changelog)
- Researcher sessions: first read `memory/research/sonification-criteria-2026-09-13.md` — the user-accepted central principles of the synthesizer — then `memory/research/research-resume-2026-09-13.md` for the handoff and current research status.
- Before proposing, specifying, handing off, or revising listening experiments, Researcher must apply
  [the listening-experiment pipeline](memory/research/listening-experiment-pipeline.md).
  Establish a nontrivial CA-to-sound question and a decision the listener's answer can change before
  drafting a detailed REQ. Routine parameter demonstrations and repeats of an already rejected sound
  do not automatically merit another listening task. The pipeline is an internal work step, not an
  extra user approval flow. Developer follows the resulting REQ and flags missing experiment intent
  without inventing additional listening cases.

Consult on demand, NOT at session start:
- `memory/archive/` — resolved questions and trimmed history (recall a settled item)
- `memory/log/` — per-session summaries (detailed record of how each fix was done)
- `memory/research/` — Researcher reports/proposals BEFORE acceptance (accepted → a
  `decisions.md` entry points back here); `memory/req-*.md` — standalone REQ briefs
- `docs/01-problem.md`, `docs/02-design-space.md`, `docs/03-prototype.md` — read-only reference; read when a decision touches design rationale (do not append)

## Gates & environment canon (Windows)
- **Gates (2026-09-23): `python check.py` after EVERY change, `python check.py --all`
  before EVERY commit.** The default run picks the gates by the files changed since HEAD
  (a map in check.py plus a search for the changed file's name in the gates' scripts) on
  top of the fast set (~0.5-2 min). `--all` runs every gate (~4 min): the gates that
  MEASURE time first and alone, the rest six at a time, longest first, OpenBLAS on two
  threads in that wave. `--dry-run` shows the pick, `--files=a,b` / `--since=REV`
  redefine "changed", `--fast` is the fast set alone (~30 s), `--jobs=1` runs one at a
  time, `-v` prints every gate's output. The map over-selects on purpose and an unknown
  file runs everything; what it still misses, `--all` catches before the commit.
  Legitimate UI change → re-bless in the SAME change: `python check.py --bless-ui`.
  A test that asserts milliseconds must carry `@timing_test` (tests/timing_gate.py),
  or the parallel wave measures the scheduler instead of the engine.
  Gates run the instrument at `CASYNTH_VOLUME=0.01` -- a check run must not play
  music at whoever is sitting there.
- Tests alone: `python tests/test_casynth_core.py` (stdlib runner, pytest not installed).
- Versioned baselines live in `tests/golden/` (in git). `artifacts/` is for transient
  diagnostics only (gitignored, swept during /dream) — never keep a baseline there.
- `git` is NOT on PowerShell's PATH here — run git via Git Bash (the Bash tool).
- Keep prototype/script stdout ASCII-only (Windows console cp1251 chokes on λ/emoji;
  set `PYTHONUTF8=1` if non-ASCII output is unavoidable).

## Working conventions (collaborator's preference)
- Propose before changing; offer devil's-advocate critique; decide one thing at a time.
- Keep changes small and verifiable; the prototype should always run.
- Collaborator works primarily in Russian; English is fine too.
- Git commit messages are always in English (even though discussion is in Russian).
