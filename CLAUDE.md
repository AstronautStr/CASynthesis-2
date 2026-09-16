# CLAUDE.md — agent operating notes

## What this project is
A synthesizer whose timbre is driven by a Cellular Automata.

## Project memory (version-controlled, under `memory/`)
Read these at the start of every substantive session (lean — live state, not history):
- `memory/current.md` — what is implemented, in-flight, prioritised next (snapshot, not a changelog)
- Researcher sessions: first read `memory/research/sonification-criteria-2026-09-13.md` — the user-accepted central principles of the synthesizer — then `memory/research/research-resume-2026-09-13.md` for the handoff and current research status.

Consult on demand, NOT at session start:
- `memory/archive/` — resolved questions and trimmed history (recall a settled item)
- `memory/log/` — per-session summaries (detailed record of how each fix was done)
- `memory/research/` — Researcher reports/proposals BEFORE acceptance (accepted → a
  `decisions.md` entry points back here); `memory/req-*.md` — standalone REQ briefs
- `docs/01-problem.md`, `docs/02-design-space.md`, `docs/03-prototype.md` — read-only reference; read when a decision touches design rationale (do not append)

## Gates & environment canon (Windows)
- **All regression gates in one command: `python check.py`** (59 unit tests +
  golden-master audio byte-exact + UI frame pixel-exact + import/init smoke).
  Legitimate UI change → re-bless in the SAME change: `python check.py --bless-ui`.
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
