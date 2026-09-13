# Demo bench (S1 + S2)

Run (Windows, project Python with deps installed): `run_demo_bench.bat`
(opens `demos/laplace_ab.json`) or
`python demo_bench.py --demo demos/laplace_ab.json`.

Offline render, no window / no audio device:
`python demo_bench.py --demo demos/laplace_ab.json --render out.wav --seconds 8 --side A|B|monitor`
(`A`/`B` = the raw side, `monitor` = what is heard incl. the A/B crossfade; default A).

Dependencies: `pip install -r requirements.txt` (numpy, scipy, pygame-ce, sounddevice).

Transport: the bench opens ON PAUSE (silent) -- release **Pause CA** to go
(field evolves + sound); while running Pause CA freezes only the automaton
(painting and sound continue).  **Stop** (S) = initial scene + pause + silence;
**Restart** (R) = initial scene, clocks and all audio tails reset, runs again.  Stop/Restart keep the
engines, parameters, volume and the selected side.  LMB paints, RMB erases.

Hotkeys: **S** = Stop, **R** = Restart, **Space** = Pause CA, **1** / **2** = select A / B.  `<<` / `>>` copy ALL settings (engine + params) B->A / A->B;
**Factory A+B** restores both sides to the scene defaults; a `*` next to A or B
marks a side that differs from those defaults.

A/B: one shared field, two sound sides computed all the time on the same
timeline.  The **A** / **B** tabs pick the side you listen to AND edit
(20 ms crossfade on switch).  The panel shows the selected side's engine
(Laplace / FFT / Walsh / Random / Granulo from `casynth_core.ENGINES`), its
knobs with the registry ranges, and a one-line summary of how A and B differ.
Settings are remembered per side x engine.

Scenes: `laplace_basic.json` = format 1 (one engine, loaded as A = B);
`laplace_ab.json` = format 2 (`variants` A/B + `listen` hint).  Strictly
validated: unknown engine / parameter, out-of-range value or bad cell -> error.

## Saving experiments and the local catalog (S4)

Recording starts by itself with Start, and starts OVER at every Restart and at
every Start after a Stop: the record's conditions are the state at that moment
(field, engines/params of A and B incl. their per-engine memory, factory
defaults, selected side, volume).  Only the **last 30 s** of audio are kept
(the command journal since that start is kept in full).  **Save** fixes the
end at the current audio block (typing the title/note afterwards does not
extend it) and stores the conditions, the journal and three WAVs (raw A, raw B,
monitor = as heard, incl. the A/B crossfade).  The live experiment keeps
running.  Records live in `lab_catalog/local/<id>/`
(`record.json` + `A.wav` `B.wav` `monitor.wav`; user data, gitignored).

**Catalog** is a separate screen: the CA is paused and the live synth muted
while it is open (only records may sound there); Esc / Back returns and resumes.
Select a record -> **Continue** (S5, see below), or **Open field anew** loads
its END state (field, engines and parameters of A and B, selected side,
volume) into a fresh live session -- fresh sound, CA paused -- that you start
manually; **Play A / Play B / Play as heard** (the live synth is
muted meanwhile); **Check reproducibility**
recomputes the experiment in a separate process from the embedded conditions
(from the recording's start; the saved window is compared) byte-exact with the
stored WAVs: "Replay matched" /
"Replay differs" / a reason why it is unavailable (e.g. the engine is not
registered any more -- the WAVs still play).  The replay output can be
listened to (**Play replay**); the originals are never modified.
Esc / Back returns to the live view.  Headless use:
`python -m casynth_lab.catalog replay lab_catalog/local <id>`.

## Continuing a find and branching (S5)

Every **Save** also stores a full **snapshot** of the runner at the same block
where the WAVs and the journal end (`state_end.json` + `state_end.npz`:
field, excitation, generation, all clocks, running/paused, volume, both sides'
engine + parameters + per-engine memory, the A/B monitor crossfade, every
oscillator phase / amplitude / tail slot / envelope of both engines, and the
commands already accepted for the future).  In the catalog **Continue**
restores that snapshot into the live session and carries on from the next
audio block -- no history is replayed, nothing is re-analysed with zeroed
phases: the following blocks and generations equal an uninterrupted run
(checked byte-exact by `tests/test_demo_lab_s5.py` for all five engines).  A
saved Pause CA stays paused (sound continues), a saved Stop stays stopped.
The session is replaced only after the snapshot has been validated; a
corrupt / incompatible / missing snapshot leaves the current session as it is
and says why ("Continue unavailable: ..." -- the WAVs still play, **Open field
anew** still works).  S4 records (no snapshot) are read as before, without
migration.

A continued session records from the snapshot on (the 30 s window starts
empty; the parent's audio is not glued in).  Its **Save** makes a **branch**:
an independent record with its own copy of the origin snapshot, its own
journal, three WAVs, end snapshot and `parent_record_id`.  The card shows
**Derived from: <title>** (click = jump to the parent); the parent is never
modified and a branch stays reproducible even when the parent folder is gone.
**Check reproducibility** of a branch restores its origin snapshot and applies
its journal.  Several Saves from one continued session share the same parent
and start point; continuing a branch makes the branch the parent.  Stop /
Restart in a continued session begin a fresh recording (fresh conditions, as
in S4) that still carries the parent link.  **Continue** on the same record
always starts from its unchanged snapshot and creates no record by itself.

## Code versions: pinned and local records (S6)

Every Save also records **which code produced it**: an explicit runtime set
(`demo_bench.py`, `casynth_lab/*.py`, `casynth_core.py`, `casynth_engine.py`,
`casynth_config.py`, `requirements.txt`, `demos/*.json`) with per-file
fingerprints, the Git commit, whether the set's CONTENT equals that commit
(staged, unstaged and new runtime files all count; docs / tests / memory do
not; CRLF checkouts are fine), and the environment (Python, exact package
versions, OS).  It is captured once per process for the code that was
imported -- not the HEAD at Save time; workers and version benches report
their own.

- **Pinned <sha>**: the runtime set matched a commit, and that commit is held
  by a permanent ref (`refs/casynth/pins/<commit>`) so it survives branch
  deletion and `git gc`.  Pinned means established origin -- whether the
  version can run here and whether the PCM reproduces are separate checks.
- **Local: <reason>**: uncommitted runtime changes, no repository, a failed
  ref, or a record from before S6 ("code version not recorded" -- never
  attributed to HEAD after the fact).  Local records keep everything else.
- **Pin to commit** (local records): after committing exactly that code the
  record is attached to the commit found (HEAD or recent history); only the
  origin fields change.  Different code -- even with identical WAVs -- is
  refused; run a new experiment instead.
- **Check reproducibility** always computes with THIS bench's code.
- **Continue**: same runtime fingerprint here -> in this bench ("this
  version"); another pinned version -> "Continue in version <sha>" starts a
  separate bench of that commit from a cached detached checkout
  (`lab_catalog/worktrees/<commit>/`, verified / repaired from the held
  commit; the main checkout, index and branch are never touched) when the
  environment is compatible (same Python major.minor, exact package
  versions).  This bench stays muted in the catalog until the other one
  closes; its saves land in the same catalog, pinned to their version, with
  the parent link.  Local record -> current code by the S5 rules.  No commit,
  no environment, or an unsupported (pre-S6) version -> a concrete reason,
  the WAVs still play.

Launch contract of a version bench / worker:
`python demo_bench.py --catalog <abs root> --record <id> --action continue|check`
(`--headless --seconds N --out file.npz` renders from the snapshot without a
window).  The first stdout line is the process's provenance (JSON).

Engines take part through two optional methods of the S3 interface
(`export_state()` / `restore_state(grid, exc, state)` + a class
`STATE_VERSION`, see `casynth_lab/engine_api.py`); the five built-in methods
get them from the shared adapter.  An engine without them still works and
saves WAVs, but its records are honestly marked as not continuable.
Formats: `record.json` `format` 2 (1 = S4, still read), runner state version
1 (`casynth_lab/runner.py`), engine state version per class, snapshot file
version 1 (`casynth_lab/snapshot.py`: JSON + npz, no pickle).

## Checking the catalog after a code change (S7)

**Check catalog** (catalog screen) recomputes EVERY record listed at that
moment -- pinned, local and old ones alike -- with the code of ONE worker
process (its provenance and environment are the report's target: `Pinned
<sha>` or `Local: <reason>`), applying the recorded conditions and journal
exactly as the single **Check reproducibility** does (same replay path, no
second algorithm; a branch starts from its origin snapshot, a plain record
from its start conditions; nothing is normalised or time-shifted).  All three
tracks of the saved window are compared with the stored WAVs: **A**, **B**
and **As heard** (monitor), each on its own.  The bench stays responsive
(progress "N of M", the current record, **Cancel**); one check at a time --
no single check and no second worker meanwhile.  An error in one record does
not stop the others; a cancel keeps the finished results and marks the rest
"not checked".  If the runtime set on disk changes during the run the run
stops: what was finished is kept, the record in progress and the rest are
"not checked" (one report never mixes two versions).

Per track the report holds one of:
- **Exact match** -- same format, length and every PCM sample (WAV header
  details are irrelevant);
- **Differs numerically** -- a comparable pair with sample or length
  differences; details: the share of differing samples and the maximum
  absolute difference in PCM units (computed without int16 overflow), a length
  mismatch is shown as such (the tail is never cut);
- **Could not check** -- no valid pair, with the concrete reason (engine no
  longer registered, snapshot of another version, unreadable WAV, ...);
  "not reproducible" is never replaced by a fresh start from the field;
- **Not checked** -- not reached, cancelled, or the target version was lost.

Reports live in `lab_catalog/local/.verify/<run>/` (`run.json` + the
recomputed WAVs of DIFFERING tracks only + `marks.json`), survive closing the
application (**Last report**, `< older` / `newer >`), and are never
recomputed on viewing.  Records, their WAVs, snapshots, Git links and parent
links are untouched; a new check makes a new report; the single check's
`<record>/replay/` never overwrites a report's pair.  Every pair is
fingerprinted: a replaced or lost file is detected when you open it and the
old verdict is not shown as valid for it.

The report screen groups records into **Differs / Exact / Failed /
Unchecked**; opening a differing record selects the changed **As heard**
track (else the first changed A/B).  Listening: track **A / B / As heard**,
version **Saved / Recomputed** (Space toggles), **Play / Stop** -- both
versions run on ONE cursor: switching continues from the same sample, never
restarts; the switch itself is smoothed by a 10 ms fade inside the player
only (the compared PCM is untouched); same gain for both, no automatic level
matching, no live synth mixed in; with different lengths the shorter version
is silence after its end.  A small numeric difference never gets an
automatic "inaudible" label: the optional **listening mark** -- *Can't hear /
Hear it / Not rated* -- belongs to that report's pair of that track, survives
a restart and is not copied to a new report.  No major-version, golden-master
or "old version needed" decision is made for you.

Headless: `python -m casynth_lab.verify run lab_catalog/local` (JSON lines).
Demo catalog for the acceptance: `python tests/s7_demo_catalog.py`, then
`run_demo_bench.bat --catalog lab_catalog\s7_demo`.

## Adding a sound engine (S3 interface)

The bench owns the field, clocks, command queue/journal, transport, the A/B
instances, the monitor and WAV export.  An engine is one class per side that
turns the field into stereo blocks.

1. **Module** — put it under `casynth_lab/` (e.g. `casynth_lab/my_engine.py`)
   and subclass `casynth_lab.SoundEngine`:
   ```python
   from casynth_lab import SoundEngine
   class MyEngine(SoundEngine):
       def init(self, grid, exc, gain): ...        # (re)start on the CURRENT field, silent
       def update_field(self, grid, exc): ...      # field changed (step / painting)
       def set_params(self, params): ...           # full validated dict (call super())
       def render(self, gain, t_samples): ...      # -> (int16 (ctx.block, 2), peak, n_clip)
       def reset(self, gain): ...                  # == init on the same field
       STATE_VERSION = 1                            # optional (S5): exact continuation
       def export_state(self): ...                 # -> dict of scalars/lists + ndarrays
       def restore_state(self, grid, exc, state): ...  # rebuild so render() continues exactly
   ```
   `self.ctx` gives `sr`, `block`, `channels`, `f0`, `level`, `rate_hz`.  Apply
   `gain` once (pre-clip).  No wall-clock time, never write to `grid`.
2. **Register** — explicitly, in one place (`casynth_lab/registry.py`, after the
   built-in five, or in your module imported from there):
   ```python
   from casynth_lab import EngineSpec, register
   register(EngineSpec('my_engine', 'My', [('depth', 'depth', 0.0, 1.0, False, 0.5)],
                       lambda ctx, params: MyEngine(ctx, params)))
   ```
   Param spec = `(arg, label, lo, hi, integer, default)`; integer 0/1 renders as
   a toggle.  The UI buttons/knobs, scene validation, commands and `--side`
   export all come from the registry -- no other Python changes.
3. **Scene** — reference it from JSON only:
   `"variants": {"B": {"engine_id": "my_engine", "engine_params": {"depth": 0.5}}}`.
4. **Check** — `python tests/test_demo_lab.py` (interface + audio references) and
   `python check.py`.  The bench rejects malformed blocks (wrong shape/dtype):
   they are replaced by silence and counted in the audio status.
