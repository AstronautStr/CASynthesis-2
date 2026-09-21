# Demo bench (S1 + S2)

Revised N1 experiments (2026-09-16): `run_network_n1.bat` opens three prepared
records in `lab_catalog/network_n1_hypotheses_2026_09_16_r2/`. Each record's Notes
starts with the researcher's hypothesis; append listening results there.
Build into an empty directory: `python demos/build_n1_hypotheses.py --root DIR`.
Both sides follow the live field. The comparisons test added field control of
damping and delayed links. Technical verification: `python demos/n1_hypotheses_report.py`.

Run (Windows, project Python with deps installed): `run_demo_bench.bat`
(opens `demos/laplace_ab.json`) or
`python demo_bench.py --demo demos/laplace_ab.json`.

Offline render, no window / no audio device:
`python demo_bench.py --demo demos/laplace_ab.json --render out.wav --seconds 8 --side A|B|monitor`
(`A`/`B` = the raw side, `monitor` = what is heard incl. the A/B crossfade; default A).

Dependencies: `pip install -r requirements.txt` (numpy, scipy, pygame-ce, sounddevice).

The bench opens on the **Catalog** screen (2026-09-14): pick a record and press
**Continue**, or double-click it -- the same thing; Esc / **Back** reaches the
live scene of the launch file.  `--live` opens the sounding field directly
(autotests, scripted launches); a continued record always opens live.

Transport: the live scene starts ON PAUSE (silent) -- release **Pause CA** to go
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
(Laplace / FFT / Walsh / Random / Granulo from `casynth_core.ENGINES`, plus
Scan / Network, see below), its knobs with the registry ranges (named modes as
word buttons), and a one-line summary of how A and B differ.
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

Every Save also records **which code produced it**: the **sound set** --
the files that can change the PCM of an experiment given its embedded
conditions (`casynth_core.py`, `casynth_engine.py`, `casynth_config.py`, the
`casynth_engines/` package -- `engine_api`, `registry`, `legacy_engine`, the
engine modules -- the casynth_lab modules that turn a scene + journal +
snapshot into blocks: `runner`, `scene`, `snapshot`, and the shims casynth_lab
keeps on the moved engine names) with per-file fingerprints, the Git commit, whether the
set's CONTENT equals that commit (staged, unstaged and new sound files all
count; the bench UI `demo_bench.py`, `audio_out` / `catalog` / `recorder` /
`verify` / `versions` / `provenance`, `requirements.txt`, `demos/*.json`,
docs / tests / memory do NOT -- editing them never re-pins an experiment;
CRLF checkouts are fine), and the environment (Python, exact package
versions, OS).  It is captured once per process for the code that was
imported -- not the HEAD at Save time; workers and version benches report
their own.  Records made with an older definition of the set -- set 1 (before 2026-09-14,
which also fingerprinted the bench) and set 2 (before the engines moved into
`casynth_engines/` on 2026-09-21) -- are compared by the content of the sound
set at their commit, so they stay continuable here when only non-sound files
changed.  Diagnosis:
`python -m casynth_lab.provenance` (this checkout) and
`python -m casynth_lab.provenance why lab_catalog/<root> [id...]` (per record:
same sound code, or which sound files differ).

- **Pinned <sha>**: the sound set matched a commit, and that commit is held
  by a permanent ref (`refs/casynth/pins/<commit>`) so it survives branch
  deletion and `git gc`.  Pinned means established origin -- whether the
  version can run here and whether the PCM reproduces are separate checks.
- **Local: <reason>**: uncommitted sound-set changes, no repository, a failed
  ref, or a record from before S6 ("code version not recorded" -- never
  attributed to HEAD after the fact).  Local records keep everything else.
- **Pin to commit** (local records): after committing exactly that code the
  record is attached to the commit found (HEAD or recent history); only the
  origin fields change.  Different code -- even with identical WAVs -- is
  refused; run a new experiment instead.
- **Check reproducibility** always computes with THIS bench's code.
- **Continue**: same sound code here -> in this bench ("same sound code as
  version <sha>"); other sound code -> "Continue in version <sha>" (the card
  names the differing sound files) starts a
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
`STATE_VERSION`, see `casynth_engines/engine_api.py`); the five built-in methods
get them from the shared adapter.  An engine without them still works and
saves WAVs, but its records are honestly marked as not continuable.
Formats: `record.json` `format` 2 (1 = S4, still read), runner state version
1 (`casynth_lab/runner.py`), engine state version per class, snapshot file
version 1 (`casynth_lab/snapshot.py`: JSON + npz, no pickle).

## Listening notes (2026-09-14)

**Notes** (live view, next to Catalog; also on the catalog screen for the
selected record) opens the free-text notes of the record the session belongs
to -- the record you continued / opened, or the last one you saved in this
session (a fresh session has none yet: Continue a record or Save first).
Text is saved as you type into `<record>/notes.md` (UTF-8; Enter = new line,
Ctrl+Backspace = erase a word, Ctrl+A/X/C/V, Esc closes).  **Since 2026-09-17 the
same form lives in a window of its own** (`casynth_lab/notes_window.py`: a second
`pygame.Window` with the bench's own drawing and text model; the main loop routes
the events that carry that window to the form): a real window with a title bar
(minimise / close), moved next to the bench; click it to write, click the bench to
play and paint -- the field is never blocked.  The window belongs to one record: it
closes when the notes would go to another record (another record selected in the
catalog, Back to the catalog, Continue / Save of another one), when the OS closes
it and with the bench; "Notes (open)" marks it in the catalog panel.  Two different
things, two names: **Description** = the text typed at
Save (`note` in record.json, never edited later, shown in the catalog panel);
**Notes** = `<record>/notes.md`, the living listening journal.  WAVs, snapshots
and record.json are never touched by Notes.  Records with notes carry a "notes"
tag in the catalog list.

The pipeline: listen -> write the impression right there -> the agent reads
it back with

    python -m casynth_lab.catalog notes lab_catalog/<root>        # Markdown: title, id, engines, text
    python -m casynth_lab.catalog notes lab_catalog/<root> --all  # + records with a Save description only

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
"not checked".  If the sound set on disk changes during the run the run
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

## Scan / Network sound demos (2026-09-14)

Two engines from `memory/req-sonification-sn-demos-2026-09-14.md`, both
treating the WHOLE field as one sound object (no per-object segmentation):

- **Scan** (`scan_surface`, `casynth_engines/scan_surface.py`): the mask becomes
  a surface (`Surface`: **Smooth** = 2*Gaussian(g, width)-1, periodic;
  **Distance** = tanh((d_dead - d_live)/width) on the torus) and a fixed
  closed path reads it once per note period (`Path`: **Ellipse**,
  **Lissajous** 2:3, **Raster** = snake through every cell, closed across the
  torus seam; `Radius X/Y` act on the two analytic paths only -- the panel
  shows `Full field` for Raster).  The reading is mean-removed, band-limited
  to K = floor(0.45*sr/f0) COMPLEX harmonics and played additively with a
  continuous phase; any change of field / surface / path / radii crossfades the
  coefficient vectors over 20 ms.  The path is drawn over the field for the
  listened side (start dot + arrow = direction), with the same geometry the
  engine reads.
- **Network** (`pm_network`, `casynth_engines/pm_network.py`): four generators at
  ratios 1..4 of f0 on one phase; six directed phase-modulation links (higher
  index modulates lower, same sample, no feedback) whose depths W are the
  field's mass under six fixed Gaussian masks (two rows x three columns, drawn
  as circles `j>i` = j modulates i); `Coupling` = beta, `Frozen links` holds
  the current W.  Runs at 4x with a linear-phase decimator and a 5 Hz
  high-pass; W / beta smoothed over 30 ms; an empty field closes a 20 ms
  gate.  The six current depths are shown as bars (tick = target).

`Level (dB)` on both = constant trim applied once with the master gain
(defaults Scan +6 dB, Network +3 dB: RMS matched to the Laplace side on the
Pulsar scene; never per record).  Both engines snapshot completely (Continue
is byte-exact; `tests/test_demo_lab_sn.py`).

Entries: **`run_scan_demo.bat`** / **`run_network_demo.bat`** open the bench
on the catalog screen (a demo scene stands behind it) with the prepared catalog
`lab_catalog/sn_demos_2026_09_14/{scan,network}/`.  Build the catalog once
(it is user data, not in Git; the build refuses an existing directory):
`python demos/build_sn_demos.py` (scenes `demos/sn_*.json` are rewritten
identically; `--scenes-only` skips the recording).  Records: 8 s static
(the CA paused from the first block; Continue = sound on, CA paused) or
12 s evolution (Pulsar at 2 steps/s; Continue keeps evolving).

| Record (catalog) | Scene | Field | A | B | Purpose |
|---|---|---|---|---|---|
| S-path: Ellipse / Lissajous (scan) | `sn_s_path_ellipse_lissajous` | F1 static | Scan Smooth Ellipse | Scan Smooth Lissajous | path only |
| S-path: Ellipse / Raster (scan) | `sn_s_path_ellipse_raster` | F1 static | Scan Smooth Ellipse | Scan Smooth Raster | path only |
| S-surface: Smooth / Distance (F1) (scan) | `sn_s_surface_f1` | F1 static | Scan Smooth Raster | Scan Distance Raster | surface only |
| S-surface: Smooth / Distance (F2) (scan) | `sn_s_surface_f2` | F2 static | same | same | field change |
| S-Laplace (scan) | `sn_s_laplace` | F3 Pulsar 12 s | Laplacian harm=0 | Scan Smooth Raster | whole method |
| S-Laplace (Distance) (scan) | `sn_s_laplace_distance` | F3 12 s | Laplacian harm=0 | Scan Distance Raster | named extra |
| S-Laplace (harm=1) (scan) | `sn_s_laplace_harm1` | F3 12 s | Laplacian harm=1 | Scan Smooth Raster | harmonicity control |
| N-links: independent / connected (top) (network) | `sn_n_links_top` | F5 top static | Network coupling 0 | Network coupling 2 | links |
| N-links: ... (bottom) (network) | `sn_n_links_bottom` | F5 bottom static | coupling 0 | coupling 2 | same count, other W |
| N-field: Pond left / right (network) | `sn_n_field_left` / `_right` | F4 static | coupling 0 | coupling 2 | small-object control |
| N-frozen: live / frozen links (network) | `sn_n_frozen` | F3 12 s | live links | frozen links | evolution of W |
| N-frozen (Kok's galaxy) (network) | `sn_n_frozen_kok` | Kok's galaxy 12 s | live links | frozen links | REQ reserve case (W moves little for both oscillators) |
| N-Laplace (network) | `sn_n_laplace` | F3 12 s | Laplacian harm=0 | Network coupling 2 | whole method |
| N-Laplace (harm=1) (network) | `sn_n_laplace_harm1` | F3 12 s | Laplacian harm=1 | Network coupling 2 | harmonicity control |

Fields: F1 `g[r,c]=1 if (r//4 + 2*(c//5)) % 3 == 0`; F2 = F1 transposed;
F3 = Pulsar (anchor 9,9); F4 = Pond at (6,6) / (6,21); F5 = rows < 16 /
rows >= 16.  Laplace control: n=12, spread=0, alpha=1, shape=1, harm=0 (1 in
the named control), fullshape=1, dyn=0.  Known near-silent case: the analytic
paths at radius 0.8 pass around the centred Pulsar (F3) and read only the
Smooth tails -- F3 is prepared with Raster only.

## N1 -- the Gutter Synthesis network driven by the field (2026-09-15)

`gutter_field` (`casynth_engines/gutter_field.py`, REQ
`memory/req-network-ca-n1-2026-09-15.md`): the eight-node Gutter Synthesis
network of the N0 reference (`demos/network_reference_n0/`) as a live bench
engine.  The model is FIXED by `casynth_engines/gutter_field_n1_config.py`
(= the Researcher's fixtures `memory/research/network-n1-fixtures-2026-09-15.json`:
44.1 kHz, explicit 8 x 24 bank frequencies, N0 node values, matrix delay
2000 + 64 samples, SVF q = 0.99, all-to-all links, the review-R1 output routes
-- node 6 has no right outlet, neither in the master sum nor at the matrix
input; `post_math = "scalar"`, `model_version = gutter_field_n1_v1`).  The
per-sample arithmetic is the scalar node port's (Java-verified) statement
order, executed by ONE numba kernel per block (~0.4 us per sample per side
instead of ~52 with per-sample numpy; without numba the same function runs as
plain Python, correct but far too slow for live use -- `numba` is in
`requirements.txt`).  `tests/test_gutter_field_n1.py` (gate 1i) checks the
kernel bit-exactly against the slow model in its scalar mode and against the
scalar node port in lockstep.

Field -> sound: the field is cut into 2 rows x 4 columns of regions (drawn
over the field with the node numbers 0..7, `i = 4 r + c`); per region
`n = live cells`, `u = n / (n + 2)`, `ratio = 2 ** clip(log2(scale) + depth *
(2 u - 1), -1, 1)`, and every bank frequency of that node becomes
`min(19000, base * ratio)` -- applied atomically at the block boundary as
float32 setFreqN messages (coefficients recomputed, all states kept, no
interpolation, no accumulation).  Nothing resets the network: painting,
evolution, parameter moves and the CA pause keep it running; Stop / Restart
are the bench transport.  Controls: `Resonators` (scale 0.5..2), `CA amount`
(depth 0..1; 0 = the field does not act), `Links` (source interaction slider
0..256, `(v/256)^2 * 5`, 50 ms ramp), `Freeze CA` (holds the control vector u
reached so far; manual knobs keep acting on it; off takes the current field;
at init the held vector is the initial field's).  The panel shows the held
u per node as bars (tick = the field's own u), the cell count, the ratio, the
current link gain and the node reset count.  Level: raw sum x 20, then the
bench gain once (0.56 effective at the defaults); the clip counter measures
before int16.  Snapshot = everything (ring delays, nodes, all filters,
coefficients / frequencies, ramps, held vector, model version): Continue is
byte-exact.

Entry: **`run_network_n1.bat`** opens the bench on the CATALOG screen of
`lab_catalog/network_n1_2026_09_15/` (every listening goes through the
experiment catalog: open the records in turn, compare A / B, write Notes;
Continue = the live field from the record's end state).  Build the catalog once
with `python demos/build_n1_demos.py` (refuses a catalog that already holds
records; `--scenes-only` rewrites `demos/network_n1_blinkers.json` +
`demos/n1_*.json` identically).  All records: 32 x 32, B3/S23, torus, four
vertical blinkers in columns 4 / 12 / 20 / 28 (rows 14..16), 12 s of evolution
at 2 generations / s unless noted:

| Record | Scene | A | B | Purpose |
|---|---|---|---|---|
| N1-follow: follows the field / Freeze CA | `network_n1_blinkers` | Gutter | Gutter, Freeze CA | the REQ's A / B |
| N1-edits: the field moved and back every 4 s | `n1_edits` (20 s, CA paused) | Gutter | Gutter, Freeze CA | the fixtures' edit schedule, no restart |
| N1-depth: CA amount 1 / 0 | `n1_depth` | Gutter depth 1 | Gutter depth 0 | the field acting vs manual scale only |
| N1-links: Links 127 / 230 | `n1_links` | Gutter Links 127 | Gutter Links 230 | coupling strength |
| N1-routes: R1 fix / N0 routing | `n1_routes` | Gutter | Gutter (N0 routes) | review R1 as a listening control (`gutter_field_n0r`: the same model with node 6 also on the right) |

Measurements (levels, spectra per 2 s window, the
fixtures' edit schedule, invariance with depth 0 / Freeze CA, block timing
p95 / p99 offline and on the device): `python demos/gutter_field_n1_report.py`
-> `demos/results/network_n1/report.{json,md}` (WAVs in `artifacts/_n1/`).

## N2 -- field events excite a delay network (2026-09-16)

REQ `memory/req-network-events-n2-2026-09-16.md`: one live field, two sides
that read it through the SAME periodic weights (`casynth_engines/periodic_readout.py`:
eight overlapping raised-cosine weights K_i, a partition of unity, continuous
on the torus -- no region borders).

- **A `gutter_field_periodic`** ("Gutter K", `casynth_engines/gutter_field_periodic.py`):
  the N1 Gutter model unchanged (source, routes, states, controls, snapshot)
  with the weighted count n_i = sum(K_i G) instead of the 2 x 4 regions and a
  fixed output trim of -16.5 dB after the x20.
- **B `ca_event_network`** ("Events", `casynth_engines/event_network.py`): a NEW model
  (not a Gutter port).  At every block boundary the cells that changed since
  the previous render, weighted by K_i and saturated (a = e / (e + 2)), strike
  a fast / slow pulse pair per node (0.25 / 2 ms, strength 0.75) that feeds
  eight delay lines D = 149..1361 samples with an orthogonal feedback matrix
  (0.25 - delta), a 6 kHz loss filter and tanh; rho ("Response" 0.60..0.95,
  20 ms ramp) scales the feedback.  Output from the loss filters with cos / sin
  panning, x96, then the bench gain.  No direct path: without events the
  output is exactly zero; after events the network decays.  Painting, CA steps
  and the initial field (once, at init / reset) are events; parameters and
  restore are not.  Only SR 44100 / block 352 / stereo.  Snapshot = delays and
  positions, filters, pulse states, previous / pending field, ramp, gain ramp,
  versions.  Panel: the last packet a_i per node (bars), the response level
  (tick), rho; overlay: node centres + half-weight ring.

Entry: **`run_network_n2.bat`** opens the bench on the CATALOG screen of
`lab_catalog/network_n2_events_2026_09_16/` (build once with
`python demos/build_n2_events.py`; refuses a catalog that holds records).
Three records, 32 x 32 torus, B3/S23, 12 s from a fresh start, evolution on,
the REQ's hypotheses in Notes: N2.1 pulsar 6 gen/s (`n2_rhythm`), N2.2 glider
16 gen/s (`n2_travel`), N2.3 R-pentomino 6 gen/s (`n2_growth`).  Gates:
`tests/test_n2_events.py` (1j; the B kernel equals an independent scalar
reference bit-exactly).  Measurements: `python demos/n2_events_report.py` ->
`demos/results/network_n2/report.{json,md}`.

## N3 -- the field tunes the resonances, its events strike them (2026-09-16)

REQ `memory/req-network-combined-n3-2026-09-16.md`: one NEW engine
`ca_tuned_events` ("Tuned", `casynth_engines/tuned_events.py`) that joins the N2
channels -- the events of the field strike, the live cells tune:

    cell changes -> pulses + the N2-B delay network -> eight tuned banks -> sound
    live cells   -> eight frequency multipliers -------------^

The network part is the N2-B sample math unchanged (pulse 0.25 / 2 ms x 0.75,
D = 149..1361, M = 0.25 - delta, 6 kHz loss, tanh) at rho = 0.88 FIXED: its
states equal `ca_event_network`'s for the same field history.  Each node's
loss-filter value drives a bank of 24 decaying complex resonators
`z' = r e^{i theta} z + l_i`, `b_i = mean Re z'`, `r = 10^(-3 / (SR T60))`, at the
EXACT N1 frequencies `CONFIG['filters_hz']` (8 x 24, never regenerated) times the
node's multiplier `ratio_i = 2^(2 u_i - 1)` (u from the periodic readout, the
N1 / N2-A law with scale 1, depth 1; float32 messages at the block boundary,
states kept across a retune).  Output: N2 cos / sin panning of the banks, one
20 Hz DC filter per channel, x4, then the bench gain.  Nothing but the banks
is heard.  Controls: `Field tuning` (Fixed / Field: the comparison switch,
retunes without reset and without an event) and `Decay` (T60 0.20..1.50 s,
default 0.80; a manual change ramps r linearly over 20 ms).  Only SR 44100 /
block 352 / stereo.  Snapshot = the whole network, Re / Im of the banks, the
frequencies (checked against the field's law on restore), r and its ramp, both
DC filters, previous / pending field, gain ramp, versions.  Panel: the last
packet a_i per node (bars), the bank level (tick), the multiplier, the mode and
the decay; overlay: node centres + half-weight ring.

Entry: **`run_network_n3.bat`** opens the bench on the CATALOG screen of
`lab_catalog/network_n3_combined_2026_09_16/` (build once with
`python demos/build_n3_combined.py`; refuses a catalog that holds records).
Both sides run the same engine on the same field: A = hits on fixed resonances
(`field_tuning 0`), B = hits + field tuning (`field_tuning 1`); the REQ's
hypotheses are in Notes.  Records (32 x 32 torus, B3/S23, evolution on, cells
= the preflight fixtures `memory/research/network-n3-preflight-2026-09-16.json`):
N3.1 `n3_cycles` 24 s at 6 gen/s -- Octagon II (period 5) for 12 s, then ONE
journalled `set_cells` of the whole field at the first block boundary at or
after 12 s puts the Tumbler (period 14) in its place (audio states and the CA
clock kept), 12 s more; N3.2 `n3_travel` glider 16 gen/s, 12 s; N3.3
`n3_growth` R-pentomino 6 gen/s, 12 s.  Gates: `tests/test_n3_tuned_events.py`
(1k; the whole chain against an independent scalar reference within 1e-12,
both modes, through a decay ramp).  Measurements:
`python demos/n3_combined_report.py` -> `demos/results/network_n3/report.{json,md}`
(scenes against the preflight, swap, tails, continuation, late edits, the two
control probes, stress probes at gain 0.04, timing, catalog check; the summary
keeps scenes and stress probes apart and lists the limitations).

## N4 -- every figure a resonator bank of its own Laplacian (2026-09-16)

REQ `memory/req-object-resonators-n4-2026-09-16.md`: one NEW engine
`ca_object_resonators` ("Objects", `casynth_engines/object_resonators.py`, geometry
in `casynth_engines/figures.py`).  Every 8-connected figure of the field (across the
torus seam) owns a bank of up to 24 decaying complex resonators at
`f_j = frequency_scale * sqrt(lambda_j)` of its own Laplacian `L = D - A` (full
8-connectivity graph, zero mode dropped, multiplicities kept, no crop / decimation,
matrix built from the canonical placement so the same shape anywhere gives the
same eigenvalues bit for bit).  At a block boundary where the field changed:
components are matched to the tracked figures (overlap graph; split / merge retire
the old ones to tails and give new identities; zero-overlap continuation by a
centre displacement <= 1.5 cells), then
`e = sum(births * K_cur) + sum(deaths * K_prev)` (new figure: births in K_cur),
`a = e / (e + 2)` into the N2 pulse pair of the figure's slot.  `K` = the detector:
Disk (default) = every cell within `R = max torus distance centre -> cell` of the
periodic centre of mass (boundary included, no hidden margin, one cell -> R 0), Own =
the figure's cells; circles are not normalised against each other.  Mode j -> mode
j on retune, new modes from zero, vanished modes ring on undriven, weights 1 / n
ramp 20 ms; panning cos / sin of `cx / (cols - 1)` (ramp 20 ms); sum of all slots ->
HP 20 Hz -> x0.5 -> bench gain (ramp 20 ms); no AGC, no division by the count.
Slots: 24 banks + 96 tails + 24 fading (quiet tails released, the quietest faded
when full, hard drops counted); "sounding X of Y" in the panel.  Controls:
`Detector` Own / Disk (masks of the next events, no packet), `Freq scale` 55..880
(retunes every slot, states kept), `Decay` 0.20..1.50 s (r ramp 20 ms).  Only SR
44100 / block 352 / stereo.  Snapshot = every slot array + the tracker (ids, slots,
cells, centres, radii) + both fields + ramps + DC filters + gain; Continue byte-exact.
Bench: the LISTENED side's figures come from the engine's `display()` (the same
geometry the audio uses): cells, circle (continued across the seam) and centre in
one stable colour per id; Own = outlined cells + a dotted reference circle; a figure
panel (id, cells, modes, lowest Hz, packet bar, level tick).  The static overlays of
the other engines are unchanged.  **Radius x** (2026-09-16, user's addition): the
Disk detector uses `R_eff = radius_mul * R` (0.25..4.0, default 1.0 = bit-exact); the
geometric R is untouched, a single cell keeps R 0, Own ignores it, a change acts on
the masks of the next events only (no packet); the circle on screen and the panel
show the effective radius; older snapshots without the parameter restore with 1.0.

Entry: **`run_object_resonators_n4.bat`** opens the bench on the CATALOG screen of
`lab_catalog/object_resonators_n4_2026_09_16/` (build once with
`python demos/build_n4_objects.py`; refuses a catalog that holds records).  Records
(32 x 32 torus, B3/S23, 6 gen/s, 12 s, cells = the preflight
`memory/research/object-resonators-n4-preflight-2026-09-16.json`): N4.1
`n4_spectrum` glider from (28, 28) across the seam, A = N3 `ca_tuned_events` (Field,
decay 0.8), B = N4 Disk; N4.2 `n4_neighbor` a still 17-cell figure (centre (11.7647,
11.7647), R 5.32962) + a blinker inside its circle, A = N4 Own, B = N4 Disk.  Gates:
`tests/test_n4_object_resonators.py` (1l; the sample path against an independent
scalar reference within 1e-12 through every ramp).  Measurements:
`python demos/n4_objects_report.py` -> `demos/results/object_resonators_n4/report.{json,md}`
(scenes against the preflight, the receiver's packets per generation, Disk - Own,
glider identity / phases, tails, continuation, late edits, stress probes at gain
0.04, the analysis cost by figure size, timing, catalog check; the summary keeps
scenes and stress apart and lists the limitations).

## Objects / Laplace -- the old Laplace against Objects with its settings (2026-09-17)

REQ `memory/req-objects-laplace-comparison-2026-09-17.md`.  The Objects engine
(`ca_object_resonators`, now `model_version ca_object_resonators_n4_v2`,
`STATE_VERSION 2`) gets a **Spectrum** switch, Figure / Laplace (registry default
Laplace; a parameter set or snapshot without the key means Figure), and the seven
settings of the old Laplace with the metadata of the core registry entry itself
(`n` / part, `spread`, `alpha`, `shape`, `harm`, `fullshape` / full, `dyn`; same
ranges, defaults and names).  Figure = the N4 spectral path as before
(`frequency_scale * sqrt(lambda)`, weights 1 / n).  Laplace = the mathematics of
`casynth_core.map_laplacian` on the figure's OWN component: `casynth_core.laplacian_modes`
(the spectral part of map_laplacian factored out on 2026-09-17; map_laplacian calls it
and stays bit-for-bit) on the full torus graph of the component for `full` = 1 (no crop,
no decimation -- the old node ceiling is not carried over), or map_laplacian on the 8 x 8
extract window for `full` = 0; the lowest selected mode is `ctx.f0` of the scene; mode
selection, guard, harmonic pull and the weight law are the old ones; `dyn` samples the
bench `exc` (the standard events_field) at the figure's cells in the graph's node order
(a weight, never a strike).  The bank is `sum(w_j Re z_j)` with `w_j = LAPLACE_GAIN (0.7)
* amplitude_j` (a constant calibration measured on the three scenes, no division by the
mode / figure count); `Freq scale` is inactive in this mode, the seven are inactive in
Figure, `dyn` is inactive while `shape` = 0 (a property of the formula).  A change of
the spectrum law, of the seven or a new `exc` on the same field (only when dyn acts)
retunes every sounding figure at once (states kept, weights ramp 20 ms) and never
strikes.  On one compact component away from the seam the frequencies and the weights /
LAPLACE_GAIN equal map_laplacian on the object's bounding box for the same seven
settings, f0 and excitation (gate: bit-exact over the 73 states of the three scenes).

**Tail rules of v2 (the three P2 of the Researcher review 2026-09-16):** modes that
vanish from a continuing figure leave its bank into a tail slot of their own (states,
frequencies, weights, panning copied; undriven), so a returning mode starts from zero
and an old tail never gets a new pulse; when every tail AND fading slot is busy nothing
is cut any more: a leaving figure fades out in place in its active slot over 20 ms
(counted "in place"; the slot is busy for those 20 ms), vanished modes fade in place
inside their bank; the only hard cut left (counted "dropped") is such in-place fading
modes when the figure grows back within those 20 ms with every pool still busy.  The N4
report now names the tails probe's peak and clip in its summary.

Bench: the panel holds the 12 settings + the figure rows (the window grows to fit);
**`copy_spectrum`** command / buttons "<< spectrum B to A" / "spectrum A to B >>" copy
only the spectrum settings both engines offer (`registry.spectrum_keys`; the pair
Laplace / Objects shares all seven, harmonic engines share `n`, nothing shared ->
rejected with a message), engines, Detector, Radius x and Decay stay; the full side copy
`<<` / `>>` is unchanged.  Scene `audio.side_gain` `{A, B}` (optional, default 1.0 =
bit-exact for older scenes / records): one constant factor on the side's pre-clip gain,
part of the record / snapshot, applied by Replay / Continue.

Entry: **`run_objects_laplace.bat`** opens the bench on the CATALOG screen of
`lab_catalog/objects_laplace_2026_09_17/` (build once with
`python demos/build_objects_laplace.py`; refuses a catalog that holds records).  All
three records: one field, 32 x 32 torus, B3/S23, 6 gen/s, 12 s, f0 110 Hz, cells = the
preflight `memory/research/object-resonators-laplace-preflight-2026-09-17.json`; A =
`laplacian` with the REQ table (n 12, spread 0, alpha 1, shape 0, harm 0, full 1, dyn 0),
B = Objects / Laplace with the same seven, Disk, Decay 0.8: L1 `ol_glider` (glider inside
the field, Radius x 1), L2 `ol_galaxy` (Kok's galaxy at (11, 11), Radius x 1), L3
`ol_neighbor` (the N4.2 receiver + a blinker three columns right, Radius x 1.5).  Side
gains B 1.08 / 0.74 / 1.25 bring the integral RMS of A and B within 0.03 dB.  The N4
catalog, its two scenes (Figure law) and Notes stay; its records (v1 engine) were Local
(built while the Radius x change was uncommitted) and are now pinned to the commit of
their sound files, so the bench runs them in a separate bench of that version ("Continue
in version 6aee6a8"; the version's own check: match).  The current code refuses their
scene documents on purpose: no parameter value is ever invented for an old record.
**Rule (2026-09-17): commit the sound code BEFORE building a catalog** -- the builders
refuse a dirty sound set (`require_pinnable()`), records must be pinned.  Gates: `tests/test_objects_laplace.py` (1m).
Measurements: `python demos/objects_laplace_report.py` ->
`demos/results/objects_laplace/report.{json,md}` (spectral equality, the seven settings on
the galaxy, levels / continuation / timing of the scenes, no packet from settings or
Restore, L3 events at x1 / x1.5 and the live lever, L2 voices and the mode-return rule,
stress probes and the cost of the Laplace law by size, catalog check).

## Objects -- event source and modal excitation (2026-09-17)

REQ `memory/req-objects-event-source-modal-2026-09-17.md`.  The Objects engine (v4,
`ca_object_resonators_n4_v4`, snapshot 4; v3 / v2 snapshots restore as Both / Uniform)
has two more word-button settings: **Events** Both / Births / Deaths selects the source of
new packets (Both = the previous path bit for bit; Births `e = |births & K_cur|`; Deaths
`e = |deaths & K_prev|`, a new bank gets no packet, a bank that loses its identity takes ONE
last packet of the deaths inside its previous mask into its tail slot -- the tail feeds its
modes until that pulse is below 1e-7, never again); **Excitation** Uniform / Birth position
distributes one packet between the driven modes: Uniform adds `a` to every mode (the previous
path), Birth position is defined for Births + Own + Laplace + full and gives mode j
`delta_j = a b_j`, `b_j = sqrt(m p_j / sum p)`, `p_j` = the born cells' summed squared
eigenvector entries averaged over the degenerate group of the mode (sign- and basis-invariant);
`sum b_j^2 = m` like Uniform.  The packet enters the pulse states OF THE MODE (per-mode pulse /
Attack states `zfm / zsm / zum`), never the output weights: a strike never recolours a ringing
tail.  Outside its combination Birth position makes NO packet (the panel says so in red,
`unsupported_packets` counts; never a uniform strike instead).  Switching either setting never
makes a packet.

Entry: **`run_objects_event_source_modal.bat`** opens the bench on the CATALOG screen of
`lab_catalog/objects_event_source_modal_2026_09_17/` (build once with
`python demos/build_objects_event_source.py`; refuses a catalog that holds records or an
uncommitted sound set).  Both records: 32 x 32 torus, B3/S23, 6 gen/s, 12 s, f0 110 Hz, cells =
the preflight `memory/research/objects-event-source-modal-preflight-2026-09-17.json`, both
sides Objects / Laplace with the preflight settings (Own, full, part 3, spread 1, harm 0.87,
Decay 1.39, Attack 4 ms, shape / alpha / dyn 0): **E1** `oes_e1` Octagon II p5, A Events Births
/ B Events Deaths (Uniform both); **M1** `oes_m1` Jam p3, Births both, A Uniform / B Birth
position.  Side gain B 0.93 on both: the integral RMS after the first 2 s of A and B within
0.1 dB (the start fills the field and is measured apart).  Gates: `tests/test_objects_event_source.py`
(1o).  Measurements: `python demos/objects_event_source_report.py` ->
`demos/results/objects_event_source/report.{json,md}` (E1 packets against independent masks,
manual events and the vanished figure's tail, M1 side-by-side equality and the preflight
coefficients, the single-birth transfer / zero participation / degenerate groups / translation /
rotation, superposition and Attack on per-mode packets, levels / continuation / timing of the
scenes, catalog check).  The bench panel: `ROW_H` 20 (16 rows), the 'Birth position' button is
92 px wide, a third header line of the Objects display shows events / excitation (`FIGURE_HEAD_H`
48).

### Birth strength (REQ section 5, 2026-09-18)

Objects v5 (`ca_object_resonators_n4_v5`, `STATE_VERSION` 5; a v4 snapshot imports as Birth
strength 1, bit for bit the v4 sound): the knob **Birth strength** (`birth_strength`, 0..4,
default 1, absent = 1) reshapes the distribution of the NEXT Birth position packets only: s = 0
takes the Uniform path itself (no participation needed, the combination is still required),
0 < s < 1 blends `(1 - s) + s b` and 1 < s <= 4 raises `b ** s`, both renormalised to
`sum c^2 = m`; s = 1 is b itself (no arithmetic).  Zero participation at s > 0 still makes no
packet; an exact zero stays zero for s >= 1.  A change of s never makes a packet and never
touches the pulse states of either path.  With Uniform the knob is stored and shown inactive
(`"1.00  Birth position only"`); the fourth header line of the Objects display gives its
meaning (`FIGURE_HEAD_H` 62; `ROW_H` 18 so the 17 rows fit a 960 px desktop).

Entry: **`run_objects_birth_strength.bat`** opens the bench on the CATALOG screen of
`lab_catalog/objects_birth_strength_2026_09_18/` (build once with
`python demos/build_objects_birth_strength.py`).  **M2** `obs_m2`: the M1 field (Jam p3), both
sides Births + Birth position with the M1 settings, A Birth strength 1 / B 4; side gain B 0.92
(A - B after 2 s at unit gains -0.73 dB); the knob moves freely in the live window (a manual
change may change the loudness: the factor is constant).  Gates:
`tests/test_objects_birth_strength.py` (1p).  Measurements:
`python demos/objects_birth_strength_report.py` -> `demos/results/objects_birth_strength/report.{json,md}`
(the law against the preflight, five engines side by side on M1, the PCM path: s = 0 == Uniform,
the pinned 2026-09-17 records against the current code, still-field changes, a knob sequence,
Continue, v4 import; levels at the five s, timing with the knob moved, catalog check).  Every
older Objects scene (`n4_*`, `ol_*`, `ora_*`, `oes_*`) carries `birth_strength: 1.0` (same sound;
their records stay pinned to their own commits).

## Objects -- decay from the history of the cells, D1-D3 (2026-09-18)

REQ `memory/req-objects-decay-2026-09-18.md`, preflight
`memory/research/objects-decay-preflight-2026-09-18.json`.  Objects v6
(`ca_object_resonators_n4_v6`, `STATE_VERSION` 6; a v5 snapshot restores as Fixed with u = 1
on its live cells -- the same sound): the knob **Decay law** (`decay_law`, buttons Fixed /
Common / Modal = Fixed / Common age / Modal age, default Fixed, absent = Fixed) right after
Decay.

- **Fixed** -- the previous path bit for bit (one global r with its 20 ms ramp).
- **History** -- every cell position holds its freshness u (born 1, dead 0, survivors keep
  theirs, `u *= exp(-B / (SR 0.5 s))` after every rendered block, also while the CA is paused).
- **Common age / Modal age** (only with Spectrum Laplace + full; the combination is refused
  by the registry, the runner at post time, the scene loader and the snapshot, and the panel
  locks it both ways): per selected Laplace mode j of a figure `q_j = sum_i P_ji u_i` (P = the
  squared eigenvector entries averaged over the degenerate group), `T_j = Decay - (Decay -
  0.08) q_j`, `gamma_j = ln(1000) / T_j`; Common gives every mode of the bank the MEAN RATE,
  Modal each its own.  The applied gamma follows the target with a 20 ms one-pole per
  sample and `r_j = exp(-gamma / SR)` replaces r for that mode -- on the whole ringing state,
  never a strike, never a weight.  Decay is then the STABLE (upper) T60; the figure rows show
  the current target T range (a target, not a measured length).  Tails keep their applied /
  target gamma and hold the last target; a switch of the law or of Decay never strikes.
- **Scene `script`** (new optional scene key): commands the runner queues at every start of
  the scene from its beginning (Start, the pause of a stopped scene released, Restart) --
  used for the two CA pauses of D2.  They stand in the journal marked `script` and the
  replay skips them (the replayed start queues them again); Continue never repeats past ones.

Entry: **`run_objects_decay.bat`** opens the bench on the CATALOG screen of
`lab_catalog/objects_decay_2026_09_18/` (build once with `python demos/build_objects_decay.py`).
Both sides: Own, Laplace, full 1, part 3, spread 1, harm 0.87, shape / alpha / dyn 0, Attack
4 ms, Events Births, Excitation Uniform, f0 110 Hz, the preflight cells.

| scene | field | rate, length | A | B | side gain B |
|---|---|---|---|---|---|
| `od_d1` | Octagon II p5 | 2 gen/s, 20 s | Fixed, Decay 0.947926 s | Common age, 1.39 s | 1.09 |
| `od_d2` | Blinker p2, two scripted pauses | 2 gen/s, 16 s | Common age, 1.39 s | Modal age, 1.39 s | 0.82 |
| `od_d3` | Jam p3 + Octagon II p5 | 6 gen/s, 18 s | Fixed, 1.39 s | Modal age, 1.39 s | 1.21 |

Side gains: A - B of the REQ windows at unit gains (D1 5..20 s, D2 the active parts after 2 s
without the pauses, D3 3..18 s).  Gates: `tests/test_objects_decay.py` (1q).  Measurements:
`python demos/objects_decay_report.py` -> `demos/results/objects_decay/report.{json,md}`.

### Objects decay control D4 (REQ `memory/req-objects-decay-control-2026-09-18.md`)

Entry: **`run_objects_decay_control.bat`** opens the bench on the CATALOG screen of
`lab_catalog/objects_decay_control_2026_09_18/` (build once with
`python demos/build_objects_decay_control.py`).  No new code in the engine: the scenes are
derived from the scene embedded in the D3 record (field = the preflight cells, the same 29);
a condition changes only the Decay law, Decay and the constant side gain of the D3 side it
starts from.  18 s at 6 gen/s (794112 samples, the length of D3), no scripted intervention.

| condition | from D3 side | Decay law | Decay | side gain |
|---|---|---|---|---|
| matched Fixed | A | Fixed | 0.671529 s | 1.220467 |
| previous Modal | B | Modal age | 1.39 s | 1.21 |
| previous long Fixed | A | Fixed | 1.39 s | 1.0 |

| scene | A | B |
|---|---|---|
| `od_d4_control` (D4.1) | matched Fixed | previous Modal (= B of D3 bit for bit) |
| `od_d4_anchor` (D4.2) | matched Fixed (= A of D4.1) | previous long Fixed (= A of D3) |

0.671529 s = ln 1000 / the mean APPLIED gamma of every driven mode of the active banks of
the D3 Modal side over the samples [132300, 793800), every mode-sample weighted equally.
Side gains: one constant per condition from the RMS of both channels over 3..18 s
(`--calibrate` measures them at unit gains; the matched Fixed is levelled to the long
Fixed).  Measurements: `python demos/objects_decay_control_report.py` ->
`demos/results/objects_decay_control/report.{json,md}`.

## Adding a sound engine (S3 interface)

The bench owns the field, clocks, command queue/journal, transport, the A/B
instances, the monitor and WAV export.  An engine is one class per side that
turns the field into stereo blocks.

1. **Module** — put it under `casynth_engines/` (e.g.
   `casynth_engines/my_engine.py`) and subclass `casynth_engines.SoundEngine`
   (`casynth_lab.SoundEngine` is the same class, re-exported):
   ```python
   from casynth_engines import SoundEngine
   class MyEngine(SoundEngine):
       SUPPORTS_TRANSPOSE = False                   # True = render() plays a note
       def init(self, grid, exc, gain=0.0): ...    # (re)start on the CURRENT field, silent
       def update_field(self, grid, exc): ...      # field changed (step / painting)
       def set_params(self, params): ...           # full validated dict (call super())
       def render(self, gain, t_samples, *, gain_prev=None, transpose=1.0): ...
                                                   # -> (int16 (ctx.block, 2), peak, n_clip)
       def reset(self, gain=0.0): ...              # == init on the same field
       def display(self): ...                      # optional read-only numbers for a UI
       def set_envelope(self, a, d, s, r, amp_slew): ...   # optional: live host envelope
       def set_rate(self, rate_hz): ...            # optional: live host tempo
       STATE_VERSION = 1                            # optional (S5): exact continuation
       def export_state(self): ...                 # -> dict of scalars/lists + ndarrays
       def restore_state(self, grid, exc, state): ...  # rebuild so render() continues exactly
   ```
   `self.ctx` gives `sr`, `block`, `channels`, `f0`, `level`, `rate_hz`.  Apply
   `gain` once (pre-clip) and GLIDE to it from the previous block's gain;
   `gain_prev` is the host overriding what you remembered.  `ctx.f0` is the
   ANALYSIS ANCHOR, never the sounding note -- the note arrives per block as
   `transpose`, a multiplier on every frequency you render (phase keeps
   accumulating, so nothing retriggers).  An engine that cannot do that leaves
   `SUPPORTS_TRANSPOSE` False and the base class refuses the note instead of
   playing a wrong one.  The VCA, the note gate and the level meter's decay are
   the host's, not yours.  No wall-clock time, never write to `grid`.  The full
   contract is the module doc of `casynth_engines/engine_api.py`.
2. **Register** — explicitly, in one place: the `_LAZY` map at the bottom of
   `casynth_engines/registry.py` (id -> your module, after the built-in five).
   The module is imported by the first `registry.get(id)`, so listing it costs
   a host nothing until it plays that engine.  In the module itself:
   ```python
   from casynth_engines import EngineSpec, register
   register(EngineSpec('my_engine', 'My', [('depth', 'depth', 0.0, 1.0, False, 0.5)],
                       lambda ctx, params: MyEngine(ctx, params),
                       plays_notes=True))        # only if SUPPORTS_TRANSPOSE is True
   ```
   Param spec = `(arg, label, lo, hi, integer, default)`; integer 0/1 renders as
   a toggle.  `plays_notes` says the engine renders the contract's `transpose`,
   so a host with a keyboard (gol_synth) may offer it; a gate checks it against
   the class's `SUPPORTS_TRANSPOSE`, and it lives on the spec so a host can ask
   without building one of every engine first.  Optional display hints (no effect on sound): `choices={'path':
   ('Ellipse', 'Lissajous', 'Raster')}` renders an integer parameter as word
   buttons; `inactive(params) -> {name: text}` shows a parameter as text when
   it does not act; `overlay(params, rows, cols) -> dict` (`polyline` of
   unwrapped cell coordinates, `start`/`ahead`, `circles`, `labels`, `text`) is
   drawn over the field for the listened side; an engine method `display()`
   returning plain numbers reaches the UI through the runner snapshot.  The UI buttons/knobs, scene validation, commands and `--side`
   export all come from the registry -- no other Python changes.
3. **Scene** — reference it from JSON only:
   `"variants": {"B": {"engine_id": "my_engine", "engine_params": {"depth": 0.5}}}`.
4. **Check** — `python tests/test_demo_lab.py` (interface + audio references) and
   `python check.py`.  The bench rejects malformed blocks (wrong shape/dtype):
   they are replaced by silence and counted in the audio status.
