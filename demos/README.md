# Demo bench (S1 + S2)

Run (Windows, project Python with deps installed): `run_demo_bench.bat`
(opens `demos/laplace_ab.json`) or
`python demo_bench.py --demo demos/laplace_ab.json`.

Offline render, no window / no audio device:
`python demo_bench.py --demo demos/laplace_ab.json --render out.wav --seconds 8 --side A|B|monitor`
(`A`/`B` = the raw side, `monitor` = what is heard incl. the A/B crossfade; default A).

Dependencies: `pip install -r requirements.txt` (numpy, scipy, pygame-ce, sounddevice).

Transport: **Start** (field evolves + sound; while running the same button is
**Stop** = full stop, back to the initial silent scene), **Pause CA** (only the
automaton freezes; painting and sound continue), **Restart** (initial scene,
clocks and all audio tails reset, scene starts again).  Stop/Restart keep the
engines, parameters, volume and the selected side.  LMB paints, RMB erases.

Hotkeys: **R** = Restart, **Space** = Pause CA, **1** / **2** = select A / B.  `<<` / `>>` copy ALL settings (engine + params) B->A / A->B;
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
Select a record -> **Open in bench** loads its END state (field, engines and
parameters of A and B, selected side, volume) into a fresh live session that
you start manually; **Play A / Play B / Play as heard** (the live synth is
muted meanwhile); **Replay from start**
recomputes the experiment in a separate process from the embedded conditions
(from the recording's start; the saved window is compared) byte-exact with the
stored WAVs: "Replay matched" /
"Replay differs" / a reason why it is unavailable (e.g. the engine is not
registered any more -- the WAVs still play).  The replay output can be
listened to (**Play replay**); the originals are never modified.
Esc / Back returns to the live view.  Headless use:
`python -m casynth_lab.catalog replay lab_catalog/local <id>`.

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
