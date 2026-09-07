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

A/B: one shared field, two sound sides computed all the time on the same
timeline.  The **A** / **B** tabs pick the side you listen to AND edit
(20 ms crossfade on switch).  The panel shows the selected side's engine
(Laplace / FFT / Walsh / Random / Granulo from `casynth_core.ENGINES`), its
knobs with the registry ranges, and a one-line summary of how A and B differ.
Settings are remembered per side x engine.

Scenes: `laplace_basic.json` = format 1 (one engine, loaded as A = B);
`laplace_ab.json` = format 2 (`variants` A/B + `listen` hint).  Strictly
validated: unknown engine / parameter, out-of-range value or bad cell -> error.
