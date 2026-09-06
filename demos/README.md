# Demo bench (S1)

Run (Windows, project Python with deps installed): `run_demo_bench.bat`
or `python demo_bench.py --demo demos/laplace_basic.json`.

Offline render, no window / no audio device:
`python demo_bench.py --demo demos/laplace_basic.json --render artifacts/s1.wav --seconds 8`

Dependencies: `pip install -r requirements.txt` (numpy, scipy, pygame-ce, sounddevice).

Buttons: **Start** (field evolves + sound), **Pause CA** (only the automaton
freezes; painting and sound continue), **Restart** (initial scene, clocks and
all audio tails reset, scene starts again).  LMB paints, RMB erases.
Scene format v1: see `laplace_basic.json` (strictly validated: unknown engine /
parameter or bad cell -> error, no fallback).
