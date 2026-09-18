@echo off
rem CASynth demo bench -- Objects decay control D4 on the field and scene of D3 (Jam p3 + Octagon II p5):
rem D4.1 matched Fixed 0.671529 s / the previous Modal age, D4.2 the same matched Fixed / the previous long Fixed 1.39 s;
rem REQ memory/req-objects-decay-control-2026-09-18.md.
rem Opens the bench on the CATALOG screen of the prepared records (build them once with
rem   python demos\build_objects_decay_control.py ); Continue = live field, Notes = hypothesis + your feedback.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\od_d4_control.json --catalog lab_catalog\objects_decay_control_2026_09_18 %*
if errorlevel 1 pause
