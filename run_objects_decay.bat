@echo off
rem CASynth demo bench -- Objects Decay law: D1 (Fixed / Common age, Octagon II p5), D2 (Common age / Modal age,
rem blinker with two CA pauses), D3 (Fixed / Modal age, Jam p3 + Octagon II p5); REQ memory/req-objects-decay-2026-09-18.md.
rem Opens the bench on the CATALOG screen of the prepared records (build them once with
rem   python demos\build_objects_decay.py ); Continue = live field, Notes = hypothesis + your feedback.
rem Decay law (Fixed / Common / Modal) and Decay are knobs of the Objects panel; under Common / Modal age
rem Decay is the stable (upper) T60 and each figure row shows its current target T range.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\od_d1.json --catalog lab_catalog\objects_decay_2026_09_18 %*
if errorlevel 1 pause
