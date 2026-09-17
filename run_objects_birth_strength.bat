@echo off
rem CASynth demo bench -- Objects Birth strength: M2 (Birth position, Birth strength 1 / 4, Jam p3),
rem Objects / Laplace on both sides, REQ 2026-09-17 section 5 (2026-09-18).
rem Opens the bench on the CATALOG screen of the prepared record (build it once with
rem   python demos\build_objects_birth_strength.py ); Continue = live field, Notes = hypothesis + your feedback.
rem Birth strength is the knob under Excitation on the Objects panel (0 = uniform strike, 1 = original,
rem >1 = stronger selection); it acts with Birth position only; a manual change may change the loudness.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\obs_m2.json --catalog lab_catalog\objects_birth_strength_2026_09_18 %*
if errorlevel 1 pause
