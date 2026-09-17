@echo off
rem CASynth demo bench -- Objects radius / attack: R1 (Radius x 1 / 2), R2 (Radius x 1 / 32), A1 (Attack 0 / 4 ms)
rem and the user's Radius 0.34 preset, all Objects / Laplace on both sides, REQ 2026-09-17.
rem Opens the bench on the CATALOG screen of the prepared records (build them once with
rem   python demos\build_objects_radius_attack.py ); Continue = live field, Notes = hypothesis + your feedback.
rem Radius x: the Min / Max fields under the slider set its range for the selected side (Enter commits, Esc cancels).
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\ora_r1.json --catalog lab_catalog\objects_radius_attack_2026_09_17 %*
if errorlevel 1 pause
