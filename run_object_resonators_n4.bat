@echo off
rem CASynth demo bench -- N4: every figure a resonator bank of its own Laplacian, struck inside its circle (REQ 2026-09-16).
rem Opens the bench on the CATALOG screen of the prepared N4 records (build them once with
rem   python demos\build_n4_objects.py ); Continue = live field, Notes = hypothesis + your feedback.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\n4_spectrum.json --catalog lab_catalog\object_resonators_n4_2026_09_16 %*
if errorlevel 1 pause
