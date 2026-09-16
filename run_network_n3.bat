@echo off
rem CASynth demo bench -- N3: the field tunes the resonances, its events strike them (REQ 2026-09-16).
rem Opens the bench on the CATALOG screen of the prepared N3 records (build them once with
rem   python demos\build_n3_combined.py ); Continue = live field, Notes = hypothesis + your feedback.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\n3_cycles.json --catalog lab_catalog\network_n3_combined_2026_09_16 %*
if errorlevel 1 pause
