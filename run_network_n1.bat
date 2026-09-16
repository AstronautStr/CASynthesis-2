@echo off
rem CASynth demo bench -- revised N1 hypotheses (2026-09-16).
rem Opens the bench on the CATALOG screen of the prepared N1 records (build them once with
rem   python demos\build_n1_hypotheses.py ); Continue = live field, Notes = hypothesis + feedback.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\n1h_rhythm.json --catalog lab_catalog\network_n1_hypotheses_2026_09_16_r2 %*
if errorlevel 1 pause
