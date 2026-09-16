@echo off
rem CASynth demo bench -- Objects / Laplace comparison: the existing Laplace synth (A) against the Objects engine in its Laplace spectrum mode (B), REQ 2026-09-17.
rem Opens the bench on the CATALOG screen of the prepared L1-L3 records (build them once with
rem   python demos\build_objects_laplace.py ); Continue = live field, Notes = hypothesis + your feedback.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\ol_glider.json --catalog lab_catalog\objects_laplace_2026_09_17 %*
if errorlevel 1 pause
