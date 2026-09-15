@echo off
rem CASynth demo bench -- N1: the Gutter Synthesis network driven by the field (REQ 2026-09-15).
rem Opens the bench on the CATALOG screen of the prepared N1 records (build them once with
rem   python demos\build_n1_demos.py ); Continue = live field, Notes = feedback.  Python path fixed at setup time.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\network_n1_blinkers.json --catalog lab_catalog\network_n1_2026_09_15 %*
if errorlevel 1 pause
