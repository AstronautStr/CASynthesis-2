@echo off
rem CASynth demo bench -- N2: field events excite a delay network vs the field retuning Gutter (REQ 2026-09-16).
rem Opens the bench on the CATALOG screen of the prepared N2 records (build them once with
rem   python demos\build_n2_events.py ); Continue = live field, Notes = hypothesis + your feedback.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\n2_rhythm.json --catalog lab_catalog\network_n2_events_2026_09_16 %*
if errorlevel 1 pause
