@echo off
rem CASynth demo bench -- Objects event source / modal excitation: E1 (Events Births / Deaths, Octagon II p5)
rem and M1 (Excitation Uniform / Birth position, Jam p3), Objects / Laplace on both sides, REQ 2026-09-17.
rem Opens the bench on the CATALOG screen of the prepared records (build them once with
rem   python demos\build_objects_event_source.py ); Continue = live field, Notes = hypothesis + your feedback.
rem Events / Excitation are the word buttons of the Objects panel; Birth position needs Births / Own / Laplace / full.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\oes_e1.json --catalog lab_catalog\objects_event_source_modal_2026_09_17 %*
if errorlevel 1 pause
