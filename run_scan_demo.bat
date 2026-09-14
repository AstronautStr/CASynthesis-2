@echo off
rem CASynth demo bench -- Scan (S) demo: F1 Smooth / Distance scene + the prepared
rem catalog lab_catalog\sn_demos_2026_09_14\scan (build it once with
rem   python demos\build_sn_demos.py ).  Python path fixed at setup time.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\sn_s_surface_f1.json --catalog lab_catalog\sn_demos_2026_09_14\scan %*
if errorlevel 1 pause
