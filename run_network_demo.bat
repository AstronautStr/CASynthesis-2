@echo off
rem CASynth demo bench -- Network (N) demo: F5 top independent / connected scene + the
rem prepared catalog lab_catalog\sn_demos_2026_09_14\network (build it once with
rem   python demos\build_sn_demos.py ).  Python path fixed at setup time.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\sn_n_links_top.json --catalog lab_catalog\sn_demos_2026_09_14\network %*
if errorlevel 1 pause
