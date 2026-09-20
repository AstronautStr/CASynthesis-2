@echo off
rem CASynth demo bench -- Laplace carriers on one Jam evolution (REQ memory/req-laplace-carriers-2026-09-20.md):
rem 1-4 each new law against the existing Laplacian on sines (Filter / Wave bank, Saw / Square),
rem 5-6 the two laws against each other on the same tracks.
rem Opens the bench on the CATALOG screen of the prepared records (build them once with
rem   python demos\build_laplace_carriers.py ); Continue = live field, Notes = hypothesis + your feedback.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\lc_saw_baseline_filter.json --catalog lab_catalog\laplace_carriers_2026_09_20 %*
if errorlevel 1 pause
