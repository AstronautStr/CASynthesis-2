@echo off
rem CASynth demo bench -- Laplace FM on one Jam evolution (REQ memory/req-laplace-fm-2026-09-20.md):
rem 1 the new FM against the existing Laplacian on sines, 2 the already listened Laplacian / Wave bank
rem (copied unchanged, no need to listen again), 3 that Wave bank against the same FM.
rem Opens the bench on the CATALOG screen of the prepared records (build them once with
rem   python demos\build_laplace_fm.py ); Continue = live field, Notes = hypothesis + your feedback.
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\lfm_baseline.json --catalog lab_catalog\laplace_fm_2026_09_20 %*
if errorlevel 1 pause
