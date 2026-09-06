@echo off
rem CASynth demo bench (S1). Python path fixed at setup time (project interpreter
rem with numpy/scipy/pygame-ce/sounddevice already installed).
set PY=C:\Users\Astro\AppData\Local\Python\pythoncore-3.14-64\python.exe
cd /d "%~dp0"
"%PY%" demo_bench.py --demo demos\laplace_basic.json %*
if errorlevel 1 pause
