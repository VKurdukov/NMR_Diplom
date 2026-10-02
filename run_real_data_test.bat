@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment .venv was not found.
  echo Run setup_pinn.bat once, then start this file again.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pinn.benchmark_real
pause

