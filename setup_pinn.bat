@echo off
setlocal
cd /d "%~dp0"
set "UV_CACHE_DIR=%CD%\.uv-cache"
echo Creating or updating the local Python environment...
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r pinn\requirements.txt
echo.
echo Done. You can now double-click run_synthetic_tests.bat or run_real_data_test.bat.
pause
