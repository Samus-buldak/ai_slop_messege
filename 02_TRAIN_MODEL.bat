@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (echo [ERROR] Missing .venv\Scripts\python.exe& pause& exit /b 1)
"%PY%" check_dataset.py --count 1000 || goto :failed
"%PY%" train.py --csv data\dataset.csv || goto :failed
echo [OK] Models stored in model\. Inspect model\test_metrics.json
pause
exit /b 0
:failed
echo [ERROR] Check output above.
pause
exit /b 1
