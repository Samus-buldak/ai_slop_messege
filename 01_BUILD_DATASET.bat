@echo off
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] Install the project venv first: py -3 -m venv .venv
  pause
  exit /b 1
)
"%PY%" protect_git.py || goto :failed
"%PY%" prepare_human.py --count 1000 || goto :failed
"%PY%" review_human.py --count 50 || goto :failed
"%PY%" generate_ai.py --count 1000 --lang en --batch 3 --parallel 3 || goto :failed
"%PY%" combine_dataset.py --count 1000 --individual || goto :failed
"%PY%" check_dataset.py --count 1000 || goto :failed
echo [OK] Created both CSVs, merged dataset.csv and 2000 individual files.
echo [NEXT] Run 02_TRAIN_MODEL.bat after manually reviewing human labels.
pause
exit /b 0
:failed
echo [ERROR] Step failed. Read messages above. Fix and rerun; generated AI rows are saved.
pause
exit /b 1
