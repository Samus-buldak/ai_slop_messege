@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
".venv-local\Scripts\python.exe" local_qwen.py train
pause
