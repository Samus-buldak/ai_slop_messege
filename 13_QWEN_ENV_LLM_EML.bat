@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Analyze .eml (attachments are detected automatically).
echo Text and images may be sent to the external .env LLM.
if "%~1"=="" (
  set /p "MAIL=Path to .eml file: "
) else (
  set "MAIL=%~1"
)
".venv-local\Scripts\python.exe" local_qwen.py analyze --llm-from-env "%MAIL%"
pause
