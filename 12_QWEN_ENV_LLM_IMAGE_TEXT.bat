@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Hybrid text + optional image analysis. A vision-capable model is required.
echo Image will be sent to the chat/vision API configured in .env.
set /p "IMG=Image file path (leave blank for text only): "
if "%IMG%"=="" (
  ".venv-local\Scripts\python.exe" local_qwen.py text --llm-from-env
) else (
  ".venv-local\Scripts\python.exe" local_qwen.py text --llm-from-env --image "%IMG%"
)
pause
