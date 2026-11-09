@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Run LM Studio. Load TWO models: qwen3-embedding-local AND your chat Qwen model.
echo Find the exact chat model identifier in LM Studio's Models screen.
set /p LOCAL_LLM_MODEL="Chat model identifier (e.g. qwen3-4b-instruct): "
if "%LOCAL_LLM_MODEL%"=="" (
  echo No chat model ID supplied. Training cancelled.
  pause
  exit /b 1
)
".venv-local\Scripts\python.exe" local_qwen.py train --with-llm
pause
