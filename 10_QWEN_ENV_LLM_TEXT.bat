@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Hybrid inference: Qwen embeddings local, original .env chat LLM over API.
echo WARNING: non-empty email segments will be sent to the original .env chat provider.
".venv-local\Scripts\python.exe" local_qwen.py text --llm-from-env %*
pause
