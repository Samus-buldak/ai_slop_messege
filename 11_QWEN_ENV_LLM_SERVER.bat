@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Hybrid server: Qwen embeddings local, original .env chat LLM over API.
echo WARNING: non-empty email segments sent for analysis may leave this computer.
".venv-local\Scripts\python.exe" local_qwen.py server --llm-from-env
pause
