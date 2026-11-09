@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Optional TF-IDF ablation/baseline (Qwen embeddings are still enabled).
".venv-local\Scripts\python.exe" local_qwen.py train --use-tfidf
pause
