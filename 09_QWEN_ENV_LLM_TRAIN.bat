@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
echo Hybrid training: embeddings via LOCAL Qwen in LM Studio, chat scoring via ORIGINAL .env API.
echo WARNING: email segments from stack/calib/test will be sent to the configured external chat API.
".venv-local\Scripts\python.exe" local_qwen.py train --llm-from-env --with-llm --llm-segmentation
pause
