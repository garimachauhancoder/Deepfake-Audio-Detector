@echo off
title Identity Shield — Deepfake Audio Forensic Console
echo ======================================================================
echo  IDENTITY SHIELD: MULTI-BRANCH SPEECH DEEPFAKE DETECTION SYSTEM
echo  MCA Dissertation Project — The NorthCap University, Gurugram
echo  Candidate: Garima Chauhan (25MCA020)
echo ======================================================================
echo.

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\activate.bat!
    echo Please run setup first or ensure python 3.12+ is installed.
    pause
    exit /b 1
)

echo [INFO] Activating Python virtual environment (.venv)...
call .venv\Scripts\activate.bat

echo [INFO] Launching Streamlit Forensic Dashboard...
echo [INFO] Local URL will open at: http://localhost:8501
echo.
streamlit run app.py

pause

