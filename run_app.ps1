# Identity Shield — Deepfake Audio Forensic Console Launcher (PowerShell)
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " IDENTITY SHIELD: MULTI-BRANCH SPEECH DEEPFAKE DETECTION SYSTEM" -ForegroundColor White
Write-Host " MCA Dissertation Project — The NorthCap University, Gurugram" -ForegroundColor Yellow
Write-Host " Candidate: Garima Chauhan (25MCA020)" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[ERROR] Virtual environment not found at .venv\Scripts\python.exe" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[INFO] Starting Streamlit Forensic Console..." -ForegroundColor Green
& ".venv\Scripts\python.exe" -m streamlit run app.py

