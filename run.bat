@echo off
title Product Studio — do not close this window
cd /d "%~dp0"

:: Keep all AI model files inside the project folder for easy cleanup
set HF_HOME=%~dp0models\cache
set HF_HUB_CACHE=%~dp0models\cache\hub
set XFORMERS_DISABLE=1

:: Activate venv
call venv\Scripts\activate.bat

:: Run environment check
echo.
echo ============================================
echo  Checking your system...
echo ============================================
python check_environment.py
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Some checks failed. Press Ctrl+C to abort,
    echo or wait 5 seconds to continue anyway...
    timeout /t 5 /nobreak >nul
)

:: Start Ollama if not already running
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    ollama ps >nul 2>&1
    if %errorlevel% neq 0 (
        echo Starting Ollama...
        start /b ollama serve
        timeout /t 3 /nobreak >nul
    )
)

echo.
echo ============================================
echo  Product Studio is starting...
echo  Open http://127.0.0.1:7860 in your browser
echo  when you see the message below.
echo ============================================
echo.

python app.py

echo.
echo App closed. You can close this window.
pause
