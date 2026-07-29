@echo off
title Product Studio — Setup
echo ============================================
echo  Product Studio — One-time Setup
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Download Python 3.11+ from https://python.org
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

:: Create virtual environment
echo [1/5] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate
call venv\Scripts\activate.bat

:: Keep model files inside project folder for easy cleanup
set HF_HOME=%~dp0models\cache
set HF_HUB_CACHE=%~dp0models\cache\hub

:: Install Python packages
echo [2/5] Installing Python packages (this may take 5-10 minutes)...

:: PyTorch with CUDA (NVIDIA GPU required)
echo   Installing PyTorch with CUDA support...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if %errorlevel% neq 0 (
    echo [WARNING] CUDA PyTorch install failed, trying CPU fallback...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

:: Remaining dependencies
echo   Installing remaining packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Check your internet connection.
    pause
    exit /b 1
)

:: Install Ollama
echo [3/5] Installing Ollama...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo Downloading Ollama...
    curl -L https://ollama.com/download/OllamaSetup.exe -o ollama_setup.exe 2>nul
    if exist ollama_setup.exe (
        start /wait ollama_setup.exe /S
        del ollama_setup.exe
    ) else (
        echo [WARNING] Could not download Ollama automatically.
        echo Download manually from: https://ollama.com/download
        echo Then run this installer again.
    )
) else (
    echo [OK] Ollama already installed
)

:: Pull vision model
echo [4/5] Downloading vision AI model (~4GB, first run only)...
ollama pull qwen2.5-vl:7b
if %errorlevel% neq 0 (
    echo [WARNING] Failed to pull vision model.
    echo Make sure Ollama is running, then run: ollama pull qwen2.5-vl:7b
)

:: Done
echo [5/5] Setup complete!
echo.
echo ============================================
echo  You can now run Product Studio by
echo  double-clicking run.bat
echo ============================================
pause
