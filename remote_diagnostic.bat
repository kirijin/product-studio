@echo off
title Product Studio — Remote Diagnostic
cd /d "%~dp0"
set REPORT=%~dp0diagnostic_report.txt

echo Product Studio — Remote Diagnostic > "%REPORT%"
echo Generated: %DATE% %TIME% >> "%REPORT%"
echo. >> "%REPORT%"

echo ============================================
echo  Product Studio — Diagnostic
echo  Saving report to: %REPORT%
echo ============================================
echo.

:: ── 1. Basic info ──────────────────────────────────────
echo [1/8] System info...
(
  echo === SYSTEM ===
  systeminfo | findstr /B /C:"OS Name" /C:"OS Version" /C:"System Type" /C:"Total Physical Memory" /C:"Available Physical Memory"
) >> "%REPORT%" 2>&1
echo CPU: >> "%REPORT%"
wmic cpu get name 2>>"%REPORT%" | findstr /V "Name" >> "%REPORT%" 2>&1
echo. >> "%REPORT%"

:: ── 2. GPU ──────────────────────────────────────────────
echo [2/8] GPU info...
(
  echo === GPU ===
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
  echo.
  nvidia-smi
) >> "%REPORT%" 2>&1
if %errorlevel% neq 0 (
  echo [WARN] nvidia-smi failed — CUDA/NVIDIA driver may not be installed >> "%REPORT%"
)

:: ── 3. Python — check multiple possible paths ────────────
echo [3/8] Python...
(
  echo === PYTHON ===
  where python 2>&1
  python --version 2>&1
  python -m pip --version 2>&1
) >> "%REPORT%" 2>&1

:: Try py launcher too
py --version >>"%REPORT%" 2>&1

:: ── 4. Python packages in venv (if exists) ───────────────
echo [4/8] Python packages...
if exist "venv\Scripts\python.exe" (
	(
	  echo === VENV PACKAGES ===
	  venv\Scripts\python.exe -m pip list --format=columns 2>&1
	) >> "%REPORT%"
	if %errorlevel% equ 0 (
		echo [OK] venv activated
	) else (
		echo venv\Scripts\python.exe exists but failed >> "%REPORT%"
	)
) else (
  echo venv not found — checking system Python packages >> "%REPORT%"
  (
    echo === SYSTEM PYTHON PACKAGES ===
    python -m pip list --format=columns 2>&1
  ) >> "%REPORT%"
)

:: Try critical imports
(
  echo === CRITICAL IMPORTS ===
  python -c "import PIL; print('PIL:', PIL.__version__)" 2>&1
  python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())" 2>&1
  python -c "import gradio; print('gradio:', gradio.__version__)" 2>&1
  python -c "import diffusers; print('diffusers:', diffusers.__version__)" 2>&1
  python -c "import rembg; print('rembg: OK')" 2>&1
  python -c "import cv2; print('cv2:', cv2.__version__)" 2>&1
) >> "%REPORT%" 2>&1

:: ── 5. Ollama ────────────────────────────────────────────
echo [5/8] Ollama...
(
  echo === OLLAMA ===
  where ollama 2>&1
  if %errorlevel% equ 0 (
    ollama --version 2>&1
    echo ---
    ollama list 2>&1
  ) else (
    echo Ollama NOT FOUND
  )
) >> "%REPORT%" 2>&1

:: ── 6. Files present ─────────────────────────────────────
echo [6/8] Project files...
(
  echo === PROJECT FILES ===
  dir /b /s *.py *.bat *.txt 2>&1
) >> "%REPORT%" 2>&1

:: ── 7. Disk + cache ──────────────────────────────────────
echo [7/8] Disk space + caches...
(
  echo === DISK ===
  wmic logicaldisk get caption,freespace,size 2>&1 | findstr /V "Caption"
  echo.
  echo === CACHE SIZES ===
  if exist "models\cache" (
    dir /s /b "models\cache" 2>nul | find /c /v "" >nul
    echo models\cache: present
  ) else (
    echo models\cache: not yet populated
  )
  if exist "%USERPROFILE%\.cache\huggingface" (
    dir /s "%USERPROFILE%\.cache\huggingface" 2>nul | find /c ":"
  ) else (
    echo HF cache: not found
  )
) >> "%REPORT%" 2>&1

:: ── 8. Pipeline dry-run ──────────────────────────────────
echo [8/8] Pipeline dry-run (syntax + imports)...
(
  echo === PIPELINE DRY-RUN ===
  python -c "
import sys, os
sys.path.insert(0, '.')
try:
    from pipeline import smart_placement, styles, bg_prompts
    print('pipeline modules: OK')
    print('Styles:', list(styles.LABEL_STYLES.keys()))
    print('Presets:', list(bg_prompts.BACKGROUND_PRESETS.keys()))
    print('Safe zone test:', 'OK' if hasattr(smart_placement, 'find_safe_zone') else 'MISSING')
except Exception as e:
    print(f'FAILED: {e}')

try:
    from pipeline import phase1_remove_bg, phase2_describe, phase3_render
    print('Phase functions defined:')
    print('  phase1:', callable(getattr(phase1_remove_bg, 'run', None)))
    print('  phase2:', callable(getattr(phase2_describe, 'run', None)))
    print('  phase3:', callable(getattr(phase3_render, 'run', None)))
except Exception as e:
    print(f'Phase import FAILED: {e}')

try:
    import check_environment
    print('check_environment: OK')
except Exception as e:
    print(f'check_environment FAILED: {e}')
" 2>&1
) >> "%REPORT%" 2>&1

:: ── Summary ──────────────────────────────────────────────
echo.
echo ============================================
echo  Done. Report saved to:
echo    %REPORT%
echo.
echo  Paste the ENTIRE contents of that file
echo  into your chat with the bot.
echo ============================================
echo.

type "%REPORT%"
pause
