@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title AriaBot

echo ================================================================
echo   AriaBot
echo   Powered by Groq + Cloudflare Workers AI + SQLite
echo ================================================================
echo.

rem --- Read IMAGE_BACKEND, COMFYUI_PATH and COMFYUI_ENGINE from tokens.txt ---
set "IMAGE_BACKEND="
set "COMFYUI_PATH="
set "COMFYUI_ENGINE="
for /f "usebackq tokens=1,* delims==" %%A in ("tokens.txt") do (
    if /i "%%A"=="IMAGE_BACKEND"   set "IMAGE_BACKEND=%%B"
    if /i "%%A"=="COMFYUI_PATH"    set "COMFYUI_PATH=%%B"
    if /i "%%A"=="COMFYUI_ENGINE"  set "COMFYUI_ENGINE=%%B"
)
if not defined COMFYUI_ENGINE set "COMFYUI_ENGINE=qwen"

rem --- Auto-launch ComfyUI only when IMAGE_BACKEND=comfyui and it is not already running ---
if /i not "!IMAGE_BACKEND!"=="comfyui" goto skipcomfy
if not defined COMFYUI_PATH goto skipcomfy
if "!COMFYUI_PATH!"=="" goto skipcomfy

rem Check if ComfyUI is already listening on port 8188 (plain socket — works regardless of HTTP response)
python -c "import socket,sys; s=socket.socket(); s.settimeout(2); r=s.connect_ex(('127.0.0.1',8188)); s.close(); sys.exit(0 if r==0 else 1)" 2>nul
if not errorlevel 1 (
    echo [ComfyUI] Already running on port 8188 — skipping launch.
    echo [ComfyUI] (Engine-scoped pack toggling is also skipped — restart ComfyUI manually if you want it to take effect.)
    echo.
    goto skipcomfy
)

rem --- Engine-scoped custom-node toggling (frees VRAM by keeping the inactive
rem     engine's heavy packs from auto-loading their models at boot). ---
echo [ComfyUI] Engine = !COMFYUI_ENGINE!  — applying engine-scoped pack manifest...
python scripts\scope_comfy_packs.py
if errorlevel 1 echo [ComfyUI] WARN: scope_comfy_packs.py exited non-zero — continuing anyway.

rem --- Optional model-path scoping: pass --extra-model-paths-config when the
rem     engine-matching yaml exists at the repo root. Cosmetic only (cleans
rem     Manager dropdowns); does NOT affect VRAM. ---
set "COMFY_EXTRA_PATHS_ARG="
set "COMFY_EXTRA_PATHS_FILE=%CD%\comfyui_extra_paths.!COMFYUI_ENGINE!.yaml"
if exist "!COMFY_EXTRA_PATHS_FILE!" (
    set "COMFY_EXTRA_PATHS_ARG=--extra-model-paths-config "!COMFY_EXTRA_PATHS_FILE!""
    echo [ComfyUI] Using engine-scoped model paths: !COMFY_EXTRA_PATHS_FILE!
)

rem --- Auto-detect GPU VRAM via nvidia-smi to pick the right ComfyUI memory
rem     mode. Maps detected GB to a flag using ComfyUI's own thresholds:
rem        >= 23 GB  -> --highvram     (24 GB cards: keep everything cached)
rem        >=  7 GB  -> --normalvram   (8/12/16 GB: aggressive eviction)
rem         <  7 GB  -> --lowvram      (UNET splitting needed; 2-3x slower)
rem     If nvidia-smi is missing or fails (non-NVIDIA card, weird driver
rem     state), pass no flag and let ComfyUI fall back to its built-in auto. ---
set "COMFY_VRAM_ARG="
set "COMFY_VRAM_MB="
for /f "usebackq tokens=1" %%V in (`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2^>nul`) do (
    if not defined COMFY_VRAM_MB set "COMFY_VRAM_MB=%%V"
)
set "_COMFY_VRAM_GB=0"
if defined COMFY_VRAM_MB set /a "_COMFY_VRAM_GB=COMFY_VRAM_MB/1024" 2>nul
if !_COMFY_VRAM_GB! GEQ 23 (
    set "COMFY_VRAM_ARG=--highvram"
) else if !_COMFY_VRAM_GB! GEQ 7 (
    set "COMFY_VRAM_ARG=--normalvram"
) else if !_COMFY_VRAM_GB! GTR 0 (
    set "COMFY_VRAM_ARG=--lowvram"
)
if defined COMFY_VRAM_ARG (
    echo [ComfyUI] Detected GPU VRAM: !COMFY_VRAM_MB! MB -^> !COMFY_VRAM_ARG!
) else (
    echo [ComfyUI] Could not detect GPU VRAM ^(nvidia-smi missing/failed^) — ComfyUI will use its built-in auto memory mode.
)

echo [ComfyUI] Starting ComfyUI from: !COMFYUI_PATH!
start "ComfyUI" /d "!COMFYUI_PATH!" python main.py --listen 127.0.0.1 --port 8188 !COMFY_EXTRA_PATHS_ARG! !COMFY_VRAM_ARG!

echo [ComfyUI] Waiting for ComfyUI to be ready on port 8188...
:waitloop
python -c "import socket,sys; s=socket.socket(); s.settimeout(2); r=s.connect_ex(('127.0.0.1',8188)); s.close(); sys.exit(0 if r==0 else 1)" 2>nul
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto waitloop
)
echo [ComfyUI] Ready!
echo.

:skipcomfy
echo [Setup] Installing / updating dependencies...
python -m pip install -r requirements.txt -q
if errorlevel 1 goto trypy

echo [Launcher] Starting bot...
python launcher.py
goto done

:trypy
py -m pip install -r requirements.txt -q
if errorlevel 1 goto nopython

echo [Launcher] Starting bot...
py launcher.py
goto done

:nopython
echo [Error] Python not found. Please install Python 3.8 or later.
echo         https://www.python.org/downloads/
echo         Make sure to check "Add Python to PATH" during installation.

:done
echo.
pause
