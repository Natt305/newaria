@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title AriaBot

echo ================================================================
echo   AriaBot
echo   Powered by Groq + Cloudflare Workers AI + SQLite
echo ================================================================
echo.

rem --- Read IMAGE_BACKEND and COMFYUI_PATH from tokens.txt ---
set "IMAGE_BACKEND="
set "COMFYUI_PATH="
for /f "usebackq tokens=1,* delims==" %%A in ("tokens.txt") do (
    if /i "%%A"=="IMAGE_BACKEND" set "IMAGE_BACKEND=%%B"
    if /i "%%A"=="COMFYUI_PATH" set "COMFYUI_PATH=%%B"
)

rem --- Auto-launch ComfyUI only when IMAGE_BACKEND=comfyui ---
if /i not "!IMAGE_BACKEND!"=="comfyui" goto skipcomfy
if not defined COMFYUI_PATH goto skipcomfy
if "!COMFYUI_PATH!"=="" goto skipcomfy

echo [ComfyUI] Starting ComfyUI from: !COMFYUI_PATH!
start "ComfyUI" /d "!COMFYUI_PATH!" python main.py --listen 127.0.0.1 --port 8188

echo [ComfyUI] Waiting for ComfyUI to be ready on port 8188...
:waitloop
python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8188/system_stats',timeout=2); sys.exit(0)" 2>nul
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
