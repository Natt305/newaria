@echo off
chcp 65001 >nul
title AriaBot

echo ================================================================
echo   AriaBot
echo   Powered by Groq + Cloudflare Workers AI + SQLite
echo ================================================================
echo.

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
