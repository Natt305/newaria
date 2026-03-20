@echo off
chcp 65001 >nul
title AriaBot

echo ================================================================
echo   AriaBot (少女樂團機器人)
echo   Powered by Groq + Cloudflare Workers AI + SQLite
echo ================================================================
echo.

REM Check for Python (try "python" then "py" launcher)
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo [Error] Python not found. Please install Python 3.8 or later.
        echo Download from: https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    set PYTHON=py
) else (
    set PYTHON=python
)

REM Install requirements if needed
if exist requirements.txt (
    echo [Setup] Installing dependencies...
    %PYTHON% -m pip install -r requirements.txt -q
    echo [Setup] Dependencies ready.
    echo.
)

REM Launch the bot
echo [Launcher] Starting bot...
%PYTHON% launcher.py

echo.
echo [Launcher] Bot has stopped.
pause
