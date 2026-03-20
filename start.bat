@echo off
chcp 65001 >nul
title 少女樂團機器人 (AriaBot)

echo ================================================================
echo   少女樂團機器人 (AriaBot)
echo   Powered by Groq + Cloudflare Workers AI + SQLite
echo ================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Python not found. Please install Python 3.8 or later.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Install requirements if needed
if exist requirements.txt (
    echo [Setup] Installing dependencies...
    pip install -r requirements.txt -q
    echo [Setup] Dependencies ready.
    echo.
)

REM Launch the bot
echo [Launcher] Starting bot...
python launcher.py

echo.
echo [Launcher] Bot has stopped.
pause
