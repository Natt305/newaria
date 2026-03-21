@echo off
chcp 65001 >nul
title AriaBot

:: ANSI color codes (Windows 10 1511+ VT100 support)
for /f %%a in ('echo prompt $E^|cmd /q /v:on /k "exit"') do set "ESC=%%a"
set "C=%ESC%[96m"
set "RED=%ESC%[91m"
set "GREEN=%ESC%[92m"
set "RST=%ESC%[0m"

echo %C%================================================================%RST%
echo %C%   AriaBot%RST%
echo %C%   Powered by Ollama / Groq + Cloudflare Workers AI + SQLite%RST%
echo %C%================================================================%RST%
echo.

echo %C%[Setup] Installing / updating dependencies...%RST%
python -m pip install -r requirements.txt -q
if errorlevel 1 goto trypy

echo %C%[Launcher] Starting bot...%RST%
python launcher.py
goto done

:trypy
py -m pip install -r requirements.txt -q
if errorlevel 1 goto nopython

echo %C%[Launcher] Starting bot...%RST%
py launcher.py
goto done

:nopython
echo %RED%[Error] Python not found. Please install Python 3.10 or later.%RST%
echo %RED%        https://www.python.org/downloads/%RST%
echo %RED%        Make sure to check "Add Python to PATH" during installation.%RST%

:done
echo.
pause
