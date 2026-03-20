@echo off
setlocal EnableDelayedExpansion
title AriaBot
color 0A

echo ============================================================
echo   AriaBot - Discord AI Bot
echo   Powered by Groq + Gemini + SQLite
echo ============================================================
echo.

set "SCRIPT_DIR=%~dp0"
set "PYTHON_DIR=%SCRIPT_DIR%python"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "PIP_EXE=%PYTHON_DIR%\Scripts\pip.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
set "PIP_URL=https://bootstrap.pypa.io/get-pip.py"
set "PYTHON_ZIP=%SCRIPT_DIR%python_embed.zip"
set "GETPIP=%SCRIPT_DIR%get-pip.py"
set "STAMP=%PYTHON_DIR%\.setup_done"

if exist "%STAMP%" goto :run_bot

echo [Setup] First-time setup - this takes about 1-2 minutes...
echo [Setup] No Python installation on your system is needed.
echo.

:: --- Download embedded Python ---
echo [1/4] Downloading portable Python...
powershell -NoProfile -Command ^
  "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%' -UseBasicParsing" 2>nul
if not exist "%PYTHON_ZIP%" (
    echo [ERROR] Could not download Python. Check your internet connection.
    pause & exit /b 1
)

:: --- Unzip embedded Python ---
echo [2/4] Unpacking Python...
powershell -NoProfile -Command ^
  "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%PYTHON_DIR%' -Force" 2>nul
del "%PYTHON_ZIP%" 2>nul

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python extraction failed.
    pause & exit /b 1
)

:: Enable site-packages (required for pip and packages)
for %%f in ("%PYTHON_DIR%\python3*._pth") do (
    powershell -NoProfile -Command ^
      "(Get-Content '%%f') -replace '#import site','import site' | Set-Content '%%f'"
)

:: --- Download pip ---
echo [3/4] Installing pip...
powershell -NoProfile -Command ^
  "Invoke-WebRequest -Uri '%PIP_URL%' -OutFile '%GETPIP%' -UseBasicParsing" 2>nul
"%PYTHON_EXE%" "%GETPIP%" --quiet 2>nul
del "%GETPIP%" 2>nul

:: --- Install packages ---
echo [4/4] Installing bot packages (discord.py, groq, etc.)...
"%PIP_EXE%" install discord.py google-genai aiohttp Pillow groq requests ^
  --quiet --disable-pip-version-check

if errorlevel 1 (
    echo [ERROR] Package installation failed.
    pause & exit /b 1
)

:: Mark setup as complete
echo setup_done > "%STAMP%"

echo.
echo [Setup] Done! Starting bot...
echo.

:run_bot
"%PYTHON_EXE%" "%SCRIPT_DIR%launcher.py"

echo.
echo Bot stopped.
pause
