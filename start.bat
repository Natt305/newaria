@echo off
setlocal enabledelayedexpansion
title AriaBot

echo ================================================================
echo   AriaBot
echo   Powered by Groq + Cloudflare Workers AI + SQLite
echo ================================================================
echo.

rem --- Read IMAGE_BACKEND, COMFYUI_PATH and COMFYUI_ENGINE from tokens.txt ---
rem NOTE: chcp 65001 (UTF-8) is intentionally NOT set yet. cmd.exe's batch
rem parser is well-known to silently mis-handle multi-byte UTF-8 inside
rem `for /f` -- it can skip lines, terminate the loop early, or assign empty
rem values, all with zero error output. tokens.txt has 100+ lines of CJK
rem comments before the IMAGE_BACKEND line, which reliably triggers the bug.
rem We parse under the legacy code page first, then switch to 65001 below
rem so the bot's own Chinese console output still renders correctly.
set "IMAGE_BACKEND="
set "COMFYUI_PATH="
set "COMFYUI_ENGINE="
set "COMFYUI_PYTHON="
for /f "usebackq tokens=1,* delims==" %%A in ("tokens.txt") do (
    if /i "%%A"=="IMAGE_BACKEND"   set "IMAGE_BACKEND=%%B"
    if /i "%%A"=="COMFYUI_PATH"    set "COMFYUI_PATH=%%B"
    if /i "%%A"=="COMFYUI_ENGINE"  set "COMFYUI_ENGINE=%%B"
    if /i "%%A"=="COMFYUI_PYTHON"  set "COMFYUI_PYTHON=%%B"
)
if not defined COMFYUI_ENGINE set "COMFYUI_ENGINE=qwen"

rem --- Switch to UTF-8 only AFTER tokens.txt has been parsed. ---
chcp 65001 >nul

rem --- Surface what we actually read so config drift is self-diagnosable. ---
echo [Config] IMAGE_BACKEND=!IMAGE_BACKEND!  COMFYUI_PATH=!COMFYUI_PATH!  COMFYUI_ENGINE=!COMFYUI_ENGINE!
echo.

rem --- Auto-launch ComfyUI only when IMAGE_BACKEND=comfyui and it is not already running ---
if /i not "!IMAGE_BACKEND!"=="comfyui" (
    echo [ComfyUI] Skipping auto-launch: IMAGE_BACKEND is "!IMAGE_BACKEND!" ^(need "comfyui"^).
    echo.
    goto skipcomfy
)
if not defined COMFYUI_PATH (
    echo [ComfyUI] Skipping auto-launch: COMFYUI_PATH is not set in tokens.txt.
    echo.
    goto skipcomfy
)
if "!COMFYUI_PATH!"=="" (
    echo [ComfyUI] Skipping auto-launch: COMFYUI_PATH is empty in tokens.txt.
    echo.
    goto skipcomfy
)

rem Check if ComfyUI is already listening on port 8188 (plain socket -- works regardless of HTTP response)
python -c "import socket,sys; s=socket.socket(); s.settimeout(2); r=s.connect_ex(('127.0.0.1',8188)); s.close(); sys.exit(0 if r==0 else 1)" 2>nul
if not errorlevel 1 (
    echo [ComfyUI] Port 8188 is already in use -- a ComfyUI ^(or other^) process is bound to it.
    echo [ComfyUI] Skipping auto-launch. If this is a stale process, close it from Task Manager and re-run start.bat.
    echo [ComfyUI] Engine-scoped pack toggling is also skipped; restart ComfyUI manually if you want it to take effect.
    echo.
    goto skipcomfy
)
echo [ComfyUI] Port 8188 is free -- proceeding with auto-launch.

rem --- Validate COMFYUI_PATH actually exists before trying to launch ---
if not exist "!COMFYUI_PATH!\" (
    echo [ComfyUI] ERROR: COMFYUI_PATH does not exist: !COMFYUI_PATH!
    echo [ComfyUI]   Check the COMFYUI_PATH value in tokens.txt and make sure the folder exists.
    echo.
    goto skipcomfy
)
if not exist "!COMFYUI_PATH!\main.py" (
    echo [ComfyUI] ERROR: main.py not found inside COMFYUI_PATH: !COMFYUI_PATH!
    echo [ComfyUI]   Make sure COMFYUI_PATH points to the ComfyUI folder that contains main.py.
    echo.
    goto skipcomfy
)

rem --- Engine-scoped custom-node toggling (frees VRAM by keeping the inactive
rem     engine's heavy packs from auto-loading their models at boot). ---
echo [ComfyUI] Engine = !COMFYUI_ENGINE! -- applying engine-scoped pack manifest...
python scripts\scope_comfy_packs.py
if errorlevel 1 (
    echo [ComfyUI] WARN: scope_comfy_packs.py exited non-zero.
    echo [ComfyUI]   This usually means 'python' on your PATH is wrong, or the script
    echo [ComfyUI]   hit a permissions error renaming a custom_nodes folder.
    echo [ComfyUI]   Continuing anyway -- pack state may not be optimal for this engine.
)

rem --- Optional model-path scoping: pass --extra-model-paths-config when the
rem     engine-matching yaml exists at the repo root. Cosmetic only (cleans
rem     Manager dropdowns); does NOT affect VRAM. ---
set "COMFY_EXTRA_PATHS_FILE="
set "_COMFY_YAML_CANDIDATE=%CD%\comfyui_extra_paths.!COMFYUI_ENGINE!.yaml"
if exist "!_COMFY_YAML_CANDIDATE!" (
    set "COMFY_EXTRA_PATHS_FILE=!_COMFY_YAML_CANDIDATE!"
    echo [ComfyUI] Using engine-scoped model paths: !_COMFY_YAML_CANDIDATE!
)

rem --- Auto-detect GPU VRAM via nvidia-smi to pick the right ComfyUI memory
rem     mode. The CONCEPTUAL mapping is:
rem        >= 24 GB cards  -> --highvram     (keep everything cached)
rem        12-23 GB cards  -> --normalvram   (aggressive eviction)
rem         8-11 GB cards  -> --normalvram   (same)
rem         <  8 GB cards  -> --lowvram      (UNET splitting; 2-3x slower)
rem     But nvidia-smi reports the BIOS-reported total in MiB minus a small
rem     reserved chunk (~20-30 MiB/GiB for ECC/firmware), so a "16 GB" card
rem     reports ~16380 MiB which integer-divides to 15 GiB; a "24 GB" card
rem     reports ~24564 MiB -> 23 GiB; an "8 GB" card reports ~8188 MiB -> 7
rem     GiB. To map those rounded-down GiB values back to the conceptual
rem     buckets above, the if-chain uses the OFF-BY-ONE-LOWER boundaries
rem     23/7 (instead of 24/8). Net effect:
rem        24 GB cards -> --highvram
rem        16 GB cards -> --normalvram
rem        12 GB cards -> --normalvram
rem         8 GB cards -> --normalvram
rem         6 GB cards -> --lowvram
rem     If nvidia-smi is missing or fails (non-NVIDIA card, weird driver
rem     state), pass no flag and let ComfyUI fall back to its built-in auto.
rem
rem     nvidia-smi search: try the well-known install path if it is not on PATH.
set "COMFY_VRAM_ARG="
set "COMFY_VRAM_MB="
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    rem nvidia-smi not on PATH -- try the default NVIDIA driver install location
    if exist "%SystemRoot%\System32\nvidia-smi.exe" (
        set "_NSMI=%SystemRoot%\System32\nvidia-smi.exe"
    ) else if exist "C:\Windows\System32\nvidia-smi.exe" (
        set "_NSMI=C:\Windows\System32\nvidia-smi.exe"
    ) else (
        set "_NSMI="
    )
) else (
    set "_NSMI=nvidia-smi"
)
if defined _NSMI (
    for /f "usebackq tokens=1" %%V in (`"!_NSMI!" --query-gpu=memory.total --format=csv,noheader,nounits 2^>nul`) do (
        if not defined COMFY_VRAM_MB set "COMFY_VRAM_MB=%%V"
    )
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
    echo [ComfyUI] Detected GPU VRAM: !COMFY_VRAM_MB! MB ^(bucket=!_COMFY_VRAM_GB! GiB^) -^> !COMFY_VRAM_ARG!
) else (
    echo [ComfyUI] Could not detect GPU VRAM ^(nvidia-smi not found^), ComfyUI will use its built-in auto memory mode.
)

rem --- Resolve which Python executable to use for ComfyUI.
rem     Priority: explicit COMFYUI_PYTHON in tokens.txt > auto-detect > system python.
rem
rem     ComfyUI Desktop installer layout:
rem       <drive>:\comfyui\resources\ComfyUI\      <- COMFYUI_PATH
rem       <drive>:\comfyui\resources\python_embeded\python.exe  <- embedded Python
rem     Venv layout:
rem       <COMFYUI_PATH>\.venv\Scripts\python.exe
if defined COMFYUI_PYTHON (
    if not "!COMFYUI_PYTHON!"=="" (
        set "_COMFY_PY=!COMFYUI_PYTHON!"
        echo [ComfyUI] Using Python from tokens.txt: !_COMFY_PY!
        goto :py_resolved
    )
)
rem Auto-detect: get the parent directory of COMFYUI_PATH (e.g. E:\comfyui\resources).
rem %%~dpP is unreliable on directory paths -- use pushd/cd .. instead.
pushd "!COMFYUI_PATH!" 2>nul
cd ..
set "_COMFY_PARENT=!CD!"
popd
echo [ComfyUI] Parent dir: !_COMFY_PARENT!

rem _COMFY_EXE   = full path to the exe (no quotes; quoted at call site)
rem _COMFY_EXTRA = words inserted between exe and main.py ("run python" for uv; else empty)
set "_COMFY_EXE="
set "_COMFY_EXTRA="

rem -----------------------------------------------------------------------
rem IMPORTANT: never use  for /r "!VARIABLE!"  -- cmd.exe prepends the CWD
rem to any variable-expanded absolute path in that position.
rem All searches below use either direct  if exist  or  pushd + for /r .
rem -----------------------------------------------------------------------

rem --- S1: ComfyUI Desktop stores its fully-installed venv in the Windows user
rem     AppData folder (Electron userData), NOT inside the ComfyUI source tree.
rem     Try every known variant of that path.
for %%A in (
    "%APPDATA%\ComfyUI Desktop\venv\Scripts\python.exe"
    "%APPDATA%\ComfyUI\venv\Scripts\python.exe"
    "%LOCALAPPDATA%\ComfyUI Desktop\venv\Scripts\python.exe"
    "%LOCALAPPDATA%\ComfyUI\venv\Scripts\python.exe"
    "%APPDATA%\ComfyUI Desktop\.venv\Scripts\python.exe"
    "%LOCALAPPDATA%\Programs\comfyui\resources\venv\Scripts\python.exe"
    "%LOCALAPPDATA%\comfyui\resources\venv\Scripts\python.exe"
) do (
    if not defined _COMFY_EXE if exist %%A (
        set "_COMFY_EXE=%%~A"
        echo [ComfyUI] Desktop AppData venv: %%~A
    )
)
if defined _COMFY_EXE goto :py_resolved

rem --- S2: .venv inside COMFYUI_PATH (only valid if it actually has packages --
rem     check for sqlalchemy as a proxy for a fully-installed env)
if exist "!COMFYUI_PATH!\.venv\Scripts\python.exe" (
    if exist "!COMFYUI_PATH!\.venv\Lib\site-packages\sqlalchemy" (
        set "_COMFY_EXE=!COMFYUI_PATH!\.venv\Scripts\python.exe"
        echo [ComfyUI] venv Python ^(packages verified^): !_COMFY_EXE!
        goto :py_resolved
    ) else (
        echo [ComfyUI] Found .venv but it has no packages -- skipping ^(run ComfyUI Desktop once to install them^)
    )
)

rem --- S3: python_embeded / python_embedded siblings (older portable installs)
for %%N in (python_embeded python_embedded) do (
    if not defined _COMFY_EXE if exist "!_COMFY_PARENT!\%%N\python.exe" (
        set "_COMFY_EXE=!_COMFY_PARENT!\%%N\python.exe"
        echo [ComfyUI] Embedded Python ^(%%N^): !_COMFY_EXE!
    )
)
if defined _COMFY_EXE goto :py_resolved

rem --- S4: uv.exe at well-known Desktop locations (last resort -- uv sync
rem     only works if pyproject.toml lists all ComfyUI deps, which it may not)
for %%U in (
    "!_COMFY_PARENT!\uv\win\uv.exe"
    "!_COMFY_PARENT!\uv\bin\uv.exe"
    "!_COMFY_PARENT!\uv\uv.exe"
) do (
    if not defined _COMFY_EXE if exist %%U (
        set "_COMFY_EXE=%%~U"
        set "_COMFY_EXTRA=run python"
        echo [ComfyUI] uv runtime ^(NOTE: run ComfyUI Desktop once first to install packages^): %%~U
    )
)
if defined _COMFY_EXE goto :py_resolved

rem --- Nothing found
echo [ComfyUI] ERROR: No Python runtime found.
echo [ComfyUI]   Open ComfyUI Desktop, let it fully load, then run this in a Command Prompt:
echo [ComfyUI]     where /r "%APPDATA%" python.exe
echo [ComfyUI]     where /r "%LOCALAPPDATA%" python.exe
echo [ComfyUI]   Find the python.exe that has ComfyUI packages, then set in tokens.txt:
echo [ComfyUI]     COMFYUI_PYTHON=^<full path to that python.exe^>
set "_COMFY_EXE=python"
:py_resolved

rem --- Write a tiny temp launcher using >> per line to avoid parenthesised-block
rem     quote-corruption. The file cd's into COMFYUI_PATH then runs the exe.
set "_COMFY_TMPBAT=%TEMP%\ariabot_comfy_launch.bat"
echo @echo off> "!_COMFY_TMPBAT!"
echo cd /d "!COMFYUI_PATH!">> "!_COMFY_TMPBAT!"

rem --- Run uv sync whenever pyproject.toml + uv.exe are present, regardless of
rem     which strategy found Python. This ensures the .venv always has all packages
rem     (it is a no-op after the first successful sync and takes ~1 s steady-state).
set "_COMFY_SYNC_EXE="
if exist "!COMFYUI_PATH!\pyproject.toml" (
    for %%U in (
        "!_COMFY_PARENT!\uv\win\uv.exe"
        "!_COMFY_PARENT!\uv\bin\uv.exe"
        "!_COMFY_PARENT!\uv\uv.exe"
        "!_COMFY_PARENT!\app.asar.unpacked\resources\uv\win\uv.exe"
    ) do (
        if not defined _COMFY_SYNC_EXE if exist %%U set "_COMFY_SYNC_EXE=%%~U"
    )
)
if defined _COMFY_SYNC_EXE (
    echo echo [ComfyUI] Syncing dependencies ^(first run installs packages -- may take a minute^)...>> "!_COMFY_TMPBAT!"
    echo "!_COMFY_SYNC_EXE!" sync>> "!_COMFY_TMPBAT!"
)

if defined COMFY_EXTRA_PATHS_FILE (
    echo "!_COMFY_EXE!" !_COMFY_EXTRA! main.py --listen 127.0.0.1 --port 8188 --extra-model-paths-config "!COMFY_EXTRA_PATHS_FILE!" !COMFY_VRAM_ARG!>> "!_COMFY_TMPBAT!"
) else (
    echo "!_COMFY_EXE!" !_COMFY_EXTRA! main.py --listen 127.0.0.1 --port 8188 !COMFY_VRAM_ARG!>> "!_COMFY_TMPBAT!"
)

echo [ComfyUI] Launch command:
type "!_COMFY_TMPBAT!"
echo [ComfyUI] Starting ComfyUI from: !COMFYUI_PATH!
echo [ComfyUI] *** Check the new "ComfyUI" window for startup errors if the bot hangs here ***
start "ComfyUI" cmd /k "!_COMFY_TMPBAT!"

rem --- Wait up to 5 minutes (150 x 2s) for ComfyUI to bind port 8188.
rem     If it never comes up, print a diagnostic and skip to the bot.
rem     Large models (Qwen GGUF) can take 60-90 s to load on first run.
echo [ComfyUI] Waiting for ComfyUI to be ready on port 8188 ^(up to 5 min^)...
set "_COMFY_WAIT=0"
:waitloop
python -c "import socket,sys; s=socket.socket(); s.settimeout(2); r=s.connect_ex(('127.0.0.1',8188)); s.close(); sys.exit(0 if r==0 else 1)" 2>nul
if not errorlevel 1 (
    echo [ComfyUI] Ready!
    echo.
    goto skipcomfy
)
set /a "_COMFY_WAIT+=1"
if !_COMFY_WAIT! GEQ 150 (
    echo.
    echo [ComfyUI] ERROR: ComfyUI did not bind port 8188 within 5 minutes.
    echo [ComfyUI] Most likely causes:
    echo [ComfyUI]   1. ComfyUI crashed -- look at the "ComfyUI" window for the error message.
    echo [ComfyUI]   2. A required custom node or model file is missing/mis-named.
    echo [ComfyUI]   3. ComfyUI's Python environment is broken -- try running it manually:
    echo [ComfyUI]      cd "!COMFYUI_PATH!" ^& python main.py --listen 127.0.0.1 --port 8188
    echo [ComfyUI] Continuing to start the bot anyway ^(image generation will fail^).
    echo.
    goto skipcomfy
)
if !_COMFY_WAIT! EQU 15 echo [ComfyUI]   Still waiting... ^(!_COMFY_WAIT! / 150^) -- large models take 60-90s to load.
if !_COMFY_WAIT! EQU 30 echo [ComfyUI]   Still waiting... ^(!_COMFY_WAIT! / 150^)
if !_COMFY_WAIT! EQU 60 echo [ComfyUI]   Still waiting... ^(!_COMFY_WAIT! / 150^) -- check the ComfyUI window for errors.
if !_COMFY_WAIT! EQU 90 echo [ComfyUI]   Still waiting... ^(!_COMFY_WAIT! / 150^)
if !_COMFY_WAIT! EQU 120 echo [ComfyUI]   Still waiting... ^(!_COMFY_WAIT! / 150^) -- nearly at timeout.
timeout /t 2 /nobreak >nul
goto waitloop

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
