"""
Build script to create a standalone .exe using PyInstaller.
Run this from the discord-bot directory: python build_exe.py
"""
import subprocess
import sys
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(SCRIPT_DIR, "dist")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")


def clean():
    for d in [DIST_DIR, BUILD_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Cleaned: {d}")
    spec_file = os.path.join(SCRIPT_DIR, "AriaBot.spec")
    if os.path.exists(spec_file):
        os.remove(spec_file)


def build():
    print("Building AriaBot.exe...")
    print("This may take a few minutes...\n")

    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--onefile",
        "--console",
        "--name", "AriaBot",
        "--add-data", f"{os.path.join(SCRIPT_DIR, 'bot.py')}{os.pathsep}.",
        "--add-data", f"{os.path.join(SCRIPT_DIR, 'groq_ai.py')}{os.pathsep}.",
        "--add-data", f"{os.path.join(SCRIPT_DIR, 'cloudflare_ai.py')}{os.pathsep}.",
        "--add-data", f"{os.path.join(SCRIPT_DIR, 'database.py')}{os.pathsep}.",
        "--add-data", f"{os.path.join(SCRIPT_DIR, 'views.py')}{os.pathsep}.",
        "--hidden-import", "discord",
        "--hidden-import", "discord.ext.commands",
        "--hidden-import", "aiohttp",
        "--hidden-import", "groq",
        "--hidden-import", "PIL",
        "--hidden-import", "sqlite3",
        "--hidden-import", "asyncio",
        "--hidden-import", "json",
        "--distpath", DIST_DIR,
        "--workpath", BUILD_DIR,
        "--specpath", SCRIPT_DIR,
        os.path.join(SCRIPT_DIR, "launcher.py"),
    ]

    result = subprocess.run(cmd, cwd=SCRIPT_DIR)

    if result.returncode == 0:
        exe_path = os.path.join(DIST_DIR, "AriaBot.exe" if sys.platform == "win32" else "AriaBot")
        if not os.path.exists(exe_path):
            exe_path = os.path.join(DIST_DIR, "AriaBot")

        print("\n" + "=" * 60)
        print("  BUILD SUCCESSFUL!")
        print("=" * 60)
        print(f"\nExecutable: {exe_path}")
        print("\nTo distribute:")
        print("1. Copy AriaBot.exe (or AriaBot) to any Windows/Linux machine")
        print("2. Run it — it will prompt for your Discord Token, Groq API Key,")
        print("   Cloudflare API Token, and Cloudflare Account ID on first run")
        print("3. Config is saved to config.json in the same folder as the exe")
        print("4. knowledge_base.db is created in the same folder as the exe")
        print("\nNo Python installation required on the target machine!")
    else:
        print("\n[ERROR] Build failed! Check the output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    clean()
    build()
