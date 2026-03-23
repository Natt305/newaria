"""
少女樂團機器人 啟動器
"""
import os
import sys


def load_tokens_file(base_path: str):
    tokens_path = os.path.join(base_path, "tokens.txt")
    if not os.path.exists(tokens_path):
        return
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key and value and not os.environ.get(key):
                    os.environ[key] = value


def load_permissions_file(base_path: str):
    permissions_path = os.path.join(base_path, "permissions.txt")
    if not os.path.exists(permissions_path):
        return

    import database

    known = set(database._DEFAULT_COMMAND_ROLES.keys())
    applied = 0
    skipped = []

    with open(permissions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, raw_value = line.partition("=")
            key = key.strip().lower()
            raw_value = raw_value.partition("#")[0].strip()  # strip inline comments

            if not key or not raw_value:
                continue

            if key not in known:
                skipped.append(key)
                continue

            if raw_value.lower() == "admin":
                role_value = "__admin__"
            elif raw_value.lower() in ("everyone", "open", "all"):
                role_value = None
            else:
                role_value = raw_value  # treat as a Discord role name

            database.set_command_role(key, role_value)
            applied += 1

    print(f"[啟動器] permissions.txt 已套用 {applied} 條權限設定。")
    if skipped:
        print(f"[啟動器] 忽略未知指令: {', '.join(skipped)}")


def main():
    print("=" * 60)
    print("  少女樂團機器人 (AriaBot)")
    print("  Powered by Ollama / Groq + Cloudflare Workers AI + SQLite")
    print("=" * 60)
    print()

    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_path)

    load_tokens_file(base_path)

    if not os.environ.get("DISCORD_BOT_TOKEN"):
        print("[Error] DISCORD_BOT_TOKEN is not set. Add it to tokens.txt or as a Replit Secret.")
        sys.exit(1)

    backend = os.environ.get("AI_BACKEND", "groq").strip().lower()
    if backend == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
        vision_model = os.environ.get("OLLAMA_VISION_MODEL", "gemma3:12b")
        print(f"[AI] Backend: Ollama @ {base_url}")
        print(f"[AI] Chat model: {model}")
        print(f"[AI] Vision model: {vision_model}")
    else:
        if not os.environ.get("GROQ_API_KEY"):
            print("[Warning] GROQ_API_KEY is not set. Text chat will be unavailable.")
        else:
            chat_model   = os.environ.get("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
            vision_model = os.environ.get("GROQ_VISION_MODEL", "").strip() or "meta-llama/llama-4-scout-17b-16e-instruct"
            print(f"[AI] Backend: Groq")
            print(f"[AI] Chat model:   {chat_model}")
            print(f"[AI] Vision model: {vision_model}")

    if not os.environ.get("CLOUDFLARE_API_TOKEN") or not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        print("[Warning] Cloudflare config is incomplete. Image generation will be disabled.")

    load_permissions_file(base_path)

    import database as _db
    _db.init_db()
    _db.migrate_thumbnails()
    _db.migrate_kb_descriptions()

    print("[啟動器] 啟動機器人...")

    import bot
    bot.main()


if __name__ == "__main__":
    import traceback
    import datetime

    try:
        main()
    except KeyboardInterrupt:
        print("\n[啟動器] 已停止。")
    except Exception:
        crash_log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crash.log")
        tb = traceback.format_exc()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(crash_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n{timestamp}\n{'=' * 60}\n{tb}\n")
        print("\n" + "=" * 60)
        print("  !! 機器人發生錯誤並已停止 !!")
        print("=" * 60)
        print(tb)
        print(f"完整錯誤已儲存至: {crash_log}")
        print("=" * 60)
    finally:
        pass  # No interactive prompt needed on Replit
