"""
少女樂團機器人 啟動器
從同目錄下的 config.txt 讀取設定，然後啟動機器人。
"""
import os
import sys

CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__)),
    "config.txt",
)

CONFIG_KEYS = [
    "DISCORD_BOT_TOKEN",
    "GROQ_API_KEY",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
]

CONFIG_TEMPLATE = """\
# ================================================================
#  少女樂團機器人 — 設定檔
#  在等號後面填入妳的金鑰，存檔後執行 start.bat 即可啟動
# ================================================================

# Discord Bot Token（從 https://discord.com/developers/applications 獲取）
DISCORD_BOT_TOKEN={DISCORD_BOT_TOKEN}

# Groq API Key（從 https://console.groq.com/keys 獲取）
GROQ_API_KEY={GROQ_API_KEY}

# Cloudflare API Token（從 https://dash.cloudflare.com/profile/api-tokens 獲取）
CLOUDFLARE_API_TOKEN={CLOUDFLARE_API_TOKEN}

# Cloudflare Account ID（在 https://dash.cloudflare.com/ 右側欄位可找到）
CLOUDFLARE_ACCOUNT_ID={CLOUDFLARE_ACCOUNT_ID}
"""


def _parse_config(path: str) -> dict:
    """Parse KEY=value lines from config.txt, ignoring comments and blanks."""
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    if key in CONFIG_KEYS and val:
                        values[key] = val
    except Exception as e:
        print(f"[啟動器] 警告: 無法讀取 config.txt: {e}")
    return values


def _write_config(values: dict):
    """Write all four keys back to config.txt, preserving the template format."""
    filled = {k: values.get(k, "") for k in CONFIG_KEYS}
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(CONFIG_TEMPLATE.format(**filled))
        print(f"[啟動器] 設定已保存到 {CONFIG_FILE}")
    except Exception as e:
        print(f"[啟動器] 警告: 無法寫入 config.txt: {e}")


def _ensure_config_exists():
    """Create a blank config.txt if it doesn't exist yet."""
    if not os.path.exists(CONFIG_FILE):
        _write_config({})
        print(f"[啟動器] 已建立設定檔: {CONFIG_FILE}")
        print("[啟動器] 請在 config.txt 中填入妳的金鑰後重新執行。")


def load_config():
    """Load config.txt into environment variables."""
    _ensure_config_exists()
    values = _parse_config(CONFIG_FILE)
    for key, val in values.items():
        if not os.environ.get(key):
            os.environ[key] = val
    if values:
        print(f"[啟動器] 從 config.txt 讀取了 {len(values)} 個設定值")


def main():
    print("=" * 60)
    print("  少女樂團機器人 (AriaBot)")
    print("  Powered by Groq + Cloudflare Workers AI + SQLite")
    print("=" * 60)
    print()

    load_config()

    if not os.environ.get("DISCORD_BOT_TOKEN"):
        print("\n[Error] DISCORD_BOT_TOKEN is not set.")
        print("Please add it as a Replit Secret named DISCORD_BOT_TOKEN,")
        print(f"or fill it into {CONFIG_FILE} and restart.")
        sys.exit(1)

    if not os.environ.get("GROQ_API_KEY"):
        print("\n[Warning] GROQ_API_KEY is not set. Text chat will be unavailable.")

    if not os.environ.get("CLOUDFLARE_API_TOKEN") or not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        print("\n[Warning] Cloudflare config is incomplete. Image generation will be disabled.")

    print("\n[啟動器] 啟動機器人...")

    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    sys.path.insert(0, base_path)

    import bot
    bot.main()


if __name__ == "__main__":
    main()
