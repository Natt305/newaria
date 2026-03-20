"""
少女樂團機器人 啟動器
"""
import os
import sys


def load_tokens_file():
    base_path = os.path.dirname(os.path.abspath(__file__))
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


def main():
    print("=" * 60)
    print("  少女樂團機器人 (AriaBot)")
    print("  Powered by Groq + Cloudflare Workers AI + SQLite")
    print("=" * 60)
    print()

    load_tokens_file()

    if not os.environ.get("DISCORD_BOT_TOKEN"):
        print("[Error] DISCORD_BOT_TOKEN is not set. Add it to tokens.txt or as a Replit Secret.")
        sys.exit(1)

    if not os.environ.get("GROQ_API_KEY"):
        print("[Warning] GROQ_API_KEY is not set. Text chat will be unavailable.")

    if not os.environ.get("CLOUDFLARE_API_TOKEN") or not os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        print("[Warning] Cloudflare config is incomplete. Image generation will be disabled.")

    print("[啟動器] 啟動機器人...")

    base_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_path)

    import bot
    bot.main()


if __name__ == "__main__":
    main()
