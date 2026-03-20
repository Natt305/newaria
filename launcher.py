"""
少女樂團機器人 啟動器
"""
import os
import sys


def main():
    print("=" * 60)
    print("  少女樂團機器人 (AriaBot)")
    print("  Powered by Groq + Cloudflare Workers AI + SQLite")
    print("=" * 60)
    print()

    if not os.environ.get("DISCORD_BOT_TOKEN"):
        print("[Error] DISCORD_BOT_TOKEN is not set. Add it as a Replit Secret.")
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
