# AriaBot — Discord AI Bot

A fully standalone Discord bot with a local SQLite knowledge base, fast text chat via **Groq**, and seamless **Gemini** image generation.

---

## Features

- **Fast text chat** — powered by Groq (Llama 3.3 70B by default), with automatic model fallback
- **Seamless image generation** — when Groq says it can't generate an image, Gemini steps in automatically and posts the image
- **Local knowledge base** — save text and image entries with `!remember` / `!saveimage`
- **Image understanding** — paste an image and ask a question; Gemini analyzes it
- **Character system** — customize the bot's name and personality at runtime
- **Grey suggestion buttons** — 3 clickable follow-up ideas after every reply
- **Standalone exe** — no Python needed on the target machine

---

## Setup

### API Keys

| Key | Required | Where to get |
|-----|----------|--------------|
| `DISCORD_BOT_TOKEN` | Yes | https://discord.com/developers/applications |
| `GROQ_API_KEY` | Yes (text) | https://console.groq.com/keys |
| `CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID` | Optional (images) | https://aistudio.google.com/apikey |

All three are free tier.

### Discord Bot Settings (Developer Portal)

Enable these intents:
- **Message Content Intent**
- **Server Members Intent**

### Running from source

```bash
cd discord-bot
pip install discord.py google-genai aiohttp Pillow pyinstaller groq

export DISCORD_BOT_TOKEN=your_token
export GROQ_API_KEY=your_groq_key
export CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID=your_gemini_key   # optional

python launcher.py
```

### Building the .exe

```bash
cd discord-bot
python build_exe.py
```

The exe appears in `dist/AriaBot` (Linux) or `dist/AriaBot.exe` (Windows).
On first run it prompts for your tokens and saves them to `config.json`.

---

## Commands

### Talking to the bot
- **@mention** or **reply** to the bot to chat
- Ask it to draw/generate images — Groq acknowledges it, Gemini generates and posts it automatically

### Character
| Command | Description |
|---------|-------------|
| `!character` | View current character |
| `!setcharacter "Name" <background>` | Change name and personality |

### Knowledge Base
| Command | Description |
|---------|-------------|
| `!remember "title" content` | Save text to KB |
| `!saveimage "title" [desc]` | Save an image attachment |
| `!setdesc <id> <description>` | Update an image entry's description |
| `!viewentry <id>` | View full entry details |
| `!knowledge [query]` | Search or list entries |
| `!forget <id>` | Delete an entry |

### Other
| Command | Description |
|---------|-------------|
| `!generate <prompt>` | Generate an image directly (Gemini) |
| `!clear` | Clear conversation history for this channel |
| `!help` | Show help |

---

## How Seamless Image Generation Works

1. User says: *"@Aria draw me a cat astronaut"*
2. Groq replies naturally (e.g. *"Creating that for you now!"*) — sent as the chat reply with suggestion buttons
3. Bot detects Groq couldn't produce an actual image
4. Gemini generates the image and posts it as a follow-up in the same channel

You get both a conversational reply **and** the generated image, automatically.

---

## Architecture

```
launcher.py      — entry point, loads/saves config.json, prompts for keys
bot.py           — Discord bot, commands, message handling, image fallback logic
groq_ai.py       — Groq text chat with image-decline detection
cloudflare_ai.py — Cloudflare Workers AI image generation (Flux 2 Dev)
database.py      — SQLite knowledge base
build_exe.py     — PyInstaller build script
```

Runtime data files (created alongside the exe):
- `knowledge_base.db` — SQLite database
- `config.json` — saved API keys
