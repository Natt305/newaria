# AriaBot — 少女樂團機器人

A Discord AI bot powered by Groq (text), Cloudflare Workers AI (image generation), and SQLite (knowledge base).

## Architecture

- **Language**: Python 3.11
- **Bot framework**: discord.py 2.x with hybrid commands (prefix `!` and slash `/`)
- **Text AI**: Groq — `llama-3.3-70b-versatile` (with fallbacks)
- **Vision AI**: Groq vision models — tries Llama 4 Scout, Llama 3.2 90B/11B in order
- **Image generation**: Cloudflare Workers AI (Flux)
- **Database**: SQLite via `data/knowledge_base.db`
- **Config persistence**: `data/character.json`, `data/settings.json`, `data/status.json`

## Files

| File | Role |
|------|------|
| `launcher.py` | Entry point — loads env vars / config.txt, then calls `bot.main()` |
| `bot.py` | All Discord commands, event handlers, conversation logic |
| `groq_ai.py` | Groq API client — text chat, image understanding, memory extraction |
| `cloudflare_ai.py` | Cloudflare Workers AI — image generation |
| `database.py` | SQLite KB, conversation history, long-term memories, settings |
| `views.py` | Discord UI components (buttons, modals, paginated views) |
| `help_config.py` | User-facing help text |
| `config.txt` | Optional fallback config (env vars take priority) |

## Required Secrets

Set these as Replit Secrets:

| Secret | Required | Purpose |
|--------|----------|---------|
| `DISCORD_BOT_TOKEN` | ✅ | Bot login |
| `GROQ_API_KEY` | ✅ | Text chat & vision |
| `CLOUDFLARE_API_TOKEN` | Optional | Image generation |
| `CLOUDFLARE_ACCOUNT_ID` | Optional | Image generation |

## Running

The workflow `Start application` runs `python3 launcher.py`.

## Known Bugs Fixed

- **`search_knowledge()` always returned empty results** — the function was using the full conversation topic as a single `LIKE '%entire phrase%'` pattern, which never matched stored entries. Fixed to split into words (same approach as `search_memories()`).
- **Slash command errors were silent** — "This interaction failed" with no feedback. Added `@bot.tree.error` handler to report errors to the user and log tracebacks.
- **Vision model retry bailed early** — a rate limit on the first model would abort instead of trying the next one. Fixed to always continue to next candidate.
- **`_client()` used stale API key** — Groq client now re-reads from env on every call.
- **Image entries saved with no description had no warning** — `saveimage` now tells the user to run `/setdesc` when auto-description fails.
