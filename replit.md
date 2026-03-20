# AriaBot — 少女樂團機器人

A Discord AI bot powered by Groq (text), Cloudflare Workers AI (image generation), and file-based plain-text storage.

## Architecture

- **Language**: Python 3.11
- **Bot framework**: discord.py 2.x with hybrid commands (prefix `!` and slash `/`)
- **Text AI**: Groq — `llama-3.3-70b-versatile` (with fallbacks)
- **Vision AI**: Groq vision models — tries Llama 4 Scout, Llama 3.2 90B/11B in order
- **Image generation**: Cloudflare Workers AI (Flux)
- **Storage**: Plain JSON/image files (human-editable) + SQLite for conversation history only

## Files

| File | Role |
|------|------|
| `launcher.py` | Entry point — loads tokens.txt, checks env vars, then calls `bot.main()` |
| `bot.py` | All Discord commands, event handlers, conversation logic |
| `groq_ai.py` | Groq API client — text chat, image understanding, memory extraction |
| `cloudflare_ai.py` | Cloudflare Workers AI — image generation |
| `database.py` | File-based storage for character, memories, knowledge; SQLite for history |
| `views.py` | Discord UI components (buttons, modals, paginated views) |
| `help_config.py` | User-facing help text |
| `tokens.txt` | Fill in API keys here (alternative to Replit Secrets) |

## Data Folder Layout

All persistent data lives under `data/` — easy to back up, transfer, or edit directly:

```
data/
  character/
    profile.json          ← Bot name & personality (edit freely)
  memories/
    memories.json         ← All long-term memories as a JSON list (edit freely)
  knowledge/
    <id>.json             ← One file per text KB entry (edit freely)
    images/
      <id>.json           ← Image metadata (title, description, tags)
      <id>.png/.jpg/…     ← The actual image file
  settings.json           ← Bot settings (suggestions, memory toggles, etc.)
  status.json             ← Persisted bot presence/status
  history.db              ← SQLite: conversation history (internal, not user-editable)
```

### Editing data directly

- **Character**: open `data/character/profile.json`, change `name` and `background`, save — takes effect on next message.
- **Memories**: open `data/memories/memories.json`, edit or delete entries in the JSON array, save.
- **Knowledge entries**: open any `data/knowledge/<id>.json`, edit `title`, `content`, or `tags`, save.
- **Image descriptions**: open `data/knowledge/images/<id>.json`, edit `image_description`, save.
- **Delete an entry**: delete the corresponding `.json` file (and image file for image entries).

## Required Secrets

Set these in `tokens.txt` **or** as Replit Secrets (Secrets take priority):

| Key | Required | Purpose |
|-----|----------|---------|
| `DISCORD_BOT_TOKEN` | ✅ | Bot login |
| `GROQ_API_KEY` | ✅ | Text chat & vision |
| `CLOUDFLARE_API_TOKEN` | Optional | Image generation |
| `CLOUDFLARE_ACCOUNT_ID` | Optional | Image generation |

## Running

The workflow `Start application` runs `python3 launcher.py`.

## Known Bugs Fixed

- **`search_knowledge()` always returned empty results** — fixed to split query into words.
- **Slash command errors were silent** — added `@bot.tree.error` handler.
- **Vision model retry bailed early** — fixed to always continue to next candidate.
- **`_client()` used stale API key** — Groq client now re-reads from env on every call.
- **Image entries saved with no description had no warning** — `saveimage` now tells the user to run `/setdesc`.
