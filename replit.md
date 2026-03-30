# AriaBot — 少女樂團機器人

A Discord AI bot powered by a configurable AI backend (Ollama or Groq), Cloudflare Workers AI (image generation), and file-based plain-text storage.

## Architecture

- **Language**: Python 3.11
- **Bot framework**: discord.py 2.x with hybrid commands (prefix `!` and slash `/`)
- **Text AI**: Configurable — Ollama (local, default `gemma3:12b`) or Groq (`llama-3.3-70b-versatile`)
- **Vision AI**: Configurable — Ollama vision model or Groq vision models
- **Image generation**: Cloudflare Workers AI (Flux)
- **Storage**: Plain JSON/image files (human-editable) + SQLite for conversation history only

## Files

| File | Role |
|------|------|
| `launcher.py` | Entry point — loads tokens.txt, checks env vars, then calls `bot.main()` |
| `bot.py` | All Discord commands, event handlers, conversation logic |
| `ai_backend.py` | Router — delegates AI calls to groq_ai or ollama_ai based on `AI_BACKEND` env var |
| `groq_ai.py` | Groq API client — text chat, image understanding, memory extraction |
| `ollama_ai.py` | Ollama API client — same interface as groq_ai, uses local Ollama server |
| `cloudflare_ai.py` | Cloudflare Workers AI — image generation |
| `database.py` | File-based storage for character, memories, knowledge; SQLite for history |
| `views.py` | Discord UI components (buttons, modals, paginated views) |
| `help_config.py` | User-facing help text |
| `tokens.txt` | Fill in API keys and backend config here (alternative to Replit Secrets) |

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

- **Character**: open `data/character/profile.json`, change `name`, `background`, and/or `personality`, save — takes effect on next message.
- **Memories**: open `data/memories/memories.json`, edit or delete entries in the JSON array, save.
- **Knowledge entries**: open any `data/knowledge/<id>.json`, edit `title`, `content`, or `tags`, save.
- **Image descriptions**: open `data/knowledge/images/<id>.json`, edit `appearance_description` (Flux-ready, bot-only) or `display_description` (user-facing lore), save.
- **Delete an entry**: delete the corresponding `.json` file (and image file for image entries).

## Configuration

Set these in `tokens.txt` **or** as Replit Secrets (Secrets take priority):

| Key | Required | Purpose |
|-----|----------|---------|
| `DISCORD_BOT_TOKEN` | ✅ | Bot login |
| `AI_BACKEND` | Optional | `groq` (default) or `ollama` |
| `GROQ_API_KEY` | If using Groq | Text chat & vision |
| `OLLAMA_BASE_URL` | If using Ollama | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | If using Ollama | Chat model (default: `gemma3:12b`) |
| `OLLAMA_VISION_MODEL` | If using Ollama | Vision model (default: `gemma3:12b`) |
| `CLOUDFLARE_API_TOKEN` | Optional | Image generation |
| `CLOUDFLARE_ACCOUNT_ID` | Optional | Image generation |
| `COMFYUI_MODE` | Optional | `multiref` (default, recommended) — spatial-masked multi-character; `refchain` — ReferenceChainConditioning custom node; `ultimate_inpaint` — SAM3 inpaint loop (requires many custom nodes) |
| `COMFYUI_USE_SAM` | Optional | `true` or `false` (default) — SAM3 auto-segmentation, only relevant for `ultimate_inpaint` mode |

### Multi-Character Photo-Referencing Modes

#### `multiref` (default — recommended, no extra node packs needed)

The primary multi-character path. Uses **only built-in ComfyUI nodes** — no SAM3, no IP-Adapter, no custom packs.  
Achieves **Mortis 9.6/10, Nina 10.0/10, distinctiveness 8/10** in automated scoring.

**Architecture (2 characters):**
1. Each character gets a **separate** `CLIPTextEncode` with scene context + character-specific appearance text.
2. Per-character `ReferenceLatent` chain injects each reference photo's latent into that character's conditioning.
3. **Spatial masking**: `SolidMask` + `MaskComposite` build left-half and right-half masks entirely in-graph (no uploads).
4. `ConditioningSetMask` locks each character's conditioning to their spatial region.
5. `ConditioningCombine` merges the two masked conditionings before the sampler.
6. Standard FLUX.2 Klein advanced sampler: `EmptyFlux2LatentImage` → `Flux2Scheduler` → `SamplerCustomAdvanced`.

**Only requires:** `ComfyUI-GGUF` (city96) for GGUF model loading. Everything else is ComfyUI core (0.8.2+).

#### `refchain` (opt-in, requires `ComfyUI-ReferenceChain` node pack)

Uses `ReferenceChainConditioning` — one node per character, handles scaling + VAE encoding internally. Falls back to `multiref` automatically if the node is not installed.

#### `ultimate_inpaint` (opt-in, requires many node packs + SAM3)

The legacy two-pass workflow: generates a base scene first, then runs per-character inpainting loops with optional SAM3 auto-segmentation. Falls back to `multiref` if the workflow fails.

**Required ComfyUI node packs** (Ultimate Inpaint only):
- `ComfyUI-Easy-Use`, `ComfyUI-Crystools`, `ComfyUI-Impact-Pack`, `ComfyUI-BRIA-RMBG`, `SAM3`, `ComfyUI-GODMT`, `ComfyUI-layerdiffuse`

## Running

The workflow `Start application` runs `python3 launcher.py`.

## Known Bugs Fixed

- **`search_knowledge()` always returned empty results** — fixed to split query into words.
- **Slash command errors were silent** — added `@bot.tree.error` handler.
- **Vision model retry bailed early** — fixed to always continue to next candidate.
- **`_client()` used stale API key** — Groq client now re-reads from env on every call.
- **Image entries saved with no description had no warning** — `saveimage` now always auto-generates `appearance_description` via vision model and tells the user to run `/editdesc` or `/editappearance`.
- **KB image description split** — image entries now have two separate fields: `appearance_description` (hidden, Flux-ready, auto-generated by vision model, used for image generation) and `display_description` (user-facing lore/background, shown in chat context and KB UI). `/setdesc` replaced by `/editdesc` (display) and `/editappearance` (appearance).
