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
| `reply_format.py` | Shared chat-formatting helpers (suggestion salvage parser, suggestion-button generator pipeline, Discord post-processor — bold dialogue, italicise narration, strip self-name prefix). Used by `lmstudio_ai.py`, `groq_ai.py`, and `ollama_ai.py` so each fix lands in one place. |
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
| `COMFYUI_ENGINE` | Optional | `qwen` (default) or `flux`. `qwen` runs Phr00t's Qwen-Image-Edit-Rapid AIO GGUF in true edit-mode (model literally sees reference photos); `flux` keeps the legacy FLUX.2 Klein workflows below. |
| `COMFYUI_QWEN_GGUF` | Required for `engine=qwen` | Qwen-Image-Edit-Rapid AIO GGUF filename, e.g. `Qwen-Rapid-NSFW-v23_Q4_K.gguf`, placed in `models/diffusion_models/` (loaded by `UnetLoaderGGUF`). |
| `COMFYUI_QWEN_VAE` | Required for `engine=qwen` | Qwen-Image VAE filename, e.g. `pig_qwen_image_vae_fp32-f16.gguf` in `models/vae/` (loaded by stock `VAELoader`). |
| `COMFYUI_QWEN_CLIP_GGUF` | Required for `engine=qwen` | Qwen 2.5 VL text-encoder GGUF filename, e.g. `Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf` in `models/clip/`. **Must** be paired with `mmproj-f16.gguf` (or `mmproj-Q8_0.gguf`) in the same folder — without mmproj the encoder is text-only and reference photos are silently ignored. |
| `COMFYUI_QWEN_STEPS` | Optional | Qwen sampling steps (default `4` — the Rapid AIO is distilled). The Qwen engine intentionally ignores any `COMFYUI_STEPS` / step overrides coming from the rest of the pipeline (which are tuned for FLUX's ~20 steps) — only this var is honored, so set it explicitly if you want anything other than 4. |
| `COMFYUI_QWEN_SAMPLER` | Optional | Qwen KSampler `sampler_name` (default `euler_ancestral`). |
| `COMFYUI_QWEN_SCHEDULER` | Optional | Qwen KSampler `scheduler` (default `beta`). |
| `COMFYUI_QWEN_WIDTH` | Optional | Qwen output width  (default `1024` — Qwen-Image native scale). |
| `COMFYUI_QWEN_HEIGHT` | Optional | Qwen output height (default `1024`). |
| `COMFYUI_ALLOW_ENGINE_FALLBACK` | Optional | `1`/`true`/`yes`/`on` re-enables the auto qwen→flux fallback when `COMFYUI_ENGINE=qwen` but the Qwen vars are unset. Off by default — the bot fails fast on misconfiguration so the active engine never silently drifts. |
| `COMFYUI_MODE` | Optional (FLUX only) | `multiref` (default, recommended) — spatial-masked multi-character; `refchain` — ReferenceChainConditioning custom node; `ultimate_inpaint` — SAM3 inpaint loop (requires many custom nodes). Ignored when `COMFYUI_ENGINE=qwen`. |
| `COMFYUI_USE_SAM` | Optional (FLUX only) | `true` or `false` (default) — SAM3 auto-segmentation, only relevant for `ultimate_inpaint` mode. |

### ComfyUI Engine: `qwen` (default)

Phr00t's **Qwen-Image-Edit-Rapid AIO** repackaged as a GGUF (e.g. `Qwen-Rapid-NSFW-v23_Q4_K.gguf` from the phil2sat GGUF release).
Selected via `COMFYUI_ENGINE=qwen` (the new default).

**Why this is the new default:** the Qwen 2.5 VL multimodal text encoder lets `TextEncodeQwenImageEditPlus` route up to four reference images through the same encoder that processes the prompt — so the model literally **sees** her looks (character portrait) and any KB photos rather than nudging a noisy latent toward them via classical img2img denoise.

**Workflow stack (built in `comfyui_ai.py`):**

1. `UnetLoaderGGUF` (city96) → MODEL
2. `CLIPLoaderGGUF` (city96, `type=qwen_image`) → CLIP — `mmproj-f16.gguf` must sit alongside the encoder GGUF in `models/clip/`, otherwise the encoder runs in text-only mode and the reference photos are silently dropped.
3. `VAELoader` → VAE (any Qwen-Image VAE; `pig_qwen_image_vae_fp32-f16.gguf` works through the stock loader).
4. Up to 4 `LoadImage` nodes (one per uploaded reference) wired into `image1..image4` slots of `TextEncodeQwenImageEditPlus` (positive). Negative is a plain empty `CLIPTextEncode`.
5. `EmptySD3LatentImage` (16-channel — Qwen-Image is a 16-channel MMDiT model).
6. `KSampler` — `cfg=1.0`, default `steps=4`, sampler `euler_ancestral`, scheduler `beta` (matches Phr00t's reference `Qwen-Rapid-AIO.json`; all overridable via env vars).
7. `VAEDecode` → `SaveImage`.

**Required ComfyUI custom node packs (manual install):**
- `ComfyUI-GGUF` (city96) — provides `UnetLoaderGGUF` and `CLIPLoaderGGUF`.
- Any pack shipping `TextEncodeQwenImageEditPlus` (Phr00t's `nodes_qwen.py` or comparable).

**Backward compatibility:** by default the bot **does not** auto-switch engines when `COMFYUI_ENGINE=qwen` is set but the Qwen vars are missing — it logs a clear WARNING and fails fast, so the active engine can never silently drift to something the user didn't ask for. If you actually want the old "fall back to FLUX" behavior, set `COMFYUI_ALLOW_ENGINE_FALLBACK=1`. To force FLUX explicitly, set `COMFYUI_ENGINE=flux`.

**Troubleshooting:**
- `/prompt` returns a node-not-found error mentioning `TextEncodeQwenImageEditPlus` → the custom node pack isn't installed; restart ComfyUI after installing it.
- Reference images are obviously ignored / output is text-only → `mmproj-*.gguf` is missing from `models/clip/`.
- `/prompt` complains about CLIP type → confirm city96's `CLIPLoaderGGUF` exposes `qwen_image` in its type dropdown (recent versions do).
- `KSampler` complains about an invalid sampler / scheduler → the user-supplied `COMFYUI_QWEN_SAMPLER` / `COMFYUI_QWEN_SCHEDULER` doesn't exist on this ComfyUI build; revert to defaults (`euler_ancestral` / `beta`).
- VRAM-tight machines: drop `COMFYUI_QWEN_WIDTH`/`COMFYUI_QWEN_HEIGHT` to `768` and stay at `COMFYUI_QWEN_STEPS=4`.

### ComfyUI Engine: `flux` (legacy)

Multi-Character Photo-Referencing Modes — only active when `COMFYUI_ENGINE=flux`:

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

**Two-pass contact-pose workflow** (`test_2pass_hug.py`):

For hugging / touching poses FLUX.2 Klein 4B cannot achieve via hard masks alone (no ControlNet for Klein exists):

1. **Pass 1** — Run the proven `ConditioningSetMask` side-by-side workflow → excellent character fidelity baseline image.
2. **Pass 2** (`build_img2img_workflow`)  — VAE-encode the Pass-1 image and feed it as an additional `ReferenceLatent` for **every** character's conditioning chain alongside their photo refs. Run a normal full 4-step schedule from `EmptyFlux2LatentImage` with the hugging scene prompt and plain `ConditioningCombine` (no spatial masks). The layout latent anchors composition; the photo refs anchor appearance; the hugging prompt steers pose.

> Note: Standard img2img via `SplitSigmasDenoise` does **not** work with FLUX.2 Klein — it is a distilled 4-step model that must always run the full sigma schedule. The reference-latent layout-anchor approach is the correct substitute.

Results: contact/hugging pose consistently detected by vision model with recognisable character features preserved.

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
