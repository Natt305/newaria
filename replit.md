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
| `database.py` | File-based storage for character, memories, knowledge, user profiles; SQLite for history |
| `views.py` | Discord UI components (buttons, modals, paginated views); includes `UserProfileView` |
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
  user_profiles/
    {discord_id}/
      images.json         ← Player reference image list (filename, mime, description, thumb)
      images/             ← Actual player reference photos + 512px thumbnails
  settings.json           ← Bot settings (suggestions, memory toggles, etc.)
  status.json             ← Persisted bot presence/status
  history.db              ← SQLite: conversation history + player profiles (internal)
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
| `COMFYUI_PREWARM` | Optional | `1`/`true`/`yes`/`on` to pre-load the active engine's model into VRAM right after `on_ready`; `0`/`false`/`no`/`off` to skip. Unset uses the per-engine default — **on for `qwen`, off for `flux`**. Submits a throwaway 256×256, 1-step txt2img through the same engine-specific builder the real pipeline uses, so the very first `!generate` skips the 30–60 s model-load cost. Runs concurrently with bot startup; failure logs a single WARN line and never blocks. Set to `0` on truly tight VRAM days. |
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
4. Up to 4 `LoadImage` nodes (one per uploaded reference) wired into `image1..image4` slots of `TextEncodeQwenImageEditPlus` (positive) and the matching `TextEncodeQwenImageEditPlus` (negative, node "11"). At the default CFG=1.5 the negative carries anatomy/feminine-build correction text; at CFG=1.0 its prompt is empty.
5. `EmptySD3LatentImage` (16-channel — Qwen-Image is a 16-channel MMDiT model). When Phr00t's **v2** `nodes_qwen.py` is detected (see below), this latent is *also* wired into the encoder's `latent_image` input — the v2 fork uses it to size the reference embeddings against the target canvas, which kills the scaling/cropping/zoom artifacts that plagued v1 multi-ref edits.
6. `KSampler` — `cfg=1.5` (default; raises to activate negative guidance), default `steps=4`, sampler `euler_ancestral`, scheduler `beta` (all overridable via env vars).
7. `VAEDecode` → `SaveImage`.

**Required ComfyUI custom node packs (manual install):**
- `ComfyUI-GGUF` (city96) — provides `UnetLoaderGGUF` and `CLIPLoaderGGUF`.
- Any pack shipping `TextEncodeQwenImageEditPlus`. **Strongly recommended: Phr00t's v2 `nodes_qwen.py`** (rename `nodes_qwen.v2.py` → `nodes_qwen.py` and drop into `<ComfyUI>/comfy_extras/`). The v1 node still works but produces visibly worse multi-reference results — faces crop, full-body refs zoom into torsos.

**Encoder v2 auto-detection.** At boot (and whenever you re-run `/diagcomfyui`) the bot probes `/object_info/TextEncodeQwenImageEditPlus` for a `latent_image` input slot — the marker for Phr00t's v2 fork — and caches the result. The boot console prints exactly one of these lines so you can verify your install at a glance:
- `✅ TextEncodeQwenImageEditPlus v2 — `latent_image` input present (scaling fix active).`
- `⚠ TextEncodeQwenImageEditPlus v1 — no `latent_image` input; falling back to legacy mode (install Phr00t's v2 nodes_qwen.py to fix scaling/cropping/zoom artifacts).`

Each `!generate` then tags its workflow log with `(latent_image: wired)`, `(latent_image: not wired (v1 node))`, or `(latent_image: not wired (unknown))` so you can confirm at a glance which path actually ran. The "unknown" tag means ComfyUI was unreachable when the bot started — re-run `/diagcomfyui` after ComfyUI comes up to populate the cache. The bot **never** sends `latent_image` to a v1 node, so a v1 install can't be poisoned with an unknown input that would 400 the prompt.

**Backward compatibility:** by default the bot **does not** auto-switch engines when `COMFYUI_ENGINE=qwen` is set but the Qwen vars are missing — it logs a clear WARNING and fails fast, so the active engine can never silently drift to something the user didn't ask for. If you actually want the old "fall back to FLUX" behavior, set `COMFYUI_ALLOW_ENGINE_FALLBACK=1`. To force FLUX explicitly, set `COMFYUI_ENGINE=flux`.

**Engine-scoped startup (VRAM hygiene):**

ComfyUI itself does not preload diffusion models, but several custom-node packs do — most notably the FLUX-side Ultimate-Inpaint stack (`SAM3`, `GroundingDINO`, `IPAdapter`, `Impact-Pack` detectors, etc.) which load their own models into VRAM at *import time*, regardless of which engine is actually generating. On a 16 GB GPU this is enough to eat ~12 GB at idle and leave no headroom for the engine you actually selected. The bot solves this by toggling pack folders **on disk** before ComfyUI boots:

- Manifest file `comfyui_engine_packs.txt` (repo root) lists each heavy pack folder with the engine it belongs to (`flux:`, `qwen:`, or `both:`/`shared:`). Packs not listed are left alone (treated as shared / safe).
- `start.bat` reads `COMFYUI_ENGINE` from `tokens.txt` (default `qwen`) and runs `scripts/scope_comfy_packs.py` *before* it launches ComfyUI. The script renames every pack listed under a non-matching engine to `<pack>.disabled` (a suffix recent ComfyUI versions skip during the custom-node scan), and strips that suffix from packs needed by the active engine. Nothing is ever deleted, and nothing under `<COMFYUI_PATH>/models/` is ever touched.
- Switching engines is therefore: **edit `COMFYUI_ENGINE` in `tokens.txt` → close ComfyUI → re-run `start.bat`**. The console prints exactly which packs were toggled before the ComfyUI window opens; if ComfyUI is already running on port 8188 the bat skips both the launch *and* the toggle (and says so) so you don't pull the rug out from under a live process.

Example manifest entries (the shipped file pre-populates the common FLUX-only packs):
```
flux: ComfyUI-SAM3
flux: ComfyUI-GroundingDINO
flux: ComfyUI_IPAdapter_plus
flux: ComfyUI-Impact-Pack
# qwen: comfyui_qwen_image_edit_plus_nodes   # add Qwen-only packs here
```
Adjust the folder names to match what's actually in your `<COMFYUI_PATH>/custom_nodes/` (different installers name folders differently) — the script logs any manifest entries it can't find so typos surface immediately.

**Optional model-path scoping (cosmetic).** If you also want the Manager / KSampler dropdowns to only list the active engine's model files, drop your models into engine-specific subfolders (e.g. `models/qwen/unet/`, `models/flux/unet/`), then **copy** `comfyui_extra_paths.qwen.example.yaml` → `comfyui_extra_paths.qwen.yaml` (and/or the FLUX one), drop the `.example` segment from the filename, and uncomment + edit the engine block inside. `start.bat` only looks for the non-example filename, so until you rename a template the feature stays completely off. When the engine-matching activated yaml exists, `start.bat` passes `--extra-model-paths-config` to ComfyUI. **This is purely cosmetic — it does not free any VRAM**; the pack toggling above is what actually fixes the idle-VRAM problem.

You can verify the toggling took effect from Discord with `/diagcomfyui` — the embed includes a "已停用的自訂節點包" field listing every `.disabled` folder currently in `custom_nodes/`. The boot console also prints the same one-line summary right after the per-node `✅/❌` block.

**Boot-time pre-warm (cuts the first-image wait):**

The first `!generate` of a session normally pays a one-time 30–60 s cost for ComfyUI to load the diffusion model into VRAM. With the engine-scoped pack toggling above freeing up headroom for the *active* engine, the bot can use that headroom proactively: as soon as `on_ready` confirms ComfyUI is reachable and the required custom nodes are present, `comfyui_ai.prewarm()` submits a tiny throwaway 256×256, 1-step txt2img through the same engine-specific workflow builder (`_build_txt2img_workflow_qwen` / `_build_txt2img_workflow`). The image is discarded; only the warm model in VRAM is kept, so the very first real `!generate` returns roughly the cold-start time faster.

- The pre-warm runs **concurrently** with the rest of bot startup (scheduled via `asyncio.create_task` + `asyncio.to_thread`) — it never blocks the bot from coming online, and any failure prints a single `[ComfyUI] WARMUP WARN: …` line and is otherwise ignored.
- The console prints a clear `WARMUP` banner before submitting so the VRAM spike that follows is obviously the warm-up and not a real request.
- Gated by `COMFYUI_PREWARM` (see env-var table). **Default: on for `qwen`, off for `flux`** — Qwen is the new default and benefits most from a hot model, while FLUX runs on tighter-VRAM setups more often. Override either way if you need to.

**VRAM management for the Qwen pipeline:**

The recommended Qwen stack is borderline on a 16 GB GPU but fits cleanly *because the workflow is sequential*, not because everything coexists in VRAM at once:

| Component | VRAM | When it's needed |
|---|---|---|
| Qwen-Rapid-NSFW UNET (Q4_K) | 12.4 GB | `KSampler` step only |
| Qwen2.5-VL-7B CLIP encoder (Q4_K_M) | 4.68 GB | `TextEncodeQwenImageEditPlus` only |
| VAE | ~0.4 GB | `VAEDecode` only |
| **Naive peak (everything resident)** | **~17.4 GB** | ❌ does NOT fit on 16 GB |
| **Sequential peak (CLIP evicted before sampling)** | **~13.0 GB** | ✅ fits cleanly |

Common misconception worth correcting: **`Qwen2.5-VL-7B` is not a separate "vision analysis" stage that runs before image generation**. It IS the text encoder our workflow uses — `CLIPLoaderGGUF` loads it with `type=qwen_image` and `TextEncodeQwenImageEditPlus` pipes both your prompt *and* up to 4 reference photos through it to produce the conditioning. Every single image generation goes through both the encoder and the sampler, in sequence, every time. ComfyUI's built-in memory manager handles the eviction automatically — once the encoder finishes its job, it's paged out of VRAM before `KSampler` loads the UNET.

The catch is that ComfyUI's default `auto` heuristic is sometimes too greedy on borderline cards (it tries to keep everything resident "just in case"), causing OOM crashes on cards that would otherwise be fine. **`start.bat` handles this for you** by reading the GPU's actual VRAM via `nvidia-smi` and passing the matching ComfyUI flag at launch — no env var, no `tokens.txt` entry needed:

The conceptual mapping is the standard ComfyUI four-row table:

| Card capacity | Flag | Why |
|---|---|---|
| ≥ 24 GB | `--highvram` | Keep everything cached for fastest steady-state |
| 12–23 GB | `--normalvram` | Aggressive eviction, no UNET splitting |
| 8–11 GB | `--normalvram` | Same as above |
| < 8 GB | `--lowvram` | Stream UNET blocks to avoid OOM; ~2–3× slower per image |
| Detection failed (no `nvidia-smi`, AMD/Intel GPU) | *(no flag)* | Falls back to ComfyUI's built-in `auto`; one WARN line printed |

In `start.bat` itself the integer-GiB cutoffs are written as `>= 23` and `>= 7` (not `>= 24` and `>= 8`) to compensate for `nvidia-smi`'s slight under-report — a "16 GB" card reports ~16380 MiB which integer-divides to 15 GiB, a "24 GB" card reports ~24564 MiB → 23 GiB, an "8 GB" card reports ~8188 MiB → 7 GiB. The lowered boundaries map those rounded-down values back to the correct conceptual buckets above. The block-leading comment in `start.bat` explains this calibration in detail.

The `start.bat` console banner shows exactly what was detected and passed, e.g. `[ComfyUI] Detected GPU VRAM: 16380 MB (bucket=15 GiB) -> --normalvram` — handy for debugging boundary cards. No user-facing override env var ships; if a power user really needs to force a different mode, edit the if-chain in `start.bat` directly.

**Realistic per-image timing on a 16 GB card with this setup:** ~15–25 s for image #1 cold, ~8–14 s steady-state once the models are cached in system RAM. With the boot-time pre-warm above (default-on for qwen), the cold path is moved to `start.bat` time and the user only ever sees the steady-state number. About 2–4 s of each generation is the CLIP→UNET swap (RAM↔VRAM at PCIe-4.0 speeds); the rest is the actual 4-step sampling + decode.

You can confirm the active VRAM headroom from Discord with `/diagcomfyui` — the embed includes a "顯存" field showing `剩餘 N MB / 共 N MB` straight from ComfyUI's `/system_stats`.

**Troubleshooting:**
- **First step for any "node-not-found 400" error: run `/diagcomfyui`.** The bot also runs the same probe automatically at startup (when `IMAGE_BACKEND=comfyui`) via `comfyui_ai.diagnose()` — it hits `/object_info` once for each required custom node and prints `✅ found / ❌ MISSING` per node, plus the `CLIPLoaderGGUF` `type` choices so you can verify `qwen_image` is one of them, plus the count and names of any engine-scoped packs currently disabled (see above). If anything is missing it logs a single prominent `[WARNING]` line telling you which custom-node pack to install (the bot still starts — the FLUX engine may still be usable). Re-run `/diagcomfyui` in any channel after installing/updating a pack and restarting ComfyUI.
- `/prompt` returns a node-not-found error mentioning `TextEncodeQwenImageEditPlus` → the custom node pack isn't installed; restart ComfyUI after installing it.
- Reference images are obviously ignored / output is text-only → `mmproj-*.gguf` is missing from `models/clip/`.
- `/prompt` complains about CLIP type → confirm city96's `CLIPLoaderGGUF` exposes `qwen_image` in its type dropdown (recent versions do).
- `KSampler` complains about an invalid sampler / scheduler → the user-supplied `COMFYUI_QWEN_SAMPLER` / `COMFYUI_QWEN_SCHEDULER` doesn't exist on this ComfyUI build; revert to defaults (`euler_ancestral` / `beta`).
- VRAM-tight machines: drop `COMFYUI_QWEN_WIDTH`/`COMFYUI_QWEN_HEIGHT` to `768` and stay at `COMFYUI_QWEN_STEPS=4`. **Most importantly, audit `comfyui_engine_packs.txt`** so every heavy FLUX-only pack you actually have installed is listed under `flux:` — the shipped defaults cover the common Ultimate-Inpaint stack, but unique installs may have other VRAM-hungry packs.
- **OOM during sampling** → check the `[ComfyUI] Detected GPU VRAM: N MB -> <flag>` line in your `start.bat` console output to confirm the auto-picked flag matches your card. If it didn't detect (you see the "ComfyUI will use its built-in auto" WARN line), `nvidia-smi` is either missing from PATH or your GPU isn't reporting; add the right `--lowvram` / `--normalvram` flag manually to the `start "ComfyUI" ...` line in `start.bat`. See the "VRAM management for the Qwen pipeline" subsection above for the math.
- A FLUX node that *should* be present is reported missing after switching to `flux` → check that the corresponding pack isn't still listed `flux:` in the manifest but somehow got missed during re-enable; `/diagcomfyui`'s "已停用的自訂節點包" field will show any leftover `.disabled` folders you can rename back manually.

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

## Scene image fidelity knobs (Qwen-Image-Edit-Rapid GGUF)

Scene-image generation drifts toward generic male proportions and away from
the reference photo's art style on some characters. The fix is purely
configuration — every knob has a sensible default that ships with the bot,
and there is no per-character hardcoding. Disable any of them per env var
when running a non-feminine character or a different art-style reference.

| Env var | Default | Effect |
|---|---|---|
| `QWEN_STYLE_POLICY` | `match_reference` | Style instruction prepended to every Qwen edit-mode prompt. `match_reference` (new default) replicates the reference's rendering, gloss, line weight, and palette verbatim. `flat_anime` is the legacy override that forces flat 2D anime regardless of the reference. `off` adds no style instruction. |
| `QWEN_FEMININE_BUILD` | `on` | Append a slim/feminine build positive suffix to every prompt and a matching negative suffix to the negative CLIPTextEncode. The negative branch is multiplied by zero at CFG=1.0, so the negative only "bites" once `QWEN_CFG > 1.0`. |
| `QWEN_REF_PREPROCESS` | `letterbox` | How v1-encoder installs resize portrait references onto the landscape canvas before upload. `letterbox` (new default) preserves the entire reference; `crop` is the legacy smart-crop that can lop off the bottom of a portrait body. v2 encoders ignore this — they handle sizing internally. |
| `QWEN_APPEND_LOOKS` | `on` | After the LLM rewrites the scene prompt, append `data/character/profile.json`'s `looks` field verbatim as a non-LLM identity anchor. Only fires when there is at least one reference photo and the engine is `qwen`. |
| `QWEN_CFG` | `1.5` | KSampler CFG. Default 1.5 activates the negative branch (anatomy/feminine-build text in node "11"), reducing broken anatomy and masculine proportions. Set to `1.0` to restore the original distilled Rapid AIO behaviour where the negative branch is mathematically zeroed. |
| `SCENE_CINEMATIC_SUFFIX` | `on` | Append the medium-close cinematic framing tail. Disable for less directed scene composition. |

Run `/scenedebug` after the next 🎬 reaction to see exactly which policy,
suffix, ref preprocessing, and CFG actually applied — the command dumps
the most recent `scene_image` and `comfyui_ai` capture snapshots
(ephemeral, role-gated).

To pick the right values empirically without guessing, run the A/B test
harness once against your character's reference photo:

```bash
python tools/scene_image_ab_test.py \
    --reference path/to/character_ref.png \
    --prompt   "Aria standing in a sunlit park, smiling at the camera" \
    --variants A,B,C,D,E,F,G,H,I,J --seeds 1,2 --out-dir ab_output/
```

Then open `ab_output/index.html` in a browser, pick the column that best
preserves the reference, and set the matching env vars in `tokens.txt`.
The harness probes the ComfyUI server's encoder version and tags variant J
accordingly (`--encoder=auto|v1|v2`); `--looks` overrides the identity
text used for variant D (auto-read from `data/character/profile.json`).
See `tools/README.md` for the full variant matrix.

## Scene image (RP mode — LM Studio)

A first-class affordance for cinematic moments in LM Studio roleplay. Off
by default; opt in per channel via `/sceneimage on`.

**What it does**

When scene mode is on for a channel, every LM Studio bot reply gets a
single 🎬 button. Click it and the bot generates a character-accurate
scene image and **edits** it onto its own message in place — no separate
reply, no doubled response. The persistent view registers in `on_ready`
via `bot.add_view(SceneImageButtonView())`, so the button keeps working
on prior bot messages even after a restart.

**Auto-trigger via `[SCENE]` / `[SCENE: ...]`**

The LM Studio system prompt teaches the model three marker forms it may emit
at the end of a paragraph that is genuinely cinematic:

- **Bare `[SCENE]`** — the bot derives the image prompt from the reply
  prose itself. Use when the prose already paints the picture and every
  character is referred to by full name.
- **`[SCENE: short cinematic description]`** — the body is taken
  **verbatim** as the image-prompt seed; the bot prose is skipped. Use
  when the model has a clearer picture in its head than the prose conveys.
  Characters should be spelled out by full name so KB photo matching can
  resolve them.
- **`[SCENE: short cinematic description | with: Name A, Name B]`** — the
  `| with:` tail explicitly pins which KB subjects' reference photos to
  use regardless of how the prose names them. The model is now instructed
  to emit this form automatically whenever a paragraph refers to a KB
  subject only by pronoun or short form (e.g. *she*, *him*, *Saki-chan*),
  when multiple subjects appear and both should be pinned, or when names
  swap mid-paragraph. Names in the `with:` list must match KB entry titles
  exactly (case-insensitive on the parser side; examples use KB title
  casing). Example the model is taught: `[SCENE: she leans close, eyes
  bright with quiet resolve | with: Saki Nikaido]`.

`lmstudio_ai.chat()` strips whichever marker fired and returns
`(…, wants_scene_image, scene_prompt)` as the 5th and 6th tuple elements.
`scene_prompt` is the `[SCENE: ...]` body (or `None` for bare `[SCENE]`).
`process_chat` then auto-runs the same runner the button uses, passing
`scene_prompt` through as `seed_override`.

**Seed source precedence (in `scene_image.run_scene_image`)**

1. `seed_override` (the `[SCENE: ...]` body) — verbatim, no prose mixing.
2. `hint_prompt` + bot-message prose, joined with double-space.
3. Generic fallback `"cinematic scene"`.

The 🎬 button click flow, the legacy `[IMAGE: ...]` re-route, and the
visual-intent re-route all leave `seed_override=None` and so keep the
existing prose-derived behaviour.

**No-duplicates guarantee — the central design rule**

When scene mode is on for a channel **and** the active backend is LM Studio,
`process_chat` skips the legacy `[IMAGE: ...] → reply.send(file)` branch
entirely. Instead it routes through `scene_image.run_scene_image` if any of
these are true on the same turn:

- `wants_scene_image` (LLM emitted `[SCENE]`)
- `image_prompt` from a legacy `[IMAGE: ...]` tag (re-routed as a hint)
- `scene_image.is_user_visual_intent(user_text)` matched

The runner has its own `(channel_id, target_message_id)` in-flight set, so
even when both `[SCENE]` and `[IMAGE: ...]` come back on the same turn,
**exactly one image is produced** — and it always edits the bot's own
message rather than sending a second one.

When scene mode is OFF (or the backend is not LM Studio), the legacy
`[IMAGE: ...]` flow runs unchanged.

**Engine tiers** (in `scene_image.run_scene_image`)

- **Qwen** (primary): full multi-ref edit through `_build_multi_edit_workflow_qwen`,
  up to 4 refs (character portrait first, then KB photos whose titles or
  aliases overlap the seed text — see *KB photo matching* below), with a
  cinematic suffix appended to the prompt (`, cinematic composition,
  soft rim light, shallow depth of field, film grain`). Pairs with the
  Qwen v2 `latent_image` scaling fix described above so the bot's looks
  survive multi-reference editing.
- **FLUX**: single-ref multiref using only the character portrait, no KB
  interleaving, same cinematic suffix.
- **Cloudflare**: plain text-to-image with the cinematic suffix; no refs.

**KB photo matching** (`scene_image._gather_refs`)

The character (bot) portrait is always loaded first and is never gated on
the seed text. Remaining ref slots (cap = 4) come from KB image entries.

Two layers decide whether a KB entry joins the reference set:

1. **Explicit override — `[SCENE: body | with: a, b, c]`.** When the model
   emits a `with:` clause as the tail of a `[SCENE: ...]` body, the names
   are matched against KB titles + aliases via strict case-insensitive
   equality. *No* substring guessing — a misspelt name silently drops
   rather than pulling in an unrelated entry. The cleaned body (with the
   `with:` tail stripped) is then used as the prompt seed.
2. **Fuzzy seed match.** Each remaining KB entry's title and aliases are
   compared against the seed text via two rules combined: word-bounded
   substring matching (so *"Saki"* matches "Saki, smiling" but never
   "ksaki"; CJK falls back to plain substring so *"妹妹"* still matches
   *"妹妹來了"*) and non-trivial-token overlap (so *"the tower"* matches
   `Tokyo Tower` via the `tower` token, but stop-words like *the / she /
   you* never win on their own).

Each KB entry contributes at most 2 photos and is added at most once. The
matcher is fully synchronous and dependency-free of network calls.

**Aliases — opt-in additive field on KB image entries**

Each image entry can carry an optional `aliases: List[str]` field, edited
via the 🏷️ **別名** button on the entry view (or programmatically via
`database.update_aliases`). Older entries with no `aliases` field continue
to load and save unchanged — they simply fall back to title-only matching.
Aliases are the recommended fix for the *"reply uses a pronoun and the KB
photo drops"* failure mode: add the character's nickname / romanisation /
short-form so the fuzzy matcher catches the next mention.

*Worked example:* KB entry `Saki Nikaido` with no aliases. The bot replies
*"she smiles at you, eyes warm"* — neither *Saki* nor *Nikaido* appears,
so the photo would silently drop and Qwen would invent a face. Either:

- add aliases like `Saki, Saki-chan` (the model usually retains *Saki*
  somewhere nearby), or
- have the model emit `[SCENE: a quiet smile in lamplight | with: Saki Nikaido]`
  to explicitly carry the subject through.

**Resolved-refs footer**

The progress message that already replies under each scene-image
generation gets a one-line *"refs: bot, Saki Nikaido, Tokyo Tower"*
footer **appended** under its final stage line (truncated to ~80 chars
for the footer itself). Operators can see at a glance both the stage the
generation finished at and which photos actually made it into the generation, so a
silently-dropped subject is immediately visible and can be fixed by
rewording the prompt, adding an alias, or using `with: ...`.

**Shared progress UX**

`scene_image.progress_bar(message, name, formatter)` is the reusable async
context manager that the legacy `[IMAGE: ...]` flow's progress block was
factored into. Same temporary-message → live-edit → delete pattern, same
`bot._format_diffuser_progress` formatter, no duplicated implementation.

**Cooldown & dedup**

- Per-channel cooldown via `commands.CooldownMapping` (3 generations / 60s)
- In-flight `set[(channel_id, target_message_id)]` rejects spam-clicks and
  dual-trigger races with a friendly ephemeral.

**Operator-facing surface**

- `/sceneimage on|off|status` (hybrid command, per-channel, persists in
  `data/settings.json` under `scene_image:{channel_id}`)
- `/diagcomfyui` now includes a "場景圖片模式 (此頻道)" field showing the
  current channel's state alongside the ComfyUI node diagnosis.
- Boot log line: `[SceneImage] Persistent view registered — buttons on bot
  messages restored across restart.`

**Files**

- `scene_image.py` — runner, progress-bar context manager, in-flight set,
  cooldown mapping, visual-intent detector, per-channel toggle helpers.
- `views.py` — `SceneImageButtonView` (persistent, fixed `custom_id`).
- `lmstudio_ai.py` — `_SCENE_MARKER_RE` (matches both `[SCENE]` and
  `[SCENE: ...]`), `[SCENE]` / `[SCENE: ...]` system-prompt directive in
  `_roleplay_format_directive`, `chat()` returns 6-tuple
  `(…, wants_scene_image, scene_prompt)`.
- `ai_backend.py` — pads non-LM-Studio chat returns to 6-tuple
  `(…, False, None)`.
- `bot.py` — scene-mode routing branch in `process_chat`, `/sceneimage`
  command, `bot.add_view(...)` in `on_ready`.

## Narrative richness dial (LM Studio only)

Controls how much narration wraps dialogue in LM Studio replies. Set in
`tokens.txt` (or as a Replit Secret) — global, applies to all channels.
Has no effect on Groq or Ollama backends.

**`LMSTUDIO_NARRATION_TARGET`** — five levels, all leaning toward more
immersive output:

| Level | Plain-prose models (MN-12B etc.) | Qwen / hauhaucs (soft hint only) |
|---|---|---|
| `terse` | No directive injected | No hint |
| `brief` | 2–3 sentences; one must cover body language or atmosphere | `<subtext>` 1–2 sentences |
| `standard` | 2–3 full paragraphs wrapping every dialogue beat | `<subtext>` 2–4 sentences |
| **`rich`** *(default for MN-12B)* | **4–5 immersive paragraphs: action, sensory detail, internal thought; 4-paragraph example** | `<subtext>` 4–6 sentences |
| `cinematic` | 6–10 literary paragraphs: extended internal monologue, environmental atmosphere, slow-burn tempo; 5-paragraph example | `<subtext>` 6–10 sentences |

**Default**: `rich` for plain-prose models (MN-12B Celeste etc.), `standard`
for Qwen/hauhaucs.

**Upward shift from the old 3-level scale**: the baseline has been deliberately
pushed higher — what was "rich" (2–3 paragraphs) is now the *floor* of
`standard`. If previous output felt verbose, drop to `brief` or `standard`;
if you want the old behaviour, `standard` is the closest match.

**Qwen caveat**: the plain-prose directive is not injected on the Qwen path
(those models use their own `<reply>/<subtext>` ChatML format). The soft hint
influences `<subtext>` length at the margin only — the model's fine-tune
training dominates. `terse` suppresses the hint entirely.

## Running

The workflow `Start application` runs `python3 launcher.py`.

## Known Bugs Fixed

- **`search_knowledge()` always returned empty results** — fixed to split query into words.
- **Slash command errors were silent** — added `@bot.tree.error` handler.
- **Vision model retry bailed early** — fixed to always continue to next candidate.
- **`_client()` used stale API key** — Groq client now re-reads from env on every call.
- **Image entries saved with no description had no warning** — `saveimage` now always auto-generates `appearance_description` via vision model and tells the user to run `/editdesc` or `/editappearance`.
- **KB image description split** — image entries now have two separate fields: `appearance_description` (hidden, Flux-ready, auto-generated by vision model, used for image generation) and `display_description` (user-facing lore/background, shown in chat context and KB UI). `/setdesc` replaced by `/editdesc` (display) and `/editappearance` (appearance).
