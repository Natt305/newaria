# AriaBot scene-image A/B test harness

`scene_image_ab_test.py` sweeps a matrix of Qwen-Image-Edit-Rapid GGUF
workflow variants against your reference photo(s) and writes an HTML
contact sheet so you can pick — by eye — the variant that best preserves
the reference character's appearance, build, and art style.

## Why

Scene images sometimes drift from the reference photo (wrong art style,
masculinized build, lost portrait body). The bot exposes those decisions
as env vars, but tuning them by trial-and-error against a live Discord
channel is slow. This harness lets you sweep them in one batch run.

## Prerequisites

- ComfyUI is running (default `http://127.0.0.1:8188`) with the same
  GGUF / VAE / CLIP files the bot is configured to use.
- Python deps already installed for AriaBot (`Pillow`, `requests`).

Set the same env vars the bot needs:

```bash
export COMFYUI_URL=http://127.0.0.1:8188
export COMFYUI_GGUF=qwen-image-edit-rapid-aio-Q5_K_S.gguf
export COMFYUI_VAE=qwen_image_vae.safetensors
export COMFYUI_CLIP=qwen_2.5_vl_7b-q4_k_m.gguf
```

## Run a sweep

```bash
python tools/scene_image_ab_test.py \
    --reference attached_assets/圖片_1777302758524.png \
    --prompt   "Aria standing in a sunlit park, smiling at the camera" \
    --variants A,B,C,D,E,F,G,H,I,J \
    --seeds    1,2 \
    --out-dir  ab_output/
```

`--refs` and `--out` are accepted as aliases of `--reference` and `--out-dir`.

Then open `ab_output/index.html` in a browser. You'll see:

- The original reference photo(s) at the top.
- One row per variant, with one image per seed in that variant.
- A collapsible `variant cfg & prompt` block under each row showing the
  exact positive/negative prompt and KSampler config that was used.

## Variant matrix

Each row only describes its discriminator vs. the baseline (A) and any
inherited overrides; identical defaults are not repeated.

| ID | What it tests vs. baseline                                                                  |
|----|--------------------------------------------------------------------------------------------|
| A  | Baseline: legacy `flat_anime` style policy, smart-crop refs, 6 steps, CFG 1.0, 1024×768.   |
| B  | A − appearance lock — `style_policy=off`, no style instruction at all.                     |
| C  | `style_policy=match_reference` (the new shipped default).                                  |
| D  | C + append the bot's `looks` identity text after the LLM-rewritten scene prompt.           |
| E  | C + feminine-build positive suffix (`QWEN_FEMININE_BUILD` analogue).                       |
| F  | C + portrait canvas (768×1024) — for refs that are taller than wide.                       |
| G  | C + feminine-build + letterbox preprocess + CFG 2.0 (activates the negative branch).       |
| H  | G + steps=8 + sampler `dpmpp_2m` / scheduler `karras`.                                     |
| I  | G + steps=10 (still on `euler_ancestral` / `beta`).                                        |
| J  | G + force the v2 encoder `latent_image` slot when the ComfyUI server supports it.          |

The harness probes `/object_info/TextEncodeQwenImageEditPlus` once at startup
to detect whether the server exposes the v2 `latent_image` input. If not,
variant J falls back to the v1 graph and prints a warning. Force the
behaviour with `--encoder=v1` or `--encoder=v2`.

`--looks` overrides the identity text used in variant D; if omitted the
harness reads `data/character/profile.json` from the repository root.

You can pass `--variants` with a subset (e.g. `C,D,G`) to skip variants
you've already ruled out, or edit the `_VARIANTS` dict in the script to
add your own row.

## Promote the winner

Once you've picked a variant, set the matching env vars in `tokens.txt`
or your shell:

```ini
QWEN_STYLE_POLICY=match_reference   # or flat_anime / off
QWEN_FEMININE_BUILD=on              # or off
QWEN_REF_PREPROCESS=letterbox       # or crop
QWEN_CFG=1.0                        # raise to activate negative branch
```

Other knobs the bot honours that the harness also exercises:

```ini
QWEN_APPEND_LOOKS=on                # append the bot's `looks` text post-LLM
SCENE_CINEMATIC_SUFFIX=on           # append the cinematic framing tail
```

…then restart the bot. Use `/scenedebug` after the next 🎬 reaction to
confirm the values you expect actually got applied — including the
resolved `ref_preprocess` mode the encoder used.

## Troubleshooting

- **HTTP 400 from /prompt about an unknown node** — your ComfyUI install
  is missing one of the custom-node packs the bot also needs. Run the
  bot once and check the boot log; the same install hints apply.
- **All variants look identical** — your CFG is 1.0 and the negative
  prompt is multiplied by zero, so the negative-only changes won't
  visibly differ from their CFG=1.0 sibling. Compare A↔C↔E↔G for the
  positive-prompt and preprocessing differences instead.
- **Variant J looks like G** — the server is on the v1 encoder; `force_v2`
  fell back gracefully. Upgrade the Qwen-Image-Edit custom-node pack to
  the fork that exposes `latent_image` on `TextEncodeQwenImageEditPlus`.
- **Script can't find `comfyui_ai`** — run from the repository root, e.g.
  `python tools/scene_image_ab_test.py ...` rather than `cd tools && python ...`.
