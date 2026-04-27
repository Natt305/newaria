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
    --refs attached_assets/圖片_1777302758524.png \
    --prompt "Aria standing in a sunlit park, smiling at the camera" \
    --variants A,B,C,D,E,F,G,H,I,J \
    --seeds 1,2 \
    --out ab_output/
```

Then open `ab_output/index.html` in a browser. You'll see:

- The original reference photo(s) at the top.
- One row per variant, with one image per seed in that variant.
- A collapsible `variant cfg & prompt` block under each row showing the
  exact positive/negative prompt and KSampler config that was used.

## Variant matrix

| ID | Style policy   | Feminine bias | Ref preprocess | Steps | CFG | Sampler / scheduler | Canvas    |
|----|----------------|---------------|----------------|-------|-----|---------------------|-----------|
| A  | flat_anime     | off           | crop           | 6     | 1.0 | euler_ancestral / beta | 1024×768 |
| B  | off            | off           | crop           | 6     | 1.0 | euler_ancestral / beta | 1024×768 |
| C  | match_reference| off           | crop           | 6     | 1.0 | euler_ancestral / beta | 1024×768 |
| D  | match_reference| on            | crop           | 6     | 1.0 | euler_ancestral / beta | 1024×768 |
| E  | match_reference| on            | letterbox      | 6     | 1.0 | euler_ancestral / beta | 1024×768 |
| F  | match_reference| off           | letterbox      | 8     | 1.0 | euler_ancestral / beta | 1024×768 |
| G  | match_reference| on            | letterbox      | 6     | 2.0 | euler_ancestral / beta | 1024×768 |
| H  | match_reference| off           | letterbox      | 6     | 1.0 | euler_ancestral / beta | 1024×1024|
| I  | match_reference| off           | letterbox      | 6     | 1.0 | euler / simple        | 1024×768 |
| J  | flat_anime     | on            | letterbox      | 6     | 1.0 | euler_ancestral / beta | 1024×768 |

A is the legacy-bot baseline; C is the new shipped default; G activates the
classifier-free-guidance negative branch (only meaningful when CFG > 1.0).

You can pass `--variants` with a subset (e.g. `C,D,E`) to skip variants
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

Other knobs the bot honours that the harness can also exercise:

```ini
QWEN_APPEND_LOOKS=on                # append the bot's `looks` text post-LLM
SCENE_CINEMATIC_SUFFIX=on           # append the cinematic framing tail
```

…then restart the bot. Use `/scenedebug` after the next 🎬 reaction to
confirm the values you expect actually got applied.

## Troubleshooting

- **HTTP 400 from /prompt about an unknown node** — your ComfyUI install
  is missing one of the custom-node packs the bot also needs. Run the
  bot once and check the boot log; the same install hints apply.
- **All variants look identical** — your CFG is 1.0 and the negative
  prompt is multiplied by zero, so the negative-only changes (G) won't
  visibly differ from their CFG=1.0 sibling. Compare A↔C↔E↔J for the
  positive-prompt and preprocessing differences instead.
- **Script can't find `comfyui_ai`** — run from the repository root, e.g.
  `python tools/scene_image_ab_test.py ...` rather than `cd tools && python ...`.
