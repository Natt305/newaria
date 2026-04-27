"""End-to-end A/B test harness for AriaBot scene-image fidelity.

Sweeps a configurable matrix of Qwen-Image-Edit-Rapid GGUF workflow variants
against one or more reference photos, saves every output PNG, and writes an
HTML contact sheet so the operator can pick the variant that best preserves
the reference character's appearance, build, and art style.

Designed to be run on the same machine that hosts ComfyUI (so reference
photos travel over localhost) and to depend on ONLY:

    - Pillow              (already a bot dep)
    - requests            (already a bot dep)
    - comfyui_ai          (re-uses _to_png / _upload_image / _submit_and_poll
                           and the new _letterbox_reference_image helper)

It does NOT import the Discord bot, the database, or LM Studio — every
prompt is supplied on the CLI or via --prompt-file. It does NOT modify
the bot's runtime state or settings — every variant builds its own
workflow JSON in-process and submits it directly to /prompt.

Usage
-----

  # 1. Set ComfyUI env vars the same way you set them for the bot:
  set COMFYUI_URL=http://127.0.0.1:8188
  set COMFYUI_GGUF=qwen-image-edit-rapid-aio-Q5_K_S.gguf
  set COMFYUI_VAE=qwen_image_vae.safetensors
  set COMFYUI_CLIP=qwen_2.5_vl_7b-q4_k_m.gguf

  # 2. Run the sweep against your reference photo(s):
  python tools/scene_image_ab_test.py ^
      --refs attached_assets/圖片_1777302758524.png ^
      --prompt "Aria standing in a sunlit park, smiling at the camera" ^
      --variants A,B,C,D,E,F,G,H,I,J ^
      --seeds 1,2,3 ^
      --out ab_output/

  # 3. Open ab_output/index.html in a browser, compare side-by-side,
  #    pick the column that best resembles the reference, then set the
  #    matching env vars in tokens.txt.

The variant matrix mirrors the production knobs added to comfyui_ai.py
in the scene-image-fidelity fix:

  A  baseline (legacy flat-anime override, smart-crop refs, 6 steps)
  B  no appearance-lock prefix at all
  C  match-reference style policy (NEW DEFAULT)
  D  match-reference + feminine build positive suffix
  E  match-reference + feminine build + letterbox refs
  F  match-reference + letterbox refs + 8 steps
  G  match-reference + letterbox refs + CFG 2.0 (negative branch active)
  H  match-reference + letterbox refs + 1024x1024 (square canvas)
  I  match-reference + letterbox refs + euler / simple sampler
  J  flat_anime + feminine build + letterbox refs (legacy + new bias)

Pick the variant id whose output you like best. Then set:

  QWEN_STYLE_POLICY=match_reference   (or flat_anime / off)
  QWEN_FEMININE_BUILD=on              (or off)
  QWEN_REF_PREPROCESS=letterbox       (or crop)
  QWEN_CFG=1.0                        (raise to activate negative branch)

…and restart the bot.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running from anywhere; the harness lives in tools/ but imports the
# top-level comfyui_ai module from the project root.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests  # noqa: E402

import comfyui_ai  # noqa: E402  — re-use upload + submit + helpers

# Same anatomy + feminine + style constants the bot uses, imported by name
# so any future tweak to the bot's defaults is picked up here too.
_ANATOMY_SUFFIX = comfyui_ai._ANATOMY_SUFFIX
_ANATOMY_NEGATIVE = comfyui_ai._ANATOMY_NEGATIVE
_FEMININE_BUILD_SUFFIX = comfyui_ai._FEMININE_BUILD_SUFFIX
_FEMININE_BUILD_NEGATIVE = comfyui_ai._FEMININE_BUILD_NEGATIVE
_STYLE_POLICY_TEXT = comfyui_ai._STYLE_POLICY_TEXT


# ── Variant matrix ────────────────────────────────────────────────────────────
#
# Each variant is a dict of overrides. Anything not specified inherits the
# baseline defaults below. The variants cover the design space the user
# wants to A/B-test in the scene-image-fidelity fix; you can add your own
# rows here and re-run the harness.
#
# Note: the harness sends a single TextEncodeQwenImageEditPlus call per
# generation and never wires the v2 `latent_image` slot — this matches
# the v1 install path so the matrix stays comparable.
_BASELINE = {
    "style_policy": "flat_anime",
    "feminine_build": False,
    "ref_preprocess": "crop",
    "anatomy_suffix": True,
    "negative_text": _ANATOMY_NEGATIVE,
    "width": 1024,
    "height": 768,
    "steps": 6,
    "sampler": "euler_ancestral",
    "scheduler": "beta",
    "cfg": 1.0,
}

_VARIANTS = {
    "A": {
        "_label": "baseline (flat-anime, crop refs, 6 steps)",
    },
    "B": {
        "_label": "no appearance-lock prefix",
        "style_policy": "off",
    },
    "C": {
        "_label": "match-reference style policy (new default)",
        "style_policy": "match_reference",
    },
    "D": {
        "_label": "match-reference + feminine-build positive suffix",
        "style_policy": "match_reference",
        "feminine_build": True,
    },
    "E": {
        "_label": "match-reference + feminine + letterbox refs",
        "style_policy": "match_reference",
        "feminine_build": True,
        "ref_preprocess": "letterbox",
    },
    "F": {
        "_label": "match-reference + letterbox + 8 steps",
        "style_policy": "match_reference",
        "ref_preprocess": "letterbox",
        "steps": 8,
    },
    "G": {
        "_label": "match-reference + letterbox + CFG 2.0 (active negative)",
        "style_policy": "match_reference",
        "feminine_build": True,
        "ref_preprocess": "letterbox",
        "cfg": 2.0,
    },
    "H": {
        "_label": "match-reference + letterbox + 1024x1024 square",
        "style_policy": "match_reference",
        "ref_preprocess": "letterbox",
        "width": 1024,
        "height": 1024,
    },
    "I": {
        "_label": "match-reference + letterbox + euler/simple sampler",
        "style_policy": "match_reference",
        "ref_preprocess": "letterbox",
        "sampler": "euler",
        "scheduler": "simple",
    },
    "J": {
        "_label": "flat_anime + feminine + letterbox (legacy override + bias)",
        "style_policy": "flat_anime",
        "feminine_build": True,
        "ref_preprocess": "letterbox",
    },
}


def _merge(variant_id: str) -> dict:
    cfg = dict(_BASELINE)
    overrides = _VARIANTS[variant_id]
    for k, v in overrides.items():
        if k.startswith("_"):
            continue
        cfg[k] = v
    cfg["_label"] = overrides.get("_label", "")
    cfg["_id"] = variant_id
    return cfg


# ── Workflow builder ─────────────────────────────────────────────────────────


def _build_workflow(
    cfg: dict,
    prompt: str,
    ref_names: List[str],
    seed: int,
    gguf: str,
    vae: str,
    clip: str,
) -> dict:
    """Build the ComfyUI API workflow JSON for one variant + one seed.

    Mirrors comfyui_ai._build_multi_edit_workflow_qwen but every parameter is
    plugged in from `cfg` so the matrix stays explicit.
    """
    n = max(1, min(len(ref_names), 4))

    style_text = _STYLE_POLICY_TEXT.get(cfg["style_policy"], "")
    suffix = _ANATOMY_SUFFIX if cfg.get("anatomy_suffix", True) else ""
    feminine_pos = _FEMININE_BUILD_SUFFIX if cfg.get("feminine_build") else ""
    enhanced_prompt = style_text + prompt + suffix + feminine_pos

    neg_parts = [_ANATOMY_NEGATIVE]
    if cfg.get("feminine_build"):
        neg_parts.append(_FEMININE_BUILD_NEGATIVE)
    negative_text = ", ".join(p for p in neg_parts if p)

    wf: dict = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": gguf},
        },
        "2": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": clip, "type": "qwen_image"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": negative_text},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": cfg["width"], "height": cfg["height"], "batch_size": 1},
        },
    }

    encoder_inputs: dict = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "prompt": enhanced_prompt,
    }
    for i in range(n):
        load_id = str(200 + i)
        wf[load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": ref_names[i]},
        }
        encoder_inputs[f"image{i + 1}"] = [load_id, 0]
    wf["10"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": encoder_inputs,
    }
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["10", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": seed,
            "steps": cfg["steps"],
            "cfg": cfg["cfg"],
            "sampler_name": cfg["sampler"],
            "scheduler": cfg["scheduler"],
            "denoise": 1.0,
        },
    }
    wf["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
    }
    wf["9"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0], "filename_prefix": f"abtest_{cfg['_id']}_"},
    }
    return wf, enhanced_prompt, negative_text


# ── Reference upload ─────────────────────────────────────────────────────────


def _load_ref_bytes(path: Path) -> Tuple[bytes, str]:
    raw = path.read_bytes()
    ext = path.suffix.lower().lstrip(".")
    mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(ext, "image/png")
    return raw, mime


def _prepare_ref(
    raw: bytes,
    mime: str,
    mode: str,
    width: int,
    height: int,
) -> bytes:
    """Apply letterbox / smart-crop / passthrough preprocessing."""
    if mode == "letterbox":
        out = comfyui_ai._letterbox_reference_image(raw, mime, width, height)
        return out or raw
    if mode == "crop":
        out = comfyui_ai._preprocess_reference_image(raw, mime, width, height)
        return out or raw
    out = comfyui_ai._to_png(raw)
    return out or raw


# ── Main loop ────────────────────────────────────────────────────────────────


def _parse_csv(s: Optional[str], default: List[str]) -> List[str]:
    if not s:
        return list(default)
    return [p.strip() for p in s.split(",") if p.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refs", required=True,
                   help="Comma-separated paths to reference images (one or more).")
    p.add_argument("--prompt", default=None,
                   help="Scene prompt to test. Use --prompt-file for long prompts.")
    p.add_argument("--prompt-file", default=None,
                   help="Path to a text file containing the scene prompt (overrides --prompt).")
    p.add_argument("--variants", default=",".join(_VARIANTS.keys()),
                   help=f"Comma-separated variant ids. Default: all of {sorted(_VARIANTS)}.")
    p.add_argument("--seeds", default="1,2",
                   help="Comma-separated integer seeds. Default: 1,2.")
    p.add_argument("--out", default="ab_output",
                   help="Output directory for PNGs and index.html. Default: ab_output/")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-job ComfyUI poll timeout in seconds. Default: 600.")
    args = p.parse_args()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt = args.prompt.strip()
    else:
        print("error: --prompt or --prompt-file is required", file=sys.stderr)
        return 2

    base_url = (os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188") or "").rstrip("/")
    gguf = (os.environ.get("COMFYUI_GGUF", "") or "").strip()
    vae = (os.environ.get("COMFYUI_VAE", "") or "").strip()
    clip = (os.environ.get("COMFYUI_CLIP", "") or "").strip()
    if not (gguf and vae and clip):
        print("error: COMFYUI_GGUF, COMFYUI_VAE, and COMFYUI_CLIP must be set",
              file=sys.stderr)
        return 2

    variant_ids = [v for v in _parse_csv(args.variants, list(_VARIANTS)) if v in _VARIANTS]
    if not variant_ids:
        print(f"error: no valid variants in --variants={args.variants!r}", file=sys.stderr)
        return 2
    seeds = []
    for s in _parse_csv(args.seeds, ["1", "2"]):
        try:
            seeds.append(int(s))
        except ValueError:
            print(f"warning: skipping non-int seed {s!r}", file=sys.stderr)
    if not seeds:
        print("error: no valid seeds", file=sys.stderr)
        return 2

    ref_paths = [Path(p) for p in _parse_csv(args.refs, [])]
    for rp in ref_paths:
        if not rp.is_file():
            print(f"error: reference not found: {rp}", file=sys.stderr)
            return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ABTest] ComfyUI: {base_url}")
    print(f"[ABTest] gguf={gguf}  vae={vae}  clip={clip}")
    print(f"[ABTest] variants={variant_ids}  seeds={seeds}  refs={[str(p) for p in ref_paths]}")
    print(f"[ABTest] prompt: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")
    print(f"[ABTest] out: {out_dir.resolve()}")

    # Load reference bytes once; we re-prepare per variant if its
    # preprocessing strategy differs.
    raw_refs: List[Tuple[bytes, str]] = [_load_ref_bytes(p) for p in ref_paths]

    client_id = str(uuid.uuid4())
    results: List[dict] = []  # one row per (variant, seed)

    for vid in variant_ids:
        cfg = _merge(vid)
        # Re-prepare and upload references for THIS variant's preprocessing
        # mode + canvas. The same physical reference may produce 2-3 distinct
        # uploads across variants (one per width/height/mode tuple) — that's
        # fine, ComfyUI dedups identical bytes server-side.
        prepared: List[bytes] = [
            _prepare_ref(raw, mime, cfg["ref_preprocess"], cfg["width"], cfg["height"])
            for raw, mime in raw_refs
        ]
        ref_names: List[str] = []
        for img_bytes in prepared:
            name = comfyui_ai._upload_image(base_url, img_bytes, requests)
            if not name:
                print(f"[ABTest] {vid}: upload failed — skipping variant.")
                ref_names = []
                break
            ref_names.append(name)
        if not ref_names:
            continue

        for seed in seeds:
            wf, full_prompt, neg = _build_workflow(cfg, prompt, ref_names, seed, gguf, vae, clip)
            label = f"{vid}_seed{seed}"
            t0 = time.time()
            print(f"\n[ABTest] === {label} — {cfg['_label']} ===")
            res = comfyui_ai._submit_and_poll(wf, base_url, client_id, args.timeout, requests)
            elapsed = time.time() - t0
            if not res:
                print(f"[ABTest] {label}: FAILED in {elapsed:.1f}s")
                results.append({
                    "id": vid, "label": cfg["_label"], "seed": seed,
                    "file": None, "elapsed": elapsed,
                    "prompt": full_prompt, "negative": neg, "config": cfg,
                })
                continue
            png_bytes, _mime = res
            fname = f"{label}.png"
            (out_dir / fname).write_bytes(png_bytes)
            print(f"[ABTest] {label}: saved {fname} ({len(png_bytes)} bytes, {elapsed:.1f}s)")
            results.append({
                "id": vid, "label": cfg["_label"], "seed": seed,
                "file": fname, "elapsed": elapsed,
                "prompt": full_prompt, "negative": neg, "config": cfg,
            })

    # JSON dump for downstream scripts.
    (out_dir / "results.json").write_text(
        json.dumps({"prompt": prompt, "results": [
            {**r, "config": {k: v for k, v in r["config"].items() if not k.startswith("_") or k == "_label"}}
            for r in results
        ]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # HTML contact sheet — one column per variant, one row per seed, plus
    # a top row showing the original reference photos for at-a-glance
    # comparison. Style intentionally minimal: no JS, no CDN, fully offline.
    html = _render_html(prompt, ref_paths, variant_ids, seeds, results)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\n[ABTest] Wrote {out_dir / 'index.html'} — open it in a browser.")
    return 0


def _render_html(
    prompt: str,
    ref_paths: List[Path],
    variant_ids: List[str],
    seeds: List[int],
    results: List[dict],
) -> str:
    by_id_seed = {(r["id"], r["seed"]): r for r in results}

    def esc(s: str) -> str:
        return (
            (s or "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    rows = []
    rows.append("<h1>AriaBot scene-image A/B test</h1>")
    rows.append(f"<p><b>Prompt:</b> {esc(prompt)}</p>")

    rows.append("<h2>Reference photo(s)</h2><div class='row'>")
    for rp in ref_paths:
        try:
            rel = os.path.relpath(rp, Path(rows and "." or "."))
        except ValueError:
            rel = str(rp)
        rows.append(f"<div class='cell'><img src='{esc(rel)}'><p>{esc(rp.name)}</p></div>")
    rows.append("</div>")

    rows.append("<h2>Outputs</h2>")
    for vid in variant_ids:
        label = _VARIANTS[vid].get("_label", "")
        rows.append(f"<h3>{esc(vid)} — {esc(label)}</h3>")
        rows.append("<div class='row'>")
        for seed in seeds:
            r = by_id_seed.get((vid, seed))
            if not r:
                rows.append(f"<div class='cell'><div class='missing'>missing</div><p>seed={seed}</p></div>")
                continue
            if not r["file"]:
                rows.append(f"<div class='cell'><div class='missing'>FAILED ({r['elapsed']:.1f}s)</div><p>seed={seed}</p></div>")
                continue
            rows.append(
                f"<div class='cell'>"
                f"<img src='{esc(r['file'])}'>"
                f"<p>seed={seed} · {r['elapsed']:.1f}s</p>"
                f"</div>"
            )
        rows.append("</div>")
        # Show the actual prompt + cfg for this variant once (first seed entry).
        first = by_id_seed.get((vid, seeds[0]))
        if first:
            cfg = first["config"]
            cfg_summary = ", ".join(
                f"{k}={v}" for k, v in cfg.items() if not k.startswith("_")
            )
            rows.append(f"<details><summary>variant cfg & prompt</summary>")
            rows.append(f"<pre>{esc(cfg_summary)}\n\nPOSITIVE:\n{esc(first['prompt'])}\n\nNEGATIVE:\n{esc(first['negative'])}</pre>")
            rows.append("</details>")

    style = """
    <style>
      body { font-family: system-ui, sans-serif; max-width: 1600px; margin: 24px auto; padding: 0 16px; }
      h1 { font-size: 22px; }
      h2 { font-size: 18px; margin-top: 32px; border-bottom: 1px solid #ccc; }
      h3 { font-size: 14px; margin-top: 16px; }
      .row { display: flex; gap: 12px; flex-wrap: wrap; }
      .cell { display: inline-block; }
      .cell img { max-width: 320px; max-height: 320px; border: 1px solid #ddd; }
      .cell p { margin: 4px 0 0; font-size: 12px; color: #555; }
      .missing { width: 320px; height: 240px; display: flex; align-items: center;
                 justify-content: center; background: #fee; color: #800; border: 1px solid #fcc; }
      pre { background: #f6f6f6; padding: 8px; font-size: 12px; overflow-x: auto; }
      details { margin: 8px 0 16px; }
    </style>
    """
    return "<!doctype html><html><head><meta charset='utf-8'><title>AriaBot A/B test</title>" + style + "</head><body>" + "\n".join(rows) + "</body></html>"


if __name__ == "__main__":
    sys.exit(main())
