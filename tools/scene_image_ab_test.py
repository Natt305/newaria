"""End-to-end A/B test harness for AriaBot scene-image fidelity.

Sweeps a configurable matrix of Qwen-Image-Edit-Rapid GGUF workflow variants
against your reference photo(s), saves every output PNG, and writes an HTML
contact sheet so you can pick the variant that best preserves the
reference character's appearance, build, and art style.

Run from the repo root:

  python tools/scene_image_ab_test.py \\
      --reference attached_assets/圖片_1777302758524.png \\
      --prompt   "Aria standing in a sunlit park, smiling at the camera" \\
      --variants A,B,C,D,E,F,G,H,I,J \\
      --seeds    1,2 \\
      --out-dir  ab_output/

Flags `--refs` / `--out` are accepted as aliases. See `tools/README.md`
for the full variant matrix and how to promote a winner.
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

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests  # noqa: E402

import comfyui_ai  # noqa: E402

_ANATOMY_SUFFIX = comfyui_ai._ANATOMY_SUFFIX
_ANATOMY_NEGATIVE = comfyui_ai._ANATOMY_NEGATIVE
_FEMININE_BUILD_SUFFIX = comfyui_ai._FEMININE_BUILD_SUFFIX
_FEMININE_BUILD_NEGATIVE = comfyui_ai._FEMININE_BUILD_NEGATIVE
_STYLE_POLICY_TEXT = comfyui_ai._STYLE_POLICY_TEXT


# Baseline matches the legacy bot path (flat-anime override, smart-crop,
# 6 steps euler_ancestral/beta @ 1024x768, CFG 1.0). Each variant overrides
# only the keys it changes.
_BASELINE = {
    "style_policy": "flat_anime",
    "feminine_build": False,
    "ref_preprocess": "crop",
    "anatomy_suffix": True,
    "append_looks": False,
    "force_v2": False,
    "width": 1024,
    "height": 768,
    "steps": 6,
    "sampler": "euler_ancestral",
    "scheduler": "beta",
    "cfg": 1.0,
}


# A-J spec — each row is documented with its discriminator vs. its parent.
_VARIANTS = {
    "A": {
        "_label": "baseline (legacy flat_anime + crop + 6 steps)",
    },
    "B": {
        "_label": "A − appearance lock (style_policy=off)",
        "style_policy": "off",
    },
    "C": {
        "_label": "match-reference style policy",
        "style_policy": "match_reference",
    },
    "D": {
        "_label": "C + append looks identity text",
        "style_policy": "match_reference",
        "append_looks": True,
    },
    "E": {
        "_label": "C + feminine-build positive suffix",
        "style_policy": "match_reference",
        "feminine_build": True,
    },
    "F": {
        "_label": "C + portrait canvas (768x1024)",
        "style_policy": "match_reference",
        "width": 768,
        "height": 1024,
    },
    "G": {
        "_label": "match-reference + feminine + letterbox + CFG 2.0",
        "style_policy": "match_reference",
        "feminine_build": True,
        "ref_preprocess": "letterbox",
        "cfg": 2.0,
    },
    "H": {
        "_label": "G + steps=8 + dpmpp_2m / karras",
        "style_policy": "match_reference",
        "feminine_build": True,
        "ref_preprocess": "letterbox",
        "cfg": 2.0,
        "steps": 8,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
    },
    "I": {
        "_label": "G + steps=10 (euler_ancestral / beta)",
        "style_policy": "match_reference",
        "feminine_build": True,
        "ref_preprocess": "letterbox",
        "cfg": 2.0,
        "steps": 10,
    },
    "J": {
        "_label": "G + force v2 encoder latent_image (if supported)",
        "style_policy": "match_reference",
        "feminine_build": True,
        "ref_preprocess": "letterbox",
        "cfg": 2.0,
        "force_v2": True,
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
    looks: str,
    ref_names: List[str],
    seed: int,
    gguf: str,
    vae: str,
    clip: str,
    encoder_v2: bool,
) -> Tuple[dict, str, str]:
    """Build the ComfyUI API workflow JSON for one variant + one seed.

    Mirrors `comfyui_ai._build_multi_edit_workflow_qwen` but every parameter
    is plugged in from `cfg` so the matrix stays explicit. When
    `cfg['force_v2']` is True (variant J), wires the v2 encoder
    `latent_image` slot; otherwise leaves it out so the graph runs on v1.
    `encoder_v2` is the actual server capability (probed once at boot) —
    when False, J falls back to v1 with a warning rather than failing the
    submission with an unknown-input 400.
    """
    n = max(1, min(len(ref_names), 4))

    style_text = _STYLE_POLICY_TEXT.get(cfg["style_policy"], "")
    suffix = _ANATOMY_SUFFIX if cfg.get("anatomy_suffix", True) else ""
    feminine_pos = _FEMININE_BUILD_SUFFIX if cfg.get("feminine_build") else ""
    enhanced_prompt = style_text + prompt + suffix + feminine_pos
    if cfg.get("append_looks") and looks:
        enhanced_prompt += (
            ". Character identity (must be preserved from reference photos): "
            + looks.strip()
        )

    neg_parts = [_ANATOMY_NEGATIVE]
    if cfg.get("feminine_build"):
        neg_parts.append(_FEMININE_BUILD_NEGATIVE)
    negative_text = ", ".join(p for p in neg_parts if p)

    use_v2_latent = bool(cfg.get("force_v2") and encoder_v2)
    if cfg.get("force_v2") and not encoder_v2:
        print(
            f"[ABTest] {cfg['_id']}: force_v2 requested but server encoder is v1 — "
            f"falling back to v1 graph for this variant."
        )

    wf: dict = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": gguf}},
        "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": clip, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": negative_text}},
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
        wf[load_id] = {"class_type": "LoadImage", "inputs": {"image": ref_names[i]}}
        encoder_inputs[f"image{i + 1}"] = [load_id, 0]
    if use_v2_latent:
        encoder_inputs["latent_image"] = ["6", 0]
    wf["10"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": encoder_inputs}
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
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}}
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
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp",
    }.get(ext, "image/png")
    return raw, mime


def _prepare_ref(raw: bytes, mime: str, mode: str, width: int, height: int) -> bytes:
    if mode == "letterbox":
        out = comfyui_ai._letterbox_reference_image(raw, mime, width, height)
        return out or raw
    if mode == "crop":
        out = comfyui_ai._preprocess_reference_image(raw, mime, width, height)
        return out or raw
    out = comfyui_ai._to_png(raw)
    return out or raw


# ── Encoder probe ────────────────────────────────────────────────────────────

def _probe_encoder_v2(base_url: str) -> bool:
    """Best-effort: return True iff /object_info advertises latent_image on
    TextEncodeQwenImageEditPlus. Defaults to False on any failure."""
    try:
        resp = requests.get(
            f"{base_url}/object_info/TextEncodeQwenImageEditPlus", timeout=10
        )
        if resp.status_code != 200:
            return False
        return comfyui_ai._has_qwen_encoder_latent_input(resp.json())
    except Exception:
        return False


# ── Looks autodetect ─────────────────────────────────────────────────────────

def _autodetect_looks() -> str:
    """Read data/character/profile.json and return its `looks` field, or ''.

    The harness avoids importing the bot's `database` module so the script
    has no Discord/SQLite dependencies; it just reads the JSON directly
    from the repository's `data/` directory.
    """
    candidates = [
        _ROOT / "data" / "character" / "profile.json",
    ]
    for p in candidates:
        if p.is_file():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                return (obj.get("looks") or "").strip()
            except Exception:
                pass
    return ""


# ── Main loop ────────────────────────────────────────────────────────────────

def _parse_csv(s: Optional[str], default: List[str]) -> List[str]:
    if not s:
        return list(default)
    return [p.strip() for p in s.split(",") if p.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Primary names match the task spec; --refs / --out are accepted aliases.
    p.add_argument("--reference", "--refs", dest="reference", required=True,
                   help="Path to the primary (character) reference image.")
    p.add_argument("--player-reference", dest="player_reference", default=None,
                   help="Path to a second (player) reference image. "
                        "When provided, enables multi-ref mode using the production "
                        "Qwen appearance-lock pipeline instead of the single-ref variant matrix.")
    p.add_argument("--player-name", dest="player_name", default="Player",
                   help="Display name for the player character in multi-ref mode. Default: Player")
    p.add_argument("--char-name", dest="char_name", default=None,
                   help="Display name for the primary character. "
                        "Auto-read from data/character/profile.json if omitted.")
    p.add_argument("--prompt", default=None,
                   help="Scene prompt to test. Use --prompt-file for long prompts.")
    p.add_argument("--prompt-file", default=None,
                   help="Path to a text file containing the scene prompt (overrides --prompt).")
    p.add_argument("--variants", default=",".join(_VARIANTS.keys()),
                   help=f"Comma-separated variant ids. Default: all of {sorted(_VARIANTS)}. "
                        "Ignored in multi-ref mode (--player-reference).")
    p.add_argument("--seeds", default="1,2",
                   help="Comma-separated integer seeds. Default: 1,2.")
    p.add_argument("--out-dir", "--out", dest="out_dir", default="ab_output",
                   help="Output directory for PNGs and index.html. Default: ab_output/")
    p.add_argument("--timeout", type=int, default=600,
                   help="Per-job ComfyUI poll timeout in seconds. Default: 600.")
    p.add_argument("--encoder", choices=("auto", "v1", "v2"), default="auto",
                   help="Encoder version override for variant J's force_v2 path. "
                        "`auto` (default) probes /object_info and picks v2 only when supported.")
    p.add_argument("--looks", default=None,
                   help="Looks/identity text appended in variant D (single-ref) or used as the "
                        "char-0 appearance hint in multi-ref mode. "
                        "Auto-read from data/character/profile.json if omitted.")
    args = p.parse_args()

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    elif args.prompt:
        prompt = args.prompt.strip()
    else:
        print("error: --prompt or --prompt-file is required", file=sys.stderr)
        return 2

    base_url = (os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188") or "").rstrip("/")
    # In multi-ref mode prefer COMFYUI_QWEN_* env vars (used by the production bot);
    # fall back to the shared COMFYUI_* vars so the script works in both configurations.
    gguf = (os.environ.get("COMFYUI_QWEN_GGUF") or os.environ.get("COMFYUI_GGUF", "") or "").strip()
    vae  = (os.environ.get("COMFYUI_QWEN_VAE")  or os.environ.get("COMFYUI_VAE",  "") or "").strip()
    clip = (os.environ.get("COMFYUI_QWEN_CLIP_GGUF") or os.environ.get("COMFYUI_CLIP", "") or "").strip()
    if not (gguf and vae and clip):
        print("error: COMFYUI_QWEN_GGUF (or COMFYUI_GGUF), COMFYUI_QWEN_VAE (or COMFYUI_VAE), "
              "and COMFYUI_QWEN_CLIP_GGUF (or COMFYUI_CLIP) must be set",
              file=sys.stderr)
        return 2

    seeds: List[int] = []
    for s in _parse_csv(args.seeds, ["1", "2"]):
        try:
            seeds.append(int(s))
        except ValueError:
            print(f"warning: skipping non-int seed {s!r}", file=sys.stderr)
    if not seeds:
        print("error: no valid seeds", file=sys.stderr)
        return 2

    char_ref_path = Path(args.reference)
    if not char_ref_path.is_file():
        print(f"error: reference not found: {char_ref_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.encoder == "v1":
        encoder_v2 = False
    elif args.encoder == "v2":
        encoder_v2 = True
    else:
        encoder_v2 = _probe_encoder_v2(base_url)

    looks = args.looks if args.looks is not None else _autodetect_looks()

    # Auto-detect char name from profile.json if not supplied
    char_name = (args.char_name or "").strip()
    if not char_name:
        try:
            _profile_path = _ROOT / "data" / "character" / "profile.json"
            if _profile_path.is_file():
                import json as _json
                _p = _json.loads(_profile_path.read_text(encoding="utf-8"))
                char_name = (_p.get("name") or "").strip() or "Character"
            else:
                char_name = "Character"
        except Exception:
            char_name = "Character"

    # ── Multi-ref mode (--player-reference provided) ──────────────────────────
    # Bypass the single-ref variant matrix and run the production Qwen multi-ref
    # appearance-lock pipeline directly with two reference photos.  Results are
    # saved as multi_seed{N}.png and included in the HTML contact sheet.
    if args.player_reference:
        player_ref_path = Path(args.player_reference)
        if not player_ref_path.is_file():
            print(f"error: player reference not found: {player_ref_path}", file=sys.stderr)
            return 2
        player_name = (args.player_name or "Player").strip()

        print(f"[ABTest] *** MULTI-REF MODE ***")
        print(f"[ABTest] ComfyUI: {base_url}")
        print(f"[ABTest] gguf={gguf}  vae={vae}  clip={clip}")
        print(f"[ABTest] encoder: {'v2' if encoder_v2 else 'v1'} (--encoder={args.encoder})")
        print(f"[ABTest] char={char_name!r}  player={player_name!r}")
        print(f"[ABTest] char_ref={char_ref_path}  player_ref={player_ref_path}")
        print(f"[ABTest] seeds={seeds}")
        print(f"[ABTest] looks: {(looks[:80] + '…') if len(looks) > 80 else (looks or '(none)')}")
        print(f"[ABTest] prompt: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")
        print(f"[ABTest] out: {out_dir.resolve()}")

        char_raw, char_mime  = _load_ref_bytes(char_ref_path)
        player_raw, player_mime = _load_ref_bytes(player_ref_path)
        # Use the default preprocess (letterbox matches production bot default)
        width, height = 1024, 768
        char_prepared   = _prepare_ref(char_raw,   char_mime,   "letterbox", width, height)
        player_prepared = _prepare_ref(player_raw, player_mime, "letterbox", width, height)

        client_id = str(uuid.uuid4())
        multi_results: List[dict] = []

        for seed in seeds:
            char_name_upload = comfyui_ai._upload_image(base_url, char_prepared, requests)
            player_name_upload = comfyui_ai._upload_image(base_url, player_prepared, requests)
            if not char_name_upload or not player_name_upload:
                print(f"[ABTest] multi seed={seed}: upload failed — skipping.")
                multi_results.append({
                    "id": "MULTI", "label": "production multi-ref lock", "seed": seed,
                    "file": None, "elapsed": 0.0, "prompt": prompt, "negative": "",
                    "config": {"_label": "production multi-ref lock"}, "encoder_used": "v2" if encoder_v2 else "v1",
                })
                continue

            subject_appearances = {char_name: looks} if looks else {}
            wf = comfyui_ai._build_multi_edit_workflow_qwen(
                prompt=prompt,
                gguf_path=gguf,
                vae_name=vae,
                clip_gguf_name=clip,
                steps=6,
                width=width,
                height=height,
                seed=seed,
                sampler_name="euler_ancestral",
                scheduler_name="beta",
                uploaded_image_names=[char_name_upload, player_name_upload],
                uploaded_subjects=[char_name, player_name],
                subject_appearances=subject_appearances,
            )
            label = f"multi_seed{seed}"
            print(f"\n[ABTest] === {label} — production appearance lock ({char_name} + {player_name}) ===")
            t0 = time.time()
            res = comfyui_ai._submit_and_poll(wf, base_url, client_id, args.timeout, requests)
            elapsed = time.time() - t0
            if not res:
                print(f"[ABTest] {label}: FAILED in {elapsed:.1f}s")
                multi_results.append({
                    "id": "MULTI", "label": "production multi-ref lock", "seed": seed,
                    "file": None, "elapsed": elapsed, "prompt": prompt, "negative": "",
                    "config": {"_label": "production multi-ref lock"}, "encoder_used": "v2" if encoder_v2 else "v1",
                })
                continue
            png_bytes, _mime = res
            fname = f"{label}.png"
            (out_dir / fname).write_bytes(png_bytes)
            print(f"[ABTest] {label}: saved {fname} ({len(png_bytes)} bytes, {elapsed:.1f}s)")
            multi_results.append({
                "id": "MULTI", "label": "production multi-ref lock", "seed": seed,
                "file": fname, "elapsed": elapsed, "prompt": prompt, "negative": "",
                "config": {"_label": "production multi-ref lock"}, "encoder_used": "v2" if encoder_v2 else "v1",
            })

        (out_dir / "results.json").write_text(
            json.dumps({"mode": "multi-ref", "prompt": prompt, "looks": looks,
                        "char": char_name, "player": player_name, "encoder_v2": encoder_v2,
                        "results": multi_results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        html = _render_multi_ref_html(
            prompt, looks, char_name, player_name,
            char_ref_path, player_ref_path, seeds, multi_results, out_dir,
        )
        (out_dir / "index.html").write_text(html, encoding="utf-8")
        print(f"\n[ABTest] Wrote {out_dir / 'index.html'} — open it in a browser.")
        return 0

    # ── Single-ref variant matrix (original behaviour) ────────────────────────
    variant_ids = [v for v in _parse_csv(args.variants, list(_VARIANTS)) if v in _VARIANTS]
    if not variant_ids:
        print(f"error: no valid variants in --variants={args.variants!r}", file=sys.stderr)
        return 2

    ref_paths = [char_ref_path]

    print(f"[ABTest] ComfyUI: {base_url}")
    print(f"[ABTest] gguf={gguf}  vae={vae}  clip={clip}")
    print(f"[ABTest] encoder: {'v2' if encoder_v2 else 'v1'} (--encoder={args.encoder})")
    print(f"[ABTest] variants={variant_ids}  seeds={seeds}  refs={[str(p) for p in ref_paths]}")
    print(f"[ABTest] looks: {(looks[:80] + '…') if len(looks) > 80 else (looks or '(none)')}")
    print(f"[ABTest] prompt: {prompt[:200]}{'…' if len(prompt) > 200 else ''}")
    print(f"[ABTest] out: {out_dir.resolve()}")

    raw_refs: List[Tuple[bytes, str]] = [_load_ref_bytes(p) for p in ref_paths]

    client_id = str(uuid.uuid4())
    results: List[dict] = []

    for vid in variant_ids:
        cfg = _merge(vid)
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
            wf, full_prompt, neg = _build_workflow(
                cfg, prompt, looks, ref_names, seed, gguf, vae, clip, encoder_v2,
            )
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
                    "prompt": full_prompt, "negative": neg,
                    "config": cfg, "encoder_used": "v2" if (cfg.get("force_v2") and encoder_v2) else "v1",
                })
                continue
            png_bytes, _mime = res
            fname = f"{label}.png"
            (out_dir / fname).write_bytes(png_bytes)
            print(f"[ABTest] {label}: saved {fname} ({len(png_bytes)} bytes, {elapsed:.1f}s)")
            results.append({
                "id": vid, "label": cfg["_label"], "seed": seed,
                "file": fname, "elapsed": elapsed,
                "prompt": full_prompt, "negative": neg,
                "config": cfg, "encoder_used": "v2" if (cfg.get("force_v2") and encoder_v2) else "v1",
            })

    (out_dir / "results.json").write_text(
        json.dumps({"prompt": prompt, "looks": looks, "encoder_v2": encoder_v2,
                    "results": [
                        {**r, "config": {k: v for k, v in r["config"].items()
                                         if not k.startswith("_") or k == "_label"}}
                        for r in results
                    ]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    html = _render_html(prompt, looks, encoder_v2, ref_paths, variant_ids, seeds, results, out_dir)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\n[ABTest] Wrote {out_dir / 'index.html'} — open it in a browser.")
    return 0


def _render_multi_ref_html(
    prompt: str,
    looks: str,
    char_name: str,
    player_name: str,
    char_ref_path: Path,
    player_ref_path: Path,
    seeds: List[int],
    results: List[dict],
    out_dir: Path,
) -> str:
    """HTML contact sheet for multi-ref mode (production appearance-lock pipeline)."""
    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def rel(p: Path) -> str:
        try:
            return os.path.relpath(p.resolve(), out_dir.resolve())
        except ValueError:
            return str(p)

    rows: List[str] = []
    rows.append("<h1>AriaBot multi-ref scene test (production appearance lock)</h1>")
    rows.append(
        f"<p><b>Prompt:</b> {esc(prompt)}</p>"
        f"<p><b>Char:</b> {esc(char_name)} · <b>Player:</b> {esc(player_name)}</p>"
        f"<p><b>Looks:</b> {esc(looks) or '(none)'}</p>"
    )

    rows.append("<h2>Reference photos</h2><div class='row'>")
    rows.append(f"<div class='cell'><img src='{esc(rel(char_ref_path))}'>"
                f"<p>slot 1: {esc(char_name)}</p></div>")
    rows.append(f"<div class='cell'><img src='{esc(rel(player_ref_path))}'>"
                f"<p>slot 2: {esc(player_name)}</p></div>")
    rows.append("</div>")

    rows.append("<h2>Outputs</h2><div class='row'>")
    for r in results:
        seed = r["seed"]
        if not r.get("file"):
            rows.append(f"<div class='cell'><div class='missing'>FAILED ({r['elapsed']:.1f}s)</div>"
                        f"<p>seed={seed}</p></div>")
        else:
            rows.append(f"<div class='cell'><img src='{esc(r['file'])}'>"
                        f"<p>seed={seed} · {r['elapsed']:.1f}s</p></div>")
    rows.append("</div>")

    rows.append("<h2>Full prompt sent to ComfyUI</h2>")
    rows.append(f"<pre>{esc(results[0]['prompt'] if results else prompt)}</pre>")

    _SHARED_STYLE = """
    <style>
      body { font-family: system-ui, sans-serif; max-width: 1600px; margin: 24px auto; padding: 0 16px; }
      h1 { font-size: 22px; }
      h2 { font-size: 18px; margin-top: 32px; border-bottom: 1px solid #ccc; }
      .row { display: flex; gap: 12px; flex-wrap: wrap; }
      .cell { display: inline-block; }
      .cell img { max-width: 400px; max-height: 400px; border: 1px solid #ddd; }
      .cell p { margin: 4px 0 0; font-size: 12px; color: #555; }
      .missing { width: 320px; height: 240px; display: flex; align-items: center;
                 justify-content: center; background: #fee; color: #800; border: 1px solid #fcc; }
      pre { background: #f6f6f6; padding: 8px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; }
    </style>
    """
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>AriaBot multi-ref test</title>" + _SHARED_STYLE
            + "</head><body>" + "\n".join(rows) + "</body></html>")


def _render_html(
    prompt: str,
    looks: str,
    encoder_v2: bool,
    ref_paths: List[Path],
    variant_ids: List[str],
    seeds: List[int],
    results: List[dict],
    out_dir: Path,
) -> str:
    by_id_seed = {(r["id"], r["seed"]): r for r in results}

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows: List[str] = []
    rows.append("<h1>AriaBot scene-image A/B test</h1>")
    rows.append(
        f"<p><b>Prompt:</b> {esc(prompt)}</p>"
        f"<p><b>Encoder:</b> {'v2' if encoder_v2 else 'v1'} · "
        f"<b>Looks:</b> {esc(looks) or '(none)'}</p>"
    )

    rows.append("<h2>Reference photo(s)</h2><div class='row'>")
    for rp in ref_paths:
        try:
            rel = os.path.relpath(rp.resolve(), out_dir.resolve())
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
                f"<div class='cell'><img src='{esc(r['file'])}'>"
                f"<p>seed={seed} · {r['elapsed']:.1f}s · enc={esc(r.get('encoder_used',''))}</p></div>"
            )
        rows.append("</div>")
        first = by_id_seed.get((vid, seeds[0]))
        if first:
            cfg_summary = ", ".join(f"{k}={v}" for k, v in first["config"].items() if not k.startswith("_"))
            rows.append("<details><summary>variant cfg & prompt</summary>")
            rows.append(
                f"<pre>{esc(cfg_summary)}\n\nPOSITIVE:\n{esc(first['prompt'])}\n\nNEGATIVE:\n{esc(first['negative'])}</pre>"
            )
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
    return ("<!doctype html><html><head><meta charset='utf-8'><title>AriaBot A/B test</title>"
            + style + "</head><body>" + "\n".join(rows) + "</body></html>")


if __name__ == "__main__":
    sys.exit(main())
