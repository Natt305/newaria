"""
ComfyUI image generation backend with two selectable engines.
Activated by setting IMAGE_BACKEND=comfyui in the environment.

Engine switch (`COMFYUI_ENGINE`):
    qwen   — DEFAULT. Phr00t's Qwen-Image-Edit-Rapid AIO repackaged as GGUF
             (`Qwen-Rapid-NSFW-v23_Q4_K.gguf` etc.). Loads via city96's
             `UnetLoaderGGUF` + `CLIPLoaderGGUF` (type=qwen_image), and uses
             `TextEncodeQwenImageEditPlus` so reference images are fed through
             the Qwen 2.5 VL multimodal encoder — the model literally "sees"
             the character portrait + KB photos rather than performing a
             classical img2img denoise.
    flux   — Original FLUX.2-klein-4B GGUF stack with ReferenceLatent /
             multiref / Ultimate-Inpaint workflows (see below). Kept for
             back-compat; pick this when you want the legacy behaviour.

Required env vars (FLUX engine — `COMFYUI_ENGINE=flux`):
    COMFYUI_GGUF     GGUF filename as known to ComfyUI (e.g. flux-2-klein-4b-Q5_K_M.gguf).
    COMFYUI_VAE      VAE filename as known to ComfyUI (e.g. flux2-vae.safetensors).
    COMFYUI_CLIP     Text-encoder filename as known to ComfyUI (e.g. qwen_3_4b.safetensors).

Required env vars (Qwen engine — `COMFYUI_ENGINE=qwen`, the default):
    COMFYUI_QWEN_GGUF        Qwen-Image-Edit-Rapid AIO GGUF filename
                             (e.g. `Qwen-Rapid-NSFW-v23_Q4_K.gguf`). Loaded by
                             `UnetLoaderGGUF` from city96's `ComfyUI-GGUF`.
    COMFYUI_QWEN_VAE         Qwen-Image VAE filename (e.g. `pig_qwen_image_vae_fp32-f16.gguf`
                             or any other Qwen-Image-compatible VAE in your `models/vae/`).
    COMFYUI_QWEN_CLIP_GGUF   Qwen 2.5 VL text encoder GGUF filename
                             (e.g. `Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf`). Place the
                             matching `mmproj-f16.gguf` (or `mmproj-Q8_0.gguf`) right
                             beside it in `models/clip/` — without mmproj the encoder
                             is text-only and reference images are silently ignored.

Optional env vars (Qwen engine):
    COMFYUI_QWEN_STEPS       Sampling steps (default: 4 — the Rapid AIO is distilled).
    COMFYUI_QWEN_SAMPLER     KSampler sampler_name (default: `euler_ancestral`).
    COMFYUI_QWEN_SCHEDULER   KSampler scheduler (default: `beta`).
    COMFYUI_QWEN_WIDTH       Output width  (default: 1024 — Qwen-Image native scale).
    COMFYUI_QWEN_HEIGHT      Output height (default: 1024).

Required ComfyUI custom node packs for the Qwen engine:
    - `ComfyUI-GGUF` (city96)             → UnetLoaderGGUF + CLIPLoaderGGUF
    - `comfyui_qwen_image_edit_plus_nodes` (or any pack shipping
       `TextEncodeQwenImageEditPlus`)     → image-aware text encoder
    VAE must be a safetensors file accepted by the stock `VAELoader`.
    If you have a GGUF VAE (e.g. `pig_qwen_image_vae_fp32-f16.gguf`), convert it
    once with: python scripts/convert_pigvae.py  then set COMFYUI_QWEN_VAE to the
    resulting .safetensors filename. VAELoaderGGUF does not exist in city96's pack.

Optional env vars (both engines):
    COMFYUI_PREWARM  `1`/`true`/`yes`/`on` to pre-load the active engine's
                     model into VRAM right after on_ready (a tiny throwaway
                     256x256, 1-step txt2img is submitted via the same
                     engine-specific builder); `0`/`false`/`no`/`off` to
                     skip. Unset uses the per-engine default — on for `qwen`,
                     off for `flux`. Cuts the first real !generate's wait by
                     the model-load cost (~30–60 s on typical setups). Runs
                     concurrently with bot startup; failure logs a single
                     WARN line and never blocks.

Optional env vars (FLUX engine):
    COMFYUI_URL      ComfyUI server address (default: http://127.0.0.1:8188).
    COMFYUI_STEPS    Number of inference steps (default: 4).
    COMFYUI_WIDTH    Output width in pixels (default: 512).
    COMFYUI_HEIGHT   Output height in pixels (default: 512).
    COMFYUI_STRENGTH         img2img denoise strength, 0.0-1.0 (default: 0.75).
    COMFYUI_FLUX_MAX_SHIFT   ModelSamplingFlux max_shift for img2img (default: 1.15).
                             Patching the FLUX timestep schedule with this value corrects
                             the noise sigma for the target resolution so the reference
                             latent is not drowned and img2img actually follows the photo.
    COMFYUI_FLUX_BASE_SHIFT  ModelSamplingFlux base_shift for img2img (default: 0.5).
    COMFYUI_WORKFLOW         Path to a custom workflow JSON file (overrides built-in template).
    COMFYUI_TIMEOUT          Max seconds to wait for job completion (default: 300).

Multi-reference mode (FLUX.2 Klein native):
    When reference_images are provided, the bot uses FLUX.2 Klein's native ReferenceLatent
    conditioning — no IP-Adapter or CLIP Vision required. Each reference image is VAE-encoded
    and injected into both positive and negative conditioning via chained ReferenceLatent nodes.
    Output resolution auto-matches the first reference image (scaled to ~1 megapixel).
    Uses the advanced sampler pipeline: EmptyFlux2LatentImage + Flux2Scheduler +
    SamplerCustomAdvanced + CFGGuider + KSamplerSelect + RandomNoise.

Ultimate Inpaint multi-character mode:
    Activated by COMFYUI_MODE=ultimate_inpaint when reference_images and subject_appearances
    are both provided. Uses the "Flux.2 Ultimate Inpaint Pro Ultra v3.1" GUI workflow which
    supports up to 4 characters each with their own ReferenceLatent conditioning, RMBG
    background removal, and InpaintCropImproved per-character inpainting pass.
    Workflow:
      1. Character reference images are uploaded to ComfyUI.
      2. An initial scene is generated via a txt2img pass and uploaded as the canvas.
      3. The Ultimate Inpaint workflow runs: for each character slot it crops the character
         region, inpaints with the reference conditioning, and stitches back.
      4. Optional SAM3 auto-segmentation detects character regions (enabled by default;
         disable with COMFYUI_USE_SAM=false).
    Additional optional env vars for this mode:
        COMFYUI_USE_SAM    Enable SAM3 auto-segmentation (default: true).

Progress reporting:
    generate_image() connects to ComfyUI's WebSocket API (ws://.../ws?clientId=…)
    and translates server-sent events to the same tag format used by the diffusers
    backend:
        STAGE:loading   — job accepted, model loading (fake LOAD ticks follow)
        LOAD:<frac>     — fractional loading progress 0.0–1.0 (approximated)
        STAGE:ready     — first inference step received, model fully loaded
        STEP:<n>/<t>    — inference step n of t completed
        STAGE:encoding  — final step done, VAE decoding / saving
    If the websockets package is not installed or the connection fails the
    on_progress callback is silently skipped and HTTP polling handles completion.
"""

import asyncio
import io
import json
import os
import tempfile
import time
import uuid
from collections import defaultdict
from typing import Callable, Coroutine, Dict, List, Optional, Tuple

DEFAULT_URL = "http://127.0.0.1:8188"
DEFAULT_STEPS = 4
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STRENGTH = 0.75
DEFAULT_TIMEOUT = 300

# Qwen-Image-Edit-Rapid AIO defaults (Phr00t / phil2sat GGUF stack).
# Distilled model: 4–6 steps + CFG 1.0 is the intended config. Phr00t's
# Qwen-Rapid-AIO.json reference uses 4; we default to 6 here for a small
# detail bump at modest cost. euler_ancestral/beta matches the reference.
# Default canvas is 1024 × 768 (4:3), the most common Discord scene aspect.
DEFAULT_QWEN_STEPS = 6
DEFAULT_QWEN_SAMPLER = "euler_ancestral"
DEFAULT_QWEN_SCHEDULER = "beta"
DEFAULT_QWEN_WIDTH = 1024
DEFAULT_QWEN_HEIGHT = 768

# Matches diffusers_worker.py — applied to every generation for anatomy quality.
_ANATOMY_SUFFIX = (
    ", perfect anatomy, correct arms, well-drawn hands, "
    "five fingers, proper limbs, symmetrical body"
)
_ANATOMY_NEGATIVE = (
    "bad anatomy, missing arms, extra arms, deformed arms, "
    "missing hands, extra hands, fused fingers, missing fingers, "
    "extra fingers, malformed hands, mutated hands, "
    "extra limbs, missing limbs, poorly drawn hands"
)


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        print(f"[ComfyUI] Invalid {key} — using default {default}.")
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        print(f"[ComfyUI] Invalid {key} — using default {default}.")
        return default


def _preprocess_reference_image(
    img_bytes: bytes,
    mime: str,
    width: int,
    height: int,
) -> Optional[bytes]:
    """Smart-crop and resize reference image to (width, height), return PNG bytes.

    Uses the same portrait-aware centering as diffusers_worker.py:
    - Portrait originals (height > width) are cropped with centering (0.5, 0.35)
      to keep faces / upper bodies in frame.
    - Landscape originals use center crop (0.5, 0.5).

    Returns PNG bytes, or None if PIL is not available / decoding fails.
    """
    try:
        from PIL import Image, ImageOps
    except ImportError:
        print("[ComfyUI] Pillow not installed — uploading reference image as-is.")
        return None

    try:
        raw = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        centering = (0.5, 0.35) if raw.height > raw.width else (0.5, 0.5)
        cropped = ImageOps.fit(raw, (width, height), method=Image.LANCZOS, centering=centering)
        print(
            f"[ComfyUI] Reference image smart-cropped to {width}x{height} "
            f"(original {raw.width}x{raw.height}, centering={centering})."
        )
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        print(f"[ComfyUI] Reference image preprocessing failed ({exc}) — uploading as-is.")
        return None


def _to_png(img_bytes: bytes) -> Optional[bytes]:
    """Convert image bytes to PNG format without resizing.

    Used before uploading reference images to ComfyUI when the workflow
    handles resizing internally (e.g. ImageScaleToTotalPixels).
    Returns PNG bytes, or None if Pillow is unavailable or decoding fails.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as exc:
        print(f"[ComfyUI] PNG conversion failed ({exc}) — uploading as-is.")
        return None


def _build_txt2img_workflow(
    prompt: str,
    gguf_path: str,
    vae_name: str,
    clip_name: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
) -> dict:
    """ComfyUI API workflow for FLUX.2-klein-4B txt2img (CLIPLoader flux2 + Qwen-3-4B)."""
    enhanced_prompt = prompt + _ANATOMY_SUFFIX
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": gguf_path},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux2"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": enhanced_prompt},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": _ANATOMY_NEGATIVE},
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "ariabot_"},
        },
    }


def _build_img2img_workflow(
    prompt: str,
    gguf_path: str,
    vae_name: str,
    clip_name: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    strength: float,
    uploaded_image_name: str,
    max_shift: float = 1.15,
    base_shift: float = 0.5,
) -> dict:
    """ComfyUI API workflow for FLUX.2-klein-4B img2img (CLIPLoader flux2 + Qwen-3-4B).

    The reference image is assumed to already be smart-cropped to (width, height)
    by _preprocess_reference_image. ImageScale is kept as a safety net in case the
    uploaded dimensions differ slightly.

    ModelSamplingFlux (node 13) patches the model's timestep schedule to be
    resolution-aware before the KSampler runs.  Without it, FLUX's default linear
    noise schedule maps denoise values to the wrong sigma for small resolutions
    (e.g. 512x512), so the reference latent is overwhelmed with noise and the
    model effectively ignores it.  The diffusers FluxImg2ImgPipeline applies this
    correction automatically; ComfyUI requires an explicit node.
    """
    enhanced_prompt = "referencing the provided image, " + prompt + _ANATOMY_SUFFIX
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": gguf_path},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux2"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": enhanced_prompt},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": _ANATOMY_NEGATIVE},
        },
        "10": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_image_name, "upload": "image"},
        },
        "11": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["10", 0],
                "upscale_method": "bilinear",
                "width": width,
                "height": height,
                "crop": "center",
            },
        },
        "12": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["11", 0], "vae": ["3", 0]},
        },
        "13": {
            "class_type": "ModelSamplingFlux",
            "inputs": {
                "model": ["1", 0],
                "max_shift": max_shift,
                "base_shift": base_shift,
                "width": width,
                "height": height,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["13", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["12", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": strength,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "ariabot_"},
        },
    }


def _build_reference_workflow(
    prompt: str,
    gguf_path: str,
    vae_name: str,
    clip_name: str,
    steps: int,
    seed: int,
    uploaded_image_names: list,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> dict:
    """ComfyUI API workflow using FLUX.2 Klein's native ReferenceLatent conditioning.

    Each reference image is VAE-encoded and injected into both positive and negative
    conditioning via chained ReferenceLatent nodes — no IP-Adapter or CLIP Vision
    required. This is the official multi-reference path for FLUX.2 Klein as documented
    at docs.comfy.org/tutorials/flux/flux-2-klein.

    Output resolution is fixed to (width, height) — the same values used for txt2img
    and img2img. ImageScaleToTotalPixels scales each reference image to the same
    pixel count (width*height) so there is no upscaling of small thumbnails and the
    generated image matches the configured size.

    Node ID scheme:
        Fixed nodes:   "1"–"5" (model loaders + text encode + zero-neg)
        Per-reference: "20i" LoadImage, "21i" ImageScaleToTotalPixels,
                       "22i" VAEEncode, "23i" ReferenceLatent (positive),
                       "24i" ReferenceLatent (negative)  — i is 0-based index
        Output chain:  "31" EmptyFlux2LatentImage, "32" Flux2Scheduler,
                       "33" KSamplerSelect, "34" RandomNoise, "35" CFGGuider,
                       "36" SamplerCustomAdvanced, "37" VAEDecode, "9" SaveImage
    """
    enhanced_prompt = prompt + _ANATOMY_SUFFIX
    workflow: dict = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": gguf_path},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux2"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": enhanced_prompt},
        },
        # Distilled FLUX2 models don't use a real negative prompt — zero it out.
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["4", 0]},
        },
    }

    # megapixels target = output resolution; prevents upscaling small thumbnails
    megapixels = round((width * height) / 1_000_000, 4)

    # Per-reference-image nodes: scale → encode → inject into conditioning
    for i, img_name in enumerate(uploaded_image_names):
        load_id    = f"20{i}"
        scale_id   = f"21{i}"
        encode_id  = f"22{i}"
        ref_pos_id = f"23{i}"
        ref_neg_id = f"24{i}"

        prev_pos = ["4", 0] if i == 0 else [f"23{i - 1}", 0]
        prev_neg = ["5", 0] if i == 0 else [f"24{i - 1}", 0]

        workflow[load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": img_name, "upload": "image"},
        }
        # Scale reference to match output pixel count.
        # resolution_steps=64 ensures output dimensions are multiples of 64 —
        # required for VAE encoding (non-aligned dims produce garbage latents).
        # lanczos gives better quality than nearest-exact when upscaling thumbnails.
        workflow[scale_id] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": [load_id, 0],
                "upscale_method": "lanczos",
                "megapixels": megapixels,
                "resolution_steps": 64,
            },
        }
        workflow[encode_id] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [scale_id, 0], "vae": ["3", 0]},
        }
        workflow[ref_pos_id] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": prev_pos, "latent": [encode_id, 0]},
        }
        workflow[ref_neg_id] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": prev_neg, "latent": [encode_id, 0]},
        }

    n = len(uploaded_image_names)
    final_pos = [f"23{n - 1}", 0]
    final_neg = [f"24{n - 1}", 0]

    # Output latent size fixed to configured width/height — no GetImageSize needed
    workflow["31"] = {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {
            "width": width,
            "height": height,
            "batch_size": 1,
        },
    }
    workflow["32"] = {
        "class_type": "Flux2Scheduler",
        "inputs": {
            "steps": steps,
            "width": width,
            "height": height,
        },
    }
    workflow["33"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
    }
    workflow["34"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }
    workflow["35"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": ["1", 0],
            "positive": final_pos,
            "negative": final_neg,
            "cfg": 1.0,
        },
    }
    workflow["36"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["34", 0],
            "guider": ["35", 0],
            "sampler": ["33", 0],
            "sigmas": ["32", 0],
            "latent_image": ["31", 0],
        },
    }
    workflow["37"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["36", 0], "vae": ["3", 0]},
    }
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["37", 0], "filename_prefix": "ariabot_"},
    }
    return workflow


def _build_per_subject_ref_workflow(
    prompt: str,
    gguf_path: str,
    vae_name: str,
    clip_name: str,
    steps: int,
    seed: int,
    uploaded_image_groups: list,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> dict:
    """Per-subject isolated ReferenceLatent workflow for multi-character scenes.

    Each subject gets its own independent ReferenceLatent chain that starts
    from the shared base text conditioning.  Subject chains are NEVER joined
    mid-chain — each subject's photos only condition that subject's identity.
    At the end, all subjects' final conditioning tensors are merged with
    ConditioningCombine before the sampler.

    This prevents the cross-subject appearance bleeding that occurs in the
    single chained workflow, where all photos feed one accumulating stream and
    later photos overwrite earlier ones regardless of which character they show.

    Args:
        uploaded_image_groups: list of lists — [[subj0_img0, subj0_img1, ...],
                                                 [subj1_img0, subj1_img1, ...], ...]
                               Each inner list contains the ComfyUI server filenames
                               for one subject's reference images.

    Node ID scheme:
        Fixed:      "1"–"5"   (loaders + text encode + zero-neg)
        Per photo:  "L{s}_{p}"  LoadImage
                    "S{s}_{p}"  ImageScaleToTotalPixels
                    "E{s}_{p}"  VAEEncode
                    "RP{s}_{p}" ReferenceLatent (positive chain)
                    "RN{s}_{p}" ReferenceLatent (negative chain)
                    s = 0-based subject index, p = 0-based photo index
        Combine:    "CP{k}"   ConditioningCombine (positive), k = 0-based step
                    "CN{k}"   ConditioningCombine (negative)
        Output:     "31"–"37", "9"  (identical to single-chain workflow)
    """
    enhanced_prompt = prompt + _ANATOMY_SUFFIX
    workflow: dict = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": gguf_path},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux2"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": enhanced_prompt},
        },
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["4", 0]},
        },
    }

    megapixels = round((width * height) / 1_000_000, 4)
    subject_final_pos: list = []  # final ReferenceLatent node ID per subject
    subject_final_neg: list = []

    for s, img_names in enumerate(uploaded_image_groups):
        for p, img_name in enumerate(img_names):
            load_id    = f"L{s}_{p}"
            scale_id   = f"S{s}_{p}"
            encode_id  = f"E{s}_{p}"
            ref_pos_id = f"RP{s}_{p}"
            ref_neg_id = f"RN{s}_{p}"

            # Each subject's chain starts independently from the base text
            # conditioning — photos from other subjects never influence this chain.
            prev_pos = ["4", 0] if p == 0 else [f"RP{s}_{p - 1}", 0]
            prev_neg = ["5", 0] if p == 0 else [f"RN{s}_{p - 1}", 0]

            workflow[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": img_name, "upload": "image"},
            }
            workflow[scale_id] = {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "image": [load_id, 0],
                    "upscale_method": "lanczos",
                    "megapixels": megapixels,
                    "resolution_steps": 64,
                },
            }
            workflow[encode_id] = {
                "class_type": "VAEEncode",
                "inputs": {"pixels": [scale_id, 0], "vae": ["3", 0]},
            }
            workflow[ref_pos_id] = {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": prev_pos, "latent": [encode_id, 0]},
            }
            workflow[ref_neg_id] = {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": prev_neg, "latent": [encode_id, 0]},
            }

        # Record the final conditioning node for this subject
        last_p = len(img_names) - 1
        subject_final_pos.append(f"RP{s}_{last_p}")
        subject_final_neg.append(f"RN{s}_{last_p}")

    # Merge all subject conditioning chains with ConditioningCombine.
    # Single subject: no combine needed — use the chain directly.
    # Multiple subjects: chain combines left-to-right:
    #   combine(s0, s1) → CP0 ; combine(CP0, s2) → CP1 ; …
    if len(subject_final_pos) == 1:
        final_pos: list = [subject_final_pos[0], 0]
        final_neg: list = [subject_final_neg[0], 0]
    else:
        cur_pos_id = subject_final_pos[0]
        cur_neg_id = subject_final_neg[0]
        for k in range(1, len(subject_final_pos)):
            cp_id = f"CP{k - 1}"
            cn_id = f"CN{k - 1}"
            workflow[cp_id] = {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [cur_pos_id, 0],
                    "conditioning_2": [subject_final_pos[k], 0],
                },
            }
            workflow[cn_id] = {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [cur_neg_id, 0],
                    "conditioning_2": [subject_final_neg[k], 0],
                },
            }
            cur_pos_id = cp_id
            cur_neg_id = cn_id
        final_pos = [cur_pos_id, 0]
        final_neg = [cur_neg_id, 0]

    workflow["31"] = {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    workflow["32"] = {
        "class_type": "Flux2Scheduler",
        "inputs": {"steps": steps, "width": width, "height": height},
    }
    workflow["33"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
    }
    workflow["34"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }
    workflow["35"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": ["1", 0],
            "positive": final_pos,
            "negative": final_neg,
            "cfg": 1.0,
        },
    }
    workflow["36"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["34", 0],
            "guider": ["35", 0],
            "sampler": ["33", 0],
            "sigmas": ["32", 0],
            "latent_image": ["31", 0],
        },
    }
    workflow["37"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["36", 0], "vae": ["3", 0]},
    }
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["37", 0], "filename_prefix": "ariabot_"},
    }
    return workflow


# ── Qwen-Image-Edit-Rapid (GGUF) builders ────────────────────────────────────
#
# These build small, monolithic ComfyUI graphs for Phr00t's
# `Qwen-Image-Edit-Rapid AIO` model repackaged as GGUF (e.g.
# `Qwen-Rapid-NSFW-v23_Q4_K.gguf`).  Edit-mode is achieved via
# `TextEncodeQwenImageEditPlus`, which routes up to four reference images
# through the Qwen 2.5 VL multimodal encoder so the model literally consumes
# the pixels rather than performing a classical img2img denoise.
#
# Required custom node packs on the ComfyUI side:
#   - city96/ComfyUI-GGUF                       → UnetLoaderGGUF, CLIPLoaderGGUF
#   - any pack shipping TextEncodeQwenImageEditPlus
# The VAE must be a safetensors file accepted by the stock VAELoader.
# GGUF VAEs (e.g. pig_qwen_image_vae_fp32-f16.gguf) cannot be loaded directly —
# convert with scripts/convert_pigvae.py, then set COMFYUI_QWEN_VAE to the
# resulting .safetensors filename. VAELoaderGGUF does not exist in city96's pack.
#
# Latent shape: Qwen-Image is a 16-channel MMDiT model, so we use
# `EmptySD3LatentImage` (also 16 channels) rather than `EmptyLatentImage`
# which only emits 4-channel SD1.5/SDXL-style latents.

def _build_txt2img_workflow_qwen(
    prompt: str,
    gguf_path: str,
    vae_name: str,
    clip_gguf_name: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    sampler_name: str,
    scheduler_name: str,
) -> dict:
    """Qwen-Image-Edit-Rapid GGUF txt2img graph.

    Stack:
        UnetLoaderGGUF                            → MODEL
        CLIPLoaderGGUF (type=qwen_image)          → CLIP
        VAELoader                                 → VAE
        CLIPTextEncode (positive + empty negative)
        EmptySD3LatentImage                       → LATENT (16-channel)
        KSampler (CFG=1.0, configurable sampler/scheduler/steps)
        VAEDecode → SaveImage

    Negative is intentionally an empty string — the model is distilled to
    CFG=1.0 so any negative is multiplied by zero anyway, and matching the
    reference Qwen-Rapid-AIO.json keeps behaviour predictable.
    """
    enhanced_prompt = prompt + _ANATOMY_SUFFIX
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": gguf_path},
        },
        "2": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": clip_gguf_name, "type": "qwen_image"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": enhanced_prompt},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": ""},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": sampler_name,
                "scheduler": scheduler_name,
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "ariabot_qwen_"},
        },
    }


def _build_multi_edit_workflow_qwen(
    prompt: str,
    gguf_path: str,
    vae_name: str,
    clip_gguf_name: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    sampler_name: str,
    scheduler_name: str,
    uploaded_image_names: List[str],
) -> dict:
    """Qwen-Image-Edit-Rapid GGUF multi-reference edit-mode graph (2–4 refs).

    Each reference image is loaded via `LoadImage` and wired into one of the
    `image1`..`image4` slots of `TextEncodeQwenImageEditPlus`, which routes
    them through the Qwen 2.5 VL multimodal encoder so the model truly "sees"
    every reference alongside the prompt — true edit-mode, not classical
    img2img. Inputs beyond 4 are silently dropped (encoder only has 4 slots).
    Negative conditioning is a plain empty `CLIPTextEncode`; with CFG=1.0
    the negative branch is multiplied by zero.

    Used by `_run_generate_qwen` whenever there are 2 or more refs. For the
    single-ref case prefer `_build_edit_workflow_qwen`, which is a thin
    wrapper around this function.
    Falls back to the txt2img graph if `uploaded_image_names` is empty.
    """
    n = min(len(uploaded_image_names), 4)
    if n < 1:
        return _build_txt2img_workflow_qwen(
            prompt, gguf_path, vae_name, clip_gguf_name,
            steps, width, height, seed, sampler_name, scheduler_name,
        )
    # In edit-mode (≥1 ref) prepend an appearance-lock instruction.  The
    # Qwen-Image-Edit model will follow explicit "do NOT change …" directives
    # even when the rest of the prompt describes the scene differently — this
    # prevents a hallucinated text colour (e.g. "ash-blonde") from overriding
    # the hair colour that is plainly visible in the reference image.
    _APPEARANCE_LOCK = (
        "Preserve the exact appearance of all characters from the reference images. "
        "Do NOT change hair colour, eye colour, skin tone, or facial features. "
        "Art style MUST be flat 2D anime illustration: clean sharp ink linework, "
        "cel-shaded flat color fills with minimal gradients, crisp contour lines, "
        "colored manga key-visual aesthetic. "
        "Do NOT replicate the rendering style of the reference images if they use "
        "3D rendering, volumetric shading, gradient shading, or webtoon-style "
        "semi-3D rendering — override those traits with flat 2D anime style. "
        "Do NOT introduce photorealism or any style other than flat 2D anime. "
    )
    enhanced_prompt = _APPEARANCE_LOCK + prompt + _ANATOMY_SUFFIX

    workflow: dict = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": gguf_path},
        },
        "2": {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": clip_gguf_name, "type": "qwen_image"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": ""},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
    }

    # LoadImage nodes per reference (IDs 200..203) wired into image1..image4.
    encoder_inputs: dict = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "prompt": enhanced_prompt,
    }
    for i in range(n):
        load_id = f"20{i}"
        workflow[load_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_image_names[i], "upload": "image"},
        }
        encoder_inputs[f"image{i + 1}"] = [load_id, 0]

    # v2-only: wire the empty latent into the encoder so the node can size
    # the reference embeddings against the target canvas — this is what
    # actually kills the scaling/cropping/zoom artifacts on multi-ref edits.
    # Gated by the diagnose() probe so v1 installs (which don't accept this
    # input) aren't poisoned with an unknown slot. When the cache is None
    # (diagnose hasn't run yet) we deliberately do NOT wire it: a one-shot
    # warning in `_run_generate_qwen` tells the operator to run /diagcomfyui.
    if _QWEN_ENCODER_V2 is True:
        encoder_inputs["latent_image"] = ["6", 0]

    workflow["10"] = {
        "class_type": "TextEncodeQwenImageEditPlus",
        "inputs": encoder_inputs,
    }
    workflow["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["10", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": seed,
            "steps": steps,
            "cfg": 1.0,
            "sampler_name": sampler_name,
            "scheduler": scheduler_name,
            "denoise": 1.0,
        },
    }
    workflow["8"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
    }
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0], "filename_prefix": "ariabot_qwen_"},
    }
    return workflow


def _build_edit_workflow_qwen(
    prompt: str,
    gguf_path: str,
    vae_name: str,
    clip_gguf_name: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    sampler_name: str,
    scheduler_name: str,
    uploaded_image_name: str,
) -> dict:
    """Qwen-Image-Edit-Rapid GGUF single-reference edit-mode graph.

    Thin convenience wrapper around `_build_multi_edit_workflow_qwen`: routes
    the one uploaded image into the `image1` slot of
    `TextEncodeQwenImageEditPlus` (slots `image2`..`image4` left empty) so
    the model sees the single reference pixel-perfect. Used by
    `_run_generate_qwen` whenever exactly one reference image is supplied
    (typically: only the bot's own portrait, no KB matches).
    """
    return _build_multi_edit_workflow_qwen(
        prompt, gguf_path, vae_name, clip_gguf_name,
        steps, width, height, seed, sampler_name, scheduler_name,
        [uploaded_image_name],
    )


def _submit_and_poll(
    workflow: dict,
    base_url: str,
    client_id: str,
    timeout: int,
    requests_mod,
) -> Optional[Tuple[bytes, str]]:
    """Submit a workflow to /prompt and poll /history until the job completes.

    Used by the Qwen engine; the FLUX engine has its own (richer) submit-and-
    poll loop with multiple fallback workflows wired in. Returns
    (image_bytes, "image/png") on success or None on failure.
    """
    try:
        payload = {"prompt": workflow, "client_id": client_id}
        resp = requests_mod.post(f"{base_url}/prompt", json=payload, timeout=15)
        if resp.status_code != 200:
            print(f"[ComfyUI] /prompt rejected (HTTP {resp.status_code}): {resp.text[:400]}")
            try:
                err = resp.json()
                node_errors = err.get("node_errors", {})
                _comfy_path = os.environ.get("COMFYUI_PATH", "").strip()
                for node_id, node_err in node_errors.items():
                    class_type = workflow.get(node_id, {}).get("class_type", node_id)
                    for e in node_err.get("errors", []):
                        print(f"[ComfyUI]   Node {node_id} ({class_type}): "
                              f"[{e.get('type', '?')}] {e.get('message', '')} -- {e.get('details', '')}")
                        if e.get("type") == "value_not_in_list":
                            _details = e.get("details", "")
                            if "vae_name:" in _details:
                                _dir = (os.path.join(_comfy_path, "models", "vae")
                                        if _comfy_path else "<ComfyUI>/models/vae/")
                                print(f"[ComfyUI] HINT: Your VAE file is not in ComfyUI's model list.")
                                print(f"[ComfyUI] HINT: Place it in: {_dir}")
                                print(f"[ComfyUI] HINT: Then close ComfyUI and re-run start.bat.")
                            elif "unet_name:" in _details:
                                _dir = (os.path.join(_comfy_path, "models", "unet")
                                        if _comfy_path else "<ComfyUI>/models/unet/")
                                print(f"[ComfyUI] HINT: Your GGUF checkpoint is not in ComfyUI's model list.")
                                print(f"[ComfyUI] HINT: Place it in: {_dir}")
                                print(f"[ComfyUI] HINT: Then close ComfyUI and re-run start.bat.")
                            elif "clip_name:" in _details:
                                _dir = (os.path.join(_comfy_path, "models", "clip")
                                        if _comfy_path else "<ComfyUI>/models/clip/")
                                print(f"[ComfyUI] HINT: Your CLIP GGUF is not in ComfyUI's model list.")
                                print(f"[ComfyUI] HINT: Place it in: {_dir}")
                                print(f"[ComfyUI] HINT: Then close ComfyUI and re-run start.bat.")
            except Exception:
                pass
            return None

        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            print(f"[ComfyUI] /prompt returned no prompt_id: {resp.text[:300]}")
            return None
        print(f"[ComfyUI] Job queued — prompt_id={prompt_id}")

        deadline = time.time() + timeout
        last_log = time.time()
        consecutive_poll_failures = 0
        MAX_CONSECUTIVE_FAILURES = 30

        while time.time() < deadline:
            time.sleep(2)

            now = time.time()
            if now - last_log >= 15:
                elapsed = int(now - (deadline - timeout))
                print(f"[ComfyUI] Still waiting… ({elapsed}s elapsed)")
                last_log = now

            try:
                hist = requests_mod.get(f"{base_url}/history/{prompt_id}", timeout=30)
            except Exception:
                consecutive_poll_failures += 1
                if consecutive_poll_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("[ComfyUI] No /history response for ~60s — ComfyUI appears down. Aborting.")
                    return None
                continue
            if hist.status_code != 200:
                consecutive_poll_failures += 1
                if consecutive_poll_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"[ComfyUI] /history failing for ~60s (HTTP {hist.status_code}) — aborting.")
                    return None
                continue
            consecutive_poll_failures = 0

            history = hist.json()
            if prompt_id not in history:
                continue

            entry = history[prompt_id]
            status_obj = entry.get("status", {})
            completed = status_obj.get("completed", False)
            status_str = status_obj.get("status_str", "")
            if not completed and status_str != "error":
                continue
            if status_str == "error":
                for msg in status_obj.get("messages", []):
                    if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
                        e = msg[1]
                        print(f"[ComfyUI] Execution error in node {e.get('node_id')} "
                              f"({e.get('node_type', '?')}): {e.get('exception_message', '')}")
                return None

            for _, output in entry.get("outputs", {}).items():
                imgs = output.get("images", [])
                if not imgs:
                    continue
                first = imgs[0]
                params = {"filename": first["filename"], "type": first.get("type", "output")}
                if first.get("subfolder"):
                    params["subfolder"] = first["subfolder"]
                view = requests_mod.get(f"{base_url}/view", params=params, timeout=30)
                if view.status_code == 200:
                    print(f"[ComfyUI] Done — {len(view.content)} bytes ('{first['filename']}')")
                    return view.content, "image/png"
            print("[ComfyUI] Job completed but no image output found.")
            return None

        print(f"[ComfyUI] Timed out after {timeout}s waiting for prompt_id={prompt_id}")
        return None
    except Exception as exc:
        print(f"[ComfyUI] Submit/poll error: {type(exc).__name__}: {exc}")
        return None


def _run_generate_qwen(
    prompt: str,
    refs: list,
    reference_image: Optional[Tuple[bytes, str]],
    client_id: str,
    seed: int,
    steps_override: Optional[int],
    width_override: Optional[int],
    height_override: Optional[int],
    timeout: int,
    base_url: str,
    requests_mod,
    reference_subjects: Optional[list] = None,
) -> Optional[Tuple[bytes, str]]:
    """Engine-specific runner for the Qwen-Image-Edit-Rapid GGUF stack.

    Reads `COMFYUI_QWEN_*` env vars, uploads up to 4 reference images, builds
    either a txt2img or an edit-mode workflow, then submits & polls. The FLUX
    engine path in `_run_generate` is left untouched.
    """
    qwen_gguf = os.environ.get("COMFYUI_QWEN_GGUF", "").strip()
    qwen_vae = os.environ.get("COMFYUI_QWEN_VAE", "").strip()
    qwen_clip = os.environ.get("COMFYUI_QWEN_CLIP_GGUF", "").strip()
    if not qwen_gguf or not qwen_vae or not qwen_clip:
        print("[ComfyUI] Qwen engine requires COMFYUI_QWEN_GGUF, COMFYUI_QWEN_VAE, "
              "and COMFYUI_QWEN_CLIP_GGUF to all be set.")
        return None

    # NOTE: `steps_override` is intentionally ignored for Qwen.
    # Upstream call sites in bot.py compute it from `COMFYUI_STEPS`, which is
    # FLUX-oriented (FLUX wants ~20 steps). The Qwen-Image-Edit-Rapid AIO
    # GGUF is distilled and runs in the 4–6 step range (DEFAULT_QWEN_STEPS).
    # Honoring a FLUX-tuned 20-step override on Qwen would waste ~3–5x the
    # wall-clock time per image with no quality gain. Qwen reads only its
    # own COMFYUI_QWEN_STEPS env var.
    if steps_override is not None and steps_override != DEFAULT_QWEN_STEPS:
        print(f"[ComfyUI] Qwen: ignoring steps_override={steps_override} "
              f"(FLUX-oriented); using COMFYUI_QWEN_STEPS / default {DEFAULT_QWEN_STEPS} instead.")
    qwen_steps = _get_int("COMFYUI_QWEN_STEPS", DEFAULT_QWEN_STEPS)
    qwen_width = width_override if width_override else _get_int(
        "COMFYUI_QWEN_WIDTH", DEFAULT_QWEN_WIDTH
    )
    qwen_height = height_override if height_override else _get_int(
        "COMFYUI_QWEN_HEIGHT", DEFAULT_QWEN_HEIGHT
    )
    qwen_sampler = os.environ.get("COMFYUI_QWEN_SAMPLER", DEFAULT_QWEN_SAMPLER).strip() or DEFAULT_QWEN_SAMPLER
    qwen_scheduler = os.environ.get("COMFYUI_QWEN_SCHEDULER", DEFAULT_QWEN_SCHEDULER).strip() or DEFAULT_QWEN_SCHEDULER

    print(f"[ComfyUI] Engine: qwen — GGUF: {qwen_gguf!r}, VAE: {qwen_vae!r}, CLIP: {qwen_clip!r}")
    print(f"[ComfyUI] Qwen settings — sampler: {qwen_sampler}/{qwen_scheduler}, "
          f"steps: {qwen_steps}, size: {qwen_width}x{qwen_height}")
    print(f"[ComfyUI] Prompt: {prompt[:200]!r}")

    # Pick reference inputs: prefer the multi-ref list; fall back to the single
    # legacy `reference_image` tuple. Cap at 4 (TextEncodeQwenImageEditPlus
    # has slots image1..image4).
    qwen_input_refs: list = []
    qwen_input_subjects: list = []
    if refs:
        qwen_input_refs = list(refs)
        if reference_subjects:
            qwen_input_subjects = [
                (str(s).strip() if s else "self") for s in reference_subjects
            ]
        else:
            qwen_input_subjects = ["self"] * len(qwen_input_refs)
        # Pad / trim subject labels to match ref count so zip() can't drop refs.
        if len(qwen_input_subjects) < len(qwen_input_refs):
            qwen_input_subjects.extend(
                ["self"] * (len(qwen_input_refs) - len(qwen_input_subjects))
            )
        elif len(qwen_input_subjects) > len(qwen_input_refs):
            qwen_input_subjects = qwen_input_subjects[: len(qwen_input_refs)]
    elif reference_image is not None:
        qwen_input_refs = [reference_image]
        qwen_input_subjects = ["self"]

    # ── Subject-aware reference selection (Pass A then Pass B) ──────────────
    # `TextEncodeQwenImageEditPlus` only has image1..image4. When the caller
    # passes multiple photos of the same subject (bot self profile + bot self
    # in a KB entry), naively taking the first 4 burns slots and starves
    # other subjects out of the encoder. Mirror the gather-side balance here
    # so any caller path benefits — not just scene_image's _gather_refs.
    if qwen_input_refs:
        _picked_indices: List[int] = []
        _used_per_label: dict = {}
        # Pass A: one ref per unique subject (caller order preserved).
        for _idx, (_ref, _subj) in enumerate(
            zip(qwen_input_refs, qwen_input_subjects)
        ):
            if len(_picked_indices) >= 4:
                break
            _key = _subj.strip().lower() or "self"
            if _used_per_label.get(_key, 0) >= 1:
                continue
            _picked_indices.append(_idx)
            _used_per_label[_key] = 1
        # Pass B: fill leftover slots with not-yet-picked refs in caller
        # order. The per-subject cap matches scene_image._gather_refs (≤2).
        if len(_picked_indices) < 4:
            _placed = set(_picked_indices)
            for _idx, (_ref, _subj) in enumerate(
                zip(qwen_input_refs, qwen_input_subjects)
            ):
                if len(_picked_indices) >= 4:
                    break
                if _idx in _placed:
                    continue
                _key = _subj.strip().lower() or "self"
                if _used_per_label.get(_key, 0) >= 2:
                    continue
                _picked_indices.append(_idx)
                _used_per_label[_key] = _used_per_label.get(_key, 0) + 1
        _picked = [qwen_input_refs[i] for i in _picked_indices]
        _picked_subjects = [qwen_input_subjects[i] for i in _picked_indices]
        if len(_picked) < len(qwen_input_refs):
            print(
                f"[ComfyUI] Qwen: subject-aware ref selection trimmed "
                f"{len(qwen_input_refs)} → {len(_picked)} (one photo per subject "
                f"first; cap=4)."
            )
        qwen_input_refs = _picked
        qwen_input_subjects = _picked_subjects

    # ── Lazy auto-diagnose: populate `_QWEN_ENCODER_V2` on first generate ──
    # `diagnose()` is normally invoked by the operator via `/diagcomfyui`,
    # but the very first `!generate` after a process restart can fire
    # before that ever happens. When the cache is still `None` and we
    # actually have a reference to upload, run diagnose() once so the
    # builder can wire `latent_image` correctly on the v2 encoder.
    global _QWEN_ENCODER_V2_WARNED
    if _QWEN_ENCODER_V2 is None and qwen_input_refs:
        print(
            "[ComfyUI] Qwen: encoder version unknown — running auto-diagnose "
            "once to populate the cache before the first edit-mode generate."
        )
        try:
            diagnose()
        except Exception as _diag_exc:
            print(
                f"[ComfyUI] Qwen: auto-diagnose failed "
                f"({type(_diag_exc).__name__}: {_diag_exc}) — proceeding "
                "without `latent_image` wiring."
            )
        _QWEN_ENCODER_V2_WARNED = True  # silence the legacy one-shot warning

    # ── v1 fallback pre-resize ─────────────────────────────────────────────
    # On v1-encoder installs, the encoder has no `latent_image` slot to
    # constrain the reference embedding to the target canvas — references
    # are propagated at their native dimensions and the result drifts
    # toward the reference's aspect ratio. Smart-crop to qwen_width ×
    # qwen_height before upload (matching the FLUX img2img path) so the
    # encoder sees a canvas-shaped reference. Skip on v2 (the encoder
    # handles sizing) and on unknown (diagnose may have been unreachable).
    if _QWEN_ENCODER_V2 is False and qwen_input_refs:
        print(
            f"[ComfyUI] Qwen v1 encoder detected — pre-resizing "
            f"{len(qwen_input_refs)} reference image(s) to "
            f"{qwen_width}x{qwen_height} before upload."
        )
        _resized: List[Tuple[bytes, str]] = []
        for _ref_b, _ref_m in qwen_input_refs:
            _png = _preprocess_reference_image(
                _ref_b, _ref_m, qwen_width, qwen_height
            )
            if _png:
                _resized.append((_png, "image/png"))
            else:
                _resized.append((_ref_b, _ref_m))
        qwen_input_refs = _resized
    elif _QWEN_ENCODER_V2 is None and qwen_input_refs:
        print(
            "[ComfyUI] Qwen: encoder v2 status still unknown after diagnose — "
            "skipping pre-resize. Reference dimensions left untouched."
        )

    uploaded: List[str] = []
    uploaded_subjects: List[str] = []
    for (ref_bytes, _ref_mime), _subj in zip(qwen_input_refs, qwen_input_subjects):
        png_bytes = _to_png(ref_bytes) or ref_bytes
        name = _upload_image(base_url, png_bytes, requests_mod)
        if name:
            uploaded.append(name)
            uploaded_subjects.append(_subj)

    # Fail loud when refs were requested but every upload failed. Falling
    # through to txt2img would silently strip identity from the prompt and
    # produce an off-character image labelled as if it were correct — much
    # worse than returning None, which the scene-image runner can surface
    # to the user as a clear failure.
    if qwen_input_refs and not uploaded:
        print(
            f"[ComfyUI] Qwen: ALL {len(qwen_input_refs)} reference upload(s) "
            f"failed — refusing to silently fall back to txt2img. Returning "
            f"None so the caller can report a clean failure to the user."
        )
        return None

    if uploaded:
        _slot_map = ", ".join(
            f"image{i + 1}={s}" for i, s in enumerate(uploaded_subjects)
        )
        print(f"[ComfyUI] Qwen: subject→slot mapping — {_slot_map}")

    # Dispatch by reference count to the matching named builder, so each
    # variant has an explicit, self-documenting entry point:
    #   0 refs   → _build_txt2img_workflow_qwen
    #   1 ref    → _build_edit_workflow_qwen      (single-image edit-mode)
    #   2–4 refs → _build_multi_edit_workflow_qwen (image1..image4)
    # Resolve the v2-encoder status into a human-readable tag for the
    # per-generate log line.
    if _QWEN_ENCODER_V2 is True:
        _latent_tag = "wired"
    elif _QWEN_ENCODER_V2 is False:
        _latent_tag = "not wired (v1 node)"
    else:
        _latent_tag = "not wired (unknown)"

    if len(uploaded) == 1:
        workflow = _build_edit_workflow_qwen(
            prompt, qwen_gguf, qwen_vae, qwen_clip,
            qwen_steps, qwen_width, qwen_height, seed,
            qwen_sampler, qwen_scheduler, uploaded[0],
        )
        print(f"[ComfyUI] Qwen edit-mode workflow with 1 reference image "
              f"(latent_image: {_latent_tag}).")
    elif len(uploaded) >= 2:
        workflow = _build_multi_edit_workflow_qwen(
            prompt, qwen_gguf, qwen_vae, qwen_clip,
            qwen_steps, qwen_width, qwen_height, seed,
            qwen_sampler, qwen_scheduler, uploaded,
        )
        print(f"[ComfyUI] Qwen multi-edit workflow with {len(uploaded)} "
              f"reference image(s) (latent_image: {_latent_tag}).")
    else:
        workflow = _build_txt2img_workflow_qwen(
            prompt, qwen_gguf, qwen_vae, qwen_clip,
            qwen_steps, qwen_width, qwen_height, seed,
            qwen_sampler, qwen_scheduler,
        )
        print(f"[ComfyUI] Qwen txt2img workflow ({qwen_width}x{qwen_height}).")

    return _submit_and_poll(workflow, base_url, client_id, timeout, requests_mod)


def _run_generate(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]],
    client_id: str,
    reference_images: Optional[list] = None,
    width_override: Optional[int] = None,
    height_override: Optional[int] = None,
    steps_override: Optional[int] = None,
    reference_subjects: Optional[list] = None,
    subject_appearances: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[bytes, str]]:
    """Blocking: submit a job to ComfyUI and return (png_bytes, mime) or None."""
    try:
        import requests as _requests
    except ImportError:
        print("[ComfyUI] 'requests' is not installed. Run: pip install requests")
        return None

    base_url = os.environ.get("COMFYUI_URL", DEFAULT_URL).rstrip("/")
    gguf_path = os.environ.get("COMFYUI_GGUF", "").strip()
    vae_name = os.environ.get("COMFYUI_VAE", "").strip()
    clip_name = os.environ.get("COMFYUI_CLIP", "").strip()
    steps = steps_override if steps_override is not None else _get_int("COMFYUI_STEPS", DEFAULT_STEPS)
    width = width_override if width_override else _get_int("COMFYUI_WIDTH", DEFAULT_WIDTH)
    height = height_override if height_override else _get_int("COMFYUI_HEIGHT", DEFAULT_HEIGHT)
    strength = _get_float("COMFYUI_STRENGTH", DEFAULT_STRENGTH)
    flux_max_shift = _get_float("COMFYUI_FLUX_MAX_SHIFT", 1.15)
    flux_base_shift = _get_float("COMFYUI_FLUX_BASE_SHIFT", 0.5)
    timeout = _get_int("COMFYUI_TIMEOUT", DEFAULT_TIMEOUT)
    custom_workflow_path = os.environ.get("COMFYUI_WORKFLOW", "").strip()
    seed = int(uuid.uuid4().int % (2**31))

    # reference_images (list) → native ReferenceLatent multi-ref path
    # reference_image  (single tuple) → legacy img2img path
    refs = reference_images or []

    # ── Engine routing ────────────────────────────────────────────────────────
    # `qwen` (default) hands the entire request off to `_run_generate_qwen`,
    # which builds + submits a Qwen-Image-Edit-Rapid GGUF workflow on its own.
    # `flux` (and any unknown value, with a warning) falls through to the
    # original FLUX.2 Klein logic below — left byte-for-byte unchanged.
    engine = os.environ.get("COMFYUI_ENGINE", "qwen").strip().lower()
    if engine not in ("qwen", "flux"):
        print(f"[ComfyUI] Unknown COMFYUI_ENGINE={engine!r} — defaulting to qwen.")
        engine = "qwen"

    # Backward-compat fallback: if the user requested `qwen` but only the
    # FLUX vars are set (e.g. they upgraded without configuring Qwen), we can
    # fall back to the FLUX engine. By default this fallback is OFF — we
    # fail fast inside `_run_generate_qwen` so a misconfiguration cannot
    # silently drift the engine in production. To opt back in to the old
    # silent-fallback behavior, set COMFYUI_ALLOW_ENGINE_FALLBACK=1 (true,
    # yes, on are also accepted).
    if engine == "qwen":
        _qwen_set = bool(
            os.environ.get("COMFYUI_QWEN_GGUF", "").strip()
            and os.environ.get("COMFYUI_QWEN_VAE", "").strip()
            and os.environ.get("COMFYUI_QWEN_CLIP_GGUF", "").strip()
        )
        _allow_fallback = os.environ.get("COMFYUI_ALLOW_ENGINE_FALLBACK", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if not _qwen_set and gguf_path and vae_name and clip_name:
            if _allow_fallback:
                print("[ComfyUI] COMFYUI_ENGINE=qwen but Qwen vars unset — "
                      "COMFYUI_ALLOW_ENGINE_FALLBACK is on, falling back to flux engine.")
                engine = "flux"
            else:
                print("[ComfyUI] WARNING: COMFYUI_ENGINE=qwen but Qwen vars are unset "
                      "(COMFYUI_QWEN_GGUF / _VAE / _CLIP_GGUF). Refusing to silently "
                      "drift to the FLUX engine. Either configure the Qwen vars, set "
                      "COMFYUI_ENGINE=flux explicitly, or set "
                      "COMFYUI_ALLOW_ENGINE_FALLBACK=1 to re-enable auto-fallback.")
                # Fall through to qwen path; it will fail-fast with the same
                # missing-vars error message.

    if engine == "qwen":
        return _run_generate_qwen(
            prompt, refs, reference_image, client_id, seed,
            steps_override, width_override, height_override,
            timeout, base_url, _requests,
            reference_subjects=reference_subjects,
        )

    # === FLUX engine (existing behaviour, untouched) ==========================
    if not gguf_path or not vae_name or not clip_name:
        print("[ComfyUI] COMFYUI_GGUF, COMFYUI_VAE, and COMFYUI_CLIP must all be set.")
        return None

    try:
        # Pre-built flat-chain fallback populated when ReferenceChainConditioning is chosen,
        # so that if /prompt rejects the node at runtime we can auto-retry without a second
        # image-upload cycle (images are already on the server).
        _refchain_fallback_wf: Optional[dict] = None
        _ultimate_sam_fallback_wf: Optional[dict] = None

        _workflow_mode = os.environ.get("COMFYUI_MODE", "").strip().lower()
        print(
            f"[ComfyUI] Workflow mode: '{_workflow_mode or 'default'}' — "
            f"refs={len(refs)}, subject_appearances={list((subject_appearances or {}).keys())}"
        )
        print(
            f"[ComfyUI] Models — GGUF: {gguf_path!r}, VAE: {vae_name!r}, CLIP: {clip_name!r}"
        )
        print(f"[ComfyUI] Prompt: {prompt[:200]!r}")

        if custom_workflow_path and os.path.exists(custom_workflow_path):
            with open(custom_workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            print(f"[ComfyUI] Using custom workflow: {custom_workflow_path}")
        elif _workflow_mode == "ultimate_inpaint" and refs:
            # ── Ultimate Inpaint multi-character path ────────────────────────────
            # Uses the Flux.2 Ultimate Inpaint Pro Ultra v3.1 GUI workflow which
            # provides per-character ReferenceLatent + InpaintCropImproved passes
            # with optional SAM3 auto-segmentation. Activated by:
            #   COMFYUI_MODE=ultimate_inpaint
            # Requires a scene/canvas image — generated via a first txt2img pass
            # and automatically uploaded before the inpainting job is submitted.
            try:
                from workflow_adapter import build_ultimate_workflow
                labels = reference_subjects or []
                unique_subjects = list(dict.fromkeys(labels)) if labels else []

                # Step 1: upload character reference images
                subject_uploaded: Dict[str, List[str]] = defaultdict(list)
                uploaded_names: List[str] = []
                for idx, (ref_bytes, ref_mime) in enumerate(refs):
                    processed = _to_png(ref_bytes)
                    up_bytes = processed if processed is not None else ref_bytes
                    up_name = _upload_image(base_url, up_bytes, _requests)
                    if up_name:
                        uploaded_names.append(up_name)
                        subj = labels[idx] if idx < len(labels) else ""
                        if subj:
                            subject_uploaded[subj].append(up_name)

                if not uploaded_names:
                    print("[ComfyUI] Ultimate: all reference uploads failed — falling back to txt2img.")
                    workflow = _build_txt2img_workflow(
                        prompt, gguf_path, vae_name, clip_name, steps, width, height, seed
                    )
                else:
                    # Step 2: generate initial scene with txt2img, then upload it
                    # Cap scene-pass steps at 4: the scene is just a canvas for inpainting;
                    # high step counts risk VRAM OOM on the RTX 3060 without quality benefit.
                    _scene_steps = min(steps, 4)
                    scene_seed = int(uuid.uuid4().int % (2**31))
                    print("[ComfyUI] Ultimate: generating initial scene via txt2img pass…")
                    _scene_wf = _build_txt2img_workflow(
                        prompt, gguf_path, vae_name, clip_name, _scene_steps, width, height, scene_seed
                    )
                    _scene_payload = {"prompt": _scene_wf, "client_id": client_id + "_scene"}
                    _scene_resp = _requests.post(f"{base_url}/prompt", json=_scene_payload, timeout=15)
                    _scene_image_filename: Optional[str] = None
                    if _scene_resp.status_code == 200:
                        _scene_pid = _scene_resp.json().get("prompt_id")
                        _scene_deadline = time.time() + timeout
                        _scene_poll_fails = 0
                        while time.time() < _scene_deadline:
                            time.sleep(2)
                            try:
                                _sh = _requests.get(f"{base_url}/history/{_scene_pid}", timeout=30)
                            except Exception:
                                _scene_poll_fails += 1
                                if _scene_poll_fails >= 30:
                                    print("[ComfyUI] Scene poll: ComfyUI appears down — aborting scene wait.")
                                    break
                                continue  # ComfyUI busy generating; retry next tick
                            if _sh.status_code != 200 or _scene_pid not in _sh.json():
                                _scene_poll_fails += 1
                                if _scene_poll_fails >= 30:
                                    print("[ComfyUI] Scene poll: ComfyUI not responding — aborting scene wait.")
                                    break
                                continue
                            _scene_poll_fails = 0  # reset on good response
                            _sj = _sh.json()[_scene_pid]
                            if not _sj.get("status", {}).get("completed"):
                                continue
                            for _sout in _sj.get("outputs", {}).values():
                                _simgs = _sout.get("images", [])
                                if _simgs:
                                    _si = _simgs[0]
                                    _sv = _requests.get(
                                        f"{base_url}/view",
                                        params={"filename": _si["filename"],
                                                "type": _si.get("type", "output"),
                                                **( {"subfolder": _si["subfolder"]} if _si.get("subfolder") else {} )},
                                        timeout=30,
                                    )
                                    if _sv.status_code == 200:
                                        _scene_image_filename = _upload_image(
                                            base_url, _sv.content, _requests
                                        )
                                    break
                            if _scene_image_filename:
                                break
                        if _scene_image_filename:
                            print(f"[ComfyUI] Ultimate: scene image uploaded as '{_scene_image_filename}'")
                        else:
                            print("[ComfyUI] Ultimate: scene generation failed — using blank canvas.")

                    # Step 3: build blank canvas if scene generation failed
                    if not _scene_image_filename:
                        try:
                            from PIL import Image as _PILImage
                            _blank = _PILImage.new("RGB", (width, height), color=(255, 255, 255))
                            _buf = io.BytesIO()
                            _blank.save(_buf, format="PNG")
                            _scene_image_filename = _upload_image(base_url, _buf.getvalue(), _requests) or "none.png"
                        except Exception:
                            _scene_image_filename = "none.png"

                    # Step 4: build and submit Ultimate Inpaint workflow
                    _per_char_prompts = [
                        subject_appearances.get(s, "") for s in list(subject_uploaded.keys())[:4]
                    ]
                    # SAM3 needs visual descriptions, not character names.
                    # Build a query from appearance data; fall back to "person, girl".
                    _sam_parts = []
                    for _sq_subj in list(subject_uploaded.keys())[:4]:
                        _sq_app = (subject_appearances or {}).get(_sq_subj, "")
                        # Take the first visual noun phrase (hair colour phrase) as the hint
                        _sq_hint = _sq_app.split(",")[0].strip() if _sq_app else ""
                        _sam_parts.append(_sq_hint if _sq_hint else "person")
                    _sam_query = ", ".join(_sam_parts) if _sam_parts else "person, girl"
                    # SAM defaults OFF: the RTX 3060 12 GB has no headroom for SAM3
                    # after the Flux model loads from the scene pass.
                    # Set COMFYUI_USE_SAM=true in tokens.txt to opt in.
                    _use_sam = os.environ.get("COMFYUI_USE_SAM", "false").strip().lower() not in ("0", "false", "no")
                    if _use_sam:
                        # Auto-detect: disable SAM if SAM3Segment isn't registered on this server
                        try:
                            _sam_check = _requests.get(
                                f"{base_url}/object_info/SAM3Segment", timeout=5
                            )
                            if _sam_check.status_code != 200:
                                print("[ComfyUI] SAM3Segment not found on server — disabling SAM automatically.")
                                _use_sam = False
                        except Exception:
                            print("[ComfyUI] SAM3Segment check failed — disabling SAM to be safe.")
                            _use_sam = False
                    if _use_sam:
                        # VRAM guard: loading SAM3 alongside the already-resident Flux
                        # model crashes the RTX 3060 (12 GB) when torch_vram_free < ~2 GB.
                        # Fail-closed: any check failure disables SAM.
                        _VRAM_SAM_MIN_MB = int(os.environ.get("COMFYUI_SAM_MIN_VRAM_MB", "2000"))
                        try:
                            _ss = _requests.get(f"{base_url}/system_stats", timeout=5)
                            if _ss.status_code == 200:
                                _tvf = _ss.json().get("devices", [{}])[0].get("torch_vram_free", 0)
                                _tvf_mb = _tvf // (1024 * 1024)
                                if _tvf_mb < _VRAM_SAM_MIN_MB:
                                    print(
                                        f"[ComfyUI] Only {_tvf_mb} MB torch VRAM free after scene pass "
                                        f"(need {_VRAM_SAM_MIN_MB} MB for SAM) — disabling SAM to prevent OOM crash."
                                    )
                                    _use_sam = False
                                else:
                                    print(f"[ComfyUI] VRAM OK ({_tvf_mb} MB free) — SAM staying enabled.")
                            else:
                                print(f"[ComfyUI] system_stats returned {_ss.status_code} — disabling SAM to be safe.")
                                _use_sam = False
                        except Exception as _vram_exc:
                            print(f"[ComfyUI] VRAM check failed ({_vram_exc}) — disabling SAM to be safe.")
                            _use_sam = False
                    # Cap at 4 steps: the user confirmed 4 gives excellent results and
                    # the per-character SAM inpaint loop is heavy enough that 8+ steps
                    # OOMs the RTX 3060 before any output is produced.
                    _inpaint_steps = min(steps, 4)
                    ultimate_wf = build_ultimate_workflow(
                        overall_prompt=prompt,
                        per_character_prompts=_per_char_prompts,
                        uploaded_filenames=dict(subject_uploaded),
                        scene_image_filename=_scene_image_filename,
                        unet_name=gguf_path,
                        vae_name=vae_name,
                        clip_name=clip_name,
                        seed=seed,
                        steps=_inpaint_steps,
                        width=width,
                        height=height,
                        sam_query=_sam_query,
                        use_sam=_use_sam,
                    )
                    if ultimate_wf is None:
                        print("[ComfyUI] Ultimate: workflow build failed — falling back to multiref.")
                        from workflow_adapter import build_multiref_workflow as _bmr_ult
                        _ult_groups = {k: v for k, v in subject_uploaded.items() if v}
                        if len(_ult_groups) >= 2:
                            workflow = _bmr_ult(
                                scene_prompt=prompt,
                                subject_filenames=_ult_groups,
                                subject_appearances=subject_appearances or {},
                                unet_name=gguf_path,
                                vae_name=vae_name,
                                clip_name=clip_name,
                                seed=seed,
                                steps=steps,
                                width=width,
                                height=height,
                            )
                        else:
                            workflow = _build_reference_workflow(
                                prompt, gguf_path, vae_name, clip_name, steps, seed,
                                uploaded_names, width=width, height=height,
                            )
                    else:
                        workflow = ultimate_wf
                        n_chars = len([v for v in subject_uploaded.values() if v])
                        print(
                            f"[ComfyUI] Ultimate Inpaint workflow — {n_chars} character(s): "
                            f"{list(subject_uploaded.keys())}, scene='{_scene_image_filename}', "
                            f"SAM={'on' if _use_sam else 'off'}, output={width}x{height}"
                        )
                        # Pre-build a SAM-off fallback in case SAM3 returns None at runtime
                        _ultimate_sam_fallback_wf: Optional[dict] = None
                        if _use_sam:
                            _ultimate_sam_fallback_wf = build_ultimate_workflow(
                                overall_prompt=prompt,
                                per_character_prompts=_per_char_prompts,
                                uploaded_filenames=dict(subject_uploaded),
                                scene_image_filename=_scene_image_filename,
                                unet_name=gguf_path,
                                vae_name=vae_name,
                                clip_name=clip_name,
                                seed=seed,
                                steps=_inpaint_steps,
                                width=width,
                                height=height,
                                sam_query=_sam_query,
                                use_sam=False,
                            )
            except Exception as _ult_exc:
                print(f"[ComfyUI] Ultimate Inpaint path failed ({_ult_exc}) — falling back to txt2img.")
                workflow = _build_txt2img_workflow(
                    prompt, gguf_path, vae_name, clip_name, steps, width, height, seed
                )
        elif refs:
            # Native FLUX.2 Klein ReferenceLatent path.
            # Convert all reference images to PNG before upload (avoids JPEG/WebP
            # decode issues inside ComfyUI's LoadImage node).
            labels = reference_subjects or []
            unique_subjects = list(dict.fromkeys(labels)) if labels else []
            n_subjects = len(unique_subjects)

            # Upload all reference images, tracking which subject each belongs to.
            uploaded_names: List[str] = []
            subject_uploaded: Dict[str, List[str]] = defaultdict(list)
            for idx, (ref_bytes, ref_mime) in enumerate(refs):
                processed = _to_png(ref_bytes)
                upload_bytes = processed if processed is not None else ref_bytes
                uploaded_name = _upload_image(base_url, upload_bytes, _requests)
                if uploaded_name:
                    uploaded_names.append(uploaded_name)
                    subj = labels[idx] if idx < len(labels) else ""
                    if subj:
                        subject_uploaded[subj].append(uploaded_name)

            if not uploaded_names:
                print("[ComfyUI] Reference: all uploads failed — falling back to txt2img.")
                workflow = _build_txt2img_workflow(
                    prompt, gguf_path, vae_name, clip_name, steps, width, height, seed
                )
            else:
                if len(uploaded_names) < len(refs):
                    print(f"[ComfyUI] Reference: partial upload ({len(uploaded_names)}/{len(refs)}) — continuing.")

                subject_label = (
                    f"{n_subjects} subject(s): {unique_subjects}"
                    if n_subjects > 1 else
                    (unique_subjects[0] if unique_subjects else "unknown")
                )

                # ── Workflow routing ──────────────────────────────────────────────
                # Mode selection:
                #   default / "multiref" → build_multiref_workflow  (primary, no deps)
                #   "refchain"           → ReferenceChainConditioning (custom node pack,
                #                         multi-subject only; falls back to multiref)
                # build_multiref_workflow is used for ALL subject counts (1 or more).
                _used_multiref = False
                _used_refchain = False

                # Detect contact/hugging pose so 2-char scenes use soft spatial masks
                # (contact_pose=True) instead of hard masks — lets arms cross the
                # centre line naturally while still anchoring each character to their half.
                _CONTACT_KW = (
                    "hug", "hugging", "embracing", "embrace",
                    "holding each other", "arms around", "arms wrapped",
                    "cuddling", "snuggling", "leaning on", "resting on",
                    "leaning against",
                )
                _contact_pose = any(kw in prompt.lower() for kw in _CONTACT_KW)

                if n_subjects >= 2 and len(subject_uploaded) >= 2:
                    if _workflow_mode == "refchain":
                        # ── opt-in ReferenceChainConditioning path (multi only) ────
                        try:
                            from workflow_adapter import build_refchain_workflow
                            _rc_check = _requests.get(
                                f"{base_url}/object_info/ReferenceChainConditioning",
                                timeout=5,
                            )
                            if _rc_check.status_code == 200 and _rc_check.json():
                                workflow = build_refchain_workflow(
                                    prompt=prompt,
                                    subject_filenames=subject_uploaded,
                                    unet_name=gguf_path,
                                    vae_name=vae_name,
                                    clip_name=clip_name,
                                    seed=seed,
                                    steps=steps,
                                    width=width,
                                    height=height,
                                )
                                _used_refchain = True
                                # Pre-build multiref fallback for auto-retry
                                from workflow_adapter import build_multiref_workflow as _bmr_rc
                                _refchain_fallback_wf = _bmr_rc(
                                    scene_prompt=prompt,
                                    subject_filenames=dict(subject_uploaded),
                                    subject_appearances=subject_appearances or {},
                                    unet_name=gguf_path,
                                    vae_name=vae_name,
                                    clip_name=clip_name,
                                    seed=seed,
                                    steps=steps,
                                    width=width,
                                    height=height,
                                    contact_pose=_contact_pose,
                                )
                                _per_char = {n: len(fs) for n, fs in subject_uploaded.items() if fs}
                                print(
                                    f"[ComfyUI] ReferenceChain workflow — {subject_label}, "
                                    f"per-char nodes: {_per_char}, output={width}x{height}"
                                )
                            else:
                                print(
                                    "[ComfyUI] ReferenceChainConditioning not on server "
                                    f"(HTTP {_rc_check.status_code}) — using multiref instead."
                                )
                        except Exception as _rc_exc:
                            print(f"[ComfyUI] ReferenceChain path failed ({_rc_exc}) — using multiref instead.")

                if not _used_refchain and len(subject_uploaded) >= 1:
                    # ── PRIMARY: build_multiref_workflow (1 or more subjects) ──────
                    # Native FLUX.2 Klein ReferenceLatent conditioning:
                    #   • Per-character CLIPTextEncode (scene + appearance text)
                    #   • Per-character isolated ReferenceLatent chain
                    #   • 2-char: ConditioningSetMask spatial split
                    #       hard masks (side-by-side) or soft masks (contact_pose)
                    #   • 1/3+ char: ConditioningCombine (no spatial split)
                    #   • Only built-in ComfyUI nodes — no custom packs required
                    try:
                        from workflow_adapter import build_multiref_workflow as _bmr
                        workflow = _bmr(
                            scene_prompt=prompt,
                            subject_filenames=dict(subject_uploaded),
                            subject_appearances=subject_appearances or {},
                            unet_name=gguf_path,
                            vae_name=vae_name,
                            clip_name=clip_name,
                            seed=seed,
                            steps=steps,
                            width=width,
                            height=height,
                            contact_pose=_contact_pose,
                        )
                        _used_multiref = True
                        _contact_note = " contact-pose" if _contact_pose else ""
                        print(
                            f"[ComfyUI] Multiref{_contact_note} workflow — {subject_label}, "
                            f"output={width}x{height}"
                        )
                        # Pre-build flat-chain fallback for auto-retry on node errors
                        _refchain_fallback_wf = _build_per_subject_ref_workflow(
                            prompt, gguf_path, vae_name, clip_name, steps, seed,
                            [list(v) for v in subject_uploaded.values() if v],
                            width=width, height=height,
                        )
                    except Exception as _mr_exc:
                        print(f"[ComfyUI] Multiref build failed ({_mr_exc}) — falling back to flat chain.")

                if not _used_multiref and not _used_refchain:
                    # Multiref build failed — flat native ReferenceLatent chain fallback.
                    print(
                        f"[ComfyUI] Flat-chain reference mode (fallback) — {subject_label}, "
                        f"{len(uploaded_names)} image(s) total, output={width}x{height}"
                    )
                    workflow = _build_reference_workflow(
                        prompt, gguf_path, vae_name, clip_name, steps, seed, uploaded_names,
                        width=width, height=height,
                    )
        elif reference_image is not None:
            img_bytes, mime = reference_image
            # Pre-process: smart-crop + convert to PNG (matches diffusers_worker.py)
            processed = _preprocess_reference_image(img_bytes, mime, width, height)
            upload_bytes = processed if processed is not None else img_bytes
            uploaded_name = _upload_image(base_url, upload_bytes, _requests)
            if uploaded_name:
                print(f"[ComfyUI] img2img mode — uploaded reference as '{uploaded_name}', strength={strength}, shift=({flux_max_shift},{flux_base_shift})")
                workflow = _build_img2img_workflow(
                    prompt, gguf_path, vae_name, clip_name,
                    steps, width, height, seed, strength, uploaded_name,
                    max_shift=flux_max_shift, base_shift=flux_base_shift,
                )
            else:
                print("[ComfyUI] Image upload failed — falling back to txt2img.")
                workflow = _build_txt2img_workflow(
                    prompt, gguf_path, vae_name, clip_name, steps, width, height, seed
                )
        else:
            workflow = _build_txt2img_workflow(
                prompt, gguf_path, vae_name, clip_name, steps, width, height, seed
            )

        if "36" in workflow:
            mode = "reference"
        elif "12" in workflow:
            mode = "img2img"
        elif "1188" in workflow:
            # Ultimate Inpaint workflow — node 1188 is its SaveImage node
            mode = "ultimate_inpaint"
        else:
            mode = "txt2img"
        size_info = f"{width}x{height}" if mode != "reference" else "auto (from reference)"
        strength_info = strength if mode == "img2img" else "n/a"
        # For the ultimate inpaint mode show the actual capped step count, not the
        # raw configured value — the cap happens before build_ultimate_workflow is called.
        _display_steps = min(steps, 4) if mode == "ultimate_inpaint" else steps
        print(f"[ComfyUI] Queuing {mode} job — size={size_info}, steps={_display_steps}, strength={strength_info}")
        print(f"[ComfyUI] Prompt: {prompt[:120]!r}")

        payload = {"prompt": workflow, "client_id": client_id}
        resp = _requests.post(f"{base_url}/prompt", json=payload, timeout=15)

        if resp.status_code != 200:
            print(f"[ComfyUI] /prompt returned HTTP {resp.status_code}.")
            _refchain_node_failed = False
            try:
                err = resp.json()
                top_error = err.get("error", {})
                if top_error:
                    print(f"[ComfyUI] Error type:    {top_error.get('type', '?')}")
                    print(f"[ComfyUI] Error message: {top_error.get('message', '?')}")
                    details = top_error.get("details", "")
                    if details:
                        print(f"[ComfyUI] Error details: {details}")
                node_errors = err.get("node_errors", {})
                for node_id, node_err in node_errors.items():
                    class_type = workflow.get(node_id, {}).get("class_type", node_id)
                    if class_type == "ReferenceChainConditioning":
                        _refchain_node_failed = True
                    errs = node_err.get("errors", [])
                    for e in errs:
                        print(f"[ComfyUI]   Node {node_id} ({class_type}): "
                              f"[{e.get('type', '?')}] {e.get('message', '')} — {e.get('details', '')}")
                if not top_error and not node_errors:
                    print(f"[ComfyUI] Raw response: {resp.text[:500]}")
            except Exception:
                print(f"[ComfyUI] Raw response: {resp.text[:500]}")

            # Guarded retry: if ReferenceChainConditioning was rejected by the server
            # (e.g. not installed despite /object_info 200), re-submit with the pre-built
            # flat-chain fallback.  Images are already uploaded — no extra upload cycle.
            # prompt_id is extracted below in the normal flow.
            if _refchain_node_failed and _refchain_fallback_wf is not None:
                print("[ComfyUI] ReferenceChainConditioning rejected — auto-retrying with flat-chain fallback.")
                workflow = _refchain_fallback_wf
                _refchain_fallback_wf = None
                payload = {"prompt": workflow, "client_id": client_id}
                resp = _requests.post(f"{base_url}/prompt", json=payload, timeout=15)
                if resp.status_code != 200:
                    print(f"[ComfyUI] Flat-chain fallback also failed (HTTP {resp.status_code}).")
                    return None
                # Fall through — prompt_id extracted below from resp
            else:
                return None

        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            print(f"[ComfyUI] Server did not return a prompt_id: {resp.text[:200]}")
            return None
        print(f"[ComfyUI] Job queued — prompt_id={prompt_id}")

        deadline = time.time() + timeout
        last_log = time.time()
        _consecutive_poll_failures = 0
        # After this many consecutive failures (~60 s) we assume ComfyUI crashed.
        _MAX_CONSECUTIVE_FAILURES = 30
        while time.time() < deadline:
            time.sleep(2)

            now = time.time()
            if now - last_log >= 15:
                elapsed = int(now - (deadline - timeout))
                try:
                    q_resp = _requests.get(f"{base_url}/queue", timeout=5)
                    if q_resp.status_code == 200:
                        q = q_resp.json()
                        running = [e for e in q.get("queue_running", []) if len(e) > 1 and e[1] == prompt_id]
                        pending = [e for e in q.get("queue_pending", []) if len(e) > 1 and e[1] == prompt_id]
                        if running:
                            print(f"[ComfyUI] Still generating… ({elapsed}s elapsed, job is running)")
                        elif pending:
                            print(f"[ComfyUI] Still waiting… ({elapsed}s elapsed, job is queued)")
                        else:
                            print(f"[ComfyUI] Still waiting… ({elapsed}s elapsed)")
                    else:
                        print(f"[ComfyUI] Still waiting… ({elapsed}s elapsed)")
                except Exception:
                    print(f"[ComfyUI] Still waiting… ({elapsed}s elapsed)")
                last_log = now

            try:
                hist_resp = _requests.get(f"{base_url}/history/{prompt_id}", timeout=30)
            except Exception:
                # ComfyUI is busy generating — its HTTP server may not respond mid-run;
                # count consecutive failures so we can detect a crash.
                _consecutive_poll_failures += 1
                if _consecutive_poll_failures >= _MAX_CONSECUTIVE_FAILURES:
                    print(f"[ComfyUI] No response for {_consecutive_poll_failures * 2}s "
                          f"— ComfyUI appears to have crashed. Aborting.")
                    return None
                continue
            if hist_resp.status_code != 200:
                _consecutive_poll_failures += 1
                if _consecutive_poll_failures >= _MAX_CONSECUTIVE_FAILURES:
                    print(f"[ComfyUI] ComfyUI returning errors for {_consecutive_poll_failures * 2}s "
                          f"— aborting.")
                    return None
                continue
            # Got a real HTTP response — reset crash counter
            _consecutive_poll_failures = 0
            history = hist_resp.json()
            if prompt_id not in history:
                continue

            job = history[prompt_id]
            status = job.get("status", {})
            completed = status.get("completed", False)
            status_str = status.get("status_str", "")

            is_error = status_str == "error"
            if not completed and not is_error:
                continue

            if is_error:
                messages = status.get("messages", [])
                _sam_caused_failure = False
                for msg in messages:
                    if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
                        err = msg[1]
                        exc_msg = err.get("exception_message", "")
                        node_type = err.get("node_type", "")
                        print(f"[ComfyUI] Execution error in node {err.get('node_id')} "
                              f"({node_type}): {exc_msg}")
                        # Detect SAM3 returning None mask (NoneType has no attribute reshape/mul/etc.)
                        if node_type in ("GrowMaskWithBlur", "MaskToSEGS", "SAM3Segment") and (
                            "NoneType" in exc_msg or "None" in exc_msg
                        ):
                            _sam_caused_failure = True
                # Auto-retry without SAM if SAM3 produced a None mask
                if _sam_caused_failure and _ultimate_sam_fallback_wf is not None:
                    print("[ComfyUI] SAM3 returned None mask — auto-retrying with SAM disabled.")
                    _ultimate_sam_fallback_wf_copy = _ultimate_sam_fallback_wf
                    _ultimate_sam_fallback_wf = None  # prevent infinite retry
                    payload = {"prompt": _ultimate_sam_fallback_wf_copy, "client_id": client_id + "_nosam"}
                    resp2 = _requests.post(f"{base_url}/prompt", json=payload, timeout=15)
                    if resp2.status_code == 200:
                        prompt_id = resp2.json().get("prompt_id")
                        if prompt_id:
                            deadline = time.time() + timeout
                            last_log = time.time()
                            continue  # re-enter polling loop with new prompt_id
                    print("[ComfyUI] SAM-off retry also failed.")
                    return None
                print(f"[ComfyUI] Job failed with status 'error'.")
                return None

            outputs = job.get("outputs", {})
            # Prefer type=output (SaveImage) over type=temp (PreviewImage) so we
            # always return the final stitched result, not an intermediate frame.
            def _img_sort_key(item):
                imgs = item[1].get("images", [])
                if not imgs:
                    return 1
                return 0 if imgs[0].get("type", "output") == "output" else 1
            for _node_id, node_out in sorted(outputs.items(), key=_img_sort_key):
                images = node_out.get("images", [])
                if images:
                    img_info = images[0]
                    filename = img_info["filename"]
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")
                    params = {"filename": filename, "type": img_type}
                    if subfolder:
                        params["subfolder"] = subfolder
                    view_resp = _requests.get(
                        f"{base_url}/view",
                        params=params,
                        timeout=30,
                    )
                    view_resp.raise_for_status()
                    image_bytes = view_resp.content
                    print(f"[ComfyUI] Done — {len(image_bytes)} bytes ('{filename}', type={img_type!r})")
                    return image_bytes, "image/png"

            print(f"[ComfyUI] Job completed but outputs contained no images.")
            print(f"[ComfyUI] Raw outputs: {json.dumps(outputs)[:400]}")
            return None

        print(f"[ComfyUI] Timed out after {timeout}s waiting for prompt_id={prompt_id}")
        return None

    except Exception as exc:
        print(f"[ComfyUI] Generation error: {type(exc).__name__}: {exc}")
        return None


def _upload_image(base_url: str, img_bytes: bytes, requests_mod) -> Optional[str]:
    """Upload PNG bytes to ComfyUI /upload/image; return the server filename or None."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            resp = requests_mod.post(
                f"{base_url}/upload/image",
                files={"image": ("reference.png", f, "image/png")},
                data={"type": "input", "overwrite": "true"},
                timeout=30,
            )
        os.unlink(tmp_path)
        resp.raise_for_status()
        name = resp.json().get("name")
        return name
    except Exception as exc:
        print(f"[ComfyUI] Upload error: {type(exc).__name__}: {exc}")
        return None


def image_ready() -> bool:
    """Return True if ComfyUI is reachable via GET /system_stats."""
    try:
        import requests as _requests
        base_url = os.environ.get("COMFYUI_URL", DEFAULT_URL).rstrip("/")
        resp = _requests.get(f"{base_url}/system_stats", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ── Module-level cache: TextEncodeQwenImageEditPlus v2 detection ─────────────
# Populated by diagnose(). True  → installed node accepts a `latent_image`
#                                  input (Phr00t's v2 fork; fixes the
#                                  scaling/cropping/zoom artifacts).
#                         False → v1-style node, no `latent_image` slot.
#                         None  → not probed yet (diagnose hasn't run, or
#                                 ComfyUI was unreachable). Builders default
#                                 to NOT wiring `latent_image` so a v1 install
#                                 isn't poisoned by an unknown input.
_QWEN_ENCODER_V2: Optional[bool] = None
# One-shot warning latch so the "wiring decision made before diagnose ran"
# warning prints once per process, not per !generate.
_QWEN_ENCODER_V2_WARNED: bool = False


# ── Custom-node packs required per engine ─────────────────────────────────────
# Used by diagnose() below. Each entry is (node_class_name, install_hint).
_REQUIRED_NODES_QWEN = [
    ("UnetLoaderGGUF",
     "city96's `ComfyUI-GGUF` "
     "(https://github.com/city96/ComfyUI-GGUF)"),
    ("CLIPLoaderGGUF",
     "city96's `ComfyUI-GGUF` "
     "(https://github.com/city96/ComfyUI-GGUF)"),
    ("TextEncodeQwenImageEditPlus",
     "any pack shipping `TextEncodeQwenImageEditPlus` "
     "(e.g. Phr00t's `comfyui_qwen_image_edit_plus_nodes`)"),
]
_REQUIRED_NODES_FLUX = [
    ("UnetLoaderGGUF",
     "city96's `ComfyUI-GGUF` "
     "(https://github.com/city96/ComfyUI-GGUF)"),
    ("Flux2Scheduler",
     "ComfyUI core 0.8.2+ (update ComfyUI itself)"),
    ("EmptyFlux2LatentImage",
     "ComfyUI core 0.8.2+ (update ComfyUI itself)"),
]


def _has_qwen_encoder_latent_input(payload: dict) -> bool:
    """Return True iff the /object_info payload for `TextEncodeQwenImageEditPlus`
    advertises a `latent_image` input slot — the marker for Phr00t's v2 fork
    of `nodes_qwen.py`. The slot may live under `required` or `optional`
    depending on how the pack author registered it, so we check both.
    Returns False on any unexpected shape; this defaults the workflow builder
    to v1-safe behaviour rather than poisoning the prompt with an unknown input.
    """
    try:
        node_info = next(iter(payload.values())) if payload else {}
        inputs = node_info.get("input", {}) or {}
        for bucket in ("required", "optional"):
            slots = inputs.get(bucket) or {}
            if isinstance(slots, dict) and "latent_image" in slots:
                return True
    except Exception:
        pass
    return False


def _extract_clip_loader_types(payload: dict) -> list:
    """Pull the list of `type` choices out of a /object_info/CLIPLoaderGGUF
    response. ComfyUI nests the inputs under
    payload[node_name]["input"]["required"]["type"] which is itself a list
    whose first element is the choices list. Returns an empty list if the
    shape doesn't match what we expect (e.g. older ComfyUI build)."""
    try:
        # /object_info/<NodeName> returns {"<NodeName>": {...info...}}
        node_info = next(iter(payload.values())) if payload else {}
        required = node_info.get("input", {}).get("required", {})
        type_field = required.get("type")
        if isinstance(type_field, list) and type_field and isinstance(type_field[0], list):
            return list(type_field[0])
    except Exception:
        pass
    return []


def diagnose() -> dict:
    """Probe ComfyUI's `/object_info` to verify the custom nodes required by
    the active `COMFYUI_ENGINE` are installed.

    Hits `/object_info/<NodeName>` once per required node. Prints a concise
    console summary (✅ found / ❌ missing per node) plus, for the Qwen engine,
    the list of `type` choices exposed by `CLIPLoaderGGUF` so the user can
    confirm `qwen_image` is one of them.

    Does not raise. Always returns a structured dict so the `/diagcomfyui`
    Discord command can render the same information in chat:

        {
            "engine":         "qwen" | "flux",
            "base_url":       "http://...",
            "reachable":      bool,                       # /system_stats responded
            "nodes":          {"NodeName": True/False, ...},
            "missing":        [(node_name, install_hint), ...],
            "clip_types":     [...],   # only for the Qwen engine; [] otherwise
            "disabled_packs": [...],   # `<pack>.disabled` folder names found
                                       # in <COMFYUI_PATH>/custom_nodes/ —
                                       # populated only when COMFYUI_PATH is
                                       # set; empty otherwise.
            "vram_free_mb":   int|None,  # devices[0].torch_vram_free / 1MB,
            "vram_total_mb":  int|None,  # devices[0].torch_vram_total / 1MB.
                                         # Both None when /system_stats does
                                         # not expose device info.
            "qwen_encoder_v2": bool|None,  # True  → installed
                                           # `TextEncodeQwenImageEditPlus`
                                           # accepts a `latent_image` input
                                           # (Phr00t v2 fork — fixes
                                           # scaling/cropping artifacts).
                                           # False → v1-style node, no slot.
                                           # None  → node missing or unparsed.
        }

    Side effect: updates the module-level `_QWEN_ENCODER_V2` cache so that
    `_build_multi_edit_workflow_qwen` can decide whether to wire `latent_image`
    without re-probing /object_info on every generate.
    """
    base_url = os.environ.get("COMFYUI_URL", DEFAULT_URL).rstrip("/")
    engine = os.environ.get("COMFYUI_ENGINE", "qwen").strip().lower()
    if engine not in ("qwen", "flux"):
        engine = "qwen"
    required = _REQUIRED_NODES_QWEN if engine == "qwen" else _REQUIRED_NODES_FLUX

    result: dict = {
        "engine":         engine,
        "base_url":       base_url,
        "reachable":      False,
        "nodes":          {},
        "missing":        [],
        "clip_types":     [],
        "disabled_packs": [],
        "vram_free_mb":   None,
        "vram_total_mb":  None,
        "qwen_encoder_v2": None,
    }

    # --- Engine-scoped pack scan (best-effort, never raises) -----------------
    # When COMFYUI_PATH is set, list folders inside custom_nodes/ that end in
    # `.disabled` so the user can verify start.bat's engine-scoped toggling
    # took effect. ComfyUI itself skips these folders during its custom-node
    # scan, so they consume zero VRAM.
    comfy_path = os.environ.get("COMFYUI_PATH", "").strip()
    if comfy_path:
        custom_nodes_dir = os.path.join(comfy_path, "custom_nodes")
        if os.path.isdir(custom_nodes_dir):
            try:
                disabled = sorted(
                    entry[: -len(".disabled")]
                    for entry in os.listdir(custom_nodes_dir)
                    if entry.endswith(".disabled")
                    and os.path.isdir(os.path.join(custom_nodes_dir, entry))
                )
                result["disabled_packs"] = disabled
            except OSError as exc:
                print(f"[ComfyUI] Diagnose: could not scan {custom_nodes_dir} "
                      f"for .disabled packs ({type(exc).__name__}: {exc})")

    print(f"[ComfyUI] Diagnose: engine={engine}  url={base_url}")
    if result["disabled_packs"]:
        _names = ", ".join(result["disabled_packs"])
        print(f"[ComfyUI] Engine-scoped packs disabled: "
              f"{len(result['disabled_packs'])} ({_names})")
    elif comfy_path:
        print("[ComfyUI] Engine-scoped packs disabled: 0")

    try:
        import requests as _requests
    except Exception as exc:
        print(f"[ComfyUI] Diagnose skipped — `requests` not importable: {exc}")
        return result

    try:
        ss = _requests.get(f"{base_url}/system_stats", timeout=5)
        result["reachable"] = (ss.status_code == 200)
    except Exception as exc:
        print(f"[ComfyUI] Diagnose: server unreachable at {base_url} ({type(exc).__name__}: {exc})")
        return result

    if not result["reachable"]:
        print(f"[ComfyUI] Diagnose: /system_stats returned HTTP {ss.status_code} — skipping node probe.")
        return result

    # --- VRAM readout from /system_stats (best-effort, never raises) ---------
    # Lets the user see at a glance how much VRAM is currently free vs. total
    # — useful for confirming the start.bat memory-mode pick (--normalvram
    # etc.) is leaving enough headroom for the active engine's workflow.
    # Fields are bytes per ComfyUI's API; we convert to MiB for display.
    try:
        _devs = ss.json().get("devices") or []
        if _devs and isinstance(_devs[0], dict):
            _free = _devs[0].get("torch_vram_free")
            _total = _devs[0].get("torch_vram_total")
            if isinstance(_free, (int, float)) and _free >= 0:
                result["vram_free_mb"] = int(_free) // (1024 * 1024)
            if isinstance(_total, (int, float)) and _total > 0:
                result["vram_total_mb"] = int(_total) // (1024 * 1024)
        if result["vram_free_mb"] is not None and result["vram_total_mb"] is not None:
            print(f"[ComfyUI] VRAM: {result['vram_free_mb']} MB free "
                  f"/ {result['vram_total_mb']} MB total")
    except Exception as exc:
        print(f"[ComfyUI] Diagnose: could not parse VRAM from /system_stats "
              f"({type(exc).__name__}: {exc})")

    _node_payloads: dict = {}  # node_name -> parsed /object_info JSON for present nodes
    for node_name, _hint in required:
        try:
            r = _requests.get(f"{base_url}/object_info/{node_name}", timeout=5)
            _rjson = r.json() if r.status_code == 200 else {}
            present = bool(_rjson)
        except Exception as exc:
            print(f"[ComfyUI] Diagnose: probe for {node_name} failed ({type(exc).__name__}: {exc})")
            present = False
            _rjson = {}
        result["nodes"][node_name] = present
        if present:
            _node_payloads[node_name] = _rjson
        marker = "✅ found" if present else "❌ MISSING"
        print(f"[ComfyUI] Diagnose:   {marker}  {node_name}")
        if not present:
            result["missing"].append((node_name, _hint))
        elif node_name == "CLIPLoaderGGUF":
            try:
                result["clip_types"] = _extract_clip_loader_types(r.json())
            except Exception:
                result["clip_types"] = []
        elif node_name == "TextEncodeQwenImageEditPlus":
            # v2 detection: Phr00t's v2 fork advertises a `latent_image`
            # input slot that fixes scaling/cropping/zoom artifacts on
            # multi-reference edits. Cache the result module-level so the
            # workflow builder can wire it without re-probing per generate.
            global _QWEN_ENCODER_V2
            try:
                is_v2 = _has_qwen_encoder_latent_input(r.json())
            except Exception:
                is_v2 = False
            result["qwen_encoder_v2"] = is_v2
            _QWEN_ENCODER_V2 = is_v2
            if is_v2:
                print("[ComfyUI] Diagnose:   ✅ TextEncodeQwenImageEditPlus v2 — "
                      "`latent_image` input present (scaling fix active).")
            else:
                print("[ComfyUI] Diagnose:   ⚠ TextEncodeQwenImageEditPlus v1 — "
                      "no `latent_image` input; falling back to legacy mode "
                      "(install Phr00t's v2 nodes_qwen.py to fix scaling/"
                      "cropping/zoom artifacts).")

    if engine == "qwen" and result["nodes"].get("CLIPLoaderGGUF"):
        types = result["clip_types"]
        if types:
            qwen_ok = "qwen_image" in types
            qmark = "✅" if qwen_ok else "❌"
            print(f"[ComfyUI] Diagnose:   {qmark} CLIPLoaderGGUF type choices: {types}")
            if not qwen_ok:
                print("[ComfyUI] Diagnose:   ⚠ `qwen_image` is NOT one of the CLIPLoaderGGUF "
                      "types — update city96's ComfyUI-GGUF to a recent commit.")
        else:
            print("[ComfyUI] Diagnose:   ⚠ Could not parse CLIPLoaderGGUF type choices "
                  "(unexpected /object_info shape).")

    if result["missing"]:
        names = ", ".join(n for n, _ in result["missing"])
        print(f"[ComfyUI] Diagnose: ⚠ MISSING required custom nodes for engine={engine}: {names}")
        for node_name, hint in result["missing"]:
            print(f"[ComfyUI] Diagnose:     • install `{node_name}` via {hint}")
        print("[ComfyUI] Diagnose: image generation requests will fail with a "
              "/prompt node-not-found 400 until these are installed.")
    else:
        print(f"[ComfyUI] Diagnose: ✅ all required custom nodes present for engine={engine}.")

    # --- Model-file reachability checks (qwen engine only) -------------------
    # Probe /object_info/<LoaderNode> for the exact filenames the env vars name
    # and confirm each one appears in ComfyUI's recognised file list.  This
    # catches "file is in the wrong directory" problems BEFORE the first
    # generate attempt, so the user sees a clear path to fix rather than an
    # opaque HTTP 400 value_not_in_list error.
    if engine == "qwen" and result["reachable"]:
        _qwen_gguf   = os.environ.get("COMFYUI_QWEN_GGUF",      "").strip()
        _qwen_vae    = os.environ.get("COMFYUI_QWEN_VAE",        "").strip()
        _qwen_clip   = os.environ.get("COMFYUI_QWEN_CLIP_GGUF",  "").strip()
        _comfy_path  = os.environ.get("COMFYUI_PATH",            "").strip()

        def _check_model_file(loader_node: str, input_field: str,
                              expected_name: str, fallback_dir: str,
                              payload_override: "Optional[dict]" = None) -> bool:
            """Return True if *expected_name* is in the loader's choices list."""
            try:
                if payload_override is not None:
                    _payload = payload_override
                else:
                    _r2 = _requests.get(f"{base_url}/object_info/{loader_node}", timeout=5)
                    _payload = _r2.json() if _r2.status_code == 200 else {}
                _node_info = next(iter(_payload.values())) if _payload else {}
                _field = (_node_info.get("input", {}).get("required", {}).get(input_field)
                          or _node_info.get("input", {}).get("optional", {}).get(input_field))
                if isinstance(_field, list) and _field and isinstance(_field[0], list):
                    choices = _field[0]
                else:
                    return False  # unexpected shape — skip check
                found = expected_name in choices
                _model_dir = (os.path.join(_comfy_path, fallback_dir)
                              if _comfy_path else f"<ComfyUI>/{fallback_dir}/")
                if found:
                    print(f"[ComfyUI] Diagnose:   ✅ {input_field} '{expected_name}' found.")
                else:
                    print(f"[ComfyUI] Diagnose:   ❌ {input_field} '{expected_name}' NOT found in ComfyUI.")
                    print(f"[ComfyUI] Diagnose:      Place it in: {_model_dir}")
                    print(f"[ComfyUI] Diagnose:      Then restart ComfyUI (close it and re-run start.bat).")
                return found
            except Exception as _exc:
                print(f"[ComfyUI] Diagnose:   ⚠ Could not check {input_field}: {_exc}")
                return True  # don't block on probe failure

        print("[ComfyUI] Diagnose: checking model files...")
        _gguf_ok = True
        _vae_ok  = True
        _clip_ok = True
        if _qwen_gguf:
            _gguf_ok = _check_model_file(
                "UnetLoaderGGUF", "unet_name", _qwen_gguf,
                os.path.join("models", "unet"),
                payload_override=_node_payloads.get("UnetLoaderGGUF"),
            )
        if _qwen_vae:
            _vae_ok = _check_model_file(
                "VAELoader", "vae_name", _qwen_vae,
                os.path.join("models", "vae"),
            )
            if not _vae_ok and _qwen_vae.lower().endswith(".gguf"):
                _st_name = _qwen_vae[:-5] + ".safetensors"
                print(
                    f"[ComfyUI] HINT: '{_qwen_vae}' is a GGUF VAE and cannot be loaded "
                    f"by the stock VAELoader (VAELoaderGGUF does not exist in city96's pack).\n"
                    f"[ComfyUI] HINT: Convert it once:  python scripts/convert_pigvae.py\n"
                    f"[ComfyUI] HINT: Then set COMFYUI_QWEN_VAE={_st_name} in tokens.txt"
                )
        if _qwen_clip:
            _clip_ok = _check_model_file(
                "CLIPLoaderGGUF", "clip_name", _qwen_clip,
                os.path.join("models", "clip"),
                payload_override=_node_payloads.get("CLIPLoaderGGUF"),
            )
        if _gguf_ok and _vae_ok and _clip_ok:
            print("[ComfyUI] Diagnose: ✅ all model files found -- image generation should work.")
        else:
            print("[ComfyUI] Diagnose: ❌ one or more model files missing from ComfyUI's model list.")
            print("[ComfyUI] Diagnose:    Fix the paths above, restart ComfyUI, then retry.")

    return result


# ── Boot-time pre-warm ────────────────────────────────────────────────────────
# Defaults for COMFYUI_PREWARM when the user has not explicitly set it:
# Qwen-Image-Edit-Rapid is the new default engine and benefits the most from a
# hot model on the very first !generate, so it warms up by default. FLUX.2
# Klein loads a heavier stack and is more likely to be running on tighter VRAM
# budgets, so it stays opt-in.
_PREWARM_DEFAULTS = {"qwen": True, "flux": False}


def _prewarm_enabled(engine: str) -> bool:
    """Return whether boot-time pre-warm should run for the given engine.

    Honors COMFYUI_PREWARM (`1/true/yes/on` or `0/false/no/off`); when the var
    is unset or unparseable, falls back to the per-engine default in
    `_PREWARM_DEFAULTS`.
    """
    raw = os.environ.get("COMFYUI_PREWARM", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return _PREWARM_DEFAULTS.get(engine, False)


def prewarm() -> dict:
    """Submit a tiny throwaway txt2img job so the active engine's model is
    already resident in VRAM before the first real `!generate` request.

    Blocking — meant to be called from `asyncio.to_thread(...)` in `on_ready`
    so it runs concurrently with the rest of bot startup. Never raises: any
    failure is logged as a single WARN line and the function returns a dict
    describing what happened. The generated image (if any) is discarded.

    Returns:
        {
            "engine":  "qwen" | "flux",
            "ran":     bool,                  # True if a /prompt was submitted
            "ok":      bool,                  # True only on full success
            "skipped": str | None,            # human-readable reason, if any
        }
    """
    base_url = os.environ.get("COMFYUI_URL", DEFAULT_URL).rstrip("/")
    engine = os.environ.get("COMFYUI_ENGINE", "qwen").strip().lower()
    if engine not in ("qwen", "flux"):
        engine = "qwen"

    result = {"engine": engine, "ran": False, "ok": False, "skipped": None}

    if not _prewarm_enabled(engine):
        print(f"[ComfyUI] WARMUP: disabled for engine={engine} "
              f"(COMFYUI_PREWARM={os.environ.get('COMFYUI_PREWARM', '<unset>')!r}). "
              f"First !generate will pay the model-load cost as before.")
        result["skipped"] = "disabled"
        return result

    try:
        import requests as _requests
    except Exception as exc:
        print(f"[ComfyUI] WARMUP WARN: `requests` not importable ({exc}) — skipping.")
        result["skipped"] = "no-requests"
        return result

    # Engine-specific tiny workflow — same builders the real pipeline uses, so
    # the model that gets loaded is exactly the one the user's first request
    # will need. Smallest sensible resolution (256x256 — multiple of 64 keeps
    # both EmptySD3LatentImage / EmptyLatentImage happy) and a single sampling
    # step keep the wall-clock cost to roughly "model load + 1 step".
    SEED = 1
    STEPS = 1
    WIDTH = 256
    HEIGHT = 256
    PROMPT = "a"  # one token — content doesn't matter, the load does

    if engine == "qwen":
        qwen_gguf = os.environ.get("COMFYUI_QWEN_GGUF", "").strip()
        qwen_vae = os.environ.get("COMFYUI_QWEN_VAE", "").strip()
        qwen_clip = os.environ.get("COMFYUI_QWEN_CLIP_GGUF", "").strip()
        if not qwen_gguf or not qwen_vae or not qwen_clip:
            print("[ComfyUI] WARMUP WARN: Qwen vars (COMFYUI_QWEN_GGUF/VAE/CLIP_GGUF) "
                  "not set — skipping pre-warm.")
            result["skipped"] = "qwen-vars-missing"
            return result
        sampler = os.environ.get("COMFYUI_QWEN_SAMPLER", DEFAULT_QWEN_SAMPLER).strip() or DEFAULT_QWEN_SAMPLER
        scheduler = os.environ.get("COMFYUI_QWEN_SCHEDULER", DEFAULT_QWEN_SCHEDULER).strip() or DEFAULT_QWEN_SCHEDULER
        workflow = _build_txt2img_workflow_qwen(
            PROMPT, qwen_gguf, qwen_vae, qwen_clip,
            STEPS, WIDTH, HEIGHT, SEED, sampler, scheduler,
        )
    else:  # flux
        gguf_path = os.environ.get("COMFYUI_GGUF", "").strip()
        vae_name = os.environ.get("COMFYUI_VAE", "").strip()
        clip_name = os.environ.get("COMFYUI_CLIP", "").strip()
        if not gguf_path or not vae_name or not clip_name:
            print("[ComfyUI] WARMUP WARN: FLUX vars (COMFYUI_GGUF/VAE/CLIP) "
                  "not set — skipping pre-warm.")
            result["skipped"] = "flux-vars-missing"
            return result
        workflow = _build_txt2img_workflow(
            PROMPT, gguf_path, vae_name, clip_name,
            STEPS, WIDTH, HEIGHT, SEED,
        )

    # The shared txt2img builders end with a `SaveImage` node ("9") that
    # writes a PNG into ComfyUI's `output/` directory. For a throwaway
    # warm-up that's pointless clutter — a stale `ariabot_*.png` would
    # accumulate in the user's outputs folder every boot. `PreviewImage`
    # has the same single `images` input but routes the result to ComfyUI's
    # `temp/` directory (which it auto-clears), so the workflow still
    # terminates correctly without polluting `output/`.
    save_node = workflow.get("9")
    if save_node and save_node.get("class_type") == "SaveImage":
        workflow["9"] = {
            "class_type": "PreviewImage",
            "inputs": {"images": save_node["inputs"]["images"]},
        }

    print("=" * 68)
    print(f"[ComfyUI] WARMUP: pre-loading engine={engine} model into VRAM")
    print(f"[ComfyUI] WARMUP:   throwaway txt2img — {WIDTH}x{HEIGHT}, {STEPS} step, result discarded.")
    print(f"[ComfyUI] WARMUP:   Expect a one-time VRAM spike — this is NOT a real request.")
    print("=" * 68)

    client_id = f"prewarm-{uuid.uuid4().hex[:8]}"
    timeout = _get_int("COMFYUI_TIMEOUT", DEFAULT_TIMEOUT)
    t0 = time.time()
    try:
        out = _submit_and_poll(workflow, base_url, client_id, timeout, _requests)
    except Exception as exc:
        print(f"[ComfyUI] WARMUP WARN: pre-warm raised "
              f"{type(exc).__name__}: {exc} — first !generate will be slow.")
        result["ran"] = True
        return result

    result["ran"] = True
    elapsed = int(time.time() - t0)
    if out is None:
        print(f"[ComfyUI] WARMUP WARN: pre-warm job did not complete after {elapsed}s — "
              f"first !generate will pay the full model-load cost.")
        return result

    result["ok"] = True
    print(f"[ComfyUI] WARMUP: ✅ engine={engine} model is now hot in VRAM "
          f"(took {elapsed}s, throwaway image discarded). "
          f"First !generate should respond promptly.")
    return result


async def _ws_progress(
    ws_url: str,
    on_progress: Callable[[str], Coroutine],
) -> None:
    """Connect to ComfyUI's WebSocket and forward live progress tags.

    Tag mapping (matches diffusers_worker.py / bot.py _format_diffuser_progress):
        execution_start          → STAGE:loading  then fake LOAD ticks every ~10 s
        first progress message   → STAGE:ready    (model fully loaded)
        progress value/max       → STEP:{value}/{max}
        all steps done           → STAGE:encoding
        execution_success / null → stop

    Falls back silently if the websockets package is unavailable or the
    connection cannot be established.
    """
    try:
        import websockets
    except ImportError:
        print("[ComfyUI] 'websockets' not installed — progress bar disabled.")
        return

    async def _fake_load_ticker(stop: asyncio.Event) -> None:
        """Emit approximate LOAD fractions while the model is loading."""
        schedule = [(0, 0.05), (10, 0.40), (30, 0.80), (60, 0.95)]
        t0 = asyncio.get_event_loop().time()
        for delay, frac in schedule:
            wait = delay - (asyncio.get_event_loop().time() - t0)
            if wait > 0:
                try:
                    await asyncio.wait_for(asyncio.shield(stop.wait()), timeout=wait)
                    return
                except asyncio.TimeoutError:
                    pass
            if stop.is_set():
                return
            try:
                await on_progress(f"LOAD:{frac:.2f}")
            except Exception:
                pass

    load_stop = asyncio.Event()
    load_task: Optional[asyncio.Task] = None
    first_step_seen = False

    try:
        async with websockets.connect(ws_url, ping_interval=20, open_timeout=10) as ws:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue

                msg_type = msg.get("type", "")
                data = msg.get("data", {})

                if msg_type == "execution_start":
                    try:
                        await on_progress("STAGE:loading")
                    except Exception:
                        pass
                    load_task = asyncio.create_task(_fake_load_ticker(load_stop))

                elif msg_type == "progress":
                    value = data.get("value", 0)
                    max_val = data.get("max", 1)

                    if not first_step_seen:
                        first_step_seen = True
                        load_stop.set()
                        if load_task and not load_task.done():
                            load_task.cancel()
                        try:
                            await on_progress("STAGE:ready")
                        except Exception:
                            pass

                    try:
                        await on_progress(f"STEP:{value}/{max_val}")
                    except Exception:
                        pass

                    if value >= max_val:
                        try:
                            await on_progress("STAGE:encoding")
                        except Exception:
                            pass

                elif msg_type in ("execution_success", "execution_error"):
                    break

                elif msg_type == "executing" and data.get("node") is None:
                    break

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"[ComfyUI] WebSocket progress error: {type(exc).__name__}: {exc}")
    finally:
        load_stop.set()
        if load_task and not load_task.done():
            load_task.cancel()


async def generate_image(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]] = None,
    reference_images: Optional[list] = None,
    on_progress: Optional[Callable[[str], Coroutine]] = None,
    width_override: Optional[int] = None,
    height_override: Optional[int] = None,
    steps_override: Optional[int] = None,
    reference_subjects: Optional[list] = None,
    subject_appearances: Optional[Dict[str, str]] = None,
) -> Optional[Tuple[bytes, str]]:
    """Generate an image via a locally-running ComfyUI instance.

    The active engine is selected by the `COMFYUI_ENGINE` env var:
        qwen (default) — Qwen-Image-Edit-Rapid AIO GGUF, edit-mode via
                         TextEncodeQwenImageEditPlus (multimodal: model
                         literally consumes the reference pixels).
        flux           — FLUX.2 Klein 4B GGUF with the native ReferenceLatent
                         multiref / Ultimate-Inpaint workflows.
    See the module docstring for the full env-var list per engine.

    Args:
        prompt:              Text prompt for image generation. An anatomy quality
                             suffix is appended automatically.
        reference_image:     Optional (bytes, mime_type) tuple — kept for backward
                             compatibility with the FLUX img2img fallback path,
                             also accepted as a single-image input on the Qwen
                             engine.
        reference_images:    Optional list of (bytes, mime_type) tuples. On the
                             Qwen engine the first 4 are wired into the
                             TextEncodeQwenImageEditPlus image1..image4 slots.
                             On the FLUX engine they drive the native
                             ReferenceLatent multi-character workflow.
        reference_subjects:  Optional list of subject labels, parallel to
                             reference_images. FLUX engine only.
        subject_appearances: Optional dict {subject_name -> appearance_text} used
                             by the FLUX AIO per-segment inpainting workflow to
                             provide per-character text prompts for each pass.
                             Ignored by the Qwen engine.
        on_progress:         Optional async callable(tag: str) for live progress.

    Returns:
        (image_bytes, "image/png") on success, or None on failure.
    """
    client_id = str(uuid.uuid4())
    base_url = os.environ.get("COMFYUI_URL", DEFAULT_URL).rstrip("/")

    gen_coro = asyncio.to_thread(
        _run_generate,
        prompt, reference_image, client_id,
        reference_images, width_override, height_override, steps_override,
        reference_subjects, subject_appearances,
    )

    if on_progress is None:
        return await gen_coro

    ws_base = base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    ws_url = f"{ws_base}/ws?clientId={client_id}"

    ws_task = asyncio.create_task(_ws_progress(ws_url, on_progress))
    try:
        result = await gen_coro
    finally:
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass

    return result
