"""
ComfyUI image generation backend using FLUX.2-klein-4B GGUF via city96's ComfyUI-GGUF extension.
Activated by setting IMAGE_BACKEND=comfyui in the environment.

Required env vars:
    COMFYUI_GGUF     GGUF filename as known to ComfyUI (e.g. flux-2-klein-4b-Q5_K_M.gguf).
    COMFYUI_VAE      VAE filename as known to ComfyUI (e.g. flux2-vae.safetensors).
    COMFYUI_CLIP     Text-encoder filename as known to ComfyUI (e.g. qwen_3_4b.safetensors).

Optional env vars:
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

    if not gguf_path or not vae_name or not clip_name:
        print("[ComfyUI] COMFYUI_GGUF, COMFYUI_VAE, and COMFYUI_CLIP must all be set.")
        return None

    # reference_images (list) → native ReferenceLatent multi-ref path
    # reference_image  (single tuple) → legacy img2img path
    refs = reference_images or []

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
                    scene_seed = int(uuid.uuid4().int % (2**31))
                    print("[ComfyUI] Ultimate: generating initial scene via txt2img pass…")
                    _scene_wf = _build_txt2img_workflow(
                        prompt, gguf_path, vae_name, clip_name, steps, width, height, scene_seed
                    )
                    _scene_payload = {"prompt": _scene_wf, "client_id": client_id + "_scene"}
                    _scene_resp = _requests.post(f"{base_url}/prompt", json=_scene_payload, timeout=15)
                    _scene_image_filename: Optional[str] = None
                    if _scene_resp.status_code == 200:
                        _scene_pid = _scene_resp.json().get("prompt_id")
                        _scene_deadline = time.time() + timeout
                        while time.time() < _scene_deadline:
                            time.sleep(2)
                            try:
                                _sh = _requests.get(f"{base_url}/history/{_scene_pid}", timeout=30)
                            except Exception:
                                continue  # ComfyUI busy generating; retry next tick
                            if _sh.status_code != 200 or _scene_pid not in _sh.json():
                                continue
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
                    _use_sam = os.environ.get("COMFYUI_USE_SAM", "true").strip().lower() not in ("0", "false", "no")
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
                            _use_sam = False
                    ultimate_wf = build_ultimate_workflow(
                        overall_prompt=prompt,
                        per_character_prompts=_per_char_prompts,
                        uploaded_filenames=dict(subject_uploaded),
                        scene_image_filename=_scene_image_filename,
                        unet_name=gguf_path,
                        vae_name=vae_name,
                        clip_name=clip_name,
                        seed=seed,
                        steps=steps,
                        width=width,
                        height=height,
                        sam_query=_sam_query,
                        use_sam=_use_sam,
                    )
                    if ultimate_wf is None:
                        print("[ComfyUI] Ultimate: workflow build failed — falling back to flat-chain.")
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
                                steps=steps,
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

                # ── ReferenceChainConditioning path ──────────────────────────────
                # Primary multi-reference path: uses ReferenceChainConditioning
                # (ComfyUI-ReferenceChain, https://github.com/remingtonspaz/ComfyUI-ReferenceChain).
                # One node takes all uploaded reference images at once, handles
                # scaling + VAE-encoding internally, and chains them as reference
                # latents on both positive and negative conditioning streams.
                #
                # Used when: 2+ subjects each have at least one reference photo.
                # Falls back to flat-chain (ReferenceLatent) if the node is not
                # installed on the ComfyUI server.
                _used_refchain = False
                if n_subjects >= 2 and len(subject_uploaded) >= 2:
                    try:
                        from workflow_adapter import build_refchain_workflow
                        # Check that ReferenceChainConditioning is registered on the server
                        _rc_check = _requests.get(
                            f"{base_url}/object_info/ReferenceChainConditioning",
                            timeout=5,
                        )
                        if _rc_check.status_code == 200 and _rc_check.json():
                            refchain_wf = build_refchain_workflow(
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
                            workflow = refchain_wf
                            _used_refchain = True
                            # Pre-build flat-chain fallback so we can auto-retry if
                            # /prompt rejects the ReferenceChainConditioning node.
                            _refchain_fallback_wf = _build_reference_workflow(
                                prompt, gguf_path, vae_name, clip_name, steps, seed,
                                uploaded_names, width=width, height=height,
                            )
                            _per_char = {n: len(fs) for n, fs in subject_uploaded.items() if fs}
                            print(
                                f"[ComfyUI] ReferenceChain workflow — {subject_label}, "
                                f"per-character nodes: {_per_char}, output size={width}x{height}"
                            )
                        else:
                            print(
                                "[ComfyUI] ReferenceChainConditioning not found on server "
                                f"(HTTP {_rc_check.status_code}) — falling back to flat chain."
                            )
                    except Exception as _rc_exc:
                        print(f"[ComfyUI] ReferenceChain path failed ({_rc_exc}) — falling back to flat chain.")

                if not _used_refchain:
                    # Fallback: flat single-chain ReferenceLatent (all images in one chain).
                    # Character differentiation is handled by the text prompt (ID anchors).
                    print(
                        f"[ComfyUI] Flat-chain reference mode — {subject_label}, "
                        f"{len(uploaded_names)} image(s) total, output size={width}x{height}"
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
        else:
            mode = "txt2img"
        size_info = f"{width}x{height}" if mode != "reference" else "auto (from reference)"
        strength_info = strength if mode == "img2img" else "n/a"
        print(f"[ComfyUI] Queuing {mode} job — size={size_info}, steps={steps}, strength={strength_info}")
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
                # just retry on the next poll tick instead of aborting.
                continue
            if hist_resp.status_code != 200:
                continue
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

    Args:
        prompt:              Text prompt for image generation. An anatomy quality
                             suffix is appended automatically.
        reference_image:     Optional (bytes, mime_type) tuple — kept for backward
                             compatibility with the img2img fallback path.
        reference_images:    Optional list of (bytes, mime_type) tuples. When
                             provided, the native FLUX.2 Klein ReferenceLatent
                             workflow is used. Takes priority over reference_image.
        reference_subjects:  Optional list of subject labels, parallel to
                             reference_images.
        subject_appearances: Optional dict {subject_name -> appearance_text} used
                             by the AIO per-segment inpainting workflow to provide
                             per-character text prompts for each inpaint pass.
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
