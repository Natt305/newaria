"""
ComfyUI image generation backend using FLUX.2-klein-4B GGUF via city96's ComfyUI-GGUF extension.
Activated by setting IMAGE_BACKEND=comfyui in the environment.

Required env vars:
    COMFYUI_GGUF     Full path to the GGUF model file (e.g. flux-2-klein-4b-Q5_K_M.gguf).
    COMFYUI_VAE      VAE model filename as known to ComfyUI (e.g. ae.safetensors).
    COMFYUI_CLIP     CLIP model filename as known to ComfyUI (e.g. t5xxl_fp8_e4m3fn.safetensors).

Optional env vars:
    COMFYUI_URL      ComfyUI server address (default: http://127.0.0.1:8188).
    COMFYUI_STEPS    Number of inference steps (default: 4).
    COMFYUI_WIDTH    Output width in pixels (default: 512).
    COMFYUI_HEIGHT   Output height in pixels (default: 512).
    COMFYUI_STRENGTH img2img denoise strength, 0.0–1.0 (default: 0.75).
    COMFYUI_WORKFLOW Path to a custom workflow JSON file (overrides built-in template).
    COMFYUI_TIMEOUT  Max seconds to wait for job completion (default: 300).
"""

import asyncio
import base64
import json
import os
import tempfile
import time
import uuid
from typing import Optional, Tuple

DEFAULT_URL = "http://127.0.0.1:8188"
DEFAULT_STEPS = 4
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STRENGTH = 0.75
DEFAULT_TIMEOUT = 300


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
    """Return a ComfyUI API-format workflow dict for text-to-image."""
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": gguf_path},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": prompt},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": ""},
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
) -> dict:
    """Return a ComfyUI API-format workflow dict for image-to-image."""
    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": gguf_path},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": prompt},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": ""},
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
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
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


def _run_generate(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]],
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
    steps = _get_int("COMFYUI_STEPS", DEFAULT_STEPS)
    width = _get_int("COMFYUI_WIDTH", DEFAULT_WIDTH)
    height = _get_int("COMFYUI_HEIGHT", DEFAULT_HEIGHT)
    strength = _get_float("COMFYUI_STRENGTH", DEFAULT_STRENGTH)
    timeout = _get_int("COMFYUI_TIMEOUT", DEFAULT_TIMEOUT)
    custom_workflow_path = os.environ.get("COMFYUI_WORKFLOW", "").strip()
    seed = int(uuid.uuid4().int % (2**31))

    if not gguf_path or not vae_name or not clip_name:
        print("[ComfyUI] COMFYUI_GGUF, COMFYUI_VAE, and COMFYUI_CLIP must all be set.")
        return None

    try:
        if custom_workflow_path and os.path.exists(custom_workflow_path):
            with open(custom_workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
            print(f"[ComfyUI] Using custom workflow: {custom_workflow_path}")
        elif reference_image is not None:
            img_bytes, _mime = reference_image
            uploaded_name = _upload_image(base_url, img_bytes, _requests)
            if uploaded_name:
                print(f"[ComfyUI] img2img mode — uploaded reference as '{uploaded_name}', strength={strength}")
                workflow = _build_img2img_workflow(
                    prompt, gguf_path, vae_name, clip_name,
                    steps, width, height, seed, strength, uploaded_name,
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

        print(f"[ComfyUI] Queuing job — size={width}x{height}, steps={steps}")
        print(f"[ComfyUI] Prompt: {prompt[:120]!r}")

        payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
        resp = _requests.post(f"{base_url}/prompt", json=payload, timeout=15)

        if resp.status_code != 200:
            print(f"[ComfyUI] /prompt returned HTTP {resp.status_code}.")
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
                    errs = node_err.get("errors", [])
                    for e in errs:
                        print(f"[ComfyUI]   Node {node_id} ({class_type}): "
                              f"[{e.get('type', '?')}] {e.get('message', '')} — {e.get('details', '')}")
                if not top_error and not node_errors:
                    print(f"[ComfyUI] Raw response: {resp.text[:500]}")
            except Exception:
                print(f"[ComfyUI] Raw response: {resp.text[:500]}")
            return None

        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            print(f"[ComfyUI] Server did not return a prompt_id: {resp.text[:200]}")
            return None
        print(f"[ComfyUI] Job queued — prompt_id={prompt_id}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(1)
            hist_resp = _requests.get(f"{base_url}/history/{prompt_id}", timeout=10)
            if hist_resp.status_code == 200:
                history = hist_resp.json()
                if prompt_id in history:
                    job = history[prompt_id]
                    outputs = job.get("outputs", {})
                    for _node_id, node_out in outputs.items():
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
                            print(f"[ComfyUI] Done — {len(image_bytes)} bytes ('{filename}')")
                            return image_bytes, "image/png"
                    print(f"[ComfyUI] Job complete but no output images found.")
                    return None
        print(f"[ComfyUI] Timed out after {timeout}s waiting for prompt_id={prompt_id}")
        return None

    except Exception as exc:
        print(f"[ComfyUI] Generation error: {type(exc).__name__}: {exc}")
        return None


def _upload_image(base_url: str, img_bytes: bytes, requests_mod) -> Optional[str]:
    """Upload image bytes to ComfyUI /upload/image; return the server filename or None."""
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


async def generate_image(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]] = None,
) -> Optional[Tuple[bytes, str]]:
    """Generate an image via a locally-running ComfyUI instance.

    Args:
        prompt:          Text prompt for image generation.
        reference_image: Optional (bytes, mime_type) tuple used as the img2img
                         reference image. Uploaded to ComfyUI via /upload/image.

    Returns:
        (image_bytes, "image/png") on success, or None on failure.
    """
    return await asyncio.to_thread(_run_generate, prompt, reference_image)
