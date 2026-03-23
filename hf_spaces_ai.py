"""
HuggingFace Spaces image generation backend using FLUX.2-klein-4B via Gradio client.
Activated by setting IMAGE_BACKEND=hf_spaces in the environment.

Required env vars:
    HF_TOKEN         HuggingFace API token (for priority access). Required.

Optional env vars:
    HF_SPACE_ID      Space to use (default: black-forest-labs/FLUX.2-klein-4B).
    HF_WIDTH         Output width in pixels (default: 680).
    HF_HEIGHT        Output height in pixels (default: 1024).
    HF_STEPS         Number of inference steps (default: 4).
    HF_GUIDANCE      Guidance scale (default: 1).
    HF_MODE          Mode choice string (default: Distilled (4 steps)).
"""

import asyncio
import os
import tempfile
from typing import Optional, Tuple


DEFAULT_SPACE_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_WIDTH = 680
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 4
DEFAULT_GUIDANCE = 1.0
DEFAULT_MODE = "Distilled (4 steps)"


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        print(f"[HFSpaces] Invalid {key} — using default {default}.")
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        print(f"[HFSpaces] Invalid {key} — using default {default}.")
        return default


def _run_generate(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]],
) -> Optional[Tuple[bytes, str]]:
    """Blocking: call the Gradio API and return (png_bytes, mime) or None."""
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("[HFSpaces] gradio_client is not installed. Run: pip install gradio_client")
        return None

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    space_id = os.environ.get("HF_SPACE_ID", DEFAULT_SPACE_ID).strip() or DEFAULT_SPACE_ID
    width = _get_int("HF_WIDTH", DEFAULT_WIDTH)
    height = _get_int("HF_HEIGHT", DEFAULT_HEIGHT)
    steps = _get_int("HF_STEPS", DEFAULT_STEPS)
    guidance = _get_float("HF_GUIDANCE", DEFAULT_GUIDANCE)
    mode = os.environ.get("HF_MODE", DEFAULT_MODE).strip() or DEFAULT_MODE

    if not hf_token:
        print("[HFSpaces] HF_TOKEN is not set — cannot authenticate.")
        return None

    ref_tmp_path: Optional[str] = None
    try:
        client = Client(space_id, hf_token=hf_token)

        input_images = []
        if reference_image is not None:
            try:
                img_bytes, _mime = reference_image
                suffix = ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(img_bytes)
                    ref_tmp_path = tmp.name
                input_images = [{"image": handle_file(ref_tmp_path), "caption": None}]
                print(f"[HFSpaces] Using reference image ({len(img_bytes)} bytes) — img2img mode")
            except Exception as exc:
                print(f"[HFSpaces] Could not encode reference image ({exc}); using txt2img.")
                input_images = []

        mode_str = mode
        print(f"[HFSpaces] Generating — space={space_id}, size={width}x{height}, steps={steps}, mode={mode_str}")
        print(f"[HFSpaces] Prompt: {prompt[:120]!r}")

        result = client.predict(
            prompt=prompt,
            input_images=input_images,
            mode_choice=mode_str,
            seed=0,
            randomize_seed=True,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            prompt_upsampling=False,
            api_name="/infer",
        )

        return _parse_result(result)

    except Exception as exc:
        print(f"[HFSpaces] Generation error: {type(exc).__name__}: {exc}")
        return None
    finally:
        if ref_tmp_path:
            try:
                os.unlink(ref_tmp_path)
            except OSError:
                pass


def _parse_result(result) -> Optional[Tuple[bytes, str]]:
    """Extract image bytes from whatever the Gradio client returns."""
    if result is None:
        print("[HFSpaces] Gradio returned None.")
        return None

    if isinstance(result, (list, tuple)):
        result = result[0]

    file_path: Optional[str] = None

    if isinstance(result, dict):
        file_path = result.get("path") or result.get("name")
    elif isinstance(result, str):
        file_path = result

    if not file_path:
        print(f"[HFSpaces] Could not extract file path from result: {type(result)} — {str(result)[:200]}")
        return None

    if not os.path.exists(file_path):
        print(f"[HFSpaces] Result file not found: {file_path}")
        return None

    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()
        print(f"[HFSpaces] Done — {len(image_bytes)} bytes from {file_path}")
        return image_bytes, "image/png"
    except Exception as exc:
        print(f"[HFSpaces] Could not read result file: {exc}")
        return None


async def generate_image(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]] = None,
) -> Optional[Tuple[bytes, str]]:
    """Generate an image via HuggingFace Spaces FLUX.2-klein-4B.

    Args:
        prompt:          Text prompt for image generation.
        reference_image: Optional (bytes, mime_type) tuple used as the img2img
                         reference image. Saved to a temp file and passed via
                         handle_file; temp file is cleaned up after the call.

    Returns:
        (image_bytes, "image/png") on success, or None on failure.
    """
    return await asyncio.to_thread(_run_generate, prompt, reference_image)
