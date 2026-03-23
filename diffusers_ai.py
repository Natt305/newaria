"""
Local image generation backend using HuggingFace diffusers Flux2KleinPipeline.
Activated by setting IMAGE_BACKEND=local_diffusers in the environment.

Required env vars:
    LOCAL_DIFFUSER_MODEL     Path to the model directory (e.g. E:\Flux_Final). Required.
    LOCAL_DIFFUSER_STEPS     Number of inference steps. Default: 8.
    LOCAL_DIFFUSER_STRENGTH  img2img strength (0.0-1.0). Default: 0.75.

Device placement is managed by pipe.enable_model_cpu_offload() and does not need
to be configured manually.
"""
import asyncio
import io
import os
from typing import Optional, Tuple

try:
    import torch
    from diffusers import Flux2KleinPipeline
    from PIL import Image
    _DEPS_AVAILABLE = True
except ImportError as _import_err:
    print(
        f"[LocalDiffusers] Dependencies not available ({_import_err}). "
        "Install torch + diffusers + Pillow on your local machine to enable this backend."
    )
    _DEPS_AVAILABLE = False

_pipeline = None


def _load_pipeline():
    """Lazily load and cache the Flux2KleinPipeline. Returns None on failure."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    model_path = os.environ.get("LOCAL_DIFFUSER_MODEL", "").strip()
    if not model_path:
        print("[LocalDiffusers] LOCAL_DIFFUSER_MODEL is not set — cannot load pipeline.")
        return None

    print(f"[LocalDiffusers] Loading pipeline from {model_path!r} ...")
    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        _pipeline = pipe
        print("[LocalDiffusers] Pipeline loaded and CPU offload enabled.")
        return _pipeline
    except Exception as exc:
        print(f"[LocalDiffusers] Failed to load pipeline: {type(exc).__name__}: {exc}")
        return None


def _run_pipeline(prompt: str, init_image=None) -> Optional[bytes]:
    """Blocking inference call. Must be run via asyncio.to_thread."""
    pipe = _load_pipeline()
    if pipe is None:
        return None

    try:
        steps = int(os.environ.get("LOCAL_DIFFUSER_STEPS", "8"))
    except ValueError:
        print("[LocalDiffusers] Invalid LOCAL_DIFFUSER_STEPS — using default 8.")
        steps = 8

    try:
        if init_image is not None:
            result = pipe(
                prompt=prompt,
                image=init_image,
                num_inference_steps=steps,
                width=512,
                height=512,
            )
        else:
            result = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                width=512,
                height=512,
            )

        img = result.images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as exc:
        print(f"[LocalDiffusers] Inference error: {type(exc).__name__}: {exc}")
        return None


async def generate_image(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]] = None,
) -> Optional[Tuple[bytes, str]]:
    """Generate an image using the local Flux2KleinPipeline.

    Args:
        prompt: Text prompt for image generation.
        reference_image: Optional (bytes, mime_type) tuple used as the img2img
                         init image. When provided the pipeline runs in
                         image-to-image mode conditioned on that reference.

    Returns:
        (png_bytes, "image/png") on success, or None on failure.
    """
    if not _DEPS_AVAILABLE:
        print("[LocalDiffusers] Cannot generate — torch/diffusers/Pillow not installed.")
        return None

    init_image = None
    if reference_image is not None:
        try:
            img_bytes, _mime = reference_image
            init_image = (
                Image.open(io.BytesIO(img_bytes))
                .convert("RGB")
                .resize((512, 512))
            )
        except Exception as exc:
            print(f"[LocalDiffusers] Could not decode reference image ({exc}); falling back to text-to-image.")

    print(
        f"[LocalDiffusers] Generating — mode={'img2img' if init_image is not None else 'txt2img'} "
        f"— prompt: {prompt[:100]!r}"
    )

    png_bytes = await asyncio.to_thread(_run_pipeline, prompt, init_image)
    if png_bytes is None:
        return None

    print(f"[LocalDiffusers] Done — {len(png_bytes)} bytes")
    return png_bytes, "image/png"
