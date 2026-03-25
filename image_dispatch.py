"""
Shared image generation dispatcher.
Routes generate_image() and image_ready() to the active backend based on
IMAGE_BACKEND env var. Imported by both bot.py and views.py so that the
routing logic lives in one place and views.py avoids a circular import with bot.

Supported backends:
    cloudflare      — cloudflare_ai.generate_image()
    hf_spaces       — hf_spaces_ai.generate_image()
    local_diffusers — diffusers_ai.generate_image()
    comfyui         — comfyui_ai.generate_image()
"""

import os
from typing import Callable, Coroutine, Optional, Tuple

_IMAGE_BACKEND = os.environ.get("IMAGE_BACKEND", "cloudflare").lower()


def image_ready() -> bool:
    """Return True if the active image backend is configured and ready."""
    if _IMAGE_BACKEND == "local_diffusers":
        return bool(os.environ.get("LOCAL_DIFFUSER_MODEL", "").strip())
    if _IMAGE_BACKEND == "hf_spaces":
        return bool(os.environ.get("HF_TOKEN", "").strip())
    if _IMAGE_BACKEND == "comfyui":
        import comfyui_ai as _comfyui_ai
        return _comfyui_ai.image_ready()
    return bool(os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID"))


def img2img_capable() -> bool:
    """Return True if the active backend supports reference image (img2img)."""
    return _IMAGE_BACKEND in ("local_diffusers", "hf_spaces", "comfyui")


async def generate_image(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]] = None,
    reference_images: Optional[list] = None,
    on_progress: Optional[Callable[[str], Coroutine]] = None,
) -> Optional[tuple]:
    """Dispatch image generation to the configured backend.

    Args:
        prompt:           Text prompt for image generation.
        reference_image:  Optional (bytes, mime_type) tuple used as the img2img
                          seed image. Passed to backends that support it
                          (local_diffusers, hf_spaces, comfyui). Ignored by Cloudflare.
        reference_images: Optional list of (bytes, mime_type) tuples. Forwarded to
                          the comfyui backend for IP-Adapter multi-character conditioning
                          when COMFYUI_IPADAPTER and COMFYUI_CLIP_VISION are set.
                          Ignored by all other backends.
        on_progress:      Optional async callable(tag: str) forwarded to
                          local_diffusers and comfyui backends for live progress
                          reporting. Silently ignored by cloudflare and hf_spaces.

    Returns:
        (image_bytes, mime_type) on success, or None on failure.
    """
    if _IMAGE_BACKEND == "local_diffusers":
        import diffusers_ai as _diffusers_ai
        return await _diffusers_ai.generate_image(
            prompt, reference_image=reference_image, on_progress=on_progress
        )
    if _IMAGE_BACKEND == "hf_spaces":
        import hf_spaces_ai as _hf_spaces_ai
        return await _hf_spaces_ai.generate_image(prompt, reference_image=reference_image)
    if _IMAGE_BACKEND == "comfyui":
        import comfyui_ai as _comfyui_ai
        return await _comfyui_ai.generate_image(
            prompt,
            reference_image=reference_image,
            reference_images=reference_images,
            on_progress=on_progress,
        )
    import cloudflare_ai as _cloudflare_ai
    return await _cloudflare_ai.generate_image(prompt)
