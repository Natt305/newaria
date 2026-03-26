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
    width_override: Optional[int] = None,
    height_override: Optional[int] = None,
    steps_override: Optional[int] = None,
    reference_subjects: Optional[list] = None,
) -> Optional[tuple]:
    """Dispatch image generation to the configured backend.

    Args:
        prompt:              Text prompt for image generation.
        reference_image:     Optional (bytes, mime_type) tuple used as img2img seed.
                             Passed to backends that support it. Ignored by Cloudflare.
        reference_images:    Optional list of (bytes, mime_type) tuples. Forwarded to
                             comfyui for ReferenceLatent conditioning.
        reference_subjects:  Optional list of subject labels, parallel to reference_images.
                             Forwarded to comfyui. When 2+ unique labels present, each
                             subject gets an isolated ReferenceLatent chain.
        on_progress:         Optional async callable(tag: str) for live progress.
        width_override / height_override: Override output dimensions. comfyui only.
        steps_override:      Override inference step count. comfyui only.

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
            width_override=width_override,
            height_override=height_override,
            steps_override=steps_override,
            reference_subjects=reference_subjects,
        )
    import cloudflare_ai as _cloudflare_ai
    return await _cloudflare_ai.generate_image(prompt)
