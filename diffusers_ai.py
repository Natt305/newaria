"""
Local image generation backend using HuggingFace diffusers Flux2KleinPipeline.
Activated by setting IMAGE_BACKEND=local_diffusers in the environment.

Required env vars:
    LOCAL_DIFFUSER_MODEL     Path to the model directory (e.g. E:\Flux_Final). Required.
    LOCAL_DIFFUSER_STEPS     Number of inference steps. Default: 8.
    LOCAL_DIFFUSER_STRENGTH  img2img strength (0.0–1.0). Default: 0.75.

The pipeline runs in a separate subprocess (diffusers_worker.py) so any GPU
crash (OOM, segfault, CUDA illegal access) cannot take down the bot process.
"""

import asyncio
import base64
import json
import os
import sys
from typing import Optional, Tuple


async def generate_image(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]] = None,
) -> Optional[Tuple[bytes, str]]:
    """Generate an image by spawning diffusers_worker.py in a subprocess.

    Args:
        prompt:          Text prompt for image generation.
        reference_image: Optional (bytes, mime_type) tuple used as the img2img
                         reference.  The worker auto-detects whether the pipeline
                         accepts image/strength kwargs and falls back to txt2img
                         if it does not.

    Returns:
        (png_bytes, "image/png") on success, or None on failure.
    """
    model_path = os.environ.get("LOCAL_DIFFUSER_MODEL", "").strip()
    if not model_path:
        print("[LocalDiffusers] LOCAL_DIFFUSER_MODEL is not set — cannot generate.")
        return None

    try:
        steps = int(os.environ.get("LOCAL_DIFFUSER_STEPS", "8"))
    except ValueError:
        print("[LocalDiffusers] Invalid LOCAL_DIFFUSER_STEPS — using default 8.")
        steps = 8

    try:
        strength = float(os.environ.get("LOCAL_DIFFUSER_STRENGTH", "0.75"))
    except ValueError:
        print("[LocalDiffusers] Invalid LOCAL_DIFFUSER_STRENGTH — using default 0.75.")
        strength = 0.75

    try:
        width = int(os.environ.get("LOCAL_DIFFUSER_WIDTH", "512"))
    except ValueError:
        print("[LocalDiffusers] Invalid LOCAL_DIFFUSER_WIDTH — using default 512.")
        width = 512

    try:
        height = int(os.environ.get("LOCAL_DIFFUSER_HEIGHT", "768"))
    except ValueError:
        print("[LocalDiffusers] Invalid LOCAL_DIFFUSER_HEIGHT — using default 768.")
        height = 768

    payload: dict = {
        "prompt": prompt,
        "steps": steps,
        "strength": strength,
        "width": width,
        "height": height,
    }

    if reference_image is not None:
        try:
            img_bytes, _mime = reference_image
            payload["image_b64"] = base64.b64encode(img_bytes).decode("ascii")
        except Exception as exc:
            print(f"[LocalDiffusers] Could not encode reference image ({exc}); using txt2img.")

    mode = "img2img" if "image_b64" in payload else "txt2img"
    print(f"[LocalDiffusers] Spawning worker — mode={mode} — prompt: {prompt[:100]!r}")

    worker_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "diffusers_worker.py"
    )
    stdin_data = json.dumps(payload).encode("utf-8")

    result = await asyncio.to_thread(_run_worker, worker_path, stdin_data)
    return result


def _run_worker(
    worker_path: str, stdin_data: bytes
) -> Optional[Tuple[bytes, str]]:
    """Blocking: spawn diffusers_worker.py, return (png_bytes, mime) or None."""
    import subprocess

    try:
        proc = subprocess.run(
            [sys.executable, worker_path],
            input=stdin_data,
            capture_output=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("[LocalDiffusers] Worker timed out after 300 s.")
        return None
    except Exception as exc:
        print(f"[LocalDiffusers] Worker launch error: {type(exc).__name__}: {exc}")
        return None

    if proc.stderr:
        for line in proc.stderr.decode("utf-8", errors="replace").splitlines():
            print(line)

    if proc.returncode != 0:
        print(f"[LocalDiffusers] Worker exited with code {proc.returncode}.")
        try:
            err = json.loads(proc.stdout)
            print(f"[LocalDiffusers] Worker error: {err.get('error', 'unknown')}")
        except Exception:
            raw = proc.stdout.decode("utf-8", errors="replace").strip()
            if raw:
                print(f"[LocalDiffusers] Worker stdout: {raw}")
        return None

    try:
        response = json.loads(proc.stdout)
    except Exception as exc:
        print(f"[LocalDiffusers] Could not parse worker output: {exc}")
        return None

    if "error" in response:
        print(f"[LocalDiffusers] Worker reported error: {response['error']}")
        return None

    if "png_b64" not in response:
        print("[LocalDiffusers] Worker returned no image data.")
        return None

    try:
        png_bytes = base64.b64decode(response["png_b64"])
    except Exception as exc:
        print(f"[LocalDiffusers] Could not decode PNG from worker: {exc}")
        return None

    print(f"[LocalDiffusers] Done — {len(png_bytes)} bytes")
    return png_bytes, "image/png"
