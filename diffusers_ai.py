"""
Local image generation backend using HuggingFace diffusers Flux2KleinPipeline.
Activated by setting IMAGE_BACKEND=local_diffusers in the environment.

Required env vars:
    LOCAL_DIFFUSER_MODEL     Path to the model directory (e.g. E:\Flux_Final). Required.
    LOCAL_DIFFUSER_STEPS     Number of inference steps. Default: 8.
    LOCAL_DIFFUSER_STRENGTH  img2img strength (0.0–1.0). Default: 0.75.

The pipeline runs in a separate subprocess (diffusers_worker.py) so any GPU
crash (OOM, segfault, CUDA illegal access) cannot take down the bot process.

Progress reporting: the worker writes "PROGRESS:<tag>" lines to stderr which
are parsed here in real time and forwarded to an optional async on_progress
callback, enabling live Discord progress-bar updates.
"""

import asyncio
import base64
import json
import os
import sys
from typing import Callable, Coroutine, Optional, Tuple

_WORKER_TIMEOUT = 300  # seconds


async def generate_image(
    prompt: str,
    reference_image: Optional[Tuple[bytes, str]] = None,
    on_progress: Optional[Callable[[str], Coroutine]] = None,
) -> Optional[Tuple[bytes, str]]:
    """Generate an image by spawning diffusers_worker.py in a subprocess.

    Args:
        prompt:          Text prompt for image generation.
        reference_image: Optional (bytes, mime_type) tuple used as the img2img
                         reference.  The worker auto-detects whether the pipeline
                         accepts image/strength kwargs and falls back to txt2img
                         if it does not.
        on_progress:     Optional async callable(tag: str) called whenever the
                         worker emits a PROGRESS: line.  Tags:
                             "STAGE:loading"   — model is being loaded
                             "STAGE:ready"     — model loaded, inference starting
                             "STEP:<n>/<t>"    — inference step n of t completed
                             "STAGE:encoding"  — inference done, encoding PNG

    Returns:
        (png_bytes, "image/png") on success, or None on failure.
    """
    model_path = os.environ.get("LOCAL_DIFFUSER_MODEL", "").strip()
    if not model_path:
        print("[LocalDiffusers] LOCAL_DIFFUSER_MODEL is not set — cannot generate.")
        return None

    try:
        steps = int(os.environ.get("LOCAL_DIFFUSER_STEPS", "4"))
    except ValueError:
        print("[LocalDiffusers] Invalid LOCAL_DIFFUSER_STEPS — using default 4.")
        steps = 4

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
        height = int(os.environ.get("LOCAL_DIFFUSER_HEIGHT", "512"))
    except ValueError:
        print("[LocalDiffusers] Invalid LOCAL_DIFFUSER_HEIGHT — using default 512.")
        height = 512

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

    try:
        result = await asyncio.wait_for(
            _run_worker_async(worker_path, stdin_data, on_progress),
            timeout=_WORKER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        print(f"[LocalDiffusers] Worker timed out after {_WORKER_TIMEOUT}s.")
        return None
    except Exception as exc:
        print(f"[LocalDiffusers] Unexpected error: {type(exc).__name__}: {exc}")
        return None

    return result


async def _run_worker_async(
    worker_path: str,
    stdin_data: bytes,
    on_progress: Optional[Callable[[str], Coroutine]],
) -> Optional[Tuple[bytes, str]]:
    """Spawn the worker subprocess, stream stderr for progress, collect stdout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            worker_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        print(f"[LocalDiffusers] Worker launch error: {type(exc).__name__}: {exc}")
        return None

    proc.stdin.write(stdin_data)
    await proc.stdin.drain()
    proc.stdin.close()

    stdout_chunks: list[bytes] = []

    async def _read_stderr() -> None:
        assert proc.stderr is not None
        while True:
            line_bytes = await proc.stderr.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if line.startswith("PROGRESS:") and on_progress is not None:
                tag = line[len("PROGRESS:"):]
                try:
                    await on_progress(tag)
                except Exception as cb_exc:
                    print(f"[LocalDiffusers] on_progress callback error: {cb_exc}")
            else:
                print(line)

    async def _read_stdout() -> None:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            stdout_chunks.append(chunk)

    await asyncio.gather(_read_stderr(), _read_stdout())
    await proc.wait()

    if proc.returncode != 0:
        print(f"[LocalDiffusers] Worker exited with code {proc.returncode}.")
        raw_stdout = b"".join(stdout_chunks)
        try:
            err = json.loads(raw_stdout)
            print(f"[LocalDiffusers] Worker error: {err.get('error', 'unknown')}")
        except Exception:
            snippet = raw_stdout.decode("utf-8", errors="replace").strip()
            if snippet:
                print(f"[LocalDiffusers] Worker stdout: {snippet}")
        return None

    raw_stdout = b"".join(stdout_chunks)
    try:
        response = json.loads(raw_stdout)
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
