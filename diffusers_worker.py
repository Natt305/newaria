"""
diffusers_worker.py — standalone image generation worker for the local diffusers backend.

Called by diffusers_ai.py as a subprocess.  Reads a JSON payload from stdin,
runs Flux2KleinPipeline, and writes a JSON response to stdout.  All log/debug
lines go to stderr so they do not corrupt the JSON stdout channel.

Input JSON (stdin):
    {
        "prompt":    "...",
        "image_b64": "...",   # optional — base64-encoded reference image bytes
        "steps":     8,
        "strength":  0.75
    }

Output JSON (stdout) on success:
    {"png_b64": "..."}

Output JSON (stdout) on failure:
    {"error": "..."}

Exit codes:
    0  success
    1  unrecoverable failure
"""

import base64
import inspect
import io
import json
import os
import sys
import traceback


def _log(msg: str) -> None:
    print(f"[Worker] {msg}", file=sys.stderr, flush=True)


def _fail(msg: str) -> None:
    print(json.dumps({"error": msg}), flush=True)
    sys.exit(1)


def main() -> None:
    try:
        raw = sys.stdin.buffer.read()
        payload = json.loads(raw)
    except Exception as exc:
        _fail(f"Failed to parse input JSON: {exc}")

    prompt: str = payload.get("prompt", "")
    image_b64: str = payload.get("image_b64", "")
    steps: int = int(payload.get("steps", 8))
    strength: float = float(payload.get("strength", 0.75))

    try:
        import torch
        from diffusers import Flux2KleinPipeline
        from PIL import Image
    except ImportError as exc:
        _fail(f"Missing dependency: {exc}")

    model_path = os.environ.get("LOCAL_DIFFUSER_MODEL", "").strip()
    if not model_path:
        _fail("LOCAL_DIFFUSER_MODEL is not set")

    _log(f"Loading pipeline from {model_path!r} ...")
    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.enable_attention_slicing(1)
        _log("Pipeline loaded (CPU offload, VAE tiling/slicing, attention slicing enabled).")
    except Exception as exc:
        _fail(f"Pipeline load failed: {type(exc).__name__}: {exc}")

    call_params = set(inspect.signature(pipe.__call__).parameters.keys())
    supports_image = "image" in call_params
    supports_strength = "strength" in call_params
    _log(f"Accepted params — image={supports_image}, strength={supports_strength}")

    init_image = None
    if image_b64 and supports_image:
        try:
            img_bytes = base64.b64decode(image_b64)
            init_image = (
                Image.open(io.BytesIO(img_bytes))
                .convert("RGB")
                .resize((512, 512))
            )
            _log("Reference image decoded and resized to 512×512.")
        except Exception as exc:
            _log(f"Could not decode reference image ({exc}) — will use txt2img.")

    def make_kwargs(use_image: bool) -> dict:
        kw = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "width": 512,
            "height": 512,
        }
        if use_image and init_image is not None:
            kw["image"] = init_image
            if supports_strength:
                kw["strength"] = strength
        return kw

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log(f"CUDA cache cleared — free VRAM: {torch.cuda.mem_get_info()[0] // 1024**2} MB")

    modes = ["img2img", "txt2img"] if init_image is not None else ["txt2img"]
    result = None
    last_error: str = ""

    for mode in modes:
        use_img = mode == "img2img"
        _log(f"Attempting {mode} ...")
        try:
            result = pipe(**make_kwargs(use_img))
            _log(f"{mode} succeeded.")
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _log(f"{mode} failed: {last_error}")
            if mode == "img2img":
                _log("Retrying as txt2img ...")

    if result is None:
        _fail(last_error or "Unknown inference error")

    img = result.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    print(json.dumps({"png_b64": png_b64}), flush=True)


if __name__ == "__main__":
    main()
