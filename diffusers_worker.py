"""
diffusers_worker.py — standalone image generation worker for the local diffusers backend.

Called by diffusers_ai.py as a subprocess.  Reads a JSON payload from stdin,
runs Flux2KleinPipeline, and writes a JSON response to stdout.  All log/debug
lines go to stderr so they do not corrupt the JSON stdout channel.

Progress reporting: lines of the form "PROGRESS:<tag>" are written to stderr
and parsed by diffusers_ai.py to drive live Discord progress updates.
Tags:
    STAGE:loading          pipeline is being loaded from disk
    STAGE:ready            pipeline loaded, about to start inference
    STEP:<n>/<total>       inference step n of total completed
    STAGE:encoding         inference done, encoding PNG

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
import pathlib
import sys
import threading
import time as _time


def _log(msg: str) -> None:
    print(f"[Worker] {msg}", file=sys.stderr, flush=True)


def _progress(tag: str) -> None:
    """Emit a progress marker that diffusers_ai.py parses in real time."""
    print(f"PROGRESS:{tag}", file=sys.stderr, flush=True)


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
    width: int = int(payload.get("width", 512))
    height: int = int(payload.get("height", 512))

    try:
        import torch
        from diffusers import Flux2KleinPipeline
        from PIL import Image, ImageOps
    except ImportError as exc:
        _fail(f"Missing dependency: {exc}")

    model_path = os.environ.get("LOCAL_DIFFUSER_MODEL", "").strip()
    if not model_path:
        _fail("LOCAL_DIFFUSER_MODEL is not set")

    _progress("STAGE:loading")
    _log(f"Loading pipeline from {model_path!r} ...")

    # ── Real loading progress via weight-file patching ────────────────────────
    # Pre-scan the model dir to know total weight bytes, then patch
    # safetensors.torch.load_file and torch.load so each completed file load
    # advances the Discord bar by its real fraction of total model size.
    _weight_exts = {".safetensors", ".bin"}
    _weight_files: dict[str, int] = {
        str(p.resolve()): p.stat().st_size
        for p in pathlib.Path(model_path).rglob("*")
        if p.suffix.lower() in _weight_exts
    }
    _total_bytes = sum(_weight_files.values()) or 1
    _loaded = [0]  # mutable int shared across both patch closures
    _log(f"Weight files: {len(_weight_files)} files, {_total_bytes // 1024**2} MB total")

    def _emit_load(filepath: str) -> None:
        size = _weight_files.get(str(pathlib.Path(filepath).resolve()), 0)
        if size:
            _loaded[0] += size
            _progress(f"LOAD:{min(_loaded[0] / _total_bytes, 0.99):.3f}")

    # Patch safetensors (primary format for modern diffusers models)
    _orig_st_load = None
    _st_module = None
    try:
        import safetensors.torch as _st_module
        _orig_st_load = _st_module.load_file

        def _patched_st_load(filename, *args, **kwargs):
            result = _orig_st_load(filename, *args, **kwargs)
            _emit_load(str(filename))
            return result

        _st_module.load_file = _patched_st_load
        _log("safetensors.torch.load_file patched for progress tracking.")
    except ImportError:
        _log("safetensors not available — load progress via torch.load only.")

    # Patch torch.load as fallback for .bin files
    _orig_torch_load = torch.load

    def _patched_torch_load(f, *args, **kwargs):
        result = _orig_torch_load(f, *args, **kwargs)
        if isinstance(f, (str, pathlib.Path)):
            fname = str(f)
        elif hasattr(f, "__fspath__"):
            fname = os.fspath(f)
        else:
            fname = getattr(f, "name", "")
        if fname:
            _emit_load(fname)
        return result

    torch.load = _patched_torch_load

    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        # Attention slicing must be set BEFORE sequential CPU offload attaches hooks
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing(1)
        # Sequential offload moves each sub-module on/off GPU one at a time —
        # far lower peak VRAM than enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        # VAE tiling/slicing are safe to set after offload
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        _log("Pipeline loaded (attention slicing → sequential CPU offload → VAE tiling+slicing).")
    except Exception as exc:
        _fail(f"Pipeline load failed: {type(exc).__name__}: {exc}")
    finally:
        # Always restore originals — even if from_pretrained raises
        if _orig_st_load is not None and _st_module is not None:
            _st_module.load_file = _orig_st_load
        torch.load = _orig_torch_load

    # Flush CUDA cache and log available VRAM before inference
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mb = torch.cuda.mem_get_info()[0] // 1024 ** 2
            _log(f"CUDA cache cleared — free VRAM: {free_mb} MB")
    except Exception:
        pass

    call_params = set(inspect.signature(pipe.__call__).parameters.keys())
    supports_image = "image" in call_params
    supports_strength = "strength" in call_params
    supports_step_end_cb = "callback_on_step_end" in call_params
    supports_legacy_cb = "callback" in call_params
    _log(f"Accepted params — image={supports_image}, strength={supports_strength}, "
         f"callback_on_step_end={supports_step_end_cb}, callback={supports_legacy_cb}")

    _progress("STAGE:ready")

    init_image = None
    if image_b64 and supports_image:
        try:
            img_bytes = base64.b64decode(image_b64)
            raw = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            # Smart crop-to-fill: for portrait sources (character art) anchor
            # the crop slightly above centre so the face/head stays in frame.
            # For landscape/square sources use dead centre.
            if raw.height > raw.width:
                centering = (0.5, 0.35)   # bias upward — keeps head/face
            else:
                centering = (0.5, 0.5)    # centre for landscape/square
            init_image = ImageOps.fit(
                raw, (width, height), method=Image.LANCZOS, centering=centering
            )
            _log(f"Reference image decoded — smart-cropped to {width}×{height} "
                 f"(original {raw.width}×{raw.height}, centering={centering}).")
        except Exception as exc:
            _log(f"Could not decode reference image ({exc}) — will use txt2img.")

    # ── Inter-step interpolation timer ───────────────────────────────────────
    # Shared state between the main thread (step callbacks) and timer thread.
    _step_state = {
        "completed": 0.0,       # fractional steps completed (updated each callback)
        "step_start": 0.0,      # monotonic time when the current step began
        "durations": [],         # duration (s) of each completed step
    }
    _infer_done = threading.Event()

    def _interpolation_thread() -> None:
        """Emit sub-step progress every 250 ms by time-interpolating within a step."""
        while not _infer_done.wait(timeout=0.25):
            durations = _step_state["durations"]
            completed = _step_state["completed"]
            if not durations or completed >= steps:
                continue
            avg_dur = sum(durations) / len(durations)
            if avg_dur <= 0:
                continue
            elapsed = _time.monotonic() - _step_state["step_start"]
            # Cap at 0.95 of expected duration so we never overshoot the next step
            within = min(elapsed / avg_dur, 0.95)
            frac = completed + within
            _progress(f"STEP:{frac:.3f}/{steps}")

    def make_kwargs(use_image: bool) -> dict:
        kw = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "width": width,
            "height": height,
        }
        if use_image and init_image is not None:
            kw["image"] = init_image
            if supports_strength:
                kw["strength"] = strength

        if supports_step_end_cb:
            def _cb_step_end(pipe, step_index, timestep, callback_kwargs):
                now = _time.monotonic()
                if _step_state["step_start"] > 0:
                    _step_state["durations"].append(now - _step_state["step_start"])
                _step_state["completed"] = float(step_index + 1)
                _step_state["step_start"] = now
                _progress(f"STEP:{step_index + 1}/{steps}")
                return callback_kwargs
            kw["callback_on_step_end"] = _cb_step_end
        elif supports_legacy_cb:
            def _cb_legacy(step, timestep, latents):
                now = _time.monotonic()
                if _step_state["step_start"] > 0:
                    _step_state["durations"].append(now - _step_state["step_start"])
                _step_state["completed"] = float(step + 1)
                _step_state["step_start"] = now
                _progress(f"STEP:{step + 1}/{steps}")
            kw["callback"] = _cb_legacy
            kw["callback_steps"] = 1

        return kw

    modes = ["img2img", "txt2img"] if init_image is not None else ["txt2img"]
    result = None
    last_error: str = ""

    # Start the interpolation timer thread just before inference begins.
    # It will begin emitting sub-step PROGRESS lines as soon as the first
    # real step completes and we have a duration estimate.
    _timer_thread = threading.Thread(target=_interpolation_thread, daemon=True)
    _timer_thread.start()

    try:
        for mode in modes:
            use_img = mode == "img2img"
            _log(f"Attempting {mode} ...")
            _step_state["step_start"] = _time.monotonic()
            try:
                result = pipe(**make_kwargs(use_img))
                _log(f"{mode} succeeded.")
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                _log(f"{mode} failed: {last_error}")
                if mode == "img2img":
                    _log("Retrying as txt2img ...")
    finally:
        _infer_done.set()
        _timer_thread.join(timeout=1.0)

    if result is None:
        _fail(last_error or "Unknown inference error")

    _progress("STAGE:encoding")
    img = result.images[0]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    print(json.dumps({"png_b64": png_b64}), flush=True)


if __name__ == "__main__":
    main()
