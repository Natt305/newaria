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
    LOAD:<frac>            fractional loading progress 0.0–1.0

Input JSON (stdin):
    {
        "prompt":    "...",
        "image_b64": "...",   # optional — base64-encoded reference image bytes
        "steps":     8,
        "strength":  0.75,
        "gguf_path": "..."    # optional — full path to .gguf transformer file
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


def _gguf_progress_thread(stop_event: threading.Event) -> None:
    """Emit indeterminate LOAD progress markers during GGUF model loading.

    GGUF loading cannot be patched like safetensors, so we emit timed
    markers so the Discord bar visibly advances while the user waits.
    Schedule: 0→0.05 immediately, then 0.20 @ 10s, 0.45 @ 25s, 0.70 @ 45s,
    0.88 @ 70s — all capped at 0.95 so the bar never claims "done" early.
    """
    schedule = [
        (0,   0.05),
        (10,  0.20),
        (25,  0.45),
        (45,  0.70),
        (70,  0.88),
        (100, 0.95),
    ]
    start = _time.monotonic()
    idx = 0
    while not stop_event.is_set() and idx < len(schedule):
        delay, frac = schedule[idx]
        elapsed = _time.monotonic() - start
        wait = delay - elapsed
        if wait > 0:
            stop_event.wait(timeout=wait)
        if stop_event.is_set():
            break
        _progress(f"LOAD:{frac:.2f}")
        idx += 1


def _load_pipeline_gguf(model_path: str, gguf_path: str, torch):
    """Load Flux2KleinPipeline with a GGUF-quantised transformer.

    The transformer is loaded from `gguf_path` using GGUFQuantizationConfig.
    Text encoders, VAE and scheduler come from `model_path` (the base
    model directory — same as the normal safetensors path).
    """
    try:
        from diffusers import Flux2KleinPipeline
        from diffusers.quantizers import GGUFQuantizationConfig
    except ImportError as exc:
        _fail(f"Missing dependency for GGUF loading: {exc}. "
              f"Ensure diffusers>=0.32 and gguf>=0.10 are installed.")

    _log(f"GGUF mode — transformer from: {gguf_path!r}")
    _log(f"GGUF mode — base model (text encoders, VAE): {model_path!r}")

    transformer = None
    used_class = None

    for cls_name in ("Flux2KleinTransformer2DModel", "FluxTransformer2DModel"):
        try:
            import diffusers as _df
            cls = getattr(_df, cls_name, None)
            if cls is None:
                continue
            _log(f"Loading transformer from GGUF using {cls_name} ...")
            transformer = cls.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
            used_class = cls_name
            _log(f"Transformer loaded via {cls_name}.")
            break
        except Exception as exc:
            _log(f"{cls_name}.from_single_file failed: {type(exc).__name__}: {exc}")

    if transformer is None:
        _fail("Could not load GGUF transformer — tried Flux2KleinTransformer2DModel "
              "and FluxTransformer2DModel. Check diffusers version and gguf file path.")

    _log(f"Loading pipeline from base model directory {model_path!r} (transformer={used_class}) ...")
    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    except Exception as exc:
        _fail(f"Pipeline.from_pretrained (GGUF mode) failed: {type(exc).__name__}: {exc}")

    return pipe


def _load_pipeline_safetensors(model_path: str, torch):
    """Load Flux2KleinPipeline from a full safetensors model directory.

    Patches safetensors.torch.load_file and torch.load to emit real LOAD:frac
    progress markers as each weight file finishes loading.
    """
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        _fail(f"Missing dependency: {exc}")

    _log(f"Safetensors mode — loading pipeline from {model_path!r} ...")

    _weight_exts = {".safetensors", ".bin"}
    _weight_files: dict[str, int] = {
        str(p.resolve()): p.stat().st_size
        for p in pathlib.Path(model_path).rglob("*")
        if p.suffix.lower() in _weight_exts
    }
    _total_bytes = sum(_weight_files.values()) or 1
    _loaded = [0]
    _log(f"Weight files: {len(_weight_files)} files, {_total_bytes // 1024**2} MB total")

    def _emit_load(filepath: str) -> None:
        size = _weight_files.get(str(pathlib.Path(filepath).resolve()), 0)
        if size:
            _loaded[0] += size
            _progress(f"LOAD:{min(_loaded[0] / _total_bytes, 0.99):.3f}")

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
    except Exception as exc:
        _fail(f"Pipeline load failed: {type(exc).__name__}: {exc}")
    finally:
        if _orig_st_load is not None and _st_module is not None:
            _st_module.load_file = _orig_st_load
        torch.load = _orig_torch_load

    return pipe


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
    gguf_path: str = payload.get("gguf_path", "").strip()

    try:
        import torch
        from PIL import Image, ImageOps
    except ImportError as exc:
        _fail(f"Missing dependency: {exc}")

    model_path = os.environ.get("LOCAL_DIFFUSER_MODEL", "").strip()
    if not model_path:
        _fail("LOCAL_DIFFUSER_MODEL is not set")

    _progress("STAGE:loading")

    if gguf_path:
        _log(f"Loading transformer from GGUF: {gguf_path!r}")
        _gguf_stop = threading.Event()
        _gguf_progress_t = threading.Thread(
            target=_gguf_progress_thread, args=(_gguf_stop,), daemon=True
        )
        _gguf_progress_t.start()
        try:
            pipe = _load_pipeline_gguf(model_path, gguf_path, torch)
        finally:
            _gguf_stop.set()
            _gguf_progress_t.join(timeout=2.0)
        _log("GGUF pipeline loaded.")
    else:
        pipe = _load_pipeline_safetensors(model_path, torch)

    # ── Post-load optimisations (same for both loading modes) ─────────────────
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(1)
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    _log("Pipeline ready (attention slicing → sequential CPU offload → VAE tiling+slicing).")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_mb = torch.cuda.mem_get_info()[0] // 1024 ** 2
            _log(f"CUDA cache cleared — free VRAM: {free_mb} MB")
    except Exception:
        pass

    call_params = set(inspect.signature(pipe.__call__).parameters.keys())
    supports_image       = "image"               in call_params
    supports_strength    = "strength"             in call_params
    supports_neg         = "negative_prompt"      in call_params
    supports_step_end_cb = "callback_on_step_end" in call_params
    supports_legacy_cb   = "callback"             in call_params
    _log(f"Accepted params — image={supports_image}, strength={supports_strength}, "
         f"negative_prompt={supports_neg}, "
         f"callback_on_step_end={supports_step_end_cb}, callback={supports_legacy_cb}")

    # ── Anatomy quality boosters ──────────────────────────────────────────────
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

    _progress("STAGE:ready")

    init_image = None
    if image_b64 and supports_image:
        try:
            img_bytes = base64.b64decode(image_b64)
            raw = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            centering = (0.5, 0.35) if raw.height > raw.width else (0.5, 0.5)
            init_image = ImageOps.fit(
                raw, (width, height), method=Image.LANCZOS, centering=centering
            )
            _log(f"Reference image decoded — smart-cropped to {width}×{height} "
                 f"(original {raw.width}×{raw.height}, centering={centering}).")
        except Exception as exc:
            _log(f"Could not decode reference image ({exc}) — will use txt2img.")

    # ── Inter-step interpolation timer ───────────────────────────────────────
    _step_state = {
        "completed": 0.0,
        "step_start": 0.0,
        "durations": [],
    }
    _infer_done = threading.Event()

    def _interpolation_thread() -> None:
        while not _infer_done.wait(timeout=0.25):
            durations = _step_state["durations"]
            completed = _step_state["completed"]
            if not durations or completed >= steps:
                continue
            avg_dur = sum(durations) / len(durations)
            if avg_dur <= 0:
                continue
            elapsed = _time.monotonic() - _step_state["step_start"]
            within = min(elapsed / avg_dur, 0.95)
            frac = completed + within
            _progress(f"STEP:{frac:.3f}/{steps}")

    def make_kwargs(use_image: bool) -> dict:
        kw = {
            "prompt": prompt + _ANATOMY_SUFFIX,
            "num_inference_steps": steps,
            "width": width,
            "height": height,
        }
        if supports_neg:
            kw["negative_prompt"] = _ANATOMY_NEGATIVE
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
