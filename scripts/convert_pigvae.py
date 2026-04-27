"""
Convert a GGUF-format VAE file to safetensors so ComfyUI's stock VAELoader
can load it.  VAELoaderGGUF does not exist in city96's ComfyUI-GGUF pack;
this one-shot conversion is the supported path.

Usage:
    # Auto-resolve path from COMFYUI_PATH env var (reads tokens.txt if set):
    python scripts/convert_pigvae.py

    # Explicit path to the .gguf file:
    python scripts/convert_pigvae.py "E:\\comfyui\\resources\\ComfyUI\\models\\vae\\pig_qwen_image_vae_fp32-f16.gguf"

Output: <same directory>/<same stem>.safetensors

After conversion, update tokens.txt:
    COMFYUI_QWEN_VAE=pig_qwen_image_vae_fp32-f16.safetensors
"""

import os
import sys
from pathlib import Path


def _resolve_gguf_path() -> Path:
    """Locate the .gguf VAE from CLI arg or COMFYUI_PATH env var."""
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            print(f"[convert_pigvae] ERROR: File not found: {p}")
            sys.exit(1)
        return p

    # Try to read COMFYUI_PATH and COMFYUI_QWEN_VAE from the environment,
    # falling back to a tokens.txt parse if the env vars are not set.
    comfyui_path = os.environ.get("COMFYUI_PATH", "").strip()
    qwen_vae = os.environ.get("COMFYUI_QWEN_VAE", "").strip()

    if not comfyui_path or not qwen_vae:
        # Try parsing tokens.txt in the current directory.
        tokens_file = Path("tokens.txt")
        if tokens_file.exists():
            for line in tokens_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line.startswith("COMFYUI_PATH="):
                    comfyui_path = comfyui_path or line.split("=", 1)[1].strip()
                elif line.startswith("COMFYUI_QWEN_VAE="):
                    qwen_vae = qwen_vae or line.split("=", 1)[1].strip()

    if not comfyui_path:
        print("[convert_pigvae] ERROR: Cannot determine COMFYUI_PATH.")
        print("  Pass the .gguf file path directly:")
        print('  python scripts/convert_pigvae.py "E:\\...\\models\\vae\\pig_qwen_image_vae_fp32-f16.gguf"')
        sys.exit(1)

    if not qwen_vae:
        qwen_vae = "pig_qwen_image_vae_fp32-f16.gguf"
        print(f"[convert_pigvae] COMFYUI_QWEN_VAE not set — defaulting to: {qwen_vae}")

    p = Path(comfyui_path) / "models" / "vae" / qwen_vae
    if not p.exists():
        print(f"[convert_pigvae] ERROR: File not found: {p}")
        print("  Check that COMFYUI_PATH and COMFYUI_QWEN_VAE are set correctly in tokens.txt.")
        sys.exit(1)
    return p


def _import_or_die(package: str, pip_name: str | None = None):
    import importlib
    try:
        return importlib.import_module(package)
    except ImportError:
        pip_name = pip_name or package
        print(f"[convert_pigvae] ERROR: '{package}' is not installed.")
        print(f"  Run: pip install {pip_name}")
        sys.exit(1)


def convert(gguf_path: Path, output_path: Path | None = None) -> Path:
    np = _import_or_die("numpy")
    torch = _import_or_die("torch")
    gguf_mod = _import_or_die("gguf")
    sf = _import_or_die("safetensors")
    from safetensors.torch import save_file

    if output_path is None:
        output_path = gguf_path.with_suffix(".safetensors")

    in_mb = gguf_path.stat().st_size / 1024 ** 2
    print(f"[convert_pigvae] Input : {gguf_path}  ({in_mb:.1f} MB)")
    print(f"[convert_pigvae] Output: {output_path}")
    print("[convert_pigvae] Reading GGUF tensors...")

    reader = gguf_mod.GGUFReader(str(gguf_path))

    tensors: dict = {}
    for tensor in reader.tensors:
        raw = tensor.data

        # GGUF stores tensor dimensions in reversed order relative to PyTorch.
        # Reverse to get the standard [out, in, h, w, ...] shape.
        shape = list(reversed(tensor.shape.tolist()))

        expected_elements = 1
        for d in shape:
            expected_elements *= d

        if raw.size != expected_elements:
            # Fallback: try GGUF-native order (rare, but handles edge cases).
            shape = tensor.shape.tolist()

        arr = raw.reshape(shape)

        # Normalise to float16 regardless of stored dtype.
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)

        tensors[tensor.name] = torch.from_numpy(arr.copy())

    n = len(tensors)
    print(f"[convert_pigvae] Converted {n} tensor(s) — saving...")
    save_file(tensors, str(output_path))

    out_mb = output_path.stat().st_size / 1024 ** 2
    print(f"[convert_pigvae] Done!  {out_mb:.1f} MB written.")
    print()
    print("Next step — update tokens.txt:")
    print(f"  COMFYUI_QWEN_VAE={output_path.name}")
    print()
    print("Then restart start.bat.  No ComfyUI node changes are needed.")
    return output_path


if __name__ == "__main__":
    gguf_path = _resolve_gguf_path()
    convert(gguf_path)
