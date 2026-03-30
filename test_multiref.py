"""
Self-test for build_multiref_workflow.

Tests:
  1. Structural validation — all node references point to existing node IDs.
  2. ComfyUI server validation — POST to /prompt and check for node errors.
  3. Full end-to-end — wait for the job to finish and save the output image.

Run directly: python test_multiref.py
"""

import io
import json
import os
import sys
import time
import uuid

TEST_COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
GGUF  = os.environ.get("COMFYUI_GGUF", "flux-2-klein-4b-Q5_K_M.gguf")
VAE   = os.environ.get("COMFYUI_VAE",  "flux2-vae.safetensors")
CLIP  = os.environ.get("COMFYUI_CLIP", "qwen_3_4b.safetensors")
W     = int(os.environ.get("COMFYUI_WIDTH",  "512"))
H     = int(os.environ.get("COMFYUI_HEIGHT", "512"))
STEPS = int(os.environ.get("COMFYUI_STEPS",  "4"))

CHAR_A_NAME = "Mortis"
CHAR_A_APP  = (
    "a young woman with long flowing dark purple hair, sharp crimson eyes, "
    "wearing a black gothic lolita dress with lace trim, pale skin"
)
CHAR_B_NAME = "Aria"
CHAR_B_APP  = (
    "a cheerful girl with short silver hair and bright blue eyes, "
    "wearing a white school uniform with blue ribbon, light skin"
)

SCENE_PROMPT = (
    f"{CHAR_A_NAME} and {CHAR_B_NAME} sitting together in a cozy cafe, "
    "warm lighting, anime style, high quality illustration"
)

REF_IMAGE_A = os.path.join(
    os.path.dirname(__file__), "attached_assets",
    "299px-Mortis_Live2D_Model_1774165684397.png"
)
REF_IMAGE_B = os.path.join(
    os.path.dirname(__file__), "attached_assets",
    "Mortis_Anime_unmasked_1774229487413.webp"
)


def _upload(base_url: str, img_path: str, requests_mod) -> str | None:
    if not os.path.exists(img_path):
        print(f"  [!] Image not found: {img_path}")
        return None
    with open(img_path, "rb") as f:
        raw = f.read()
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
    except Exception as e:
        print(f"  [!] PIL conversion failed ({e}), uploading as-is")
    files = {"image": ("ref.png", io.BytesIO(raw), "image/png")}
    try:
        r = requests_mod.post(f"{base_url}/upload/image", files=files, timeout=15)
        if r.status_code == 200:
            name = r.json().get("name", "")
            print(f"  [+] Uploaded → {name}")
            return name
        print(f"  [!] Upload failed HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"  [!] Upload error: {e}")
    return None


def _validate_workflow_structure(wf: dict) -> list[str]:
    """Check that every node reference [node_id, slot] points to an existing node."""
    errors = []
    node_ids = set(wf.keys())
    for nid, node in wf.items():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                ref_id = val[0]
                if ref_id not in node_ids:
                    errors.append(
                        f"  Node {nid} ({node.get('class_type','?')}).inputs[{key}] "
                        f"→ '{ref_id}' does not exist"
                    )
    return errors


def run():
    try:
        import requests
    except ImportError:
        print("[FAIL] 'requests' not installed.")
        sys.exit(1)

    print("=" * 60)
    print("  MultiRef Workflow Self-Test")
    print("=" * 60)
    print(f"  ComfyUI URL : {TEST_COMFYUI_URL}")
    print(f"  GGUF        : {GGUF}")
    print(f"  VAE         : {VAE}")
    print(f"  CLIP        : {CLIP}")
    print(f"  Resolution  : {W}x{H}  steps={STEPS}")
    print()

    # ── 0. Check ComfyUI is reachable ─────────────────────────────────────────
    print("[0] Checking ComfyUI reachability …")
    try:
        r = requests.get(f"{TEST_COMFYUI_URL}/system_stats", timeout=10)
        if r.status_code != 200:
            print(f"  [FAIL] /system_stats returned {r.status_code}")
            sys.exit(1)
        devices = r.json().get("devices", [{}])
        vram_free_mb = devices[0].get("torch_vram_free", 0) // (1024 * 1024) if devices else 0
        print(f"  [OK] ComfyUI online — VRAM free: {vram_free_mb} MB")
    except Exception as e:
        print(f"  [FAIL] Cannot reach ComfyUI: {e}")
        sys.exit(1)
    print()

    # ── 1. Upload reference images ─────────────────────────────────────────────
    print("[1] Uploading reference images …")
    name_a = _upload(TEST_COMFYUI_URL, REF_IMAGE_A, requests)
    name_b = _upload(TEST_COMFYUI_URL, REF_IMAGE_B, requests)
    if not name_a or not name_b:
        print("  [FAIL] Could not upload one or both reference images.")
        sys.exit(1)
    print()

    subject_filenames = {
        CHAR_A_NAME: [name_a],
        CHAR_B_NAME: [name_b],
    }
    subject_appearances = {
        CHAR_A_NAME: CHAR_A_APP,
        CHAR_B_NAME: CHAR_B_APP,
    }

    # ── 2. Build the workflow ──────────────────────────────────────────────────
    print("[2] Building multiref workflow …")
    from workflow_adapter import build_multiref_workflow
    seed = int(uuid.uuid4().int % (2 ** 31))
    wf = build_multiref_workflow(
        scene_prompt=SCENE_PROMPT,
        subject_filenames=subject_filenames,
        subject_appearances=subject_appearances,
        unet_name=GGUF,
        vae_name=VAE,
        clip_name=CLIP,
        seed=seed,
        steps=STEPS,
        width=W,
        height=H,
    )
    print(f"  [+] Workflow built — {len(wf)} nodes")
    print()

    # ── 3. Structural validation ───────────────────────────────────────────────
    print("[3] Structural validation (node reference integrity) …")
    errs = _validate_workflow_structure(wf)
    if errs:
        print(f"  [FAIL] {len(errs)} broken reference(s):")
        for e in errs:
            print(e)
        sys.exit(1)
    print("  [OK] All node references are valid")
    print()

    # ── 4. Strip _meta before submitting ──────────────────────────────────────
    clean_wf = {
        nid: {k: v for k, v in node.items() if k != "_meta"}
        for nid, node in wf.items()
    }

    # ── 5. Submit to ComfyUI /prompt ──────────────────────────────────────────
    print("[4] Submitting to ComfyUI /prompt …")
    client_id = str(uuid.uuid4())
    payload = {"prompt": clean_wf, "client_id": client_id}
    resp = requests.post(f"{TEST_COMFYUI_URL}/prompt", json=payload, timeout=15)
    if resp.status_code != 200:
        print(f"  [FAIL] /prompt returned HTTP {resp.status_code}")
        try:
            err = resp.json()
            top = err.get("error", {})
            if top:
                print(f"  Error: {top.get('type')} — {top.get('message')} — {top.get('details','')}")
            for nid, ne in err.get("node_errors", {}).items():
                ct = clean_wf.get(nid, {}).get("class_type", nid)
                for e in ne.get("errors", []):
                    print(f"  Node {nid} ({ct}): [{e.get('type')}] {e.get('message')} — {e.get('details','')}")
        except Exception:
            print(f"  Raw: {resp.text[:500]}")
        sys.exit(1)

    prompt_id = resp.json().get("prompt_id")
    print(f"  [OK] Job queued — prompt_id={prompt_id}")
    print()

    # ── 6. Poll for completion ────────────────────────────────────────────────
    print("[5] Waiting for ComfyUI to finish …")
    deadline = time.time() + 300
    poll_fail = 0
    while time.time() < deadline:
        time.sleep(3)
        try:
            h = requests.get(f"{TEST_COMFYUI_URL}/history/{prompt_id}", timeout=15)
        except Exception:
            poll_fail += 1
            if poll_fail >= 20:
                print("  [FAIL] ComfyUI not responding.")
                sys.exit(1)
            continue
        poll_fail = 0
        if h.status_code != 200 or prompt_id not in h.json():
            continue
        job = h.json()[prompt_id]
        if not job.get("status", {}).get("completed"):
            elapsed = int(deadline - time.time())
            print(f"  … still running ({elapsed}s remaining)", end="\r")
            continue

        # Collect output images
        out_images = []
        for out in job.get("outputs", {}).values():
            for img in out.get("images", []):
                out_images.append(img)

        if not out_images:
            print("\n  [FAIL] Job completed but no images in output.")
            sys.exit(1)

        print(f"\n  [OK] Job completed — {len(out_images)} image(s)")
        print()

        # ── 7. Download and save ──────────────────────────────────────────────
        print("[6] Downloading output image …")
        img_info = out_images[0]
        params = {"filename": img_info["filename"], "type": img_info.get("type", "output")}
        if img_info.get("subfolder"):
            params["subfolder"] = img_info["subfolder"]
        dl = requests.get(f"{TEST_COMFYUI_URL}/view", params=params, timeout=30)
        if dl.status_code != 200:
            print(f"  [FAIL] Download HTTP {dl.status_code}")
            sys.exit(1)

        out_path = os.path.join(os.path.dirname(__file__), "attached_assets", "_test_multiref_output.png")
        with open(out_path, "wb") as f:
            f.write(dl.content)
        print(f"  [OK] Saved to: {out_path}")
        print()
        print("=" * 60)
        print("  PASS — multiref workflow is working correctly.")
        print("=" * 60)
        return

    print("\n  [FAIL] Timed out waiting for ComfyUI.")
    sys.exit(1)


if __name__ == "__main__":
    run()
