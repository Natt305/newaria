"""
test_inpaint_hugging.py
-----------------------
Three-pass inpaint test for contact/hugging poses.

Pass 1 — layout:   scene text only → free pose generation (2 chars hugging)
Pass 2a — Mortis:  inpaint left half with Mortis reference + appearance
Pass 2b — Nina:    inpaint right half of Pass-2a result with Nina reference

No SAM, no ControlNet, no extra models required.
"""

import io, os, sys, time, uuid, requests
from PIL import Image

BASE  = os.environ.get("COMFYUI_URL", "https://gina-fatherless-shavonda.ngrok-free.dev")
GGUF  = os.environ.get("COMFYUI_GGUF",  "flux-2-klein-4b-Q5_K_M.gguf")
VAE   = os.environ.get("COMFYUI_VAE",   "flux2-vae.safetensors")
CLIP  = os.environ.get("COMFYUI_CLIP",  "qwen_3_4b.safetensors")

MORTIS_REF = "attached_assets/Mortis_Anime_unmasked_1774831761833.webp"
NINA_REF   = "attached_assets/圖片_1774831943302.png"

MORTIS_APPEARANCE = (
    "1girl, Mortis, long silver white hair, red beret hat, golden amber eyes, "
    "pale skin, black gothic military coat, crimson red shorts, gray tights, "
    "red platform mary jane shoes, cross brooch, elegant gothic lolita aesthetic"
)
NINA_APPEARANCE = (
    "1girl, Tokazaki Nina (仁菜), short brown hair, small twin tails, blue eyes, "
    "light skin, white graphic t-shirt, dark navy fur-trim hooded bomber jacket, "
    "red tartan plaid skirt, leather belt, white socks, black sneakers, "
    "casual Japanese school girl aesthetic"
)

SCENE_PROMPT = (
    "Mortis and Nina hugging each other warmly on a rooftop at sunset, "
    "side by side, arms around each other, full body view, "
    "anime illustration, dramatic sky background"
)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_KEY     = os.environ.get("GROQ_API_KEY", "")

WIDTH, HEIGHT = 512, 768


# ── helpers ───────────────────────────────────────────────────────────────────

def img_to_png_bytes(path: str) -> bytes:
    img = Image.open(path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    buf = io.BytesIO()
    bg.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

def bytes_to_b64(b: bytes) -> str:
    import base64
    return base64.b64encode(b).decode()

def upload_to_comfyui(raw_bytes: bytes, fname: str) -> str:
    r = requests.post(
        f"{BASE}/upload/image",
        files={"image": (fname, raw_bytes, "image/png")},
        data={"overwrite": "true"},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    sub = d.get("subfolder", "")
    return f"{sub}/{d['name']}" if sub else d["name"]

def submit(workflow: dict) -> str:
    clean = {nid: {k: v for k, v in n.items() if k != "_meta"}
             for nid, n in workflow.items()}
    r = requests.post(f"{BASE}/prompt",
                      json={"prompt": clean, "client_id": str(uuid.uuid4())},
                      timeout=15)
    if r.status_code != 200:
        print(f"  ✗ submit failed {r.status_code}: {r.text[:400]}")
        sys.exit(1)
    return r.json()["prompt_id"]

def wait_and_download(pid: str, label: str, save_path: str) -> bytes:
    print(f"  Waiting for {label}...", end="", flush=True)
    for _ in range(240):
        time.sleep(2)
        h = requests.get(f"{BASE}/history/{pid}", timeout=10).json()
        entry = h.get(pid, {})
        if entry.get("status", {}).get("completed"):
            imgs = []
            for node_out in entry.get("outputs", {}).values():
                imgs.extend(node_out.get("images", []))
            if imgs:
                fn, sf = imgs[-1]["filename"], imgs[-1].get("subfolder","")
                url = f"{BASE}/view?filename={fn}&subfolder={sf}&type=output"
                data = requests.get(url, timeout=30).content
                with open(save_path, "wb") as f:
                    f.write(data)
                print(f" ✓ saved → {save_path}")
                return data
        print(".", end="", flush=True)
    print(" TIMEOUT")
    sys.exit(1)

def vision_score(img_bytes: bytes, char_name: str, appearance: str) -> dict:
    if not GROQ_KEY:
        return {"skipped": True}
    import groq as groq_module
    client = groq_module.Groq(api_key=GROQ_KEY)
    b64 = bytes_to_b64(img_bytes)
    prompt = (
        f'You are a quality-control reviewer for an AI image generation system.\n'
        f'Reference character: {char_name} — {appearance}\n\n'
        f'Score how well the character in the image matches the reference (0–10 each):\n'
        f'hair, eyes, outfit, accessories, overall likeness.\n'
        f'Reply with JSON only: {{"hair":N,"eyes":N,"outfit":N,"accessories":N,"likeness":N,"notes":"..."}}'
    )
    resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        max_tokens=300,
    )
    import json, re
    txt = resp.choices[0].message.content
    m = re.search(r'\{.*\}', txt, re.S)
    return json.loads(m.group()) if m else {"raw": txt}


# ── main ──────────────────────────────────────────────────────────────────────

from workflow_adapter import build_layout_workflow, build_inpaint_char_workflow

import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--mode", choices=["text", "ref", "both"], default="text",
                help="text=text-only inpaint (default), ref=with ReferenceLatent, both=run both")
_args = _p.parse_args()

print("═" * 60)
print(f"  TEST: Inpaint Hugging — Mortis & Nina (mode={_args.mode})")
print("═" * 60)

seed = int(uuid.uuid4().int % (2**31))

# Step 1: Upload reference photos
print("\n── Step 1: Upload reference photos ──")
mortis_raw = img_to_png_bytes(MORTIS_REF)
nina_raw   = img_to_png_bytes(NINA_REF)
mortis_cf  = upload_to_comfyui(mortis_raw, "Mortis.png")
nina_cf    = upload_to_comfyui(nina_raw,   "Nina.png")
print(f"  ✓ Mortis → {mortis_cf}")
print(f"  ✓ Nina   → {nina_cf}")

# ── Pass 1: layout ────────────────────────────────────────────────────────────
print("\n── Pass 1: Generate hugging layout (no character refs) ──")
wf1 = build_layout_workflow(
    scene_prompt=SCENE_PROMPT,
    unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
    seed=seed, steps=4, width=WIDTH, height=HEIGHT,
)
print(f"  Nodes: {len(wf1)}")
pid1 = submit(wf1)
print(f"  ✓ submitted — {pid1}")
layout_bytes = wait_and_download(pid1, "layout", "attached_assets/_inpaint_pass1_layout.png")

# Upload layout image back to ComfyUI for use in Pass 2
layout_cf = upload_to_comfyui(layout_bytes, "inpaint_layout.png")
print(f"  ✓ layout re-uploaded → {layout_cf}")

# ── Pass 2a: inpaint Mortis (left) ───────────────────────────────────────────
print("\n── Pass 2a: Inpaint Mortis into left half ──")
wf2a = build_inpaint_char_workflow(
    layout_image_filename=layout_cf,
    char_name="Mortis",
    char_filenames=[mortis_cf],
    char_appearance=MORTIS_APPEARANCE,
    scene_prompt=SCENE_PROMPT,
    side="left",
    unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
    seed=seed + 1, steps=6, width=WIDTH, height=HEIGHT, overlap=96,
)
print(f"  Nodes: {len(wf2a)}")
pid2a = submit(wf2a)
print(f"  ✓ submitted — {pid2a}")
pass2a_bytes = wait_and_download(pid2a, "Mortis inpaint", "attached_assets/_inpaint_pass2a_mortis.png")

# Upload Pass-2a result for Pass-2b
pass2a_cf = upload_to_comfyui(pass2a_bytes, "inpaint_after_mortis.png")
print(f"  ✓ pass-2a re-uploaded → {pass2a_cf}")

# ── Pass 2b: inpaint Nina (right) ────────────────────────────────────────────
print("\n── Pass 2b: Inpaint Nina into right half ──")
wf2b = build_inpaint_char_workflow(
    layout_image_filename=pass2a_cf,
    char_name="Nina",
    char_filenames=[nina_cf],
    char_appearance=NINA_APPEARANCE,
    scene_prompt=SCENE_PROMPT,
    side="right",
    unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
    seed=seed + 2, steps=6, width=WIDTH, height=HEIGHT, overlap=96,
)
print(f"  Nodes: {len(wf2b)}")
pid2b = submit(wf2b)
print(f"  ✓ submitted — {pid2b}")
final_bytes = wait_and_download(pid2b, "Nina inpaint", "attached_assets/_inpaint_final.png")

# ── Vision analysis ───────────────────────────────────────────────────────────
print("\n── Step 3: Vision analysis ──")
if GROQ_KEY:
    m_score = vision_score(final_bytes, "Mortis", MORTIS_APPEARANCE)
    n_score = vision_score(final_bytes, "Nina",   NINA_APPEARANCE)
    def avg(s): return round(sum(s.get(k, 0) for k in ["hair","eyes","outfit","accessories","likeness"]) / 5, 1)
    print(f"\n  Mortis  avg {avg(m_score)}/10  {m_score}")
    print(f"  Nina    avg {avg(n_score)}/10  {n_score}")
else:
    print("  (GROQ_API_KEY not set — skipping vision scoring)")

print("\n══ Done. Final image: attached_assets/_inpaint_final.png ══")
