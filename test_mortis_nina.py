"""
Test: store Mortis & 仁菜 as KB entries, then run multiref workflow with them.
Run manually: python test_mortis_nina.py
"""
import io, json, os, sys, time, uuid
sys.path.insert(0, ".")

import requests
from PIL import Image

BASE  = os.getenv("COMFYUI_URL", "https://gina-fatherless-shavonda.ngrok-free.dev")
GGUF  = os.getenv("COMFYUI_GGUF", "flux-2-klein-4b-Q5_K_M.gguf")
VAE   = os.getenv("COMFYUI_VAE",  "flux2-vae.safetensors")
CLIP  = os.getenv("COMFYUI_CLIP", "qwen_3_4b.safetensors")

MORTIS_PATH = "attached_assets/Mortis_Anime_unmasked_1774831761833.webp"
NINA_PATH   = "attached_assets/圖片_1774831943302.png"

MORTIS_APPEARANCE = (
    "1girl, Mortis, long silver white hair, red beret hat, golden amber eyes, "
    "pale skin, black gothic military coat, crimson red shorts, gray tights, "
    "red platform mary jane shoes, cross brooch, elegant gothic lolita aesthetic"
)
NINA_APPEARANCE = (
    "1girl, Tokazaki Nina, short brown hair, small twin tails, blue eyes, "
    "light skin, white graphic t-shirt, dark navy fur-trim hooded bomber jacket, "
    "red tartan plaid skirt, leather belt, white socks, black sneakers, "
    "casual Japanese school girl aesthetic"
)

# ── helpers ──────────────────────────────────────────────────────────────────

def to_png_bytes(path: str) -> bytes:
    img = Image.open(path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    rgb = bg.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()

def upload_to_comfyui(raw: bytes, label: str) -> str:
    r = requests.post(
        f"{BASE}/upload/image",
        files={"image": (f"{label}.png", io.BytesIO(raw), "image/png")},
        timeout=15,
    )
    r.raise_for_status()
    name = r.json().get("name")
    print(f"  ✓ uploaded {label} → {name}")
    return name

def poll_and_download(pid: str, out_path: str, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        h = requests.get(f"{BASE}/history/{pid}", timeout=15)
        if h.status_code != 200 or pid not in h.json():
            continue
        j = h.json()[pid]
        if not j.get("status", {}).get("completed"):
            continue
        imgs = [i for o in j.get("outputs", {}).values() for i in o.get("images", [])]
        if not imgs:
            print("  ✗ completed but no output images"); return False
        ii = imgs[0]
        params = {"filename": ii["filename"], "type": ii.get("type", "output")}
        if ii.get("subfolder"):
            params["subfolder"] = ii["subfolder"]
        dl = requests.get(f"{BASE}/view", params=params, timeout=30)
        with open(out_path, "wb") as f:
            f.write(dl.content)
        print(f"  ✓ saved → {out_path}"); return True
    print("  ✗ timed out"); return False

# ── Step 1: KB entries ────────────────────────────────────────────────────────

import database

print("\n=== Step 1: Create KB image entries ===")

mortis_bytes = to_png_bytes(MORTIS_PATH)
nina_bytes   = to_png_bytes(NINA_PATH)

mortis_id = database.add_image_entry(
    title="Mortis",
    image_bytes=mortis_bytes,
    mime_type="image/png",
    appearance_description=MORTIS_APPEARANCE,
    display_description="Mortis — silver-haired gothic lolita with a red beret. Calm, mysterious demeanor.",
    tags="character,mortis,gothic",
)
print(f"  ✓ Mortis saved as KB entry #{mortis_id}")

nina_id = database.add_image_entry(
    title="仁菜",
    image_bytes=nina_bytes,
    mime_type="image/png",
    appearance_description=NINA_APPEARANCE,
    display_description="Tokazaki Nina (仁菜) — energetic brown-haired girl with a casual street fashion style.",
    tags="character,nina,仁菜,casual",
)
print(f"  ✓ 仁菜 saved as KB entry #{nina_id}")

# ── Step 2: Upload reference images to ComfyUI ─────────────────────────────

print("\n=== Step 2: Upload reference images to ComfyUI ===")
mortis_cf = upload_to_comfyui(mortis_bytes, "Mortis")
nina_cf   = upload_to_comfyui(nina_bytes, "Nina")

# ── Step 3: Build and submit multiref workflow ────────────────────────────

print("\n=== Step 3: Build & submit multiref workflow ===")
from workflow_adapter import build_multiref_workflow

scene = (
    "Mortis and Nina standing together on a rooftop at sunset, "
    "full body view, anime illustration, dramatic sky background"
)

wf = build_multiref_workflow(
    scene_prompt=scene,
    subject_filenames={"Mortis": [mortis_cf], "Nina": [nina_cf]},
    subject_appearances={"Mortis": MORTIS_APPEARANCE, "Nina": NINA_APPEARANCE},
    unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
    seed=int(uuid.uuid4().int % (2**31)),
    steps=4, width=512, height=768,
)
print(f"  Nodes: {len(wf)}")

clean = {nid: {k: v for k, v in node.items() if k != "_meta"} for nid, node in wf.items()}
r = requests.post(
    f"{BASE}/prompt",
    json={"prompt": clean, "client_id": str(uuid.uuid4())},
    timeout=15,
)
if r.status_code != 200:
    print(f"  ✗ submit failed {r.status_code}: {r.text[:600]}"); sys.exit(1)
pid = r.json()["prompt_id"]
print(f"  ✓ submitted — prompt_id: {pid}")

# ── Step 4: Wait for output ───────────────────────────────────────────────

print("\n=== Step 4: Waiting for ComfyUI output... ===")
ok = poll_and_download(pid, "attached_assets/_test_mortis_nina_output.png")

if ok:
    print("\n✓ TEST PASSED — output: attached_assets/_test_mortis_nina_output.png")
    print(f"  KB entries: Mortis=#{mortis_id}, 仁菜=#{nina_id}")
else:
    print("\n✗ TEST FAILED"); sys.exit(1)
