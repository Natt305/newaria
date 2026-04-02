"""
test_2pass_hug.py
-----------------
Two-pass contact-pose test:

  Pass 1 — side-by-side with hard spatial masks (proven, Mortis~8/10 Nina~10/10)
  Pass 2 — img2img on the Pass-1 output: nudge characters into a hugging pose
            using both ReferenceLatents + a hugging scene prompt, denoise ~0.6

Run: python test_2pass_hug.py [--denoise 0.55] [--steps 8]
"""

import argparse, base64, io, json, os, re, sys, time, uuid
import requests
from PIL import Image

sys.path.insert(0, ".")
from workflow_adapter import build_multiref_workflow, build_img2img_workflow

# ── config ────────────────────────────────────────────────────────────────────
BASE  = os.environ.get("COMFYUI_URL", "https://gina-fatherless-shavonda.ngrok-free.dev")
GGUF  = os.environ.get("COMFYUI_GGUF", "flux-2-klein-4b-Q5_K_M.gguf")
VAE   = os.environ.get("COMFYUI_VAE",  "flux2-vae.safetensors")
CLIP  = os.environ.get("COMFYUI_CLIP", "qwen_3_4b.safetensors")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

MORTIS_PATH = "attached_assets/Mortis_Anime_unmasked_1774831761833.webp"
NINA_PATH   = "attached_assets/圖片_1774831943302.png"

MORTIS_APPEARANCE = (
    "beautiful anime girl, Mortis, long straight silver-white hair, small red beret, "
    "bright golden amber eyes, pale skin, black frilled gothic lolita dress with bell skirt, "
    "white ruffled collar cravat, long black coat over dress, silver cross brooch on chest, "
    "dark gray opaque tights, red platform mary jane shoes with black straps"
)
NINA_APPEARANCE = (
    "beautiful anime girl, Tokazaki Nina, short brown hair with twin tails, bright blue eyes, "
    "light skin, white graphic t-shirt under dark navy hooded bomber jacket with white fur collar trim, "
    "red plaid tartan pleated mini skirt, leather belt, white knee socks, black canvas sneakers"
)

# Pass 1: proven side-by-side — ConditioningSetMask hard masks deliver 9.6/10 and 10/10
PASS1_PROMPT = (
    "Mortis and Nina standing side by side on a rooftop at sunset, "
    "full body view, anime illustration, dramatic sky background"
)
# Pass 2: ref-guided pose nudge toward contact / hugging
PASS2_PROMPT = (
    "Mortis and Nina hugging each other warmly on a rooftop at sunset, "
    "arms around each other, leaning close, full body view, "
    "anime illustration, dramatic sky background"
)

WIDTH, HEIGHT = 512, 768

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--seed",  type=int, default=-1, help="Pass-1 seed (-1 = random)")
ap.add_argument("--sweep", type=int, default=1,
                help="Run this many different seeds for Pass 1 and pick the best (default 1)")
args = ap.parse_args()

base_seed = args.seed if args.seed >= 0 else int(uuid.uuid4().int % (2**31))

# ── helpers ───────────────────────────────────────────────────────────────────
def to_png_bytes(path: str) -> bytes:
    img = Image.open(path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    buf = io.BytesIO()
    bg.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

def upload(raw: bytes, fname: str) -> str:
    r = requests.post(
        f"{BASE}/upload/image",
        files={"image": (fname, raw, "image/png")},
        data={"overwrite": "true"},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    sub = d.get("subfolder", "")
    return f"{sub}/{d['name']}" if sub else d["name"]

def submit(wf: dict) -> str:
    clean = {nid: {k: v for k, v in n.items() if k != "_meta"}
             for nid, n in wf.items()}
    r = requests.post(f"{BASE}/prompt",
                      json={"prompt": clean, "client_id": str(uuid.uuid4())},
                      timeout=15)
    if r.status_code != 200:
        print(f"  ✗ submit error {r.status_code}: {r.text[:400]}")
        sys.exit(1)
    return r.json()["prompt_id"]

def wait_download(pid: str, label: str, save: str) -> bytes:
    print(f"  Waiting for {label}...", end="", flush=True)
    for _ in range(300):
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
                with open(save, "wb") as f:
                    f.write(data)
                print(f" ✓ → {save}")
                return data
        print(".", end="", flush=True)
    print(" TIMEOUT")
    sys.exit(1)

def score(img_bytes: bytes, char: str, appearance: str) -> dict:
    if not GROQ_KEY:
        return {}
    from groq import Groq
    b64 = base64.b64encode(img_bytes).decode()
    client = Groq(api_key=GROQ_KEY)
    prompt = (
        f"Quality-control reviewer. Reference character: {char} — {appearance}\n"
        f"Score how well the character in the image matches (0–10 each): "
        f"hair, eyes, outfit, accessories, overall likeness.\n"
        f"Also note whether the two characters appear to be hugging or touching.\n"
        f"Reply JSON only: {{\"hair\":N,\"eyes\":N,\"outfit\":N,\"accessories\":N,"
        f"\"likeness\":N,\"contact\":true/false,\"notes\":\"...\"}}"
    )
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        max_tokens=350,
    )
    txt = resp.choices[0].message.content
    m = re.search(r'\{.*\}', txt, re.S)
    return json.loads(m.group()) if m else {"raw": txt}

def avg_score(s: dict) -> float:
    keys = ["hair", "eyes", "outfit", "accessories", "likeness"]
    vals = [s.get(k, 0) for k in keys if isinstance(s.get(k), (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else 0.0

# ── MAIN ──────────────────────────────────────────────────────────────────────
print("═" * 62)
print(f"  2-Pass Hug Test")
print(f"  PASS 1 → ConditioningSetMask HARD MASK  (side-by-side)")
print(f"  PASS 2 → IMG2IMG pose transfer          (hugging)")
print(f"  base_seed={base_seed}  sweep={args.sweep}")
print("═" * 62)

# ── Step 0: Upload reference photos ───────────────────────────────────────────
print("\n── Step 0: Upload reference photos ──")
mortis_raw = to_png_bytes(MORTIS_PATH)
nina_raw   = to_png_bytes(NINA_PATH)
mortis_cf  = upload(mortis_raw, "Mortis.png")
nina_cf    = upload(nina_raw,   "Nina.png")
print(f"  Mortis → {mortis_cf}")
print(f"  Nina   → {nina_cf}")

# ── PASS 1: ConditioningSetMask HARD MASK side-by-side ────────────────────────
# Architecture:
#   CLIPTextEncode (per char) → ReferenceLatent chain
#   SolidMask + MaskComposite → ConditioningSetMask (mask bounds, strength=1.0)
#   ConditioningCombine → CFGGuider → SamplerCustomAdvanced
#
# Sweep N seeds; keep the one with the highest combined vision score.
# ─────────────────────────────────────────────────────────────────────────────
sweep_seeds = [
    (base_seed + i * 373) % (2**31)   # spread seeds across the 32-bit range
    for i in range(args.sweep)
]

best_seed   = sweep_seeds[0]
best_bytes  = None
best_total  = -1.0

for idx, p1_seed in enumerate(sweep_seeds):
    tag = f"[{idx+1}/{args.sweep}] seed={p1_seed}"
    print(f"\n── PASS 1 {tag} — Hard-mask side-by-side ──")

    wf1 = build_multiref_workflow(
        scene_prompt=PASS1_PROMPT,
        subject_filenames={"Mortis": [mortis_cf], "Nina": [nina_cf]},
        subject_appearances={"Mortis": MORTIS_APPEARANCE, "Nina": NINA_APPEARANCE},
        unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
        seed=p1_seed, steps=4, width=WIDTH, height=HEIGHT,
        contact_pose=False,     # ← HARD MASK (ConditioningSetMask, mask bounds)
    )
    print(f"  Hard-mask nodes: {len(wf1)}")
    p1_bytes = wait_download(
        submit(wf1), f"Pass 1 seed={p1_seed}",
        f"attached_assets/_2pass_p1_{idx}.png",
    )

    if GROQ_KEY:
        m1 = score(p1_bytes, "Mortis", MORTIS_APPEARANCE)
        n1 = score(p1_bytes, "Nina",   NINA_APPEARANCE)
        total = avg_score(m1) + avg_score(n1)
        print(f"  Mortis {avg_score(m1)}/10 · Nina {avg_score(n1)}/10 · total={total:.1f}")
        print(f"    Mortis: {m1.get('notes','')[:90]}")
        print(f"    Nina:   {n1.get('notes','')[:90]}")
        if total > best_total:
            best_total = total
            best_seed  = p1_seed
            best_bytes = p1_bytes
    else:
        # No scoring — keep the last seed
        best_seed  = p1_seed
        best_bytes = p1_bytes

# Save the winning Pass-1 image
import shutil
shutil.copy(f"attached_assets/_2pass_p1_{sweep_seeds.index(best_seed)}.png",
            "attached_assets/_2pass_p1_sidebyside.png")
pass1_cf = upload(best_bytes, "2pass_p1.png")
print(f"\n  ✓ Best Pass-1: seed={best_seed}  score={best_total:.1f}  → {pass1_cf}")

# ── PASS 2: IMG2IMG pose transfer (layout ReferenceLatent) ────────────────────
# Architecture:
#   LoadImage(pass1) → VAEEncode → layout_latent
#   Per char: CLIPTextEncode → ReferenceLatent(photo) → ReferenceLatent(layout)
#   ConditioningCombine (plain, no spatial masks) → CFGGuider
#   EmptyFlux2LatentImage → Flux2Scheduler → SamplerCustomAdvanced
#
# FLUX.2 Klein is a distilled 4-step model: SplitSigmasDenoise produces pure
# noise.  The layout ReferenceLatent is the correct img2img substitute — it
# anchors composition from Pass 1 while the hugging prompt steers the pose.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── PASS 2 — IMG2IMG pose transfer (hugging) ──")
print(f"  Input:       Pass-1 side-by-side ({pass1_cf})")
print(f"  Method:      layout VAEEncode → ReferenceLatent anchor per character")
print(f"  Mask:        NONE (plain ConditioningCombine — let chars move together)")

wf2 = build_img2img_workflow(
    input_image_filename=pass1_cf,
    scene_prompt=PASS2_PROMPT,
    subject_filenames={"Mortis": [mortis_cf], "Nina": [nina_cf]},
    subject_appearances={"Mortis": MORTIS_APPEARANCE, "Nina": NINA_APPEARANCE},
    unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
    seed=best_seed + 1, steps=4,
    width=WIDTH, height=HEIGHT,
)
print(f"  Img2img nodes: {len(wf2)}")
pass2_bytes = wait_download(
    submit(wf2), "Pass 2 img2img",
    "attached_assets/_2pass_p2_hugging.png",
)

# ── Vision analysis ───────────────────────────────────────────────────────────
print("\n── Vision analysis: img2img result ──")
if GROQ_KEY:
    m2 = score(pass2_bytes, "Mortis", MORTIS_APPEARANCE)
    n2 = score(pass2_bytes, "Nina",   NINA_APPEARANCE)
    print(f"  Mortis  {avg_score(m2)}/10  contact={m2.get('contact')}")
    print(f"    {m2.get('notes','')}")
    print(f"  Nina    {avg_score(n2)}/10  contact={n2.get('contact')}")
    print(f"    {n2.get('notes','')}")
    if m2.get("contact") or n2.get("contact"):
        print("\n  ✓ CONTACT DETECTED — characters are hugging!")
    else:
        print("\n  ✗ No contact — try a different base_seed")
else:
    print("  (GROQ_API_KEY not set — skipping vision scoring)")

print("\n══ Done ══")
print(f"  Pass 1 (hard mask):  attached_assets/_2pass_p1_sidebyside.png  (seed={best_seed})")
print(f"  Pass 2 (img2img):    attached_assets/_2pass_p2_hugging.png")
