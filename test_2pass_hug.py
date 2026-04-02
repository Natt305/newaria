"""
test_2pass_hug.py
-----------------
Two-pass contact-pose test:

  Pass 1 — side-by-side with hard spatial masks (proven, Mortis~8/10 Nina~10/10)
  Pass 2 — img2img on the Pass-1 output: nudge characters into a hugging pose
            using both ReferenceLatents + a hugging scene prompt, denoise ~0.6

Run: python test_2pass_hug.py [--denoise 0.55] [--steps 8]
"""

import argparse, base64, io, json, os, re, shutil, sys, time, uuid
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
    "1girl, Mortis, long pale mint-grey hair, light silver with subtle green tint, "
    "deep burgundy red beret hat, warm amber eyes, pale skin, "
    "black gothic military coat, crimson red shorts, dark gray tights, "
    "dark red platform mary jane shoes, cross brooch with green gem, elegant gothic lolita aesthetic"
)
NINA_APPEARANCE = (
    "1girl, Tokazaki Nina (仁菜), short brown hair, small twin tails, blue eyes, "
    "light skin, white graphic t-shirt, dark navy fur-trim hooded bomber jacket, "
    "red tartan plaid skirt, leather belt, white socks, black sneakers, "
    "casual Japanese school girl aesthetic"
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
ap.add_argument("--seed",     type=int, default=-1, help="Pass-1 seed (-1 = random)")
ap.add_argument("--p2-tries", type=int, default=1,
                help="How many different seeds to try for Pass 2 (pick best)")
args = ap.parse_args()

seed = args.seed if args.seed >= 0 else int(uuid.uuid4().int % (2**31))

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

def score(img_bytes: bytes, char: str, appearance: str,
          ref_bytes: bytes | None = None) -> dict:
    """Score a generated image for character fidelity.

    When ref_bytes is supplied the vision model can directly compare the
    generated image against the real reference photo (same method as
    test_mortis_nina.py), giving much more accurate scores than text-only.
    """
    if not GROQ_KEY:
        return {}
    from groq import Groq
    b64     = base64.b64encode(img_bytes).decode()
    client  = Groq(api_key=GROQ_KEY)

    content: list = []
    if ref_bytes:
        ref_b64 = base64.b64encode(ref_bytes).decode()
        content += [
            {"type": "text",
             "text": f"REFERENCE PHOTO — {char} (use this as the ground truth appearance):"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{ref_b64}"}},
        ]

    content += [
        {"type": "text", "text": (
            f"GENERATED IMAGE — should contain {char}: {appearance}\n"
            f"Score how well {char} in the generated image matches "
            f"the reference photo and description (0–10 each): "
            f"hair, eyes, outfit, accessories, overall likeness.\n"
            f"Also note whether the two characters appear to be hugging or touching.\n"
            f"Reply JSON only: {{\"hair\":N,\"eyes\":N,\"outfit\":N,\"accessories\":N,"
            f"\"likeness\":N,\"contact\":true/false,\"notes\":\"...\"}}"
        )},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]

    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": content}],
                max_tokens=350,
                temperature=0.1,
            )
            txt = resp.choices[0].message.content
            m = re.search(r'\{.*\}', txt, re.S)
            return json.loads(m.group()) if m else {"raw": txt}
        except Exception as e:
            wait = 2 ** attempt * 3
            print(f"\n  [score] Groq error (attempt {attempt+1}/4): {e} — retrying in {wait}s")
            time.sleep(wait)
    return {"raw": "scoring failed after retries"}

def avg_score(s: dict) -> float:
    keys = ["hair", "eyes", "outfit", "accessories", "likeness"]
    vals = [s.get(k, 0) for k in keys if isinstance(s.get(k), (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else 0.0

# ── MAIN ──────────────────────────────────────────────────────────────────────
print("═" * 62)
print(f"  2-Pass Hug Test  seed={seed}")
print(f"  PASS 1 → ConditioningSetMask HARD MASK  (side-by-side)")
print(f"  PASS 2 → IMG2IMG pose transfer          (hugging)")
print("═" * 62)

# ── Step 0: Upload reference photos ───────────────────────────────────────────
print("\n── Step 0: Upload reference photos ──")
mortis_raw = to_png_bytes(MORTIS_PATH)
nina_raw   = to_png_bytes(NINA_PATH)
mortis_cf  = upload(mortis_raw, "Mortis.png")
nina_cf    = upload(nina_raw,   "Nina.png")
print(f"  Mortis → {mortis_cf}")
print(f"  Nina   → {nina_cf}")

# ── PASS 1: ConditioningSetMask HARD MASK ─────────────────────────────────────
# CLIPTextEncode → ReferenceLatent chain
# SolidMask + MaskComposite → ConditioningSetMask (mask bounds, strength=1.0)
# ConditioningCombine → CFGGuider → SamplerCustomAdvanced
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── PASS 1: Hard-mask side-by-side (seed={seed}) ──")
wf1 = build_multiref_workflow(
    scene_prompt=PASS1_PROMPT,
    subject_filenames={"Mortis": [mortis_cf], "Nina": [nina_cf]},
    subject_appearances={"Mortis": MORTIS_APPEARANCE, "Nina": NINA_APPEARANCE},
    unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
    seed=seed, steps=4, width=WIDTH, height=HEIGHT,
    contact_pose=False,     # ← HARD MASK (ConditioningSetMask, mask bounds)
)
print(f"  Nodes: {len(wf1)}")
pass1_bytes = wait_download(
    submit(wf1), "Pass 1 hard-mask",
    "attached_assets/_2pass_p1_sidebyside.png",
)

if GROQ_KEY:
    print("\n  Pass 1 scores (with reference photos):")
    m1 = score(pass1_bytes, "Mortis", MORTIS_APPEARANCE, ref_bytes=mortis_raw)
    n1 = score(pass1_bytes, "Nina",   NINA_APPEARANCE,   ref_bytes=nina_raw)
    print(f"    Mortis {avg_score(m1)}/10 · {m1.get('notes','')[:80]}")
    print(f"    Nina   {avg_score(n1)}/10 · {n1.get('notes','')[:80]}")

pass1_cf = upload(pass1_bytes, "2pass_p1.png")
print(f"\n  Pass-1 re-uploaded → {pass1_cf}")

# ── PASS 2: IMG2IMG pose transfer ─────────────────────────────────────────────
# LoadImage(pass1) → VAEEncode → layout_latent
# Per char: CLIPTextEncode → ReferenceLatent(photo) → ReferenceLatent(layout)
#           → ConditioningSetAreaStrength(2.0) → ConditioningCombine
# EmptyFlux2LatentImage → Flux2Scheduler → SamplerCustomAdvanced
#
# Note: SplitSigmasDenoise is BROKEN for FLUX.2 Klein (distilled, 4-step).
# The layout ReferenceLatent is the correct img2img approach for this model.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n── PASS 2: IMG2IMG pose transfer  ({args.p2_tries} tr{'y' if args.p2_tries==1 else 'ies'}) ──")
print(f"  Input: {pass1_cf}")

best_p2_bytes  = None
best_p2_total  = -1.0
best_p2_seed   = seed + 1

for p2_k in range(args.p2_tries):
    p2_seed = seed + 1 + p2_k
    tag = f"[{p2_k+1}/{args.p2_tries}] seed={p2_seed}"
    print(f"\n  {tag}")
    wf2 = build_img2img_workflow(
        input_image_filename=pass1_cf,
        scene_prompt=PASS2_PROMPT,
        subject_filenames={"Mortis": [mortis_cf], "Nina": [nina_cf]},
        subject_appearances={"Mortis": MORTIS_APPEARANCE, "Nina": NINA_APPEARANCE},
        unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
        seed=p2_seed, steps=4,
        width=WIDTH, height=HEIGHT,
    )
    p2_bytes = wait_download(
        submit(wf2), f"Pass 2 {tag}",
        f"attached_assets/_2pass_p2_{p2_k}.png",
    )

    if GROQ_KEY:
        m2 = score(p2_bytes, "Mortis", MORTIS_APPEARANCE, ref_bytes=mortis_raw)
        n2 = score(p2_bytes, "Nina",   NINA_APPEARANCE,   ref_bytes=nina_raw)
        total = avg_score(m2) + avg_score(n2)
        contact = m2.get("contact") or n2.get("contact")
        print(f"    Mortis {avg_score(m2)}/10 · Nina {avg_score(n2)}/10 · "
              f"total={total:.1f} · contact={contact}")
        print(f"      M: {m2.get('notes','')[:75]}")
        print(f"      N: {n2.get('notes','')[:75]}")
        if total > best_p2_total:
            best_p2_total = total
            best_p2_seed  = p2_seed
            best_p2_bytes = p2_bytes
    else:
        best_p2_bytes = p2_bytes

best_idx = best_p2_seed - (seed + 1)
shutil.copy(f"attached_assets/_2pass_p2_{best_idx}.png",
            "attached_assets/_2pass_p2_hugging.png")

print(f"\n  ✓ Best Pass-2: seed={best_p2_seed}  total={best_p2_total:.1f}")

# For multi-try: re-score the winner cleanly; single-try already printed above
if GROQ_KEY and args.p2_tries > 1:
    print("\n── Best Pass-2 final scores ──")
    m2 = score(best_p2_bytes, "Mortis", MORTIS_APPEARANCE, ref_bytes=mortis_raw)
    n2 = score(best_p2_bytes, "Nina",   NINA_APPEARANCE,   ref_bytes=nina_raw)
    print(f"  Mortis  {avg_score(m2)}/10  contact={m2.get('contact')}")
    print(f"    {m2.get('notes','')}")
    print(f"  Nina    {avg_score(n2)}/10  contact={n2.get('contact')}")
    print(f"    {n2.get('notes','')}")
    if m2.get("contact") or n2.get("contact"):
        print("\n  ✓ CONTACT DETECTED — characters are hugging!")

print("\n══ Done ══")
print(f"  Pass 1 (hard mask, seed={seed}):        attached_assets/_2pass_p1_sidebyside.png")
print(f"  Pass 2 (img2img, seed={best_p2_seed}):  attached_assets/_2pass_p2_hugging.png")
