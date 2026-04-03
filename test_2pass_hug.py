"""
test_hug.py  (formerly test_2pass_hug.py)
------------------------------------------
Single-pass hugging pose test using multiple ReferenceLatent nodes.

Both character reference photos are fed directly as ReferenceLatents
inside build_multiref_workflow (contact_pose=True) — no img2img / no
two-pass needed.  The soft spatial masks let arms cross the centre line
naturally while keeping each character's appearance on the correct side.

Run:
    python test_2pass_hug.py [--seed N] [--tries N] [--no-score]
"""

import argparse, base64, io, json, os, re, shutil, sys, time, uuid
import requests
from PIL import Image

sys.path.insert(0, ".")
from workflow_adapter import build_multiref_workflow

# ── config ────────────────────────────────────────────────────────────────────
BASE     = os.environ.get("COMFYUI_URL", "https://gina-fatherless-shavonda.ngrok-free.dev")
GGUF     = os.environ.get("COMFYUI_GGUF", "flux-2-klein-4b-Q5_K_M.gguf")
VAE      = os.environ.get("COMFYUI_VAE",  "flux2-vae.safetensors")
CLIP     = os.environ.get("COMFYUI_CLIP", "qwen_3_4b.safetensors")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

MORTIS_PATH = "attached_assets/Mortis_Anime_unmasked_1774831761833.webp"
NINA_PATH   = "attached_assets/圖片_1774831943302.png"

MORTIS_APPEARANCE = (
    "1girl, Mortis, long pale silver-grey hair with subtle mint-green tint, "
    "deep burgundy red flat beret, warm amber-grey eyes, pale skin, "
    "black tailcoat open at the front revealing the waist, coat splits open at center, "
    "gold-beige trim and stud details along coat edges, flared coat tails behind, "
    "crimson red high-waisted shorts clearly visible at center (solid plain, no plaid, no pattern, gold buttons), "
    "white lace jabot at collar, green gem cross brooch, black wrist cuffs, "
    "dark gray opaque tights covering legs fully (no bare legs), "
    "dark red platform mary jane shoes, gothic lolita aristocrat aesthetic"
)
NINA_APPEARANCE = (
    "1girl, Tokazaki Nina (仁菜), short brown hair, small twin tails, blue eyes, "
    "light skin, white graphic t-shirt, dark navy fur-trim hooded bomber jacket, "
    "red tartan plaid skirt, leather belt, white socks, black sneakers, "
    "no beret, no hat, casual Japanese school girl aesthetic"
)

SCENE_PROMPT = (
    "only two girls, Mortis and Nina hugging each other warmly on a rooftop at sunset, "
    "arms wrapped around each other's backs, leaning close together, "
    "full body view, exactly two characters, no extra people, "
    "anime illustration, dramatic sky background"
)

WIDTH, HEIGHT = 640, 960

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--seed",     type=int, default=-1,
                help="Base seed (-1 = random)")
ap.add_argument("--tries",    type=int, default=1,
                help="How many seeds to try; picks best scored result")
ap.add_argument("--no-score", action="store_true",
                help="Skip Groq scoring (useful when daily token quota is spent)")
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
                fn, sf = imgs[-1]["filename"], imgs[-1].get("subfolder", "")
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
    if not GROQ_KEY:
        return {}
    from groq import Groq
    b64    = base64.b64encode(img_bytes).decode()
    client = Groq(api_key=GROQ_KEY)
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
            err_str = str(e)
            if "tokens per day" in err_str or "TPD" in err_str:
                print(f"\n  [score] Daily token quota reached — skipping score")
                return {"raw": "quota exceeded"}
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
print(f"  Single-pass Hug Test  base_seed={seed}  tries={args.tries}")
print(f"  contact_pose=True → soft spatial masks + multiple ReferenceLatents")
print("═" * 62)

print("\n── Upload reference photos ──")
mortis_raw = to_png_bytes(MORTIS_PATH)
nina_raw   = to_png_bytes(NINA_PATH)
mortis_cf  = upload(mortis_raw, "Mortis.png")
nina_cf    = upload(nina_raw,   "Nina.png")
print(f"  Mortis → {mortis_cf}")
print(f"  Nina   → {nina_cf}")

best_bytes = None
best_total = -1.0
best_seed  = seed

for k in range(args.tries):
    k_seed = seed + k
    tag    = f"[{k+1}/{args.tries}] seed={k_seed}"
    print(f"\n── {tag} ──")

    wf = build_multiref_workflow(
        scene_prompt=SCENE_PROMPT,
        subject_filenames={"Mortis": [mortis_cf], "Nina": [nina_cf]},
        subject_appearances={"Mortis": MORTIS_APPEARANCE, "Nina": NINA_APPEARANCE},
        unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
        seed=k_seed, steps=4, width=WIDTH, height=HEIGHT,
        contact_pose=True,
    )
    print(f"  Nodes: {len(wf)}")
    out_bytes = wait_download(
        submit(wf), tag,
        f"attached_assets/_hug_{k}.png",
    )

    if GROQ_KEY and not args.no_score:
        ms = score(out_bytes, "Mortis", MORTIS_APPEARANCE, ref_bytes=mortis_raw)
        ns = score(out_bytes, "Nina",   NINA_APPEARANCE,   ref_bytes=nina_raw)
        total   = avg_score(ms) + avg_score(ns)
        contact = ms.get("contact") or ns.get("contact")
        print(f"  Mortis {avg_score(ms)}/10 · Nina {avg_score(ns)}/10 · "
              f"total={total:.1f} · contact={contact}")
        print(f"    M: {ms.get('notes','')[:80]}")
        print(f"    N: {ns.get('notes','')[:80]}")
        if total > best_total:
            best_total = total
            best_seed  = k_seed
            best_bytes = out_bytes
    else:
        best_bytes = out_bytes
        best_seed  = k_seed

best_idx = best_seed - seed
shutil.copy(f"attached_assets/_hug_{best_idx}.png",
            "attached_assets/_hug_best.png")

print(f"\n  ✓ Best: seed={best_seed}  total={best_total:.1f}")
print("\n══ Done ══")
for k in range(args.tries):
    print(f"  try {k+1}: attached_assets/_hug_{k}.png")
print(f"  best:   attached_assets/_hug_best.png")
