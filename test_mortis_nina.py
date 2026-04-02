"""
Test: store Mortis & 仁菜 as KB entries, run multiref workflow,
then deeply analyse the output using the Groq vision model.
Run manually: python test_mortis_nina.py
"""
import base64, io, json, os, sys, time, uuid
sys.path.insert(0, ".")

import requests
from PIL import Image
from groq import Groq

BASE  = os.getenv("COMFYUI_URL", "https://gina-fatherless-shavonda.ngrok-free.dev")
GGUF  = os.getenv("COMFYUI_GGUF", "flux-2-klein-4b-Q5_K_M.gguf")
VAE   = os.getenv("COMFYUI_VAE",  "flux2-vae.safetensors")
CLIP  = os.getenv("COMFYUI_CLIP", "qwen_3_4b.safetensors")

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

SCENE_PROMPT = (
    "Mortis and Nina standing side by side on a rooftop at sunset, "
    "full body view, anime illustration, dramatic sky background"
)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── helpers ──────────────────────────────────────────────────────────────────

def img_to_png_bytes(path: str) -> bytes:
    img = Image.open(path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    buf = io.BytesIO()
    bg.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()

def to_b64_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()

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

def poll_and_download(pid: str, out_path: str, timeout: int = 180) -> bool:
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

# ── vision analysis ──────────────────────────────────────────────────────────

def analyse_output(mortis_raw: bytes, nina_raw: bytes, output_raw: bytes) -> dict:
    """
    Send reference images + generated output to Groq vision model.
    Returns a dict: {mortis_score, nina_score, verdict, report}
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # --- Per-character analysis calls ---
    analyses = {}
    for char_name, ref_raw, expected in [
        ("Mortis", mortis_raw, MORTIS_APPEARANCE),
        ("Nina (仁菜)", nina_raw, NINA_APPEARANCE),
    ]:
        print(f"  Analysing {char_name}...")
        prompt = f"""You are a strict quality-control reviewer for an AI image generation system.

I will show you:
1. A REFERENCE IMAGE of the character "{char_name}"
2. A GENERATED SCENE that should contain this character alongside another character.

Your job: locate the character in the generated scene that most resembles "{char_name}" and score how well it matches the reference.

Expected visual features: {expected}

Score each feature 0–10 and give a total average score:
- Hair colour & style
- Eye colour
- Outfit / clothing match
- Accessories (hats, brooches, shoes, etc.)
- Overall likeness & distinctiveness (does it look like this specific character, not generic?)

After scoring, give a one-sentence verdict: PASS (≥7.0 average) or FAIL (<7.0).

Respond in this exact JSON format (no markdown fences):
{{
  "hair_score": <0-10>,
  "eyes_score": <0-10>,
  "outfit_score": <0-10>,
  "accessories_score": <0-10>,
  "likeness_score": <0-10>,
  "average": <float>,
  "verdict": "PASS" or "FAIL",
  "notes": "<one sentence>"
}}"""

        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",       "text": "REFERENCE IMAGE:"},
                    {"type": "image_url",  "image_url": {"url": to_b64_url(ref_raw)}},
                    {"type": "text",       "text": "GENERATED SCENE:"},
                    {"type": "image_url",  "image_url": {"url": to_b64_url(output_raw)}},
                    {"type": "text",       "text": prompt},
                ],
            }],
            temperature=0.1,
            max_tokens=400,
        )
        raw_text = resp.choices[0].message.content.strip()
        # strip markdown fences if model adds them
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        try:
            analyses[char_name] = json.loads(raw_text.strip())
        except json.JSONDecodeError:
            analyses[char_name] = {"average": 0.0, "verdict": "FAIL", "notes": raw_text[:200], "parse_error": True}

    # --- Distinctiveness check ---
    print("  Checking overall character distinctiveness...")
    dist_resp = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",       "text": "REFERENCE A — Mortis (gothic lolita, silver hair, red beret):"},
                {"type": "image_url",  "image_url": {"url": to_b64_url(mortis_raw)}},
                {"type": "text",       "text": "REFERENCE B — Nina 仁菜 (casual school girl, short brown hair):"},
                {"type": "image_url",  "image_url": {"url": to_b64_url(nina_raw)}},
                {"type": "text",       "text": "GENERATED SCENE (should contain both characters):"},
                {"type": "image_url",  "image_url": {"url": to_b64_url(output_raw)}},
                {"type": "text",       "text": (
                    "Are the two characters in the generated scene visually distinct from each other? "
                    "Do they look like two DIFFERENT people rather than near-clones? "
                    "Score the inter-character distinctiveness 0–10. "
                    "Respond in JSON (no fences): "
                    '{"distinctiveness_score": <0-10>, "verdict": "PASS" or "FAIL", "notes": "<one sentence>"}'
                )},
            ],
        }],
        temperature=0.1,
        max_tokens=150,
    )
    dist_text = dist_resp.choices[0].message.content.strip()
    if dist_text.startswith("```"):
        dist_text = dist_text.split("```")[1]
        if dist_text.startswith("json"):
            dist_text = dist_text[4:]
    try:
        dist = json.loads(dist_text.strip())
    except json.JSONDecodeError:
        dist = {"distinctiveness_score": 0, "verdict": "FAIL", "notes": dist_text[:200]}

    return {"per_character": analyses, "distinctiveness": dist}

# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  TEST: Mortis & 仁菜 — Multi-Character Generation")
print("═"*60)

# ── Step 1: Load images ───────────────────────────────────────────────────────
print("\n── Step 1: Load reference images ──")
mortis_raw = img_to_png_bytes(MORTIS_PATH)
nina_raw   = img_to_png_bytes(NINA_PATH)
print(f"  Mortis: {len(mortis_raw)//1024} KB   Nina: {len(nina_raw)//1024} KB")

# ── Step 2: KB entries ────────────────────────────────────────────────────────
print("\n── Step 2: Store KB image entries ──")
import database

mortis_id = database.add_image_entry(
    title="Mortis",
    image_bytes=mortis_raw,
    mime_type="image/png",
    appearance_description=MORTIS_APPEARANCE,
    display_description="Mortis — silver-haired gothic lolita with a red beret. Calm, mysterious demeanor.",
    tags="character,mortis,gothic",
)
nina_id = database.add_image_entry(
    title="仁菜",
    image_bytes=nina_raw,
    mime_type="image/png",
    appearance_description=NINA_APPEARANCE,
    display_description="Tokazaki Nina (仁菜) — energetic brown-haired girl with a casual street fashion style.",
    tags="character,nina,仁菜,casual",
)
print(f"  ✓ Mortis → KB #{mortis_id}   仁菜 → KB #{nina_id}")

# ── Step 3: Upload to ComfyUI ─────────────────────────────────────────────────
print("\n── Step 3: Upload reference images to ComfyUI ──")
mortis_cf = upload_to_comfyui(mortis_raw, "Mortis")
nina_cf   = upload_to_comfyui(nina_raw,   "Nina")

# ── Step 4: Build & submit workflow ──────────────────────────────────────────
print("\n── Step 4: Build & submit multiref workflow ──")
from workflow_adapter import build_multiref_workflow

wf = build_multiref_workflow(
    scene_prompt=SCENE_PROMPT,
    subject_filenames={"Mortis": [mortis_cf], "Nina": [nina_cf]},
    subject_appearances={"Mortis": MORTIS_APPEARANCE, "Nina": NINA_APPEARANCE},
    unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
    seed=int(uuid.uuid4().int % (2**31)),
    steps=4, width=512, height=768,
    contact_pose=False,
)
print(f"  Nodes: {len(wf)}")
clean = {nid: {k: v for k, v in n.items() if k != "_meta"} for nid, n in wf.items()}
r = requests.post(f"{BASE}/prompt", json={"prompt": clean, "client_id": str(uuid.uuid4())}, timeout=15)
if r.status_code != 200:
    print(f"  ✗ submit failed {r.status_code}: {r.text[:400]}"); sys.exit(1)
pid = r.json()["prompt_id"]
print(f"  ✓ submitted — prompt_id: {pid}")

# ── Step 5: Wait for output ───────────────────────────────────────────────────
print("\n── Step 5: Waiting for ComfyUI output... ──")
OUT = "attached_assets/_test_mortis_nina_output.png"
ok = poll_and_download(pid, OUT)
if not ok:
    print("✗ GENERATION FAILED"); sys.exit(1)

output_raw = open(OUT, "rb").read()

# ── Step 6: Vision analysis ───────────────────────────────────────────────────
print("\n── Step 6: Deep visual analysis via vision model ──")
analysis = analyse_output(mortis_raw, nina_raw, output_raw)

print("\n" + "─"*60)
print("  ANALYSIS REPORT")
print("─"*60)

all_pass = True
for char, scores in analysis["per_character"].items():
    avg = scores.get("average", 0.0)
    verd = scores.get("verdict", "FAIL")
    status = "✓ PASS" if verd == "PASS" else "✗ FAIL"
    if verd != "PASS":
        all_pass = False
    print(f"\n  [{status}] {char}  (avg {avg:.1f}/10)")
    for k in ("hair_score","eyes_score","outfit_score","accessories_score","likeness_score"):
        if k in scores:
            print(f"    {k:<22} {scores[k]:>4}/10")
    print(f"    Notes: {scores.get('notes','')}")

dist = analysis["distinctiveness"]
dist_verd = dist.get("verdict", "FAIL")
dist_score = dist.get("distinctiveness_score", 0)
dist_status = "✓ PASS" if dist_verd == "PASS" else "✗ FAIL"
if dist_verd != "PASS":
    all_pass = False
print(f"\n  [{dist_status}] Inter-character distinctiveness  ({dist_score}/10)")
print(f"    Notes: {dist.get('notes','')}")

print("\n" + "═"*60)
if all_pass:
    print("  OVERALL VERDICT: ✓ PASS — generation is satisfying")
else:
    print("  OVERALL VERDICT: ✗ FAIL — generation needs improvement")
print("═"*60 + "\n")

if not all_pass:
    sys.exit(1)
