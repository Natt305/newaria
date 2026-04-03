"""
Self-test: validates the full image-generation pipeline using real KB data.

Steps:
  1. Connect to ComfyUI (ngrok URL from env).
  2. Load image entries from the knowledge base (database.py).
  3. Fetch FULL-resolution PNG images via database.get_kb_image_full().
  4. Upload them to ComfyUI.
  5. Build build_multiref_workflow with the stored appearance_description per character.
  6. Submit, poll, download output.
  7. (Optional) Score output with Groq vision model.

Run:
    python test_self.py [--no-score] [--single] [--entry-ids A B]
"""
import argparse, base64, io, json, os, re, sys, time, uuid
sys.path.insert(0, ".")

import requests

import database
from workflow_adapter import build_multiref_workflow


def _validate_workflow_structure(wf: dict) -> list:
    """Check that every node reference [node_id, slot] points to an existing node."""
    errors = []
    node_ids = set(wf.keys())
    for nid, node in wf.items():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                ref_id = val[0]
                if ref_id not in node_ids:
                    errors.append(
                        f"Node {nid} ({node.get('class_type','?')}).inputs[{key}] "
                        f"→ '{ref_id}' does not exist"
                    )
    return errors

BASE  = os.environ.get("COMFYUI_URL",  "https://gina-fatherless-shavonda.ngrok-free.dev").rstrip("/")
GGUF  = os.environ.get("COMFYUI_GGUF", "flux-2-klein-4b-Q5_K_M.gguf")
VAE   = os.environ.get("COMFYUI_VAE",  "flux2-vae.safetensors")
CLIP  = os.environ.get("COMFYUI_CLIP", "qwen_3_4b.safetensors")
W     = int(os.environ.get("COMFYUI_WIDTH",  "512"))
H     = int(os.environ.get("COMFYUI_HEIGHT", "768"))
STEPS = int(os.environ.get("COMFYUI_STEPS",  "4"))

ap = argparse.ArgumentParser()
ap.add_argument("--no-score", action="store_true", help="Skip Groq scoring")
ap.add_argument("--single",   action="store_true", help="Test lone-character path (1 subject)")
ap.add_argument("--entry-ids", nargs="+", type=int, default=None,
                help="Force specific KB entry IDs (e.g. --entry-ids 1 2)")
args = ap.parse_args()

SEP = "═" * 62


# ── helpers ───────────────────────────────────────────────────────────────────

def to_b64(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode()

def upload(raw: bytes, label: str) -> str:
    r = requests.post(
        f"{BASE}/upload/image",
        files={"image": (f"{label}.png", io.BytesIO(raw), "image/png")},
        data={"overwrite": "true"},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    sub = d.get("subfolder", "")
    name = f"{sub}/{d['name']}" if sub else d["name"]
    print(f"  ✓ Uploaded {label!r} → {name}")
    return name

def submit_workflow(wf: dict) -> str:
    clean = {nid: {k: v for k, v in n.items() if k != "_meta"} for nid, n in wf.items()}
    r = requests.post(f"{BASE}/prompt",
                      json={"prompt": clean, "client_id": str(uuid.uuid4())},
                      timeout=15)
    if r.status_code != 200:
        try:
            err = r.json()
            top = err.get("error", {})
            print(f"  ✗ Submit error HTTP {r.status_code}: {top.get('type')} — {top.get('message')}")
            for nid, ne in err.get("node_errors", {}).items():
                ct = clean.get(nid, {}).get("class_type", nid)
                for e in ne.get("errors", []):
                    print(f"    Node {nid} ({ct}): [{e.get('type')}] {e.get('message')} — {e.get('details','')}")
        except Exception:
            print(f"  ✗ Submit error HTTP {r.status_code}: {r.text[:400]}")
        sys.exit(1)
    pid = r.json()["prompt_id"]
    print(f"  ✓ Submitted — prompt_id: {pid}")
    return pid

def poll_download(pid: str, out_path: str, timeout: int = 300) -> bytes:
    deadline = time.time() + timeout
    print(f"  Polling...", end="", flush=True)
    while time.time() < deadline:
        time.sleep(3)
        h = requests.get(f"{BASE}/history/{pid}", timeout=15)
        if h.status_code != 200 or pid not in h.json():
            print(".", end="", flush=True)
            continue
        j = h.json()[pid]
        if not j.get("status", {}).get("completed"):
            print(".", end="", flush=True)
            continue
        imgs = [i for o in j.get("outputs", {}).values() for i in o.get("images", [])]
        if not imgs:
            print(f"\n  ✗ Completed but no output images")
            sys.exit(1)
        ii = imgs[0]
        params = {"filename": ii["filename"], "type": ii.get("type", "output")}
        if ii.get("subfolder"):
            params["subfolder"] = ii["subfolder"]
        dl = requests.get(f"{BASE}/view", params=params, timeout=30)
        data = dl.content
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"\n  ✓ Saved → {out_path}")
        return data
    print(f"\n  ✗ Timed out after {timeout}s")
    sys.exit(1)

def score_character(out_bytes: bytes, ref_bytes: bytes | None,
                    char_name: str, appearance: str) -> dict:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return {}
    from groq import Groq
    client = Groq(api_key=groq_key)
    content: list = []
    if ref_bytes:
        content += [
            {"type": "text", "text": f"REFERENCE PHOTO — {char_name}:"},
            {"type": "image_url", "image_url": {"url": to_b64(ref_bytes)}},
        ]
    content += [
        {"type": "text", "text": (
            f"GENERATED IMAGE — should contain {char_name}: {appearance}\n"
            f"Score how well {char_name} in the generated image matches "
            f"the reference and description (0–10 each): "
            f"hair, eyes, outfit, accessories, overall likeness.\n"
            f"Reply JSON only (no fences): "
            f'{{\"hair\":N,\"eyes\":N,\"outfit\":N,\"accessories\":N,\"likeness\":N,\"notes\":\"...\"}}'
        )},
        {"type": "image_url", "image_url": {"url": to_b64(out_bytes)}},
    ]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": content}],
                max_tokens=300, temperature=0.1,
            )
            txt = resp.choices[0].message.content
            m = re.search(r'\{.*\}', txt, re.S)
            return json.loads(m.group()) if m else {"raw": txt}
        except Exception as e:
            if "tokens per day" in str(e) or "TPD" in str(e):
                print(f"  [score] Daily quota reached — skipping")
                return {"quota": True}
            time.sleep(2 ** attempt * 2)
    return {}

def avg(s: dict) -> float:
    keys = ["hair", "eyes", "outfit", "accessories", "likeness"]
    vals = [s[k] for k in keys if isinstance(s.get(k), (int, float))]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  AriaBot Self-Test: full KB → ComfyUI image generation pipeline")
print(SEP)
print(f"  ComfyUI : {BASE}")
print(f"  GGUF    : {GGUF}  steps={STEPS}  size={W}x{H}")

# ── Step 0: ComfyUI reachability ──────────────────────────────────────────────
print(f"\n[0] Checking ComfyUI...")
try:
    r = requests.get(f"{BASE}/system_stats", timeout=10)
    assert r.status_code == 200, f"HTTP {r.status_code}"
    vram = r.json().get("devices", [{}])[0].get("torch_vram_free", 0) // (1024*1024)
    print(f"  ✓ Online — VRAM free: {vram} MB")
except Exception as e:
    print(f"  ✗ Cannot reach ComfyUI: {e}")
    sys.exit(1)

# ── Step 1: Load KB entries ───────────────────────────────────────────────────
print(f"\n[1] Loading KB image entries...")
all_entries = database.get_image_entries()
entries_with_photos = [e for e in all_entries if database.get_entry_image_count(e.get("id", 0)) > 0]
print(f"  Found {len(all_entries)} KB image entries, {len(entries_with_photos)} with photos")

if args.entry_ids:
    selected = [e for e in entries_with_photos if e.get("id") in args.entry_ids]
    if not selected:
        print(f"  ✗ No entries found with IDs {args.entry_ids}")
        sys.exit(1)
elif args.single:
    selected = entries_with_photos[:1]
else:
    selected = entries_with_photos[:2]

if not selected:
    print("  ✗ No KB entries with photos found. Add some via /addimage first.")
    sys.exit(1)

n_use = 1 if args.single else min(2, len(selected))
selected = selected[:n_use]
for e in selected:
    print(f"  → #{e['id']} {e.get('title','?')!r}  ({database.get_entry_image_count(e['id'])} photo(s))")

# ── Step 2: Fetch full-resolution images ──────────────────────────────────────
print(f"\n[2] Fetching full-resolution images from DB...")
subject_raw: dict = {}    # {title: first_full_png_bytes}   — for scoring
subject_fnames_cf: dict = {}  # {title: [comfyui_filename,...]}  — for workflow

for e in selected:
    eid   = e["id"]
    title = e.get("title", f"entry_{eid}")
    count = database.get_entry_image_count(eid)
    loaded = []
    for idx in range(1, count + 1):
        result = database.get_kb_image_full(eid, idx)
        if result:
            img_bytes, _ = result
            if not subject_raw.get(title):
                subject_raw[title] = img_bytes
            loaded.append(img_bytes)
    if not loaded:
        print(f"  ✗ Could not load any images for {title!r} (id={eid})")
        sys.exit(1)
    print(f"  ✓ {title!r}: {len(loaded)} image(s) loaded (full-res PNG)")

    # Upload to ComfyUI
    cf_names = []
    for i, raw in enumerate(loaded[:3]):  # up to 3 per subject
        cf = upload(raw, f"{title}_{i+1}")
        cf_names.append(cf)
    subject_fnames_cf[title] = cf_names

# ── Step 3: Get appearance descriptions ───────────────────────────────────────
print(f"\n[3] Reading appearance descriptions...")
subject_appearances: dict = {}
for e in selected:
    title = e.get("title", "?")
    app = (e.get("appearance_description") or "").strip()
    if not app:
        app = (e.get("display_description") or "").strip()
    subject_appearances[title] = app
    preview = app[:80] + "..." if len(app) > 80 else app
    print(f"  {title!r}: {preview!r}")

# ── Step 4: Build workflow ────────────────────────────────────────────────────
names_str = " & ".join(subject_fnames_cf.keys())
if len(selected) == 1:
    title0 = selected[0].get("title", "character")
    SCENE = f"{title0} standing in a cozy library, full body view, anime illustration, warm lighting"
    contact = False
else:
    titles = [e.get("title","?") for e in selected]
    SCENE = (f"{titles[0]} and {titles[1]} standing side by side in a park at sunset, "
             f"full body view, anime illustration")
    contact = False

print(f"\n[4] Building multiref workflow for: {names_str}")
print(f"  Scene: {SCENE!r}")
print(f"  contact_pose={contact}")

seed = int(uuid.uuid4().int % (2**31))
wf = build_multiref_workflow(
    scene_prompt=SCENE,
    subject_filenames=subject_fnames_cf,
    subject_appearances=subject_appearances,
    unet_name=GGUF, vae_name=VAE, clip_name=CLIP,
    seed=seed, steps=STEPS, width=W, height=H,
    contact_pose=contact,
)
print(f"  ✓ Workflow built — {len(wf)} nodes, seed={seed}")

# ── Step 5: Structural validation ─────────────────────────────────────────────
print(f"\n[5] Structural validation...")
try:
    errs = _validate_workflow_structure(wf)
    if errs:
        print(f"  ✗ {len(errs)} broken node reference(s):")
        for er in errs:
            print(f"    {er}")
        sys.exit(1)
    print(f"  ✓ All node references valid")
except AttributeError:
    print("  (skipped — _validate_workflow_structure not exported)")

# ── Step 6: Submit ────────────────────────────────────────────────────────────
print(f"\n[6] Submitting to ComfyUI...")
pid = submit_workflow(wf)

# ── Step 7: Poll & download ───────────────────────────────────────────────────
print(f"\n[7] Waiting for output...")
out_path = "attached_assets/_self_test_output.png"
out_bytes = poll_download(pid, out_path)

# ── Step 8: Groq scoring ──────────────────────────────────────────────────────
if not args.no_score and os.environ.get("GROQ_API_KEY"):
    print(f"\n[8] Scoring with Groq vision model...")
    all_pass = True
    for e in selected:
        title = e.get("title", "?")
        app   = subject_appearances.get(title, "")
        ref_b = subject_raw.get(title)
        sc    = score_character(out_bytes, ref_b, title, app)
        if sc.get("quota"):
            print(f"  [{title}] Quota exceeded — skipping")
            continue
        if not sc or "raw" in sc:
            print(f"  [{title}] Scoring failed: {sc}")
            continue
        avg_sc = avg(sc)
        verdict = "✓ PASS" if avg_sc >= 7.0 else "✗ FAIL"
        if avg_sc < 7.0:
            all_pass = False
        print(f"  [{verdict}] {title}  avg={avg_sc}/10")
        print(f"    hair={sc.get('hair','?')}  eyes={sc.get('eyes','?')}  "
              f"outfit={sc.get('outfit','?')}  accessories={sc.get('accessories','?')}  "
              f"likeness={sc.get('likeness','?')}")
        print(f"    {sc.get('notes','')}")
    print(f"\n  Overall: {'✓ PASS' if all_pass else '✗ FAIL'}")
else:
    print(f"\n[8] Scoring skipped (--no-score or no GROQ_API_KEY)")

print(f"\n{SEP}")
print(f"  DONE — output saved to: {out_path}")
print(SEP + "\n")
