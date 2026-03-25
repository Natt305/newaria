"""
Groq AI client for fast text chat and image understanding.
Falls back to Cloudflare Workers AI image generation when the model's response
indicates it would generate an image but cannot.
"""
import os
import asyncio
import re
import base64
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from groq import AsyncGroq

# ── Per-day exhaustion tracking ───────────────────────────────────────────────
# Maps model_name -> Unix timestamp when it was marked exhausted.
# A model is considered exhausted until the next midnight UTC after that point.
_daily_exhausted: dict[str, float] = {}

# ── Refusal-detection for enhance_image_prompt ────────────────────────────────
# When vision models are unavailable and the fallback text model receives a
# request that depends on seeing images, it produces a polite refusal.  These
# phrases identify that case so we can discard the response and return the
# original prompt instead of sending a refusal as an image-generation prompt.
_REFUSAL_PHRASES: tuple[str, ...] = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "unable to view",
    "unable to see",
    "cannot view",
    "can't view",
    "without being able to view",
    "without being able to see",
    "without access to the",
    "no image was provided",
    "no reference image",
    "don't have access to",
    "do not have access to",
    "cannot access the image",
)

def _mark_daily_exhausted(model: str) -> None:
    """Record that this model hit its daily token limit."""
    _daily_exhausted[model] = time.time()
    print(f"[Groq] {model} hit daily token limit — skipping until midnight UTC.")

def _is_daily_exhausted(model: str) -> bool:
    """Return True if the model is still within its daily-exhaustion window."""
    ts = _daily_exhausted.get(model)
    if ts is None:
        return False
    exhausted_at = datetime.fromtimestamp(ts, tz=timezone.utc)
    next_midnight = (exhausted_at + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    if datetime.now(timezone.utc) >= next_midnight:
        del _daily_exhausted[model]   # expired — remove so it's tried again
        return False
    return True

def _is_daily_limit_error(err: str) -> bool:
    """Return True if the error string indicates a daily/total quota, not a per-minute spike."""
    markers = ("per_day", "per day", "daily", "tokens_per_day", "day limit", "24-hour")
    return any(m in err for m in markers)

_DEFAULT_CHAT_MODEL   = "llama-3.3-70b-versatile"
_DEFAULT_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def _default_model() -> str:
    """Return the configured chat model (GROQ_MODEL env var, else built-in default)."""
    return os.environ.get("GROQ_MODEL", "").strip() or _DEFAULT_CHAT_MODEL

def _default_vision_model() -> str:
    """Return the configured vision model (GROQ_VISION_MODEL env var, else built-in default)."""
    return os.environ.get("GROQ_VISION_MODEL", "").strip() or _DEFAULT_VISION_MODEL

# Resolved at import time so the module-level DEFAULT_MODEL stays compatible.
DEFAULT_MODEL = _default_model()

# ── Vision model candidates (tried in order until one succeeds) ───────────────
# Llama 4 Scout is the primary confirmed multimodal model on Groq.
# GROQ_VISION_MODEL is dynamically prepended at call time.
VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # 750 T/s, 20 MB, preview
    "llama-3.2-90b-vision-preview",               # legacy fallback
    "llama-3.2-11b-vision-preview",               # legacy fallback
]

# ── Text chat fallback chain (priority-ordered, highest quality first) ────────
# When the primary model (GROQ_MODEL) is rate-limited, each entry is tried
# in sequence.  Lower tiers trade quality for availability / speed.
#
#  Tier 1 — production, large  (~120–70 B params)
#    openai/gpt-oss-120b       500 T/s  131K ctx  production
#    llama-3.3-70b-versatile   280 T/s  131K ctx  production  ← default
#  Tier 2 — production, fast   (~20–32 B params)
#    qwen/qwen3-32b            400 T/s  131K ctx  preview
#    openai/gpt-oss-20b       1000 T/s  131K ctx  production
#  Tier 3 — extended context / compound
#    moonshotai/kimi-k2-instruct-0905  200 T/s  262K ctx  preview
#    groq/compound              450 T/s  131K ctx  production system
#    groq/compound-mini         450 T/s  131K ctx  production system
#  Tier 4 — fast last resort  (~8 B params)
#    llama-3.1-8b-instant      560 T/s  131K ctx  production
FALLBACK_MODELS = [
    # Tier 1
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    # Tier 2
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
    # Tier 3
    "moonshotai/kimi-k2-instruct-0905",
    "groq/compound",
    "groq/compound-mini",
    # Tier 4 removed — Ollama is used as the final fallback instead
]

IMAGE_TRIGGER_PHRASES = [
    # English declines
    "i can't generate",
    "i cannot generate",
    "i'm unable to generate",
    "i am unable to generate",
    "i don't have the ability to generate",
    "i do not have the ability to generate",
    "i can't create images",
    "i cannot create images",
    "i'm not able to generate",
    "i am not able to generate",
    "as a text-based",
    "as a language model",
    "i can't display",
    "i cannot display",
    "i can't produce images",
    "i cannot produce images",
    "i'm unable to create visual",
    "only generate text",
    "text-based ai",
    "text only",
    "no image generation",
    "can't render",
    "cannot render",
    # Chinese declines (Traditional / Simplified common forms)
    "文字ai",
    "文字 ai",
    "我是一個文字",
    "我是個文字",
    "只是一個文字",
    "只是個文字",
    "只能以文字",
    "只能用文字",
    "無法生成圖",
    "無法為您生成",
    "無法創作圖",
    "沒有辦法真的",
    "沒有生成圖像",
    "試著想像",
    "用文字描述",
    "以文字描述",
    "用文字表達",
    "以文字表達",
    "想像一下",
]

IMAGE_REQUEST_PATTERNS = [
    # English
    re.compile(r"\bgenerate\b.{0,30}\bimage\b", re.I),
    re.compile(r"\bcreate\b.{0,30}\bimage\b", re.I),
    re.compile(r"\bdraw\b", re.I),
    re.compile(r"\bpaint\b.{0,30}\bpicture\b", re.I),
    re.compile(r"\bmake\b.{0,30}\bimage\b", re.I),
    re.compile(r"\bshow\b.{0,30}\bpicture\b", re.I),
    re.compile(r"\billustrate\b", re.I),
    # Chinese explicit image requests (Traditional & Simplified)
    re.compile(r"(生成|畫|繪|製作|創作|做).{0,20}(圖|圖片|圖像|插圖|照片)", re.I),
    re.compile(r"(幫我|幫|請|可以|能不能).{0,10}(畫|生成|繪製|做).{0,20}(圖|圖片|圖像)", re.I),
    re.compile(r"(圖片|圖像|照片|插圖).{0,10}(生成|製作|創作)", re.I),
    # Chinese implicit visual requests ("I want to see you doing X")
    re.compile(r"想看.{0,40}(的樣子|樣子|彈|唱|跳|畫面|場景|你|妳)", re.I),
    re.compile(r"(讓我看|給我看|讓我瞧|讓我欣賞).{0,30}(樣子|彈|唱|跳|妳|你)", re.I),
    re.compile(r"(看看妳|看看你|看妳|看你).{0,30}(的樣子|樣子|彈|唱|跳)", re.I),
    re.compile(r"(我想看|我要看|我想瞧|我要瞧).{0,30}(妳|你)", re.I),
    re.compile(r"(妳|你)(彈|唱|跳|演奏|表演).{0,20}(的樣子|樣子|畫面)", re.I),
]

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_RE_UNCLOSED = re.compile(r"<think>.*", re.DOTALL)

_IMAGE_MARKER_RE = re.compile(
    r"\[(?:IMAGE|圖像生成|圖像|生成圖像|生成图像|图像生成|图像|GENERATE IMAGE|GEN IMAGE):\s*(.+?)\]",
    re.I | re.S,
)

_SELF_REF_RE = re.compile(
    r"\b(selfie|self.?portrait|photo of me|picture of me|my face|my photo|my picture|"
    r"what i look like|how i look|my appearance|my outfit|my hair|my eyes|"
    r"myself|me posing|me standing|me sitting|portrait of me|"
    r"show me you|show yourself|your photo|your picture|your selfie|your appearance|"
    r"photo of you|picture of you|what you look like|how you look)\b"
    r"|自拍|我的照片|我的樣子|我的臉|我的外貌|拍一張我|我的自拍"
    r"|妳的照片|你的照片|妳的樣子|你的樣子|妳的自拍|你的自拍|妳的外貌|你的外貌"
    r"|想看妳|想看你|看看妳|看看你|給我看妳|給我看你|讓我看妳|讓我看你"
    r"|妳的臉|你的臉|看妳的|看你的|妳照片|你照片",
    re.I,
)


def is_self_referential_image(prompt: str) -> bool:
    """Return True if an image prompt is about the bot's own appearance."""
    return bool(_SELF_REF_RE.search(prompt))


# ── Memory helpers ────────────────────────────────────────────────────────────

_RECALL_RE = re.compile(
    r"(你記得嗎|記得嗎|你還記得|還記得|妳記得嗎|妳還記得|"
    r"上次|之前|以前|我之前|我們之前|我說過|你說過|妳說過|"
    r"之前說|之前說過|你提到|妳提到|我提到|之前聊|上回|"
    r"幾天前|幾週前|幾個月前|好久以前|很久以前|"
    r"do you remember|remember when|remember that|remember what|"
    r"you mentioned|i mentioned|we talked about|you told me|i told you|"
    r"last time|earlier|before|a while ago|ages ago|long ago|"
    r"way back|back when|didn't (you|i|we)|don't you remember)",
    re.I,
)


def is_recall_request(text: str) -> bool:
    """Return True if the user's message contains recall/memory trigger phrases."""
    return bool(_RECALL_RE.search(text))


async def extract_memories(exchange: str, bot_name: str) -> list:
    """Extract 0–3 memorable facts from a conversation exchange using LLM.

    Returns a list of short summary strings (empty list on failure or nothing notable).
    The exchange should look like:
        'User (Name): <user_text>\\n<bot_name>: <bot_response>'
    """
    import json as _json

    if not exchange or len(exchange.strip()) < 30:
        return []

    client = _client()
    if not client:
        return []

    system = (
        f"You are a memory extractor for {bot_name}.\n"
        "Given a conversation exchange, extract 0 to 3 facts that are genuinely worth remembering long-term.\n"
        "Only extract personal, specific facts: names, preferences, jobs, hobbies, relationships, "
        "locations, events, goals, feelings, or important statements the user shared.\n"
        "Skip greetings, generic questions, image requests, and anything trivial or repetitive.\n"
        "Each fact MUST name the person and what they shared, written in Traditional Chinese.\n"
        "Keep each fact under 60 characters.\n"
        "Return ONLY a valid JSON array of strings. Return [] if nothing is worth remembering.\n"
        'Example: ["Alice 說她是一名遊戲設計師，喜歡貓咪", "Bob 提到他來自台灣，最喜歡的樂團是五月天"]'
    )

    messages = [{"role": "user", "content": f"Extract memorable facts:\n\n{exchange[:1500]}"}]

    try:
        text, *_ = await chat(messages, system_prompt=system, model=DEFAULT_MODEL)
        if not text:
            return []
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            arr = _json.loads(text[start:end])
            if isinstance(arr, list):
                return [str(s).strip() for s in arr[:3] if isinstance(s, str) and s.strip()]
    except Exception as e:
        print(f"[Memory] Extract error: {e}")

    return []


_groq_client: Optional[AsyncGroq] = None
_groq_client_key: str = ""


def _client() -> Optional[AsyncGroq]:
    global _groq_client, _groq_client_key
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None
    if _groq_client is None or key != _groq_client_key:
        _groq_client = AsyncGroq(api_key=key)
        _groq_client_key = key
    return _groq_client


async def understand_image(
    image_bytes: bytes,
    mime_type: str,
    question: str = "Describe this image in detail.",
) -> Optional[str]:
    """Analyze an image using Groq vision models.

    Tries each candidate in VISION_MODELS until one works.
    Returns the description string on success, or None on total failure.
    """
    client = _client()
    if not client:
        print("[Groq Vision] No API key — skipping image analysis")
        return None

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    configured_vision = _default_vision_model()
    vision_order = [configured_vision] + [m for m in VISION_MODELS if m != configured_vision]

    for model in vision_order:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                temperature=0.5,
                max_tokens=1024,
            )
            text = _THINK_RE.sub("", response.choices[0].message.content or "").strip()
            if text:
                print(f"[Groq Vision] Success with model: {model}")
                return text
            print(f"[Groq Vision] Empty response from {model}, trying next...")
        except Exception as e:
            err = str(e)
            print(f"[Groq Vision] {model} failed: {err}")
            # Always try the next candidate regardless of error type
            continue

    print("[Groq Vision] All vision model candidates exhausted — image analysis unavailable")
    return None


_NEGATION_RE = re.compile(
    r"無法|不能|沒辦法|做不到|辦不到|can'?t|cannot|unable|not able|no way",
    re.I,
)


def response_declines_image(text: str, messages: list | None = None) -> bool:
    """Return True if the reply says it can't generate an image.

    Two-pass check:
    1. Fast path — exact phrase list (language-specific known phrases).
    2. Context-gated heuristic — fires only when the user clearly wanted an image
       (messages is provided and user_wants_image() returns non-None) AND the
       reply contains any negation/inability word but no [IMAGE:] marker.
       This catches novel phrasings like "抱歉，我無法提供" that the phrase list misses.
    """
    lower = text.lower()
    if any(phrase in lower for phrase in IMAGE_TRIGGER_PHRASES):
        return True
    if messages is not None and not _IMAGE_MARKER_RE.search(text):
        if user_wants_image(messages) and _NEGATION_RE.search(text):
            return True
    return False


def user_wants_image(messages: list) -> Optional[str]:
    """
    If the most recent user message is an image request, extract and return
    a clean image prompt string. Otherwise return None.
    """
    for msg in reversed(messages):
        if msg["role"] == "user":
            content = msg["content"]
            for pat in IMAGE_REQUEST_PATTERNS:
                if pat.search(content):
                    cleaned = re.sub(
                        r"(?i)(please\s+)?(generate|create|draw|make|show|paint|illustrate)\s+(me\s+)?(an?\s+)?",
                        "", content,
                    ).strip()
                    return cleaned if cleaned else content
            return None
    return None


async def enhance_image_prompt(
    raw_prompt: str,
    character_context: str = "",
    subject_references: dict = None,
    reference_images: Optional[list] = None,
) -> str:
    """Translate and expand a raw (possibly Chinese) prompt into a rich English
    image-generation prompt suitable for Cloudflare Workers AI.

    character_context: text description of the character's appearance (written + photo-derived).
    subject_references: {name: description} for named KB subjects — used as ground truth.
    reference_images: list of (bytes, mime_type) thumbnail tuples. When provided, the
        rewrite call uses a vision model that can literally see the reference photos,
        which significantly improves accuracy for traits like eye color and hair tone.

    Returns the enhanced prompt, or the original if enhancement fails.
    """
    client = _client()
    if not client:
        return raw_prompt

    has_images = bool(reference_images)

    # If images were provided but ALL vision models are daily-exhausted, any text-model
    # fallback will refuse (it cannot see the photos).  When there is also no stored text
    # context to work from, skip the API call entirely and return the raw prompt unchanged
    # rather than risk sending an apology sentence to the image generator.
    if has_images and not character_context and not subject_references:
        configured_vision = _default_vision_model()
        all_vision = [configured_vision] + [m for m in VISION_MODELS if m != configured_vision]
        if all(_is_daily_exhausted(m) for m in all_vision):
            print("[Groq] All vision models exhausted and no text context — skipping enhancement, using raw prompt")
            return raw_prompt

    # Strip any ART STYLE section from stored character descriptions before using
    # them as ground truth — stored descriptions may say "ART STYLE: 3D rendered" etc.
    # which would override the fixed 2D anime target even inside VERIFIED GROUND TRUTH.
    _art_style_line_re = re.compile(r"(?m)^ART STYLE[:\s][^\n]*\n?", re.IGNORECASE)
    if character_context:
        character_context = _art_style_line_re.sub("", character_context)

    ref_block = ""
    if character_context and character_context.strip():
        if has_images:
            # Reference photos are attached — the vision model reads appearance
            # directly from them.  The stored text is a secondary fallback only.
            # Do NOT instruct the model to enumerate all traits; instead keep
            # appearance brief and let scene/action dominate the output.
            ref_block += (
                "\n[CHARACTER APPEARANCE — SECONDARY REFERENCE (photos are primary)]\n"
                "Reference photos are attached above. Read appearance from the photos, not this text.\n"
                "Use the text below ONLY as a fallback for details not clearly visible in the photos.\n"
                "Do NOT reproduce this block exhaustively — include just a brief identifying tag "
                "(character name + one or two signature visual traits, e.g. hair colour + eye colour). "
                "Spend the rest of the prompt on: scene, environment, lighting, mood, pose, "
                "and what the character(s) are doing or feeling.\n"
                "EXCEPTION: any ART STYLE line in this block is irrelevant — art style is always 2D anime.\n"
                f"{character_context.strip()}\n"
            )
        else:
            # No photos — text is the only appearance source.  Enumerate fully.
            ref_block += (
                "\n[CHARACTER APPEARANCE — VERIFIED GROUND TRUTH]\n"
                "The following contains confirmed, factual appearance details for the character.\n"
                "Sections marked HIGHEST PRIORITY must be followed exactly above all else.\n"
                "You MUST use ALL of these physical traits in your output. Do NOT invent, alter, or substitute any of them.\n"
                "Eye color, hair color, hairstyle, skin tone, AND OUTFIT described here are FINAL — "
                "they override anything conflicting in the raw prompt or your own assumptions. "
                "If an outfit is described here, reproduce it in full detail in the output. "
                "Do NOT summarize, omit, or replace any garment piece listed.\n"
                "EXCEPTION: any ART STYLE line in this block is irrelevant — the art style is always fixed as 2D anime.\n"
                f"{character_context.strip()}\n"
            )
    if subject_references:
        for name, desc in subject_references.items():
            ref_block += (
                f"\n[SUBJECT REFERENCE — {name.upper()} — AUTHORITATIVE TEXT — HIGHEST PRIORITY]\n"
                f"The following are the confirmed, verified appearance details for '{name}'. "
                f"This text is the ABSOLUTE AUTHORITY for every physical trait listed. "
                f"It overrides reference photos, raw prompt words, and any other source. "
                f"You MUST reproduce every trait here exactly. "
                f"Do NOT invent, alter, or omit any detail:\n"
                f"{desc.strip()}\n"
                f"\nCRITICAL — if reference photos are also attached, they do NOT override this text. "
                f"Photos may only be used to understand pose, setting, or details NOT covered by this text. "
                f"Every color, garment piece, and physical trait named above supersedes what any photo appears to show.\n"
            )

    # When photos are present, priority order is:
    # [SUBJECT REFERENCE] text > [CHARACTER APPEARANCE] text > photos > raw prompt words.
    # Photos fill in details not covered by the stored text descriptions.
    if has_images:
        image_note = (
            "\n[PRIORITY RULING — READ BEFORE EVERYTHING ELSE — MANDATORY OVERRIDES]\n"
            "Reference photos are attached. Priority order (highest first):\n"
            "  1. [SUBJECT REFERENCE] text blocks — absolute authority, override everything\n"
            "  2. [CHARACTER APPEARANCE] text blocks — authoritative, override photos\n"
            "  3. Reference photos — used only for details NOT covered by the text blocks above\n"
            "  4. Raw prompt appearance words — DISCARD (written without photos, unreliable)\n"
            "\n"
            "OVERRIDE 1 — COLORS: discard raw prompt colors, but trust [SUBJECT REFERENCE] text.\n"
            "The raw image request was written WITHOUT access to reference photos or verified data. "
            "Every color word in the raw prompt (hair color, eye color, skin tone) is a guess — DISCARD them. "
            "HOWEVER: any color stated in a [SUBJECT REFERENCE] or [CHARACTER APPEARANCE] block above "
            "is verified and correct — use it exactly as written. "
            "Only fall back to reading colors from the reference photos for traits NOT covered by those text blocks.\n"
            "\n"
            "OVERRIDE 2 — OUTFIT: if a [SUBJECT REFERENCE] block lists the outfit, reproduce it exactly. "
            "Only reconstruct outfit from photos when no text description is present.\n"
            "\n"
            "OVERRIDE 3 — ART STYLE: FIXED — DO NOT DERIVE FROM REFERENCE PHOTOS.\n"
            "The target art style is ALWAYS: "
            "'clean 2D anime illustration, flat cel-shaded coloring, "
            "soft gradient shading, vivid saturated color palette, anime digital art'.\n"
            "Reference photos are used ONLY to extract character appearance (hair, eyes, outfit). "
            "NEVER use the reference photo's rendering style (3D, photorealistic, semi-realistic, CGI, etc.) "
            "as the basis for the art style description. "
            "Even if the reference looks like a 3D model, game render, or photograph, "
            "the output prompt must always describe clean 2D anime illustration style.\n"
            "\n"
            "STRUCTURE (hairstyle shape, bang style, character identity) — use the text descriptions as the authority.\n"
        )
    else:
        image_note = ""

    # Multi-character composition and interaction block.
    # Activated when >= 2 reference images are provided OR when the raw prompt
    # already contains spatial position terms injected by _build_spatial_prefix
    # (e.g. "(left) Aria, (right) Mortis").
    _spatial_terms = ("(left)", "(right)", "(center)", "(far-left)", "(far-right)",
                      "(center-left)", "(center-right)")
    _has_spatial_terms = any(t in raw_prompt for t in _spatial_terms)
    n_subjects_hint = max(
        len(reference_images) if reference_images else 0,
        len(subject_references) if subject_references else 0,
        2 if _has_spatial_terms else 0,
    )
    if n_subjects_hint >= 2:
        _raw_lower = raw_prompt.lower()
        _interaction_verbs = (
            "together", "hug", "hugging", "fight", "fighting", "battle", "sparring",
            "talk", "talking", "chat", "chatting", "play", "playing", "dance", "dancing",
            "hold hands", "holding hands", "interact", "interacting", "alongside",
            "side by side", "with each other", "facing each other", "looking at each other",
            "sitting with", "standing with", "walk with", "walking with",
            "lean", "leaning", "embrace", "embracing", "wrestle", "wrestling",
            "一起", "抱", "打架", "說話", "跳舞", "握手", "互動", "並肩",
        )
        _has_interaction = any(v in _raw_lower for v in _interaction_verbs)
        multi_char_note = (
            f"\n[MULTI-CHARACTER SCENE — {n_subjects_hint} SUBJECTS — MANDATORY RULES]\n"
            "This image contains MULTIPLE characters. Apply ALL rules below without exception:\n"
            "\n"
            "RULE 1 — COMPOSITION (NON-NEGOTIABLE): All characters must appear FULLY VISIBLE "
            "inside the frame. No character may be cut off at any edge. "
            "Use a centered composition — all subjects grouped near the center of the image, "
            "close enough together that both fit within a portrait-format frame without cropping. "
            "Use phrasing such as 'centered composition', 'both characters fully visible in frame', "
            "'full-body view of all characters', 'close together in the center of the scene'.\n"
            "\n"
            "RULE 2 — PROXIMITY: Characters must be near each other — not pushed to opposite far "
            "edges with empty space between them. They share the same immediate space.\n"
        )
        if _has_interaction:
            multi_char_note += (
                "\n"
                "RULE 3 — INTERACTION (MANDATORY — user's request implies shared action): "
                "The characters are actively engaged with each other, not merely standing side by side. "
                "You MUST explicitly describe the physical interaction or shared activity: "
                "body language, eye contact, gesture, touch, or shared motion showing they are "
                "reacting to and engaging with each other. "
                "Examples: 'leaning toward each other', 'arms around each other's shoulders', "
                "'locked in combat with swords clashing', 'mid-conversation, one gesturing toward the other', "
                "'side by side with intertwined arms'. "
                "Be specific and vivid — do NOT write that they are simply 'standing together'.\n"
            )
        if has_images:
            multi_char_note += (
                "\n"
                f"RULE — IDENTIFY EACH CHARACTER BRIEFLY: There are {n_subjects_hint} characters. "
                "For each, include only a short identifying tag (name + one or two visual identifiers "
                "read from the reference photos — e.g. hair colour, hat, or signature outfit piece). "
                "Do NOT exhaustively list every appearance detail. "
                "Reserve the majority of the prompt for: scene setting, environment, "
                "lighting, mood, pose, and what the characters are doing or feeling together.\n"
            )
        else:
            multi_char_note += (
                "\n"
                f"RULE — DESCRIBE EACH CHARACTER SEPARATELY: There are {n_subjects_hint} characters. "
                "Give each their own appearance block — hair, eyes, outfit — labelled clearly so the "
                "image model knows which traits belong to which subject.\n"
            )
    else:
        multi_char_note = ""

    system = (
        "You are an expert image-prompt writer for AI image generators.\n"
        "Given a user's image request (which may be in Chinese or English), "
        "rewrite it as a single, rich English prompt for an AI image model.\n"
        f"{image_note}"
        f"{multi_char_note}"
        f"{ref_block}"
        "Rules:\n"
        "- Output ONLY the prompt text — no intro, no quotes, no explanation.\n"
        "- Always write in English.\n"
        "- Aim for 150-300 words.\n"
        "- Be specific: include subject, art style, lighting, colors, mood, and setting.\n"
        "- Do NOT start with 'Generate', 'Create', 'Draw', 'An image of', etc.\n"
        "- Physical details from references (hair color, eye color, hairstyle, skin tone) "
        "ALWAYS take priority over anything in the raw prompt. Incorporate them naturally.\n"
        "- PROPS / INSTRUMENTS / HELD OBJECTS are MANDATORY: any object the character is "
        "holding, playing, or interacting with (guitar, microphone, sword, cup, book, etc.) "
        "MUST appear in the output. These come from the request text and are never overridden "
        "by appearance references. If the request says 'playing guitar', the guitar MUST be in the prompt.\n"
        "- ART STYLE (mandatory, always fixed — never derived from reference photos): "
        "The art style is ALWAYS 'clean 2D anime illustration' regardless of what the reference looks like. "
        "Reference photos may be 3D models, photographs, game renders — ignore their rendering style entirely. "
        "Always output this fixed style description, expanded across all axes:\n"
        "  * RENDERING TYPE: clean cell-shaded 2D anime illustration\n"
        "  * LINE ART: soft anti-aliased lines\n"
        "  * SHADING STYLE: flat cel-shaded with soft gradient shading\n"
        "  * COLOR PALETTE: vivid saturated color palette\n"
        "  * STYLE MARKERS: anime digital art, 2D illustration\n"
        "NEVER write 'semi-realistic', '3D rendered', 'photorealistic', 'CGI', or any rendering style "
        "derived from the reference photo — these will cause Flux to generate a photorealistic result.\n"
        "- BANG ASYMMETRY (MANDATORY): if the reference or character context shows a gap or "
        "parting in the bangs on one specific side, you MUST include it — it is not optional. "
        "State the side explicitly: e.g. 'straight-across bangs with a small gap on the right side'. "
        "Never write just 'straight-across bangs' if a gap is visible — that silently drops the detail.\n"
        "- HAIRSTYLE is critical — describe it precisely using ALL of the following dimensions:\n"
        "  * bang style: blunt/straight-across, asymmetric with a gap/part on one side, "
        "side-swept, parted center, no bangs, etc. — be specific about asymmetry if present\n"
        "  * whether hair is completely loose and flowing, or tied/braided/pinned\n"
        "  * presence or ABSENCE of an ahoge (antenna hair) — if not visible in reference, "
        "explicitly write 'no ahoge' in the prompt so the model does not add one\n"
        "  * side-framing strands — note if longer strands frame the sides of the face\n"
        "  * exact length: very long past waist, shoulder-length, short bob, etc.\n"
        "- Do NOT invent or add hairstyle elements (ahoge, twin-tails, clips, ribbons) "
        "that are not visible in the reference. Only describe what you actually see.\n"
        "- DISCARD raw prompt appearance words — ZERO TOLERANCE RULE: "
        "The raw prompt's color words (hair color, eye color, skin tone) are FABRICATIONS — written without photos. "
        "They MUST NOT appear anywhere in your output. This is non-negotiable. "
        "Do a mental check before writing: if a color word came from the raw prompt and not from the reference photo, DELETE IT. "
        "Example failure: raw prompt says 'silver hair, pink eyes' → your output contains 'silver hair' or 'pink eyes' → WRONG. "
        "Example success: raw prompt says 'silver hair, pink eyes' → you look at the photo → photo shows pale mint-green hair and warm amber eyes → you write 'pale mint-green hair' and 'warm amber-brown eyes'. "
        "Rewrite ALL physical appearance from scratch using ONLY the reference photos and character context. "
        "The final output must contain exactly ONE description of each trait — yours from the photos, not the raw prompt's guess.\n"
        "- HAIR COLOR (FLUX-CRITICAL — read every word):\n"
        "  MANDATORY ORDERING RULE: the primary COLOR HUE must come FIRST. "
        "NEVER lead with a lightness/achromatic word like 'near-white', 'almost-white', 'off-white', 'pale', or 'silver'. "
        "These words make Flux render grey/silver hair and IGNORE the actual tint. "
        "CORRECT pattern: '[color] hair, extremely desaturated / very pale, NOT silver, NOT grey' "
        "e.g. 'very pale seafoam mint-green hair — NOT silver, NOT grey, NOT white — extremely low saturation but clearly green-tinted'. "
        "WRONG pattern: 'near-white hair with a barely-there mint tint' — Flux sees 'near-white' and renders silver. "
        "ANTI-SILVER TRAP: any description that starts with 'near-white', 'almost-white', 'silver', or 'pale grey' "
        "will cause Flux to render achromatic silver/grey hair. PUT THE COLOR FIRST. "
        "When saturation is extremely low (hair looks almost white), describe it as: "
        "'[color] hair, almost achromatic but still distinctly [color]-tinted, NOT silver, NOT grey, NOT pure white'. "
        "The negation 'NOT silver, NOT grey' is MANDATORY whenever the hair has a tint. "
        "SATURATION MUST BE EXPLICIT: never omit 'extremely desaturated' / 'very low saturation' when applicable — "
        "Flux will default to vivid saturated color if saturation is not stated. "
        "Example — correct output for pale mint-green hair: "
        "'very pale seafoam mint-green hair (extremely desaturated, almost achromatic, "
        "but unmistakably cool-green-tinted — NOT silver, NOT grey, NOT white)'\n"
        "- EYE COLOR (three-axis required): describe eye color using ALL THREE axes — "
        "never collapse into a single color name:\n"
        "  * HUE FAMILY: e.g. 'warm amber-yellow', 'gray-green', 'khaki', 'warm olive-brown'\n"
        "  * SATURATION: read from the photo and state exactly — 'vivid' / 'high saturation' if the "
        "eyes are bright and saturated; 'muted' / 'very low saturation' if they are dull or washed out. "
        "NEVER assume muted — state what you actually see.\n"
        "  * BRIGHTNESS/CONTRAST: e.g. 'medium brightness', 'slightly dark', 'light'\n"
        "  Example (vivid amber): 'vivid warm amber-yellow eyes, high saturation, medium brightness'\n"
        "  Example (muted amber): 'low-saturation warm amber-yellow eyes, muted and soft, medium brightness'\n"
        "  Example (muted olive): 'very low-saturation muted olive-brown eyes with yellowish-green undertone, soft, medium brightness'\n"
        "  FORBIDDEN as bare single words (without saturation qualifier): 'amber eyes', 'golden eyes', "
        "'honey eyes' — always pair hue with explicit saturation level. "
        "NEVER write a shorthand color first and then the three-axis version — write ONLY the three-axis version.\n"
        "- EYELASHES (three-axis required): describe lash appearance using ALL THREE axes:\n"
        "  * HUE: read the actual color from the reference — it can be warm OR cool. "
        "e.g. 'warm gray-brown', 'pale brown', 'warm beige', 'cool greenish-grey', 'desaturated olive-grey', "
        "'grey-green with brown undertone', 'muted sage-grey'. "
        "IMPORTANT: eyelash color is often influenced by hair color undertones — "
        "a mint-haired character may have cool greenish-grey lashes, not warm brown ones. "
        "Read the reference and choose the hue that matches what you see.\n"
        "  * DARKNESS/CONTRAST: e.g. 'very low contrast against skin', 'soft', 'not dark', 'moderate contrast'. "
        "CRITICAL: Flux AI defaults to near-black lashes. If lashes in the reference are NOT near-black, "
        "you MUST explicitly state they are NOT dark — e.g. 'lashes are NOT black or dark brown, "
        "they are a soft [hue] with moderate/low contrast'. This override is mandatory.\n"
        "  * WEIGHT: e.g. 'fine and light', 'delicate', 'subtle'\n"
        "  Example (cool-toned): 'cool greenish-grey lashes, NOT black or dark brown, moderate contrast, fine and delicate'\n"
        "  Example (warm-toned): 'pale warm-brown lashes, very low contrast, fine and delicate'\n"
        "  NOT: just 'light lashes' or 'gray lashes' — still too vague, Flux will render dark\n"
        "- COMPLEXION/SKIN TONE: describe exact shade, e.g. 'very pale porcelain skin', "
        "'fair skin with a cool undertone', 'light warm ivory skin'.\n"
        "- OUTFIT (mandatory, extremely detailed, piece-by-piece reconstruction): "
        "List EVERY individual garment piece and accessory as its own entry. "
        "NEVER merge multiple pieces into one phrase. "
        "LAYERING ORDER — always list from outermost to innermost. "
        "For outer garments that drape over inner ones, explicitly state the relationship "
        "e.g. 'worn over the inner blouse', 'draping over the shoulders and chest'. "
        "This is critical — if the outer layer is omitted or merged with the inner layer, the visual will be wrong.\n"
        "GARMENT VOCABULARY — use precise type names. Critical distinctions:\n"
        "  • CAPE / MANTLE / CLOAK = open-front outer garment that drapes from the shoulders, "
        "NO sleeves, hangs freely. Key visual: if you see a wide circle-cut or flared black fabric "
        "that hangs from the shoulders and opens at the front (like a circular cloak), it is a CAPE, "
        "NOT a coat or jacket. Coats have sleeves. Describe as: "
        "'open-front black shoulder cape; wide circle-cut, [trim details], draping to [length]'.\n"
        "  • SHOULDER COVERS / EPAULETTES = the structured black padded sections at the very top of "
        "the shoulders where the cape attaches. They look like built-in shoulder pads or caps. "
        "Describe as a SEPARATE entry: 'charcoal-black shoulder epaulettes; structured, gold-decorated, cape-attachment'. "
        "Do NOT omit them or merge them into the cape entry.\n"
        "  • JABOT = a SMALL decorative ruffled chest piece worn at the sternum only. "
        "SIZE IS CRITICAL: palm-sized, sternum level only, does NOT extend below the chest, NOT a skirt or bib. "
        "Secured with a brooch or gemstone pin. "
        "Describe as: 'off-white ruffled jabot; sternum-level, palm-sized, NOT below chest, [color]-gemstone brooch'. "
        "NEVER call it a 'collar' or 'detachable collar'. "
        "NEVER describe it as reaching the waist or thighs.\n"
        "  • CAPE + ATTACHED SKIRT: A black outer garment may have BOTH a shoulder-cape section (upper) "
        "AND a wide A-line/circle-cut flared skirt below the waist (same fabric, same trim continuing along the hem). "
        "The SKIRT portion MUST be listed as a SEPARATE entry — do NOT fold it into the cape entry or omit it. "
        "Describe as: 'black circle-cut overskirt; [trim details e.g. cream-dotted hem trim], draping to [length]'.\n"
        "  • RED STRUCTURED COLLAR / COLLAR-CAPE: If deep red/crimson structured panels or wing-shaped collar pieces "
        "frame the neckline (visible at the inner opening of the black cape, sitting as rigid V-shaped or swept-back panels), "
        "this is a RED COLLAR-CAPE or STRUCTURED COLLAR — list as a SEPARATE entry: "
        "'deep-crimson structured collar-cape; V-shaped rigid panels, neckline framing'. "
        "NEVER omit or merge into the black outer cape.\n"
        "  • SHORTS vs VEST — KEY DISTINCTION: "
        "A VEST covers only the torso, has NO waistband, and ends at the natural waist. "
        "HIGH-WAIST SHORTS have a defined waistband at the natural waist AND a short leg portion below. "
        "If a red/crimson garment has buttons on the torso AND a visible waistband with short leg below → HIGH-WAIST SHORTS. "
        "NEVER call high-waist shorts a 'vest' even if buttons are visible on the torso portion.\n"
        "  • TIGHTS / HOSIERY RULE: If a form-fitting knit or sheer layer covers the legs from upper thigh "
        "to ankle (no visible waistband, no crotch seam, hugs the leg contour) → these are TIGHTS or OPAQUE TIGHTS, "
        "NOT trousers or pants. Describe as: 'dark grey opaque tights'. "
        "NEVER call tights 'trousers', 'pants', or 'leggings' unless the reference clearly shows "
        "a structured waistband and loose/straight leg silhouette.\n"
        "  • BROOCH RULE: A gemstone pin/brooch is ALWAYS described as part of the garment it fastens. "
        "If it fastens a white ruffled jabot, include it in the jabot entry's details. "
        "NEVER attribute a green gemstone brooch to a red or black garment.\n"
        "For EACH piece you MUST include ALL of the following — skipping any makes the description incomplete:\n"
        "  1. EXACT COLOR(S): precise shade names — 'deep crimson', 'off-white', 'charcoal-black', 'navy-midnight-blue'\n"
        "  2. GARMENT TYPE: exact name from the vocabulary above\n"
        "  3. KEY DETAILS: every visually important feature — collar shape, trim color, button style/count, embroidery, ruffles, cutouts, lace, hardware, brooch color. "
        "Use hyphenated compound terms ('gold-braid', 'dome-button', 'double-row'). Be specific, not vague.\n"
        "Format per piece: '[exact-color] [type]; [detail, detail, detail]' — join pieces with ' + '.\n"
        "Include ALL accessories: hats, belts, bags, jewelry, gloves, wrist cuffs, socks, tights, shoes — each as its own entry.\n"
        "BAD: 'black jacket' — NO color precision, NO details\n"
        "BAD: 'long black coat' for an open-front cape — use 'open-front black shoulder cape; [details]'\n"
        "BAD: 'white collar with brooch' for a jabot — use 'off-white ruffled jabot; sternum-level, palm-sized, green-gemstone brooch'\n"
        "GOOD: 'charcoal-black blazer; high-collar, gold-braid lapels, dome-button front, structured shoulders'\n"
        "NEVER write vague single-phrase outfit summaries like 'gothic uniform', 'casual outfit', 'school uniform', "
        "'military-style outfit', or 'black and red outfit' — these are automatic failures.\n"
        "⚠️ FORMAT-ONLY EXAMPLE — THIS IS A HYPOTHETICAL DIFFERENT CHARACTER. "
        "DO NOT COPY THIS OUTFIT INTO YOUR OUTPUT. "
        "YOUR OUTPUT MUST DESCRIBE WHAT YOU ACTUALLY SEE IN THE REFERENCE PHOTOS, NOT THIS EXAMPLE.\n"
        "Format example: short dark auburn wavy bob, slight inward curl at ends, no bangs, no ahoge, "
        "warm hazel eyes with greenish undertone, warm medium saturation, soft brightness, "
        "warm light-brown lashes, low contrast, fine and delicate, "
        "light warm ivory skin with golden undertone, "
        "outfit: dusty-rose turtleneck sweater; oversized, ribbed cuffs and hem + "
        "cream trousers; wide-leg, high-waist, pleated-front + "
        "white loafers; moc-toe stitching, tan sole, no hardware, "
        "anime art style\n"
        "END FORMAT-ONLY EXAMPLE — your output must describe the character in the reference photos, NOT the example above.\n"
        "CRITICAL REMINDER — OUTFIT: every piece must have exact color, garment type, and specific key details. "
        "EVERY garment and accessory must be its own entry joined by ' + '. "
        "If any piece is missing color precision or key details, your output is WRONG. "
        "If you collapse the entire outfit into fewer than 4 separate entries, your output is WRONG.\n"
        "MANDATORY PRE-SUBMISSION CHECKLIST — before writing your final output, verify ALL of these are present:\n"
        "  ✓ HAIR COLOR — exact hue name FIRST (not 'near-white', not 'silver'), "
        "then saturation level, then 'NOT silver, NOT grey' negation if tinted. "
        "ANTI-SILVER CHECK: scan your output — if 'silver', 'grey hair', 'near-white', or 'almost white' "
        "appears BEFORE the actual color name, rewrite it. Color hue always leads.\n"
        "  ✓ EYE COLOR — hue family + saturation + brightness (three axes). NEVER write 'amber', 'gold', "
        "'honey', or 'warm amber' as bare single words — always pair with explicit saturation. "
        "Use 'vivid / high saturation' if eyes are bright; 'muted / very low-saturation' if dull. "
        "Example (vivid amber): 'vivid warm amber-yellow eyes, high saturation, medium brightness'. "
        "Example (muted amber): 'low-saturation warm amber-yellow eyes, muted and soft, medium brightness'. "
        "Example (muted olive): 'very low-saturation muted olive-brown eyes with yellowish-green undertone, soft'.\n"
        "  ✓ EYELASHES — hue (cool or warm) + darkness override (state explicitly if NOT near-black) + weight. "
        "This field is MANDATORY. An output that does not contain eyelash description is INCOMPLETE.\n"
        "  ✓ SKIN TONE — precise shade\n"
        "  ✓ OUTFIT — every piece has exact color, garment type, and specific key details. "
        "If the reference shows a white ruffled chest piece with a gemstone brooch, "
        "it MUST appear as 'off-white ruffled jabot; sternum-level, palm-sized, NOT below chest, green-gemstone brooch'. "
        "If the reference shows a wide flared fabric from the shoulders (no sleeves), "
        "it MUST appear as 'open-front [color] shoulder cape; [details]'. "
        "Red shorts with gold buttons MUST appear as a separate lower-body entry visible below the jabot. "
        "These cannot be omitted or merged into other garments.\n"
        "  ✓ ART STYLE — must be 'clean 2D anime illustration, flat cel-shaded, anime digital art' (fixed, never from photos)\n"
        "  ✓ SCENE/POSE/SETTING — extracted from the request text\n"
        "  ✓ PROPS/INSTRUMENTS — any object mentioned in the request (guitar, microphone, etc.) must be present\n"
        "If any item from this checklist is missing from your output, add it before finalizing.\n"
    )

    if has_images:
        # Split the instruction into two explicit source directives:
        # 1. From the text → scene/pose/setting only (appearance words are fabrications)
        # 2. From the reference photos → ALL appearance (must be present in the output)
        # This prevents the model from interpreting "don't copy appearance from text"
        # as "omit appearance entirely" — a common failure mode.
        user_content = (
            "Image request (reference photos attached above).\n\n"
            "Your final output must be a single unified paragraph — no section headers, no labels, no bullet points.\n\n"
            "Instructions for building the output:\n"
            "  (1) SCENE/POSE/SETTING/PROPS — read from the text below. "
            "Extract everything that is NOT physical appearance: scene, pose, action, setting, mood, "
            "background, lighting, AND any objects, props, or instruments being held or used "
            "(e.g. guitar, microphone, book, cup, sword, flower — whatever the character is holding or interacting with). "
            "Props and instruments are SCENE elements, not appearance. "
            "They come from the text and MUST appear in your output — never omit them. "
            "DISCARD ONLY: hair color, eye color, skin tone, and clothing/outfit descriptions — "
            "those were written without photos and are fabrications. "
            "Everything else (action, props, instruments, location) must be preserved.\n"
            "  (2) APPEARANCE — read from the reference photos. "
            "Your output MUST include ALL of the following — any omission makes the output wrong:\n"
            "    • HAIR: exact shade (e.g. 'pale seafoam mint-green, low saturation'), hairstyle, length\n"
            "    • EYES: write ONLY the three-axis form: '[saturation] [hue] eyes, [saturation], [brightness]'. "
            "Read saturation from the photo — use 'vivid / high saturation' if eyes are bright, "
            "'muted / very low saturation' if dull. NEVER assume muted. "
            "FORBIDDEN as bare words without saturation: 'amber', 'golden', 'honey' alone — "
            "always pair hue with explicit saturation. "
            "Examples: 'vivid warm amber-yellow eyes, high saturation' OR 'low-saturation warm amber-yellow eyes, muted and soft'.\n"
            "    • EYELASHES: this field is MANDATORY and must be present. "
            "Write: '[hue], [darkness statement — state NOT black/dark brown if that is the case], [weight]'. "
            "Example: 'cool greenish-grey lashes, NOT black or dark brown, moderate contrast, fine and delicate'.\n"
            "    • SKIN TONE: exact shade\n"
            "    • OUTFIT: every garment piece as '[exact-color] [type]; [detail, detail, ...]', joined by ' + '. "
            "When reading the photo, actively LOOK FOR and name these elements if present:\n"
            "      — Wide flared/circle-cut fabric hanging from the shoulders WITHOUT sleeves "
            "(not a coat — no sleeves, drapes freely) → this is a SHOULDER CAPE. "
            "Describe as: 'open-front [color] shoulder cape; wide circle-cut, [trim details], draping to [length]'.\n"
            "      — Structured padded black shoulder caps at the shoulder attachment point of the cape "
            "→ these are SHOULDER EPAULETTES. List as a separate entry.\n"
            "      — White ruffled V-shaped or waterfall fabric at the chest/sternum level "
            "(NOT at the neck) with a gemstone pin at its center → this is a JABOT. "
            "Describe as: 'off-white ruffled jabot; sternum-level, V-shaped, green-gemstone brooch'.\n"
            "      — Black outer garment with BOTH a shoulder-cape section AND a wide A-line flared skirt below the waist "
            "→ the SKIRT is a SEPARATE entry 'black circle-cut overskirt; cream-dotted hem trim, draping to mid-thigh'. "
            "Do NOT omit the skirt or fold it into the cape entry.\n"
            "      — Deep red/crimson rigid V-shaped or wing-like panels at the neckline inside the black cape opening "
            "→ RED COLLAR-CAPE or STRUCTURED COLLAR. List as a SEPARATE entry: "
            "'deep-crimson structured collar-cape; V-shaped rigid panels, neckline framing'.\n"
            "      — Red/crimson garment at the torso with a defined waistband AND short leg visible below waist "
            "→ HIGH-WAIST SHORTS, NOT a vest. Describe as: 'deep-crimson high-waist shorts; double-row gold buttons'.\n"
            "      — Form-fitting leg layer from thigh to ankle with no visible waistband or crotch seam "
            "→ TIGHTS or OPAQUE TIGHTS, NOT trousers. E.g. 'dark grey opaque tights'.\n"
            "      List garments from outermost to innermost.\n\n"
            f"Text: {raw_prompt}"
        )
    else:
        # No reference photos — text-only path.
        # LLMs attend much more reliably to user-message content than deep system-prompt
        # blocks.  When we have authoritative character data (subject refs or character
        # context), echo it directly into the user turn so it cannot be ignored.
        _char_snippets: list[str] = []
        if subject_references:
            for _sname, _sdesc in subject_references.items():
                _char_snippets.append(
                    f"[MANDATORY APPEARANCE FOR '{_sname}' — COPY EVERY DETAIL EXACTLY]\n"
                    f"{_sdesc.strip()}\n"
                    f"[END OF '{_sname}' APPEARANCE — every piece above MUST appear in your output]"
                )
        if character_context and character_context.strip():
            _char_snippets.append(
                f"[MANDATORY CHARACTER APPEARANCE — COPY EVERY DETAIL EXACTLY]\n"
                f"{character_context.strip()}\n"
                f"[END OF CHARACTER APPEARANCE — every piece above MUST appear in your output]"
            )
        if _char_snippets:
            user_content = (
                f"Image request: {raw_prompt}\n\n"
                + "\n\n".join(_char_snippets)
                + "\n\nCRITICAL: your output MUST reproduce the outfit, hair, eyes, skin, and all "
                "accessories EXACTLY as listed above. Do NOT invent, swap, or omit any piece. "
                "Do NOT summarize the outfit — list every individual garment."
            )
        else:
            user_content = f"Image request: {raw_prompt}"
    messages_list = [{"role": "user", "content": user_content}]

    # When reference images are provided, pass them via context_images so chat()
    # handles model selection (vision-first), multimodal message injection, and the
    # full vision-model fallback chain — the same way it does for conversational calls.
    if has_images:
        print(f"[Groq] enhance_image_prompt: using vision model with {len(reference_images)} reference image(s)")

    try:
        enhanced, *_ = await chat(
            messages_list,
            system_prompt=system,
            model=DEFAULT_MODEL,
            context_images=reference_images if has_images else None,
            max_tokens=8192,
        )
        if enhanced:
            enhanced = _THINK_RE.sub("", enhanced)
            enhanced = _THINK_RE_UNCLOSED.sub("", enhanced).strip()
            if len(enhanced) > 5:
                # Detect refusals: text-model fallbacks that couldn't see the
                # reference images produce polite apologies.  Using that text as
                # an image prompt would generate garbage, so fall through to the
                # raw_prompt return at the bottom instead.
                lower = enhanced.lower()
                if any(phrase in lower for phrase in _REFUSAL_PHRASES):
                    print(f"[Groq] Enhancement returned a refusal — discarding, using raw prompt")
                    return raw_prompt
                print(f"[Groq] Prompt enhanced: {enhanced}")
                return enhanced
    except Exception as e:
        print(f"[Groq] Prompt enhancement failed: {e}")
    return raw_prompt


async def chat(
    messages: list,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    context_images: Optional[list] = None,
    max_tokens: int = 1024,
) -> tuple[str, Optional[str], bool]:
    """
    Send a chat request to Groq.

    context_images: optional list of (bytes, mime_type) tuples to inject as
    visual content into the last user message. When provided, a vision-capable
    model is preferred automatically.

    Returns (response_text, image_prompt_or_None, prompt_from_marker).
    prompt_from_marker=True means the image prompt came from the [IMAGE: ...]
    tag and is already well-crafted English — the caller should skip a full
    LLM rewrite and only apply KB enrichment or character context injection.
    prompt_from_marker=False means the prompt is raw user text that still
    needs translation/expansion via enhance_image_prompt.
    If image_prompt is set, the caller should generate an image with Cloudflare.
    """
    client = _client()
    if not client:
        return "Groq API key is not configured. Please set GROQ_API_KEY.", None, False

    groq_messages = []
    if system_prompt:
        groq_messages.append({"role": "system", "content": system_prompt})

    recent = messages[-20:]
    # Build vision-capable messages (multimodal) and a plain-text fallback version.
    # We need the text-only version ready for when we fall back to non-vision models;
    # sending image_url parts to a text model causes errors or garbled output.
    groq_messages_text: list = []
    for i, msg in enumerate(recent):
        content = msg["content"]
        if context_images and i == len(recent) - 1 and msg["role"] == "user":
            image_parts = []
            for img_bytes, img_mime in context_images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                data_url = f"data:{img_mime};base64,{b64}"
                image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
            image_parts.append({"type": "text", "text": content if isinstance(content, str) else str(content)})
            groq_messages.append({"role": msg["role"], "content": image_parts})
            groq_messages_text.append({"role": msg["role"], "content": content if isinstance(content, str) else str(content)})
        else:
            groq_messages.append({"role": msg["role"], "content": content})
            groq_messages_text.append({"role": msg["role"], "content": content})

    # Prefer vision models when images are attached
    if context_images:
        configured_vision = _default_vision_model()
        vision_order = [configured_vision] + [m for m in VISION_MODELS if m != configured_vision]
        models_to_try = vision_order + [m for m in FALLBACK_MODELS if m not in vision_order]
    else:
        models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]

    # Skip models that already hit their daily token limit today
    available = [m for m in models_to_try if not _is_daily_exhausted(m)]
    if len(available) < len(models_to_try):
        skipped = [m for m in models_to_try if _is_daily_exhausted(m)]
        print(f"[Groq] Skipping daily-exhausted models: {skipped}")
    models_to_try = available

    for attempt_model in models_to_try:
        # Use text-only messages when falling back from vision to a text model so we
        # don't send multimodal content (image_url parts) to a model that can't handle it.
        is_vision_model = attempt_model in VISION_MODELS
        msgs_for_attempt = groq_messages if (not context_images or is_vision_model) else groq_messages_text
        if context_images and not is_vision_model:
            print(f"[Groq] Falling back to text model {attempt_model} — stripping image_url parts from messages")
        try:
            response = await client.chat.completions.create(
                model=attempt_model,
                messages=msgs_for_attempt,
                temperature=0.8,
                max_tokens=max_tokens,
            )
            text = _THINK_RE.sub("", response.choices[0].message.content or "").strip()

            # Primary: bot used the [IMAGE: ...] marker → generate silently
            marker_match = _IMAGE_MARKER_RE.search(text)
            if marker_match:
                img_prompt = marker_match.group(1).strip()
                # Strip the marker (and any surrounding whitespace) from the text
                clean_text = _IMAGE_MARKER_RE.sub("", text).strip()
                print(f"[Groq] Image prompt from marker (already enhanced): {img_prompt[:80]!r}")
                # If nothing meaningful remains, return no text (fully silent)
                return (clean_text or None), img_prompt, True

            # Fallback: bot said it can't generate — route to image generator anyway
            # Do NOT call enhance_image_prompt here; the caller (bot.py) handles it
            # so it can also inject character context when needed.
            if response_declines_image(text, messages=messages):
                img_prompt = user_wants_image(messages)
                if img_prompt:
                    print(f"[Groq] Image prompt from fallback (raw, needs enhancement): {img_prompt[:80]!r}")
                    return None, img_prompt, False

            return text, None, False

        except Exception as e:
            err = str(e).lower()
            if "model" in err and ("not found" in err or "deprecated" in err or "invalid" in err):
                print(f"[Groq] Model {attempt_model} unavailable, trying next...")
                continue
            if "rate limit" in err or "429" in err:
                if _is_daily_limit_error(err):
                    _mark_daily_exhausted(attempt_model)
                else:
                    print(f"[Groq] Rate limited on {attempt_model}, trying next...")
                    await asyncio.sleep(2)
                continue
            print(f"[Groq] Error with {attempt_model}: {e}")
            continue

    # All Groq models exhausted — attempt Ollama as a last resort
    print("[Groq] All models failed. Falling back to Ollama...")
    try:
        import ollama_ai
        return await ollama_ai.chat(
            messages,
            system_prompt=system_prompt,
            context_images=context_images,
        )
    except Exception as e:
        print(f"[Groq→Ollama] Fallback also failed: {e}")
        return "I'm having trouble connecting to my AI right now. Please try again in a moment.", None, False


async def generate_image_comment(
    image_prompt: str,
    bot_name: str,
    character_background: str,
    user_request: str = "",
    history: list = None,
) -> str:
    """Generate a short in-character comment to accompany a generated image.

    Uses the recent conversation history so the comment continues the chat
    naturally rather than feeling like a disconnected reaction.
    Returns a 1–2 sentence comment string, or empty string on failure.
    """
    client = _client()
    if not client:
        return ""

    system = (
        f"You are {bot_name}. {character_background}\n"
        "You are mid-conversation and have just drawn/created an image for the user. "
        "Write ONE short sentence (two at most) to send alongside it — something that flows "
        "naturally from the conversation you were just having, like a real person would say.\n"
        "Rules:\n"
        "- Stay in the mood and tone of the conversation. If you were joking, keep joking. "
        "If you were being sincere, be sincere.\n"
        "- Reference the specific thing that was asked for when it feels natural — don't just react generically.\n"
        "- Language: default Traditional Chinese (繁體中文). Only switch if the user wrote a full sentence in another language. Never use Simplified Chinese.\n"
        "- Do NOT say 'Here is', 'Here's', 'I generated', 'I created', '好的', '完成了', or anything that sounds like a task report.\n"
        "- Do NOT describe the image — the user can see it.\n"
        "- Sound like yourself, not an assistant completing a request.\n"
    )

    # Build the message with conversation context + the image request
    context_lines = []
    if history:
        for msg in history[-6:]:
            role_label = "User" if msg["role"] == "user" else bot_name
            context_lines.append(f"{role_label}: {msg['content'][:200]}")
    if user_request:
        context_lines.append(f"User: {user_request}")

    context_block = "\n".join(context_lines)
    msg = (
        f"Conversation so far:\n{context_block}\n\n"
        f"(You just created an image of: {image_prompt[:200]})\n"
        "Write your one-sentence reaction that continues this conversation naturally."
    ) if context_block else (
        f"You just created an image of: {image_prompt[:200]}. "
        "Write one natural in-character sentence to send with it."
    )

    messages = [{"role": "user", "content": msg}]
    try:
        text, *_ = await chat(messages, system_prompt=system, model=DEFAULT_MODEL)
        if text and len(text.strip()) > 2:
            return text.strip()
    except Exception as e:
        print(f"[Groq] Image comment generation failed: {e}")

    return ""


async def generate_suggestions(
    topic: str,
    bot_name: str,
    character_background: str,
    count: int = 3,
    guiding_prompt: str = "",
    language_sample: str = "",
) -> list:
    """Generate short follow-up suggestion buttons (max 80 chars each).

    language_sample should be a snippet of the bot's latest reply so the AI
    can detect and mirror the correct language automatically.
    If guiding_prompt is provided it is prepended to the default instruction.
    Returns concise suggestions suitable for Discord button labels.
    """
    import json

    lang_instruction = ""
    if language_sample and language_sample.strip():
        lang_instruction = (
            f"IMPORTANT: Write every suggestion in the EXACT same language as this sample text "
            f"(detect it automatically — do NOT translate): \"{language_sample[:120]}\"\n"
        )

    base = (
        f"{lang_instruction}"
        f"You are {bot_name}. {character_background}\n"
        f"Generate exactly {count} follow-up messages a user might naturally send next in the conversation.\n"
        f"Write them as the USER speaking to you — casual, warm, and conversational.\n"
        f"Each should be 10–75 characters. No punctuation at the end. No quotes.\n"
        f"Return ONLY a valid JSON array of strings. No markdown, no code fences.\n"
    )
    if guiding_prompt:
        system = f"{guiding_prompt}\n\n{base}"
    else:
        system = base

    if topic and topic.strip():
        prompt = f"Context of the conversation so far:\n{topic[:400]}\n\nGenerate {count} natural follow-up messages the user might send next."
    else:
        prompt = f"Generate {count} casual opening messages someone might send to start chatting with {bot_name}."

    messages = [{"role": "user", "content": prompt}]

    try:
        text, *_ = await chat(messages, system_prompt=system)
        if not text:
            return []
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            arr = json.loads(text[start:end])
            if isinstance(arr, list) and len(arr) > 0:
                # Enforce Discord button label hard limit (80 chars)
                clean = []
                for s in arr[:count]:
                    s = str(s).strip().rstrip(".")
                    if len(s) > 80:
                        s = s[:77] + "..."
                    clean.append(s)
                return clean
    except Exception as e:
        print(f"[Groq] Suggestion parse error: {e}")

    return []  # silently return no buttons rather than wrong-language fallbacks
