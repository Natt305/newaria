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

_IMAGE_MARKER_RE = re.compile(
    r"\[(?:IMAGE|圖像生成|圖像|生成圖像|生成图像|图像生成|图像|GENERATE IMAGE|GEN IMAGE):\s*(.+?)\]",
    re.I | re.S,
)

_SELF_REF_RE = re.compile(
    r"\b(selfie|self.?portrait|photo of me|picture of me|my face|my photo|my picture|"
    r"what i look like|how i look|my appearance|my outfit|my hair|my eyes|"
    r"myself|me posing|me standing|me sitting|portrait of me)\b"
    r"|自拍|我的照片|我的樣子|我的臉|我的外貌|拍一張我|我的自拍",
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


def response_declines_image(text: str) -> bool:
    """Return True if Groq's reply says it can't generate an image."""
    lower = text.lower()
    return any(phrase in lower for phrase in IMAGE_TRIGGER_PHRASES)


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


async def enhance_image_prompt(raw_prompt: str, character_context: str = "") -> str:
    """Translate and expand a raw (possibly Chinese) prompt into a rich English
    image-generation prompt suitable for Cloudflare Workers AI.

    If character_context is provided (appearance descriptions of the bot's character),
    it will be used to ground self-referential prompts like 'selfie' or 'photo of me'.

    Returns the enhanced prompt, or the original if enhancement fails.
    """
    client = _client()
    if not client:
        return raw_prompt

    char_block = ""
    if character_context and character_context.strip():
        char_block = (
            f"\nThe image may involve the character whose appearance is described below. "
            f"If the prompt is self-referential (e.g. 'selfie', 'photo of me', 'my face', 'what I look like'), "
            f"use these appearance details as the subject of the image:\n"
            f"{character_context.strip()}\n"
        )

    system = (
        "You are an expert image-prompt writer for AI image generators.\n"
        "Given a user's image request (which may be in Chinese or English), "
        "rewrite it as a single, rich English prompt for an AI image model.\n"
        f"{char_block}"
        "Rules:\n"
        "- Output ONLY the prompt text — no intro, no quotes, no explanation.\n"
        "- Always write in English.\n"
        "- Be specific: include subject, art style, lighting, colors, mood, and setting.\n"
        "- Aim for 20-60 words.\n"
        "- Do NOT start with 'Generate', 'Create', 'Draw', 'An image of', etc.\n"
        "Good output: vibrant cherry blossom park in Kyoto at sunset, soft golden light, "
        "anime art style, petals drifting in the breeze, peaceful atmosphere\n"
    )

    messages_list = [{"role": "user", "content": f"Image request: {raw_prompt}"}]
    try:
        enhanced, *_ = await chat(messages_list, system_prompt=system, model=DEFAULT_MODEL)
        if enhanced and len(enhanced.strip()) > 5:
            print(f"[Groq] Prompt enhanced: {enhanced[:120]}")
            return enhanced.strip()
    except Exception as e:
        print(f"[Groq] Prompt enhancement failed: {e}")
    return raw_prompt


async def chat(
    messages: list,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
    context_images: Optional[list] = None,
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
    for i, msg in enumerate(recent):
        content = msg["content"]
        # Inject context_images into the last user message as multimodal content
        if context_images and i == len(recent) - 1 and msg["role"] == "user":
            image_parts = []
            for img_bytes, img_mime in context_images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                data_url = f"data:{img_mime};base64,{b64}"
                image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
            image_parts.append({"type": "text", "text": content if isinstance(content, str) else str(content)})
            content = image_parts
        groq_messages.append({"role": msg["role"], "content": content})

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
        try:
            response = await client.chat.completions.create(
                model=attempt_model,
                messages=groq_messages,
                temperature=0.8,
                max_tokens=1024,
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
            if response_declines_image(text):
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
