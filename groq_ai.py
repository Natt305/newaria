"""
Groq AI client for fast text chat and image understanding.
Falls back to Cloudflare Workers AI image generation when the model's response
indicates it would generate an image but cannot.
"""
import os
import asyncio
import re
import base64
from typing import Optional
from groq import AsyncGroq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Vision model candidates tried in order until one works
VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "llama-4-scout-17b-16e-instruct",
    "meta-llama/Llama-4-Scout-17B-16E",
]

FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
]

IMAGE_TRIGGER_PHRASES = [
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
    # Chinese (Traditional & Simplified)
    re.compile(r"(生成|畫|繪|製作|創作|做).{0,20}(圖|圖片|圖像|插圖|照片)", re.I),
    re.compile(r"(幫我|幫|請|可以|能不能).{0,10}(畫|生成|繪製|做).{0,20}(圖|圖片|圖像)", re.I),
    re.compile(r"(圖片|圖像|照片|插圖).{0,10}(生成|製作|創作)", re.I),
]

_IMAGE_MARKER_RE = re.compile(r"\[IMAGE:\s*(.+?)\]", re.I | re.S)

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
        text, _ = await chat(messages, system_prompt=system, model=DEFAULT_MODEL)
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


def _client() -> Optional[AsyncGroq]:
    if not GROQ_API_KEY:
        return None
    return AsyncGroq(api_key=GROQ_API_KEY)


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

    for model in VISION_MODELS:
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
            text = (response.choices[0].message.content or "").strip()
            if text:
                print(f"[Groq Vision] Success with model: {model}")
                return text
            print(f"[Groq Vision] Empty response from {model}, trying next...")
        except Exception as e:
            err = str(e)
            print(f"[Groq Vision] {model} failed: {err}")
            # If it's clearly a model-not-found error, try the next candidate
            if any(k in err.lower() for k in ("model", "not found", "invalid", "deprecated", "does not exist")):
                continue
            # For other errors (rate limit, auth) skip remaining candidates
            break

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
                    cleaned = re.sub(r"(?i)(please\s+)?(generate|create|draw|make|show|paint|illustrate)\s+(me\s+)?(an?\s+)?", "", content).strip()
                    return cleaned if cleaned else content
            return None
    return None


async def chat(
    messages: list,
    system_prompt: str = "",
    model: str = DEFAULT_MODEL,
) -> tuple[str, Optional[str]]:
    """
    Send a chat request to Groq.

    Returns (response_text, image_prompt_or_None).
    If image_prompt is set, the caller should generate an image with Gemini.
    """
    client = _client()
    if not client:
        return "Groq API key is not configured. Please set GROQ_API_KEY.", None

    groq_messages = []
    if system_prompt:
        groq_messages.append({"role": "system", "content": system_prompt})
    for msg in messages[-20:]:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]

    for attempt_model in models_to_try:
        try:
            response = await client.chat.completions.create(
                model=attempt_model,
                messages=groq_messages,
                temperature=0.8,
                max_tokens=1024,
            )
            text = (response.choices[0].message.content or "").strip()

            # Primary: bot used the [IMAGE: ...] marker → generate silently
            marker_match = _IMAGE_MARKER_RE.search(text)
            if marker_match:
                img_prompt = marker_match.group(1).strip()
                # Strip the marker (and any surrounding whitespace) from the text
                clean_text = _IMAGE_MARKER_RE.sub("", text).strip()
                # If nothing meaningful remains, return no text (fully silent)
                return (clean_text or None), img_prompt

            # Fallback: bot said it can't generate — route to image generator anyway
            if response_declines_image(text):
                img_prompt = user_wants_image(messages)
                if img_prompt:
                    return None, img_prompt

            return text, None

        except Exception as e:
            err = str(e).lower()
            if "model" in err and ("not found" in err or "deprecated" in err or "invalid" in err):
                print(f"[Groq] Model {attempt_model} unavailable, trying next...")
                continue
            if "rate limit" in err or "429" in err:
                print(f"[Groq] Rate limited on {attempt_model}, trying next...")
                await asyncio.sleep(2)
                continue
            print(f"[Groq] Error with {attempt_model}: {e}")
            continue

    return "I'm having trouble connecting to my text AI right now. Please try again in a moment.", None


async def generate_suggestions(
    topic: str,
    bot_name: str,
    character_background: str,
    count: int = 3,
    guiding_prompt: str = "",
) -> list:
    """Generate short follow-up suggestion buttons (max 80 chars each).

    If guiding_prompt is provided it fully replaces the default system instruction,
    so callers can control the tone/style/content of suggestions.
    Returns concise suggestions suitable for Discord button labels.
    """
    import json

    base = (
        f"You are {bot_name}. {character_background}\n"
        f"Generate exactly {count} follow-up questions or suggestions.\n"
        f"Return ONLY a valid JSON array of strings. No markdown, no code fences.\n"
        f'Example: ["Can you tell me more about that", "Show me some examples of this", "What are the other options"]'
    )
    if guiding_prompt:
        # Custom prompt guides tone/style/length; JSON requirement is always enforced
        system = f"{guiding_prompt}\n\n{base}"
    else:
        system = base + "\nEach suggestion should be 30–70 characters, natural and conversational, no punctuation at the end."

    if topic and topic.strip():
        prompt = f"Context: {topic[:300]}\n\nGenerate {count} brief, punchy follow-ups."
    else:
        prompt = f"Generate {count} short, interesting opening lines for {bot_name}."

    messages = [{"role": "user", "content": prompt}]

    try:
        text, _ = await chat(messages, system_prompt=system)
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            arr = json.loads(text[start:end])
            if isinstance(arr, list) and len(arr) > 0:
                # Ensure each suggestion is short enough for button labels
                clean = []
                for s in arr[:count]:
                    s = str(s).strip().rstrip(".")
                    if len(s) > 80:
                        s = s[:77] + "..."
                    clean.append(s)
                return clean
    except Exception as e:
        print(f"[Groq] Suggestion parse error: {e}")

    return [
        f"Tell me more",
        "What else?",
        "Show examples",
    ]
