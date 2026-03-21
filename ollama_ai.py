"""
Ollama AI client — local model backend using Ollama's OpenAI-compatible API.
Mirrors the public interface of groq_ai.py so ai_backend.py can delegate to either.
"""
import os
import re
import base64
import asyncio
import json as _json
from typing import Optional
import aiohttp

DEFAULT_MODEL = "gemma3:12b"
DEFAULT_VISION_MODEL = "gemma3:12b"

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
    re.compile(r"\bgenerate\b.{0,30}\bimage\b", re.I),
    re.compile(r"\bcreate\b.{0,30}\bimage\b", re.I),
    re.compile(r"\bdraw\b", re.I),
    re.compile(r"\bpaint\b.{0,30}\bpicture\b", re.I),
    re.compile(r"\bmake\b.{0,30}\bimage\b", re.I),
    re.compile(r"\bshow\b.{0,30}\bpicture\b", re.I),
    re.compile(r"\billustrate\b", re.I),
    re.compile(r"(生成|畫|繪|製作|創作|做).{0,20}(圖|圖片|圖像|插圖|照片)", re.I),
    re.compile(r"(幫我|幫|請|可以|能不能).{0,10}(畫|生成|繪製|做).{0,20}(圖|圖片|圖像)", re.I),
    re.compile(r"(圖片|圖像|照片|插圖).{0,10}(生成|製作|創作)", re.I),
]

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


def _base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def _vision_model() -> str:
    return os.environ.get("OLLAMA_VISION_MODEL", DEFAULT_VISION_MODEL)


def is_self_referential_image(prompt: str) -> bool:
    return bool(_SELF_REF_RE.search(prompt))


def is_recall_request(text: str) -> bool:
    return bool(_RECALL_RE.search(text))


def response_declines_image(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in IMAGE_TRIGGER_PHRASES)


def user_wants_image(messages: list) -> Optional[str]:
    for msg in reversed(messages):
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, str):
                for pat in IMAGE_REQUEST_PATTERNS:
                    if pat.search(content):
                        cleaned = re.sub(
                            r"(?i)(please\s+)?(generate|create|draw|make|show|paint|illustrate)\s+(me\s+)?(an?\s+)?",
                            "", content,
                        ).strip()
                        return cleaned if cleaned else content
            return None
    return None


async def _call_ollama(
    messages: list,
    model: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> Optional[str]:
    """Make a single chat completion request to Ollama. Returns the text response or None."""
    url = f"{_base_url()}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"[Ollama] HTTP {resp.status}: {body[:200]}")
                    return None
                data = await resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
    except aiohttp.ClientConnectorError:
        print(f"[Ollama] Cannot connect to {_base_url()} — is Ollama running?")
        return None
    except Exception as e:
        print(f"[Ollama] Request error: {e}")
        return None


async def chat(
    messages: list,
    system_prompt: str = "",
    model: str = "",
    context_images: Optional[list] = None,
) -> tuple[str, Optional[str]]:
    """Send a chat request to Ollama. Returns (response_text, image_prompt_or_None).

    context_images: optional list of (bytes, mime_type) tuples injected as visual
    content into the last user message. Switches to the vision model automatically.
    """
    # Use vision model when context images are provided
    if context_images:
        active_model = _vision_model()
    else:
        active_model = model if model else _model()

    ollama_messages = []
    if system_prompt:
        ollama_messages.append({"role": "system", "content": system_prompt})

    recent = messages[-20:]
    for i, msg in enumerate(recent):
        content = msg["content"]
        if isinstance(content, list):
            text_parts = [p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = " ".join(text_parts)
        # Inject context_images into the last user message
        if context_images and i == len(recent) - 1 and msg["role"] == "user":
            image_parts = []
            for img_bytes, img_mime in context_images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                data_url = f"data:{img_mime};base64,{b64}"
                image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
            image_parts.append({"type": "text", "text": content if isinstance(content, str) else str(content)})
            content = image_parts
        ollama_messages.append({"role": msg["role"], "content": content})

    text = await _call_ollama(ollama_messages, model=active_model)

    if text is None:
        return "I'm having trouble connecting to Ollama right now. Please make sure Ollama is running.", None

    marker_match = _IMAGE_MARKER_RE.search(text)
    if marker_match:
        img_prompt = marker_match.group(1).strip()
        clean_text = _IMAGE_MARKER_RE.sub("", text).strip()
        return (clean_text or None), img_prompt

    if response_declines_image(text):
        img_prompt = user_wants_image(messages)
        if img_prompt:
            img_prompt = await enhance_image_prompt(img_prompt)
            return None, img_prompt

    return text, None


async def understand_image(
    image_bytes: bytes,
    mime_type: str,
    question: str = "Describe this image in detail.",
) -> Optional[str]:
    """Analyze an image using Ollama's vision-capable model."""
    model = _vision_model()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": question},
            ],
        }
    ]

    url = f"{_base_url()}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 1024,
        "stream": False,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"[Ollama Vision] HTTP {resp.status}: {body[:200]}")
                    return None
                data = await resp.json()
                text = (data["choices"][0]["message"]["content"] or "").strip()
                if text:
                    print(f"[Ollama Vision] Success with model: {model}")
                    return text
                print(f"[Ollama Vision] Empty response from {model}")
                return None
    except aiohttp.ClientConnectorError:
        print(f"[Ollama Vision] Cannot connect to {_base_url()} — is Ollama running?")
        return None
    except Exception as e:
        print(f"[Ollama Vision] Error: {e}")
        return None


async def extract_memories(exchange: str, bot_name: str) -> list:
    """Extract 0–3 memorable facts from a conversation exchange."""
    if not exchange or len(exchange.strip()) < 30:
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
        text, _ = await chat(messages, system_prompt=system)
        if not text:
            return []
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            arr = _json.loads(text[start:end])
            if isinstance(arr, list):
                return [str(s).strip() for s in arr[:3] if isinstance(s, str) and s.strip()]
    except Exception as e:
        print(f"[Ollama Memory] Extract error: {e}")

    return []


async def enhance_image_prompt(raw_prompt: str, character_context: str = "") -> str:
    """Translate and expand a raw prompt into a rich English image-generation prompt."""
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
        enhanced, _ = await chat(messages_list, system_prompt=system)
        if enhanced and len(enhanced.strip()) > 5:
            print(f"[Ollama] Prompt enhanced: {enhanced[:120]}")
            return enhanced.strip()
    except Exception as e:
        print(f"[Ollama] Prompt enhancement failed: {e}")
    return raw_prompt


async def generate_image_comment(
    image_prompt: str,
    bot_name: str,
    character_background: str,
    user_request: str = "",
    history: list = None,
) -> str:
    """Generate a short in-character comment to accompany a generated image."""
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
        text, _ = await chat(messages, system_prompt=system)
        if text and len(text.strip()) > 2:
            return text.strip()
    except Exception as e:
        print(f"[Ollama] Image comment generation failed: {e}")

    return ""


async def generate_suggestions(
    topic: str,
    bot_name: str,
    character_background: str,
    count: int = 3,
    guiding_prompt: str = "",
) -> list:
    """Generate short follow-up suggestion buttons (max 80 chars each)."""
    base = (
        f"You are {bot_name}. {character_background}\n"
        f"Generate exactly {count} follow-up messages a user might naturally send next in the conversation.\n"
        f"Write them as the USER speaking to you — casual, warm, and conversational, like something you'd actually type to a friend.\n"
        f"Each should be 40–75 characters. No punctuation at the end. No quotes.\n"
        f"Return ONLY a valid JSON array of strings. No markdown, no code fences.\n"
        f'Example: ["That\'s so cool, what made you get into that", "Okay but wait, how does that actually work", "Can you show me a picture of that"]'
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
        text, _ = await chat(messages, system_prompt=system)
        if not text:
            return []
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            arr = _json.loads(text[start:end])
            if isinstance(arr, list) and len(arr) > 0:
                clean = []
                for s in arr[:count]:
                    s = str(s).strip().rstrip(".")
                    if len(s) > 80:
                        s = s[:77] + "..."
                    clean.append(s)
                return clean
    except Exception as e:
        print(f"[Ollama] Suggestion parse error: {e}")

    return [
        "Hmm, tell me more about that",
        "Wait, I want to know more",
        "Okay that's interesting, go on",
    ]
