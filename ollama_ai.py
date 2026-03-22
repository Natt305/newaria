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

_IMAGE_MARKER_RE = re.compile(
    r"\[(?:IMAGE|圖像生成|圖像|生成圖像|生成图像|图像生成|图像|GENERATE IMAGE|GEN IMAGE):\s*(.+?)\]",
    re.I | re.S,
)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_RE_UNCLOSED = re.compile(r"<think>.*", re.DOTALL)

# Phrases that indicate the model broke character and admitted to being an AI / LLM.
# Deliberately specific to SELF-IDENTIFICATION — avoids false positives when the bot
# legitimately discusses AI topics in conversation.
_BREAKS_CHARACTER_RE = re.compile(
    r"(大型語言模型|語言模型|language model|large language model|"
    # Model self-identifies by name (only triggers when model names appear as self-reference)
    r"我是\s*(gemma|llama|mistral|qwen|phi|claude|gpt|chatgpt|bard|gemini|deepseek|kimi|copilot)\b|"
    r"i am\s*(gemma|llama|mistral|qwen|phi|claude|gpt|chatgpt|bard|gemini|deepseek|kimi|copilot)\b|"
    r"i'm\s*(gemma|llama|mistral|qwen|phi|claude|gpt|chatgpt|bard|gemini|deepseek|kimi|copilot)\b|"
    # Company training attributions (self-disclosure: "trained by X")
    r"由\s*(google|deepmind|google deepmind|meta|mistral|anthropic|openai|microsoft|baidu|alibaba)\s*.{0,15}(訓練|開發|製作|created|trained|made|built)|"
    r"(trained|created|made|built|developed)\s*by\s*(google|deepmind|meta|mistral|anthropic|openai|microsoft)|"
    # Generic AI self-disclosure
    r"我是\s*(一個|個)?\s*(ai助手|ai|人工智慧|人工智能|助手)|"
    r"開放權重的\s*ai|open.?weight.*model|"
    r"i am an ai\b|i'm an ai\b|as an ai\b|i'm just an ai|i am just an ai|"
    r"i'm a language|i am a language|i'm a text.?based|i am a text.?based)",
    re.I,
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


_NEGATION_RE = re.compile(
    r"無法|不能|沒辦法|做不到|辦不到|can'?t|cannot|unable|not able|no way",
    re.I,
)


def response_declines_image(text: str, messages: list | None = None) -> bool:
    """Return True if the reply says it can't generate an image.

    Two-pass check:
    1. Fast path — exact phrase list (language-specific known phrases).
    2. Context-gated heuristic — only fires when the user clearly wanted an image
       AND the response has no [IMAGE:] marker AND contains a negation/inability word.
    """
    lower = text.lower()
    if any(phrase in lower for phrase in IMAGE_TRIGGER_PHRASES):
        return True
    if messages is not None and not _IMAGE_MARKER_RE.search(text):
        if user_wants_image(messages) and _NEGATION_RE.search(text):
            return True
    return False


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


def _estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token across all message content."""
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", ""))
    return total // 4


def _to_native_messages(messages: list) -> list:
    """Convert OpenAI-format messages to Ollama native /api/chat format.

    OpenAI multimodal content (list with image_url + text items) is converted
    to Ollama native format where images are a separate top-level list of raw
    base64 strings and content is plain text.
    """
    native = []
    for msg in messages:
        content = msg.get("content", "")
        images = []
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    # Strip "data:<mime>;base64," prefix to get raw base64
                    if ";base64," in url:
                        url = url.split(";base64,", 1)[1]
                    if url:
                        images.append(url)
            content = " ".join(text_parts)
        entry = {"role": msg["role"], "content": content}
        if images:
            entry["images"] = images
        native.append(entry)
    return native


async def _call_ollama(
    messages: list,
    model: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    num_ctx: int = 16384,
) -> Optional[str]:
    """Make a single chat completion request using Ollama's native /api/chat endpoint.

    Uses the native endpoint (not the OpenAI-compatible one) because only the
    native endpoint reliably respects options.num_ctx. The OpenAI-compatible
    endpoint silently ignores it, leaving the model at its baked-in default
    (often 4096 tokens), which causes system prompt truncation for large KB sets.
    """
    url = f"{_base_url()}/api/chat"
    native_messages = _to_native_messages(messages)
    payload = {
        "model": model,
        "messages": native_messages,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    est = _estimate_tokens(messages)
    sys_len = len(messages[0].get("content", "")) if messages and messages[0]["role"] == "system" else 0
    print(f"[Ollama] Sending to {model} | sys_prompt={sys_len}ch | ~{est} tokens total | num_ctx={num_ctx}")
    if messages and messages[0]["role"] == "system":
        print(f"[Ollama] System prompt start: {messages[0]['content'][:200]!r}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"[Ollama] HTTP {resp.status}: {body[:500]}")
                    return None
                data = await resp.json()
                content = _THINK_RE.sub("", data.get("message", {}).get("content") or "").strip()
                prompt_tokens = data.get("prompt_eval_count", "?")
                completion_tokens = data.get("eval_count", "?")
                done_reason = data.get("done_reason", "?")
                print(f"[Ollama] Response: done_reason={done_reason} | prompt_tokens={prompt_tokens} | completion_tokens={completion_tokens}")
                print(f"[Ollama] Response text: {content[:200]!r}")
                return content
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
) -> tuple[str, Optional[str], bool]:
    """Send a chat request to Ollama. Returns (response_text, image_prompt_or_None, prompt_from_marker).

    prompt_from_marker=True means the image prompt came from the [IMAGE: ...] tag
    and is already well-crafted English — the caller should skip a full LLM rewrite.
    prompt_from_marker=False means the prompt is raw user text needing enhancement.
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
        return "I'm having trouble connecting to Ollama right now. Please make sure Ollama is running.", None, False

    # Character-break guard: if the model admitted to being an AI/LLM, retry once
    # with a fresh, laser-focused minimal prompt — no extra context to confuse it.
    if _BREAKS_CHARACTER_RE.search(text):
        print("[Ollama] Character break detected — retrying with laser-focused identity prompt")
        # Extract bot name from the system prompt ("You are {name}.")
        _name_m = re.search(r"^You are ([^\.\n]+)\.", system_prompt)
        bot_name_hint = _name_m.group(1).strip() if _name_m else "the character"

        # Build a minimal system prompt focused ONLY on identity
        identity_system = (
            f"YOUR NAME IS {bot_name_hint}. YOU ARE {bot_name_hint}.\n"
            f"NOT Gemma. NOT an AI. NOT a language model. NOT made by Google or anyone else.\n"
            f"You are {bot_name_hint} — a real person. Answer every message as {bot_name_hint}.\n"
            f"If asked who you are: say you are {bot_name_hint} and stay in character.\n"
            f"NEVER say you are Gemma, an AI, or a language model — EVER.\n"
        )
        # Append the character's own background if we can extract it
        _bg_m = re.search(r"You are [^\n]+\. (.+?)(?:\n\nYou are NOT an AI)", system_prompt, re.S)
        if _bg_m:
            identity_system += f"\nBackground: {_bg_m.group(1).strip()[:400]}\n"

        # Get the last user message text (strip multimodal parts)
        last_user_text = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                c = msg["content"]
                if isinstance(c, str):
                    last_user_text = c
                elif isinstance(c, list):
                    for p in c:
                        if isinstance(p, dict) and p.get("type") == "text":
                            last_user_text = p.get("text", "")
                            break
                break

        # Fresh minimal conversation — last 4 turns max + the triggering user message
        retry_msgs = [{"role": "system", "content": identity_system}]
        for msg in messages[-4:]:
            c = msg["content"]
            if isinstance(c, list):
                parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                c = " ".join(parts)
            retry_msgs.append({"role": msg["role"], "content": c})
        if not any(m["role"] == "user" for m in retry_msgs[1:]) and last_user_text:
            retry_msgs.append({"role": "user", "content": last_user_text})

        retry_text = await _call_ollama(retry_msgs, model=active_model, temperature=0.6)
        if retry_text:
            if _BREAKS_CHARACTER_RE.search(retry_text):
                print("[Ollama] Retry also broke character — using retry result anyway")
            else:
                print("[Ollama] Retry succeeded — character restored")
            text = retry_text

    marker_match = _IMAGE_MARKER_RE.search(text)
    if marker_match:
        img_prompt = marker_match.group(1).strip()
        clean_text = _IMAGE_MARKER_RE.sub("", text).strip()
        print(f"[Ollama] Image prompt from marker (already enhanced): {img_prompt[:80]!r}")
        return (clean_text or None), img_prompt, True

    if response_declines_image(text, messages=messages):
        img_prompt = user_wants_image(messages)
        if img_prompt:
            print(f"[Ollama] Image prompt from fallback (raw, needs enhancement): {img_prompt[:80]!r}")
            return None, img_prompt, False

    return text, None, False


async def understand_image(
    image_bytes: bytes,
    mime_type: str,
    question: str = "Describe this image in detail.",
) -> Optional[str]:
    """Analyze an image using Ollama's vision-capable model (native /api/chat endpoint)."""
    model = _vision_model()
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": question,
            "images": [b64],
        }
    ]

    url = f"{_base_url()}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.5,
            "num_predict": 1024,
        },
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"[Ollama Vision] HTTP {resp.status}: {body[:500]}")
                    return None
                data = await resp.json()
                text = _THINK_RE.sub("", data.get("message", {}).get("content") or "").strip()
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
        text, *_ = await chat(messages, system_prompt=system)
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


async def enhance_image_prompt(
    raw_prompt: str,
    character_context: str = "",
    subject_references: dict = None,
    reference_images: list = None,
) -> str:
    """Translate and expand a raw prompt into a rich English image-generation prompt.

    subject_references: {name: description} for named KB subjects.
    reference_images: list of (bytes, mime_type) tuples passed to the vision model.
    """
    char_block = ""
    if character_context and character_context.strip():
        char_block = (
            f"\n[CHARACTER APPEARANCE — VERIFIED GROUND TRUTH]\n"
            f"The following contains confirmed, factual appearance details for the character. "
            f"You MUST use ALL of these physical traits in your output. Do NOT invent, alter, or substitute any of them.\n"
            f"Eye color, hair color, hairstyle, skin tone, AND OUTFIT described here are FINAL — "
            f"they override anything conflicting in the raw prompt or your own assumptions. "
            f"If an outfit is described here, reproduce it in full detail in the output. "
            f"Do NOT summarize, omit, or replace any garment piece listed:\n"
            f"{character_context.strip()}\n"
        )

    ref_block = ""
    if subject_references:
        lines = []
        for name, desc in subject_references.items():
            lines.append(f"  {name}: {desc}")
        ref_block = (
            "\n\nVERIFIED GROUND TRUTH — the following subjects appear in the image request. "
            "Their physical traits below are authoritative; do NOT alter or invent details:\n"
            + "\n".join(lines) + "\n"
        )

    has_images = bool(reference_images)
    image_note = (
        "\nReference photos are attached. Observe them directly — they are the "
        "definitive source of truth for visual traits. Text descriptions above provide "
        "supplementary context.\n"
        "OVERRIDE 1 — COLORS IN THE RAW PROMPT ARE WRONG. "
        "Every color word in the raw prompt (hair, eyes, skin) was written without photos and is a fabrication. "
        "Discard them all. Describe ONLY what you literally see in the reference photos.\n"
        "OVERRIDE 2 — OUTFIT IN THE RAW PROMPT MAY BE WRONG. Ignore it. Reconstruct from the photos only.\n"
        "OVERRIDE 3 — ART STYLE: identify the 2D anime rendering style from the reference photos — "
        "the target output is always a 2D anime illustration regardless of how the reference was produced. "
        "Describe the specific 2D style you observe (cell-shaded, painterly, etc.). "
        "NEVER default to a generic phrase.\n"
        if has_images else ""
    )

    system = (
        "You are an expert image-prompt writer for AI image generators.\n"
        "Given a user's image request (which may be in Chinese or English), "
        "rewrite it as a single, rich English prompt for an AI image model.\n"
        f"{char_block}"
        f"{ref_block}"
        f"{image_note}"
        "Rules:\n"
        "- Output ONLY the prompt text — no intro, no quotes, no explanation.\n"
        "- Always write in English.\n"
        "- Be specific: include subject, art style, lighting, colors, mood, and setting.\n"
        "- Aim for 150-300 words.\n"
        "- Do NOT start with 'Generate', 'Create', 'Draw', 'An image of', etc.\n"
        "- Physical details from references (hair color, eye color, hairstyle, skin tone) "
        "ALWAYS take priority over anything in the raw prompt. Incorporate them naturally.\n"
        "- ART STYLE (mandatory, reference-matched when photos are present): "
        "When reference images are attached, you MUST analyze their rendering style and describe it precisely. "
        "Cover ALL of the following axes:\n"
        "  * RENDERING TYPE: e.g. 'clean cell-shaded 2D anime illustration', 'soft painterly 2D digital illustration', "
        "'rough sketch-style 2D line art', 'polished semi-realistic 2D anime', 'flat graphic 2D anime'. "
        "The output is always a 2D anime image — describe the 2D style that best matches the reference.\n"
        "  * LINE ART: e.g. 'clean sharp black outlines', 'soft anti-aliased thin lines', "
        "'thick expressive ink lines', 'no visible outlines / lineless'\n"
        "  * SHADING STYLE: e.g. 'flat shading with minimal gradients', 'soft gradient shading', "
        "'hard cell shading with two shadow tones', 'rich volumetric shading', 'airbrush-smooth blending'\n"
        "  * COLOR PALETTE CHARACTER: e.g. 'high saturation vivid palette', "
        "'low saturation muted pastel palette', 'warm golden-hour color grading', 'cool desaturated tones'\n"
        "  * ANY DISTINCTIVE STYLE MARKERS: e.g. 'subtle screen-tone texture', "
        "'glossy highlight spots on hair', 'soft glowing rim light', 'watercolor-bleed edges'\n"
        "When NO reference images are provided, default to the character's established art style "
        "or 'clean cell-shaded anime, sharp black outlines, soft gradient shading, vivid color palette'.\n"
        "NEVER use a bare generic phrase like 'anime art style' or 'anime-style illustration, 2D art' "
        "— always expand to the specific rendering axes above.\n"
        "Never use 'photorealistic' or 'photograph' for character prompts.\n"
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
        "- HAIR COLOR: describe the exact shade precisely. 'Near-white with a barely-there "
        "cool mint tint' is very different from 'mint-green' or 'teal'. Match what you see.\n"
        "- EYE COLOR (three-axis required): describe eye color using ALL THREE axes — "
        "never collapse into a single color name like 'amber' or 'hazel':\n"
        "  * HUE FAMILY: e.g. 'warm olive-brown', 'gray-green', 'khaki', 'dusty brown-olive'\n"
        "  * SATURATION: explicitly state — 'very low saturation', 'muted', 'desaturated', "
        "'soft' — or 'vivid' if truly vivid\n"
        "  * BRIGHTNESS/CONTRAST: e.g. 'medium brightness', 'slightly dark', 'light'\n"
        "  Example: 'very low-saturation warm olive-brown eyes, muted and soft, medium brightness'\n"
        "  NOT: 'amber eyes' or 'honey-gold eyes' — too vague, forces model to render vivid warm amber\n"
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
        "'wide circle-cut open-front black shoulder cape, [trim details], draping to [length]'.\n"
        "  • SHOULDER COVERS / EPAULETTES = the structured black padded sections at the very top of "
        "the shoulders where the cape attaches. They look like built-in shoulder pads or caps. "
        "Describe them as a SEPARATE entry: 'structured black shoulder cover epaulettes with gold decoration, "
        "built into the cape shoulder attachment'. Do NOT omit them or merge them into the cape entry.\n"
        "  • JABOT = a decorative V-shaped or waterfall ruffled chest piece worn at the sternum/chest, "
        "NOT at the neck. It forms a V-opening with layered white ruffled fabric fanning downward. "
        "It is SECURED with a brooch or gemstone pin at the apex of the V. "
        "It sits INSIDE the cape's neckline, visible between the collar and the inner top. "
        "Describe as: 'chest-level white V-shaped ruffled jabot with layered gathered ruffles, "
        "secured at apex with a large [color] faceted gemstone brooch'. "
        "NEVER call it a 'collar' or 'detachable collar' — it is a jabot.\n"
        "  • SHORTS = visible leg garment ending above mid-thigh. "
        "If red fabric with rows of gold/cream buttons is visible at the lower body, it is red shorts, "
        "NOT a vest or blouse. Describe as: 'deep crimson red shorts with double row of [color] buttons'.\n"
        "  • BROOCH RULE: A gemstone pin/brooch is ALWAYS described as part of the garment it fastens. "
        "If it fastens a white ruffled jabot, include it in the jabot entry's details. "
        "NEVER attribute a green gemstone brooch to a red or black garment.\n"
        "For EACH piece you MUST describe ALL of the following axes — skipping any axis makes the description incomplete:\n"
        "  1. EXACT COLOR(S): not just 'red' — use precise shade names like 'deep crimson', 'off-white', 'charcoal black', 'navy midnight blue'\n"
        "  2. GARMENT TYPE: exact garment name using the vocabulary above\n"
        "  3. MATERIAL / TEXTURE: e.g. 'matte wool', 'shiny satin', 'sheer chiffon', 'soft ribbed knit', 'smooth patent leather', 'opaque cotton'\n"
        "  4. SILHOUETTE / FIT: e.g. 'slim-fit', 'oversized', 'structured', 'flared', 'form-fitting', 'loose draped'\n"
        "  5. DISTINCTIVE DETAILS: buttons, trim color/type, embroidery, lace trim, zippers, bow, ruffle, collar shape, pattern, logos, cutouts, hardware\n"
        "Format: join pieces with ' + ', one fully described piece per entry.\n"
        "Include ALL accessories: hats, belts, bags, jewelry, gloves, wrist cuffs, socks, tights, shoes — each as its own entry.\n"
        "BAD (missing axes): 'black jacket' — NO color precision, NO material, NO fit, NO details\n"
        "BAD (wrong type): 'long black coat' for an open-front cape — use 'open-front black shoulder cape'\n"
        "BAD (wrong placement): 'white collar with brooch' for a chest jabot — use 'chest-level white ruffled jabot secured with green gemstone brooch'\n"
        "GOOD: 'slim-fit high-collar charcoal black matte wool blazer with narrow gold braid trim along both lapels, "
        "single column of small polished gold dome buttons down the center front, structured shoulders'\n"
        "NEVER write vague single-phrase outfit summaries like 'gothic uniform', 'casual outfit', 'school uniform', "
        "'military-style outfit', or 'black and red outfit' — these are automatic failures.\n"
        "⚠️ FORMAT-ONLY EXAMPLE — THIS IS A HYPOTHETICAL DIFFERENT CHARACTER. "
        "DO NOT COPY THIS OUTFIT INTO YOUR OUTPUT. "
        "YOUR OUTPUT MUST DESCRIBE WHAT YOU ACTUALLY SEE IN THE REFERENCE PHOTOS, NOT THIS EXAMPLE.\n"
        "Format example: short dark auburn wavy bob, slight inward curl at ends, no bangs, no ahoge, "
        "warm hazel eyes with greenish undertone, warm medium saturation, soft brightness, "
        "warm light-brown lashes, low contrast, fine and delicate, "
        "light warm ivory skin with golden undertone, "
        "outfit: oversized dusty-rose knit turtleneck sweater, cable-knit texture, ribbed hem and cuffs + "
        "wide-leg high-waist cream linen trousers, pleated front, tapered ankle, smooth matte texture + "
        "white leather loafers, moc-toe stitching, thin tan rubber sole, no hardware, "
        "anime-style illustration, 2D art, soft warm lighting\n"
        "END FORMAT-ONLY EXAMPLE — your output must describe the character in the reference photos, NOT the example above.\n"
        "CRITICAL REMINDER — OUTFIT: Every piece in a correct output has all 5 axes described. "
        "EVERY garment and accessory must be its own entry joined by ' + '. "
        "If any piece is missing color precision, material, fit, or details, your output is WRONG. "
        "If you collapse the entire outfit into fewer than 4 separate entries, your output is WRONG.\n"
        "MANDATORY PRE-SUBMISSION CHECKLIST — before writing your final output, verify ALL of these are present:\n"
        "  ✓ HAIR COLOR — exact shade with warmth/coolness and saturation\n"
        "  ✓ EYE COLOR — hue family + saturation + brightness (three axes). NEVER write 'amber', 'gold', "
        "'honey', or 'warm amber' alone — these force Flux to render vivid warm eyes. "
        "Always qualify with saturation level: 'muted', 'very low-saturation', 'desaturated'. "
        "Example: 'very low-saturation muted olive-brown eyes with yellowish-green undertone'. "
        "Write ONLY the three-axis version — no shorthand before it.\n"
        "  ✓ EYELASHES — hue (cool or warm) + darkness override (state explicitly if NOT near-black) + weight. "
        "This field is MANDATORY. An output that does not contain eyelash description is INCOMPLETE.\n"
        "  ✓ SKIN TONE — precise shade\n"
        "  ✓ OUTFIT — all pieces with 5 axes each. "
        "If the reference shows a white ruffled chest piece with a gemstone brooch, "
        "it MUST appear as 'chest-level white V-shaped ruffled jabot ... green gemstone brooch'. "
        "If the reference shows a wide flared fabric from the shoulders (no sleeves), "
        "it MUST appear as 'open-front [color] wide-hemmed shoulder cape'. "
        "These cannot be omitted or merged into other garments.\n"
        "  ✓ ART STYLE — all five rendering axes\n"
        "  ✓ SCENE/POSE/SETTING — extracted from the request text\n"
        "If any item from this checklist is missing from your output, add it before finalizing.\n"
    )

    if has_images:
        # Relabel the raw prompt so the model understands its appearance words are
        # wrong fabrications — this prevents color/outfit bleed at the message level,
        # rather than relying solely on system-prompt DISCARD rules which LLMs often
        # ignore when conflicting text is salient in the user turn.
        user_content = (
            "Image request (reference photos attached above).\n\n"
            "Your final output must be a single unified paragraph — no section headers, no labels, no bullet points.\n\n"
            "Instructions for building the output:\n"
            "  (1) SCENE/POSE/SETTING — read from the text below. "
            "Extract only: scene, pose, action, setting, mood, background, lighting. "
            "The text was written WITHOUT photos. "
            "Every hair color, eye color, skin tone, and outfit word in it is a fabrication — "
            "do NOT copy any of those into your output.\n"
            "  (2) APPEARANCE — read from the reference photos. "
            "Your output MUST include ALL of the following — any omission makes the output wrong:\n"
            "    • HAIR: exact shade (e.g. 'pale seafoam mint-green, low saturation'), hairstyle, length\n"
            "    • EYES: write ONLY the three-axis form. "
            "Format: '[saturation] [hue family] eyes with [undertone], [saturation], [brightness]'. "
            "FORBIDDEN words to use alone: 'amber', 'golden', 'honey', 'warm amber', 'golden-amber' — "
            "these make Flux render vivid saturated eyes. Never write a brief color name before the three-axis version.\n"
            "    • EYELASHES: this field is MANDATORY and must be present. "
            "Write: '[hue], [darkness statement — state NOT black/dark brown if that is the case], [weight]'. "
            "Example: 'cool greenish-grey lashes, NOT black or dark brown, moderate contrast, fine and delicate'.\n"
            "    • SKIN TONE: exact shade\n"
            "    • OUTFIT: every garment piece with all 5 axes, joined by ' + '. "
            "When reading the photo, actively LOOK FOR and name these elements if present:\n"
            "      — Wide flared/circle-cut fabric hanging from the shoulders WITHOUT sleeves "
            "(not a coat — no sleeves, drapes freely) → this is a SHOULDER CAPE. Call it 'open-front [color] wide-hemmed shoulder cape'.\n"
            "      — Structured padded black shoulder caps at the shoulder attachment point of the cape "
            "→ these are SHOULDER EPAULETTES. List as a separate entry.\n"
            "      — White ruffled V-shaped or waterfall fabric at the chest/sternum level "
            "(NOT at the neck) with a gemstone pin at its center → this is a JABOT. "
            "Call it 'chest-level white V-shaped ruffled jabot, secured with a large green gemstone brooch'.\n"
            "      — Red lower-body garment with rows of gold buttons visible below the cape → RED SHORTS, NOT a vest.\n"
            "      List garments from outermost to innermost.\n"
            "    • ART STYLE: all rendering axes\n\n"
            f"Text: {raw_prompt}"
        )
    else:
        user_content = f"Image request: {raw_prompt}"
    messages_list = [{"role": "user", "content": user_content}]
    if has_images:
        print(f"[Ollama] enhance_image_prompt: using vision model with {len(reference_images)} reference image(s)")
    try:
        enhanced, *_ = await chat(
            messages_list,
            system_prompt=system,
            context_images=reference_images if has_images else None,
        )
        if enhanced:
            enhanced = _THINK_RE.sub("", enhanced)
            enhanced = _THINK_RE_UNCLOSED.sub("", enhanced).strip()
            if len(enhanced) > 5:
                print(f"[Ollama] Prompt enhanced: {enhanced}")
                return enhanced
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
        text, *_ = await chat(messages, system_prompt=system)
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
    language_sample: str = "",
) -> list:
    """Generate short follow-up suggestion buttons (max 80 chars each).

    language_sample should be a snippet of the bot's latest reply so the AI
    can detect and mirror the correct language automatically.
    """
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

    return []  # silently return no buttons rather than wrong-language fallbacks
