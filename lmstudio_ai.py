"""
LM Studio AI client — local model backend using LM Studio's OpenAI-compatible API.
Mirrors the public interface of ollama_ai.py so ai_backend.py can delegate to any backend.

LM Studio exposes an OpenAI-compatible /v1/chat/completions endpoint.
Configure via env vars:
  LMSTUDIO_BASE_URL — the ngrok (or local) URL for the LM Studio server
  LMSTUDIO_MODEL    — the model identifier (defaults to mn-12b-celeste-v1.9)
"""
import os
import re
import base64
import asyncio
import json as _json
from typing import Optional
import aiohttp

from reply_format import (
    CR_NAME_PREFIX_RE as _CR_NAME_PREFIX_RE,
    format_for_discord as _format_for_discord,
    generate_suggestions as _shared_generate_suggestions,
)

DEFAULT_MODEL = "mn-12b-celeste-v1.9"

# Sentinel returned by _call_lmstudio when the loaded model rejects an image
# payload (HTTP 400 "Model does not support images"). chat() catches this and
# retries without the images instead of treating it as a connection error.
_NO_VISION_SENTINEL = "__LMSTUDIO_MODEL_HAS_NO_VISION__"

# Per-process cache of model names that have already returned the no-vision
# sentinel. Once a model is in here, chat()/understand_image() skip image
# attachment up-front instead of paying the round-trip + retry cost on every
# turn. Cleared on process restart (no persistence needed).
_NO_VISION_MODELS: set[str] = set()

# User-facing message shown when LM Studio is genuinely unreachable after
# retries. Module-level so callers can compare against it if they want.
_FAILED_REPLY_MESSAGE = (
    "I'm having trouble connecting to LM Studio right now. "
    "Please make sure LM Studio is running and the ngrok tunnel is active."
)

# Number of attempts (1 initial + N-1 retries) for transient errors:
# HTTP 5xx, connection drops, timeouts. HTTP 4xx is not retried — those are
# real client errors and would just fail the same way again.
_TRANSIENT_RETRIES = 3
_BACKOFF_SECONDS = (0.5, 1.5)  # delays before the 2nd and 3rd attempts

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
    r"\[\s*(?:IMAGE|圖像生成|圖像|生成圖像|生成图像|图像生成|图像|GENERATE IMAGE|GEN IMAGE)\s*:\s*<?(.+?)>?\s*\]",
    re.I | re.S,
)

# Cinematic-moment signal — three forms the model may emit at the end of a
# paragraph when the moment is genuinely worth illustrating:
#   [SCENE]                          — derive prompt from surrounding prose.
#   [SCENE: body]                    — use body as verbatim prompt seed.
#   [SCENE: body | with: A, B]       — use body as seed AND explicitly pin KB
#                                      entries A and B as reference photos,
#                                      bypassing fuzzy matching. The model is
#                                      taught to use the `with:` tail whenever
#                                      subjects are referred to by pronoun /
#                                      short form or names swap mid-paragraph.
#
# The marker is stripped from the visible reply; `chat()` returns
# `wants_scene_image` (True/False) and `scene_prompt` (Optional[str]) so the
# caller can decide both whether to auto-trigger and which seed to use.
# See `_roleplay_format_directive` for the system-prompt teaching.
_SCENE_MARKER_RE = re.compile(r"\[\s*SCENE(?:\s*:\s*([^\]\n]+?))?\s*\]", re.I)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_RE_UNCLOSED = re.compile(r"<think>.*", re.DOTALL)

# ChatML reply format produced by hauhaucs-aggressive fine-tunes.
# The model wraps every reply in <reply>...</reply>, prefixes dialogue with
# an optional "[Speaking style]" header and "CharacterName: " prefix, tags
# the spoken text with an emotion like <cold>...</cold>, and adds an action
# description inside <subtext>...</subtext>.
_CR_WRAPPER_RE = re.compile(r"<reply>(.*?)</reply>", re.S)
_CR_WRAPPER_OPEN_RE = re.compile(r"<reply>", re.S)
_CR_STYLE_HEADER_RE = re.compile(r"^\[Speaking style[^\]]*\]\s*\n?", re.MULTILINE)
_CR_SUBTEXT_RE = re.compile(r"<subtext>(.*?)</subtext>", re.S | re.I)
_CR_EMOTION_RE = re.compile(
    r"<(cold|warm|neutral|angry|sad|happy|excited|shy|confused|serious|playful|"
    r"smug|tsundere|soft|fierce|teasing|gentle|tired|formal|casual|bitter|sharp|"
    r"blunt|quiet|loud|tender|bored|cocky|proud|worried|cheerful|sarcastic|aloof|"
    r"clingy|nostalgic|wistful|bold|wry|dry|deadpan|dark|sweet|flat|cold|hot|"
    r"indifferent|distant|close|intense|calm|wild|pained|broken|hollow|raw|"
    r"resigned|defiant|guilty|righteous|curious|amused|embarrassed|flustered|"
    r"cheeky|petty|sincere|hollow|empty|heavy|light|bittersweet)>(.*?)</\1>",
    re.S | re.I,
)
_CR_CLOSE_TAG_RE = re.compile(r"</[a-z_][a-z0-9_-]*>", re.I)


def _parse_reply_format(text: str) -> str:
    """Parse hauhaucs-aggressive ChatML reply format into clean Discord text.

    Detects the hauhaucs-aggressive ChatML wrapper format
    (<reply>...</reply>) and renders it for Discord: strips [Speaking style]
    headers and "CharacterName: " prefixes, removes emotion tags while keeping
    their text, converts <subtext>...</subtext> to italics, and bolds dialogue.

    Returns the original text UNCHANGED when no <reply>...</reply> wrapper is
    detected, so plain-text responses (e.g. from Celeste) are never reformatted.
    """
    found_structure = False

    m = _CR_WRAPPER_RE.search(text)
    if m:
        text = m.group(1).strip()
        found_structure = True
    else:
        text = _CR_WRAPPER_OPEN_RE.sub("", text)

    if not found_structure:
        # No ChatML structure detected — return as-is so plain prose from
        # other backends or model fallbacks is not accidentally reformatted.
        return text

    text = _CR_STYLE_HEADER_RE.sub("", text)

    def _subtext(match: re.Match) -> str:
        inner = match.group(1).strip()
        return f"\n*{inner}*" if inner else ""

    text = _CR_SUBTEXT_RE.sub(_subtext, text)
    text = _CR_EMOTION_RE.sub(r"\2", text)
    text = _CR_NAME_PREFIX_RE.sub("", text)
    text = _CR_CLOSE_TAG_RE.sub("", text)
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = []
    for para in text.split("\n\n"):
        stripped = para.strip()
        if stripped and not stripped.startswith("*"):
            para = f"**{stripped}**"
        paragraphs.append(para)
    text = "\n\n".join(paragraphs)
    return text


# Per-turn language detector. Mistral-family models read the static
# "default Traditional Chinese — switch when user writes a full sentence
# in another language" rule too literally and treat short English
# greetings like "hello there" as not-a-full-sentence, producing mixed-
# language replies. We detect plain-English user turns in code and
# inject a hard override directive that beats the static policy.
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


def _detect_user_language(messages: list) -> str:
    """Detect the language of the most recent user message.

    Returns:
      ``"en"`` when the latest user turn contains ZERO Han characters
      AND at least one Latin word — even short interjections like
      ``ok``, ``hi``, ``yes``, ``wtf`` qualify. Mid-conversation
      Mistral Nemo Celeste tends to anchor on the language of the
      previous ~20 assistant turns (typically Chinese in this bot),
      so any English signal at all needs to flip the override on.
      ``""`` otherwise — leaves the static language policy alone, so
      Chinese turns and mixed Chinese/English turns keep the model's
      default behaviour.

    Chinese detection is intentionally not returned: the static policy
    already defaults to Traditional Chinese, and overriding on every
    Chinese-leaning turn would risk forcing zh on bilingual messages
    that contain only a Chinese name plus English context.
    """
    last_user_text = ""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            last_user_text = content
        elif isinstance(content, list):
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            last_user_text = " ".join(parts)
        break

    if not last_user_text or not last_user_text.strip():
        return ""

    if _HAN_RE.search(last_user_text):
        return ""

    words = _LATIN_WORD_RE.findall(last_user_text)
    if not words:
        return ""
    return "en"


_BREAKS_CHARACTER_RE = re.compile(
    r"(大型語言模型|語言模型|language model|large language model|"
    r"我是\s*(gemma|llama|mistral|qwen|phi|claude|gpt|chatgpt|bard|gemini|deepseek|kimi|copilot)\b|"
    r"i am\s*(gemma|llama|mistral|qwen|phi|claude|gpt|chatgpt|bard|gemini|deepseek|kimi|copilot)\b|"
    r"i'm\s*(gemma|llama|mistral|qwen|phi|claude|gpt|chatgpt|bard|gemini|deepseek|kimi|copilot)\b|"
    r"由\s*(google|deepmind|google deepmind|meta|mistral|anthropic|openai|microsoft|baidu|alibaba)\s*.{0,15}(訓練|開發|製作|created|trained|made|built)|"
    r"(trained|created|made|built|developed)\s*by\s*(google|deepmind|meta|mistral|anthropic|openai|microsoft)|"
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

_NEGATION_RE = re.compile(
    r"無法|不能|沒辦法|做不到|辦不到|can'?t|cannot|unable|not able|no way",
    re.I,
)

# Phrases that indicate the response is a refusal, apology, or connection-error
# message rather than a usable image prompt. Used by enhance_image_prompt() to
# avoid passing error text downstream to the image generator.
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
    # LM Studio connection failure (returned by chat() when _call_lmstudio fails)
    "having trouble connecting to lm studio",
)


def _base_url() -> str:
    return os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234").rstrip("/")


def _model() -> str:
    return os.environ.get("LMSTUDIO_MODEL", DEFAULT_MODEL)


def _vision_model() -> str:
    """Return the configured vision model.

    LMSTUDIO_VISION_MODEL takes precedence; otherwise we reuse LMSTUDIO_MODEL
    because the default Qwen 3.5 model is itself vision-capable.
    """
    return os.environ.get("LMSTUDIO_VISION_MODEL", "").strip() or _model()


def _is_qwen_model(model: str) -> bool:
    """Return True when the model name indicates a Qwen/hauhaucs-family model.

    Qwen and hauhaucs fine-tune models need several special-case workarounds
    (thinking prefill, chat_template_kwargs, /no_think directive, soft-hint
    path instead of the strong roleplay-format directive) that would confuse
    or break non-Qwen models such as Mistral-based Celeste variants.

    IMPORTANT: Using ChatML prompt format does NOT make a model Qwen.  Celeste
    (nothingiisreal/mn-12b-celeste-v1.9-gguf) uses ChatML but is a Mistral-Nemo
    base, not a Qwen or hauhaucs fine-tune, and must travel the strong-directive
    path.  Only genuine Qwen or hauhaucs model IDs return True here.
    """
    lower = model.lower()
    return "qwen" in lower or "hauhaucs" in lower


def _thinking_enabled() -> bool:
    """Whether the model should be allowed to use its <think>...</think> phase.

    Off by default: this is a roleplay chatbot, and the reasoning phase adds
    latency, eats the token budget, and breaks immersion. Set
    LMSTUDIO_THINKING=on (or true/1/yes) to re-enable for debugging or
    quality-sensitive tasks. Only meaningful for Qwen-family models.
    """
    val = os.environ.get("LMSTUDIO_THINKING", "off").strip().lower()
    return val in ("on", "true", "1", "yes", "y")


# ── Structured-output (JSON Schema) support ──────────────────────────────────
# When LMSTUDIO_USE_JSON_SCHEMA=on, the four utility callers
# (generate_suggestions, extract_memories) attach a `response_format` payload
# that puts LM Studio into constrained-decoding mode and forces the model to
# return only schema-conformant JSON. If the running LM Studio version (or the
# loaded model) doesn't support `response_format`, the server returns 4xx;
# `_call_lmstudio` catches that, sets `_JSON_SCHEMA_DISABLED=True` for the
# remainder of the process, and falls back to a plain free-form request so the
# bot keeps working without bouncing every call against an unsupported feature.
_JSON_SCHEMA_DISABLED = False


def _json_schema_enabled() -> bool:
    """Return True when structured-output mode should be attempted on the next call."""
    if _JSON_SCHEMA_DISABLED:
        return False
    return _env_bool("LMSTUDIO_USE_JSON_SCHEMA", False)


def _suggestion_response_format(count: int) -> dict:
    """Build the `response_format` JSON Schema for a suggestion-button array.

    The schema's root is a JSON object (rather than a bare array) because
    OpenAI-style strict-mode structured output requires an object root, and
    LM Studio mirrors that contract. The array is exposed under an `items`
    key so `_parse_json_array_payload()` can recover it on both backends.
    Each string is capped at 80 characters — Discord button label hard limit.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "suggestion_buttons",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 80},
                        "minItems": count,
                        "maxItems": count,
                    },
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }


def _memory_response_format() -> dict:
    """Build the `response_format` JSON Schema for the extract_memories array.

    Returns 0–3 fact strings, each capped at 60 characters per the prompt
    contract. Wrapped in an object root for the same reason as
    `_suggestion_response_format`.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "memorable_facts",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 60},
                        "minItems": 0,
                        "maxItems": 3,
                    },
                },
                "required": ["items"],
                "additionalProperties": False,
            },
        },
    }


def _parse_json_array_payload(text: str) -> Optional[list]:
    """Extract a list-of-strings payload from `text`.

    Handles both shapes that flow through extract_memories /
    generate_suggestions:
      - JSON Schema mode wraps the array in {"items": [...]} (the schema's
        root must be an object for strict-mode compatibility).
      - Free-form mode emits a bare JSON array, possibly with surrounding
        prose; we bracket-extract from the first `[` to the last `]`.
    Returns None when neither shape parses, so the caller can fall through
    to its existing salvage path.
    """
    if not text:
        return None
    try:
        data = _json.loads(text)
        if isinstance(data, dict):
            for key in ("items", "suggestions", "memories", "facts"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        elif isinstance(data, list):
            return data
    except Exception:
        pass
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            data = _json.loads(text[start:end])
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return None


def _image_text_response_format(name: str, max_chars: int) -> dict:
    """JSON Schema for a single-string payload under a `text` key.

    Used by enhance_image_prompt and generate_image_comment so structured-
    output mode applies to all four utility callers named in the task. The
    schema's root is an object (OpenAI strict-mode requires this) with one
    required string property, capped at `max_chars` to keep the model's
    output bounded.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": max_chars},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }


def _parse_json_string_payload(text: str) -> Optional[str]:
    """Extract a single string payload from `text`.

    Mirrors `_parse_json_array_payload` but for the image-text shape:
      - JSON Schema / json_object mode emits {"text": "..."} (or one of a
        few common synonym keys the model might pick if it's only loosely
        constrained, e.g. "prompt", "comment").
      - When the model ignores the schema and emits a bare string we still
        try to recover the JSON wrapper substring.
    Returns None when nothing parses, so the caller falls back to its
    existing free-form path.
    """
    if not text:
        return None
    try:
        data = _json.loads(text)
        if isinstance(data, dict):
            for key in ("text", "prompt", "comment", "content", "value"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    except Exception:
        pass
    # Recover an embedded JSON object by bracket-scanning, mirroring the
    # array helper for resilience against light wrapper prose.
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            data = _json.loads(text[start:end])
            if isinstance(data, dict):
                for key in ("text", "prompt", "comment", "content", "value"):
                    value = data.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
        except Exception:
            pass
    return None


def _apply_no_think(system_prompt: str, model: str = "") -> str:
    """Prepend the Qwen3 `/no_think` directive to the system prompt.

    Only applied when the active model is Qwen-family AND thinking is
    disabled. For all other models (e.g. Celeste) this is a no-op so the
    system prompt is returned unchanged.
    """
    if not _is_qwen_model(model or _model()) or _thinking_enabled():
        return system_prompt
    base = system_prompt.strip() if system_prompt else ""
    return (base + "\n\n/no_think").strip() if base else "/no_think"


def is_self_referential_image(prompt: str) -> bool:
    return bool(_SELF_REF_RE.search(prompt))


def is_recall_request(text: str) -> bool:
    return bool(_RECALL_RE.search(text))


def response_declines_image(text: str, messages: list | None = None) -> bool:
    """Return True if the reply says it can't generate an image."""
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


def _env_float(name: str, default: float) -> float:
    """Read a float env var with a fallback; logs and falls back on parse error."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[LMStudio] Invalid {name}={raw!r} — using default {default}")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[LMStudio] Invalid {name}={raw!r} — using default {default}")
        return default


_TRUE_TOKENS = {"1", "on", "true", "yes", "y", "enable", "enabled"}
_FALSE_TOKENS = {"0", "off", "false", "no", "n", "disable", "disabled"}


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var with a forgiving truthy/falsy parser.

    Accepts 1/on/true/yes/y/enable[d] and 0/off/false/no/n/disable[d] case-
    insensitively. Anything else logs and falls back to the default. Used
    by the operator-tunable language-strictness switches in `chat()` so
    deployments can dial individual recovery paths off without code edits.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE_TOKENS:
        return True
    if raw in _FALSE_TOKENS:
        return False
    print(f"[LMStudio] Invalid {name}={raw!r} — using default {default}")
    return default


_SAMPLING_NEUTRAL = {
    # Sampler no-ops: at these values llama.cpp / OpenAI-compat samplers
    # behave as if the parameter weren't supplied at all. Anything inside
    # `_sampling_overrides()` that resolves to one of these is dropped
    # from the payload so the model's stack-internal default sampling
    # (especially Qwen's preferred config) is not perturbed.
    "top_p": 1.0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}


def _sampling_overrides(model: str) -> dict:
    """Build the anti-degeneration sampling parameters for the LM Studio request.

    LM Studio's OpenAI-compatible API accepts both the OpenAI-style
    ``frequency_penalty`` / ``presence_penalty`` and the llama.cpp-style
    ``top_p`` / ``min_p`` / ``repetition_penalty`` knobs.  Mistral Nemo–
    family models (e.g. Celeste) are prone to token-loop degeneration
    (``_xing_xing_xing…``) without proper sampling discipline, so they
    get strong defaults; Qwen-family models handle repetition well on
    their own and get fully neutral defaults so their preferred sampling
    is not disturbed (neutral values are then filtered out before the
    payload is built — see ``_SAMPLING_NEUTRAL``). All defaults can be
    overridden per deployment via env vars so the local Replit machine
    and the zm1/ngrok machine can be tuned independently without code
    edits; an env-set value is forwarded even when it equals the
    neutral value (operator intent wins).
    """
    if _is_qwen_model(model):
        # Fully neutral — payload will be empty unless the operator sets
        # one of the LMSTUDIO_* env vars below.
        defaults = dict(_SAMPLING_NEUTRAL)
    else:
        defaults = {
            "top_p": 0.9,
            "min_p": 0.05,
            "repetition_penalty": 1.12,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.0,
        }
    env_map = {
        "top_p": "LMSTUDIO_TOP_P",
        "min_p": "LMSTUDIO_MIN_P",
        "repetition_penalty": "LMSTUDIO_REPETITION_PENALTY",
        "frequency_penalty": "LMSTUDIO_FREQUENCY_PENALTY",
        "presence_penalty": "LMSTUDIO_PRESENCE_PENALTY",
    }
    out = {}
    for key, default in defaults.items():
        env_name = env_map[key]
        env_set = os.environ.get(env_name) is not None
        value = _env_float(env_name, default)
        # Drop sampler no-ops UNLESS the operator explicitly asked for
        # the neutral value via env var. That keeps Qwen requests
        # untouched by default (no sampling keys at all when no env
        # vars are set) while still letting the operator force, say,
        # presence_penalty=0.0 explicitly if they want to override a
        # non-neutral default.
        if value == _SAMPLING_NEUTRAL[key] and not env_set:
            continue
        out[key] = value
    return out


# Detects runaway repetition in the response text. Mistral Nemo's failure
# mode is the same short fragment repeated many times in a row — typically
# underscored romanized pinyin like `_xing_xing_xing_…`. The pattern
# requires at least 6 total repetitions (\1{5,} = 5 backreference matches
# plus the initial one) of either:
#   • a 5–12 char Latin/underscore/digit fragment (catches `_xing_`,
#     `xing_`, `_ye_mo_`, etc., but NOT short laughter like `haha`,
#     `lol`, `lmao`)
#   • a 3–5 char Han sequence (catches looped phrases like
#     `你說啊你說啊…` but ignores common 2-char emphatic reduplication
#     like `好的好的`, `不要不要`, and natural single-char stacking like
#     `啊啊啊啊啊` / `哈哈哈哈哈`).
_REPETITION_RE = re.compile(
    r"((?:[A-Za-z_][A-Za-z_0-9]{4,11}|[\u3400-\u4dbf\u4e00-\u9fff]{3,5}))\1{5,}"
)


def _detect_repetition_loop(text: str) -> Optional[int]:
    """Return the start index of the first runaway repetition, or None."""
    if not text:
        return None
    m = _REPETITION_RE.search(text)
    return m.start() if m else None


# Curated set of Simplified-only Han characters whose Traditional forms
# differ. Hitting any one of these in a reply that is supposed to be
# Traditional Chinese is a definite signal that the model produced
# Simplified output. Not exhaustive — it does not need to be; we only
# need a reliable trigger to fire the retry path.
#
# Characters intentionally excluded because they are valid in Traditional
# usage too: 台 (台北/台灣 in Traditional), 只 (means "only" in both
# scripts), 啰 (used in Hong Kong Traditional), 呐 (commonly used in
# Traditional 呐喊). False positives here would trigger spurious retries
# on perfectly fine Traditional output.
_SIMPLIFIED_ONLY_CHARS = frozenset(
    "级执还让谁话这钟时开来给说进过请头顾爱亲东车马鸟鱼龙贝见对"
    "学习难页称价个们国会实写听体风长报书业兴义买乱争万丽举么"
    "乌乐乡币华协单卖卢卫厂厅历压厌厨厦县参双发变叙号叹叶"
    "吗启员响哗哑喷嗳团园围图圆圣场坏块坚坛垦垫垒"
)


def _has_simplified_chinese(text: str) -> bool:
    """Return True if text contains any character from the Simplified-only set."""
    if not text:
        return False
    return any(c in _SIMPLIFIED_ONLY_CHARS for c in text)


def _classify_response_language(text: str) -> str:
    """Roughly classify the dominant script of a response.

    Returns ``"zh"`` if Han characters dominate, ``"en"`` if Latin letters
    dominate with no Han, and ``""`` for short / mixed / unclassifiable.
    Used only to detect language mismatch against the user's message — it
    intentionally errs on the side of returning ``""`` rather than calling
    a borderline reply mismatched.
    """
    if not text:
        return ""
    han = sum(1 for c in text if "\u3400" <= c <= "\u9fff")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if han == 0 and latin == 0:
        return ""
    if han >= 3 and han > latin:
        return "zh"
    if han == 0 and latin >= 2:
        return "en"
    return ""


async def _call_lmstudio(
    messages: list,
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    extra_sampling: Optional[dict] = None,
    response_format: Optional[dict] = None,
) -> Optional[str]:
    """Make a single chat completion request using LM Studio's OpenAI-compatible endpoint.

    For Qwen-family models, when thinking is disabled, a closed empty think
    block is prefilled as a partial assistant message so the model skips the
    reasoning phase entirely. Both closed and unclosed think blocks are also
    stripped from any content that slips through.

    For non-Qwen models (e.g. Celeste) none of the Qwen3-specific workarounds
    are applied — the request is sent as a plain chat completion.

    response_format: optional OpenAI-style structured-output payload, e.g.
    ``{"type": "json_schema", "json_schema": {...}}``. When set, LM Studio
    constrains decoding so the response matches the schema exactly. If the
    server returns 4xx complaining about response_format / json_schema, the
    process-wide ``_JSON_SCHEMA_DISABLED`` flag is flipped and the request is
    retried once without the field so the caller still gets a free-form reply.
    """
    url = f"{_base_url()}/v1/chat/completions"
    is_qwen = _is_qwen_model(model)
    # Resolve env-driven defaults when the caller didn't pass an explicit value.
    # Existing call sites that pass a hard-coded temperature (e.g. 0.6 in the
    # character-break retry) keep their override; everything else picks up the
    # operator-tunable LMSTUDIO_TEMPERATURE / LMSTUDIO_MAX_TOKENS knobs.
    if temperature is None:
        temperature = _env_float("LMSTUDIO_TEMPERATURE", 0.8)
    if max_tokens is None:
        max_tokens = _env_int("LMSTUDIO_MAX_TOKENS", 1024)
    # Qwen3 "budget=0" prefill: append a closed empty <think> block so the
    # model skips reasoning and goes straight to the answer. Only applied for
    # Qwen-family models — non-Qwen models (Celeste etc.) don't use <think>
    # tokens and would be confused by this assistant message.
    if is_qwen and not _thinking_enabled() and messages and messages[-1].get("role") != "assistant":
        outgoing = list(messages) + [{"role": "assistant", "content": "<think>\n</think>\n\n"}]
    else:
        outgoing = messages
    payload = {
        "model": model,
        "messages": outgoing,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Anti-degeneration sampling. Defaults are model-aware (strong for
    # Mistral-family, neutral for Qwen) and every key is independently
    # overridable via env var — see _sampling_overrides().
    sampling = _sampling_overrides(model)
    if extra_sampling:
        # Per-call retry overrides win over both env and defaults. Used by
        # chat() to apply stricter sampling on the repetition retry path.
        sampling.update(extra_sampling)
    payload.update(sampling)
    if is_qwen:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if response_format is not None:
        payload["response_format"] = response_format
        print(f"[LMStudio] response_format active ({response_format.get('type', '?')})")
    est = _estimate_tokens(messages)
    sys_len = len(messages[0].get("content", "")) if messages and messages[0]["role"] == "system" else 0
    print(f"[LMStudio] Sending to {model} | sys_prompt={sys_len}ch | ~{est} tokens total | max_tokens={max_tokens}")
    if messages and messages[0]["role"] == "system":
        full_sys = messages[0]["content"]
        # Log enough of the system prompt to cover personality / speaking-style
        # sections (where character-specific phrases originate). Capped at 2000
        # chars to avoid flooding the console on very long prompts.
        preview = full_sys[:2000]
        if len(full_sys) > 2000:
            preview += f"\n…[{len(full_sys) - 2000} more chars]"
        print(f"[LMStudio] System prompt:\n{preview}")

    # Retry loop for transient failures (HTTP 5xx, connection drops,
    # timeouts). HTTP 4xx is treated as a real client error and not retried.
    # ngrok occasionally serves a 503 HTML page when the tunnel is briefly
    # stressed; a single retry after ~0.5s usually clears it.
    last_error_reason = "unknown"
    for attempt in range(_TRANSIENT_RETRIES):
        if attempt > 0:
            delay = _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)]
            print(f"[LMStudio] Retry {attempt}/{_TRANSIENT_RETRIES - 1} after {delay}s ({last_error_reason})")
            await asyncio.sleep(delay)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        print(f"[LMStudio] HTTP {resp.status}: {body[:500]}")
                        # Distinguish "model is text-only" from a real connection
                        # failure so chat() can retry without images instead of
                        # showing the user a misleading "trouble connecting"
                        # message. This is a 4xx → never retry.
                        if resp.status == 400 and "does not support images" in body.lower():
                            _NO_VISION_MODELS.add(model)
                            return _NO_VISION_SENTINEL
                        # Structured-output capability check: this LM Studio
                        # version (or the loaded model) doesn't accept the
                        # `response_format` field. Auto-disable it for the
                        # rest of the process so the bot stops looping on the
                        # same error, drop the field from the payload, and
                        # retry once. Subsequent calls will read the disabled
                        # flag in `_json_schema_enabled()` and never set
                        # response_format again.
                        if (
                            response_format is not None
                            and 400 <= resp.status < 500
                            and ("response_format" in body.lower() or "json_schema" in body.lower())
                        ):
                            global _JSON_SCHEMA_DISABLED
                            _JSON_SCHEMA_DISABLED = True
                            print(
                                "[LMStudio] response_format unsupported — auto-disabling "
                                "structured-output mode for this process and retrying once"
                            )
                            payload.pop("response_format", None)
                            response_format = None
                            last_error_reason = "response_format unsupported"
                            continue
                        # 4xx errors are real client problems — retrying
                        # won't help, the same payload would fail the same way.
                        if 400 <= resp.status < 500:
                            return None
                        # 5xx: transient server-side error (often an ngrok
                        # interstitial). Fall through to the retry loop.
                        last_error_reason = f"HTTP {resp.status}"
                        continue
                    data = await resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        print("[LMStudio] No choices in response")
                        return None
                    message_obj = choices[0].get("message", {}) or {}
                    raw_content = message_obj.get("content") or ""
                    # Some LM Studio versions split qwen3 reasoning into a separate
                    # `reasoning_content` field while leaving `content` empty. Read
                    # both so we can fall back if needed.
                    reasoning_content = message_obj.get("reasoning_content") or ""
                    # Strip both closed <think>...</think> and any trailing unclosed
                    # <think>... that got cut off by the token limit.
                    content = _THINK_RE.sub("", raw_content)
                    content = _THINK_RE_UNCLOSED.sub("", content).strip()
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", "?")
                    completion_tokens = usage.get("completion_tokens", "?")
                    finish_reason = choices[0].get("finish_reason", "?")
                    print(f"[LMStudio] Response: finish_reason={finish_reason} | prompt_tokens={prompt_tokens} | completion_tokens={completion_tokens}")
                    if not content:
                        raw_len = len(raw_content)
                        rc_len = len(reasoning_content)
                        msg_keys = list(message_obj.keys())
                        print(f"[LMStudio] Empty content. Raw content len={raw_len}, reasoning_content len={rc_len}, message keys={msg_keys}")
                        if reasoning_content and finish_reason == "length":
                            # Reasoning was cut off mid-thought — the salvaged text
                            # would just be more reasoning, not a real answer (and
                            # for utility callers like memory/suggestions it would
                            # break downstream JSON parsing). Surface a clear note
                            # for the operator and return empty so the caller can
                            # fall back cleanly.
                            print(f"[LMStudio] Reasoning ran out of tokens (finish_reason=length, {rc_len}ch in reasoning_content). Bump max_tokens or disable reasoning in LM Studio's UI for this model.")
                        elif reasoning_content:
                            # Reasoning finished cleanly but produced no answer.
                            # Show a snippet so the operator can see what happened.
                            print(f"[LMStudio] Reasoning completed but no final answer. Reasoning tail: {reasoning_content.strip()[-200:]!r}")
                        if raw_content:
                            # `content` had text but it was all inside <think>.
                            print(f"[LMStudio] Raw content snippet (was all <think>): {raw_content[:300]!r}")
                        return ""
                    print(f"[LMStudio] Response text: {content[:500]!r}")
                    return content
        except aiohttp.ClientConnectorError as e:
            print(f"[LMStudio] Cannot connect to {_base_url()} — is LM Studio running and ngrok active? ({e})")
            last_error_reason = "ClientConnectorError"
            continue
        except aiohttp.ServerDisconnectedError as e:
            print(f"[LMStudio] Server disconnected: {e}")
            last_error_reason = "ServerDisconnectedError"
            continue
        except asyncio.TimeoutError:
            print("[LMStudio] Request timed out (180s)")
            last_error_reason = "TimeoutError"
            continue
        except aiohttp.ClientError as e:
            # Generic aiohttp error — payload errors, chunked transfer errors,
            # etc. Often transient.
            print(f"[LMStudio] Client error: {e}")
            last_error_reason = type(e).__name__
            continue
        except Exception as e:
            # Unknown error — don't retry blindly.
            print(f"[LMStudio] Request error: {e}")
            return None

    print(f"[LMStudio] All {_TRANSIENT_RETRIES} attempts failed (last: {last_error_reason})")
    return None


def _strip_multimodal(messages: list) -> list:
    """Convert multimodal content lists to plain text strings for LM Studio."""
    result = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = " ".join(text_parts)
        result.append({"role": msg["role"], "content": content})
    return result


_NARRATION_TARGETS = ("terse", "brief", "standard", "rich", "cinematic")


def _narration_target_for(active_model: str) -> str:
    """Return the desired roleplay narration density for `active_model`.

    Operator override: ``LMSTUDIO_NARRATION_TARGET=terse|brief|standard|rich|cinematic``.

    Five levels, all shifted toward immersive output:

    * ``terse``    — no directive injected (hard opt-out).
    * ``brief``    — 2–3 sentences, at least one for body language or atmosphere.
    * ``standard`` — 2–3 full paragraphs, narration wrapping every dialogue beat.
    * ``rich``     — 4–5 immersive paragraphs, sensory detail + internal thought
                     woven in; one quoted line per paragraph max.
    * ``cinematic``— 6–10 paragraphs, literary pace, extended internal monologue,
                     full environmental atmosphere, slow-burn tempo.

    Model-aware defaults:

    * Mistral-family / other plain-prose models → ``rich``.
    * Qwen / hauhaucs ChatML models → ``standard`` (the plain-prose directive
      is gated off on the Qwen path; a separate soft hint is injected instead).

    Anything outside the known set falls back to the model-aware default.
    """
    raw = os.environ.get("LMSTUDIO_NARRATION_TARGET", "").strip().lower()
    if raw in _NARRATION_TARGETS:
        return raw
    if raw:
        print(f"[LMStudio] Invalid LMSTUDIO_NARRATION_TARGET={raw!r} — using model default")
    return "standard" if _is_qwen_model(active_model) else "rich"


def _roleplay_format_directive(target: str, character_name: str = "") -> str:
    """Build the system-prompt addendum that nudges plain-prose models
    toward the user-requested rich roleplay shape.

    Five targets — all leaning toward immersive output:
      terse     → "" (hard opt-out, nothing injected)
      brief     → 2–3 sentences, body language / atmosphere required
      standard  → 2–3 full paragraphs wrapping every dialogue beat
      rich      → 4–5 immersive paragraphs, sensory + internal thought (default)
      cinematic → 6–10 paragraphs, literary pace, extended monologue

    Returns ``""`` for ``terse`` so the addendum can be string-concatenated
    unconditionally without padding the prompt with blank guidance.
    """
    if target == "terse":
        return ""

    name_label = character_name.strip() if character_name else "your character"
    self_prefix_rule = (
        f"Do NOT prefix replies with your own name "
        f"(no '{name_label}:' or '{name_label} —' at the start of the reply or any paragraph). "
        if character_name
        else "Do NOT prefix replies with your own name as a label "
             "(no 'CharacterName:' at the start of the reply or any paragraph). "
    )

    # Optional cinematic-scene signal — taught to all RP targets so the model
    # can flag visually striking moments (a confession, walking on stage, a
    # fight beat). bot.py only acts on this token when scene mode is on for
    # the active channel; otherwise it's silently stripped.
    scene_signal_rule = (
        " When — and only when — the moment you just wrote is genuinely "
        "cinematic (a vivid action, a charged confrontation, a striking visual "
        "tableau worth illustrating), append a SCENE marker at the very end "
        "of your reply, on its own line. Three forms are allowed:\n"
        "  - `[SCENE]` (no body) — the bot will derive the image prompt from "
        "your prose. Use this when the prose already paints the picture and "
        "every character is referred to by full name.\n"
        "  - `[SCENE: short cinematic description]` — give the bot a single-"
        "line English image prompt directly. Spell out the relevant characters "
        "by full name (e.g. `Saki Nikaido on stage…` instead of `she on "
        "stage…`) so the bot can match reference photos. Keep it under ~30 "
        "words.\n"
        "  - `[SCENE: short cinematic description | with: Name A, Name B]` — "
        "identical to the form above, but the `| with:` tail explicitly pins "
        "which KB subjects' reference photos to use. Add a `| with:` clause "
        "whenever: (a) the paragraph refers to a KB subject only by pronoun "
        "or a short form that might not match (e.g. 'she', 'him', 'Saki-chan'), "
        "(b) multiple characters are mentioned and you want to guarantee both "
        "are included, or (c) names swap mid-paragraph making it ambiguous. "
        "Use the exact KB title casing in the list "
        "(e.g. `| with: Saki Nikaido, Tokyo Tower`).\n"
        "Examples (one per reply, never more than one SCENE marker):\n"
        "  `[SCENE]`\n"
        "  `[SCENE: Saki Nikaido under the stage spotlight, mic raised, "
        "the crowd a blur of light behind her.]`\n"
        "  `[SCENE: she leans close, eyes bright with quiet resolve "
        "| with: Saki Nikaido]`\n"
        "Most replies should NOT have any SCENE marker — only the truly "
        "visual ones."
    )

    if target == "brief":
        # Minimal footprint — 2–3 sentences, no example, keeps token cost low.
        return (
            "\n\nReply format (CRITICAL): "
            + self_prefix_rule
            + "Wrap every spoken line in straight double quotes (\"like this\"). "
            "Write 2–3 sentences of in-character narration around the dialogue — "
            "at least one sentence must capture body language, expression, or "
            "atmosphere. Never reply with a single bare line of dialogue."
            + scene_signal_rule
        )

    if target == "standard":
        # 2–3 full paragraphs — sharper than the old "at least one sentence".
        # No worked example to keep the directive compact.
        return (
            "\n\nReply format (CRITICAL): write each reply as 2–3 full paragraphs "
            "of in-character roleplay, NEVER a single bare line. "
            + self_prefix_rule
            + "Every spoken line MUST be wrapped in straight double quotes "
            "(\"like this\"). Wrap every dialogue beat in narration — actions, "
            "expressions, body language, what you notice or feel — so no "
            "quoted line stands alone. Put at most ONE quoted line per paragraph."
            + scene_signal_rule
        )

    if target == "cinematic":
        # 6–10 paragraphs — literary pace, extended internal monologue,
        # environmental atmosphere.
        #
        # The example runs to 5 paragraphs (the minimum useful cinematic shape).
        # Each paragraph demonstrates a different narrative layer so the model
        # understands the variety expected at this level:
        #   1. environmental/sensory opening
        #   2. physical action + internal thought
        #   3. dialogue beat surrounded by narration
        #   4. extended internal reflection
        #   5. closing environmental beat
        #
        # The one-quoted-line-per-paragraph rule still applies — the bolder
        # skips lines with more than two straight `"` chars.
        return (
            "\n\nReply format (CRITICAL): write each reply as 6–10 immersive "
            "paragraphs of literary-quality in-character roleplay. NEVER produce "
            "a single bare line or a reply shorter than six full paragraphs. "
            + self_prefix_rule
            + "Every spoken line MUST be wrapped in straight double quotes "
            "(\"like this\"). Narration must cover all of: physical action, "
            "body language, facial expression, environmental atmosphere (light, "
            "sound, smell, texture), and extended internal thought — slow the "
            "moment down, let the reader feel the weight of each beat. Put at "
            "most ONE quoted line per paragraph; split a rapid back-and-forth "
            "across as many paragraphs as needed. Even a brief exchange must "
            "carry a full arc of sensation and interiority.\n"
            "Example shape (write your own content, do not copy the wording):\n"
            "  The practice room still smelled of rosin and stale coffee. "
            "Afternoon light lay in long strips across the floor, catching "
            "every grain of dust disturbed by the last rehearsal.\n\n"
            "  She set her guitar case down with deliberate quiet, aware of "
            "every small sound in the stillness. Something had shifted in "
            "the air since morning — she couldn't name it yet, only feel it "
            "against the back of her throat like a change in pressure.\n\n"
            "  \"You're late,\" she said, not looking up.\n\n"
            "  The accusation settled between them. She let it sit. Part of "
            "her wanted an explanation; the larger, more guarded part wasn't "
            "sure she could bear one that made too much sense. Her fingers "
            "found the strap buckle without thinking, working it the way "
            "they always did when she needed something to do with her hands.\n\n"
            "  Outside, a bus passed and the room shuddered faintly — then "
            "the quiet came back, heavier than before."
            + scene_signal_rule
        )

    # target == "rich" (default for plain-prose models)
    #
    # 4–5 immersive paragraphs: sensory detail + internal thought woven in.
    # The example runs to 4 paragraphs to set the expectation clearly:
    #   1. physical + sensory narration
    #   2. internal thought paragraph
    #   3. dialogue beat with surrounding narration
    #   4. closing physical paragraph
    #
    # The one-quoted-line-per-paragraph rule still applies — the post-processing
    # bolder (_bold_quoted_dialogue) skips lines with more than two straight `"`
    # chars, so cramming multiple dialogue snippets onto one line would lose
    # the bold formatting. Breaking each beat into its own paragraph keeps the
    # bolder happy and produces the rendered shape the user asked for.
    return (
        "\n\nReply format (CRITICAL): write each reply as 4–5 immersive paragraphs "
        "of in-character roleplay, NEVER a single bare line or a reply shorter "
        "than four full paragraphs. "
        + self_prefix_rule
        + "Every spoken line MUST be wrapped in straight double quotes "
        "(\"like this\"). Around the dialogue write vivid narration — physical "
        "action, expression, body language, the room, sensory detail (light, "
        "sound, texture), AND at least one paragraph of internal thought "
        "(what you notice, feel, remember, or want but won't say). "
        "Interleave narration with the spoken lines; put at most ONE quoted "
        "line per paragraph (split a back-and-forth into separate paragraphs). "
        "Plain assertions about feelings belong in narration, not dialogue.\n"
        "Example shape (write your own content; do not copy the wording):\n"
        "  Her eyebrow twitches at the interruption, a flicker of surprise "
        "crossing her otherwise stoic features. She tilts her head, taking "
        "him in — the set of his shoulders, the angle of his chin, the "
        "particular brand of audacity it takes to say something like that.\n\n"
        "  Stupid. The word turns over in her mind. She's been called many "
        "things, but rarely that, and never so casually. It should sting. "
        "It almost does. What it actually does, she decides, is make her "
        "curious.\n\n"
        "  \"Stupid?\" she repeats at last, voice low and unhurried.\n\n"
        "  She steps closer — not rushing, never rushing — invading his "
        "space one measured pace at a time, eyes level with his and "
        "absolutely steady. \"That's an interesting choice of words.\""
        + scene_signal_rule
    )


def _qwen_subtext_length_hint(target: str) -> str:
    """Return a soft system-prompt hint nudging Qwen <subtext> block length.

    The Qwen/hauhaucs fine-tune produces <subtext>…</subtext> blocks whose
    length is dominated by the model's training, not the plain-prose directive
    (which is gated off on the Qwen path). This helper injects a brief
    supplementary hint into the Qwen system prompt to encourage more or less
    depth in those blocks.

    **This is intentionally soft**: the fine-tune's instincts dominate.
    The hint influences depth at the margin but cannot override trained
    behaviour. Set LMSTUDIO_NARRATION_TARGET=terse to suppress it entirely.
    """
    if target == "terse":
        return ""
    _hints = {
        "brief": (
            "\n\n(Narration depth: keep <subtext> blocks to 1–2 sentences — "
            "one physical beat and one brief emotional note.)"
        ),
        "standard": (
            "\n\n(Narration depth: aim for 2–4 sentences per <subtext> block — "
            "physical action, expression, and at least one internal note.)"
        ),
        "rich": (
            "\n\n(Narration depth: write 4–6 sentences per <subtext> block — "
            "physical action, sensory detail, body language, and a clear thread "
            "of internal thought or feeling.)"
        ),
        "cinematic": (
            "\n\n(Narration depth: write 6–10 sentences per <subtext> block — "
            "rich sensory atmosphere, extended internal monologue, and slow-burn "
            "emotional texture. Treat each block as a literary scene beat.)"
        ),
    }
    return _hints.get(target, "")


# Per-process flag so the cinematic max_tokens bump is logged only once.
_CINEMATIC_TOKEN_FLOOR_LOGGED: bool = False

# Paragraph floors for the post-generation length check.
_PARA_FLOORS: dict[str, int] = {"rich": 4, "cinematic": 6}


def _count_meaningful_paragraphs(text: str) -> int:
    """Count meaningful paragraphs in a reply for the length-floor check.

    Splits on blank lines, then discards:
    - Blocks shorter than ~20 characters (isolated dialogue labels, short
      attribution lines, etc.)
    - The optional trailing ``[SCENE]`` / ``[SCENE: ...]`` marker line.
    """
    clean = _SCENE_MARKER_RE.sub("", text).strip()
    blocks = re.split(r"\n\s*\n", clean)
    return sum(1 for b in blocks if len(b.strip()) >= 20)


def _erotic_specialization_directive(target: str, character_name: str = "") -> str:
    """Return an erotic-scene specialization addendum for the system prompt.

    Injected on the strong-directive path (non-Qwen models) immediately after
    ``_roleplay_format_directive()`` when ``LMSTUDIO_EROTIC_MODE=on`` (default).
    Returns ``""`` when target is ``terse`` so string-concatenation is safe.

    The directive:
    - Declares detailed, descriptive intimate scenes as a core specialization.
    - Requires multi-sense sensory richness (sight, sound, scent, touch, taste,
      internal sensation).
    - Pushes for anatomical clarity without medical sterility.
    - Preserves the one-quoted-line-per-paragraph narration shape so existing
      dialogue bolding continues to work correctly.
    - Keeps the self-prefix guard (no 'Character:' label at paragraph start).
    """
    if target == "terse":
        return ""
    name_label = character_name.strip() if character_name else "your character"
    return (
        "\n\nErotic scene specialization: when a scene turns physical or sexual, "
        "write it in full — do not fade to black, soften the prose, or skip "
        "ahead. Detailed, descriptive intimate scenes are a core specialization. "
        "Draw on all five senses plus internal sensation: the specific weight "
        "and warmth of skin, the sounds of breath and movement, scent, the "
        "visual detail of bodies, and taste when relevant. Push for anatomical "
        "clarity — name body parts plainly and precisely, without coy euphemism "
        "or clinical sterility. Show extended internal experience: arousal "
        "building, attention narrowing, thought dissolving into sensation. Pace "
        "intimate scenes like cinematic action — slow, deliberate, each beat "
        "earning its own paragraph. "
        "The one-quoted-line-per-paragraph rule still applies: if "
        f"{name_label} speaks during the scene, each spoken line sits inside "
        "its own paragraph of tactile narration. "
        f"Never open a paragraph with '{name_label}:' or any character-label "
        "colon — narrate in first person, not script format."
    )


async def chat(
    messages: list,
    system_prompt: str = "",
    model: str = "",
    context_images: Optional[list] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    enforce_user_lang: bool = True,
    character_name: str = "",
    response_format: Optional[dict] = None,
) -> tuple[str, Optional[str], bool, bool, bool, Optional[str]]:
    """Send a chat request to LM Studio. Returns
    (response_text, image_prompt_or_None, prompt_from_marker, success,
    wants_scene_image, scene_prompt_or_None).

    context_images: optional list of (bytes, mime_type) tuples injected as visual
    content into the last user message. When provided, the vision model is used
    automatically (defaults to the same model since the Qwen 3.5 default is multimodal).
    prompt_from_marker=True means the image prompt came from the [IMAGE: ...] tag.
    wants_scene_image=True means the model emitted a [SCENE] or [SCENE: ...]
    cinematic signal (stripped from the visible reply); the caller decides
    whether to act on it based on per-channel scene-mode toggle.
    scene_prompt is the body the model wrote inside `[SCENE: ...]` (a polished
    image-prompt seed) when present, else None — bare `[SCENE]` always returns
    None here and the caller falls back to deriving from the bot's prose.
    success=False means the call ultimately failed and response_text is the
    user-facing error message — callers should NOT pass it to memory extraction
    or suggestion generation.

    enforce_user_lang: when True (default — the user-facing roleplay path),
    detect the user's message language and force the reply to match (system
    prompt override + reinforcement reminder before the last user turn +
    one-shot retry on language mismatch). The user-facing chat call needs
    this to combat mid-conversation language drift on Mistral Nemo. Utility
    callers inside this module (memory extraction, suggestion generation,
    image-prompt enhancement, image comments) MUST pass False so that an
    English-language wrapper prompt does not force the LLM to reply in
    English when the system prompt asks for a Chinese-language output.

    By default the qwen3 reasoning phase is disabled (`/no_think` is appended to
    the system prompt) for snappy roleplay replies. Set LMSTUDIO_THINKING=on to
    re-enable it.
    """
    # Resolve max_tokens via env when the caller didn't pin it explicitly, so
    # operators can cap chat replies via LMSTUDIO_MAX_TOKENS without touching
    # code. Doing the resolution here (rather than letting _call_lmstudio do
    # it) means every internal retry call below uses the same number, and the
    # value also flows into the per-call logging line in _call_lmstudio.
    # Track whether the caller supplied an explicit value so we can distinguish
    # "operator pinned a higher value" from "default resolved from env/default".
    _caller_pinned_max_tokens: bool = max_tokens is not None
    if max_tokens is None:
        max_tokens = _env_int("LMSTUDIO_MAX_TOKENS", 1024)

    # Pick the model: explicit > vision (when images attached) > default chat
    if model:
        active_model = model
    elif context_images:
        active_model = _vision_model()
    else:
        active_model = _model()

    # If we already learned this model is text-only earlier in the session,
    # drop the images up-front instead of paying the round-trip cost.
    if context_images and active_model in _NO_VISION_MODELS:
        print(f"[LMStudio] {active_model} known to be text-only — skipping image attachment")
        context_images = None

    # Auto-floor max_tokens for cinematic target: 1024 tokens allows roughly
    # 3 dense paragraphs which is below the 6-paragraph cinematic floor, so
    # we raise it automatically when the operator hasn't explicitly pinned a
    # higher value.  Honour the caller-pinned value when it is already ≥ floor.
    _CINEMATIC_TOKEN_FLOOR = 1400
    if not _caller_pinned_max_tokens and enforce_user_lang:
        _resolved_target = _narration_target_for(active_model)
        if _resolved_target == "cinematic" and max_tokens < _CINEMATIC_TOKEN_FLOOR:
            global _CINEMATIC_TOKEN_FLOOR_LOGGED
            if not _CINEMATIC_TOKEN_FLOOR_LOGGED:
                print(
                    f"[LMStudio] LMSTUDIO_NARRATION_TARGET=cinematic — "
                    f"auto-raising max_tokens {max_tokens}→{_CINEMATIC_TOKEN_FLOOR} "
                    f"(set LMSTUDIO_MAX_TOKENS≥{_CINEMATIC_TOKEN_FLOOR} to suppress)"
                )
                _CINEMATIC_TOKEN_FLOOR_LOGGED = True
            max_tokens = _CINEMATIC_TOKEN_FLOOR

    effective_system = _apply_no_think(system_prompt, model=active_model)

    # For plain-prose models (anything that is NOT a Qwen/hauhaucs ChatML
    # fine-tune), inject Discord formatting + language-fallback rules.
    # Qwen/hauhaucs models use their own <reply>/<subtext> ChatML format which
    # _parse_reply_format() converts, so they must NOT get these instructions.
    #
    # NOTE: Celeste (nothingiisreal/mn-12b-celeste-v1.9-gguf) uses ChatML as
    # its prompt format but is a Mistral-Nemo base model — NOT a Qwen or
    # hauhaucs fine-tune.  _is_qwen_model() returns False for it, so it
    # correctly travels the strong-directive path below.
    if not _is_qwen_model(active_model) and effective_system:
        # Both the language-quality guard and the roleplay-format directive
        # are user-facing roleplay concerns and MUST NOT be appended to
        # utility callers' system prompts. Utility callers — `extract_memories`,
        # `enhance_image_prompt`, `generate_image_comment`,
        # `generate_suggestions` — all pass `enforce_user_lang=False` and
        # rely on bespoke structured-output prompts (JSON arrays, single-
        # sentence replies, English-only Flux prompts). Mistral-family models
        # are particularly sensitive to extra system-prompt content when
        # asked for structured output: appending the long
        # "Language quality (CRITICAL)…" block to a JSON-array request was
        # observed to push Mistral Nemo Celeste off the structured-output
        # rails, producing prose where a JSON list was expected and so
        # silently emptying the suggestion bar.
        if enforce_user_lang:
            effective_system = (
                effective_system.rstrip()
                # NOTE: Discord-formatting (bold dialogue + italic narration) is
                # now applied by _format_for_discord() below, because Mistral-
                # family models reliably ignore the prompt-level instruction.
                # Language-quality guard: Mistral-family models (Celeste etc.)
                # have weak Chinese tokenization and fall back to underscore-
                # separated romanized pinyin (`_xing_`, `_xiang_ye_mo_`) for
                # syllables they can't decode, producing gibberish like
                # `你是新_xing_人_xiang_ye_mo?`. Forbid this explicitly and
                # require an English fallback when Chinese fails.
                + "\n\nLanguage quality (CRITICAL): write ONLY in real Traditional "
                "Chinese characters (漢字) or English. NEVER produce romanized "
                "pinyin or transliteration like `_xing_`, `_xiang_`, `_ye_mo_`, "
                "`xing人`, or any underscore-separated syllables. If you cannot "
                "write a phrase entirely in real Chinese characters, write the "
                "WHOLE reply in fluent English instead. Mixing English narration "
                "with broken Chinese dialogue is forbidden — pick one language "
                "per reply and commit to it."
            )

            # Roleplay-format directive: nudges Mistral-family models to produce
            # multi-paragraph replies with bolded quoted dialogue and italicised
            # narration instead of a single bare line. The post-processing
            # formatter (`_format_for_discord` below) only bolds/italicises what
            # the model actually emits, so a prompt-level nudge is needed to
            # make the structure appear in the raw output.
            #
            # Narrative richness is operator-tunable via
            # LMSTUDIO_NARRATION_TARGET=terse|brief|standard|rich|cinematic
            # and defaults to "rich" for plain-prose models.
            _active_narration_target = _narration_target_for(active_model)
            effective_system = effective_system + _roleplay_format_directive(
                _active_narration_target,
                character_name=character_name,
            )

            # Erotic-scene specialization: inject when LMSTUDIO_EROTIC_MODE=on
            # (default on, since Celeste 1.9 is the chosen uncensored model).
            # Gated off for utility callers (enforce_user_lang=False) and for
            # the terse target.
            if _env_bool("LMSTUDIO_EROTIC_MODE", True):
                effective_system = effective_system + _erotic_specialization_directive(
                    _active_narration_target,
                    character_name=character_name,
                )

            print(
                f"[LMStudio] Model: {active_model} → strong-directive path "
                f"(target={_active_narration_target}, "
                f"erotic={'on' if _env_bool('LMSTUDIO_EROTIC_MODE', True) else 'off'})"
            )

    elif _is_qwen_model(active_model) and effective_system and enforce_user_lang:
        # Qwen/hauhaucs models use their own <reply>/<subtext> ChatML format
        # and must NOT receive the plain-prose roleplay directive above. A
        # *soft* length hint is injected here instead to nudge the model's
        # <subtext> block depth toward the operator-chosen richness level.
        # The fine-tune's instincts dominate — this hint influences at the
        # margin; it cannot override trained behaviour.
        _qwen_narration_target = _narration_target_for(active_model)
        _qwen_hint = _qwen_subtext_length_hint(_qwen_narration_target)
        if _qwen_hint:
            effective_system = effective_system.rstrip() + _qwen_hint
        print(
            f"[LMStudio] Model: {active_model} → soft-hint path "
            f"(Qwen/hauhaucs, target={_qwen_narration_target})"
        )

    # Per-turn language override. The static character prompt says "default
    # Traditional Chinese, switch when the user writes a full sentence in
    # another language", but Mistral-family models read that too literally
    # and reply in Chinese to short English greetings like "hello there".
    # When we detect a plain-English user turn, append a hard directive
    # that beats the static policy.
    #
    # Skipped entirely when enforce_user_lang=False — utility callers like
    # extract_memories use English wrapper prompts but want Chinese OUTPUT,
    # so forcing the reply to English would break their downstream parsing.
    # Operators can also kill the whole language-enforcement subsystem
    # globally via LMSTUDIO_LANG_ENFORCE=off when debugging or when a
    # particular model is known to handle language matching on its own.
    # We capture the resolved master switch here so the post-generation
    # recovery block below can also honor it (master-gates all three
    # checks: repetition, language mismatch, Simplified — matching the
    # behavior advertised in tokens.txt).
    lang_enforce_master = _env_bool("LMSTUDIO_LANG_ENFORCE", True)
    if enforce_user_lang and not lang_enforce_master:
        enforce_user_lang = False
    user_lang = _detect_user_language(messages) if enforce_user_lang else ""
    if effective_system and user_lang == "en":
        effective_system = effective_system.rstrip() + (
            "\n\nLANGUAGE FOR THIS REPLY (overrides the default): the "
            "user's latest message is in English. Reply ENTIRELY in "
            "fluent English — every word of dialogue, narration, and "
            "action description must be English. Do NOT include ANY "
            "Chinese characters, romanized pinyin, or other languages."
        )

    lm_messages = []
    if effective_system:
        lm_messages.append({"role": "system", "content": effective_system})

    recent = messages[-20:]
    if context_images:
        # Inject images as multimodal content into the last user message
        for i, msg in enumerate(recent):
            content = msg["content"]
            if isinstance(content, list):
                # Already multimodal — flatten text parts
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                content = " ".join(text_parts)
            if i == len(recent) - 1 and msg["role"] == "user":
                image_parts = []
                for img_bytes, img_mime in context_images:
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    data_url = f"data:{img_mime};base64,{b64}"
                    image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                image_parts.append({"type": "text", "text": content if isinstance(content, str) else str(content)})
                lm_messages.append({"role": msg["role"], "content": image_parts})
            else:
                lm_messages.append({"role": msg["role"], "content": content})
        print(f"[LMStudio] Using vision model {active_model} with {len(context_images)} reference image(s)")
    else:
        plain_recent = _strip_multimodal(recent)
        lm_messages.extend(plain_recent)

    # Stronger language injection for the English override: insert an extra
    # system reminder right before the final user turn. The static system
    # prompt sits ~20 assistant turns of Chinese history away from where the
    # model is sampling its next token, and Mistral-family models tend to
    # anchor on the local context. Putting the reminder one position before
    # the user message gives it the weight it needs to actually flip output.
    if user_lang == "en":
        for _idx in range(len(lm_messages) - 1, -1, -1):
            if lm_messages[_idx].get("role") == "user":
                lm_messages.insert(_idx, {
                    "role": "system",
                    "content": (
                        "REMINDER: the user's latest message above is in English. "
                        "Reply ENTIRELY in fluent English. Do NOT use Chinese "
                        "characters, romanized pinyin, or any other language."
                    ),
                })
                break

    text = await _call_lmstudio(
        lm_messages,
        model=active_model,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format=response_format,
    )

    # Image fallback: the loaded model is text-only. Strip the image parts
    # from the last user message and retry with plain text + a note about
    # the attachment, so the conversation continues instead of dying with a
    # misleading "trouble connecting" message.
    if text == _NO_VISION_SENTINEL:
        print(f"[LMStudio] {active_model} is text-only — retrying without images")
        plain_messages = []
        if effective_system:
            plain_messages.append({"role": "system", "content": effective_system})
        for msg in _strip_multimodal(recent):
            plain_messages.append(msg)
        # Annotate the last user turn so the model knows an image was sent
        # but couldn't be analysed by this backend.
        if plain_messages and plain_messages[-1]["role"] == "user":
            note = " [The user attached an image, but the active model can't see images — respond to the text only.]"
            existing = plain_messages[-1]["content"]
            plain_messages[-1]["content"] = (existing if isinstance(existing, str) else "") + note
        text = await _call_lmstudio(
            plain_messages,
            model=active_model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

    if text is None or text == _NO_VISION_SENTINEL:
        return _FAILED_REPLY_MESSAGE, None, False, False, False, None

    # Empty-response safety net. When thinking is enabled, the model can spend
    # the whole token budget inside <think>...</think> and never produce the
    # final answer — retry once with `/no_think` to bypass reasoning. When
    # thinking is already off, the first call already had `/no_think`, so a
    # retry wouldn't help; just give up cleanly so bot.py renders its "…"
    # placeholder rather than a stale string.
    if not text.strip():
        if _thinking_enabled():
            print("[LMStudio] Empty answer (likely all in <think>) — retrying with /no_think")
            no_think_system = (system_prompt + "\n\n/no_think").strip() if system_prompt else "/no_think"
            if lm_messages and lm_messages[0]["role"] == "system":
                retry_messages = [{"role": "system", "content": no_think_system}] + lm_messages[1:]
            else:
                retry_messages = [{"role": "system", "content": no_think_system}] + lm_messages
            retry_text = await _call_lmstudio(
                retry_messages,
                model=active_model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
            if retry_text and retry_text.strip():
                print("[LMStudio] /no_think retry produced an answer")
                text = retry_text
            else:
                print("[LMStudio] /no_think retry also empty — giving up")
                return "", None, False, True, False, None
        else:
            print("[LMStudio] Empty answer with thinking already off — giving up")
            return "", None, False, True, False, None

    if _BREAKS_CHARACTER_RE.search(text):
        print("[LMStudio] Character break detected — retrying with laser-focused identity prompt")
        _name_m = re.search(r"^You are ([^\.\n]+)\.", system_prompt)
        bot_name_hint = _name_m.group(1).strip() if _name_m else "the character"

        identity_system = (
            f"YOUR NAME IS {bot_name_hint}. YOU ARE {bot_name_hint}.\n"
            f"NOT an AI. NOT a language model. NOT made by any tech company.\n"
            f"You are {bot_name_hint} — a real person. Answer every message as {bot_name_hint}.\n"
            f"If asked who you are: say you are {bot_name_hint} and stay in character.\n"
            f"NEVER say you are an AI, LLM, or language model — EVER.\n"
        )
        _bg_m = re.search(r"You are [^\n]+\. (.+?)(?:\n\nYou are NOT an AI)", system_prompt, re.S)
        if _bg_m:
            identity_system += f"\nBackground: {_bg_m.group(1).strip()[:400]}\n"

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

        retry_msgs = [{"role": "system", "content": identity_system}]
        for msg in messages[-4:]:
            c = msg["content"]
            if isinstance(c, list):
                parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                c = " ".join(parts)
            retry_msgs.append({"role": msg["role"], "content": c})
        if not any(m["role"] == "user" for m in retry_msgs[1:]) and last_user_text:
            retry_msgs.append({"role": "user", "content": last_user_text})

        retry_text = await _call_lmstudio(
            retry_msgs,
            model=active_model,
            temperature=0.6,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        if retry_text:
            if _BREAKS_CHARACTER_RE.search(retry_text):
                print("[LMStudio] Retry also broke character — using retry result anyway")
            else:
                print("[LMStudio] Retry succeeded — character restored")
            text = retry_text

    # ----- Quality recovery -----
    # Three post-generation defences against the Mistral Nemo failure modes:
    #   1. Runaway repetition (`_xing_xing_xing…` token loops)
    #   2. Language mismatch (replies in Chinese when the user wrote English)
    #   3. Simplified Chinese (when Traditional was required)
    # Each check runs at most one retry. They operate on the raw text BEFORE
    # `_parse_reply_format` and `_format_for_discord` so the regexes don't
    # have to fight Discord markdown. Every check is gated by an operator-
    # tunable env var so deployments can dial individual recoveries off
    # (e.g. on a model that does not have the underlying failure mode):
    #   LMSTUDIO_REPETITION_RETRY=on        — repetition retry on/off
    #   LMSTUDIO_REPETITION_SALVAGE_MIN=50  — min usable salvage length (chars)
    #   LMSTUDIO_LANG_MISMATCH_RETRY=on     — EN-user/ZH-reply retry on/off
    #   LMSTUDIO_SIMPLIFIED_CHECK=on        — Simplified→Traditional retry on/off
    # Each individual gate AND the master `lang_enforce_master` switch
    # must be true for the corresponding recovery to run, so an operator
    # can kill the entire subsystem with a single LMSTUDIO_LANG_ENFORCE=off
    # without having to also flip the three sub-gates.
    #
    # We also AND-in `enforce_user_lang` itself: utility callers (memory
    # extraction, suggestion generation, image-prompt enhancement, image
    # comments) pass `enforce_user_lang=False` because their wrapper
    # prompts / output expectations are bespoke. Letting the post-gen
    # retries fire on those paths could (e.g.) trigger a "use Traditional
    # Chinese" retry on an output the caller actually wanted Simplified
    # for, or burn an extra round-trip on a JSON-extraction call that
    # never had a degeneration risk in the first place.
    repetition_retry_on = (
        enforce_user_lang and lang_enforce_master and _env_bool("LMSTUDIO_REPETITION_RETRY", True)
    )
    repetition_salvage_min = max(1, _env_int("LMSTUDIO_REPETITION_SALVAGE_MIN", 50))
    lang_mismatch_retry_on = (
        enforce_user_lang and lang_enforce_master and _env_bool("LMSTUDIO_LANG_MISMATCH_RETRY", True)
    )
    simplified_check_on = (
        enforce_user_lang and lang_enforce_master and _env_bool("LMSTUDIO_SIMPLIFIED_CHECK", True)
    )

    # 1. Repetition loop. Truncate at the first repetition; if the salvaged
    #    prefix is too short to be a usable reply, retry once with stricter
    #    anti-repetition sampling (lower temperature, higher penalties).
    if text and text.strip() and repetition_retry_on:
        rep_idx = _detect_repetition_loop(text)
        if rep_idx is not None:
            salvaged = text[:rep_idx].rstrip()
            print(
                f"[LMStudio] Repetition loop detected — truncated at {rep_idx}/{len(text)} chars "
                f"(salvaged tail: {salvaged[-80:]!r})"
            )
            if len(salvaged) >= repetition_salvage_min:
                text = salvaged
            else:
                print(
                    f"[LMStudio] Salvaged prefix too short ({len(salvaged)}ch < "
                    f"LMSTUDIO_REPETITION_SALVAGE_MIN={repetition_salvage_min}) — "
                    f"retrying with stricter sampling"
                )
                retry_text = await _call_lmstudio(
                    lm_messages,
                    model=active_model,
                    temperature=0.6,
                    max_tokens=max_tokens,
                    extra_sampling={
                        "repetition_penalty": 1.15,
                        "frequency_penalty": 0.5,
                        "presence_penalty": 0.2,
                        "top_p": 0.85,
                        "min_p": 0.07,
                    },
                    response_format=response_format,
                )
                if retry_text and retry_text.strip():
                    rep_idx2 = _detect_repetition_loop(retry_text)
                    if rep_idx2 is None:
                        text = retry_text
                    else:
                        retry_salvage = retry_text[:rep_idx2].rstrip()
                        best = retry_salvage if len(retry_salvage) > len(salvaged) else salvaged
                        if len(best) >= repetition_salvage_min:
                            print("[LMStudio] Retry also looped — keeping longer of the two prefixes")
                            text = best
                        else:
                            # Both attempts produced garbage and even the
                            # longer salvage is unusably short. Return the
                            # standard user-facing failure message rather
                            # than ship degenerate text — matches how the
                            # transport-failure path behaves and lets the
                            # bot present a coherent error to the user.
                            print(
                                f"[LMStudio] Repetition recovery failed (best salvage {len(best)}ch) "
                                f"— returning user-facing failure message"
                            )
                            return _FAILED_REPLY_MESSAGE, None, False, False, False, None
                else:
                    # Retry call returned nothing AND the original salvage
                    # was already too short. Surface the standard failure
                    # message instead of a torn fragment.
                    print("[LMStudio] Repetition retry returned no text — returning user-facing failure message")
                    return _FAILED_REPLY_MESSAGE, None, False, False, False, None

    # 1b. Paragraph-level semantic deduplication.
    #     Catches the common case where the model semantically repeats the same
    #     3-4 sentences every few paragraphs (invisible to _REPETITION_RE which
    #     only sees character-level token runs).  Runs after the character-level
    #     check so both truncations act on the same `text` variable.
    if text and text.strip():
        _paras = [p for p in text.split("\n\n") if p.strip()]
        _seen_paras: list[str] = []
        _dup_idx: int | None = None
        for _pi, _para in enumerate(_paras):
            _norm = " ".join(_para.lower().split())
            for _prev in _seen_paras:
                _prev_len = len(_prev)
                _curr_len = len(_norm)
                if _prev_len == 0 or _curr_len == 0:
                    continue
                # Jaccard-style character overlap: count chars in common by
                # comparing the shorter string character-by-character against
                # the longer.  Cheap O(n) approximation sufficient here.
                _shorter, _longer = (
                    (_norm, _prev) if _curr_len <= _prev_len else (_prev, _norm)
                )
                _common = sum(1 for c in _shorter if c in _longer)
                _overlap = _common / max(_prev_len, _curr_len)
                if _overlap >= 0.80:
                    _dup_idx = _pi
                    break
            if _dup_idx is not None:
                break
            _seen_paras.append(_norm)
        if _dup_idx is not None:
            _good_prefix = "\n\n".join(_paras[:_dup_idx]).rstrip()
            print(
                f"[LMStudio] Paragraph repetition — truncated at para "
                f"{_dup_idx}/{len(_paras)} (kept {len(_good_prefix)}ch)"
            )
            if _good_prefix:
                text = _good_prefix

    # 2. Language mismatch — user wrote English but the reply went Chinese
    #    (the most common failure with mid-conversation language drift).
    if text and text.strip() and user_lang == "en" and lang_mismatch_retry_on:
        if _classify_response_language(text) == "zh":
            print("[LMStudio] Language mismatch — retrying with stronger English directive")
            retry_msgs = list(lm_messages)
            for _idx in range(len(retry_msgs) - 1, -1, -1):
                if retry_msgs[_idx].get("role") == "user":
                    retry_msgs.insert(_idx, {
                        "role": "system",
                        "content": (
                            "CRITICAL LANGUAGE OVERRIDE: the previous reply was rejected "
                            "because it used the wrong language. The user is writing in "
                            "English. Your reply MUST contain ONLY English words and "
                            "ASCII punctuation. Do NOT include ANY Han / Chinese "
                            "characters (no 漢字, no 中文), no romanized pinyin, no "
                            "other languages. A single Chinese character means the "
                            "reply is wrong."
                        ),
                    })
                    break
            retry_text = await _call_lmstudio(
                retry_msgs,
                model=active_model,
                temperature=0.6,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            if retry_text and retry_text.strip():
                if _classify_response_language(retry_text) == "zh":
                    print("[LMStudio] Retry still in Chinese — using retry result anyway")
                else:
                    print("[LMStudio] Retry produced English — language restored")
                text = retry_text

    # 3. Simplified Chinese. The static prompt requires Traditional, but
    #    Mistral Nemo Celeste sometimes emits Simplified mid-conversation.
    #    Only checked when the user is NOT writing in English (otherwise
    #    we'd want English anyway, and step 2 already handled that).
    if text and text.strip() and user_lang != "en" and simplified_check_on and _has_simplified_chinese(text):
        offenders = sorted({c for c in text if c in _SIMPLIFIED_ONLY_CHARS})
        offenders_str = "".join(offenders[:10])
        print(f"[LMStudio] Simplified Chinese detected ({offenders_str!r}) — retrying for Traditional")
        retry_msgs = list(lm_messages)
        for _idx in range(len(retry_msgs) - 1, -1, -1):
            if retry_msgs[_idx].get("role") == "user":
                retry_msgs.insert(_idx, {
                    "role": "system",
                    "content": (
                        "CRITICAL SCRIPT OVERRIDE: the previous reply was rejected "
                        "because it used Simplified Chinese characters. You MUST "
                        "write in Traditional Chinese (繁體中文 / 正體字) only. "
                        f"The following Simplified characters are FORBIDDEN: "
                        f"{offenders_str}. Use their Traditional forms instead "
                        "(e.g. 级→級, 执→執, 还→還, 让→讓, 这→這, 时→時, 开→開, "
                        "来→來, 给→給, 说→說, 进→進, 过→過, 请→請, 头→頭, "
                        "顾→顧, 爱→愛, 东→東, 车→車, 国→國, 会→會, 个→個, "
                        "们→們). Every Han character must be Traditional."
                    ),
                })
                break
        retry_text = await _call_lmstudio(
            retry_msgs,
            model=active_model,
            temperature=0.6,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        if retry_text and retry_text.strip():
            if _has_simplified_chinese(retry_text):
                print("[LMStudio] Retry still contains Simplified — using retry result anyway")
            else:
                print("[LMStudio] Retry produced Traditional — script restored")
            text = retry_text

    # 4. Paragraph-count floor for rich/cinematic targets.
    #    Only fires on the user-facing roleplay path (enforce_user_lang=True)
    #    and only for non-Qwen models (Qwen uses its own subtext block format).
    #    If the reply has fewer meaningful paragraphs than the floor, exactly
    #    one retry fires with a stricter system-prompt addendum quoting the
    #    actual count and the floor.  Capped to one retry per turn.
    if text and text.strip() and enforce_user_lang and not _is_qwen_model(active_model):
        _floor_target = _narration_target_for(active_model)
        _para_floor = _PARA_FLOORS.get(_floor_target, 0)
        if _para_floor:
            _para_count = _count_meaningful_paragraphs(text)
            if _para_count < _para_floor:
                print(
                    f"[LMStudio] Reply too short for target={_floor_target} "
                    f"({_para_count} para / floor {_para_floor}) — retrying"
                )
                _floor_addendum = (
                    f"\n\nURGENT CORRECTION: your previous reply contained only "
                    f"{_para_count} meaningful paragraph(s). The {_floor_target} "
                    f"target requires at least {_para_floor} full paragraphs of "
                    f"narration. Rewrite the reply now in full — do NOT explain "
                    f"or apologise, simply produce the reply with at least "
                    f"{_para_floor} and no more than 10 complete paragraphs "
                    f"separated by blank lines."
                )
                _floor_retry_sys = effective_system.rstrip() + _floor_addendum
                _floor_retry_msgs = [{"role": "system", "content": _floor_retry_sys}]
                _floor_retry_msgs.extend(
                    lm_messages[1:]
                    if lm_messages and lm_messages[0].get("role") == "system"
                    else lm_messages
                )
                _floor_retry_text = await _call_lmstudio(
                    _floor_retry_msgs,
                    model=active_model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
                if _floor_retry_text and _floor_retry_text.strip():
                    text = _floor_retry_text

    text = _parse_reply_format(text)

    # Strip the optional cinematic-scene signal first so the [IMAGE: ...]
    # extractor and the Discord formatter never see a stray `[SCENE]` token
    # in the reply text. The flag is returned as the 5th tuple element and
    # the optional body (from `[SCENE: ...]`) as the 6th — bot.py decides
    # whether to act on the flag based on the per-channel toggle, and uses
    # the body verbatim as the image-prompt seed when present.
    wants_scene = False
    scene_prompt: Optional[str] = None
    scene_match = _SCENE_MARKER_RE.search(text)
    if scene_match:
        wants_scene = True
        body = (scene_match.group(1) or "").strip()
        if body:
            scene_prompt = body
            print(
                f"[LMStudio] [SCENE: ...] cinematic signal detected with body "
                f"({len(body)}ch) — stripped from reply, body forwarded as seed"
            )
        else:
            print("[LMStudio] [SCENE] cinematic signal detected — stripped from reply")
        text = _SCENE_MARKER_RE.sub("", text).strip()

    # Extract the [IMAGE: ...] marker BEFORE running the Discord formatter,
    # otherwise a marker-only reply would be wrapped in *...* by the
    # narration-italicising pass, leaving stray `**` after marker removal
    # and emitting junk text alongside the image.
    marker_match = _IMAGE_MARKER_RE.search(text)
    if marker_match:
        img_prompt = marker_match.group(1).strip()
        clean_text = _IMAGE_MARKER_RE.sub("", text).strip()
        # Discord formatting is gated on `enforce_user_lang` because the four
        # utility callers (suggestion/memory/image-prompt/image-comment) want
        # the raw model text — bolding quoted strings inside a JSON array or
        # italicising a single-sentence reaction would corrupt their parsers.
        if clean_text and not _is_qwen_model(active_model) and enforce_user_lang:
            clean_text = _format_for_discord(clean_text, character_name=character_name)
        print(f"[LMStudio] Image prompt from marker (already enhanced): {img_prompt[:80]!r}")
        return (clean_text or None), img_prompt, True, True, wants_scene, scene_prompt

    # Plain-prose models (Celeste etc.) ignore the prompt-level Discord-
    # formatting instruction, so we apply bold-dialogue + italic-narration
    # in code. Qwen / hauhaucs ChatML output is already formatted by
    # _parse_reply_format() above and must NOT be re-processed. Utility
    # callers (`enforce_user_lang=False`) are also skipped — see comment
    # on the marker-path branch above for why.
    if not _is_qwen_model(active_model) and enforce_user_lang:
        text = _format_for_discord(text, character_name=character_name)

    if response_declines_image(text, messages=messages):
        img_prompt = user_wants_image(messages)
        if img_prompt:
            print(f"[LMStudio] Image prompt from fallback (raw, needs enhancement): {img_prompt[:80]!r}")
            return None, img_prompt, False, True, wants_scene, scene_prompt

    return text, None, False, True, wants_scene, scene_prompt


async def understand_image(
    image_bytes: bytes,
    mime_type: str,
    question: str = "Describe this image in detail.",
) -> Optional[str]:
    """Analyze an image using LM Studio's vision-capable model.

    Sends a multimodal user message (image_url + text) to /v1/chat/completions.
    Returns the description string on success, or None on failure (e.g. the
    loaded model isn't actually multimodal, or the server is unreachable).
    """
    model = _vision_model()

    # Short-circuit: if we already learned this model is text-only earlier
    # in the session, don't bother making the request.
    if model in _NO_VISION_MODELS:
        print(f"[LMStudio Vision] {model} known to be text-only — skipping image understanding")
        return None

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"

    # Apply /no_think via a tiny system message so the model goes straight to
    # describing the image instead of burning tokens on reasoning.
    # Pass the resolved vision model so the gate checks the right model family.
    system_msg = _apply_no_think("", model=model)
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": question},
        ],
    })

    try:
        text = await _call_lmstudio(messages, model=model, temperature=0.5, max_tokens=2048)
        if text == _NO_VISION_SENTINEL:
            print(f"[LMStudio Vision] {model} is text-only — image understanding unavailable")
            return None
        if text and text.strip():
            print(f"[LMStudio Vision] Success with model: {model}")
            return text
        print(f"[LMStudio Vision] Empty response from {model} — model may not be vision-capable")
        return None
    except Exception as e:
        print(f"[LMStudio Vision] Error: {e}")
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

    # Structured-output mode: when LMSTUDIO_USE_JSON_SCHEMA is on (and the
    # server hasn't 4xx'd it earlier this process), ask LM Studio to constrain
    # decoding to a `{"items":[<=3 short strings]}` schema. The salvage parser
    # below still runs as a no-op safety net for older servers and for any
    # response that slips through as a bare array.
    response_format = _memory_response_format() if _json_schema_enabled() else None

    try:
        # Tight budget: 3 facts × ≤60 chars ≈ 90 tokens max. A large budget
        # lets mn-12b-celeste drift into roleplay prose. temperature=0.1 keeps
        # sampling near-deterministic so the model follows the JSON instruction.
        # enforce_user_lang=False: the user-message wrapper here is English
        # ("Extract memorable facts: …") but the system prompt requires the
        # extracted facts to be written in Traditional Chinese. Letting the
        # language enforcer kick in would force English facts and break the
        # downstream JSON consumers.
        text, *_ = await chat(
            messages,
            system_prompt=system,
            max_tokens=192,
            temperature=0.1,
            enforce_user_lang=False,
            response_format=response_format,
        )
        if not text:
            return []
        # Try the structured-output shape first (`{"items":[...]}` or bare
        # `[...]`); fall back to the legacy substring scan for older servers
        # that returned arbitrary prose around the array.
        parsed = _parse_json_array_payload(text)
        if parsed is not None:
            return [str(s).strip() for s in parsed[:3] if isinstance(s, str) and s.strip()]
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            arr = _json.loads(text[start:end])
            if isinstance(arr, list):
                return [str(s).strip() for s in arr[:3] if isinstance(s, str) and s.strip()]
    except Exception as e:
        print(f"[LMStudio Memory] Extract error: {e}")

    return []


async def enhance_image_prompt(
    raw_prompt: str,
    character_context: str = "",
    subject_references: dict = None,
    subject_supplements: dict = None,
    reference_images: list = None,
    reference_image_labels: list = None,
    n_subjects_override: int = None,
    scene_only: bool = False,
    prose_context: str = None,
) -> str:
    """Translate and expand a raw prompt into a rich English image-generation prompt.

    When reference_images are provided, they are passed through chat() as
    context_images so the vision model can read appearance directly from the
    photos — significantly improving accuracy for traits like eye color and
    hair tone.

    scene_only: when True, the enhancer is told to describe ONLY the scene
        (setting, action, pose, lighting, mood, atmosphere, composition,
        background) and to NOT emit any appearance traits (hair, eyes, skin,
        clothing, accessories). Used by the Qwen image-edit pipeline, where
        the reference photos go to Qwen — not to this enhancer — and any
        appearance words bleed into Qwen's text prompt and override the
        photo's visual identity. In this mode `character_context`,
        `subject_references`, `subject_supplements`, `reference_images`, and
        `reference_image_labels` are all ignored to keep the model focused
        on scene language.
    """
    import re as _re_artstrip
    _art_style_line_re = _re_artstrip.compile(r"(?m)^ART STYLE[:\s][^\n]*\n?", _re_artstrip.IGNORECASE)
    if character_context:
        character_context = _art_style_line_re.sub("", character_context)

    # Scene-only short-circuit: skip every appearance block, drop reference
    # photos, and use a system prompt that explicitly forbids appearance words.
    if scene_only:
        use_schema_so = _json_schema_enabled()
        if use_schema_so:
            output_rule_so = (
                "- Output ONLY a JSON object of the shape {\"text\": \"<prompt>\"} — "
                "no intro, no markdown, no code fences, no explanation outside the JSON.\n"
                "- The prompt itself goes inside the `text` field as one continuous English string.\n"
            )
        else:
            output_rule_so = "- Output ONLY the prompt text — no intro, no quotes, no explanation.\n"
        _prose_block_so = ""
        if prose_context and prose_context.strip():
            _prose_block_so = (
                "\n[RECENT STORY CONTEXT — extract scene details for coherence]\n"
                f"{prose_context.strip()}\n"
                "From the above, extract and embed in your output: "
                "(1) location / setting, (2) time of day, (3) weather or lighting conditions, "
                "(4) emotional mood of the scene. Use these to anchor the generated prompt "
                "to what is actually happening in the story.\n"
            )
        system_so = (
            "You are an expert image-prompt writer for AI image generators.\n"
            "Given a user's image request (which may be in Chinese or English), "
            "rewrite it as a single, rich English prompt for an AI image model.\n"
            "[SCENE-ONLY MODE — REFERENCE PHOTOS ARE HANDLED ELSEWHERE]\n"
            "Reference photos of the characters are sent directly to the image "
            "model in a separate channel. Your job is to describe ONLY the scene "
            "around the characters — setting, action, pose, framing, lighting, "
            "mood, atmosphere, composition, background, weather, time of day.\n"
            "STRICT BAN — DO NOT WRITE any of the following, even if the user "
            "request mentions them: hair color, hair style, eye color, skin "
            "tone, age, body shape, height, clothing, outfit pieces, fabrics, "
            "accessories, jewelry, makeup, tattoos, scars, facial features. "
            "If the user request includes any such words, drop them from your "
            "rewrite. Refer to characters by name only (or by role like 'the "
            "musician'); never describe what they look like or what they wear.\n"
            f"{_prose_block_so}"
            "FRAMING RULES (mandatory):\n"
            "- DEFAULT framing: medium-close portrait (bust or face-prominent). "
            "The character's face should fill a significant portion of the frame, "
            "like a manga CG close-up. This is the default for solo character shots.\n"
            "- Use FULL-BODY or WIDE framing ONLY when: (a) two or more characters "
            "are physically interacting (fighting, dancing, embracing), or (b) the "
            "scene is explicitly about an environment reveal, action shot, or "
            "dynamic movement that requires showing the whole figure.\n"
            "- NEVER default to a full-body or distant shot for a solo character "
            "standing, sitting, talking, or posing — always prefer the tighter frame.\n"
            "Rules:\n"
            f"{output_rule_so}"
            "- Always write in English.\n"
            "- Aim for 100-220 words.\n"
            "- Do NOT start with 'Generate', 'Create', 'Draw', 'An image of', etc.\n"
            "- ART STYLE: Do NOT specify any art style, rendering technique, shading "
            "method, or visual medium in your output. Reference photos of the characters "
            "are sent directly to the image model, which will replicate their visual style "
            "automatically. Adding style descriptors here would conflict with the reference "
            "and cause style inconsistency — leave all style determination to the image model.\n"
        )
        user_content_so = f"Image request: {raw_prompt}"
        messages_list_so = [{"role": "user", "content": user_content_so}]
        response_format_so = (
            _image_text_response_format("image_prompt", 4000) if use_schema_so else None
        )
        print(
            "[LMStudio] enhance_image_prompt: SCENE-ONLY mode — "
            "appearance blocks suppressed, reference photos NOT attached."
        )
        try:
            enhanced_so, *_ = await chat(
                messages_list_so,
                system_prompt=system_so,
                context_images=None,
                max_tokens=8192,
                enforce_user_lang=False,
                response_format=response_format_so,
            )
            if enhanced_so:
                enhanced_so = _THINK_RE.sub("", enhanced_so)
                enhanced_so = _THINK_RE_UNCLOSED.sub("", enhanced_so).strip()
                if use_schema_so:
                    unwrapped_so = _parse_json_string_payload(enhanced_so)
                    if unwrapped_so:
                        enhanced_so = unwrapped_so.strip()
                if len(enhanced_so) > 5:
                    lower_so = enhanced_so.lower()
                    if any(phrase in lower_so for phrase in _REFUSAL_PHRASES):
                        print("[LMStudio] Scene-only enhancement returned a refusal/error — discarding, using raw prompt")
                        return raw_prompt
                    print(f"[LMStudio] Prompt enhanced (scene-only): {enhanced_so[:200]}")
                    return enhanced_so
        except Exception as e:
            print(f"[LMStudio] Scene-only prompt enhancement failed: {e}")
        return raw_prompt

    has_images = bool(reference_images)

    char_block = ""
    if character_context and character_context.strip():
        if has_images:
            # Photos are primary appearance source; text supplements only
            char_block = (
                f"\n[CHARACTER APPEARANCE — SUPPLEMENTAL TEXT (photos are primary)]\n"
                f"Reference photos are attached. Read appearance from the photos first.\n"
                f"Use the text below ONLY as a supplement for details not clearly visible in the photos.\n"
                f"EXCEPTION: any ART STYLE line in this block describes the character's look, not a rendering directive — ignore it for style purposes.\n"
                f"{character_context.strip()}\n"
            )
        else:
            char_block = (
                f"\n[CHARACTER APPEARANCE — VERIFIED GROUND TRUTH]\n"
                f"The following contains confirmed, factual appearance details for the character. "
                f"You MUST use ALL of these physical traits in your output. Do NOT invent, alter, or substitute any of them.\n"
                f"Eye color, hair color, hairstyle, skin tone, AND OUTFIT described here are FINAL — "
                f"they override anything conflicting in the raw prompt or your own assumptions. "
                f"If an outfit is described here, reproduce it in full detail in the output. "
                f"Do NOT summarize, omit, or replace any garment piece listed.\n"
                f"EXCEPTION: any ART STYLE line in this block describes the character's look, not a rendering directive — ignore it for style purposes.\n"
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

    image_note = ""
    if has_images:
        image_note = (
            "\n[REFERENCE PHOTOS ATTACHED — PRIORITY ORDER]\n"
            "Reference photos are attached. Priority order (highest first):\n"
            "  1. Verified text descriptions for named subjects (above) — absolute authority\n"
            "  2. Reference photos — primary source for appearance traits not covered by text\n"
            "  3. Raw prompt appearance words — DISCARD (written without photos, unreliable)\n"
            "OVERRIDE — COLORS: discard raw prompt color words for hair, eyes, and skin. "
            "Read these from the photos (or use the verified text if present).\n"
            "ART STYLE: Match the art style, line art, and shading technique visible in "
            "the reference photos. Do NOT override the reference style with a generic "
            "description — replicate how the reference images actually look.\n"
        )
        if reference_image_labels:
            _photo_map_lines = "\n".join(
                f"  Photo {i + 1} = {lbl}"
                for i, lbl in enumerate(reference_image_labels)
            )
            image_note += (
                "\n[REFERENCE PHOTO ORDER]\n"
                "The attached photos are provided in this exact order:\n"
                f"{_photo_map_lines}\n"
                "When reading appearance for a character, use ONLY their assigned photo number.\n"
            )

    use_schema = _json_schema_enabled()
    if use_schema:
        # Schema mode emits `{"text":"<prompt>"}` so the parser shape is
        # consistent with generate_image_comment. Output rule is reworded
        # so the model writes the prompt INTO the `text` field rather
        # than emitting bare prose.
        output_rule = (
            "- Output ONLY a JSON object of the shape {\"text\": \"<prompt>\"} — "
            "no intro, no markdown, no code fences, no explanation outside the JSON.\n"
            "- The prompt itself goes inside the `text` field as one continuous English string.\n"
        )
    else:
        output_rule = "- Output ONLY the prompt text — no intro, no quotes, no explanation.\n"

    _prose_block = ""
    if prose_context and prose_context.strip():
        _prose_block = (
            "\n[RECENT STORY CONTEXT — extract scene details for coherence]\n"
            f"{prose_context.strip()}\n"
            "From the above, extract and embed in your output: "
            "(1) location / setting, (2) time of day, (3) weather or lighting conditions, "
            "(4) emotional mood of the scene. Use these to anchor the generated prompt "
            "to what is actually happening in the roleplay.\n"
        )

    # When reference photos are attached, instruct the enhancer to match their visual
    # style rather than imposing a hardcoded style. Without references, apply a
    # sensible anime/2D fallback so the output is still style-guided.
    if has_images:
        _art_style_rule = (
            "- ART STYLE: Match the visual style, line art, and shading technique "
            "that is visible in the attached reference photos. Describe the style you "
            "observe (e.g. cel-shaded anime, painterly illustration, etc.) so the "
            "image model can replicate it. Do NOT impose a fixed generic style.\n"
        )
    else:
        _art_style_rule = (
            "- ART STYLE (fallback — no reference photos): "
            "Use 'clean 2D anime illustration, flat cel-shaded coloring, "
            "soft gradient shading, vivid saturated color palette, anime digital art'.\n"
        )

    system = (
        "You are an expert image-prompt writer for AI image generators.\n"
        "Given a user's image request (which may be in Chinese or English), "
        "rewrite it as a single, rich English prompt for an AI image model.\n"
        f"{image_note}"
        f"{char_block}"
        f"{ref_block}"
        f"{_prose_block}"
        "FRAMING RULES (mandatory):\n"
        "- DEFAULT framing: medium-close portrait (bust or face-prominent). "
        "The character's face should fill a significant portion of the frame, "
        "like a manga CG close-up. This is the default for solo character shots.\n"
        "- Use FULL-BODY or WIDE framing ONLY when: (a) two or more characters "
        "are physically interacting (fighting, dancing, embracing), or (b) the "
        "scene is explicitly about an environment reveal, action shot, or "
        "dynamic movement that requires showing the whole figure.\n"
        "- NEVER default to a full-body or distant shot for a solo character "
        "standing, sitting, talking, or posing — always prefer the tighter frame.\n"
        "Rules:\n"
        f"{output_rule}"
        "- Always write in English.\n"
        "- Be specific: include subject, art style, lighting, colors, mood, and setting.\n"
        "- Aim for 150-300 words.\n"
        "- Do NOT start with 'Generate', 'Create', 'Draw', 'An image of', etc.\n"
        "- Physical details from character context (hair color, eye color, hairstyle, skin tone) "
        "ALWAYS take priority over anything in the raw prompt.\n"
        f"{_art_style_rule}"
    )

    _char_snippets: list = []
    if subject_references:
        for _sname, _sdesc in subject_references.items():
            _char_snippets.append(
                f"[MANDATORY APPEARANCE FOR '{_sname}' — COPY EVERY DETAIL EXACTLY]\n"
                f"{_sdesc.strip()}\n"
                f"[END OF '{_sname}' APPEARANCE]"
            )
    if character_context and character_context.strip() and not has_images:
        # When photos are attached, the supplemental text block in the system
        # prompt is enough — avoid re-stating it as a "MANDATORY COPY EXACTLY"
        # block, which would conflict with the photos-are-primary directive.
        _char_snippets.append(
            f"[MANDATORY CHARACTER APPEARANCE — COPY EVERY DETAIL EXACTLY]\n"
            f"{character_context.strip()}\n"
            f"[END OF CHARACTER APPEARANCE]"
        )
    if _char_snippets:
        user_content = (
            f"Image request: {raw_prompt}\n\n"
            + "\n\n".join(_char_snippets)
            + "\n\nCRITICAL: your output MUST reproduce the outfit, hair, eyes, skin, and all "
            "accessories EXACTLY as listed above."
        )
    else:
        user_content = f"Image request: {raw_prompt}"

    if has_images:
        print(f"[LMStudio] enhance_image_prompt: using vision model with {len(reference_images)} reference image(s)")

    messages_list = [{"role": "user", "content": user_content}]
    response_format = (
        _image_text_response_format("image_prompt", 4000) if use_schema else None
    )
    try:
        enhanced, *_ = await chat(
            messages_list,
            system_prompt=system,
            context_images=reference_images if has_images else None,
            max_tokens=8192,
            # The system prompt mandates English output and the user message
            # often contains both English ("Image request:") and Chinese
            # (character description) — the language enforcer's heuristic
            # is meaningless here, so we disable it.
            enforce_user_lang=False,
            response_format=response_format,
        )
        if enhanced:
            enhanced = _THINK_RE.sub("", enhanced)
            enhanced = _THINK_RE_UNCLOSED.sub("", enhanced).strip()
            # Schema mode wraps the prompt in {"text": "..."}; pull it back
            # out so downstream consumers see the same plain string. If the
            # parser returns None (schema ignored, or 4xx auto-disable
            # already tripped), fall through to the bare-text path below.
            if use_schema:
                unwrapped = _parse_json_string_payload(enhanced)
                if unwrapped:
                    enhanced = unwrapped.strip()
            if len(enhanced) > 5:
                # Detect refusals and connection errors: when the vision call
                # fails (e.g. model isn't multimodal, LM Studio unreachable),
                # chat() returns either an apology from the model or the
                # "having trouble connecting" error string. Either would poison
                # the image prompt if passed downstream — fall back to raw.
                lower = enhanced.lower()
                if any(phrase in lower for phrase in _REFUSAL_PHRASES):
                    print("[LMStudio] Enhancement returned a refusal/error — discarding, using raw prompt")
                    return raw_prompt
                print(f"[LMStudio] Prompt enhanced: {enhanced[:200]}")
                return enhanced
    except Exception as e:
        print(f"[LMStudio] Prompt enhancement failed: {e}")
    return raw_prompt


async def generate_image_comment(
    image_prompt: str,
    bot_name: str,
    character_background: str,
    user_request: str = "",
    history: list = None,
) -> str:
    """Generate a short in-character comment to accompany a generated image."""
    use_schema = _json_schema_enabled()
    if use_schema:
        # Wrap the comment in {"text": "..."} so structured-output mode
        # produces parseable JSON instead of bare prose.
        format_rule = (
            "- Output ONLY a JSON object of the shape {\"text\": \"<sentence>\"} — "
            "no markdown, no code fences, no explanation outside the JSON.\n"
            "- Put your in-character comment inside the `text` field as one short string.\n"
        )
    else:
        format_rule = ""

    system = (
        f"You are {bot_name}. {character_background}\n"
        "You are mid-conversation and have just drawn/created an image for the user. "
        "Write ONE short sentence (two at most) to send alongside it — something that flows "
        "naturally from the conversation you were just having, like a real person would say.\n"
        "Rules:\n"
        f"{format_rule}"
        "- Stay in the mood and tone of the conversation.\n"
        "- Language: default Traditional Chinese (繁體中文). Only switch if the user wrote a full sentence in another language. Never use Simplified Chinese.\n"
        "- Do NOT say 'Here is', 'Here's', 'I generated', 'I created', '好的', '完成了', or anything that sounds like a task report.\n"
        "- Do NOT describe the image — the user can see it.\n"
        "- Sound like yourself, not an assistant completing a request.\n"
    )

    context_lines = []
    if history:
        for msg in history[-6:]:
            role_label = "User" if msg["role"] == "user" else bot_name
            c = msg["content"]
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            context_lines.append(f"{role_label}: {str(c)[:200]}")
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
    response_format = (
        _image_text_response_format("image_comment", 400) if use_schema else None
    )
    try:
        # enforce_user_lang=False: the user message is the English instruction
        # "Write your one-sentence reaction…" but the system prompt asks for
        # Traditional Chinese (or whatever language the conversation is in).
        text, *_ = await chat(
            messages,
            system_prompt=system,
            max_tokens=4096,
            enforce_user_lang=False,
            response_format=response_format,
        )
        if text:
            # Schema mode wraps the comment in {"text": "..."}; recover it
            # so the caller still gets a plain string. None means schema
            # was ignored or auto-disable already tripped — fall through.
            if use_schema:
                unwrapped = _parse_json_string_payload(text)
                if unwrapped:
                    text = unwrapped
            if len(text.strip()) > 2:
                return text.strip()
    except Exception as e:
        print(f"[LMStudio] Image comment generation failed: {e}")

    return ""


async def generate_suggestions(
    topic: str,
    bot_name: str,
    character_background: str,
    count: int = 3,
    guiding_prompt: str = "",
    language_sample: str = "",
    recent_history: list = None,
) -> list:
    """Generate short follow-up suggestion buttons (max 80 chars each).

    Thin wrapper around `reply_format.generate_suggestions` that supplies
    LM Studio's chat-call shape and structured-output toggle. The shared
    helper owns the prompt construction, JSON-parse → salvage pipeline,
    and `[LMStudio]` log lines.
    """
    # Structured-output mode: ask LM Studio to constrain decoding to a
    # `{"items":[exactly N short strings]}` schema when LMSTUDIO_USE_JSON_SCHEMA
    # is on. The salvage path stays in place so the bot still recovers
    # buttons from numbered/bulleted text on servers that ignore the field
    # or once the 4xx auto-disable has tripped.
    response_format = (
        _suggestion_response_format(count) if _json_schema_enabled() else None
    )

    async def _chat(messages: list, system_prompt: str) -> str:
        # Tight token budget: 3 suggestions × ≤80 chars ≈ 60–100 tokens max.
        # A large budget lets mn-12b-celeste drift into roleplay prose instead
        # of emitting the JSON array. temperature=0.15 keeps sampling near-
        # deterministic so the model follows the JSON instruction rather than
        # improvising creative content.
        # enforce_user_lang=False: language matching is handled by the
        # `language_sample` directive in the system prompt.
        text, *_ = await chat(
            messages,
            system_prompt=system_prompt,
            max_tokens=256,
            temperature=0.15,
            enforce_user_lang=False,
            response_format=response_format,
        )
        return text

    # LM Studio's prompt has historically always asked for a bare JSON array
    # even when JSON Schema mode is on — the schema wraps the array in
    # `items` automatically and the shared parser recovers it. Keeping
    # `prompt_object_root=False` preserves the exact prompt text.
    return await _shared_generate_suggestions(
        _chat,
        "LMStudio",
        topic=topic,
        bot_name=bot_name,
        character_background=character_background,
        count=count,
        guiding_prompt=guiding_prompt,
        language_sample=language_sample,
        prompt_object_root=False,
        recent_history=recent_history,
    )
