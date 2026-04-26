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
_CR_NAME_PREFIX_RE = re.compile(r"^[A-Z][A-Za-z]{1,24}(?: [A-Z][A-Za-z]{1,24})?: ", re.MULTILINE)
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


# Discord-format post-processor for plain-prose models like Celeste.
# Mistral-family roleplay fine-tunes ignore the prompt-level "wrap dialogue
# in **bold**" instruction and emit raw prose with quoted dialogue. We bold
# the quoted segments and italicise pure-narration paragraphs in code so
# the reply renders correctly in Discord regardless of model compliance.

# Quote pairs we treat as dialogue. Lone apostrophes (`'`) are deliberately
# excluded — they collide with English contractions ("Kelly's", "You're").
_DIALOGUE_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),       # straight ASCII double quotes
    ("\u201c", "\u201d"),  # curly double quotes "..."
    ("\u300c", "\u300d"),  # Chinese corner brackets 「...」
    ("\u300e", "\u300f"),  # Chinese white corner brackets 『...』
)


# Patterns for detecting already-formatted spans. Used to mask them out
# before applying quote-bolding so we never re-wrap quotes that are
# already inside **bold** or *italic* spans (or merely adjacent to them).
_PROTECTED_BOLD_RE = re.compile(r"\*\*[^*\n]+?\*\*")
_PROTECTED_ITALIC_RE = re.compile(r"\*[^*\n]+?\*")
_PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")


def _bold_quoted_dialogue(text: str) -> str:
    """Wrap quoted dialogue segments in **bold** when not already wrapped.

    Idempotency strategy: before processing, all existing `**...**` and
    `*...*` spans are extracted into placeholders, so quotes that already
    live inside formatted spans (or are merely adjacent to them) are
    never re-wrapped. After processing, the placeholders are restored.
    Re-running on the helper's own output is a guaranteed no-op.

    For same-char delimiters (straight `"`), nested quotes within a
    single line — e.g. `"He said "go" quietly"` — would corrupt the
    markdown, so any line whose `"` count is odd or greater than 2 is
    left untouched for that pair. Asymmetric delimiters (`「」`, `『』`,
    curly `""`) are immune to nesting confusion since the open and
    close characters differ.

    Single-quote `'…'` dialogue is also bolded, gated by negative
    letter-boundary lookarounds so apostrophes inside contractions
    (`Kelly's`, `You're`, `don't`) are never matched as dialogue quotes.
    Lines with more than 4 single-quote chars are skipped to avoid
    contraction-heavy prose accidentally triggering matches.
    """
    placeholders: list[str] = []

    def _stash(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    masked = _PROTECTED_BOLD_RE.sub(_stash, text)
    masked = _PROTECTED_ITALIC_RE.sub(_stash, masked)

    for open_q, close_q in _DIALOGUE_QUOTE_PAIRS:
        if open_q == close_q:
            inner = rf"[^{re.escape(close_q)}\n]+?"
            pattern = re.compile(
                rf"{re.escape(open_q)}({inner}){re.escape(close_q)}"
            )
            new_lines = []
            for line in masked.split("\n"):
                count = line.count(open_q)
                if count == 0 or count % 2 != 0 or count > 2:
                    new_lines.append(line)
                else:
                    new_lines.append(pattern.sub(rf"**{open_q}\1{close_q}**", line))
            masked = "\n".join(new_lines)
        else:
            inner = rf"[^{re.escape(open_q)}{re.escape(close_q)}\n]+?"
            pattern = re.compile(
                rf"{re.escape(open_q)}({inner}){re.escape(close_q)}"
            )
            masked = pattern.sub(rf"**{open_q}\1{close_q}**", masked)

    # Single-quote dialogue: open `'` must NOT be preceded by a letter,
    # close `'` must NOT be followed by a letter. This filters out every
    # apostrophe inside a word.
    sq_pattern = re.compile(r"(?<![A-Za-z])'([^'\n]+?)'(?![A-Za-z])")
    new_lines = []
    for line in masked.split("\n"):
        count = line.count("'")
        if count == 0 or count > 4:
            new_lines.append(line)
        else:
            new_lines.append(sq_pattern.sub(r"**'\1'**", line))
    masked = "\n".join(new_lines)

    def _unstash(m: re.Match) -> str:
        return placeholders[int(m.group(1))]

    return _PLACEHOLDER_RE.sub(_unstash, masked)


def _italicize_pure_narration(text: str) -> str:
    """Wrap whole paragraphs of unformatted narration in *italics*.

    Only touches paragraphs that contain NO existing markdown (`*`, `_`)
    and NO dialogue quotes. Mixed paragraphs (italic narration + bolded
    dialogue) are left alone — their dialogue is already handled by
    `_bold_quoted_dialogue`, and re-wrapping the whole thing would
    double-up markers.
    """
    quote_chars = {open_q for open_q, _ in _DIALOGUE_QUOTE_PAIRS} | {
        close_q for _, close_q in _DIALOGUE_QUOTE_PAIRS
    }
    out_paragraphs = []
    for para in text.split("\n\n"):
        stripped = para.strip()
        if not stripped:
            out_paragraphs.append(para)
            continue
        if "*" in stripped or "_" in stripped:
            out_paragraphs.append(para)
            continue
        if any(q in stripped for q in quote_chars):
            out_paragraphs.append(para)
            continue
        # Pure narration paragraph — italicise while preserving any
        # surrounding whitespace.
        leading_len = len(para) - len(para.lstrip())
        trailing_len = len(para) - len(para.rstrip())
        leading = para[:leading_len]
        trailing = para[len(para) - trailing_len:] if trailing_len else ""
        out_paragraphs.append(f"{leading}*{stripped}*{trailing}")
    return "\n\n".join(out_paragraphs)


def _format_for_discord(text: str) -> str:
    """Convert plain-prose roleplay output into Discord-ready markdown.

    Two passes:
      1. Bold any quoted dialogue segments not already inside **...**.
      2. Italicise paragraphs that are pure narration with no existing
         formatting and no dialogue quotes.

    Designed for Mistral-family models (Celeste etc.) that ignore the
    prompt-level formatting instruction. Qwen / hauhaucs ChatML output is
    handled by `_parse_reply_format()` and must NOT pass through this.
    """
    if not text:
        return text
    text = _bold_quoted_dialogue(text)
    text = _italicize_pure_narration(text)
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
    """Return True when the model name indicates a Qwen-family model.

    Qwen models need several special-case workarounds (thinking prefill,
    chat_template_kwargs, /no_think directive) that would confuse or break
    non-Qwen models such as Mistral-based Celeste variants.
    """
    return "qwen" in model.lower()


def _thinking_enabled() -> bool:
    """Whether the model should be allowed to use its <think>...</think> phase.

    Off by default: this is a roleplay chatbot, and the reasoning phase adds
    latency, eats the token budget, and breaks immersion. Set
    LMSTUDIO_THINKING=on (or true/1/yes) to re-enable for debugging or
    quality-sensitive tasks. Only meaningful for Qwen-family models.
    """
    val = os.environ.get("LMSTUDIO_THINKING", "off").strip().lower()
    return val in ("on", "true", "1", "yes", "y")


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
            "repetition_penalty": 1.08,
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
) -> Optional[str]:
    """Make a single chat completion request using LM Studio's OpenAI-compatible endpoint.

    For Qwen-family models, when thinking is disabled, a closed empty think
    block is prefilled as a partial assistant message so the model skips the
    reasoning phase entirely. Both closed and unclosed think blocks are also
    stripped from any content that slips through.

    For non-Qwen models (e.g. Celeste) none of the Qwen3-specific workarounds
    are applied — the request is sent as a plain chat completion.
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


async def chat(
    messages: list,
    system_prompt: str = "",
    model: str = "",
    context_images: Optional[list] = None,
    max_tokens: Optional[int] = None,
    enforce_user_lang: bool = True,
) -> tuple[str, Optional[str], bool, bool]:
    """Send a chat request to LM Studio. Returns (response_text, image_prompt_or_None, prompt_from_marker, success).

    context_images: optional list of (bytes, mime_type) tuples injected as visual
    content into the last user message. When provided, the vision model is used
    automatically (defaults to the same model since the Qwen 3.5 default is multimodal).
    prompt_from_marker=True means the image prompt came from the [IMAGE: ...] tag.
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

    effective_system = _apply_no_think(system_prompt, model=active_model)

    # For plain-prose models (anything that is NOT a Qwen/hauhaucs ChatML
    # fine-tune), inject Discord formatting + language-fallback rules.
    # Qwen/hauhaucs models use their own <reply>/<subtext> ChatML format which
    # _parse_reply_format() converts, so they must NOT get these instructions.
    if not _is_qwen_model(active_model) and effective_system:
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

    text = await _call_lmstudio(lm_messages, model=active_model, max_tokens=max_tokens)

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
        text = await _call_lmstudio(plain_messages, model=active_model, max_tokens=max_tokens)

    if text is None or text == _NO_VISION_SENTINEL:
        return _FAILED_REPLY_MESSAGE, None, False, False

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
            retry_text = await _call_lmstudio(retry_messages, model=active_model, max_tokens=max_tokens)
            if retry_text and retry_text.strip():
                print("[LMStudio] /no_think retry produced an answer")
                text = retry_text
            else:
                print("[LMStudio] /no_think retry also empty — giving up")
                return "", None, False, True
        else:
            print("[LMStudio] Empty answer with thinking already off — giving up")
            return "", None, False, True

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

        retry_text = await _call_lmstudio(retry_msgs, model=active_model, temperature=0.6, max_tokens=max_tokens)
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
    repetition_retry_on = lang_enforce_master and _env_bool("LMSTUDIO_REPETITION_RETRY", True)
    repetition_salvage_min = max(1, _env_int("LMSTUDIO_REPETITION_SALVAGE_MIN", 50))
    lang_mismatch_retry_on = lang_enforce_master and _env_bool("LMSTUDIO_LANG_MISMATCH_RETRY", True)
    simplified_check_on = lang_enforce_master and _env_bool("LMSTUDIO_SIMPLIFIED_CHECK", True)

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
                            return _FAILED_REPLY_MESSAGE, None, False, False
                else:
                    # Retry call returned nothing AND the original salvage
                    # was already too short. Surface the standard failure
                    # message instead of a torn fragment.
                    print("[LMStudio] Repetition retry returned no text — returning user-facing failure message")
                    return _FAILED_REPLY_MESSAGE, None, False, False

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
        )
        if retry_text and retry_text.strip():
            if _has_simplified_chinese(retry_text):
                print("[LMStudio] Retry still contains Simplified — using retry result anyway")
            else:
                print("[LMStudio] Retry produced Traditional — script restored")
            text = retry_text

    text = _parse_reply_format(text)

    # Extract the [IMAGE: ...] marker BEFORE running the Discord formatter,
    # otherwise a marker-only reply would be wrapped in *...* by the
    # narration-italicising pass, leaving stray `**` after marker removal
    # and emitting junk text alongside the image.
    marker_match = _IMAGE_MARKER_RE.search(text)
    if marker_match:
        img_prompt = marker_match.group(1).strip()
        clean_text = _IMAGE_MARKER_RE.sub("", text).strip()
        if clean_text and not _is_qwen_model(active_model):
            clean_text = _format_for_discord(clean_text)
        print(f"[LMStudio] Image prompt from marker (already enhanced): {img_prompt[:80]!r}")
        return (clean_text or None), img_prompt, True, True

    # Plain-prose models (Celeste etc.) ignore the prompt-level Discord-
    # formatting instruction, so we apply bold-dialogue + italic-narration
    # in code. Qwen / hauhaucs ChatML output is already formatted by
    # _parse_reply_format() above and must NOT be re-processed.
    if not _is_qwen_model(active_model):
        text = _format_for_discord(text)

    if response_declines_image(text, messages=messages):
        img_prompt = user_wants_image(messages)
        if img_prompt:
            print(f"[LMStudio] Image prompt from fallback (raw, needs enhancement): {img_prompt[:80]!r}")
            return None, img_prompt, False, True

    return text, None, False, True


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

    try:
        # Larger budget than chat() default — qwen3 reasoning often needs
        # 2-3K tokens before producing the JSON output. Falls back gracefully
        # to [] when LM Studio still hides everything in reasoning_content.
        # enforce_user_lang=False: the user-message wrapper here is English
        # ("Extract memorable facts: …") but the system prompt requires the
        # extracted facts to be written in Traditional Chinese. Letting the
        # language enforcer kick in would force English facts and break the
        # downstream JSON consumers.
        text, *_ = await chat(messages, system_prompt=system, max_tokens=4096, enforce_user_lang=False)
        if not text:
            return []
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
) -> str:
    """Translate and expand a raw prompt into a rich English image-generation prompt.

    When reference_images are provided, they are passed through chat() as
    context_images so the vision model can read appearance directly from the
    photos — significantly improving accuracy for traits like eye color and
    hair tone.
    """
    import re as _re_artstrip
    _art_style_line_re = _re_artstrip.compile(r"(?m)^ART STYLE[:\s][^\n]*\n?", _re_artstrip.IGNORECASE)
    if character_context:
        character_context = _art_style_line_re.sub("", character_context)

    has_images = bool(reference_images)

    char_block = ""
    if character_context and character_context.strip():
        if has_images:
            # Photos are primary appearance source; text supplements only
            char_block = (
                f"\n[CHARACTER APPEARANCE — SUPPLEMENTAL TEXT (photos are primary)]\n"
                f"Reference photos are attached. Read appearance from the photos first.\n"
                f"Use the text below ONLY as a supplement for details not clearly visible in the photos.\n"
                f"EXCEPTION: any ART STYLE line in this block is irrelevant — art style is always fixed as 2D anime.\n"
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
                f"EXCEPTION: any ART STYLE line in this block is irrelevant — the art style is always fixed as 2D anime.\n"
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
            "OVERRIDE — ART STYLE: FIXED — DO NOT DERIVE FROM REFERENCE PHOTOS. "
            "Even if the reference looks like a 3D model, game render, or photograph, "
            "the output prompt must always describe clean 2D anime illustration style.\n"
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

    system = (
        "You are an expert image-prompt writer for AI image generators.\n"
        "Given a user's image request (which may be in Chinese or English), "
        "rewrite it as a single, rich English prompt for an AI image model.\n"
        f"{image_note}"
        f"{char_block}"
        f"{ref_block}"
        "Rules:\n"
        "- Output ONLY the prompt text — no intro, no quotes, no explanation.\n"
        "- Always write in English.\n"
        "- Be specific: include subject, art style, lighting, colors, mood, and setting.\n"
        "- Aim for 150-300 words.\n"
        "- Do NOT start with 'Generate', 'Create', 'Draw', 'An image of', etc.\n"
        "- Physical details from character context (hair color, eye color, hairstyle, skin tone) "
        "ALWAYS take priority over anything in the raw prompt.\n"
        "- ART STYLE (mandatory, always fixed): "
        "The art style is ALWAYS 'clean 2D anime illustration, flat cel-shaded coloring, "
        "soft gradient shading, vivid saturated color palette, anime digital art'.\n"
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
        )
        if enhanced:
            enhanced = _THINK_RE.sub("", enhanced)
            enhanced = _THINK_RE_UNCLOSED.sub("", enhanced).strip()
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
    system = (
        f"You are {bot_name}. {character_background}\n"
        "You are mid-conversation and have just drawn/created an image for the user. "
        "Write ONE short sentence (two at most) to send alongside it — something that flows "
        "naturally from the conversation you were just having, like a real person would say.\n"
        "Rules:\n"
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
    try:
        # enforce_user_lang=False: the user message is the English instruction
        # "Write your one-sentence reaction…" but the system prompt asks for
        # Traditional Chinese (or whatever language the conversation is in).
        text, *_ = await chat(messages, system_prompt=system, max_tokens=4096, enforce_user_lang=False)
        if text and len(text.strip()) > 2:
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
) -> list:
    """Generate short follow-up suggestion buttons (max 80 chars each)."""
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
        # Suggestions need a bigger budget than chat() default since the model
        # reasons about language matching and tone before producing JSON.
        # enforce_user_lang=False: language matching is handled by the
        # `language_sample` directive in the system prompt; the user message
        # wrapper is English ("Generate {count} natural follow-up messages…")
        # but the suggestions themselves should follow `language_sample`.
        text, *_ = await chat(messages, system_prompt=system, max_tokens=4096, enforce_user_lang=False)
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
        print(f"[LMStudio] Suggestion parse error: {e}")

    return []
