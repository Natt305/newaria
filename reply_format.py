"""Shared chat-formatting utilities for AI backends.

Centralises the suggestion-salvage parser, the suggestion-button generator
pipeline, and the Discord post-processor (quote bolding, narration
italicising, self-name prefix stripping) that were previously duplicated
across `lmstudio_ai.py`, `groq_ai.py`, and `ollama_ai.py`. Backend modules
should import from here so any fix lands in one place.

Public surface:

* `salvage_suggestions(text, count)`
* `suggestion_log_snippet(text, limit=200)`
* `generate_suggestions(chat_fn, log_prefix, *, ...)`
* `SUGGESTION_LIST_PREFIX_RE`
* `bold_quoted_dialogue(text)`
* `italicize_pure_narration(text)`
* `strip_self_name_prefix(text, character_name="")`
* `format_for_discord(text, character_name="")`
* `DIALOGUE_QUOTE_PAIRS`, `PROTECTED_BOLD_RE`, `PROTECTED_ITALIC_RE`,
  `PLACEHOLDER_RE`, `CR_NAME_PREFIX_RE`, `SELF_NAME_SEP_CLASS`
"""

from __future__ import annotations

import json as _json
import re
from typing import Any, Awaitable, Callable, Optional


# ---------------------------------------------------------------------------
# Suggestion salvage parser
# ---------------------------------------------------------------------------

# Strips numbered/bulleted list prefixes ("1.", "1)", "1:", "-", "*", "•")
# so the salvage parser can recover suggestions from non-JSON output shapes
# that Mistral-family models often emit instead of a clean JSON array.
SUGGESTION_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.):])\s+")


def salvage_suggestions(text: str, count: int) -> list:
    """Recover up to ``count`` suggestions from non-JSON model output.

    Handles the common Mistral-family output shapes that appear when JSON
    extraction fails: numbered lists (``1. Hi``, ``1) Hi``, ``1: Hi``),
    bullet lists (``- Hi``, ``* Hi``, ``• Hi``), and plain newline-
    separated lines. Strips enclosing ASCII or curly quotes, trailing
    punctuation, and only keeps lines whose visible text is between 5 and
    80 characters (Discord button label hard limit).
    """
    if not text:
        return []

    def _clean(candidate: str) -> Optional[str]:
        line = candidate.strip()
        if not line:
            return None
        # Drop a trailing comma left over from JSON-list shapes
        line = line.rstrip(",").strip()
        # Drop enclosing quotes (straight or curly)
        if len(line) >= 2 and line[0] in "\"'\u201c\u2018" and line[-1] in "\"'\u201d\u2019":
            line = line[1:-1].strip()
        # Drop trailing sentence punctuation per the original suggestion contract
        line = line.rstrip(".!?。！？").strip()
        if not (5 <= len(line) <= 80):
            return None
        # Skip lines that look like wrapper prose ("Here are 3 suggestions:")
        if line.endswith(":"):
            return None
        return line

    out: list = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        # Drop list prefixes (numbered/bulleted)
        line = SUGGESTION_LIST_PREFIX_RE.sub("", line, count=1).strip()
        # Mistral-family models often emit each suggestion as its own
        # one-element JSON array on a separate line:
        #   ["I didn't mean any harm, really."]
        #   ["I got lost on my way back."]
        # The top-level JSON parser fails on the second line, so we land
        # here. Try to JSON-decode each line first to peel the `[ ]`
        # (and inner quotes) off cleanly. A list with multiple strings
        # is also expanded so we don't lose entries.
        candidates: list[str] = []
        try:
            parsed = _json.loads(line)
            if isinstance(parsed, str):
                candidates.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        candidates.append(item)
            # Other JSON scalars (`null`, `false`, numbers, objects) are
            # dropped — falling back to the raw line would let `false` slip
            # through as a button label.
        except Exception:
            # JSON parse failed — but the line may still be a single-quoted
            # or curly-quoted "array" shape (`['text']`, `[“text”]`) that
            # is technically not JSON. Strip a single pair of outer
            # brackets here; `_clean` then strips the inner quote pair
            # (single, double, or curly) on the next pass.
            inner = line
            if len(inner) >= 2 and inner[0] == "[" and inner[-1] == "]":
                inner = inner[1:-1].strip().rstrip(",").strip()
            candidates.append(inner)

        for cand in candidates:
            cleaned = _clean(cand)
            if cleaned is not None:
                out.append(cleaned)
                if len(out) >= count:
                    return out
    return out


def suggestion_log_snippet(text: str, limit: int = 200) -> str:
    """Collapse newlines + truncate raw model output for a one-line log."""
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


# ---------------------------------------------------------------------------
# Post-generation perspective filter
# ---------------------------------------------------------------------------

# Matches *action* or *narration* asterisk formatting — a marker of bot-
# voiced roleplay prose, not casual player messages.
_ASTERISK_ACTION_RE = re.compile(r"\*[^*]+\*")

# Third-person verb list covering typical roleplay narration patterns
# ("Kelly smiled", "Kelly said", "Kelly's eyes…").
_BOT_THIRD_PERSON_VERB_RE_TEMPLATE = (
    r"\b{name}\b.{{0,6}}"
    r"(said|says|smiled|smiles|nods|nodded|asked|asks|"
    r"replied|replies|looked|looks|walked|walks|turned|turns|"
    r"thought|thinks|felt|feels|stood|sat|sits|sighed|sighs|"
    r"whispered|whispers|laughed|laughs|glanced|glances)"
)


def _is_bot_voiced(text: str, bot_name: str) -> bool:
    """Return True when a suggestion sounds like the bot speaking, not the player.

    Criteria (any one sufficient):
    a. Longer than 90 characters — casual player messages are short.
    b. Uses ``*action*`` or ``*narration*`` asterisk formatting — bot's style.
    c. References ``bot_name`` in a third-person narration pattern
       ("Kelly smiled…", "Kelly's voice…") rather than as a direct address
       ("Hey Kelly, …" is fine and returns False).
    """
    if len(text) > 90:
        return True
    if _ASTERISK_ACTION_RE.search(text):
        return True
    if bot_name:
        first_name = bot_name.split()[0]
        third_person_re = re.compile(
            _BOT_THIRD_PERSON_VERB_RE_TEMPLATE.format(name=re.escape(first_name)),
            re.I,
        )
        if third_person_re.search(text):
            return True
        if re.search(r"\b" + re.escape(first_name) + r"'s\b", text, re.I):
            return True
    return False


# ---------------------------------------------------------------------------
# Discord-format post-processor for plain-prose models like Celeste.
# ---------------------------------------------------------------------------
# Mistral-family roleplay fine-tunes ignore the prompt-level "wrap dialogue
# in **bold**" instruction and emit raw prose with quoted dialogue. We bold
# the quoted segments and italicise pure-narration paragraphs in code so
# the reply renders correctly in Discord regardless of model compliance.

# Quote pairs we treat as dialogue. Lone apostrophes (`'`) are deliberately
# excluded — they collide with English contractions ("Kelly's", "You're").
DIALOGUE_QUOTE_PAIRS: tuple[tuple[str, str], ...] = (
    ('"', '"'),       # straight ASCII double quotes
    ("\u201c", "\u201d"),  # curly double quotes "..."
    ("\u300c", "\u300d"),  # Chinese corner brackets 「...」
    ("\u300e", "\u300f"),  # Chinese white corner brackets 『...』
)


# Patterns for detecting already-formatted spans. Used to mask them out
# before applying quote-bolding so we never re-wrap quotes that are
# already inside **bold** or *italic* spans (or merely adjacent to them).
PROTECTED_BOLD_RE = re.compile(r"\*\*[^*\n]+?\*\*")
PROTECTED_ITALIC_RE = re.compile(r"\*[^*\n]+?\*")
PLACEHOLDER_RE = re.compile(r"\x00(\d+)\x00")


# Title-Case "CharacterName: " prefix used by both `_parse_reply_format`
# (ChatML path) and `strip_self_name_prefix` (Discord post-processor)
# heuristic fallback. Kept here so the two paths stay in lockstep.
CR_NAME_PREFIX_RE = re.compile(
    r"^[A-Z][A-Za-z]{1,24}(?: [A-Z][A-Za-z]{1,24})?: ", re.MULTILINE
)


# Punctuation that Mistral-family models put between their own character
# name and the rest of a reply when they ignore the "no name prefix"
# instruction. Covers `:`, em-dash `—`, en-dash `–`, and ASCII `-`.
SELF_NAME_SEP_CLASS = r"[:\u2014\u2013\-]"


def bold_quoted_dialogue(text: str) -> str:
    """Wrap quoted dialogue segments in **bold** when not already wrapped.

    Idempotency strategy: before processing, all existing `**...**` and
    `*...*` spans are extracted into placeholders, so quotes that already
    live inside formatted spans (or are merely adjacent to them) are
    never re-wrapped. After processing, the placeholders are restored.
    Re-running on the helper's own output is a guaranteed no-op.

    For same-char delimiters (straight `"`), the regex is non-greedy
    so quotes are paired in order (1↔2, 3↔4, …). The common Mistral
    shape ``"…dialogue…" *narration* "…dialogue…"`` has 4 quotes on
    one line and pairs cleanly; only an ODD count is rejected as
    unbalanced. Truly nested same-quote dialogue (e.g.
    ``"He said "go" quietly"``) would mis-pair, but that pattern is
    virtually never produced by the models we target — they switch to
    curly or single quotes for nesting. Asymmetric delimiters (`「」`,
    `『』`, curly `""`) are immune to nesting confusion since the open
    and close characters differ.

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

    masked = PROTECTED_BOLD_RE.sub(_stash, text)
    masked = PROTECTED_ITALIC_RE.sub(_stash, masked)

    for open_q, close_q in DIALOGUE_QUOTE_PAIRS:
        if open_q == close_q:
            inner = rf"[^{re.escape(close_q)}\n]+?"
            pattern = re.compile(
                rf"{re.escape(open_q)}({inner}){re.escape(close_q)}"
            )
            new_lines = []
            for line in masked.split("\n"):
                count = line.count(open_q)
                if count == 0 or count % 2 != 0:
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

    return PLACEHOLDER_RE.sub(_unstash, masked)


def italicize_pure_narration(text: str) -> str:
    """Wrap whole paragraphs of unformatted narration in *italics*.

    Only touches paragraphs that contain NO existing markdown (`*`, `_`)
    and NO dialogue quotes. Mixed paragraphs (italic narration + bolded
    dialogue) are left alone — their dialogue is already handled by
    `bold_quoted_dialogue`, and re-wrapping the whole thing would
    double-up markers.
    """
    quote_chars = {open_q for open_q, _ in DIALOGUE_QUOTE_PAIRS} | {
        close_q for _, close_q in DIALOGUE_QUOTE_PAIRS
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


def strip_self_name_prefix(text: str, character_name: str = "") -> str:
    """Strip a leading self-name label from each paragraph of `text`.

    Mistral-family models (Celeste etc.) frequently begin a reply (or each
    paragraph of a reply) with the bot's own character name as a label, e.g.
    ``Kelly Gray: ...``, ``Kelly Gray — ...``, or just ``Kelly: ...``. The
    Discord-formatter then has nothing to do with the line — there are no
    quotes to bold — and `italicize_pure_narration` wraps the whole thing
    in ``*...*``, producing the redundant ``*Kelly Gray: ...*`` look
    reported by users.

    Two modes:

    * **Name-aware** (`character_name` provided): strip ONLY exact matches
      of the bot's own name (including the first token of a multi-word
      name, so ``Kelly Gray`` also matches a bare ``Kelly:``). This is
      safe to apply unconditionally because legitimate prefixes like
      ``Note:`` or another character's quoted line cannot match.

    * **Heuristic fallback** (no name): use the same Title-Case heuristic
      that `_parse_reply_format` already uses on the Qwen path
      (`CR_NAME_PREFIX_RE`). Conservative — accepts a single word or
      two-word Title-Case label followed by ``: ``.

    Operates per paragraph (split on ``\\n\\n``), so a colon in the middle
    of narration (e.g. ``She glanced at the clock: 11:47 PM.``) is never
    touched. A leading ``*`` from pre-existing italics is preserved by
    re-attaching it after the strip.
    """
    if not text:
        return text

    name = (character_name or "").strip()
    name_pattern: Optional[re.Pattern] = None
    if name:
        # Build the per-name pattern once. Match either the full name
        # (case-insensitive) or just its first token, followed by an
        # optional `*` (preceding italic marker survives re-attachment),
        # one of the separator characters, and optional whitespace.
        first_token = name.split()[0]
        alternatives = [re.escape(name)]
        if first_token and first_token.lower() != name.lower():
            alternatives.append(re.escape(first_token))
        name_pattern = re.compile(
            rf"^\s*({'|'.join(alternatives)})\s*{SELF_NAME_SEP_CLASS}\s*",
            re.IGNORECASE,
        )

    out_paragraphs: list[str] = []
    for para in text.split("\n\n"):
        # Preserve a leading `*` (italic opener) so the rest of the
        # formatter still sees the same wrapping after the strip.
        leading_star = ""
        body = para
        stripped_body = body.lstrip()
        if stripped_body.startswith("*") and not stripped_body.startswith("**"):
            leading_ws = body[: len(body) - len(stripped_body)]
            leading_star = leading_ws + "*"
            body = stripped_body[1:]

        if name_pattern is not None:
            new_body = name_pattern.sub("", body, count=1)
        else:
            # Heuristic: Title-Case word, optional second Title-Case
            # word, then `: `. Re-uses the existing Qwen-path constant
            # so the two paths stay in lockstep.
            new_body = CR_NAME_PREFIX_RE.sub("", body, count=1)

        if leading_star:
            new_body = f"{leading_star}{new_body}"
        out_paragraphs.append(new_body)

    return "\n\n".join(out_paragraphs)


def format_for_discord(text: str, character_name: str = "") -> str:
    """Convert plain-prose roleplay output into Discord-ready markdown.

    Three passes:
      0. Strip a leading self-name label from each paragraph
         (e.g. ``Kelly Gray: ...`` → ``...``) so the bolder/italicizer
         doesn't have to deal with a redundant prefix.
      1. Bold any quoted dialogue segments not already inside **...**.
      2. Italicise paragraphs that are pure narration with no existing
         formatting and no dialogue quotes.

    Designed for Mistral-family models (Celeste etc.) that ignore the
    prompt-level formatting instruction. Qwen / hauhaucs ChatML output is
    handled by `_parse_reply_format()` and must NOT pass through this.
    """
    if not text:
        return text
    text = strip_self_name_prefix(text, character_name=character_name)
    text = bold_quoted_dialogue(text)
    text = italicize_pure_narration(text)
    return text


# ---------------------------------------------------------------------------
# Shared suggestion-button generator pipeline
# ---------------------------------------------------------------------------
# The three backend modules (`lmstudio_ai`, `groq_ai`, `ollama_ai`) used to
# carry near-identical ~100-line `generate_suggestions` implementations.
# They diverged only in (a) which `chat()` flavour they invoked, (b) their
# `[Backend]` log prefix, and (c) whether the prompt asked for an `{"items":
# [...]}` object root or a bare JSON array. Every other line — prompt
# construction, JSON-parse → bracket-extract → salvage → 80-char clamp —
# was copy-pasted three times, which is exactly the bug pattern the
# `salvage_suggestions` consolidation just fixed. Anything that flows
# through the JSON-or-prose recovery pipeline now lives here so a future
# tweak (length cap, language-detection nudge, log shape) lands in one
# place.

# Discord button labels are hard-capped at 80 visible characters; the
# 77-char ellipsis trim is part of the "behaviour is identical" contract.
_BUTTON_LABEL_MAX = 80
_BUTTON_LABEL_ELLIPSIS_AT = 77


def _clamp_button_label(s: str) -> str:
    """Clamp a button label to 80 chars, replacing the tail with `...`."""
    if len(s) > _BUTTON_LABEL_MAX:
        s = s[:_BUTTON_LABEL_ELLIPSIS_AT] + "..."
    return s


def _clean_json_label(s: Any) -> str:
    """Normalise a JSON-extracted label: stringify, strip, drop trailing dot, clamp."""
    return _clamp_button_label(str(s).strip().rstrip("."))


def _parse_json_array_payload(text: str) -> Optional[list]:
    """Extract a list-of-strings payload from `text`.

    Mirrors the per-backend helpers of the same name. Handles both shapes
    that flow through `generate_suggestions`:
      - JSON-Schema / json_object mode wraps the array in `{"items":[...]}`
        (the schema's root must be an object for OpenAI strict-mode
        compatibility).
      - Free-form mode emits a bare JSON array, possibly with surrounding
        prose; we bracket-extract from the first `[` to the last `]`.
    Returns ``None`` when neither shape parses, so the caller can fall
    through to its salvage path.
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


# Type alias for the per-backend chat callable supplied to
# `generate_suggestions`. The wrapper is responsible for applying any
# backend-specific kwargs (max_tokens, response_format, enforce_user_lang)
# and returning just the model's raw text reply.
ChatFn = Callable[[list, str], Awaitable[str]]


async def generate_suggestions(
    chat_fn: ChatFn,
    log_prefix: str,
    *,
    topic: str,
    bot_name: str,
    character_background: str,
    count: int = 3,
    guiding_prompt: str = "",
    language_sample: str = "",
    prompt_object_root: bool = False,
    recent_history: list = None,
) -> list:
    """Generate short follow-up suggestion buttons (max 80 chars each).

    Backend-agnostic implementation of the JSON-parse → bracket-extract →
    salvage pipeline previously copy-pasted across the three AI modules.

    Parameters
    ----------
    chat_fn:
        Async callable ``(messages, system_prompt) -> str`` that invokes
        the backend's underlying `chat()` with whatever extra kwargs that
        backend needs (e.g. `response_format`, `max_tokens=4096`,
        `enforce_user_lang=False` on LM Studio). Must return the model's
        raw text reply, or an empty string on failure.
    log_prefix:
        Bracketed backend name used in log lines (e.g. ``"LMStudio"``,
        ``"Groq"``, ``"Ollama"``).
    prompt_object_root:
        When True, the "Return ONLY ..." line in the system prompt asks
        for a ``{"items":[...]}`` object root instead of a bare array.
        Set this when the backend has constrained-decoding mode active
        (Groq json_object, Ollama format=json). Leave False on LM Studio:
        its JSON Schema mode wraps the array in `items` automatically and
        the prompt has historically always asked for a bare array.
    recent_history:
        Optional list of recent conversation messages in
        ``[{"role": "user"|"assistant", "content": str}, ...]`` format.
        The last 2–4 exchange pairs are injected as few-shot examples
        before the final generation request so the model can match tone,
        vocabulary, and energy from the live conversation rather than
        relying on the character description alone.
    """
    lang_instruction = ""
    if language_sample and language_sample.strip():
        lang_instruction = (
            f"IMPORTANT: Write every suggestion in the EXACT same language as this sample text "
            f"(detect it automatically — do NOT translate): \"{language_sample[:120]}\"\n"
        )

    if prompt_object_root:
        return_line = (
            'Return ONLY a valid JSON object of the shape {"items": [<suggestion strings>]}. '
            'No markdown, no code fences.\n'
        )
    else:
        return_line = "Return ONLY a valid JSON array of strings. No markdown, no code fences.\n"

    base = (
        f"{lang_instruction}"
        f"You are writing short follow-up messages FROM the user's side, directed at "
        f"{bot_name} during roleplay. Do NOT write as {bot_name} — write as the USER.\n"
        f"NEVER write dialogue that {bot_name} would say. "
        f"NEVER continue from {bot_name}'s voice. "
        f"NEVER use asterisks (*) for actions or narration — those belong to {bot_name}'s replies. "
        f"Each suggestion MUST be a message typed by the human player. "
        f"If a line sounds like something {bot_name} would say — formal, authoritative, "
        f"in-character narration — discard it and write a casual human reply instead.\n"
        f"{bot_name}'s personality and appearance for context: {character_background}\n"
        f"Generate exactly {count} follow-up messages a user might naturally send next in the conversation.\n"
        f"Write them as the USER speaking to {bot_name} — casual, warm, and conversational.\n"
        f"Each should be 10–75 characters. No punctuation at the end. No quotes.\n"
        f"{return_line}"
    )
    if guiding_prompt:
        system = f"{guiding_prompt}\n\n{base}"
    else:
        system = base

    if topic and topic.strip():
        prompt = (
            f"Context of the conversation so far:\n{topic[:400]}\n\n"
            f"Generate {count} natural follow-up messages the user might send next."
        )
    else:
        prompt = (
            f"Generate {count} casual opening messages someone might send to start chatting with {bot_name}."
        )

    # Inject recent conversation turns as labeled lines embedded in the user
    # prompt rather than as separate role messages. This prevents smaller LM
    # Studio models from continuing in the last speaker's voice (which would
    # be the bot's, since assistant turns always come last in the history).
    # By collapsing history into a single user-role block with explicit
    # "Player: …" / "[bot_name]: …" labels, the final message the model sees
    # is always the user-role generation request — keeping it player-voiced.
    if recent_history:
        slice_start = max(0, len(recent_history) - 8)
        labeled_lines = []
        for msg in recent_history[slice_start:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not (role in ("user", "assistant") and content):
                continue
            label = "Player" if role == "user" else bot_name
            labeled_lines.append(f"{label}: {content[:300]}")
        if labeled_lines:
            history_block = "\n".join(labeled_lines)
            prompt = (
                f"Recent conversation (Player speaks to {bot_name}; "
                f"use this for tone/topic reference only):\n"
                f"{history_block}\n\n"
                f"{prompt}"
            )

    messages = [{"role": "user", "content": prompt}]

    text = ""
    try:
        text = await chat_fn(messages, system)
        if not text:
            print(f"[{log_prefix}] Suggestion: empty model response — no buttons")
            return []
        # Schema / json_object mode emits `{"items":[...]}`; legacy mode
        # emits a bare array. The shared parser handles both shapes.
        parsed = _parse_json_array_payload(text)
        if parsed is not None and len(parsed) > 0:
            candidates = [_clean_json_label(s) for s in parsed[:count]]
            filtered = [s for s in candidates if not _is_bot_voiced(s, bot_name)]
            dropped = len(candidates) - len(filtered)
            if dropped:
                print(f"[{log_prefix}] Suggestion: filtered {dropped} bot-voiced item(s)")
            if filtered:
                return filtered
        # Bracket-extract fallback for prose with an embedded array.
        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            arr = _json.loads(text[start:end])
            if isinstance(arr, list) and len(arr) > 0:
                candidates = [_clean_json_label(s) for s in arr[:count]]
                filtered = [s for s in candidates if not _is_bot_voiced(s, bot_name)]
                dropped = len(candidates) - len(filtered)
                if dropped:
                    print(f"[{log_prefix}] Suggestion (bracket): filtered {dropped} bot-voiced item(s)")
                if filtered:
                    return filtered
    except Exception as e:
        print(
            f"[{log_prefix}] Suggestion JSON parse error: {e} — "
            f"raw: {suggestion_log_snippet(text)!r}"
        )

    # Salvage path: when JSON extraction fails, recover suggestions from a
    # numbered/bulleted list or plain newline-separated lines. The salvage
    # parser already strips quotes/punctuation/length-filters to 5–80
    # chars, so we only re-clamp here (mirrors the original behaviour:
    # JSON path runs `.strip().rstrip(".")` + clamp; salvage path only
    # clamps).
    salvaged = salvage_suggestions(text, count)
    if salvaged:
        candidates = [_clamp_button_label(s) for s in salvaged]
        filtered = [s for s in candidates if not _is_bot_voiced(s, bot_name)]
        dropped = len(candidates) - len(filtered)
        if dropped:
            print(f"[{log_prefix}] Suggestion (salvage): filtered {dropped} bot-voiced item(s)")
        if filtered:
            print(
                f"[{log_prefix}] Suggestion: salvaged {len(filtered)} from non-JSON output — "
                f"raw: {suggestion_log_snippet(text)!r}"
            )
            return filtered

    print(
        f"[{log_prefix}] Suggestion: salvage failed too — "
        f"raw: {suggestion_log_snippet(text)!r}"
    )
    return []  # silently return no buttons rather than wrong-language fallbacks
