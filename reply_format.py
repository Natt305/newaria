"""Shared chat-formatting utilities for AI backends.

Centralises the suggestion-salvage parser and the Discord post-processor
(quote bolding, narration italicising, self-name prefix stripping) that
were previously duplicated across `lmstudio_ai.py`, `groq_ai.py`, and
`ollama_ai.py`. Backend modules should import from here so any fix lands
in one place.

Public surface:

* `salvage_suggestions(text, count)`
* `suggestion_log_snippet(text, limit=200)`
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
from typing import Optional


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
