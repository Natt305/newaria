"""
Scene image runner for LM Studio RP-mode replies.

Single shared entry point (`run_scene_image`) used by all three triggers:
  - 🎬 button click on a bot message
  - LLM-emitted [SCENE] auto-trigger
  - Visual-intent re-route (when scene mode is on for the channel)
  - Legacy [IMAGE: ...] re-route (when scene mode is on for the channel)

Guarantees:
  - At most one image per (channel_id, target_message_id) — in-flight set
  - Per-channel cooldown (3 generations / 60s)
  - Image is attached IN-PLACE via bot_message.edit(attachments=[...]),
    never as a separate reply (kills the doubled-response feel)
  - Same temporary-message progress bar UX the legacy flow uses

Per-channel toggle persists in the existing settings store under the key
`scene_image:{channel_id}` (bool, default False).

Engine tiers, in order of fidelity:
  - Qwen (ComfyUI):   full multi-ref edit + cinematic suffix, up to 4 refs
  - FLUX (ComfyUI):   single-ref multiref + cinematic suffix
  - Cloudflare:       plain text-to-image with cinematic suffix
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Optional

import discord
from discord.ext import commands

import database
import image_dispatch
import ai_backend as groq_ai


CINEMATIC_SUFFIX = (
    ", medium-close portrait shot, face prominent in frame, "
    "dramatic lighting, expressive composition"
)


def _append_looks_enabled() -> bool:
    """Return True when QWEN_APPEND_LOOKS resolves to on (default on)."""
    raw = (os.environ.get("QWEN_APPEND_LOOKS", "on") or "on").strip().lower()
    return raw in ("on", "true", "1", "yes", "enabled", "enable")


def _cinematic_suffix_enabled() -> bool:
    """Return True when SCENE_CINEMATIC_SUFFIX resolves to on (default on)."""
    raw = (os.environ.get("SCENE_CINEMATIC_SUFFIX", "on") or "on").strip().lower()
    return raw in ("on", "true", "1", "yes", "enabled", "enable")


# ── Scene-state / clothing-override helpers ───────────────────────────────────

# Regex patterns that signal the character is NOT in their default outfit.
# Each entry is (compiled_regex, human_readable_label).
#
# Design notes:
#   - bath action patterns do NOT include "bathroom" (a location noun that does
#     not imply undressed state on its own) — only action/state variants like
#     "bath", "bathing", "bathtub", "bathwater" are matched.
#   - "bed" is matched only as an explicit "in bed" phrase to avoid false
#     positives from words like "bedroom" or "bedside".
#   - "changing" is matched only when adjacent to clothes-related context to
#     avoid matching phrases like "a changing situation".
_SCENE_CLOTHING_STATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bshower(?:ing)?\b", re.I), "in the shower (unclothed)"),
    # Match bath ACTIONS only — not "bathroom" (location noun, may be clothed)
    (re.compile(r"\bbath(?:ing|tub|water)?\b", re.I), "in the bath (unclothed)"),
    (re.compile(r"\bnude\b", re.I), "nude"),
    (re.compile(r"\bnaked\b", re.I), "naked"),
    (re.compile(r"\bundress(?:ed|ing)?\b", re.I), "undressed"),
    (re.compile(r"\bno\s+clothes\b", re.I), "without clothes"),
    (re.compile(r"\bwithout\s+clothes\b", re.I), "without clothes"),
    (re.compile(r"\bbare\s+skin\b", re.I), "bare skin showing"),
    (re.compile(r"\bbare\s+bod(?:y|ies)\b", re.I), "bare body"),
    (re.compile(r"\btowel\b", re.I), "wrapped in a towel (just bathed)"),
    (re.compile(r"\bwashing\s+(?:her|his|their|up|body|hair)\b", re.I), "washing (unclothed)"),
    # Sleeping / resting in bed — typically undressed or in nightwear
    (re.compile(r"\bsleep(?:ing)?\b", re.I), "sleeping (nightwear/undressed)"),
    (re.compile(r"\basleep\b", re.I), "asleep (nightwear/undressed)"),
    (re.compile(r"\bin\s+bed\b", re.I), "in bed (nightwear/undressed)"),
    # Actively changing clothes
    (re.compile(r"\bchanging\s+(?:clothes|outfits?|into|out of)\b", re.I), "changing clothes"),
    (re.compile(r"\bgetting\s+(?:dressed|undressed|changed)\b", re.I), "changing clothes"),
    # Explicit intimate states
    (re.compile(r"\bmaking\s+love\b", re.I), "intimate (unclothed)"),
    (re.compile(r"\bhaving\s+sex\b", re.I), "intimate (unclothed)"),
]

# Compiled word-boundary regex for outfit/weapon sentence detection.
# Using a single combined regex (instead of substring membership) prevents
# false matches from short tokens: e.g. "tied" won't match \btie\b,
# "undressed" won't double-match via "dress".
_OUTFIT_SENTENCE_RE: re.Pattern = re.compile(
    r"\b(?:"
    + "|".join(re.escape(w) for w in sorted([
        "wear", "wearing", "wears", "wore", "worn",
        "outfit", "clothes", "clothing", "garment", "attire",
        "dressed", "undressed",
        "coat", "jacket", "uniform", "suit", "dress", "shirt", "blouse",
        "pants", "trousers", "skirt", "shorts", "sweater", "hoodie",
        "holster", "gun", "pistol", "weapon", "rifle", "firearm",
        "sword", "blade", "combat", "tactical", "ammo",
        "belt", "strap", "harness",
        "gloves", "boots", "shoes", "heels", "sneakers",
        "scarf", "tie",
    ], key=len, reverse=True))  # longer tokens first to avoid partial matches
    + r")\b",
    re.I,
)


def _detect_scene_clothing_override(text: str) -> tuple[bool, str]:
    """Return (triggered, state_label) when *text* implies the character is not
    wearing their default outfit (shower, bath, nude, sleeping, etc.).

    Checks both the seed and the enriched prompt text combined — pass
    ``seed + " " + enriched_base`` as *text*.
    """
    for pattern, label in _SCENE_CLOTHING_STATE_PATTERNS:
        if pattern.search(text):
            return True, label
    return False, ""


def _strip_outfit_from_looks(looks_text: str, *, fallback_to_empty: bool = False) -> str:
    """Remove clothing/weapon/outfit sentences from a character looks description.

    Splits *looks_text* on sentence boundaries and drops any sentence that
    contains an outfit-related keyword, keeping only base identity traits
    (hair colour, eye colour, skin tone, facial features, build/height).

    Args:
        looks_text: The full character looks description.
        fallback_to_empty: When True, returns '' if all sentences are filtered
            (safe for scene-state override mode — avoids re-introducing outfit
            text). When False (default), returns the original text as a safety
            fallback (for non-override callers).
    """
    sentences = re.split(r"(?<=[.!?])\s+", looks_text.strip())
    kept: list[str] = []
    for sentence in sentences:
        if _OUTFIT_SENTENCE_RE.search(sentence):
            continue
        kept.append(sentence)
    result = " ".join(kept).strip()
    if result:
        return result
    return "" if fallback_to_empty else looks_text


# Last-generation debug capture for /scenedebug. Overwritten per generate.
_LAST_SCENE_GENERATION: dict = {}


def get_last_scene_generation() -> dict:
    """Return a snapshot of the most recent scene-image generation."""
    return dict(_LAST_SCENE_GENERATION)


def _record_last_scene_generation(**fields) -> None:
    try:
        _LAST_SCENE_GENERATION.clear()
        _LAST_SCENE_GENERATION.update(fields)
        _LAST_SCENE_GENERATION["timestamp"] = time.time()
    except Exception as exc:
        print(f"[SceneImage] _record_last_scene_generation failed: {type(exc).__name__}: {exc}")


# ── State ─────────────────────────────────────────────────────────────────────

_inflight: set[tuple[str, int]] = set()
_cooldown = commands.CooldownMapping.from_cooldown(
    3, 60.0, commands.BucketType.channel
)

# ── Per-channel location cache ────────────────────────────────────────────────
# Keyed by str(channel_id). Stores the last known location/setting phrase
# extracted from an enhanced scene prompt. Used as a fallback anchor when the
# LMStudio scene-only enhancer omits location on emotionally-charged turns.

_scene_location_cache: dict[str, str] = {}

# TTL for persisted scene location entries. Entries older than this are
# discarded on load and on write. Configurable via SCENE_LOCATION_TTL_DAYS
# (default: 7 days). Set to 0 to disable expiry.
try:
    _SCENE_LOCATION_TTL_SECONDS: float = float(os.getenv("SCENE_LOCATION_TTL_DAYS", "7")) * 86400.0
except (ValueError, TypeError):
    print("[SceneImage] Invalid SCENE_LOCATION_TTL_DAYS value; defaulting to 7 days.")
    _SCENE_LOCATION_TTL_SECONDS = 7 * 86400.0

# Populate the cache from the persistent settings store on module load.
# Each persisted entry uses the key "scene_location:{channel_id}" and is
# stored as {"location": str, "ts": float}. Plain-string entries (legacy
# format, no timestamp) are treated as expired and removed immediately.
def _load_location_cache() -> None:
    """Load non-stale scene location entries from persistent store into cache."""
    now = time.time()
    for raw_key, value in database.get_settings_by_prefix("scene_location:").items():
        channel_id = raw_key[len("scene_location:"):]
        location: str | None = None

        if isinstance(value, dict):
            ts = value.get("ts")
            loc = value.get("location")
            if (
                isinstance(ts, (int, float))
                and isinstance(loc, str)
                and loc
                and (_SCENE_LOCATION_TTL_SECONDS <= 0 or (now - ts) < _SCENE_LOCATION_TTL_SECONDS)
            ):
                location = loc
        # Plain strings (old format, no timestamp) are treated as expired — fall through.

        if location:
            _scene_location_cache[channel_id] = location
        else:
            # Entry is stale or in legacy format — purge it from the store.
            database.delete_setting(raw_key)
            print(f"[SceneImage] Expired stale location entry for channel {channel_id}")

_load_location_cache()

# Two patterns cover the two main phrasings the scene-only enhancer uses:
#   1. Prepositional:  "in/inside/at/within [article] [adj…] NOUN"
#   2. Sentence-start: "[article] [adj…] NOUN" at the start of a sentence
# Both share the same noun list — common English setting nouns.
_LOCATION_NOUN_GROUP = (
    r"(?:room|chamber|bedchamber|boudoir|parlour|salon|foyer|office|corridor|"
    r"bedroom|hallway|hall|street|rooftop|roof|"
    r"kitchen|bathroom|bath|library|forest|garden|courtyard|alley|laboratory|lab|"
    r"dungeon|arena|balcony|tavern|inn|castle|palace|mansion|apartment|"
    r"classroom|gym|hospital|clinic|park|plaza|beach|shore|mountain|field|meadow|"
    r"cave|tunnel|bridge|studio|stage|workshop|warehouse|basement|attic|"
    r"lobby|staircase|terrace|patio|porch|yard|road|path|throne)"
)

# Pattern 1: preposition-led phrase (most reliable)
_LOCATION_EXTRACT_RE = re.compile(
    r"\b(?:in|inside|within|at|on)\b\s+(?:an?\b|the\b|this\b|her\b|his\b|their\b)?\s*"
    r"((?:[\w''-]+\s+){0,4}" + _LOCATION_NOUN_GROUP + r")",
    re.IGNORECASE,
)

# Pattern 2: sentence-initial noun phrase (no preposition).
# Matches at the very start of the string or right after a sentence boundary.
_LOCATION_SENTENCE_START_RE = re.compile(
    r"(?:(?:^|(?<=\. )|(?<=\.\n)))"
    r"(?:an?\b|the\b|this\b|her\b|his\b|their\b)?\s*"
    r"((?:[\w''-]+\s+){0,4}" + _LOCATION_NOUN_GROUP + r")",
    re.IGNORECASE | re.MULTILINE,
)


def clear_scene_location_cache(channel_id) -> None:
    """Remove the cached scene location for a channel (call on /clear)."""
    _scene_location_cache.pop(str(channel_id), None)
    database.delete_setting(f"scene_location:{channel_id}")


def _update_location_cache(channel_id: str, enhanced_prompt: str) -> None:
    """Scan an enhanced prompt and cache the first location phrase found.

    Tries the preposition-led pattern first (most reliable), then falls back
    to the sentence-initial pattern to catch outputs that open directly with
    a noun phrase (e.g. "Bedchamber at night, candlelight…").
    """
    m = _LOCATION_EXTRACT_RE.search(enhanced_prompt)
    if m is None:
        m = _LOCATION_SENTENCE_START_RE.search(enhanced_prompt)
    if m:
        location = m.group(1).strip().rstrip(".,;:")
        if location:
            _scene_location_cache[channel_id] = location
            database.set_setting(
                f"scene_location:{channel_id}",
                {"location": location, "ts": time.time()},
            )
            print(f"[SceneImage] Location cache updated for channel {channel_id}: {location!r}")


# ── Per-channel toggle ────────────────────────────────────────────────────────

def _setting_key(channel_id) -> str:
    return f"scene_image:{channel_id}"


def is_scene_mode_on(channel_id) -> bool:
    return bool(database.get_setting(_setting_key(channel_id)))


def set_scene_mode(channel_id, enabled: bool) -> bool:
    return database.set_setting(_setting_key(channel_id), bool(enabled))


# ── Auto-trigger toggle (guild-wide default + per-channel override) ───────────
#
# Keys:
#   scene_auto:guild:{guild_id}  — guild-wide default (bool or None)
#   scene_auto:{channel_id}      — per-channel override (True/False/None)
#
# Resolution order: channel-explicit → guild default → False (off).
# Storing None in a channel key removes the override so the guild default
# applies again (uses set_setting(key, None) convention).

def _auto_guild_key(guild_id) -> str:
    return f"scene_auto:guild:{guild_id}"


def _auto_channel_key(channel_id) -> str:
    return f"scene_auto:{channel_id}"


def is_auto_scene_on(channel_id, guild_id=None) -> bool:
    """Check if auto-trigger is enabled.  Channel override wins over guild default."""
    chan_val = database.get_setting(_auto_channel_key(channel_id))
    if chan_val is not None:
        return bool(chan_val)
    if guild_id is not None:
        guild_val = database.get_setting(_auto_guild_key(guild_id))
        if guild_val is not None:
            return bool(guild_val)
    return False


def set_auto_scene_mode(channel_id, enabled) -> bool:
    """Set per-channel auto-trigger override.  Pass None to clear (inherit guild)."""
    return database.set_setting(_auto_channel_key(channel_id), enabled)


def set_guild_auto_scene_mode(guild_id, enabled: bool) -> bool:
    """Set guild-wide auto-trigger default."""
    return database.set_setting(_auto_guild_key(guild_id), bool(enabled))


# ── Visual-intent reuse ───────────────────────────────────────────────────────

# Compact visual-intent regex used when scene mode is on but the LLM did not
# emit [SCENE] and there's no [IMAGE: ...] tag — covers EN/ZH "show me / I
# want to see" phrasing. Looser than lmstudio_ai.IMAGE_REQUEST_PATTERNS so
# RP-mode users don't have to phrase requests as image commands.
_VISUAL_INTENT_RE = re.compile(
    r"(?:show\s+me|let\s+me\s+see|i\s+want\s+to\s+see|"
    r"draw\s+me|paint\s+me|illustrate|"
    r"我想看|我要看|讓我看|給我看|看看妳|看看你|"
    r"幫我畫|畫一下|畫個|長什麼樣|長甚麼樣)",
    re.I,
)


def is_user_visual_intent(text: str) -> bool:
    """True if the user message reads like a visual request (loose detector)."""
    if not text:
        return False
    if _VISUAL_INTENT_RE.search(text):
        return True
    # Also defer to the lmstudio backend's own self-ref detector when present
    try:
        return bool(groq_ai.is_self_referential_image(text))
    except Exception:
        return False


# ── SCENE marker (re-exported so bot.py can strip server-side too) ────────────

SCENE_MARKER_RE = re.compile(r"\[\s*SCENE(?:\s*:\s*([^\]\n]+?))?\s*\]", re.I)


# ── Progress bar context manager (factored out of process_chat) ───────────────

@asynccontextmanager
async def progress_bar(
    target_message: discord.Message,
    name: str,
    formatter: Callable[[str, str], Optional[str]],
    *,
    enabled: bool = True,
):
    """Reusable progress UX matching the legacy [IMAGE:] flow.

    Yields ``(on_progress, set_final)`` where:

    - ``on_progress(tag)`` is the awaitable progress callback (None when disabled).
    - ``set_final(text)`` lets the caller replace the live progress text with a
      final summary (e.g. *"refs: bot, Saki Nikaido, Tokyo Tower"*) which stays
      visible after the runner exits. When ``set_final`` is never called, the
      progress message is deleted on exit as before.
    """
    if not enabled:
        yield (None, lambda _t: None)
        return

    progress_msg: Optional[discord.Message] = None
    try:
        progress_msg = await target_message.reply(
            formatter("STAGE:loading", name) or "正在生成圖像…",
            mention_author=False,
        )
    except discord.HTTPException:
        progress_msg = None

    if progress_msg is None:
        yield (None, lambda _t: None)
        return

    queue: asyncio.Queue[str] = asyncio.Queue()
    final_holder: dict = {"text": None}
    # Track the last live progress text so the final footer can be APPENDED
    # under it rather than replacing the whole message — keeps the "what
    # stage did we end at" context visible alongside "which refs were used".
    last_rendered: dict = {"text": formatter("STAGE:loading", name) or "正在生成圖像…"}

    async def on_progress(tag: str) -> None:
        await queue.put(tag)

    def set_final(text: str) -> None:
        if text:
            final_holder["text"] = text

    async def poller() -> None:
        while True:
            tag = await queue.get()
            while not queue.empty():
                try:
                    tag = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            content = formatter(tag, name)
            if content:
                last_rendered["text"] = content
                try:
                    await progress_msg.edit(content=content)
                except discord.HTTPException:
                    pass
            await asyncio.sleep(0.5)

    poller_task = asyncio.create_task(poller())
    try:
        yield (on_progress, set_final)
    finally:
        poller_task.cancel()
        try:
            await poller_task
        except asyncio.CancelledError:
            pass
        final_text = final_holder["text"]
        if final_text:
            base = (last_rendered["text"] or "").rstrip()
            combined = f"{base}\n{final_text}" if base else final_text
            # Discord caps message edits at 2000 chars — be defensive even
            # though the formatter output is short and the footer is ~80.
            if len(combined) > 2000:
                combined = combined[: 2000 - len(final_text) - 1].rstrip() + "\n" + final_text
            try:
                await progress_msg.edit(content=combined)
            except discord.HTTPException:
                pass
        else:
            try:
                await progress_msg.delete()
            except discord.HTTPException:
                pass


# ── Reference-image assembly ──────────────────────────────────────────────────

# Stop-words filtered out of token-overlap matching so "the tower" matches
# `Tokyo Tower` (via "tower") but `the` alone never wins. EN articles +
# pronouns + auxiliaries; CJK has no equivalent set since CJK matching uses
# substring rather than tokens.
_TOKEN_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "is", "was", "are", "were", "be", "been", "being",
    "she", "he", "it", "they", "you", "i", "we", "us", "me",
    "him", "her", "his", "hers", "their", "theirs", "them",
    "my", "mine", "our", "ours", "your", "yours", "its",
    "this", "that", "these", "those",
    "as", "by", "do", "does", "did", "have", "has", "had",
    "from", "but", "not", "no", "so", "if", "then", "than", "too", "very",
    "out", "up", "down", "over", "into", "onto", "off", "about", "around",
})

# Word-character class used for ASCII boundary checks. CJK characters and
# punctuation count as boundaries so "saki" matches "Saki," / "Saki。" but
# not "ksaki" or "asaki".
_ASCII_WORD = re.compile(r"[a-z0-9]+")
_ASCII_ONLY = re.compile(r"^[\x00-\x7f]+$")

# `[SCENE: body | with: Saki, Tokyo Tower]` — captured by upstream marker
# parsing as just the body; we strip the trailing `| with: ...` here so the
# names can override the fuzzy matcher.
_WITH_CLAUSE_RE = re.compile(r"\|\s*with\s*:\s*(.+)$", re.I)


def _parse_with_clause(body: str) -> tuple[str, list[str]]:
    """Pull the optional `| with: a, b, c` tail out of a marker body.

    Returns ``(cleaned_body, [name, ...])``. When no clause is present the
    body is returned unchanged with an empty list. Names that are blank
    after stripping are dropped.
    """
    body = (body or "").strip()
    if not body:
        return body, []
    m = _WITH_CLAUSE_RE.search(body)
    if not m:
        return body, []
    raw = m.group(1)
    cleaned = body[: m.start()].rstrip(" ,;|\t")
    names = [n.strip() for n in raw.split(",") if n.strip()]
    return cleaned, names


def _name_tokens(name: str) -> set[str]:
    """Non-trivial ASCII tokens from `name` for token-overlap matching.

    Tokens shorter than 3 chars and tokens in the stop-word list are dropped
    so a multi-word title like *Tokyo Tower* contributes ``{"tokyo", "tower"}``
    while *the city* contributes only ``{"city"}``.
    """
    if not name:
        return set()
    return {
        t for t in _ASCII_WORD.findall(name.lower())
        if len(t) >= 3 and t not in _TOKEN_STOPWORDS
    }


def _seed_token_set(seed: str) -> set[str]:
    """All ASCII word tokens in the seed (no length / stop-word filtering —
    the filtering happens on the name side so common seed words like *the*
    only match when a name actually contains them)."""
    if not seed:
        return set()
    return set(_ASCII_WORD.findall(seed.lower()))


def _has_substring_match(name: str, seed_lower: str) -> bool:
    """Substring match with word-boundary safety for ASCII names.

    Pure-ASCII names require ASCII word boundaries on both sides so *she*
    matches "She smiles" but never "ashes". CJK or mixed-script names use
    plain substring matching since CJK has no inter-word whitespace.
    """
    name_l = name.strip().lower()
    if len(name_l) < 2 or not seed_lower:
        return False
    if _ASCII_ONLY.match(name_l):
        pattern = r"(?<![a-z0-9])" + re.escape(name_l) + r"(?![a-z0-9])"
        return bool(re.search(pattern, seed_lower))
    return name_l in seed_lower


def _name_matches_seed(name: str, seed_lower: str, seed_tokens: set[str]) -> bool:
    """True when `name` (a title or alias) overlaps the seed text.

    Combines the word-bounded substring rule with token-overlap so single
    multi-word titles still match short-form mentions ("the tower" → *Tokyo
    Tower* via the *tower* token).
    """
    name = (name or "").strip()
    if len(name) < 2 or not seed_lower:
        return False
    if _has_substring_match(name, seed_lower):
        return True
    name_tokens = _name_tokens(name)
    if name_tokens and (name_tokens & seed_tokens):
        return True
    return False


def _entry_aliases(entry: dict) -> list[str]:
    """Read the optional `aliases` field as a clean list of strings.

    Tolerates legacy entries that lack the field, mis-typed entries that
    stored a comma-separated string, and entries with the field set to a
    non-list value.
    """
    raw = entry.get("aliases")
    if not raw:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(p).strip() for p in raw if isinstance(p, str) and p.strip()]
    return []


def _entry_matches_seed(entry: dict, seed_lower: str, seed_tokens: set[str]) -> bool:
    """Title OR any alias overlaps the seed via the fuzzy rule."""
    title = (entry.get("title") or "").strip()
    if title and _name_matches_seed(title, seed_lower, seed_tokens):
        return True
    for alias in _entry_aliases(entry):
        if _name_matches_seed(alias, seed_lower, seed_tokens):
            return True
    return False


def _entry_matches_explicit(entry: dict, query: str) -> bool:
    """Strict match for `with: ...` names — case-insensitive equality against
    the title or any alias. No substring / token guessing here so a misspelt
    name silently drops rather than pulling in an unrelated entry."""
    q = (query or "").strip().lower()
    if not q:
        return False
    title = (entry.get("title") or "").strip().lower()
    if title and q == title:
        return True
    for alias in _entry_aliases(entry):
        if q == alias.strip().lower():
            return True
    return False


def _format_refs_footer(subjects: list, bot_name: str) -> str:
    """Render a one-line *"refs: bot, Saki, Tokyo Tower"* footer.

    The bot's own slot collapses to the literal word ``bot`` so the line is
    short; KB titles keep their original casing. Empty input returns the
    empty string. The whole line is truncated to ~80 chars (with an ellipsis)
    to avoid pushing the progress UI off-screen on mobile clients.
    """
    if not subjects:
        return ""
    bot_marker = (bot_name or "self").strip().lower()
    pretty: list[str] = []
    seen: set[str] = set()
    for s in subjects:
        label = (s or "").strip()
        if not label:
            continue
        if label.lower() == bot_marker:
            label = "bot"
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        pretty.append(label)
    if not pretty:
        return ""
    line = "refs: " + ", ".join(pretty)
    if len(line) > 80:
        line = line[:77].rstrip(", ") + "…"
    return line


def _gather_refs(
    seed_text: str,
    explicit_subjects: Optional[list[str]] = None,
) -> tuple[list, list, dict]:
    """Build ``(refs, subjects, appearances)`` from character + KB matches.

    Subject-aware slot balancing: ``TextEncodeQwenImageEditPlus`` only has
    ``image1..image4`` slots, so when the scene contains multiple unique
    subjects (the bot self + KB matches) we MUST give each subject at least
    one photo before doubling up on the bot — otherwise a 3-subject scene
    burns 2 of 4 slots on the bot and starves the others out of the
    encoder.

    Two-pass selection within a 4-slot cap:

    1. **Subject discovery** — collect candidate subjects in priority order:
       bot self (if it has photos), explicit ``with:`` subjects (strict
       case-insensitive equality vs KB title/aliases), then fuzzy KB
       matches against the seed via the word-bounded substring /
       non-trivial-token rule (``_name_matches_seed``).
    2. **Pass A — one photo per subject** (in priority order) until either
       the slot cap is reached or every subject has its first photo.
    3. **Pass B — fill remaining slots** with extra photos from the same
       subjects (up to 2 per subject), preserving priority order.

    The single-subject case (only the bot, or only one KB match) keeps
    today's "up to 2 photos" behaviour because Pass A places one photo and
    Pass B fills the second slot from the same subject. Each KB entry / the
    bot contributes at most 2 photos total; a subject that already has 2
    photos in ``refs`` is skipped in Pass B.
    """
    MAX_TOTAL = 4
    refs: list = []
    subjects: list = []
    appearances: dict = {}

    seed_lower = (seed_text or "").lower()
    seed_tokens = _seed_token_set(seed_text)

    char = database.get_character()
    bot_name = (char.get("name") or "").strip()
    looks = (char.get("looks") or "").strip()

    char_count = database.get_character_image_count()
    bot_label = bot_name or "self"

    try:
        kb_entries = database.get_image_entries()
    except Exception:
        kb_entries = []

    # ── Subject discovery ────────────────────────────────────────────────
    # Each candidate is (label, loader, max_photos). loader(idx_1based)
    # returns a (bytes, mime) tuple or None. max_photos is the hard cap on
    # how many photos this subject may contribute across both passes.
    candidates: list = []
    seen_keys: set = set()
    used_entry_ids: set = set()

    def _key(label: str) -> str:
        return (label or "self").strip().lower()

    if char_count > 0:
        candidates.append((
            bot_label,
            lambda i: database.get_character_image(i),
            min(char_count, 2),
            "self",
        ))
        seen_keys.add(_key(bot_label))
        if bot_name and looks:
            appearances[bot_name] = looks

    def _add_kb_entry(entry: dict) -> None:
        entry_id = entry.get("id")
        if entry_id is None or entry_id in used_entry_ids:
            return
        title = (entry.get("title") or "").strip()
        if not title or _key(title) in seen_keys:
            return
        n_imgs = database.get_entry_image_count(entry_id)
        if n_imgs <= 0:
            return
        used_entry_ids.add(entry_id)
        seen_keys.add(_key(title))
        candidates.append((
            title,
            lambda i, eid=entry_id: database.get_kb_image_full(eid, i),
            min(n_imgs, 2),
            "kb",
        ))
        desc = (entry.get("appearance_description") or "").strip()
        if desc and title not in appearances:
            appearances[title] = desc

    # 1a. Explicit `with: ...` overrides — strict equality against KB titles.
    if explicit_subjects:
        for name in explicit_subjects:
            for entry in kb_entries:
                if entry.get("id") in used_entry_ids:
                    continue
                if _entry_matches_explicit(entry, name):
                    _add_kb_entry(entry)
                    break

    # 1b. Fuzzy seed-driven KB matches.
    for entry in kb_entries:
        if entry.get("id") in used_entry_ids:
            continue
        if _entry_matches_seed(entry, seed_lower, seed_tokens):
            _add_kb_entry(entry)

    if not candidates:
        return refs, subjects, appearances

    # Track per-subject usage so Pass B doesn't exceed the per-subject cap.
    used_by_label: dict = {}

    def _take_from(label: str, loader, max_photos: int) -> bool:
        """Try to take one more photo from *label*. Returns True on success."""
        if len(refs) >= MAX_TOTAL:
            return False
        used = used_by_label.get(label, 0)
        if used >= max_photos:
            return False
        idx = used + 1
        full = loader(idx)
        if full is None:
            # Mark as exhausted so we don't keep retrying the same index.
            used_by_label[label] = max_photos
            return False
        refs.append(full)
        subjects.append(label)
        used_by_label[label] = used + 1
        return True

    # ── Pass A: one photo per unique subject (priority order) ─────────────
    for label, loader, max_photos, _src in candidates:
        if len(refs) >= MAX_TOTAL:
            break
        _take_from(label, loader, max_photos)

    # ── Pass B: fill remaining slots from same candidates, ≤2 each ────────
    if len(refs) < MAX_TOTAL:
        for label, loader, max_photos, _src in candidates:
            while len(refs) < MAX_TOTAL and _take_from(label, loader, max_photos):
                pass
            if len(refs) >= MAX_TOTAL:
                break

    return refs, subjects, appearances


# ── Prompt derivation ─────────────────────────────────────────────────────────

async def _derive_prompt(
    seed: str,
    looks: str,
    bot_name: str,
    llm_refs: Optional[list] = None,
    llm_ref_labels: Optional[list] = None,
    scene_only: bool = False,
    prose_context: Optional[str] = None,
) -> str:
    """Run prompt enhancement against the active backend's enhancer.

    When ``llm_refs`` is provided (list of ``(bytes, mime)`` tuples, one per
    unique subject), they are forwarded to ``enhance_image_prompt`` as
    ``reference_images`` so the LLM can read character appearance directly
    from the photos instead of inventing traits. ``llm_ref_labels`` is the
    matching list of subject names used to build the photo-order map.

    ``scene_only=True`` switches the enhancer into "describe scene only,
    NO appearance words" mode. We use it on the Qwen-edit pipeline whenever
    reference photos are headed to Qwen — Qwen reads the character identity
    from the photos and any appearance words in the text prompt would
    overpower the photo's visual identity. In that mode the
    ``character_context`` and ``llm_refs`` are intentionally NOT forwarded
    to the enhancer (the photos go to Qwen, not here).

    Falls back to the seed itself on any failure — scene-image generation is
    never blocked by enhancement issues.
    """
    char_context = ""
    if looks and not scene_only:
        char_context = (
            f"[Authoritative written appearance — {bot_name or 'character'}]\n"
            f"{looks}"
        )
    has_llm_refs = bool(llm_refs)
    if scene_only:
        print(
            "[SceneImage] derive_prompt: scene-only mode — appearance text and "
            "reference photos withheld from the enhancer (photos go to Qwen)."
        )
    elif has_llm_refs:
        print(f"[SceneImage] passing {len(llm_refs)} LLM ref image(s) to enhancer: {llm_ref_labels}")
    try:
        enriched = await groq_ai.enhance_image_prompt(
            seed,
            character_context=char_context,
            reference_images=llm_refs if (has_llm_refs and not scene_only) else None,
            reference_image_labels=llm_ref_labels if (has_llm_refs and not scene_only) else None,
            scene_only=scene_only,
            prose_context=prose_context,
        )
        if enriched and isinstance(enriched, str):
            return enriched.strip()
    except Exception as exc:
        print(
            f"[SceneImage] enhance_image_prompt failed: "
            f"{type(exc).__name__}: {exc} — falling back to seed"
        )
    return seed.strip()


# ── Public runner ─────────────────────────────────────────────────────────────

async def run_scene_image(
    *,
    bot_message: discord.Message,
    channel: discord.abc.Messageable,
    channel_id,
    trigger: str,
    hint_prompt: Optional[str] = None,
    seed_override: Optional[str] = None,
    acker: Optional[Callable[[str], Awaitable[None]]] = None,
    prose_context: Optional[str] = None,
) -> None:
    """Generate a scene image and edit it onto `bot_message` in place.

    Args:
        bot_message:    The bot's own message to attach the image onto. Must
                        already be sent (not a Webhook return). The image is
                        attached via `.edit(attachments=[...])`.
        channel:        For typing context.
        channel_id:     Stringified channel id (used for cooldown + dedup).
        trigger:        One of "button" | "scene_tag" | "intent_reroute"
                        | "image_tag_reroute" — diagnostic only.
        hint_prompt:    Optional seed text. For [IMAGE: ...] re-route this is
                        the marker body. For intent re-route it's the user
                        message text. For bare [SCENE] / button it's None
                        and we derive from `bot_message.content`. Combined
                        with the bot prose unless `seed_override` is set.
        seed_override:  When the model emitted `[SCENE: short cinematic
                        description]`, the body comes through here and is
                        used VERBATIM as the prompt seed — the bot prose is
                        skipped because the model already handed us the
                        polished prompt it had in mind. Mutually exclusive
                        with the prose-derive path; takes precedence over
                        `hint_prompt` and the bot-message text.
        acker:          Optional async ephemeral-acknowledger (button path).
    """
    key = (str(channel_id), bot_message.id)

    # In-flight dedup — every trigger consults the same set so a click
    # cannot race with a [SCENE] auto-trigger on the same target.
    if key in _inflight:
        if acker is not None:
            try:
                await acker("⏳ 場景圖片正在生成中，請稍候…")
            except Exception:
                pass
        return

    # Per-channel cooldown
    bucket = _cooldown.get_bucket(bot_message)
    if bucket is not None:
        retry_after = bucket.update_rate_limit()
        if retry_after:
            if acker is not None:
                try:
                    await acker(f"⏳ 此頻道生成過於頻繁，請 {retry_after:.0f} 秒後再試。")
                except Exception:
                    pass
            return

    if not image_dispatch.image_ready():
        if acker is not None:
            try:
                await acker("⚠️ 尚未設定圖像生成後端，無法生成場景圖片。")
            except Exception:
                pass
        return

    _inflight.add(key)
    try:
        char = database.get_character()
        bot_name = (char.get("name") or "").strip()
        looks = (char.get("looks") or "").strip()

        # 1. Prompt seed
        # When the model emitted `[SCENE: short cinematic description]`, the
        # body lands in `seed_override` and is used VERBATIM — the bot's
        # reply prose is intentionally skipped because the model already
        # handed us the polished image-prompt it had in mind, and mixing in
        # the surrounding prose would dilute that signal.
        # Otherwise: hint_prompt (if any) + bot_message.content, with a
        # generic fallback if both are empty.
        explicit_subjects: list[str] = []
        if seed_override and seed_override.strip():
            cleaned, explicit_subjects = _parse_with_clause(seed_override.strip())
            seed = cleaned or "cinematic scene"
        else:
            seed_parts = []
            if hint_prompt:
                seed_parts.append(hint_prompt.strip())
            # Always append bot's own message text so derived scene matches the prose
            body = (bot_message.content or "").strip()
            if body:
                seed_parts.append(body)
            seed = "  ".join(p for p in seed_parts if p) or "cinematic scene"

        # Inject the bot's own name into the seed when it's missing. The LLM
        # often emits `[SCENE: she on stage…]` (pronoun only), which makes
        # _gather_refs miss the bot's KB photos and the Qwen dispatcher fall
        # back to plain txt2img. Prepending `"{bot_name}: "` here guarantees:
        #   (a) _gather_refs' fuzzy seed matcher still picks up the bot,
        #   (b) the derived prompt names the rendered subject, and
        #   (c) the multi-edit / edit workflow triggers when photos exist.
        # Skipped cleanly when the bot has no name configured or no saved
        # photo (in which case there's nothing to anchor to anyway).
        if bot_name and database.get_character_image_count() > 0:
            if bot_name.lower() not in seed.lower():
                seed = f"{bot_name}: {seed}"
                print(f"[SceneImage] injected bot name into seed ({bot_name!r})")

        # 2. Engine-tier detection (needed before enhancement so Qwen can
        # pass reference photos to the prompt enhancer).
        backend = os.environ.get("IMAGE_BACKEND", "cloudflare").lower()
        engine = os.environ.get("COMFYUI_ENGINE", "qwen").lower()

        # For Qwen: gather refs NOW so the enhancer sees character photos.
        # Both the bot's own character photo and each matched KB entry photo
        # are forwarded — one photo per unique subject, labelled by name.
        # This lets the LLM read actual hair/eye colour instead of inventing
        # them. The full refs list (up to 2 photos per subject) is preserved
        # for ComfyUI; we only de-duplicate for the LLM call.
        _qwen_prefetch: Optional[tuple] = None
        llm_refs: list = []
        llm_ref_labels: list = []
        if backend == "comfyui" and engine == "qwen":
            _early_refs, _early_subjects, _early_appearances = _gather_refs(
                seed, explicit_subjects
            )
            _qwen_prefetch = (_early_refs, _early_subjects, _early_appearances)
            _seen_llm: set = set()
            for _r, _s in zip(_early_refs, _early_subjects):
                _s_key = (_s or "self").strip().lower()
                if _s_key not in _seen_llm:
                    _seen_llm.add(_s_key)
                    llm_refs.append(_r)
                    llm_ref_labels.append((_s or "self").strip())

        # On the Qwen-edit pipeline, reference photos go directly to Qwen via
        # `TextEncodeQwenImageEditPlus` — feeding any appearance words into
        # the text prompt would override the visual identity baked into the
        # photo. Switch the enhancer into scene-only mode whenever Qwen has
        # at least one reference photo to consume.
        _scene_only = bool(llm_refs) and backend == "comfyui" and engine == "qwen"

        # Inject cached location as a seed anchor so the scene-only enhancer
        # honours the established setting even on emotionally-charged turns
        # where it tends to skip the location extraction step.
        _str_channel_id = str(channel_id)
        _cached_loc = _scene_location_cache.get(_str_channel_id)
        if _cached_loc:
            _seed_lower = seed.lower()
            _loc_lower = _cached_loc.lower()
            # Only prepend if no location word from the cache is already present
            if _loc_lower not in _seed_lower:
                seed = f"[Location: {_cached_loc}] {seed}"
                print(f"[SceneImage] Injected cached location into seed: {_cached_loc!r}")

        enriched_base = await _derive_prompt(
            seed, looks, bot_name,
            llm_refs=llm_refs if llm_refs else None,
            llm_ref_labels=llm_ref_labels if llm_ref_labels else None,
            scene_only=_scene_only,
            prose_context=prose_context,
        )
        enriched_base = enriched_base.rstrip(",. \n")

        # Update the location cache from the freshly-enhanced prompt so future
        # turns carry the correct setting forward.
        _update_location_cache(_str_channel_id, enriched_base)

        # 3. Engine-tier ref assembly (Qwen reuses the prefetched result)
        gen_kwargs: dict = {}
        resolved_subjects: list[str] = []
        all_subjects: list[str] = []
        if backend == "comfyui" and engine == "qwen":
            refs, subjects, appearances = _qwen_prefetch  # type: ignore[misc]
            if refs:
                gen_kwargs["reference_images"] = refs
                gen_kwargs["reference_subjects"] = subjects
                gen_kwargs["subject_appearances"] = appearances
            resolved_subjects = subjects
            all_subjects = subjects
        elif backend == "comfyui":
            # FLUX — single-ref char only, no KB interleaving
            refs, subjects, _ = _gather_refs(seed, explicit_subjects)
            if refs:
                gen_kwargs["reference_image"] = refs[0]
                gen_kwargs["reference_images"] = refs[:1]
                gen_kwargs["reference_subjects"] = subjects[:1]
            resolved_subjects = subjects[:1]
            all_subjects = subjects[:1]
        elif backend in ("local_diffusers", "hf_spaces"):
            refs, subjects, _ = _gather_refs(seed, explicit_subjects)
            if refs:
                gen_kwargs["reference_image"] = refs[0]
            resolved_subjects = subjects[:1]
            all_subjects = subjects[:1]
        # cloudflare: text-only, no refs

        # Carry resolved subject names into the final prompt: append a short
        # "featuring …" tail naming any matched subjects that don't already
        # appear in the enriched body. Gated to the ComfyUI-Qwen path because
        # that's where multi-ref edit runs — FLUX/local_diffusers/hf_spaces
        # already get bot context through `_derive_prompt`'s `character_context`,
        # and Cloudflare is text-only with no refs to name. Cap to the first 3
        # subjects to avoid over-padding short prompts.
        enriched_lower = enriched_base.lower()
        featured: list[str] = []
        if backend == "comfyui" and engine == "qwen":
            for s in all_subjects[:3]:
                s_clean = (s or "").strip()
                if not s_clean or s_clean.lower() == "self":
                    continue
                if s_clean.lower() in enriched_lower:
                    continue
                if s_clean.lower() in (f.lower() for f in featured):
                    continue
                featured.append(s_clean)
        _cinematic_on = _cinematic_suffix_enabled()
        _suffix = CINEMATIC_SUFFIX if _cinematic_on else ""
        if featured:
            enriched = enriched_base + ", featuring " + ", ".join(featured) + _suffix
        else:
            enriched = enriched_base + _suffix

        # Append `looks` post-LLM as a non-LLM identity anchor (gated by
        # QWEN_APPEND_LOOKS, default on; engine==qwen + refs required).
        #
        # Scene-state awareness: when the scene implies the character is not in
        # their default outfit (shower, bath, nude, undressed, etc.) the full
        # looks text is NOT used as the anchor — only base identity traits
        # (hair, eyes, skin) are kept and an explicit SCENE STATE OVERRIDE
        # instruction is prepended so Qwen does not render the default
        # coat/holster/weapons.  Normal-outfit scenes are unaffected.
        _looks_appended = False
        _state_triggered = False
        _state_label = ""
        if (
            _append_looks_enabled()
            and backend == "comfyui"
            and engine == "qwen"
            and refs
        ):
            # Detect scene-state clothing override FIRST, independently of
            # whether looks text is available.  This ensures the SCENE STATE
            # OVERRIDE instruction is injected even when the character's looks
            # field is empty, preventing default-outfit rendering in those cases.
            _combined_ctx = seed + " " + enriched_base
            _state_triggered, _state_label = _detect_scene_clothing_override(_combined_ctx)

            try:
                _char = database.get_character() or {}
                _looks_text = (_char.get("looks") or "").strip()
            except Exception:
                _looks_text = ""

            if _state_triggered:
                # Always inject the state override prefix regardless of looks availability.
                _state_prefix = (
                    f"SCENE STATE OVERRIDE — character is {_state_label}: "
                    f"do NOT render coat, holster, weapons, or any default outfit items; "
                    f"scene context takes absolute priority over default clothing. "
                )
                if _looks_text:
                    # Strip outfit/weapon sentences — keep hair, eyes, skin only.
                    # fallback_to_empty=True prevents re-introducing outfit text
                    # in the anchor when all sentences happened to be outfit-related.
                    _identity_only = _strip_outfit_from_looks(_looks_text, fallback_to_empty=True)
                    if _identity_only:
                        enriched = (
                            _state_prefix
                            + enriched
                            + ". Character base identity only (outfit overridden by scene): "
                            + _identity_only
                        )
                    else:
                        # All looks sentences were outfit-related — state prefix only;
                        # Qwen reads base identity from reference photos directly.
                        enriched = _state_prefix + enriched
                else:
                    # No looks text — state prefix alone is sufficient.
                    enriched = _state_prefix + enriched
                _looks_appended = True
                print(
                    f"[SceneImage] Scene-state override active ({_state_label!r}): "
                    f"outfit stripped from looks anchor, state prefix prepended "
                    f"({'identity anchor included' if (_looks_text and _identity_only) else 'no identity anchor — photos only'})."
                )
            elif _looks_text:
                # Normal-outfit scene — append full looks as identity anchor.
                enriched = (
                    enriched
                    + ". Character identity (must be preserved from reference photos): "
                    + _looks_text
                )
                _looks_appended = True
                print(
                    f"[SceneImage] Appended {len(_looks_text)}-char looks identity suffix "
                    f"post-LLM (QWEN_APPEND_LOOKS=on)."
                )

        # Capture per-generate metadata for /scenedebug. Best-effort, never raises.
        try:
            _record_last_scene_generation(
                enriched_base=enriched_base,
                enriched_final=enriched,
                looks_appended=_looks_appended,
                cinematic_suffix=_cinematic_on,
                ref_labels=list(resolved_subjects or []),
                ref_count=len(refs or []),
                backend=backend,
                engine=engine,
                scene_only=bool(_scene_only),
                bot_name=bot_name,
                trigger=trigger,
                channel_id=str(getattr(channel, "id", "")),
                scene_state_override=_state_triggered,
                scene_state_label=_state_label,
            )
        except Exception as _dbg_exc:
            print(f"[SceneImage] debug capture failed: {type(_dbg_exc).__name__}: {_dbg_exc}")

        # 3. Run generation with the shared progress bar
        # `_format_diffuser_progress` lives in bot.py — lazy import to avoid
        # a circular module load at import time.
        try:
            from bot import _format_diffuser_progress as _fmt_progress
        except Exception:
            _fmt_progress = lambda tag, name="": None  # type: ignore

        progress_enabled = backend in ("local_diffusers", "comfyui")

        result = None
        async with progress_bar(
            bot_message,
            bot_name or "場景",
            _fmt_progress,
            enabled=progress_enabled,
        ) as (on_progress, set_final):
            if on_progress is not None:
                gen_kwargs["on_progress"] = on_progress
            try:
                async with channel.typing():
                    result = await image_dispatch.generate_image(
                        enriched, **gen_kwargs
                    )
            except Exception as exc:
                print(
                    f"[SceneImage] generate_image raised "
                    f"{type(exc).__name__}: {exc} (trigger={trigger})"
                )
                result = None

            # Log resolved reference subjects for ops visibility (the most
            # common silent failure mode is "the AI invented a face because
            # the KB photo didn't match the pronoun-only paragraph").
            # Not surfaced in Discord — we delete the progress message on
            # exit so the channel stays clean after generation.
            footer = _format_refs_footer(resolved_subjects, bot_name)
            if footer:
                print(f"[SceneImage] {footer}")

        # 4. Attach in place
        if (
            result
            and isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], bytes)
        ):
            img_bytes, mime = result
            ext = mime.split("/")[-1] if isinstance(mime, str) and "/" in mime else "png"
            try:
                file = discord.File(io.BytesIO(img_bytes), filename=f"scene.{ext}")
                await bot_message.edit(attachments=[file])
                if acker is not None:
                    try:
                        await acker("🎬 場景已生成 ✓")
                    except Exception:
                        pass
                return
            except discord.HTTPException as exc:
                # Strict edit-in-place contract: scene mode must NEVER post a
                # separate image message. If the in-place edit fails (rare —
                # message deleted, lost permissions, etc.) we report the
                # failure and stop. This is the central no-doubled-images
                # guarantee — never weaken it with a `channel.send` fallback.
                print(
                    f"[SceneImage] edit(attachments=) failed: {exc} — "
                    f"NOT falling back to a new message (strict in-place rule)."
                )
                if acker is not None:
                    try:
                        await acker(
                            "⚠️ 無法將場景圖片附加到原訊息 "
                            f"(`{type(exc).__name__}`)，已取消以避免重複貼圖。"
                        )
                    except Exception:
                        pass
                return

        # 5. Failure
        print(f"[SceneImage] no image produced for trigger={trigger}")
        if acker is not None:
            try:
                await acker("⚠️ 場景圖片生成失敗，請稍後再試。")
            except Exception:
                pass
    finally:
        _inflight.discard(key)


# ── Button click handler (called from views.SceneImageButtonView) ─────────────

async def handle_button_click(interaction: discord.Interaction) -> None:
    """Persistent-view button callback — defers, then dispatches the runner."""
    msg = interaction.message
    channel = interaction.channel
    if msg is None or channel is None:
        try:
            await interaction.response.send_message(
                "⚠️ 無法辨識此訊息。", ephemeral=True
            )
        except discord.HTTPException:
            pass
        return

    # Defer ephemerally so the user sees an immediate ack
    try:
        await interaction.response.defer(ephemeral=True, thinking=False)
    except discord.HTTPException:
        pass

    async def _ack(text: str) -> None:
        try:
            await interaction.followup.send(text, ephemeral=True)
        except discord.HTTPException:
            pass

    _prose_ctx: Optional[str] = None
    try:
        from bot import get_channel_context as _gcc, _build_prose_context as _bpc
        _prose_ctx = _bpc(_gcc(str(channel.id)))
    except Exception:
        pass

    await run_scene_image(
        bot_message=msg,
        channel=channel,
        channel_id=str(channel.id),
        trigger="button",
        hint_prompt=None,
        acker=_ack,
        prose_context=_prose_ctx,
    )
