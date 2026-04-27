"""Per-channel character appearance state cache.

Tracks six categories of appearance state that persist across roleplay turns
and are injected into every subsequent scene image:

  outfit       — current clothing (e.g. "red dress", "jeans and t-shirt")
  body_state   — clothed / undressed / towel / pajamas / robe / ...
  accessories  — persistent worn items (collar, bracelet, earrings, watch…)
                 survive scene overrides (appear even in shower/nude scenes)
  restraints   — bindings (rope around wrists, handcuffs, chains…)
                 survive scene overrides
  wounds       — injuries (bandage on arm, black eye, cut on cheek, bruised…)
                 survive scene overrides; clear when RP implies healing
  marks        — lasting marks (scar, revealed tattoo…)
                 survive scene overrides; clear when RP implies recovery

State is in-memory only (same lifecycle as conversation_contexts in bot.py).
The LLM extractor detects both EXPLICIT and IMPLIED changes: "the user cut the
ropes" removes restraints even without a literal "she is no longer tied" line.
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class CharacterState:
    outfit: str = ""
    body_state: str = "clothed"   # "clothed" | "undressed" | "naked" | "towel" | ...
    accessories: list[str] = field(default_factory=list)
    restraints:  list[str] = field(default_factory=list)
    wounds:      list[str] = field(default_factory=list)
    marks:       list[str] = field(default_factory=list)
    updated_at: int = 0           # monotone turn counter for diagnostics

    def is_empty(self) -> bool:
        """Return True when no runtime state has been accumulated."""
        return (
            not self.outfit
            and self.body_state == "clothed"
            and not self.accessories
            and not self.restraints
            and not self.wounds
            and not self.marks
        )

    def persistent_items(self) -> list[str]:
        """All items that must always appear in images regardless of scene override.

        Accessories, restraints, wounds, and marks persist even through
        nude/shower scenes — a collar stays on while showering, bound wrists
        are still visible in an intimate scene, bandages don't vanish.
        """
        items: list[str] = []
        items.extend(self.accessories)
        items.extend(self.restraints)
        items.extend(self.wounds)
        items.extend(self.marks)
        return items

    def is_undressed(self) -> bool:
        """Return True when body_state indicates the character is not clothed."""
        return self.body_state.lower() in (
            "undressed", "naked", "nude", "bare", "unclothed",
        )

    def format_debug(self) -> str:
        """One-line summary for /scenedebug output."""
        return (
            f"outfit={self.outfit!r} body={self.body_state!r} "
            f"acc={self.accessories} restraints={self.restraints} "
            f"wounds={self.wounds} marks={self.marks} turn={self.updated_at}"
        )


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_states: dict[str, CharacterState] = {}
_turn_counters: dict[str, int] = {}


def get_state(channel_id: str) -> CharacterState:
    """Return the current appearance state for a channel (blank default if none)."""
    return _states.get(channel_id, CharacterState())


def reset_state(channel_id: str | None = None) -> None:
    """Clear appearance state for one channel, or ALL channels when None."""
    if channel_id is None:
        _states.clear()
        _turn_counters.clear()
        print("[CharState] All channel states cleared.")
    else:
        removed = _states.pop(channel_id, None)
        _turn_counters.pop(channel_id, None)
        if removed:
            print(f"[CharState] State cleared for channel {channel_id}.")


def _next_turn(channel_id: str) -> int:
    _turn_counters[channel_id] = _turn_counters.get(channel_id, 0) + 1
    return _turn_counters[channel_id]


# ---------------------------------------------------------------------------
# Pre-filter — prevents LLM calls on turns with no appearance signals
# ---------------------------------------------------------------------------

_NEEDS_UPDATE_RE = re.compile(
    r"""
    # Clothing / outfit change signals
    \b(?:wear|wearing|wears|wore|worn|puts?\s+on|takes?\s+off|took\s+off
    |dress(?:ed|es|ing)?|undress(?:ed|es|ing)?|strips?\s+(?:off|down)?|stripped
    |chang(?:es?|ed|ing)\s+(?:into|out\s+of|clothes?|outfits?)
    |outfit|clothes|clothing|shirt|blouse|dress|skirt|pants|trousers|suit
    |uniform|lingerie|pyjamas?|pajamas?|robe|nightgown|nightwear
    |towel|bra|underwear|swimsuit|bikini|naked|nude)\b

    # Accessories
    |\b(?:collar|bracelet|earring|necklace|anklet|choker|ring|watch
    |glasses|sunglasses|hat|cap|headband|tiara)\b

    # Restraints and liberation
    |\b(?:rope|tied|bind|bound|restrained|handcuff|cuff|shackle|chain(?:ed)?
    |strapped|gagged|blindfolded
    |untied|unbound|freed|free(?:d|s)?|rescue(?:d|s)?|escaped?
    |cut\s+(?:the\s+)?rope|cut\s+(?:her\s+)?free|pulled?\s+free
    |slipped?\s+(?:out|free|loose)|loosen(?:ed)?|released?|liberated?)\b

    # Injuries, wounds, and recovery
    |\b(?:bandage|bruise|bruised|wound(?:ed)?|injur(?:y|ied|es)
    |hurt|battered|bloody|bleed(?:ing)?|scar(?:red)?
    |cut|gash|lacerat|black\s+eye|swollen|stitche?s?
    |heal(?:ed|ing|s)?|recover(?:ed|ing|y)?|rested?|treated?
    |cleaned?\s+up|patched?\s+up|tended?\s+to)\b
    """,
    re.I | re.VERBOSE,
)


def _needs_update(user_text: str, bot_reply: str) -> bool:
    """Return True if the exchange might contain appearance changes."""
    combined = (user_text or "") + " " + (bot_reply or "")
    return bool(_NEEDS_UPDATE_RE.search(combined))


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

ChatFn = Callable[[list, str], Awaitable[str]]

_EXTRACT_SYSTEM_TEMPLATE = """\
You are a character appearance tracker for a roleplay chatbot.
Given a conversation exchange, determine what changed about {bot_name}'s appearance.

Track SIX categories (only report changes — leave out unchanged categories):
  outfit       — what they are wearing right now (full description, e.g. "red dress and heels")
  body_state   — "clothed", "undressed", "naked", "towel", "pajamas", "robe", or short phrase
  accessories  — persistent worn items: collar, bracelet, earrings, hat, glasses, watch, etc.
  restraints   — physical bindings: rope around wrists, handcuffs, chains, etc.
  wounds       — injuries: bandage on arm, black eye, cut on cheek, bruised face, etc.
  marks        — lasting marks: scar revealed for the first time, new tattoo shown, etc.

CRITICAL RULES:
1. Detect BOTH EXPLICIT and IMPLIED changes.
   - "the user cut the ropes" → remove all rope restraints (even if not stated directly)
   - "{bot_name} was rescued/saved/freed" while bound → remove restraints
   - "after a week of rest" / "the wounds healed" / "she was treated" → remove wounds
   - "she grabbed a robe" → body_state becomes "robe"
   - "she stepped out of the shower in a towel" → body_state becomes "towel"
2. RE-DRESSING: When the character puts clothes back on after a nude/undressed/shower
   scene, you MUST set body_state to "clothed" AND update outfit with what they wore.
   Examples:
   - "She slipped back into her uniform" → body_state "clothed", outfit "uniform"
   - "She got dressed" / "she put her clothes back on" → body_state "clothed",
     outfit = whatever they put on (or the outfit they had before if not specified)
   - "she pulled on her jeans and t-shirt" → body_state "clothed", outfit "jeans and t-shirt"
   - "she wrapped herself in a towel and then got dressed" → body_state "clothed"
   IMPORTANT: Never leave body_state as "naked"/"undressed"/"nude" after a clear
   dressing action. Always update it to "clothed" (or the appropriate cover state).
3. Accessories, restraints, wounds, and marks use ADDITIVE lists (added/removed).
   Outfit and body_state fully replace the previous value.
4. Only output changes. Use null for any category with no change.
5. Do NOT invent changes not supported by the text.
6. Keep descriptions concise (under 60 chars per item).

Current state (for reference — do not repeat unless it changed):
{current_state}

Return ONLY a valid JSON object with these exact keys (null = no change):
{{
  "outfit": null,
  "body_state": null,
  "accessories_added": null,
  "accessories_removed": null,
  "restraints_added": null,
  "restraints_removed": null,
  "wounds_added": null,
  "wounds_removed": null,
  "marks_added": null,
  "marks_removed": null
}}"""


def _state_summary(state: CharacterState) -> str:
    parts = [
        f"outfit={state.outfit!r}",
        f"body_state={state.body_state!r}",
        f"accessories={state.accessories}",
        f"restraints={state.restraints}",
        f"wounds={state.wounds}",
        f"marks={state.marks}",
    ]
    return ", ".join(parts)


async def _extract_state_delta(
    user_text: str,
    bot_reply: str,
    current_state: CharacterState,
    bot_name: str,
    chat_fn: ChatFn,
) -> dict:
    """Call the LLM to extract an appearance delta for this exchange.

    Returns a (possibly empty) dict on success; empty dict on failure.
    """
    exchange = (
        f"Player: {user_text[:600]}\n"
        f"{bot_name}: {bot_reply[:800]}"
    )
    system = _EXTRACT_SYSTEM_TEMPLATE.format(
        bot_name=bot_name,
        current_state=_state_summary(current_state),
    )
    messages = [
        {"role": "user", "content": f"Analyze this exchange for appearance changes:\n\n{exchange}"}
    ]

    try:
        text = await chat_fn(messages, system)
        if not text:
            return {}

        text = text.strip()
        # Extract JSON object from response (model may include prose around it)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            print(f"[CharState] No JSON object found in extractor response: {text[:200]!r}")
            return {}

        parsed = _json.loads(text[start:end])
        if not isinstance(parsed, dict):
            print(f"[CharState] Extractor returned non-dict JSON: {text[start:end][:200]!r}")
            return {}

        return parsed

    except _json.JSONDecodeError as e:
        print(f"[CharState] JSON parse error in extractor: {e} — raw: {text[start:end][:200]!r}")
        return {}
    except Exception as e:
        print(f"[CharState] Extractor call failed: {type(e).__name__}: {e}")
        return {}


def _clean_str_list(raw) -> list[str]:
    """Coerce a raw JSON value to a clean list of non-empty strings."""
    if not isinstance(raw, list):
        return []
    return [str(s).strip() for s in raw if isinstance(s, str) and str(s).strip()]


def _apply_delta(state: CharacterState, delta: dict, turn: int) -> CharacterState:
    """Merge a delta dict into the state in-place and return the updated state."""
    outfit = delta.get("outfit")
    if outfit and isinstance(outfit, str) and outfit.strip():
        state.outfit = outfit.strip()

    body_state = delta.get("body_state")
    if body_state and isinstance(body_state, str) and body_state.strip():
        state.body_state = body_state.strip().lower()
    elif outfit and isinstance(outfit, str) and outfit.strip() and state.body_state != "clothed":
        # Fallback: a new outfit was set but the LLM forgot to update body_state.
        # Covers naked/undressed AND transitional-cover states (towel, robe, etc.)
        # so the SCENE STATE OVERRIDE is cleared whenever the character gets dressed.
        state.body_state = "clothed"

    for added_key, removed_key, attr in [
        ("accessories_added",  "accessories_removed",  "accessories"),
        ("restraints_added",   "restraints_removed",   "restraints"),
        ("wounds_added",       "wounds_removed",       "wounds"),
        ("marks_added",        "marks_removed",        "marks"),
    ]:
        added   = _clean_str_list(delta.get(added_key))
        removed = _clean_str_list(delta.get(removed_key))
        current: list[str] = getattr(state, attr)

        # Add new items (deduplicate by value)
        existing_lower = {x.lower() for x in current}
        for item in added:
            if item.lower() not in existing_lower:
                current.append(item)
                existing_lower.add(item.lower())

        # Remove items (case-insensitive match)
        if removed:
            lower_removed = {r.lower() for r in removed}
            setattr(state, attr, [
                x for x in current if x.lower() not in lower_removed
            ])

    state.updated_at = turn
    return state


# ---------------------------------------------------------------------------
# Public update entry point
# ---------------------------------------------------------------------------

async def update_state(
    channel_id: str,
    user_text: str,
    bot_reply: str,
    bot_name: str,
    chat_fn: ChatFn,
) -> CharacterState:
    """Update appearance state for a channel after a roleplay exchange.

    1. Runs the regex pre-filter — returns early if no appearance keywords.
    2. Calls the LLM extractor to get a delta.
    3. Merges the delta into the stored state.
    Returns the (possibly unchanged) state.
    """
    if not _needs_update(user_text, bot_reply):
        return get_state(channel_id)

    current = _states.get(channel_id, CharacterState())
    turn = _next_turn(channel_id)
    print(f"[CharState] ch={channel_id} turn={turn} — pre-filter triggered, calling extractor…")

    delta = await _extract_state_delta(user_text, bot_reply, current, bot_name, chat_fn)
    if not delta:
        return current

    # Log non-null / non-empty changes for ops visibility
    meaningful = {
        k: v for k, v in delta.items()
        if v not in (None, [], "")
    }
    if not meaningful:
        return current

    print(f"[CharState] Applying delta: {meaningful}")
    updated = _apply_delta(current, delta, turn)
    _states[channel_id] = updated
    return updated
