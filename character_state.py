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

State is backed by the SQLite database (history.db, character_state table).
On first access per channel per session the state is lazily loaded from the DB,
so bot restarts mid-roleplay do not lose outfit/accessory/wound/restraint history.
Every meaningful appearance change is persisted immediately after being applied.
The LLM extractor detects both EXPLICIT and IMPLIED changes: "the user cut the
ropes" removes restraints even without a literal "she is no longer tied" line.
"""

from __future__ import annotations

import copy as _copy
import json as _json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import database as _db

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
    # Items removed from accessories because of a captive/disarmed event.
    # Stored so they can be reliably restored when the character escapes or is freed,
    # even if the LLM no longer has the weapon in its context window.
    # NOT shown in images until the character is freed (then auto-promoted back to accessories).
    suspended_accessories: list[str] = field(default_factory=list)
    updated_at: int = 0           # monotone turn counter for diagnostics

    def is_empty(self) -> bool:
        """Return True when no runtime visible state has been accumulated.

        suspended_accessories is metadata — it does not affect emptiness.
        """
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
        Suspended accessories are NOT included — they are absent until restored.
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
            f"wounds={self.wounds} marks={self.marks} "
            f"suspended={self.suspended_accessories} turn={self.updated_at}"
        )


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_states: dict[str, CharacterState] = {}
_turn_counters: dict[str, int] = {}
_loaded_from_db: set[str] = set()   # channels whose DB row has been fetched this session

# Appearance transition history: channel_id -> list of {"turn": int, "changes": [str]}
_history: dict[str, list[dict]] = {}
_HISTORY_MAX: int = 10  # max entries kept per channel


# ---------------------------------------------------------------------------
# DB serialisation helpers
# ---------------------------------------------------------------------------

def _state_to_dict(state: CharacterState) -> dict:
    return {
        "outfit":                state.outfit,
        "body_state":            state.body_state,
        "accessories":           state.accessories,
        "restraints":            state.restraints,
        "wounds":                state.wounds,
        "marks":                 state.marks,
        "suspended_accessories": state.suspended_accessories,
    }


def _state_from_dict(d: dict) -> CharacterState:
    return CharacterState(
        outfit=d.get("outfit", ""),
        body_state=d.get("body_state", "clothed"),
        accessories=list(d.get("accessories") or []),
        restraints=list(d.get("restraints") or []),
        wounds=list(d.get("wounds") or []),
        marks=list(d.get("marks") or []),
        suspended_accessories=list(d.get("suspended_accessories") or []),
    )


def _load_from_db(channel_id: str) -> CharacterState | None:
    """Attempt to load state, history, and turn counter from the database. Returns None if no row exists."""
    try:
        row = _db.get_character_state(channel_id)
        if row is None:
            return None
        state = _state_from_dict(row)
        print(f"[CharState] Loaded persisted state for channel {channel_id}: {state.format_debug()}")
        saved_history = _db.get_character_history(channel_id)
        if saved_history:
            _history[channel_id] = saved_history
            print(f"[CharState] Restored {len(saved_history)} history entries for channel {channel_id}")
        saved_counter = _db.get_character_turn_counter(channel_id)
        if saved_counter:
            _turn_counters[channel_id] = saved_counter
            print(f"[CharState] Restored turn counter {saved_counter} for channel {channel_id}")
        return state
    except Exception as e:
        print(f"[CharState] Could not load state from DB for channel {channel_id}: {e}")
        return None


def get_state(channel_id: str) -> CharacterState:
    """Return the current appearance state for a channel.

    On first access per session, tries to restore the state from the database
    so that bot restarts mid-RP don't reset outfit/accessory/wound history.
    """
    if channel_id not in _states and channel_id not in _loaded_from_db:
        _loaded_from_db.add(channel_id)
        loaded = _load_from_db(channel_id)
        if loaded is not None:
            _states[channel_id] = loaded
    return _states.get(channel_id, CharacterState())


def reset_state(channel_id: str | None = None) -> None:
    """Clear appearance state for one channel, or ALL channels when None."""
    if channel_id is None:
        _states.clear()
        _turn_counters.clear()
        _loaded_from_db.clear()
        _history.clear()
        _db.delete_character_state(None)
        print("[CharState] All channel states cleared.")
    else:
        removed = _states.pop(channel_id, None)
        _turn_counters.pop(channel_id, None)
        _loaded_from_db.discard(channel_id)
        _history.pop(channel_id, None)
        _db.delete_character_state(channel_id)
        if removed:
            print(f"[CharState] State cleared for channel {channel_id}.")


def _next_turn(channel_id: str) -> int:
    _turn_counters[channel_id] = _turn_counters.get(channel_id, 0) + 1
    return _turn_counters[channel_id]


def _diff_states(before: CharacterState, after: CharacterState) -> list[str]:
    """Return a list of human-readable descriptions of what changed between two states."""
    changes: list[str] = []
    if before.outfit != after.outfit:
        changes.append(f"outfit: {before.outfit!r} → {after.outfit!r}")
    if before.body_state != after.body_state:
        changes.append(f"body_state: {before.body_state!r} → {after.body_state!r}")
    for attr in ("accessories", "restraints", "wounds", "marks", "suspended_accessories"):
        before_set = set(getattr(before, attr))
        after_set = set(getattr(after, attr))
        added = after_set - before_set
        removed = before_set - after_set
        if added:
            changes.append(f"{attr} +{sorted(added)}")
        if removed:
            changes.append(f"{attr} -{sorted(removed)}")
    return changes


def _record_history(channel_id: str, turn: int, before: CharacterState, after: CharacterState) -> None:
    """Append a history entry if anything actually changed, then persist to DB."""
    changes = _diff_states(before, after)
    if not changes:
        return
    entry = {"turn": turn, "changes": changes}
    bucket = _history.setdefault(channel_id, [])
    bucket.append(entry)
    if len(bucket) > _HISTORY_MAX:
        del bucket[: len(bucket) - _HISTORY_MAX]
    _db.set_character_history(channel_id, bucket)


def get_history(channel_id: str, n: int = 5) -> list[dict]:
    """Return the last *n* state-transition entries for a channel (oldest first).

    Each entry is a dict with keys:
      ``turn``    – turn counter when the change was applied
      ``changes`` – list of human-readable change strings
    """
    return list(_history.get(channel_id, [])[-n:])


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

    # Captive / disarmed transitions (may imply removal of carried items)
    |\b(?:captur(?:ed?|ing)|imprison(?:ed|ing)?|dungeon|taken\s+(?:prisoner|captive)
    |disarm(?:ed|ing)?|kidnapp(?:ed|ing)?|confiscat(?:ed?|ing)
    |enslav(?:ed|ing)?|locked\s+(?:up|in|away)|seized?\s+(?:her|his|their)
    |stripped?\s+of\s+(?:her|his|their)\s+(?:weapon|gun|pistol|equipment|belonging))\b
    """,
    re.I | re.VERBOSE,
)


def _needs_update(user_text: str, bot_reply: str) -> bool:
    """Return True if the exchange might contain appearance changes."""
    combined = (user_text or "") + " " + (bot_reply or "")
    return bool(_NEEDS_UPDATE_RE.search(combined))


# ---------------------------------------------------------------------------
# Suspension / restoration regexes
# ---------------------------------------------------------------------------

# Detects captive/disarm transitions — signals that portable accessories
# removed this turn should be suspended rather than permanently deleted.
_CAPTIVE_TRANSITION_RE: re.Pattern = re.compile(
    r"\b(?:captur(?:ed?|ing)|imprison(?:ed|ing)?|dungeon|cell|cage|prison|"
    r"taken\s+(?:prisoner|captive)|disarm(?:ed|ing)?|kidnapp(?:ed|ing)?|"
    r"confiscat(?:ed?|ing)|enslav(?:ed|ing)?|locked\s+(?:up|in|away)|"
    r"stripped?\s+of\s+(?:her|his|their)\s+(?:weapon|gun|pistol|equipment|belonging))\b",
    re.I,
)

# Detects freed/escape transitions — signals that suspended accessories
# should be promoted back to active accessories.
# Mirrors _FREED_SCENE_RE in scene_image.py (kept in sync manually).
_FREED_TRANSITION_RE: re.Pattern = re.compile(
    r"\b(?:"
    r"escap(?:ed?|es|ing)\s+from\b|"
    r"escap(?:ed?|es)\s+(?:the\s+)?(?:dungeon|cell|cage|prison|captivity|custody|captors?)\b|"
    r"broke?\s+free\b|broken\s+free\b|broke?\s+out\s+of\b|"
    r"(?:was|were|been|got|gets?|getting)\s+set\s+free\b|"
    r"(?:was|were|been|got|gets?|getting)\s+freed\b|"
    r"set\s+(?:her|him|them|you|me)\s+free\b|"
    r"finally\s+free\b|"
    r"(?:was|were|been)\s+released\s+from\b|"
    r"(?:was|were|been)\s+liberat(?:ed?|ing)\b|"
    r"return(?:ed)?\s+(?:her|his|their|your)\s+"
    r"(?:weapon|gun|pistol|rifle|equipment|belonging|holster|knife|sword|blade|bag|gear)\b|"
    r"(?:her|his|their|your)\s+(?:weapon|gun|pistol|rifle|equipment|belonging|"
    r"holster|knife|sword|blade|bag|gear)\s+(?:was|were)\s+(?:returned|given\s+back|handed\s+back)\b|"
    r"retriev(?:ed?|ing)\s+(?:her|his|their|your)\s+"
    r"(?:weapon|gun|pistol|rifle|equipment|belonging|holster|knife|sword|blade|bag|gear)\b|"
    r"re-?arm(?:ed|ing)?\b|rearmed\b|"
    r"got\s+(?:her|his|their|your)\s+(?:weapon|gun|pistol|rifle|equipment|belonging|"
    r"holster|knife|sword|blade|bag|gear)\s+back\b|"
    r"reclaim(?:ed?|ing)\s+(?:her|his|their|your)\b"
    r")",
    re.I,
)

# Matches portable/carried items (weapons, bags, tools) that are suspended on
# capture and restored on escape — NOT worn/attached items like collars or cuffs.
_PORTABLE_ITEM_RE: re.Pattern = re.compile(
    r"\b(?:holster|gun|pistol|revolver|rifle|shotgun|firearm|weapon|"
    r"sword|blade|knife|dagger|saber|sabre|axe|bow|crossbow|baton|taser|"
    r"bag|pouch|satchel|backpack|kit|device|gadget|tool)\b",
    re.I,
)


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
7. CAPTIVE / DISARMED TRANSITIONS — implied removal of carried items:
   When the text clearly implies the character has been CAPTURED, IMPRISONED, taken to
   a DUNGEON or CELL, DISARMED, or had their belongings CONFISCATED/SEIZED, remove any
   carried/portable accessories from the accessories list.
   "Carried/portable" = weapons (gun, revolver, pistol, rifle, shotgun, knife, sword,
   blade, dagger, holster, axe, taser, baton), bags, pouches, tools, and tech-devices.
   DO NOT remove these items in a captive transition — they stay on the body:
     collar, choker, cuffs, shackles, chains, rope, leash, bandages, piercings,
     bracelets, anklets, tattoos, scars, marks.
   Examples:
   - "dragged into the dungeon" → remove revolver, holster from accessories_removed
   - "they took her weapons" / "she was disarmed" → remove all weapon accessories
   - "stripped of her equipment" → remove weapon/tool accessories
   - "collared and led to a cell" → keep collar (worn); add restraint if newly applied
   - "she surrendered her gun and was locked up" → remove gun from accessories_removed
   Only apply this rule when captivity/disarming is CLEARLY described — do NOT remove
   items for a passing mention of a dungeon as background scenery.
8. FREED / ESCAPED / RE-ARMED — implied restoration of carried items:
   When the text clearly implies {bot_name} has ESCAPED, been FREED, RELEASED, had
   their BELONGINGS RETURNED, or is explicitly RE-ARMED, re-add previously carried
   accessories that were logically removed during capture or disarming.
   "Previously carried" = weapons (gun, revolver, pistol, rifle, shotgun, knife, sword,
   blade, dagger, holster, axe, taser, baton), bags, pouches, tools, and tech-devices.
   Only re-add an item if the current accessories list does NOT already contain it and
   the prior state or context implies it was carried before capture.
   Examples:
   - "she escaped from the dungeon" / "she broke free" and had a revolver before → re-add revolver
   - "they returned her belongings" / "her gun was handed back" → re-add the weapon
   - "she retrieved her holster and pistol" → re-add gun and holster
   - "she was released and rearmed" → re-add previously held weapons
   - "she slipped away undetected and recovered her weapon" → re-add weapon
   Only apply this rule when freedom/rearming is CLEARLY described — do NOT add items
   for a passing mention of escape as background or hypothetical.

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
    if state.suspended_accessories:
        parts.append(
            f"suspended_accessories={state.suspended_accessories} "
            f"(confiscated during capture — NOT on body now; "
            f"use accessories_added to restore ONLY if she has clearly escaped or been rearmed)"
        )
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

    current = get_state(channel_id)
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
    before_snapshot = _copy.copy(current)
    before_snapshot.accessories           = list(current.accessories)
    before_snapshot.restraints            = list(current.restraints)
    before_snapshot.wounds                = list(current.wounds)
    before_snapshot.marks                 = list(current.marks)
    before_snapshot.suspended_accessories = list(current.suspended_accessories)
    updated = _apply_delta(current, delta, turn)

    # ── Suspension / restoration logic ────────────────────────────────────
    # These are deterministic post-delta adjustments keyed on the exchange text
    # so they don't require any LLM prompt changes.
    exchange_text = (user_text or "") + " " + (bot_reply or "")

    # Captive transition → suspend portable items that were removed this turn
    removed_accessories = set(before_snapshot.accessories) - set(updated.accessories)
    if removed_accessories and _CAPTIVE_TRANSITION_RE.search(exchange_text):
        portable_removed = [
            item for item in removed_accessories
            if _PORTABLE_ITEM_RE.search(item)
        ]
        if portable_removed:
            existing_lower = {x.lower() for x in updated.suspended_accessories}
            for item in portable_removed:
                if item.lower() not in existing_lower:
                    updated.suspended_accessories.append(item)
                    existing_lower.add(item.lower())
            print(f"[CharState] Suspended on capture: {portable_removed}")

    # Freed / escaped transition → promote suspended items back to accessories
    if updated.suspended_accessories and _FREED_TRANSITION_RE.search(exchange_text):
        existing_acc_lower = {x.lower() for x in updated.accessories}
        promoted: list[str] = []
        for item in updated.suspended_accessories:
            if item.lower() not in existing_acc_lower:
                updated.accessories.append(item)
                existing_acc_lower.add(item.lower())
                promoted.append(item)
        updated.suspended_accessories = []
        if promoted:
            print(f"[CharState] Restored from suspension on escape: {promoted}")
        else:
            print(f"[CharState] Cleared suspended_accessories on escape (already in accessories).")
    # ── End suspension / restoration ──────────────────────────────────────

    _states[channel_id] = updated
    _db.set_character_state(channel_id, _state_to_dict(updated))
    _db.set_character_turn_counter(channel_id, turn)
    _record_history(channel_id, turn, before_snapshot, updated)
    return updated
