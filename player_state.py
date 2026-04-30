"""Per-channel, per-player appearance state cache.

Tracks six categories of appearance state for the human player, mirroring
the character_state system.  State is keyed by (channel_id, discord_id) so
each player's outfit persists independently across turns and bot restarts.

  outfit       — current clothing (e.g. "leather jacket and jeans")
  body_state   — clothed / undressed / towel / pajamas / robe / ...
  accessories  — persistent worn items (bracelet, necklace, glasses, …)
                 survive scene overrides (appear even in shower/nude scenes)
  restraints   — bindings (rope around wrists, handcuffs, chains…)
                 survive scene overrides
  wounds       — injuries (bandage on arm, black eye, cut on cheek, bruised…)
                 survive scene overrides; clear when RP implies healing
  marks        — lasting marks (scar, revealed tattoo…)
                 survive scene overrides; clear when RP implies recovery

State is backed by the SQLite database (history.db, player_state table).
On first access per (channel, player) per session the state is lazily loaded
from the DB.  Every meaningful appearance change is persisted immediately.
"""

from __future__ import annotations

import copy as _copy
import json as _json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import database as _db

# ---------------------------------------------------------------------------
# State dataclass (identical structure to CharacterState)
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    outfit: str = ""
    body_state: str = "clothed"
    accessories: list[str] = field(default_factory=list)
    restraints:  list[str] = field(default_factory=list)
    wounds:      list[str] = field(default_factory=list)
    marks:       list[str] = field(default_factory=list)
    # Items removed from accessories because of a captive/disarmed event.
    # Stored so they can be reliably restored when the player escapes or is freed,
    # even if the LLM no longer has the item in its context window.
    # NOT shown in images until the player is freed (then auto-promoted back to accessories).
    suspended_accessories: list[str] = field(default_factory=list)
    updated_at: int = 0

    def is_empty(self) -> bool:
        """suspended_accessories is metadata — it does not affect emptiness."""
        return (
            not self.outfit
            and self.body_state == "clothed"
            and not self.accessories
            and not self.restraints
            and not self.wounds
            and not self.marks
        )

    def persistent_items(self) -> list[str]:
        """Suspended accessories are NOT included — they are absent until restored."""
        items: list[str] = []
        items.extend(self.accessories)
        items.extend(self.restraints)
        items.extend(self.wounds)
        items.extend(self.marks)
        return items

    def is_undressed(self) -> bool:
        return self.body_state.lower() in (
            "undressed", "naked", "nude", "bare", "unclothed",
        )

    def format_debug(self) -> str:
        return (
            f"outfit={self.outfit!r} body={self.body_state!r} "
            f"acc={self.accessories} restraints={self.restraints} "
            f"wounds={self.wounds} marks={self.marks} "
            f"suspended={self.suspended_accessories} turn={self.updated_at}"
        )


# ---------------------------------------------------------------------------
# In-memory store  (keyed by "channel_id:discord_id")
# ---------------------------------------------------------------------------

_states: dict[str, PlayerState] = {}
_turn_counters: dict[str, int] = {}
_loaded_from_db: set[str] = set()
_history: dict[str, list[dict]] = {}
_HISTORY_MAX: int = 10


def _key(channel_id: str, discord_id: str) -> str:
    return f"{channel_id}:{discord_id}"


# ---------------------------------------------------------------------------
# DB serialisation helpers
# ---------------------------------------------------------------------------

def _state_to_dict(state: PlayerState) -> dict:
    return {
        "outfit":                state.outfit,
        "body_state":            state.body_state,
        "accessories":           state.accessories,
        "restraints":            state.restraints,
        "wounds":                state.wounds,
        "marks":                 state.marks,
        "suspended_accessories": state.suspended_accessories,
    }


def _state_from_dict(d: dict) -> PlayerState:
    return PlayerState(
        outfit=d.get("outfit", ""),
        body_state=d.get("body_state", "clothed"),
        accessories=list(d.get("accessories") or []),
        restraints=list(d.get("restraints") or []),
        wounds=list(d.get("wounds") or []),
        marks=list(d.get("marks") or []),
        suspended_accessories=list(d.get("suspended_accessories") or []),
    )


def _load_from_db(channel_id: str, discord_id: str) -> PlayerState | None:
    try:
        row = _db.get_player_state(channel_id, discord_id)
        if row is None:
            return None
        state = _state_from_dict(row)
        k = _key(channel_id, discord_id)
        print(f"[PlayerState] Loaded persisted state ch={channel_id} user={discord_id}: {state.format_debug()}")
        saved_history = _db.get_player_history(channel_id, discord_id)
        if saved_history:
            _history[k] = saved_history
        saved_counter = _db.get_player_turn_counter(channel_id, discord_id)
        if saved_counter:
            _turn_counters[k] = saved_counter
        return state
    except Exception as e:
        print(f"[PlayerState] Could not load state from DB ch={channel_id} user={discord_id}: {e}")
        return None


def get_state(channel_id: str, discord_id: str) -> PlayerState:
    """Return the current appearance state for a (channel, player) pair.

    Lazily loads from DB on first access per session.
    """
    k = _key(channel_id, discord_id)
    if k not in _states and k not in _loaded_from_db:
        _loaded_from_db.add(k)
        loaded = _load_from_db(channel_id, discord_id)
        if loaded is not None:
            _states[k] = loaded
    return _states.get(k, PlayerState())


def reset_state(channel_id: str | None = None, discord_id: str | None = None) -> None:
    """Clear appearance state.

    - Both None: clear all state for all channels and players.
    - Only channel_id: clear all players in that channel.
    - Both set: clear that specific player in that channel.
    """
    if channel_id is None and discord_id is None:
        _states.clear()
        _turn_counters.clear()
        _loaded_from_db.clear()
        _history.clear()
        _db.delete_player_state(None, None)
        print("[PlayerState] All player states cleared.")
    elif discord_id is None:
        # Clear all keys belonging to this channel
        keys_to_remove = [k for k in list(_states) if k.startswith(f"{channel_id}:")]
        for k in keys_to_remove:
            _states.pop(k, None)
            _turn_counters.pop(k, None)
            _loaded_from_db.discard(k)
            _history.pop(k, None)
        _db.delete_player_state(channel_id, None)
        print(f"[PlayerState] All player states cleared for channel {channel_id}.")
    else:
        k = _key(channel_id, discord_id)
        removed = _states.pop(k, None)
        _turn_counters.pop(k, None)
        _loaded_from_db.discard(k)
        _history.pop(k, None)
        _db.delete_player_state(channel_id, discord_id)
        if removed:
            print(f"[PlayerState] State cleared ch={channel_id} user={discord_id}.")


def _next_turn(channel_id: str, discord_id: str) -> int:
    k = _key(channel_id, discord_id)
    _turn_counters[k] = _turn_counters.get(k, 0) + 1
    return _turn_counters[k]


def _diff_states(before: PlayerState, after: PlayerState) -> list[str]:
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


def _record_history(channel_id: str, discord_id: str, turn: int, before: PlayerState, after: PlayerState) -> None:
    changes = _diff_states(before, after)
    if not changes:
        return
    k = _key(channel_id, discord_id)
    entry = {"turn": turn, "changes": changes}
    bucket = _history.setdefault(k, [])
    bucket.append(entry)
    if len(bucket) > _HISTORY_MAX:
        del bucket[: len(bucket) - _HISTORY_MAX]
    _db.set_player_history(channel_id, discord_id, bucket)


def get_history(channel_id: str, discord_id: str, n: int = 5) -> list[dict]:
    """Return the last *n* state-transition entries for a (channel, player) pair."""
    k = _key(channel_id, discord_id)
    return list(_history.get(k, [])[-n:])


# ---------------------------------------------------------------------------
# Pre-filter — same keywords as character_state
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
    combined = (user_text or "") + " " + (bot_reply or "")
    return bool(_NEEDS_UPDATE_RE.search(combined))


# ---------------------------------------------------------------------------
# Suspension / restoration regexes (mirrors character_state.py)
# ---------------------------------------------------------------------------

_CAPTIVE_TRANSITION_RE: re.Pattern = re.compile(
    r"\b(?:captur(?:ed?|ing)|imprison(?:ed|ing)?|dungeon|cell|cage|prison|"
    r"taken\s+(?:prisoner|captive)|disarm(?:ed|ing)?|kidnapp(?:ed|ing)?|"
    r"confiscat(?:ed?|ing)|enslav(?:ed|ing)?|locked\s+(?:up|in|away)|"
    r"stripped?\s+of\s+(?:his|her|their)\s+(?:weapon|gun|pistol|equipment|belonging))\b",
    re.I,
)

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
You are a player appearance tracker for a roleplay chatbot.
Given a conversation exchange, determine what changed about {player_name}'s appearance.

Track SIX categories (only report changes — leave out unchanged categories):
  outfit       — what they are wearing right now (full description, e.g. "leather jacket and jeans")
  body_state   — "clothed", "undressed", "naked", "towel", "pajamas", "robe", or short phrase
  accessories  — persistent worn items: bracelet, necklace, earrings, hat, glasses, watch, etc.
  restraints   — physical bindings: rope around wrists, handcuffs, chains, etc.
  wounds       — injuries: bandage on arm, black eye, cut on cheek, bruised face, etc.
  marks        — lasting marks: scar revealed for the first time, new tattoo shown, etc.

CRITICAL RULES:
1. Only track changes to {player_name}'s appearance — NOT the bot character's appearance.
2. Detect BOTH EXPLICIT and IMPLIED changes.
   - "the ropes were cut" → remove all rope restraints on {player_name}
   - "{player_name} was freed/rescued" while bound → remove restraints
   - "after a week of rest" / "{player_name}'s wounds healed" → remove wounds
   - "{player_name} grabbed a robe" → body_state becomes "robe"
   - "{player_name} stepped out of the shower in a towel" → body_state "towel"
3. RE-DRESSING: When {player_name} puts clothes back on after being nude/undressed,
   you MUST set body_state to "clothed" AND update outfit with what they wore.
   - "he slipped back into his coat" → body_state "clothed", outfit "coat"
   - "she got dressed" → body_state "clothed", outfit = what they put on
   IMPORTANT: Never leave body_state as "naked"/"undressed"/"nude" after a clear dressing action.
4. Accessories, restraints, wounds, and marks use ADDITIVE lists (added/removed).
   Outfit and body_state fully replace the previous value.
5. Only output changes. Use null for any category with no change.
6. Do NOT invent changes not supported by the text.
7. Keep descriptions concise (under 60 chars per item).
8. CAPTIVE / DISARMED TRANSITIONS — implied removal of carried items:
   When the text clearly implies {player_name} has been CAPTURED, IMPRISONED, taken to
   a DUNGEON or CELL, DISARMED, or had their belongings CONFISCATED/SEIZED, remove any
   carried/portable accessories from the accessories list.
   "Carried/portable" = weapons (gun, revolver, pistol, rifle, shotgun, knife, sword,
   blade, dagger, holster, axe, taser, baton), bags, pouches, tools, and tech-devices.
   DO NOT remove these items in a captive transition — they stay on the body:
     collar, choker, cuffs, shackles, chains, rope, leash, bandages, piercings,
     bracelets, anklets, tattoos, scars, marks.
   Examples:
   - "{player_name} was dragged into the dungeon" → remove any weapon accessories
   - "they took {player_name}'s weapons" → remove all weapon accessories
   - "{player_name} was disarmed and locked up" → remove gun/holster from accessories
   - "collared and led to a cell" → keep collar (worn); add restraint if newly applied
   Only apply this rule when captivity/disarming is CLEARLY described — do NOT remove
   items for a passing mention of a dungeon as background scenery.
9. FREED / ESCAPED / RE-ARMED — implied restoration of carried items:
   When the text clearly implies {player_name} has ESCAPED, been FREED, RELEASED, had
   their BELONGINGS RETURNED, or is explicitly RE-ARMED, re-add previously carried
   accessories that were logically removed during capture or disarming.
   "Previously carried" = weapons (gun, revolver, pistol, rifle, shotgun, knife, sword,
   blade, dagger, holster, axe, taser, baton), bags, pouches, tools, and tech-devices.
   Only re-add an item if the current accessories list does NOT already contain it and
   the prior state or context implies it was carried before capture.
   Examples:
   - "{player_name} escaped from the dungeon" / "{player_name} broke free" and had a weapon before → re-add it
   - "they returned {player_name}'s belongings" / "{player_name}'s gun was handed back" → re-add the weapon
   - "{player_name} retrieved their holster and pistol" → re-add gun and holster
   - "{player_name} was released and rearmed" → re-add previously held weapons
   - "{player_name} slipped away and recovered their weapon" → re-add weapon
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


def _state_summary(state: PlayerState) -> str:
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
            f"use accessories_added to restore ONLY if they have clearly escaped or been rearmed)"
        )
    return ", ".join(parts)


async def _extract_state_delta(
    user_text: str,
    bot_reply: str,
    current_state: PlayerState,
    player_name: str,
    chat_fn: ChatFn,
) -> dict:
    exchange = (
        f"Player ({player_name}): {user_text[:600]}\n"
        f"Bot: {bot_reply[:800]}"
    )
    system = _EXTRACT_SYSTEM_TEMPLATE.format(
        player_name=player_name,
        current_state=_state_summary(current_state),
    )
    messages = [
        {"role": "user", "content": f"Analyze this exchange for appearance changes to {player_name}:\n\n{exchange}"}
    ]

    try:
        text = await chat_fn(messages, system)
        if not text:
            return {}

        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            print(f"[PlayerState] No JSON object found in extractor response: {text[:200]!r}")
            return {}

        parsed = _json.loads(text[start:end])
        if not isinstance(parsed, dict):
            print(f"[PlayerState] Extractor returned non-dict JSON: {text[start:end][:200]!r}")
            return {}

        return parsed

    except _json.JSONDecodeError as e:
        print(f"[PlayerState] JSON parse error in extractor: {e} — raw: {text[start:end][:200]!r}")
        return {}
    except Exception as e:
        print(f"[PlayerState] Extractor call failed: {type(e).__name__}: {e}")
        return {}


def _clean_str_list(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(s).strip() for s in raw if isinstance(s, str) and str(s).strip()]


def _apply_delta(state: PlayerState, delta: dict, turn: int) -> PlayerState:
    outfit = delta.get("outfit")
    if outfit and isinstance(outfit, str) and outfit.strip():
        state.outfit = outfit.strip()

    body_state = delta.get("body_state")
    if body_state and isinstance(body_state, str) and body_state.strip():
        state.body_state = body_state.strip().lower()
    elif outfit and isinstance(outfit, str) and outfit.strip() and state.body_state != "clothed":
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

        existing_lower = {x.lower() for x in current}
        for item in added:
            if item.lower() not in existing_lower:
                current.append(item)
                existing_lower.add(item.lower())

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
    discord_id: str,
    user_text: str,
    bot_reply: str,
    player_name: str,
    chat_fn: ChatFn,
) -> PlayerState:
    """Update player appearance state after a roleplay exchange.

    1. Runs the regex pre-filter — returns early if no appearance keywords.
    2. Calls the LLM extractor to get a delta.
    3. Merges the delta into the stored state.
    Returns the (possibly unchanged) state.
    """
    # exchange_text is needed for deterministic suspension/restoration logic which
    # must run independently of the extractor (even if extractor returns nothing).
    exchange_text = (user_text or "") + " " + (bot_reply or "")

    if not _needs_update(user_text, bot_reply):
        return get_state(channel_id, discord_id)

    current = get_state(channel_id, discord_id)
    turn = _next_turn(channel_id, discord_id)
    k = _key(channel_id, discord_id)
    print(f"[PlayerState] ch={channel_id} user={discord_id} turn={turn} — pre-filter triggered, calling extractor…")

    delta = await _extract_state_delta(user_text, bot_reply, current, player_name, chat_fn)
    if not delta:
        # Extractor returned nothing — still apply deterministic freed restoration
        # so escape turns reliably reinstate suspended accessories.
        if current.suspended_accessories and _FREED_TRANSITION_RE.search(exchange_text):
            before_snap = _copy.copy(current)
            before_snap.suspended_accessories = list(current.suspended_accessories)
            existing_acc_lower = {x.lower() for x in current.accessories}
            promoted = []
            for item in current.suspended_accessories:
                if item.lower() not in existing_acc_lower:
                    current.accessories.append(item)
                    existing_acc_lower.add(item.lower())
                    promoted.append(item)
            current.suspended_accessories = []
            if promoted:
                print(f"[PlayerState] Restored from suspension on escape (no extractor delta) for {discord_id}: {promoted}")
            _states[k] = current
            _db.set_player_state(channel_id, discord_id, _state_to_dict(current))
            _db.set_player_turn_counter(channel_id, discord_id, turn)
            _record_history(channel_id, discord_id, turn, before_snap, current)
        return current

    meaningful = {
        kk: v for kk, v in delta.items()
        if v not in (None, [], "")
    }

    if not meaningful:
        # Extractor returned all-null delta — still apply deterministic freed restoration
        if current.suspended_accessories and _FREED_TRANSITION_RE.search(exchange_text):
            before_snap = _copy.copy(current)
            before_snap.suspended_accessories = list(current.suspended_accessories)
            existing_acc_lower = {x.lower() for x in current.accessories}
            promoted = []
            for item in current.suspended_accessories:
                if item.lower() not in existing_acc_lower:
                    current.accessories.append(item)
                    existing_acc_lower.add(item.lower())
                    promoted.append(item)
            current.suspended_accessories = []
            if promoted:
                print(f"[PlayerState] Restored from suspension on escape (no visual delta) for {discord_id}: {promoted}")
            _states[k] = current
            _db.set_player_state(channel_id, discord_id, _state_to_dict(current))
            _db.set_player_turn_counter(channel_id, discord_id, turn)
            _record_history(channel_id, discord_id, turn, before_snap, current)
        return current

    print(f"[PlayerState] Applying delta for {discord_id}: {meaningful}")
    before_snapshot = _copy.copy(current)
    before_snapshot.accessories           = list(current.accessories)
    before_snapshot.restraints            = list(current.restraints)
    before_snapshot.wounds                = list(current.wounds)
    before_snapshot.marks                 = list(current.marks)
    before_snapshot.suspended_accessories = list(current.suspended_accessories)
    updated = _apply_delta(current, delta, turn)

    # ── Suspension / restoration logic (post-delta) ────────────────────────
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
            print(f"[PlayerState] Suspended on capture for {discord_id}: {portable_removed}")

    # Freed / escaped transition → promote any remaining suspended items
    # (belt-and-suspenders; also catches items extractor may have added to suspended)
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
            print(f"[PlayerState] Restored from suspension on escape for {discord_id}: {promoted}")
        else:
            print(f"[PlayerState] Cleared suspended_accessories on escape for {discord_id}.")
    # ── End suspension / restoration ──────────────────────────────────────

    _states[k] = updated
    _db.set_player_state(channel_id, discord_id, _state_to_dict(updated))
    _db.set_player_turn_counter(channel_id, discord_id, turn)
    _record_history(channel_id, discord_id, turn, before_snapshot, updated)
    return updated
