"""Shared regex patterns for appearance-state suspension/restoration logic.

Imported by character_state.py, player_state.py, and scene_image.py.
All freed/escape detection lives here so the patterns stay in sync.

Shared vocabulary constants
----------------------------
_FREED_PRONOUNS            — pronoun alternation used in weapon-retrieval phrases
_FREED_WEAPON_NOUNS        — weapon/gear noun alternation shared by both freed patterns
_FREED_ESCAPE_CORE         — escape/break-free sub-patterns identical in both patterns
_FREED_WEAPON_RETRIEVAL    — weapon-retrieval sub-patterns identical in both patterns

Both _FREED_TRANSITION_RE and _FREED_SCENE_RE are built from these constants so that
adding a new noun or escape verb to one automatically keeps the other aligned.
"""

import re

_CAPTIVE_TRANSITION_RE: re.Pattern = re.compile(
    r"\b(?:captur(?:ed?|ing)|imprison(?:ed|ing)?|dungeon|cell|cage|prison|"
    r"taken\s+(?:prisoner|captive)|disarm(?:ed|ing)?|kidnapp(?:ed|ing)?|"
    r"confiscat(?:ed?|ing)|enslav(?:ed|ing)?|locked\s+(?:up|in|away)|"
    r"stripped?\s+of\s+(?:her|his|their)\s+(?:weapon|gun|pistol|equipment|belonging))\b",
    re.I,
)

# ── Shared vocabulary fragments ───────────────────────────────────────────────

# Pronoun alternation reused in retrieval phrases.
_FREED_PRONOUNS: str = r"her|his|their|your"

# Weapon / gear nouns shared by every retrieval and retrieval-check branch.
# Add new nouns here — both freed patterns pick them up automatically.
_FREED_WEAPON_NOUNS: str = (
    r"weapon|gun|pistol|rifle|equipment|belonging|"
    r"holster|knife|sword|blade|bag|gear"
)

# Escape and break-free sub-patterns that are identical in both compiled regexes.
# These are conservative enough to be safe for both transition and scene use.
_FREED_ESCAPE_CORE: str = (
    r"escap(?:ed?|es|ing)\s+from\b|"
    r"escap(?:ed?|es)\s+(?:the\s+)?(?:dungeon|cell|cage|prison|captivity|custody|captors?)\b|"
    r"broke?\s+free\b|broken\s+free\b|broke?\s+out\s+of\b|"
    r"(?:was|were|been|got|gets?|getting)\s+set\s+free\b|"
    r"set\s+(?:her|him|them|you|me)\s+free\b"
)

# Weapon-retrieval sub-patterns shared verbatim by both freed patterns.
# Built from the shared noun/pronoun constants so a single edit propagates.
_FREED_WEAPON_RETRIEVAL: str = (
    r"return(?:ed)?\s+(?:{p})\s+(?:{n})\b|"
    r"(?:{p})\s+(?:{n})\s+(?:was|were)\s+(?:returned|given\s+back|handed\s+back)\b|"
    r"retriev(?:ed?|ing)\s+(?:{p})\s+(?:{n})\b|"
    r"re-?arm(?:ed|ing)?\b|rearmed\b|"
    r"got\s+(?:{p})\s+(?:{n})\s+back\b|"
    r"reclaim(?:ed?|ing)\s+(?:{p})\b"
).format(p=_FREED_PRONOUNS, n=_FREED_WEAPON_NOUNS)

# ── Compiled patterns ─────────────────────────────────────────────────────────

# More permissive: used for state-machine transitions.
# Includes loose forms ("finally free", "was freed", "was liberated") that are
# safe for determining whether a state change occurred.
_FREED_TRANSITION_RE: re.Pattern = re.compile(
    r"\b(?:"
    + _FREED_ESCAPE_CORE + r"|"
    r"finally\s+free\b|"
    r"(?:was|were|been|got|gets?|getting)\s+freed\b|"
    r"(?:was|were|been)\s+released\s+from\b|"
    r"(?:was|were|been)\s+liberat(?:ed?|ing)\b|"
    + _FREED_WEAPON_RETRIEVAL +
    r")",
    re.I,
)

# More conservative: used by scene_image.py to override weapon suppression.
# Requires directional/contextual anchors to avoid false positives
# ("she felt free", "he released an arrow", "the bird was freed" must NOT fire).
_FREED_SCENE_RE: re.Pattern = re.compile(
    r"\b(?:"
    + _FREED_ESCAPE_CORE + r"|"
    r"(?:was|were|been|got|gets?|getting)\s+(?:finally\s+|just\s+|at\s+last\s+)?freed\s+from\b|"
    r"finally\s+free\s+from\s+(?:captivity|prison|jail|custody|captors?|bonds?|chains|shackles|dungeon|cell|cage|confinement|imprisonment)\b|"
    r"(?:was|were|been)\s+(?:finally\s+|just\s+|at\s+last\s+)?released\s+from\b|"
    r"(?:was|were|been)\s+(?:finally\s+|just\s+|at\s+last\s+)?liberat(?:ed?)\s+from\b|"
    + _FREED_WEAPON_RETRIEVAL +
    r")",
    re.I,
)

_PORTABLE_ITEM_RE: re.Pattern = re.compile(
    r"\b(?:holster|gun|pistol|revolver|rifle|shotgun|firearm|weapon|"
    r"sword|blade|knife|dagger|saber|sabre|axe|bow|crossbow|baton|taser|"
    r"bag|pouch|satchel|backpack|kit|device|gadget|tool)\b",
    re.I,
)
