"""Shared regex patterns for appearance-state suspension/restoration logic.

Imported by character_state.py, player_state.py, and scene_image.py.
All freed/escape detection lives here so the patterns stay in sync.
"""

import re

_CAPTIVE_TRANSITION_RE: re.Pattern = re.compile(
    r"\b(?:captur(?:ed?|ing)|imprison(?:ed|ing)?|dungeon|cell|cage|prison|"
    r"taken\s+(?:prisoner|captive)|disarm(?:ed|ing)?|kidnapp(?:ed|ing)?|"
    r"confiscat(?:ed?|ing)|enslav(?:ed|ing)?|locked\s+(?:up|in|away)|"
    r"stripped?\s+of\s+(?:her|his|their)\s+(?:weapon|gun|pistol|equipment|belonging))\b",
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

# Detects a "freed / escaped / re-armed" scene — used by scene_image.py to
# override weapon suppression when the character has just escaped, been
# released, or retrieved their belongings.  More conservative than
# _FREED_TRANSITION_RE: requires directional/contextual anchors to avoid
# false positives ("she felt free", "he released an arrow", etc.).
_FREED_SCENE_RE: re.Pattern = re.compile(
    r"\b(?:"
    # Escape — must be directional ("escaped from") or target a captivity noun
    r"escap(?:ed?|es|ing)\s+from\b|"
    r"escap(?:ed?|es)\s+(?:the\s+)?(?:dungeon|cell|cage|prison|captivity|custody|captors?)\b|"
    # Breaking free — specific enough already
    r"broke?\s+free\b|broken\s+free\b|broke?\s+out\s+of\b|"
    # Freed/released — require explicit captivity context to avoid false positives
    # ("she felt free", "he released an arrow", "the bird was freed" must NOT fire).
    r"(?:was|were|been|got|gets?|getting)\s+set\s+free\b|"
    r"(?:was|were|been|got|gets?|getting)\s+(?:finally\s+|just\s+|at\s+last\s+)?freed\s+from\b|"
    r"set\s+(?:her|him|them|you|me)\s+free\b|"
    r"finally\s+free\s+from\s+(?:captivity|prison|jail|custody|captors?|bonds?|chains|shackles|dungeon|cell|cage|confinement|imprisonment)\b|"
    r"(?:was|were|been)\s+(?:finally\s+|just\s+|at\s+last\s+)?released\s+from\b|"
    r"(?:was|were|been)\s+(?:finally\s+|just\s+|at\s+last\s+)?liberat(?:ed?)\s+from\b|"
    # Weapon / belonging explicitly returned or retrieved
    r"return(?:ed)?\s+(?:her|his|their|your)\s+(?:weapon|gun|pistol|rifle|"
    r"equipment|belonging|holster|knife|sword|blade|bag|gear)\b|"
    r"(?:her|his|their|your)\s+(?:weapon|gun|pistol|rifle|equipment|belonging|"
    r"holster|knife|sword|blade|bag|gear)\s+(?:was|were)\s+(?:returned|given\s+back|handed\s+back)\b|"
    r"retriev(?:ed?|ing)\s+(?:her|his|their|your)\s+(?:weapon|gun|pistol|rifle|"
    r"equipment|belonging|holster|knife|sword|blade|bag|gear)\b|"
    r"re-?arm(?:ed|ing)?\b|rearmed\b|"
    r"got\s+(?:her|his|their|your)\s+(?:weapon|gun|pistol|rifle|equipment|belonging|"
    r"holster|knife|sword|blade|bag|gear)\s+back\b|"
    r"reclaim(?:ed?|ing)\s+(?:her|his|their|your)\b"
    r")",
    re.I,
)
