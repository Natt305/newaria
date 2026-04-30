"""Shared regex patterns for appearance-state suspension/restoration logic.

Imported by character_state.py and player_state.py.
scene_image.py uses its own _FREED_SCENE_RE (image-context variant) but
references the same vocabulary for consistency.
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
