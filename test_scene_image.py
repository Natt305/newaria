"""
Unit tests for the weapon-filtering logic in scene_image.py.

Specifically verifies that _WEAPON_PERSISTENT_RE:
  - matches actual weapon / holster items so they are suppressed during
    scene-state overrides (shower, nude, intimate scenes)
  - does NOT match non-weapon accessories (collar, cuffs, strap, chain,
    bandages, rope) so those survive scene-state overrides

Also exercises the filtering helper used in the _state_triggered block
(lines 1452-1478 of scene_image.py) end-to-end on a mixed persistent-item
list.

Run: python test_scene_image.py
"""

import sys
sys.path.insert(0, ".")

from scene_image import _WEAPON_PERSISTENT_RE

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f"\n         {detail}" if detail else ""))
    _results.append((label, condition, detail))


# ── _WEAPON_PERSISTENT_RE — items that MUST match (weapons) ───────────────────

WEAPON_ITEMS = [
    "holster",
    "leather holster",
    "gun",
    "pistol",
    "revolver",
    "rifle",
    "shotgun",
    "firearm",
    "weapon",
    "sword",
    "blade",
    "knife",
    "dagger",
    "saber",
    "sabre",
    "axe",
    "bow",
    "crossbow",
    "baton",
    "taser",
    "carries a pistol in a hip holster",
    "tactical holster with pistol",
    "sheathed sword at her side",
]

# ── _WEAPON_PERSISTENT_RE — items that must NOT match (non-weapon accessories) ─

SAFE_ITEMS = [
    "leather collar",
    "collar",
    "ankle cuffs",
    "wrist cuffs",
    "cuffs",
    "strap",
    "ankle strap",
    "leather strap",
    "chain",
    "ankle chain",
    "thin gold chain",
    "bandages",
    "bandage on her arm",
    "rope",
    "silk rope bracelet",
    "choker",
    "leash",
    "harness",
]


def test_weapon_items_matched():
    """Every actual weapon / holster item must be caught by the regex."""
    for item in WEAPON_ITEMS:
        matched = bool(_WEAPON_PERSISTENT_RE.search(item))
        check(
            f"WEAPON matched: {item!r}",
            matched,
            f"expected a match but got none",
        )


def test_safe_items_not_matched():
    """Non-weapon accessories must NOT be caught by the regex."""
    for item in SAFE_ITEMS:
        matched = bool(_WEAPON_PERSISTENT_RE.search(item))
        check(
            f"SAFE not matched: {item!r}",
            not matched,
            f"unexpected match — item would be incorrectly suppressed in scene override",
        )


def test_case_insensitive():
    """Regex must be case-insensitive for all weapon keywords."""
    cases = ["GUN", "Holster", "SWORD", "Pistol", "KNIFE"]
    for word in cases:
        matched = bool(_WEAPON_PERSISTENT_RE.search(word))
        check(f"case-insensitive match: {word!r}", matched, "expected match, got none")


def test_filtering_mixed_persistent_list():
    """Simulate the filtering that the _state_triggered block performs.

    Given a mixed list of persistent items (some weapons, some safe accessories),
    the filter must:
      - remove all weapon items
      - keep all non-weapon accessories intact
    """
    persistent_items = [
        "leather collar",
        "carries a holster with pistol",
        "ankle cuffs",
        "sheathed knife at her belt",
        "thin silver chain around wrist",
        "tactical baton clipped to belt",
        "silk rope bracelet",
        "bandages wrapped around forearm",
    ]

    expected_kept = {
        "leather collar",
        "ankle cuffs",
        "thin silver chain around wrist",
        "silk rope bracelet",
        "bandages wrapped around forearm",
    }
    expected_removed = {
        "carries a holster with pistol",
        "sheathed knife at her belt",
        "tactical baton clipped to belt",
    }

    safe = [item for item in persistent_items if not _WEAPON_PERSISTENT_RE.search(item)]
    removed = [item for item in persistent_items if _WEAPON_PERSISTENT_RE.search(item)]

    for item in expected_kept:
        check(
            f"kept after filter: {item!r}",
            item in safe,
            f"item was unexpectedly removed",
        )

    for item in expected_removed:
        check(
            f"removed by filter: {item!r}",
            item in removed,
            f"item was not removed — weapon would appear in override scene",
        )

    check(
        "no unexpected items kept",
        set(safe) == expected_kept,
        f"safe list mismatch: {set(safe)} != {expected_kept}",
    )

    check(
        "no unexpected items removed",
        set(removed) == expected_removed,
        f"removed list mismatch: {set(removed)} != {expected_removed}",
    )


def test_word_boundary_respected():
    """The \b word-boundary anchor must prevent over-matching inside compound words.

    Words like 'holstered', 'gunshot', 'sabers', 'knifepoint' embed a weapon
    root but are NOT weapon items themselves.  The regex must leave them alone
    so that descriptors containing those substrings are not incorrectly
    suppressed during scene-state overrides.
    """
    compound_words = [
        "sabers",
        "holstered",
        "gunshot",
        "knifepoint",
    ]
    for word in compound_words:
        matched = bool(_WEAPON_PERSISTENT_RE.search(word))
        check(
            f"compound word not matched: {word!r}",
            not matched,
            "word-boundary anchor should prevent this match",
        )


# ── Run all tests ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Weapon items matched ──────────────────────────────────────────────")
    test_weapon_items_matched()

    print("\n── Safe accessories not matched ──────────────────────────────────────")
    test_safe_items_not_matched()

    print("\n── Case insensitivity ────────────────────────────────────────────────")
    test_case_insensitive()

    print("\n── Mixed persistent-list filter ──────────────────────────────────────")
    test_filtering_mixed_persistent_list()

    print("\n── Word-boundary behaviour ───────────────────────────────────────────")
    test_word_boundary_respected()

    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        sys.exit(1)
    else:
        print("  — all good!")
