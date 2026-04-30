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

from scene_image import (
    _WEAPON_PERSISTENT_RE, _CLOTHING_ACCESSORY_PERSISTENT_RE,
    _FREED_SCENE_RE, _detect_captive_scene,
    _OUTFIT_SENTENCE_RE, _strip_outfit_from_looks,
)

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


# ── _FREED_SCENE_RE — patterns that must NOT fire (false-positive guards) ──────

FALSE_POSITIVE_PHRASES = [
    # Generic use of "released" — not about captivity
    "she released an arrow",
    "he released a long breath",
    "they released a new album",
    # Generic use of "free" / "freed" — not captivity-escape
    "he felt free",
    "she finally felt free",
    "the bird was freed",
    "he was finally free from his worries",
    "she was finally free from all obligations",
    # "liberated" without explicit captivity source
    "they were liberated",
    "she was liberating herself emotionally",
]

# ── _FREED_SCENE_RE — patterns that MUST fire (genuine freedom signals) ────────

TRUE_POSITIVE_PHRASES = [
    # Released FROM a place/state of captivity
    "she was released from prison",
    "he was released from custody",
    "she was released from captivity",
    # Freed FROM captivity
    "she was freed from the dungeon",
    "he was freed from captivity",
    "they got freed from their chains",
    # Finally free FROM captivity noun
    "she was finally free from captivity",
    "he was finally free from prison",
    "she was finally free from confinement",
    # Liberated FROM a place
    "they were liberated from the dungeon",
    "she was liberated from captivity",
    # Explicit escape
    "she escaped from the cell",
    "he escaped from captivity",
    "she broke free",
    "he broke out of the cage",
    # Belongings returned
    "returned her weapon",
    "retrieved his gun",
    "got their gear back",
]


def test_freed_scene_no_false_positives():
    """Generic uses of freedom/release words must NOT trigger the freed-scene override."""
    for phrase in FALSE_POSITIVE_PHRASES:
        matched = bool(_FREED_SCENE_RE.search(phrase))
        check(
            f"FREED false-positive guard: {phrase!r}",
            not matched,
            "phrase incorrectly matched — would disable captive-scene weapon filtering",
        )


def test_freed_scene_true_positives():
    """Genuine freedom-from-captivity phrases must always trigger the freed-scene override."""
    for phrase in TRUE_POSITIVE_PHRASES:
        matched = bool(_FREED_SCENE_RE.search(phrase))
        check(
            f"FREED true positive: {phrase!r}",
            matched,
            "phrase did not match — captive character would incorrectly lose their weapons",
        )


def test_captive_scene_fires_despite_incidental_freedom_words():
    """A captive scene that uses freedom words in passing must still suppress weapons.

    e.g. "She released an arrow while chained to the dungeon wall" is a captive scene
    even though it contains "released".
    """
    captive_with_incidental_freedom = [
        "She released an arrow while chained to the dungeon wall.",
        "He felt free in his mind while locked in a cell.",
        "They released their breath as guards locked the cell door.",
        "She was finally free from fear yet still sat in her prison cell.",
    ]
    for text in captive_with_incidental_freedom:
        is_captive, label = _detect_captive_scene(text)
        check(
            f"captive fires despite freedom word: {text[:60]!r}",
            is_captive,
            f"captive detector did not fire — weapons would not be suppressed (label={label!r})",
        )


def test_captive_scene_yields_on_genuine_freedom():
    """A captive scene where the character genuinely escapes must NOT suppress weapons."""
    genuine_freedom_texts = [
        "She escaped from the dungeon, sword back in hand.",
        "He was finally released from prison after years of captivity.",
        "They broke free from their chains and retrieved their weapons.",
        "She was freed from captivity and rearmed.",
    ]
    for text in genuine_freedom_texts:
        is_captive, label = _detect_captive_scene(text)
        check(
            f"captive yields on genuine freedom: {text[:60]!r}",
            not is_captive,
            f"captive detector fired — weapons would be incorrectly suppressed (label={label!r})",
        )


def test_headwear_matched_by_outfit_sentence_re():
    """Headwear and outerwear terms must be caught by _OUTFIT_SENTENCE_RE."""
    headwear_sentences = [
        "A black hat completes her look.",
        "Her signature cowboy hat sits atop her curly hair.",
        "She sports a red cap on her head.",
        "A wide-brimmed beret rests on her head.",
        "Her vintage fedora gives her a mysterious air.",
        "A white bonnet frames her face.",
        "She wears a motorcycle helmet.",
        "A tactical visor shades her eyes.",
        # New outerwear terms (Task #17)
        "A long trenchcoat falls to her knees.",
        "Her windbreaker keeps off the chill.",
        "A heavy anorak covers her shoulders.",
        "A hooded cloak drapes her figure.",
        "A velvet cape flows behind her.",
        "A silk cardigan rests over her shoulders.",
        "A fitted vest sits over her shirt.",
        "A silk wrap covers her arms.",
        "Her knitwear is soft and form-fitting.",
    ]
    for sent in headwear_sentences:
        check(
            f"headwear caught: {sent[:60]!r}",
            bool(_OUTFIT_SENTENCE_RE.search(sent)),
            "_OUTFIT_SENTENCE_RE did not match — headwear would leak into erotic prompt",
        )


def test_headwear_stripped_from_looks_by_strip_outfit():
    """_strip_outfit_from_looks removes hat sentences during scene-state override."""
    hat_only_looks = (
        "She has long auburn hair and green eyes. "
        "A black cowboy hat sits atop her head. "
        "Her skin is lightly tanned."
    )
    result = _strip_outfit_from_looks(hat_only_looks, fallback_to_empty=True)
    check(
        "hat sentence removed from looks",
        "hat" not in result.lower(),
        f"hat still present in stripped result: {result!r}",
    )
    check(
        "hair/eyes retained after hat strip",
        "auburn hair" in result and "green eyes" in result,
        f"identity traits lost: {result!r}",
    )


def test_hat_with_collar_collar_rescued():
    """A sentence mentioning hat AND collar: hat sentence dropped, collar phrase rescued."""
    mixed = "She wears a black hat and a leather collar around her neck."
    result = _strip_outfit_from_looks(mixed, fallback_to_empty=True)
    check(
        "hat+collar sentence: hat keyword absent from result",
        "hat" not in result.lower(),
        f"hat still present: {result!r}",
    )
    check(
        "hat+collar sentence: collar phrase rescued",
        "collar" in result.lower(),
        f"collar not rescued: {result!r}",
    )


def test_non_headwear_outfit_words_still_caught():
    """Existing outfit terms (shirt, boots, etc.) are still caught after the addition."""
    existing_terms = [
        ("shirt", "She wears a white shirt."),
        ("boots", "She has tall leather boots."),
        ("gloves", "Black gloves cover her hands."),
        ("scarf", "A red scarf wraps her neck."),
    ]
    for term, sent in existing_terms:
        check(
            f"existing term still caught: {term!r}",
            bool(_OUTFIT_SENTENCE_RE.search(sent)),
            f"regression: {term!r} no longer matched",
        )


# ── _CLOTHING_ACCESSORY_PERSISTENT_RE — clothing accessory filter tests ────────

CLOTHING_ACCESSORY_ITEMS = [
    "hat",
    "black hat",
    "cowboy hat",
    "baseball cap",
    "cap",
    "beret",
    "red beret",
    "fedora",
    "vintage fedora",
    "bonnet",
    "visor",
    "tactical visor",
    "helmet",
    "motorcycle helmet",
    "beanie",
    "stetson",
    "trilby",
    "bowler",
    "coat",
    "long coat",
    "overcoat",
    "jacket",
    "leather jacket",
    "parka",
    "blazer",
    "gloves",
    "glove",
    "black gloves",
    "mittens",
    "mitten",
    "scarf",
    "red scarf",
    "shawl",
    "trenchcoat",
    "long trenchcoat",
    "windbreaker",
    "anorak",
    "hoodie",
    "black hoodie",
    "cardigan",
    "vest",
    "leather vest",
    "cloak",
    "hooded cloak",
    "cape",
    "velvet cape",
    "wrap",
    "silk wrap",
    "knitwear",
]

CLOTHING_ACCESSORY_SAFE_ITEMS = [
    "leather collar",
    "collar",
    "ankle cuffs",
    "wrist cuffs",
    "cuffs",
    "leash",
    "chain",
    "ankle chain",
    "thin gold chain",
    "rope",
    "silk rope bracelet",
    "harness",
    "choker",
    "bandages",
    "strap",
]


def test_clothing_accessories_matched():
    """Every hat/outerwear/accessory item must be caught by _CLOTHING_ACCESSORY_PERSISTENT_RE."""
    for item in CLOTHING_ACCESSORY_ITEMS:
        matched = bool(_CLOTHING_ACCESSORY_PERSISTENT_RE.search(item))
        check(
            f"CLOTHING matched: {item!r}",
            matched,
            "expected a match but got none — item would leak into erotic prompt",
        )


def test_restraint_accessories_not_matched_by_clothing_re():
    """Restraint accessories must NOT be caught by _CLOTHING_ACCESSORY_PERSISTENT_RE."""
    for item in CLOTHING_ACCESSORY_SAFE_ITEMS:
        matched = bool(_CLOTHING_ACCESSORY_PERSISTENT_RE.search(item))
        check(
            f"RESTRAINT safe from clothing RE: {item!r}",
            not matched,
            "unexpected match — restraint item would be incorrectly stripped during erotic scene",
        )


def test_clothing_filter_strips_hat_in_erotic_scene():
    """Simulate _state_triggered erotic scene: hat is stripped, collar survives."""
    persistent_items = [
        "leather collar",
        "black cowboy hat",
        "ankle cuffs",
        "red scarf",
        "thin silver chain",
    ]

    expected_kept = {"leather collar", "ankle cuffs", "thin silver chain"}
    expected_removed = {"black cowboy hat", "red scarf"}

    filtered = [item for item in persistent_items if not _CLOTHING_ACCESSORY_PERSISTENT_RE.search(item)]
    removed = [item for item in persistent_items if _CLOTHING_ACCESSORY_PERSISTENT_RE.search(item)]

    for item in expected_kept:
        check(
            f"erotic scene kept: {item!r}",
            item in filtered,
            "item was unexpectedly removed",
        )
    for item in expected_removed:
        check(
            f"erotic scene removed: {item!r}",
            item in removed,
            "item was not removed — clothing would appear in erotic prompt",
        )


def test_clothing_filter_not_applied_in_captive_only_scene():
    """In a captive-only (non-nude) scene, hats in accessories must NOT be stripped.

    The clothing filter only fires when _state_triggered (nude/undressed) is True.
    A captive-only scene with _state_triggered=False must leave clothing items intact.
    This test documents the expected policy: no _CLOTHING_ACCESSORY_PERSISTENT_RE
    filtering when only _captive_triggered is True.
    """
    persistent_items = [
        "leather collar",
        "black cowboy hat",
        "ankle cuffs",
    ]

    # Simulate captive-only: only _WEAPON_PERSISTENT_RE is applied (not clothing RE)
    after_weapon_filter = [item for item in persistent_items if not _WEAPON_PERSISTENT_RE.search(item)]

    check(
        "captive-only: hat retained (clothing filter not applied)",
        "black cowboy hat" in after_weapon_filter,
        "hat was stripped in captive-only path — clothing filter should not fire here",
    )
    check(
        "captive-only: collar retained",
        "leather collar" in after_weapon_filter,
        "collar was incorrectly stripped",
    )


def test_collar_always_survives_clothing_re():
    """Collar must survive _CLOTHING_ACCESSORY_PERSISTENT_RE in all scene types."""
    restraint_items = ["collar", "leather collar", "leash", "ankle cuffs", "wrist cuffs", "chain"]
    for item in restraint_items:
        matched = bool(_CLOTHING_ACCESSORY_PERSISTENT_RE.search(item))
        check(
            f"collar/restraint always survives clothing RE: {item!r}",
            not matched,
            "restraint item matched clothing RE — would be incorrectly stripped",
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

    print("\n── Freed-scene false-positive guards ─────────────────────────────────")
    test_freed_scene_no_false_positives()

    print("\n── Freed-scene true positives ────────────────────────────────────────")
    test_freed_scene_true_positives()

    print("\n── Captive fires despite incidental freedom words ────────────────────")
    test_captive_scene_fires_despite_incidental_freedom_words()

    print("\n── Captive yields on genuine freedom ─────────────────────────────────")
    test_captive_scene_yields_on_genuine_freedom()

    print("\n── Headwear matched by _OUTFIT_SENTENCE_RE ───────────────────────────")
    test_headwear_matched_by_outfit_sentence_re()

    print("\n── Headwear stripped from looks text ─────────────────────────────────")
    test_headwear_stripped_from_looks_by_strip_outfit()

    print("\n── Hat+collar: collar rescued, hat stripped ───────────────────────────")
    test_hat_with_collar_collar_rescued()

    print("\n── Existing outfit terms regression check ────────────────────────────")
    test_non_headwear_outfit_words_still_caught()

    print("\n── Clothing accessories matched by _CLOTHING_ACCESSORY_PERSISTENT_RE ─")
    test_clothing_accessories_matched()

    print("\n── Restraint accessories safe from clothing RE ───────────────────────")
    test_restraint_accessories_not_matched_by_clothing_re()

    print("\n── Clothing filter strips hat in erotic scene ────────────────────────")
    test_clothing_filter_strips_hat_in_erotic_scene()

    print("\n── Clothing filter NOT applied in captive-only scene ─────────────────")
    test_clothing_filter_not_applied_in_captive_only_scene()

    print("\n── Collar always survives clothing RE ────────────────────────────────")
    test_collar_always_survives_clothing_re()

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
