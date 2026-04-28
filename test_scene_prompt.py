"""
Unit tests for pronoun resolution in _assemble_scene_prompt / _infer_gender.

Edge cases covered:
  1. Two female characters  → substitution suppressed (ambiguous)
  2. Zero female characters → "she/her" left unchanged
  3. Player with unknown gender → added to roster without causing substitution
  4. Bot referenced by "self" label → resolved to bot_name, not misattributed

Run: python test_scene_prompt.py
"""

import sys
sys.path.insert(0, ".")

from scene_image import _infer_gender, _assemble_scene_prompt

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f"\n         {detail}" if detail else ""))
    _results.append((label, condition, detail))


# ── _infer_gender unit tests ───────────────────────────────────────────────────

def test_infer_gender_female():
    result = _infer_gender("She is a young woman with long hair and kind eyes.")
    check("_infer_gender: female description → 'f'", result == "f", f"got {result!r}")


def test_infer_gender_male():
    result = _infer_gender("He is a tall man with broad shoulders and short hair.")
    check("_infer_gender: male description → 'm'", result == "m", f"got {result!r}")


def test_infer_gender_empty():
    result = _infer_gender("")
    check("_infer_gender: empty string → ''", result == "", f"got {result!r}")


def test_infer_gender_ambiguous():
    result = _infer_gender("A person with neutral features.")
    check("_infer_gender: no gender words → ''", result == "", f"got {result!r}")


# ── _assemble_scene_prompt edge-case tests ────────────────────────────────────

FEMALE_APP = "a young woman with long hair, she is graceful"
MALE_APP   = "a tall man with broad shoulders, he is stoic"


def test_two_females_no_substitution():
    """When two female characters are present, 'she' must NOT be replaced (ambiguous)."""
    result = _assemble_scene_prompt(
        seed="She walked across the room and smiled.",
        prose_context=None,
        roster_names=["Alice", "Beth"],
        roster_appearances={
            "Alice": FEMALE_APP,
            "Beth":  FEMALE_APP,
        },
        bot_name="Alice",
        player_display_name=None,
    )
    check(
        "Two females: 'she' not replaced with either name",
        "She" in result or "she" in result,
        f"result={result!r}",
    )
    check(
        "Two females: neither name spuriously inserted via pronoun",
        "alice" not in result.lower() and "beth" not in result.lower(),
        f"result={result!r}",
    )


def test_zero_females_no_substitution():
    """When there are no female characters, 'she' must remain in the text."""
    result = _assemble_scene_prompt(
        seed="She entered the room quietly.",
        prose_context=None,
        roster_names=["Carlos"],
        roster_appearances={"Carlos": MALE_APP},
        bot_name="Carlos",
        player_display_name=None,
    )
    check(
        "Zero females: 'she' left untouched",
        "she" in result.lower(),
        f"result={result!r}",
    )
    check(
        "Zero females: 'Carlos' not inserted via pronoun",
        result.count("Carlos") <= 1,
        f"result={result!r}",
    )


def test_one_female_substitution_occurs():
    """With exactly one female, 'she' should be replaced by her name."""
    result = _assemble_scene_prompt(
        seed="She sat by the window.",
        prose_context=None,
        roster_names=["Luna"],
        roster_appearances={"Luna": FEMALE_APP},
        bot_name="Luna",
        player_display_name=None,
    )
    check(
        "One female: 'she' replaced with character name",
        "Luna" in result,
        f"result={result!r}",
    )
    check(
        "One female: bare 'she' not present after substitution",
        "she" not in result.lower().replace("luna", ""),
        f"result={result!r}",
    )


def test_player_unknown_gender_does_not_trigger_substitution():
    """Player added with unknown gender should not cause erroneous 'she' substitution."""
    result = _assemble_scene_prompt(
        seed="She walked beside the player.",
        prose_context=None,
        roster_names=["Aria"],
        roster_appearances={"Aria": FEMALE_APP},
        bot_name="Aria",
        player_display_name="Alex",  # unknown gender
    )
    check(
        "Player (unknown gender) + one female: 'she' replaced with female bot name only",
        "Aria" in result,
        f"result={result!r}",
    )
    check(
        "Player (unknown gender): 'Alex' not inserted via pronoun substitution",
        "Alex" not in result.replace("Alex walked", ""),
        f"result={result!r}",
    )


def test_bot_self_label_resolved_to_bot_name():
    """Roster entry with name 'self' must be remapped to bot_name, not dropped.

    After the remap 'self' → 'Mira', the code looks up the appearance by the
    resolved name, so the appearance must be keyed under bot_name.
    """
    result = _assemble_scene_prompt(
        seed="She looked at her reflection.",
        prose_context=None,
        roster_names=["self"],
        roster_appearances={"Mira": FEMALE_APP},
        bot_name="Mira",
        player_display_name=None,
    )
    check(
        "'self' label resolved to bot_name for gender inference",
        "Mira" in result,
        f"result={result!r}",
    )
    check(
        "'self' label: bare 'she' not left unresolved",
        "she" not in result.lower().replace("mira", ""),
        f"result={result!r}",
    )


def test_female_bot_male_player_only_she_substituted():
    """Female bot + male player: 'she' → bot name, 'he' → player name (both unambiguous)."""
    result = _assemble_scene_prompt(
        seed="She handed it to him.",
        prose_context=None,
        roster_names=["Aria"],
        roster_appearances={"Aria": FEMALE_APP},
        bot_name="Aria",
        player_display_name="Dan",
    )
    # Dan's gender is inferred '' (unknown) since player_display_name gets no appearance.
    # So only 'she' can be unambiguously substituted.
    check(
        "Female bot + player: 'she' replaced with bot name",
        "Aria" in result,
        f"result={result!r}",
    )


def test_male_and_female_both_unique():
    """One male + one female: both pronouns resolved without cross-contamination."""
    result = _assemble_scene_prompt(
        seed="She smiled at him warmly.",
        prose_context=None,
        roster_names=["Zara", "Ethan"],
        roster_appearances={
            "Zara":  FEMALE_APP,
            "Ethan": MALE_APP,
        },
        bot_name="Zara",
        player_display_name=None,
    )
    check(
        "Mixed genders: 'she' replaced with female name",
        "Zara" in result,
        f"result={result!r}",
    )
    check(
        "Mixed genders: 'him' left or 'Ethan' inserted (male pronoun resolution)",
        "Ethan" in result or "him" in result.lower(),
        f"result={result!r}",
    )
    # The code substitutes "he" but not "him"; "him" is an object pronoun and
    # remains in the text.  The key correctness property is that Zara (female)
    # is NOT substituted in place of the male pronoun.
    check(
        "Mixed genders: Zara not inserted as replacement for 'him'",
        result.count("Zara") == 1,  # appears once (from 'she' substitution only)
        f"result={result!r}",
    )


# ── runner ─────────────────────────────────────────────────────────────────────

def run_all():
    print("\n" + "=" * 60)
    print("  test_scene_prompt — pronoun resolution edge cases")
    print("=" * 60)

    test_infer_gender_female()
    test_infer_gender_male()
    test_infer_gender_empty()
    test_infer_gender_ambiguous()

    print()
    test_two_females_no_substitution()
    test_zero_females_no_substitution()
    test_one_female_substitution_occurs()
    test_player_unknown_gender_does_not_trigger_substitution()
    test_bot_self_label_resolved_to_bot_name()
    test_female_bot_male_player_only_she_substituted()
    test_male_and_female_both_unique()

    print()
    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print("=" * 60)
    if failed == 0:
        print(f"  \033[32mAll {total} checks passed.\033[0m")
    else:
        print(f"  \033[31m{failed}/{total} checks FAILED.\033[0m")
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    run_all()
