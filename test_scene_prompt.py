"""
Unit tests for pronoun resolution in _assemble_scene_prompt / _infer_gender.

Edge cases covered:
  1. Two female characters  → substitution suppressed (ambiguous) for both "she" and "her"
  2. Zero female characters → "she/her" left unchanged
  3. Player with unknown gender → added to roster without causing substitution
  4. Bot referenced by "self" label → resolved to bot_name, not misattributed
  5. One female character   → "her" (object pronoun) substituted alongside "she"
  6. 'her' before numeral+noun phrase (e.g. 'her two sisters') → possessive, preserved
  7. Player with looks-only profile (appearance in roster_appearances, keyed by display
     name) → correctly inferred as female, making pronouns ambiguous (task #18 regression)

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


def test_infer_gender_name_excluded_female_word():
    """A character named 'Lady' with neutral appearance must not score as female
    just because 'lady' appears in the appearance text as part of their name.
    """
    result = _infer_gender("Lady has a neutral, androgynous appearance.", name="Lady")
    check(
        "_infer_gender: name token 'lady' excluded → ''",
        result == "",
        f"got {result!r}",
    )


def test_infer_gender_name_excluded_male_word():
    """A character named 'He' with neutral appearance must not score as male."""
    result = _infer_gender("He is an indistinct figure of average build.", name="He")
    check(
        "_infer_gender: name token 'he' excluded → ''",
        result == "",
        f"got {result!r}",
    )


def test_infer_gender_name_excluded_multiword():
    """A character named 'Miss Chen' must not have 'miss' counted as a gender hit."""
    result = _infer_gender("Miss Chen has a neutral style and short dark hair.", name="Miss Chen")
    check(
        "_infer_gender: multi-word name tokens excluded → ''",
        result == "",
        f"got {result!r}",
    )


def test_infer_gender_name_boy_not_classified_male():
    """A character named 'Boy' with neutral text must not score as male."""
    result = _infer_gender("Boy stands quietly in the corner with an unreadable expression.", name="Boy")
    check(
        "_infer_gender: name 'boy' excluded → ''",
        result == "",
        f"got {result!r}",
    )


def test_infer_gender_name_male_not_classified_male():
    """A character named 'Male' with neutral text must not score as male."""
    result = _infer_gender("Male has an average build and unremarkable features.", name="Male")
    check(
        "_infer_gender: name 'male' excluded → ''",
        result == "",
        f"got {result!r}",
    )


def test_infer_gender_name_gentleman_not_classified_male():
    """A character named 'Gentleman' with neutral text must not score as male."""
    result = _infer_gender("Gentleman moves with a calm and composed manner.", name="Gentleman")
    check(
        "_infer_gender: name 'gentleman' excluded → ''",
        result == "",
        f"got {result!r}",
    )


def test_infer_gender_name_woman_not_classified_female():
    """A character named 'Woman' with neutral text must not score as female."""
    result = _infer_gender("Woman has an unassuming look and keeps to herself.", name="Woman")
    check(
        "_infer_gender: name 'woman' excluded → ''",
        result == "",
        f"got {result!r}",
    )


def test_infer_gender_her_alone_not_female():
    """'her' alone (possessive) must NOT infer female after tightening.

    Before the word-set tightening, 'her' was in _FEMALE_GENDER_WORDS and this
    description would have returned 'f'.  A neutral scene description that
    happens to use the possessive 'her' must now return '' so that pronoun
    resolution is not triggered incorrectly.
    """
    result = _infer_gender("The teacher passed her the assignment on Monday.")
    check(
        "_infer_gender: possessive 'her' alone → '' (not 'f')",
        result == "",
        f"got {result!r}",
    )


def test_infer_gender_miss_alone_not_female():
    """'miss' as a verb must NOT infer female after tightening.

    Before the tightening, 'miss' was in _FEMALE_GENDER_WORDS and a sentence
    like "don't miss the opportunity" would have incorrectly returned 'f'.
    """
    result = _infer_gender("Don't miss the opportunity to attend the event.")
    check(
        "_infer_gender: verbal 'miss' alone → '' (not 'f')",
        result == "",
        f"got {result!r}",
    )


def test_infer_gender_him_alone_not_male():
    """'him' as object pronoun must NOT infer male after tightening.

    Before the tightening, 'him' was in _MALE_GENDER_WORDS and a neutral
    sentence like "the crowd cheered for him" would have returned 'm'.
    """
    result = _infer_gender("The crowd cheered for him loudly.")
    check(
        "_infer_gender: object 'him' alone → '' (not 'm')",
        result == "",
        f"got {result!r}",
    )


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


def test_one_female_her_substitution():
    """With exactly one female, 'her' (object pronoun) should also be replaced by her name."""
    result = _assemble_scene_prompt(
        seed="She handed it to her.",
        prose_context=None,
        roster_names=["Luna"],
        roster_appearances={"Luna": FEMALE_APP},
        bot_name="Luna",
        player_display_name=None,
    )
    check(
        "One female: 'her' replaced with character name",
        result.count("Luna") == 2,
        f"result={result!r}",
    )
    check(
        "One female: bare 'her' not present after substitution",
        "her" not in result.lower().replace("luna", ""),
        f"result={result!r}",
    )


def test_possessive_her_not_substituted():
    """Possessive 'her' (followed by a noun) must NOT be replaced with the character name."""
    result = _assemble_scene_prompt(
        seed="She looked at her reflection in the mirror.",
        prose_context=None,
        roster_names=["Mira"],
        roster_appearances={"Mira": FEMALE_APP},
        bot_name="Mira",
        player_display_name=None,
    )
    check(
        "Possessive 'her': subject 'she' replaced with name",
        "Mira" in result,
        f"result={result!r}",
    )
    check(
        "Possessive 'her': 'her reflection' not mangled to 'Mira reflection'",
        "Mira reflection" not in result,
        f"result={result!r}",
    )
    check(
        "Possessive 'her': possessive 'her' preserved before noun",
        "her reflection" in result.lower(),
        f"result={result!r}",
    )


def test_object_her_followed_by_adverb_is_substituted():
    """Object 'her' followed by an adverb (e.g. 'to her yesterday') must still be replaced."""
    result = _assemble_scene_prompt(
        seed="She gave the key to her yesterday.",
        prose_context=None,
        roster_names=["Mira"],
        roster_appearances={"Mira": FEMALE_APP},
        bot_name="Mira",
        player_display_name=None,
    )
    check(
        "Object 'her' + adverb: 'she' replaced with name",
        "Mira" in result,
        f"result={result!r}",
    )
    check(
        "Object 'her' + adverb: 'to her yesterday' replaced with 'to Mira yesterday'",
        "to Mira yesterday" in result,
        f"result={result!r}",
    )
    check(
        "Object 'her' + adverb: bare 'her' not present after substitution",
        " her " not in result and not result.endswith(" her"),
        f"result={result!r}",
    )


def test_possessive_her_multiword_adjective_noun():
    """Possessive 'her' followed by an adjective+noun phrase must NOT be replaced."""
    cases = [
        ("She tossed her long hair over one shoulder.", "her long hair"),
        ("She narrowed her bright eyes at the horizon.", "her bright eyes"),
        ("She adjusted her silver bracelet carefully.", "her silver bracelet"),
    ]
    for seed, expected_phrase in cases:
        result = _assemble_scene_prompt(
            seed=seed,
            prose_context=None,
            roster_names=["Mira"],
            roster_appearances={"Mira": FEMALE_APP},
            bot_name="Mira",
            player_display_name=None,
        )
        check(
            f"Possessive 'her' + adj+noun: 'she' replaced with name in {seed!r}",
            "Mira" in result,
            f"result={result!r}",
        )
        check(
            f"Possessive 'her' + adj+noun: phrase {expected_phrase!r} preserved",
            expected_phrase in result.lower(),
            f"result={result!r}",
        )
        check(
            f"Possessive 'her' + adj+noun: no 'Mira' inserted before adjective in {seed!r}",
            "Mira " + expected_phrase.split()[1] not in result,
            f"result={result!r}",
        )


def test_possessive_her_before_numeral_noun():
    """Possessive 'her' followed by a cardinal number + noun must NOT be replaced.

    Cardinal numbers (two, three, four …) are not in _FUNC_WORDS and don't
    end in -ly, so the possessive guard treats them as content words and
    preserves 'her'.
    """
    cases = [
        ("She clutched her two sisters close.", "her two sisters"),
        ("She wore her three rings proudly.", "her three rings"),
        ("She could feel her five senses sharpen.", "her five senses"),
        ("She counted her seven steps to the door.", "her seven steps"),
    ]
    for seed, expected_phrase in cases:
        result = _assemble_scene_prompt(
            seed=seed,
            prose_context=None,
            roster_names=["Mira"],
            roster_appearances={"Mira": FEMALE_APP},
            bot_name="Mira",
            player_display_name=None,
        )
        check(
            f"Numeral phrase: 'she' replaced in {seed!r}",
            "Mira" in result,
            f"result={result!r}",
        )
        check(
            f"Numeral phrase: {expected_phrase!r} preserved in {seed!r}",
            expected_phrase in result.lower(),
            f"result={result!r}",
        )
        check(
            f"Numeral phrase: no name inserted before number in {seed!r}",
            "Mira " + expected_phrase.split()[1] not in result,
            f"result={result!r}",
        )


def test_possessive_her_before_digit_numeral():
    """Possessive 'her' followed by a digit numeral + noun must NOT be replaced.

    Digit strings (e.g. "2", "10") are not in _FUNC_WORDS and don't end in
    '-ly', so the possessive guard already fires and preserves 'her'.
    """
    cases = [
        ("She clutched her 2 sisters close.", "her 2 sisters"),
        ("She wore her 3 rings proudly.", "her 3 rings"),
        ("She counted her 10 steps to the door.", "her 10 steps"),
        ("She carried her 5 bags inside.", "her 5 bags"),
    ]
    for seed, expected_phrase in cases:
        result = _assemble_scene_prompt(
            seed=seed,
            prose_context=None,
            roster_names=["Mira"],
            roster_appearances={"Mira": FEMALE_APP},
            bot_name="Mira",
            player_display_name=None,
        )
        check(
            f"Digit numeral phrase: 'she' replaced in {seed!r}",
            "Mira" in result,
            f"result={result!r}",
        )
        check(
            f"Digit numeral phrase: {expected_phrase!r} preserved in {seed!r}",
            expected_phrase in result.lower(),
            f"result={result!r}",
        )
        check(
            f"Digit numeral phrase: no name inserted before digit in {seed!r}",
            "Mira " + expected_phrase.split()[1] not in result,
            f"result={result!r}",
        )


def test_possessive_her_before_ordinal():
    """Possessive 'her' followed by an ordinal word + noun must NOT be replaced.

    Ordinal words (first, second, third …) are not in _FUNC_WORDS and don't
    end in '-ly', so the possessive guard already fires and preserves 'her'.
    """
    cases = [
        ("She seized her first chance.", "her first chance"),
        ("She remembered her second attempt.", "her second attempt"),
        ("She slipped on her third ring.", "her third ring"),
        ("She felt her fourth heartbeat skip.", "her fourth heartbeat"),
        ("She treasured her fifth gift.", "her fifth gift"),
    ]
    for seed, expected_phrase in cases:
        result = _assemble_scene_prompt(
            seed=seed,
            prose_context=None,
            roster_names=["Mira"],
            roster_appearances={"Mira": FEMALE_APP},
            bot_name="Mira",
            player_display_name=None,
        )
        check(
            f"Ordinal phrase: 'she' replaced in {seed!r}",
            "Mira" in result,
            f"result={result!r}",
        )
        check(
            f"Ordinal phrase: {expected_phrase!r} preserved in {seed!r}",
            expected_phrase in result.lower(),
            f"result={result!r}",
        )
        check(
            f"Ordinal phrase: no name inserted before ordinal in {seed!r}",
            "Mira " + expected_phrase.split()[1] not in result,
            f"result={result!r}",
        )


def test_possessive_his_not_substituted():
    """Possessive 'his' (followed by a noun) must NOT be replaced with the character name."""
    result = _assemble_scene_prompt(
        seed="He grabbed his coat from the chair.",
        prose_context=None,
        roster_names=["Ethan"],
        roster_appearances={"Ethan": MALE_APP},
        bot_name="Ethan",
        player_display_name=None,
    )
    check(
        "Possessive 'his': 'his coat' not mangled to 'Ethan coat'",
        "Ethan coat" not in result,
        f"result={result!r}",
    )
    check(
        "Possessive 'his': possessive 'his' preserved before noun",
        "his coat" in result.lower(),
        f"result={result!r}",
    )
    check(
        "Possessive 'his': subject 'he' still replaced with name",
        "Ethan" in result,
        f"result={result!r}",
    )


def test_two_females_her_not_substituted():
    """When two female characters are present, 'her' must NOT be replaced (ambiguous)."""
    result = _assemble_scene_prompt(
        seed="She handed it to her.",
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
        "Two females: 'her' not replaced (ambiguous)",
        "her" in result.lower(),
        f"result={result!r}",
    )


def test_male_and_female_both_unique():
    """One male + one female: both subject and object pronouns resolved without cross-contamination."""
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
        "Mixed genders: 'him' replaced with male name",
        "Ethan" in result,
        f"result={result!r}",
    )
    check(
        "Mixed genders: Zara not inserted as replacement for 'him'",
        result.count("Zara") == 1,  # appears once (from 'she' substitution only)
        f"result={result!r}",
    )


# ── Character name contains gender word (task #7 regression) ─────────────────

def test_character_named_lady_not_misclassified():
    """A character whose name is 'Lady' must not be inferred as female solely
    because their name token appears in their appearance description.

    Without the name-exclusion fix, 'lady' in the appearance text would match
    _FEMALE_GENDER_WORDS and force gender 'f', potentially making pronoun
    substitution ambiguous when it should not be.
    """
    result = _assemble_scene_prompt(
        seed="He stood next to the figure.",
        prose_context=None,
        roster_names=["Lady", "Carlos"],
        roster_appearances={
            "Lady":   "Lady has a neutral, androgynous appearance with short dark hair.",
            "Carlos": MALE_APP,
        },
        bot_name="Lady",
        player_display_name=None,
    )
    check(
        "Character named 'Lady' (neutral appearance): 'he' still replaced with male name",
        "Carlos" in result,
        f"result={result!r}",
    )
    check(
        "Character named 'Lady' (neutral appearance): 'Lady' not inserted via pronoun",
        result.count("Lady") == 0,
        f"result={result!r}",
    )


def test_character_named_miss_not_misclassified():
    """A character named 'Miss Chen' with neutral appearance must not be inferred
    as female just because 'miss' appears in their appearance text.
    """
    result = _assemble_scene_prompt(
        seed="She walked into the room.",
        prose_context=None,
        roster_names=["Miss Chen"],
        roster_appearances={
            "Miss Chen": "Miss Chen has a neutral style and short dark hair.",
        },
        bot_name="Miss Chen",
        player_display_name=None,
    )
    check(
        "Character named 'Miss Chen' (neutral appearance): 'she' not replaced (gender unknown)",
        "she" in result.lower(),
        f"result={result!r}",
    )
    check(
        "Character named 'Miss Chen': name not spuriously inserted via pronoun",
        "Miss Chen" not in result,
        f"result={result!r}",
    )


# ── Looks-only player gender inference (task #18 regression) ─────────────────

def test_looks_only_female_player_she_not_substituted():
    """Regression test for task #18 bug.

    Before the fix, player_display_name was added to the roster with an empty
    gender string ('') regardless of roster_appearances — so a female-looking
    player was mis-identified as gender-unknown.  With only the bot counted as
    female, 'she' was (incorrectly) unambiguous and resolved to the bot name.

    After the fix, roster_appearances is consulted using player_display_name as
    the key, so Yuki is inferred as female.  Two females → 'she' is ambiguous
    and must be left unchanged.
    """
    result = _assemble_scene_prompt(
        seed="She walked to the window.",
        prose_context=None,
        roster_names=["Aria"],
        roster_appearances={
            "Aria": FEMALE_APP,
            "Yuki": FEMALE_APP,   # player appearance, keyed by display name
        },
        bot_name="Aria",
        player_display_name="Yuki",
    )
    check(
        "Looks-only female player + female bot: 'she' preserved (two females → ambiguous)",
        "she" in result.lower(),
        f"result={result!r}",
    )
    check(
        "Looks-only female player: 'Aria' not injected via pronoun (would signal pre-fix bug)",
        "Aria" not in result,
        f"result={result!r}",
    )


def test_looks_only_female_player_her_not_substituted():
    """Same regression as above but for the object/possessive pronoun 'her'."""
    result = _assemble_scene_prompt(
        seed="The camera focused on her.",
        prose_context=None,
        roster_names=["Aria"],
        roster_appearances={
            "Aria": FEMALE_APP,
            "Yuki": FEMALE_APP,
        },
        bot_name="Aria",
        player_display_name="Yuki",
    )
    check(
        "Looks-only female player + female bot: 'her' preserved (two females → ambiguous)",
        "her" in result.lower(),
        f"result={result!r}",
    )
    check(
        "Looks-only female player 'her': 'Aria' not injected via pronoun",
        "Aria" not in result,
        f"result={result!r}",
    )


def test_looks_only_male_player_she_substituted_to_bot():
    """Player with male appearance in roster_appearances → only bot is female
    → 'she' unambiguously resolves to the bot name, 'him' to the player name.
    """
    result = _assemble_scene_prompt(
        seed="She sat across from him.",
        prose_context=None,
        roster_names=["Aria"],
        roster_appearances={
            "Aria":  FEMALE_APP,
            "Kenji": MALE_APP,    # player's appearance, keyed by display name
        },
        bot_name="Aria",
        player_display_name="Kenji",
    )
    check(
        "Looks-only male player + female bot: 'she' substituted to bot name",
        "Aria" in result,
        f"result={result!r}",
    )
    check(
        "Looks-only male player + female bot: 'him' substituted to player name",
        "Kenji" in result,
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
    test_infer_gender_name_excluded_female_word()
    test_infer_gender_name_excluded_male_word()
    test_infer_gender_name_excluded_multiword()
    test_infer_gender_name_boy_not_classified_male()
    test_infer_gender_name_male_not_classified_male()
    test_infer_gender_name_gentleman_not_classified_male()
    test_infer_gender_name_woman_not_classified_female()
    test_infer_gender_her_alone_not_female()
    test_infer_gender_miss_alone_not_female()
    test_infer_gender_him_alone_not_male()

    print()
    test_two_females_no_substitution()
    test_zero_females_no_substitution()
    test_one_female_substitution_occurs()
    test_player_unknown_gender_does_not_trigger_substitution()
    test_bot_self_label_resolved_to_bot_name()
    test_female_bot_male_player_only_she_substituted()
    test_one_female_her_substitution()
    test_possessive_her_not_substituted()
    test_object_her_followed_by_adverb_is_substituted()
    test_possessive_her_multiword_adjective_noun()
    test_possessive_her_before_numeral_noun()
    test_possessive_her_before_digit_numeral()
    test_possessive_her_before_ordinal()
    test_possessive_his_not_substituted()
    test_two_females_her_not_substituted()
    test_male_and_female_both_unique()

    print()
    test_character_named_lady_not_misclassified()
    test_character_named_miss_not_misclassified()

    print()
    test_looks_only_female_player_she_not_substituted()
    test_looks_only_female_player_her_not_substituted()
    test_looks_only_male_player_she_substituted_to_bot()

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
