"""
Integration test for the `with:` clause pipeline.

Verifies — without a live LM Studio or ComfyUI session — that:

  1. `_SCENE_MARKER_RE` (lmstudio_ai) correctly extracts the raw body
     `"she smiles | with: Saki Nikaido"` from a model reply.

  2. `scene_prompt` returned by the `chat()` pipeline carries that body
     verbatim (the part that scene_image.py will later parse).

  3. `_parse_with_clause` (scene_image) splits the body into
     scene_prompt="she smiles" and explicit_subjects=["Saki Nikaido"].

  4. `_entry_matches_explicit` resolves "Saki Nikaido" against a KB
     entry whose title is "Saki Nikaido" (exact match) and also via
     an alias, while rejecting unrelated entries and near-misses.

  5. `_gather_refs` (scene_image) — with the database layer monkeypatched
     to inject a synthetic KB — resolves the `with:` name to the correct
     KB entry and surfaces it in the returned subjects list.

  6. `lmstudio_ai.chat()` — with `_call_lmstudio` stubbed to return a
     pronoun-heavy reply containing `[SCENE: she smiles | with: Saki Nikaido]`
     — returns `wants_scene=True`, `scene_prompt=="she smiles | with: Saki Nikaido"`,
     and strips the token from the visible reply; feeding that `scene_prompt`
     through `_parse_with_clause` + `_gather_refs` produces the expected subjects.

Run directly:
    python test_with_clause.py
"""

import asyncio
import sys
import types
import re

# ── helpers ───────────────────────────────────────────────────────────────────

PASS_COUNT = 0
FAIL_COUNT = 0


def ok(label: str) -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  [PASS] {label}")


def fail(label: str, detail: str = "") -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    msg = f"  [FAIL] {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)


def expect_eq(label: str, got, expected) -> None:
    if got == expected:
        ok(label)
    else:
        fail(label, f"got {got!r}, expected {expected!r}")


def expect_true(label: str, value) -> None:
    if value:
        ok(label)
    else:
        fail(label, f"expected truthy, got {value!r}")


def expect_false(label: str, value) -> None:
    if not value:
        ok(label)
    else:
        fail(label, f"expected falsy, got {value!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  _SCENE_MARKER_RE  — raw regex extraction
# ═══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, ".")
import lmstudio_ai  # noqa: E402  (needed for _SCENE_MARKER_RE and chat())
from lmstudio_ai import _SCENE_MARKER_RE  # use production symbol — not a local copy

print("\n── 1. _SCENE_MARKER_RE (lmstudio_ai) ──")

REPLY_WITH_CLAUSE = (
    'She tilts her head and lets out a soft laugh.\n'
    '[SCENE: she smiles | with: Saki Nikaido]\n'
    '"You always know exactly what to say," she murmurs.'
)

REPLY_BARE = "She waves goodbye. [SCENE]"
REPLY_NO_WITH = "The sun sets beautifully. [SCENE: golden hour over the city]"

m = _SCENE_MARKER_RE.search(REPLY_WITH_CLAUSE)
expect_true("regex matches reply containing [SCENE: ... | with: ...]", m)
if m:
    body = (m.group(1) or "").strip()
    expect_eq(
        "body extracted verbatim (including with-clause)",
        body,
        "she smiles | with: Saki Nikaido",
    )

m_bare = _SCENE_MARKER_RE.search(REPLY_BARE)
expect_true("regex matches bare [SCENE]", m_bare)
if m_bare:
    expect_eq("bare [SCENE] body is empty string", (m_bare.group(1) or "").strip(), "")

m_no_with = _SCENE_MARKER_RE.search(REPLY_NO_WITH)
expect_true("regex matches [SCENE: ...] without with clause", m_no_with)
if m_no_with:
    expect_eq(
        "body without with-clause returned as-is",
        (m_no_with.group(1) or "").strip(),
        "golden hour over the city",
    )

stripped = _SCENE_MARKER_RE.sub("", REPLY_WITH_CLAUSE).strip()
expect_false("[SCENE: ...] token removed from visible text", "[SCENE" in stripped)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  _parse_with_clause  (scene_image)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── 2. _parse_with_clause (scene_image) ──")

from scene_image import _parse_with_clause  # noqa: E402

body_cleaned, names = _parse_with_clause("she smiles | with: Saki Nikaido")
expect_eq("scene_prompt cleaned to 'she smiles'", body_cleaned, "she smiles")
expect_eq("with-clause produces one name", names, ["Saki Nikaido"])

body_cleaned2, names2 = _parse_with_clause("two friends laugh | with: Alice, Bob")
expect_eq("prompt cleaned when two names present", body_cleaned2, "two friends laugh")
expect_eq("two names split correctly", names2, ["Alice", "Bob"])

body_cleaned3, names3 = _parse_with_clause("a peaceful garden scene")
expect_eq("body unchanged when no with-clause", body_cleaned3, "a peaceful garden scene")
expect_eq("empty names list when no with-clause", names3, [])

body_cleaned4, names4 = _parse_with_clause("")
expect_eq("empty input returns empty body", body_cleaned4, "")
expect_eq("empty input returns empty list", names4, [])

body_cleaned5, names5 = _parse_with_clause("  with: Leading spaces  | with: Trailing Name  ")
expect_eq("trailing name trimmed", names5, ["Trailing Name"])


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  _entry_matches_explicit  (scene_image)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── 3. _entry_matches_explicit (scene_image) ──")

from scene_image import _entry_matches_explicit  # noqa: E402

SAKI_ENTRY = {
    "id": 1,
    "title": "Saki Nikaido",
    "aliases": "Saki,二階堂サキ",
    "appearance_description": "pink twin-tails, punk outfit",
}

UNRELATED_ENTRY = {
    "id": 2,
    "title": "Bocchi The Rock",
    "aliases": "Bocchi,Hitori Gotoh",
    "appearance_description": "pink long hair, shy",
}

expect_true(
    "exact title match 'Saki Nikaido'",
    _entry_matches_explicit(SAKI_ENTRY, "Saki Nikaido"),
)
expect_true(
    "case-insensitive title match 'saki nikaido'",
    _entry_matches_explicit(SAKI_ENTRY, "saki nikaido"),
)
expect_true(
    "alias match 'Saki'",
    _entry_matches_explicit(SAKI_ENTRY, "Saki"),
)
expect_true(
    "alias match '二階堂サキ'",
    _entry_matches_explicit(SAKI_ENTRY, "二階堂サキ"),
)
expect_false(
    "partial name 'Saki N' does NOT match (no substring guessing)",
    _entry_matches_explicit(SAKI_ENTRY, "Saki N"),
)
expect_false(
    "unrelated entry does not match 'Saki Nikaido'",
    _entry_matches_explicit(UNRELATED_ENTRY, "Saki Nikaido"),
)
expect_false(
    "empty query returns False",
    _entry_matches_explicit(SAKI_ENTRY, ""),
)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  _gather_refs end-to-end with mocked database
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── 4. _gather_refs end-to-end (mocked database) ──")

import database as _db_module  # noqa: E402  (needed for monkeypatching)
import scene_image  # noqa: E402

# Build a minimal in-memory KB:  Saki Nikaido (id=10)  +  Bocchi (id=11)
_KB_ENTRIES = [
    {
        "id": 10,
        "title": "Saki Nikaido",
        "aliases": "Saki,二階堂サキ",
        "appearance_description": "pink twin-tails, punk outfit",
    },
    {
        "id": 11,
        "title": "Bocchi The Rock",
        "aliases": "Bocchi",
        "appearance_description": "pink long hair, shy",
    },
]

_KB_IMAGES = {
    10: [("saki_full.png", b"\x00", "image/png")],
    11: [("bocchi_full.png", b"\x00", "image/png")],
}

# Patch database functions used by _gather_refs
_db_module_orig_get_character = _db_module.get_character
_db_module_orig_get_character_image_count = _db_module.get_character_image_count
_db_module_orig_get_image_entries = _db_module.get_image_entries
_db_module_orig_get_entry_image_count = _db_module.get_entry_image_count
_db_module_orig_get_kb_image_full = _db_module.get_kb_image_full

_db_module.get_character = lambda: {"name": "TestBot", "looks": "tall, silver hair"}
_db_module.get_character_image_count = lambda: 0  # no bot photos — keeps refs list simple
_db_module.get_image_entries = lambda: _KB_ENTRIES
_db_module.get_entry_image_count = lambda entry_id: len(_KB_IMAGES.get(entry_id, []))
def _mock_get_kb_image_full(entry_id, image_index=1):
    """Mock that mirrors database.get_kb_image_full's None-on-out-of-range
    behaviour. The real implementation returns None when image_index is out
    of bounds; the test mock previously raised IndexError, which now hits
    the relaxed Pass-B cap (single-subject scenes ask for higher indices)."""
    images = _KB_IMAGES.get(entry_id) or []
    if image_index < 1 or image_index > len(images):
        return None
    return images[image_index - 1]


_db_module.get_kb_image_full = _mock_get_kb_image_full

try:
    # ── 4a. Explicit with: Saki Nikaido should pull Saki's entry ──────────────
    refs, subjects, appearances, _decision = scene_image._gather_refs(
        "she smiles softly",
        explicit_subjects=["Saki Nikaido"],
    )

    expect_true("refs list is non-empty for explicit Saki Nikaido", len(refs) > 0)
    expect_true("'Saki Nikaido' appears in subjects", "Saki Nikaido" in subjects)
    expect_false("'Bocchi The Rock' NOT in subjects (not in with-clause)", "Bocchi The Rock" in subjects)
    expect_true("Saki's appearance description in appearances dict", "Saki Nikaido" in appearances)
    expect_eq(
        "Saki's appearance text correct",
        appearances["Saki Nikaido"],
        "pink twin-tails, punk outfit",
    )

    # ── 4b. No explicit clause + seed mentioning Bocchi → fuzzy picks Bocchi ──
    refs2, subjects2, appearances2, _decision2 = scene_image._gather_refs(
        "Bocchi stands alone with her guitar",
        explicit_subjects=[],
    )

    expect_true("fuzzy seed match: refs non-empty for Bocchi seed", len(refs2) > 0)
    expect_true("fuzzy seed match: Bocchi in subjects", "Bocchi The Rock" in subjects2)
    expect_false("fuzzy seed match: Saki NOT in subjects", "Saki Nikaido" in subjects2)

    # ── 4c. Unknown explicit name falls back gracefully (empty refs) ───────────
    refs3, subjects3, _, _decision3 = scene_image._gather_refs(
        "mysterious stranger smiles",
        explicit_subjects=["NonExistentCharacter"],
    )
    expect_false(
        "unknown explicit name yields no refs (strict matching, no fallback)",
        "NonExistentCharacter" in subjects3,
    )

    # ── 4d. Full marker string parsed and forwarded correctly ─────────────────
    full_marker_body = "she smiles | with: Saki Nikaido"
    cleaned_seed, explicit = _parse_with_clause(full_marker_body)
    refs4, subjects4, _, _decision4 = scene_image._gather_refs(cleaned_seed, explicit_subjects=explicit)

    expect_eq("full pipeline: cleaned seed is 'she smiles'", cleaned_seed, "she smiles")
    expect_eq("full pipeline: explicit is ['Saki Nikaido']", explicit, ["Saki Nikaido"])
    expect_true("full pipeline: Saki in final subjects", "Saki Nikaido" in subjects4)

finally:
    # Restore original database functions
    _db_module.get_character = _db_module_orig_get_character
    _db_module.get_character_image_count = _db_module_orig_get_character_image_count
    _db_module.get_image_entries = _db_module_orig_get_image_entries
    _db_module.get_entry_image_count = _db_module_orig_get_entry_image_count
    _db_module.get_kb_image_full = _db_module_orig_get_kb_image_full


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  lmstudio_ai.chat()  — full pipeline with stubbed _call_lmstudio
# ═══════════════════════════════════════════════════════════════════════════════

print("\n── 5. lmstudio_ai.chat() with stubbed _call_lmstudio ──")

_CHAT_STUB_REPLY = (
    'She tilts her head and lets out a soft laugh.\n'
    '[SCENE: she smiles | with: Saki Nikaido]\n'
    '"You always know exactly what to say," she murmurs.'
)

_orig_call_lmstudio = lmstudio_ai._call_lmstudio


async def _stub_call(*args, **kwargs):
    return _CHAT_STUB_REPLY


lmstudio_ai._call_lmstudio = _stub_call

try:
    result = asyncio.run(lmstudio_ai.chat(
        messages=[{"role": "user", "content": "She is acting mysterious again."}],
        system_prompt="You are a test character.",
        enforce_user_lang=False,
    ))

    reply_text, img_prompt, _pfm, success, wants_scene, scene_prompt = result

    expect_true("chat() call succeeded (success flag)", success)
    expect_true("chat() wants_scene is True", wants_scene)
    expect_eq(
        "chat() scene_prompt == 'she smiles | with: Saki Nikaido'",
        scene_prompt,
        "she smiles | with: Saki Nikaido",
    )
    expect_false(
        "chat() [SCENE] tag stripped from visible reply text",
        "[SCENE" in (reply_text or ""),
    )

    if scene_prompt:
        cleaned_chat, explicit_chat = _parse_with_clause(scene_prompt)
        expect_eq("chat→parse: seed is 'she smiles'", cleaned_chat, "she smiles")
        expect_eq("chat→parse: explicit is ['Saki Nikaido']", explicit_chat, ["Saki Nikaido"])

        _db_module.get_character = lambda: {"name": "TestBot", "looks": "tall, silver hair"}
        _db_module.get_character_image_count = lambda: 0
        _db_module.get_image_entries = lambda: _KB_ENTRIES
        _db_module.get_entry_image_count = lambda entry_id: len(_KB_IMAGES.get(entry_id, []))
        _db_module.get_kb_image_full = _mock_get_kb_image_full
        try:
            refs_chat, subjects_chat, _, _decision_chat = scene_image._gather_refs(
                cleaned_chat, explicit_subjects=explicit_chat,
            )
            expect_true("chat→gather: Saki Nikaido in subjects", "Saki Nikaido" in subjects_chat)
        finally:
            _db_module.get_character = _db_module_orig_get_character
            _db_module.get_character_image_count = _db_module_orig_get_character_image_count
            _db_module.get_image_entries = _db_module_orig_get_image_entries
            _db_module.get_entry_image_count = _db_module_orig_get_entry_image_count
            _db_module.get_kb_image_full = _db_module_orig_get_kb_image_full

finally:
    lmstudio_ai._call_lmstudio = _orig_call_lmstudio


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

print()
print("=" * 60)
total = PASS_COUNT + FAIL_COUNT
print(f"  Results: {PASS_COUNT}/{total} passed")
if FAIL_COUNT == 0:
    print("  PASS — with: clause pipeline is working correctly.")
else:
    print(f"  FAIL — {FAIL_COUNT} test(s) failed.")
print("=" * 60)

sys.exit(0 if FAIL_COUNT == 0 else 1)
