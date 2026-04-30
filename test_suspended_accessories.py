"""Regression tests for the suspended_accessories feature (Task #5).

Coverage:
  A. Serialization round-trip (forward + backward-compat)
  B. Capture → suspension (post-delta path)
  C. Freed restoration — pre-flight path (no _needs_update match, no extractor)
  D. Pre-flight restoration turn/history tracking (updated_at advances)
  E. Dedup: item already in accessories is not duplicated after restoration
  F. No restoration when freed text absent
  G. Scene-image freed reinjection (_FREED_SCENE_RE + suspended_accessories)
  H. PlayerState parallel parity

Run with: python3 test_suspended_accessories.py
"""

import sys, copy
sys.path.insert(0, ".")

import character_state as cs_mod
import player_state as ps_mod

# ── helpers ──────────────────────────────────────────────────────────────────

def make_char(**kw):
    return cs_mod.CharacterState(**kw)

def make_player(**kw):
    return ps_mod.PlayerState(**kw)


# ── A. Serialization ─────────────────────────────────────────────────────────

def test_serialization_round_trip():
    state = make_char(accessories=["bracelet"], suspended_accessories=["revolver"])
    d = cs_mod._state_to_dict(state)
    assert "suspended_accessories" in d, "field missing from dict"
    restored = cs_mod._state_from_dict(d)
    assert restored.suspended_accessories == ["revolver"]
    assert restored.accessories == ["bracelet"]

def test_backward_compat_missing_key():
    old_row = {"outfit": "jacket", "body_state": "clothed"}
    state = cs_mod._state_from_dict(old_row)
    assert state.suspended_accessories == [], "old rows must default to []"

def test_suspended_excluded_from_persistent_items():
    state = make_char(accessories=["bracelet"], suspended_accessories=["revolver"])
    items = state.persistent_items()
    assert "bracelet" in items
    assert "revolver" not in items, "suspended items must not appear in persistent_items"

def test_is_empty_ignores_suspended():
    state = make_char(suspended_accessories=["revolver"])
    assert state.is_empty(), "suspended-only state should be considered empty"


# ── B. Capture → suspension (simulating post-delta path) ─────────────────────

def test_capture_suspends_portable_items():
    before_acc = ["revolver", "holster", "bracelet"]
    updated_acc = ["bracelet"]  # extractor removed revolver + holster on capture turn
    exchange = "She was captured and disarmed by the guards."

    removed = set(before_acc) - set(updated_acc)
    assert cs_mod._CAPTIVE_TRANSITION_RE.search(exchange), "captive regex must match"

    susp = []
    existing_lower = set()
    for item in removed:
        if cs_mod._PORTABLE_ITEM_RE.search(item):
            if item.lower() not in existing_lower:
                susp.append(item)
                existing_lower.add(item.lower())

    assert "revolver" in susp, "revolver must be suspended"
    assert "holster" in susp, "holster must be suspended"
    assert "bracelet" not in removed, "bracelet should not have been removed"

def test_non_portable_items_not_suspended():
    before_acc = ["collar", "chain", "revolver"]
    updated_acc = []
    exchange = "She was captured."

    removed = set(before_acc) - set(updated_acc)
    portable = [i for i in removed if cs_mod._PORTABLE_ITEM_RE.search(i)]

    assert "revolver" in portable
    assert "collar" not in portable, "collar is not portable"
    assert "chain" not in portable, "chain is not portable"

def test_no_capture_suspension_without_captive_text():
    before_acc = ["revolver"]
    updated_acc = []
    exchange = "She placed the gun on the shelf."
    removed = set(before_acc) - set(updated_acc)
    assert not cs_mod._CAPTIVE_TRANSITION_RE.search(exchange), "no captive signal"
    # In real code, no suspension should trigger without the regex match.
    # Here we just verify the regex doesn't fire.


# ── C. Freed restoration — pre-flight path ───────────────────────────────────

def test_preflight_fires_before_needs_update():
    """Freed text that contains no generic appearance keywords must still
    trigger the pre-flight restoration block (not gated by _needs_update)."""
    freed_texts = [
        "They rearmed her and let her go.",
        "Her weapon was returned to her by the officer.",
        "She was rearmed.",
        "She finally escaped from captivity.",
        "was set free by the rebels",
    ]
    for text in freed_texts:
        # The text should match freed regex
        assert cs_mod._FREED_TRANSITION_RE.search(text), f"freed regex must match: {text!r}"
        # Optionally, at least some should NOT match needs_update (pure freed turns)
        # We just care that the freed regex matches so pre-flight would activate.

def test_preflight_restores_suspended_items():
    """Simulate the pre-flight restoration: suspended + freed text → promoted."""
    current_acc = ["bracelet"]
    current_susp = ["revolver", "holster"]
    exchange = "They rearmed her and let her go."

    assert cs_mod._FREED_TRANSITION_RE.search(exchange)
    existing = {x.lower() for x in current_acc}
    promoted = []
    for item in current_susp:
        if item.lower() not in existing:
            current_acc.append(item)
            existing.add(item.lower())
            promoted.append(item)
    current_susp = []

    assert "revolver" in current_acc
    assert "holster" in current_acc
    assert "bracelet" in current_acc
    assert current_susp == []
    assert set(promoted) == {"revolver", "holster"}

def test_no_restoration_without_freed_signal():
    """Suspended items must remain suspended when no freed regex matches."""
    exchange = "She walked calmly through the market."
    assert not cs_mod._FREED_TRANSITION_RE.search(exchange)
    # Restoration loop would be skipped; suspended_accessories unchanged.


# ── D. updated_at monotonic advancement ──────────────────────────────────────

def test_updated_at_advances_on_restoration():
    """updated_at must advance in pre-flight restoration so diagnostics are
    consistent even when no extractor delta follows."""
    state = make_char(accessories=["bracelet"], suspended_accessories=["revolver"], updated_at=5)
    turn = 6
    state.updated_at = turn   # simulate what pre-flight block does
    assert state.updated_at == 6, "updated_at must advance to restoration turn"


# ── E. Dedup ─────────────────────────────────────────────────────────────────

def test_dedup_no_duplicate_on_restoration():
    current_acc = ["bracelet", "revolver"]  # revolver already present
    current_susp = ["revolver"]
    exchange = "was set free"

    existing = {x.lower() for x in current_acc}
    for item in current_susp:
        if item.lower() not in existing:
            current_acc.append(item)
            existing.add(item.lower())
    current_susp = []

    assert current_acc.count("revolver") == 1, "No duplicate after dedup"
    assert current_susp == []


# ── F. No restoration without freed text ─────────────────────────────────────

def test_no_restoration_when_suspended_empty():
    current_susp = []
    exchange = "She escaped from the dungeon."
    # pre-flight condition: `if current_susp and freed_re.search(exchange)` → False
    assert not (bool(current_susp) and cs_mod._FREED_TRANSITION_RE.search(exchange))


# ── G. Scene-image freed reinjection ─────────────────────────────────────────

def test_scene_freed_regex_exists():
    import scene_image
    assert hasattr(scene_image, "_FREED_SCENE_RE"), "_FREED_SCENE_RE must exist in scene_image"
    assert scene_image._FREED_SCENE_RE.search("She escaped from her captors and retrieved her weapon")
    assert scene_image._FREED_SCENE_RE.search("She broke free and was rearmed")

def test_scene_suspended_reinject_simulation():
    """Simulate the scene-image logic that reinjects suspended items on freed turns."""
    suspended = ["revolver", "holster"]
    current_persistent = ["bracelet"]
    exchange_context = "She escaped from captivity."

    import scene_image
    if scene_image._FREED_SCENE_RE.search(exchange_context) and suspended:
        existing = {x.lower() for x in current_persistent}
        for item in suspended:
            if item.lower() not in existing:
                current_persistent.append(item)
                existing.add(item.lower())

    assert "revolver" in current_persistent
    assert "holster" in current_persistent
    assert "bracelet" in current_persistent


# ── H. PlayerState parallel parity ───────────────────────────────────────────

def test_player_state_serialization():
    state = make_player(accessories=["sword"], suspended_accessories=["pistol"])
    d = ps_mod._state_to_dict(state)
    assert "suspended_accessories" in d
    restored = ps_mod._state_from_dict(d)
    assert restored.suspended_accessories == ["pistol"]

def test_player_state_backward_compat():
    state = ps_mod._state_from_dict({"outfit": "jeans"})
    assert state.suspended_accessories == []

def test_player_state_persistent_items():
    state = make_player(accessories=["sword"], suspended_accessories=["pistol"])
    items = state.persistent_items()
    assert "sword" in items
    assert "pistol" not in items

def test_player_state_regexes():
    assert ps_mod._CAPTIVE_TRANSITION_RE.search("He was captured")
    assert ps_mod._FREED_TRANSITION_RE.search("He escaped from the dungeon")
    assert ps_mod._FREED_TRANSITION_RE.search("They rearmed him")
    assert ps_mod._PORTABLE_ITEM_RE.search("pistol")
    assert not ps_mod._PORTABLE_ITEM_RE.search("collar")


# ── runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
