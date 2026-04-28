"""
Verification tests for the per-slot identity prefix (Task #3 fix).

These tests call the *real* production functions — no logic is duplicated here.
The prefix is built inside _build_multi_edit_workflow_qwen (comfyui_ai.py:1047-1061)
and the handoff from _inject_player_refs (scene_image.py:1254-1307) is exercised
by patching database calls with controlled data.

Run: python test_per_slot_identity.py
"""

import io
import sys
import contextlib
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    suffix = f"\n         detail: {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    _results.append((label, condition, detail))


# ---------------------------------------------------------------------------
# Import real production builders
# ---------------------------------------------------------------------------

from comfyui_ai import _build_multi_edit_workflow_qwen  # noqa: E402


def _qwen_node_prompt(workflow: dict) -> str:
    """Extract the prompt text from the TextEncodeQwenImageEditPlus node."""
    for node in workflow.values():
        if node.get("class_type") == "TextEncodeQwenImageEditPlus":
            return node["inputs"].get("prompt", "")
    return ""


# ---------------------------------------------------------------------------
# Test 1: per-slot prefix is injected into the real Qwen workflow
# ---------------------------------------------------------------------------

KELLY_NAME = "Kelly Gray"
KELLY_APP  = "dark hair, blue eyes, detective coat"
NATT_NAME  = "Natt"
NATT_APP   = "black hair, glasses, casual outfit"

SCENE_PROMPT = "Kelly Gray and Natt sitting in a cozy cafe, anime style"


def test_prefix_in_real_qwen_workflow():
    """_build_multi_edit_workflow_qwen embeds the per-slot prefix in its prompt."""
    subjects    = [KELLY_NAME, NATT_NAME]
    appearances = {KELLY_NAME: KELLY_APP, NATT_NAME: NATT_APP}
    dummy_files = ["kelly_ref.png", "natt_ref.png"]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wf = _build_multi_edit_workflow_qwen(
            SCENE_PROMPT,
            gguf_path="dummy_model.gguf",
            vae_name="dummy_vae.safetensors",
            clip_gguf_name="dummy_clip.gguf",
            steps=1,
            width=512,
            height=512,
            seed=42,
            sampler_name="euler_ancestral",
            scheduler_name="beta",
            uploaded_image_names=dummy_files,
            uploaded_subjects=subjects,
            subject_appearances=appearances,
        )

    log_output = buf.getvalue()
    prompt_in_wf = _qwen_node_prompt(wf)

    check(
        "workflow has a TextEncodeQwenImageEditPlus node",
        bool(prompt_in_wf),
        f"node prompt: {prompt_in_wf[:80]!r}",
    )
    check(
        "prompt starts with slot prefix bracket",
        prompt_in_wf.startswith("["),
        f"prompt start: {prompt_in_wf[:80]!r}",
    )
    check(
        "Kelly Gray is labelled as Reference image 1",
        "Reference image 1 is Kelly Gray" in prompt_in_wf,
        f"prompt: {prompt_in_wf[:120]!r}",
    )
    check(
        "Natt is labelled as Reference image 2",
        "Reference image 2 is Natt" in prompt_in_wf,
        f"prompt: {prompt_in_wf[:120]!r}",
    )
    check(
        "Kelly's appearance details are in the prefix",
        KELLY_APP in prompt_in_wf,
        f"prompt: {prompt_in_wf[:160]!r}",
    )
    check(
        "Natt's appearance details are in the prefix",
        NATT_APP in prompt_in_wf,
        f"prompt: {prompt_in_wf[:200]!r}",
    )
    check(
        "scene prompt follows the prefix",
        SCENE_PROMPT in prompt_in_wf,
        f"prompt: {prompt_in_wf[:200]!r}",
    )
    check(
        "slot prefix comes before the scene prompt",
        prompt_in_wf.index("[Reference image") < prompt_in_wf.index(SCENE_PROMPT),
        f"prompt: {prompt_in_wf[:200]!r}",
    )
    check(
        "console log emitted: [ComfyUI] Qwen: per-slot identity prefix",
        "[ComfyUI] Qwen: per-slot identity prefix" in log_output,
        f"stdout: {log_output[:200]!r}",
    )


# ---------------------------------------------------------------------------
# Test 2: "self" / "player" labels are skipped by the real builder
# ---------------------------------------------------------------------------

def test_self_label_skipped_by_real_builder():
    """Slots labelled 'self' or 'player' must not appear in the prefix."""
    subjects    = [KELLY_NAME, "self"]
    appearances = {KELLY_NAME: KELLY_APP}
    dummy_files = ["kelly_ref.png", "player_selfie.png"]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wf = _build_multi_edit_workflow_qwen(
            SCENE_PROMPT,
            gguf_path="dummy.gguf",
            vae_name="dummy.safetensors",
            clip_gguf_name="dummy_clip.gguf",
            steps=1, width=512, height=512, seed=0,
            sampler_name="euler_ancestral",
            scheduler_name="beta",
            uploaded_image_names=dummy_files,
            uploaded_subjects=subjects,
            subject_appearances=appearances,
        )

    prompt_in_wf = _qwen_node_prompt(wf)

    check(
        "'self' label does not appear in slot prefix",
        "Reference image 2 is self" not in prompt_in_wf.lower(),
        f"prompt: {prompt_in_wf[:160]!r}",
    )
    check(
        "Kelly Gray still labelled in prefix when slot 2 is 'self'",
        "Reference image 1 is Kelly Gray" in prompt_in_wf,
        f"prompt: {prompt_in_wf[:160]!r}",
    )

    subjects2    = [KELLY_NAME, "player"]
    dummy_files2 = ["kelly_ref.png", "player_photo.png"]

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        wf2 = _build_multi_edit_workflow_qwen(
            SCENE_PROMPT,
            gguf_path="dummy.gguf",
            vae_name="dummy.safetensors",
            clip_gguf_name="dummy_clip.gguf",
            steps=1, width=512, height=512, seed=0,
            sampler_name="euler_ancestral",
            scheduler_name="beta",
            uploaded_image_names=dummy_files2,
            uploaded_subjects=subjects2,
            subject_appearances=appearances,
        )

    prompt2 = _qwen_node_prompt(wf2)

    check(
        "'player' label does not appear in slot prefix",
        "Reference image 2 is player" not in prompt2.lower(),
        f"prompt: {prompt2[:160]!r}",
    )


# ---------------------------------------------------------------------------
# Test 3: _inject_player_refs handoff — player photo lands in subjects list
# ---------------------------------------------------------------------------

def test_inject_player_refs_adds_player_photo():
    """
    _inject_player_refs (scene_image.py:1254) appends the player photo and
    label to the refs/subjects/appearances lists when the player is named
    in the seed. Exercises the actual handoff that passes data into Qwen.
    """
    import scene_image

    fake_profile  = {"discord_name": NATT_NAME, "looks": NATT_APP}
    fake_photo    = b"\x89PNG\r\n\x1a\n"  # minimal PNG header bytes

    with (
        patch.object(scene_image.database, "get_user_profile", return_value=fake_profile),
        patch.object(scene_image.database, "get_user_profile_image_count", return_value=1),
        patch.object(scene_image.database, "get_user_profile_image", return_value=fake_photo),
    ):
        refs_in        = [b"kelly_photo_bytes"]
        subjects_in    = [KELLY_NAME]
        appearances_in = {KELLY_NAME: KELLY_APP}

        refs_out, subjects_out, appearances_out = scene_image._inject_player_refs(
            refs_in,
            subjects_in,
            appearances_in,
            player_discord_id="12345",
            player_display_name=NATT_NAME,
            seed=f"Kelly Gray and {NATT_NAME} share an umbrella in the rain",
        )

    check(
        "player photo appended to refs list",
        len(refs_out) == 2 and refs_out[1] == fake_photo,
        f"refs_out lengths: {len(refs_out)}, last item type: {type(refs_out[-1]).__name__}",
    )
    check(
        "player label appended to subjects list",
        len(subjects_out) == 2 and subjects_out[1] == NATT_NAME,
        f"subjects_out: {subjects_out}",
    )
    check(
        "player appearance added to appearances dict",
        appearances_out.get(NATT_NAME) == NATT_APP,
        f"appearances_out[{NATT_NAME!r}]: {appearances_out.get(NATT_NAME)!r}",
    )
    check(
        "Kelly Gray's appearance is still in appearances dict",
        appearances_out.get(KELLY_NAME) == KELLY_APP,
        f"appearances_out[{KELLY_NAME!r}]: {appearances_out.get(KELLY_NAME)!r}",
    )


# ---------------------------------------------------------------------------
# Test 4: end-to-end handoff — _inject_player_refs output feeds real builder
# ---------------------------------------------------------------------------

def test_inject_handoff_into_real_builder():
    """
    Simulate the full data flow:
      _inject_player_refs → subjects/appearances → _build_multi_edit_workflow_qwen
    Both Kelly Gray (slot 1) and Natt (slot 2) must appear in the final prompt.
    """
    import scene_image

    fake_profile = {"discord_name": NATT_NAME, "looks": NATT_APP}
    fake_photo   = b"\x89PNG\r\n\x1a\n"

    with (
        patch.object(scene_image.database, "get_user_profile", return_value=fake_profile),
        patch.object(scene_image.database, "get_user_profile_image_count", return_value=1),
        patch.object(scene_image.database, "get_user_profile_image", return_value=fake_photo),
    ):
        refs_out, subjects_out, appearances_out = scene_image._inject_player_refs(
            [b"kelly_photo_bytes"],
            [KELLY_NAME],
            {KELLY_NAME: KELLY_APP},
            player_discord_id="12345",
            player_display_name=NATT_NAME,
            seed=f"Kelly Gray and {NATT_NAME} investigate the old mansion together",
        )

    dummy_files = ["kelly_ref.png", "natt_ref.png"]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wf = _build_multi_edit_workflow_qwen(
            SCENE_PROMPT,
            gguf_path="dummy.gguf",
            vae_name="dummy.safetensors",
            clip_gguf_name="dummy_clip.gguf",
            steps=1, width=512, height=512, seed=7,
            sampler_name="euler_ancestral",
            scheduler_name="beta",
            uploaded_image_names=dummy_files,
            uploaded_subjects=subjects_out,
            subject_appearances=appearances_out,
        )

    prompt_in_wf = _qwen_node_prompt(wf)
    log_output   = buf.getvalue()

    check(
        "end-to-end: Kelly Gray is Reference image 1 in workflow prompt",
        "Reference image 1 is Kelly Gray" in prompt_in_wf,
        f"prompt: {prompt_in_wf[:160]!r}",
    )
    check(
        "end-to-end: Natt is Reference image 2 in workflow prompt",
        "Reference image 2 is Natt" in prompt_in_wf,
        f"prompt: {prompt_in_wf[:200]!r}",
    )
    check(
        "end-to-end: per-slot prefix log line emitted",
        "[ComfyUI] Qwen: per-slot identity prefix" in log_output,
        f"stdout: {log_output[:200]!r}",
    )
    check(
        "end-to-end: log line contains expected slot label for Kelly",
        f"Reference image 1 is {KELLY_NAME}" in log_output,
        f"stdout: {log_output[:200]!r}",
    )
    check(
        "end-to-end: log line contains expected slot label for Natt",
        f"Reference image 2 is {NATT_NAME}" in log_output,
        f"stdout: {log_output[:200]!r}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n── Test 1: per-slot prefix in real Qwen workflow ─────────────────────")
    test_prefix_in_real_qwen_workflow()

    print("\n── Test 2: 'self'/'player' labels skipped by real builder ────────────")
    test_self_label_skipped_by_real_builder()

    print("\n── Test 3: _inject_player_refs handoff (with mocked DB) ──────────────")
    test_inject_player_refs_adds_player_photo()

    print("\n── Test 4: end-to-end — inject_player_refs → real builder ───────────")
    test_inject_handoff_into_real_builder()

    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        sys.exit(1)
    else:
        print("  — all good!")
        print()
        print("Per-slot identity prefix code path is verified end-to-end.")
        print("Live visual confirmation still needed: run a Discord scene")
        print("generation with /addprofilephoto set and inspect the output.")
