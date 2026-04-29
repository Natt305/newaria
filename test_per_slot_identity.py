"""
Verification tests for per-slot identity prefix + dual-encoder architecture (Task #8 fix).

These tests call the *real* production functions — no logic is duplicated here.
The prefix and workflow graph are built inside _build_multi_edit_workflow_qwen
(comfyui_ai.py); _inject_player_refs (scene_image.py) handoff is exercised by
patching database calls with controlled data.

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


def _positive_node_prompt(workflow: dict) -> str:
    """Return the prompt from node '10' (positive TextEncodeQwenImageEditPlus)."""
    node = workflow.get("10", {})
    if node.get("class_type") == "TextEncodeQwenImageEditPlus":
        return node["inputs"].get("prompt", "")
    return ""


def _negative_node_prompt(workflow: dict) -> str:
    """Return the prompt from node '11' (negative TextEncodeQwenImageEditPlus)."""
    node = workflow.get("11", {})
    if node.get("class_type") == "TextEncodeQwenImageEditPlus":
        return node["inputs"].get("prompt", "")
    return ""


def _ksampler_node(workflow: dict) -> dict:
    """Return the KSampler node inputs dict."""
    return workflow.get("7", {}).get("inputs", {})


# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

KELLY_NAME = "Kelly Gray"
KELLY_APP  = "dark hair, blue eyes, detective coat"
NATT_NAME  = "Natt"
NATT_APP   = "black hair, glasses, casual outfit"

SCENE_PROMPT = "Kelly Gray and Natt sitting in a cozy cafe, anime style"

_BASE_KWARGS = dict(
    gguf_path="dummy_model.gguf",
    vae_name="dummy_vae.safetensors",
    clip_gguf_name="dummy_clip.gguf",
    steps=1,
    width=512,
    height=512,
    seed=42,
    sampler_name="lcm",
    scheduler_name="sgm_uniform",
)


# ---------------------------------------------------------------------------
# Test 1: per-slot prefix format in positive encoder prompt
# ---------------------------------------------------------------------------

def test_prefix_in_real_qwen_workflow():
    """_build_multi_edit_workflow_qwen embeds the name-only per-slot prefix
    in the positive encoder prompt using 'image N is [Name].' syntax."""
    subjects    = [KELLY_NAME, NATT_NAME]
    appearances = {KELLY_NAME: KELLY_APP, NATT_NAME: NATT_APP}
    dummy_files = ["kelly_ref.png", "natt_ref.png"]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wf = _build_multi_edit_workflow_qwen(
            SCENE_PROMPT,
            **_BASE_KWARGS,
            uploaded_image_names=dummy_files,
            uploaded_subjects=subjects,
            subject_appearances=appearances,
        )

    log_output = buf.getvalue()
    prompt_in_wf = _positive_node_prompt(wf)

    check(
        "positive node (10) is a TextEncodeQwenImageEditPlus",
        wf.get("10", {}).get("class_type") == "TextEncodeQwenImageEditPlus",
        f"node 10 class_type: {wf.get('10', {}).get('class_type')!r}",
    )
    check(
        "positive prompt is non-empty",
        bool(prompt_in_wf),
        f"node prompt: {prompt_in_wf[:80]!r}",
    )
    check(
        "prefix uses 'image 1 is Kelly Gray' format (not 'Reference image')",
        "image 1 is Kelly Gray" in prompt_in_wf,
        f"prompt start: {prompt_in_wf[:120]!r}",
    )
    check(
        "prefix uses 'image 2 is Natt' format (not 'Reference image')",
        "image 2 is Natt" in prompt_in_wf,
        f"prompt: {prompt_in_wf[:120]!r}",
    )
    check(
        "prefix does NOT contain verbose appearance text for Kelly",
        KELLY_APP not in prompt_in_wf[:100],
        f"prefix region: {prompt_in_wf[:100]!r}",
    )
    check(
        "prefix does NOT contain verbose appearance text for Natt",
        NATT_APP not in prompt_in_wf[:100],
        f"prefix region: {prompt_in_wf[:100]!r}",
    )
    check(
        "prefix does NOT use old bracket '[Reference image …]' format",
        "[Reference image" not in prompt_in_wf,
        f"prompt: {prompt_in_wf[:160]!r}",
    )
    check(
        "scene prompt is present in the positive prompt",
        SCENE_PROMPT in prompt_in_wf,
        f"prompt: {prompt_in_wf[:200]!r}",
    )
    check(
        "slot prefix comes before the scene prompt",
        prompt_in_wf.index("image 1 is") < prompt_in_wf.index(SCENE_PROMPT),
        f"prompt: {prompt_in_wf[:200]!r}",
    )
    check(
        "console log emitted: [ComfyUI] Qwen: per-slot identity prefix",
        "[ComfyUI] Qwen: per-slot identity prefix" in log_output,
        f"stdout: {log_output[:200]!r}",
    )


# ---------------------------------------------------------------------------
# Test 2: dual-encoder architecture — node "11" replaces CLIPTextEncode "5"
# ---------------------------------------------------------------------------

def test_dual_encoder_architecture():
    """Workflow must use two TextEncodeQwenImageEditPlus nodes (positive + negative).
    The old CLIPTextEncode node '5' must be absent.
    KSampler must reference node '11' for its negative input."""
    subjects    = [KELLY_NAME, NATT_NAME]
    appearances = {KELLY_NAME: KELLY_APP, NATT_NAME: NATT_APP}
    dummy_files = ["kelly_ref.png", "natt_ref.png"]

    wf = _build_multi_edit_workflow_qwen(
        SCENE_PROMPT,
        **_BASE_KWARGS,
        uploaded_image_names=dummy_files,
        uploaded_subjects=subjects,
        subject_appearances=appearances,
    )

    check(
        "node '5' (CLIPTextEncode) is NOT present in workflow",
        "5" not in wf,
        f"node 5 class_type: {wf.get('5', {}).get('class_type', 'absent')!r}",
    )
    check(
        "node '11' (negative TextEncodeQwenImageEditPlus) IS present",
        wf.get("11", {}).get("class_type") == "TextEncodeQwenImageEditPlus",
        f"node 11 class_type: {wf.get('11', {}).get('class_type', 'absent')!r}",
    )
    check(
        "KSampler positive input references node '10'",
        _ksampler_node(wf).get("positive") == ["10", 0],
        f"positive: {_ksampler_node(wf).get('positive')!r}",
    )
    check(
        "KSampler negative input references node '11'",
        _ksampler_node(wf).get("negative") == ["11", 0],
        f"negative: {_ksampler_node(wf).get('negative')!r}",
    )


# ---------------------------------------------------------------------------
# Test 3: negative encoder receives same image slots as positive encoder
# ---------------------------------------------------------------------------

def test_negative_encoder_images_match_positive():
    """Both encoders must wire the same reference images into the same slots."""
    dummy_files = ["kelly_ref.png", "natt_ref.png"]

    wf = _build_multi_edit_workflow_qwen(
        SCENE_PROMPT,
        **_BASE_KWARGS,
        uploaded_image_names=dummy_files,
        uploaded_subjects=[KELLY_NAME, NATT_NAME],
        subject_appearances={},
    )

    pos_inputs = wf.get("10", {}).get("inputs", {})
    neg_inputs = wf.get("11", {}).get("inputs", {})

    check(
        "positive encoder has image1 slot",
        "image1" in pos_inputs,
        f"pos inputs keys: {list(pos_inputs.keys())}",
    )
    check(
        "negative encoder has image1 slot",
        "image1" in neg_inputs,
        f"neg inputs keys: {list(neg_inputs.keys())}",
    )
    check(
        "image1 slot matches in both encoders",
        pos_inputs.get("image1") == neg_inputs.get("image1"),
        f"pos={pos_inputs.get('image1')!r}, neg={neg_inputs.get('image1')!r}",
    )
    check(
        "image2 slot matches in both encoders",
        pos_inputs.get("image2") == neg_inputs.get("image2"),
        f"pos={pos_inputs.get('image2')!r}, neg={neg_inputs.get('image2')!r}",
    )
    check(
        "positive encoder wired to correct VAE (node '3')",
        pos_inputs.get("vae") == ["3", 0],
        f"pos vae: {pos_inputs.get('vae')!r}",
    )
    check(
        "negative encoder wired to correct VAE (node '3')",
        neg_inputs.get("vae") == ["3", 0],
        f"neg vae: {neg_inputs.get('vae')!r}",
    )


# ---------------------------------------------------------------------------
# Test 4: negative encoder node carries anatomy/feminine text when CFG > 1.0
#          and an empty prompt when CFG = 1.0
# ---------------------------------------------------------------------------

def test_negative_encoder_prompt_gated_by_cfg():
    """Negative encoder (node '11') must carry the anatomy/feminine-build negative
    text when QWEN_CFG > 1.0 (negative branch active), and an empty prompt when
    QWEN_CFG = 1.0 (negative branch mathematically zeroed)."""
    import os as _os

    shared_kwargs = dict(
        uploaded_image_names=["kelly.png", "natt.png"],
        uploaded_subjects=[KELLY_NAME, NATT_NAME],
        subject_appearances={},
    )

    # ── sub-test A: CFG > 1.0 → negative carries anatomy text ────────────
    _os.environ["QWEN_CFG"] = "1.5"
    try:
        wf_active = _build_multi_edit_workflow_qwen(
            SCENE_PROMPT, **_BASE_KWARGS, **shared_kwargs
        )
    finally:
        del _os.environ["QWEN_CFG"]

    neg_active = _negative_node_prompt(wf_active)

    check(
        "CFG=1.5: negative encoder (node '11') has a non-empty prompt",
        bool(neg_active),
        f"neg prompt: {neg_active[:200]!r}",
    )
    check(
        "CFG=1.5: negative prompt contains anatomy correction text",
        any(kw in neg_active.lower() for kw in ("anatomy", "limb", "hand", "finger", "deform")),
        f"neg prompt: {neg_active[:200]!r}",
    )

    # ── sub-test B: CFG = 1.0 → negative is empty ─────────────────────────
    _os.environ["QWEN_CFG"] = "1.0"
    try:
        wf_zero = _build_multi_edit_workflow_qwen(
            SCENE_PROMPT, **_BASE_KWARGS, **shared_kwargs
        )
    finally:
        del _os.environ["QWEN_CFG"]

    neg_zero = _negative_node_prompt(wf_zero)

    check(
        "CFG=1.0: negative encoder (node '11') has an empty prompt",
        neg_zero == "",
        f"neg prompt: {neg_zero[:200]!r}",
    )
    check(
        "positive encoder (node '10') has a non-empty prompt",
        bool(_positive_node_prompt(wf_active)),
        f"pos prompt: {_positive_node_prompt(wf_active)[:80]!r}",
    )


# ---------------------------------------------------------------------------
# Test 5: "self" / "player" labels are skipped by the real builder
# ---------------------------------------------------------------------------

def test_self_label_skipped_by_real_builder():
    """Slots labelled 'self' or 'player' must not appear in the prefix."""
    subjects    = [KELLY_NAME, "self"]
    appearances = {KELLY_NAME: KELLY_APP}
    dummy_files = ["kelly_ref.png", "player_selfie.png"]

    wf = _build_multi_edit_workflow_qwen(
        SCENE_PROMPT,
        **_BASE_KWARGS,
        uploaded_image_names=dummy_files,
        uploaded_subjects=subjects,
        subject_appearances=appearances,
    )

    prompt_in_wf = _positive_node_prompt(wf)

    check(
        "'self' label does not appear in slot prefix",
        "image 2 is self" not in prompt_in_wf.lower(),
        f"prompt: {prompt_in_wf[:160]!r}",
    )
    check(
        "Kelly Gray still labelled in prefix when slot 2 is 'self'",
        "image 1 is Kelly Gray" in prompt_in_wf,
        f"prompt: {prompt_in_wf[:160]!r}",
    )

    subjects2    = [KELLY_NAME, "player"]
    dummy_files2 = ["kelly_ref.png", "player_photo.png"]

    wf2 = _build_multi_edit_workflow_qwen(
        SCENE_PROMPT,
        **_BASE_KWARGS,
        uploaded_image_names=dummy_files2,
        uploaded_subjects=subjects2,
        subject_appearances=appearances,
    )

    prompt2 = _positive_node_prompt(wf2)

    check(
        "'player' label does not appear in slot prefix",
        "image 2 is player" not in prompt2.lower(),
        f"prompt: {prompt2[:160]!r}",
    )


# ---------------------------------------------------------------------------
# Test 6: _inject_player_refs handoff — player photo lands in subjects list
# ---------------------------------------------------------------------------

def test_inject_player_refs_adds_player_photo():
    """
    _inject_player_refs (scene_image.py) appends the player photo and
    label to the refs/subjects/appearances lists when the player is named
    in the seed. Exercises the actual handoff that passes data into Qwen.
    """
    import scene_image

    fake_profile  = {"discord_name": NATT_NAME, "looks": NATT_APP}
    fake_photo    = b"\x89PNG\r\n\x1a\n"

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
# Test 7: end-to-end handoff — _inject_player_refs output feeds real builder
# ---------------------------------------------------------------------------

def test_inject_handoff_into_real_builder():
    """
    Simulate the full data flow:
      _inject_player_refs → subjects/appearances → _build_multi_edit_workflow_qwen
    Both Kelly Gray (slot 1) and Natt (slot 2) must appear in the final prompt
    with the new 'image N is [Name].' format.
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
            **_BASE_KWARGS,
            uploaded_image_names=dummy_files,
            uploaded_subjects=subjects_out,
            subject_appearances=appearances_out,
        )

    prompt_in_wf = _positive_node_prompt(wf)
    log_output   = buf.getvalue()

    check(
        "end-to-end: 'image 1 is Kelly Gray' in workflow prompt",
        "image 1 is Kelly Gray" in prompt_in_wf,
        f"prompt: {prompt_in_wf[:160]!r}",
    )
    check(
        "end-to-end: 'image 2 is Natt' in workflow prompt",
        "image 2 is Natt" in prompt_in_wf,
        f"prompt: {prompt_in_wf[:200]!r}",
    )
    check(
        "end-to-end: per-slot prefix log line emitted",
        "[ComfyUI] Qwen: per-slot identity prefix" in log_output,
        f"stdout: {log_output[:200]!r}",
    )
    check(
        "end-to-end: log line contains new-format label for Kelly",
        f"image 1 is {KELLY_NAME}" in log_output,
        f"stdout: {log_output[:200]!r}",
    )
    check(
        "end-to-end: log line contains new-format label for Natt",
        f"image 2 is {NATT_NAME}" in log_output,
        f"stdout: {log_output[:200]!r}",
    )
    check(
        "end-to-end: node '11' (negative encoder) is present",
        wf.get("11", {}).get("class_type") == "TextEncodeQwenImageEditPlus",
        f"node 11: {wf.get('11', {}).get('class_type', 'absent')!r}",
    )
    check(
        "end-to-end: CLIPTextEncode node '5' is absent",
        "5" not in wf,
        f"node 5 present: {'5' in wf}",
    )


# ---------------------------------------------------------------------------
# Test 8: player NOT in seed → refs/subjects unchanged (no bleed-in)
# ---------------------------------------------------------------------------

def test_no_injection_when_player_not_in_seed():
    """When the player's name does not appear in the scene seed and no
    second-person pronouns are present, _inject_player_refs must leave
    refs, subjects, and appearances completely unchanged — no style bleed."""
    import scene_image

    fake_profile = {"discord_name": NATT_NAME, "looks": NATT_APP}
    fake_photo   = b"\x89PNG\r\n\x1a\n"

    seed_without_player = "Kelly Gray walks alone through a rain-soaked alley"

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
            seed=seed_without_player,
        )

    check(
        "no player photo injected when player absent from seed",
        len(refs_out) == 1,
        f"refs_out length: {len(refs_out)}",
    )
    check(
        "subjects unchanged when player absent from seed",
        subjects_out == [KELLY_NAME],
        f"subjects_out: {subjects_out}",
    )
    check(
        "appearances unchanged when player absent from seed",
        list(appearances_out.keys()) == [KELLY_NAME],
        f"appearances keys: {list(appearances_out.keys())}",
    )
    check(
        "Kelly Gray is still sole subject (no style bleed)",
        subjects_out[0] == KELLY_NAME,
        f"subjects_out: {subjects_out}",
    )


# ---------------------------------------------------------------------------
# Test 9: player in seed but zero photos → no ref injection, looks only
# ---------------------------------------------------------------------------

def test_no_photo_injection_when_photo_count_zero():
    """When a player IS named in the seed but has uploaded zero photos,
    no image should be appended to refs — only the looks text may be
    added to appearances so the model renders them neutrally."""
    import scene_image

    fake_profile = {"discord_name": NATT_NAME, "looks": NATT_APP}

    seed_with_player = f"Kelly Gray and {NATT_NAME} investigate the old mansion"

    with (
        patch.object(scene_image.database, "get_user_profile", return_value=fake_profile),
        patch.object(scene_image.database, "get_user_profile_image_count", return_value=0),
        patch.object(scene_image.database, "get_user_profile_image", return_value=None),
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
            seed=seed_with_player,
        )

    check(
        "no extra ref appended when player has 0 photos",
        len(refs_out) == 1,
        f"refs_out length: {len(refs_out)}",
    )
    check(
        "subjects list unchanged when player has 0 photos",
        subjects_out == [KELLY_NAME],
        f"subjects_out: {subjects_out}",
    )
    check(
        "player looks text injected into appearances (text-only fallback)",
        appearances_out.get(NATT_NAME) == NATT_APP,
        f"appearances_out[{NATT_NAME!r}]: {appearances_out.get(NATT_NAME)!r}",
    )
    check(
        "Kelly Gray appearance preserved alongside player looks",
        appearances_out.get(KELLY_NAME) == KELLY_APP,
        f"appearances_out[{KELLY_NAME!r}]: {appearances_out.get(KELLY_NAME)!r}",
    )


# ---------------------------------------------------------------------------
# Test 10: no discord_id → function is a pure pass-through
# ---------------------------------------------------------------------------

def test_no_discord_id_is_passthrough():
    """When player_discord_id is None or empty, _inject_player_refs must
    return the exact same objects unchanged — sentinel for anonymous users."""
    import scene_image

    refs_in        = [b"kelly_photo_bytes"]
    subjects_in    = [KELLY_NAME]
    appearances_in = {KELLY_NAME: KELLY_APP}

    refs_out, subjects_out, appearances_out = scene_image._inject_player_refs(
        refs_in,
        subjects_in,
        appearances_in,
        player_discord_id=None,
        player_display_name=NATT_NAME,
        seed=f"Kelly Gray and {NATT_NAME} share an umbrella",
    )

    check(
        "refs list unchanged when no discord_id",
        refs_out is refs_in,
        f"refs identity: same={refs_out is refs_in}",
    )
    check(
        "subjects list unchanged when no discord_id",
        subjects_out is subjects_in,
        f"subjects identity: same={subjects_out is subjects_in}",
    )
    check(
        "appearances dict unchanged when no discord_id",
        appearances_out is appearances_in,
        f"appearances identity: same={appearances_out is appearances_in}",
    )


# ---------------------------------------------------------------------------
# Test 11: multi-ref uses per-slot appearance lock (not generic policy text)
# ---------------------------------------------------------------------------

def test_multi_ref_per_slot_appearance_lock():
    """For ≥2 refs with named slots the generic style-policy text is replaced
    by a character-led 6-part lock:
      1. Style directive  — image 1 (Kelly Gray) is the art-style authority,
         expressed with rich detail vocabulary (gloss, highlight placement,
         pupil detail, skin shading, hair rendering, eye design, line-art
         weight, colour palette) so the model gets the same fine-grained
         style anchor that the single-ref `match_reference` policy provides.
      2. Main-subject directive — Kelly is always the primary subject
      3. Char-0 full lock       — Kelly kept exactly as in image 1
      4. Char-1+ likeness       — Natt provides face/hair/eye from image 2,
         art style from image 1, bidirectional accessory ban
      5. Absurd-prop omission   — drop props when scene context makes them
         nonsensical (e.g. weapon in a shower scene)
      6. Mirror logic           — accurate, logical mirror reflections;
         hands don't pass through mirrors
    The generic 'Replicate the visual style of the reference images exactly'
    must NOT appear — it would route the model to blend across all refs."""
    wf = _build_multi_edit_workflow_qwen(
        SCENE_PROMPT,
        **_BASE_KWARGS,
        uploaded_image_names=["kelly_ref.png", "natt_ref.png"],
        uploaded_subjects=[KELLY_NAME, NATT_NAME],
        subject_appearances={KELLY_NAME: KELLY_APP, NATT_NAME: NATT_APP},
    )
    p = _positive_node_prompt(wf)

    check(
        "multi-ref: style-from-image-1 directive present for Kelly",
        f"from image 1 ({KELLY_NAME}) for the entire scene" in p,
        f"prompt: {p[:400]!r}",
    )
    check(
        "multi-ref: rich style vocabulary present (gloss / highlight / pupil / skin shading)",
        all(tok in p for tok in (
            "gloss", "highlight placement", "pupil detail", "skin shading",
            "hair rendering", "eye design",
        )),
        f"prompt: {p[:600]!r}",
    )
    check(
        "multi-ref: 'Do NOT introduce photorealism' guard present",
        "Do NOT introduce photorealism" in p,
        f"prompt: {p[:600]!r}",
    )
    check(
        "multi-ref: main-subject directive present for Kelly",
        f"{KELLY_NAME} will always be the main subject" in p,
        f"prompt: {p[:600]!r}",
    )
    check(
        "multi-ref: Kelly kept EXACTLY as in image 1 (CAPS lock, beats scene description)",
        f"Keep {KELLY_NAME} EXACTLY as shown in image 1" in p,
        f"prompt: {p[:400]!r}",
    )
    check(
        "multi-ref: explicit eye/hair/outfit override for Kelly present",
        f"Do NOT change {KELLY_NAME}'s eye colour, hair colour, or outfit to match the scene description" in p,
        f"prompt: {p[:400]!r}",
    )
    check(
        "multi-ref: Natt rendered with ONLY face/hair/eye from image 2 (no accessory bleed)",
        f"Render {NATT_NAME} with ONLY the face shape, hair colour, and eye colour from image 2" in p,
        f"prompt: {p[:400]!r}",
    )
    check(
        "multi-ref: bidirectional accessory ban present ('and vice versa')",
        "and vice versa" in p,
        f"prompt: {p[:600]!r}",
    )
    check(
        "multi-ref: Natt's hat/clothing explicitly excluded from Kelly Gray",
        f"Do NOT copy {NATT_NAME}'s hat, clothing, or accessories onto {KELLY_NAME}" in p,
        f"prompt: {p[:400]!r}",
    )
    check(
        "multi-ref: generic 'Replicate the visual style' NOT in prompt",
        "Replicate the visual style" not in p,
        f"prompt: {p[:300]!r}",
    )
    check(
        "multi-ref: generic 'Do NOT change hair colour' NOT in prompt",
        "Do NOT change hair colour" not in p,
        f"prompt: {p[:300]!r}",
    )
    check(
        "multi-ref: absurd-prop omission clause present",
        "absurd" in p.lower() and "weapon in a shower" in p,
        f"prompt: {p[:600]!r}",
    )
    check(
        "multi-ref: mirror-logic clause IS in prompt",
        "mirror reflections are accurate" in p
        and "hands don't go through mirrors" in p,
        f"prompt: {p[:600]!r}",
    )


# ---------------------------------------------------------------------------
# Test 11b: hybrid negative — close-up suppressor removed, other guards stay
# ---------------------------------------------------------------------------

def test_negative_guards_all_present():
    """`_MIRROR_AND_QUALITY_NEGATIVE` must contain all required guard phrases:
    close-up suppressor, mirror artifacts, quality push, chat/thought bubble
    suppression, hands-through-mirror, wearables-on-wrong-characters.
    Clone suppression is in `_ANATOMY_NEGATIVE`; feminine-build stays in
    `_FEMININE_BUILD_NEGATIVE`."""
    from comfyui_ai import (
        _MIRROR_AND_QUALITY_NEGATIVE, _ANATOMY_NEGATIVE, _FEMININE_BUILD_NEGATIVE,
    )

    check(
        "negative: 'unwanted super close up shots' present in mirror/quality negative",
        "unwanted super close up shots" in _MIRROR_AND_QUALITY_NEGATIVE,
        f"_MIRROR_AND_QUALITY_NEGATIVE: {_MIRROR_AND_QUALITY_NEGATIVE!r}",
    )
    check(
        "negative: mirror artifact suppression preserved",
        all(tok in _MIRROR_AND_QUALITY_NEGATIVE for tok in (
            "ghost objects in mirror", "illogical mirror reflections",
            "hands that go through mirror", "wearables on wrong characters",
        )),
        f"_MIRROR_AND_QUALITY_NEGATIVE: {_MIRROR_AND_QUALITY_NEGATIVE!r}",
    )
    check(
        "negative: quality push and chat-bubble suppression preserved",
        all(tok in _MIRROR_AND_QUALITY_NEGATIVE for tok in (
            "poor or mediocre quality artstyle",
            "text chat bubbles", "thought bubbles",
        )),
        f"_MIRROR_AND_QUALITY_NEGATIVE: {_MIRROR_AND_QUALITY_NEGATIVE!r}",
    )
    check(
        "negative: clone suppression still in anatomy negative",
        all(tok in _ANATOMY_NEGATIVE for tok in (
            "duplicate characters", "twin characters", "cloned person",
            "multiple instances of same character", "mirror character clone",
        )),
        f"_ANATOMY_NEGATIVE: {_ANATOMY_NEGATIVE!r}",
    )
    check(
        "negative: feminine-build negative untouched",
        "masculine build" in _FEMININE_BUILD_NEGATIVE
        and "broad shoulders" in _FEMININE_BUILD_NEGATIVE,
        f"_FEMININE_BUILD_NEGATIVE: {_FEMININE_BUILD_NEGATIVE!r}",
    )


# ---------------------------------------------------------------------------
# Test 12: single-ref still uses the configured style-policy text (no change)
# ---------------------------------------------------------------------------

def test_single_ref_keeps_generic_appearance_lock():
    """For exactly 1 ref, the per-slot override must NOT fire — the generic
    style-policy text is correct for single-ref and must still be present."""
    wf = _build_multi_edit_workflow_qwen(
        SCENE_PROMPT,
        **_BASE_KWARGS,
        uploaded_image_names=["kelly_ref.png"],
        uploaded_subjects=[KELLY_NAME],
        subject_appearances={KELLY_NAME: KELLY_APP},
    )
    p = _positive_node_prompt(wf)

    check(
        "single-ref: generic appearance lock IS still in prompt",
        "Preserve the exact appearance" in p or "Do NOT change hair colour" in p
        or "Replicate the visual style" in p,
        f"prompt: {p[:300]!r}",
    )
    check(
        "single-ref: per-slot 'keep character in image' NOT injected",
        "keep character in image" not in p,
        f"prompt: {p[:300]!r}",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n── Test 1: per-slot prefix format (name-only, no appearance text) ────")
    test_prefix_in_real_qwen_workflow()

    print("\n── Test 2: dual-encoder architecture (node '11' replaces '5') ────────")
    test_dual_encoder_architecture()

    print("\n── Test 3: negative encoder receives same image slots as positive ─────")
    test_negative_encoder_images_match_positive()

    print("\n── Test 4: negative encoder CFG-gated prompt (anatomy text when CFG > 1.0) ──")
    test_negative_encoder_prompt_gated_by_cfg()

    print("\n── Test 5: 'self'/'player' labels skipped by real builder ────────────")
    test_self_label_skipped_by_real_builder()

    print("\n── Test 6: _inject_player_refs handoff (with mocked DB) ─────────────")
    test_inject_player_refs_adds_player_photo()

    print("\n── Test 7: end-to-end — inject_player_refs → real builder ───────────")
    test_inject_handoff_into_real_builder()

    print("\n── Test 8: player NOT in seed → refs/subjects unchanged (no bleed) ──")
    test_no_injection_when_player_not_in_seed()

    print("\n── Test 9: player in seed, 0 photos → looks text only, no ref ────────")
    test_no_photo_injection_when_photo_count_zero()

    print("\n── Test 10: no discord_id → pure pass-through ───────────────────────")
    test_no_discord_id_is_passthrough()

    print("\n── Test 11: multi-ref → character-led style, player likeness only ──────")
    test_multi_ref_per_slot_appearance_lock()

    print("\n── Test 11b: all negative guards present (close-up, mirror, quality, etc.) ───────")
    test_negative_guards_all_present()

    print("\n── Test 12: single-ref → generic policy lock preserved (no override) ─")
    test_single_ref_keeps_generic_appearance_lock()

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
        print("Qwen dual-encoder architecture and per-slot prefix verified end-to-end.")
        print("All no-photo / no-player paths confirmed: zero style bleed when")
        print("no profile photo is uploaded or the player is not in the scene seed.")
        print()
        print("Multi-ref character-led style verified:")
        print("  image 1 (bot character) drives art style + appearance for the whole scene.")
        print("  image 2+ (player/secondary) provide face/hair/eye likeness only,")
        print("  rendered in the bot character's art style.")
        print()
        print("Live visual confirmation still needed: run a Discord scene")
        print("generation with /addprofilephoto set and inspect the output.")
