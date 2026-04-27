"""
AI backend router — delegates all calls to either groq_ai or ollama_ai
based on the AI_BACKEND environment variable (default: groq).

Usage in other modules:
    import ai_backend as groq_ai   # drop-in replacement
"""
import os
from typing import Optional


def _backend():
    return os.environ.get("AI_BACKEND", "groq").strip().lower()


def _mod():
    if _backend() == "ollama":
        import ollama_ai
        return ollama_ai
    elif _backend() == "lmstudio":
        import lmstudio_ai
        return lmstudio_ai
    import groq_ai
    return groq_ai


# ── Pure helpers (no IO) — delegate straight through ──────────────────────────

def is_self_referential_image(prompt: str) -> bool:
    return _mod().is_self_referential_image(prompt)


def is_recall_request(text: str) -> bool:
    return _mod().is_recall_request(text)


def response_declines_image(text: str) -> bool:
    return _mod().response_declines_image(text)


def user_wants_image(messages: list) -> Optional[str]:
    return _mod().user_wants_image(messages)


# ── Async functions ────────────────────────────────────────────────────────────

async def chat(
    messages: list,
    system_prompt: str = "",
    model: Optional[str] = None,
    context_images: Optional[list] = None,
    character_name: str = "",
) -> tuple[str, Optional[str], bool, bool, bool, Optional[str]]:
    """Returns (response_text, image_prompt_or_None, prompt_from_marker,
    success, wants_scene_image, scene_prompt_or_None).

    Only the LM Studio backend currently emits the [SCENE] / [SCENE: ...]
    cinematic signal. For Groq / Ollama the 5th element is always padded to
    False and the 6th to None so the call site (`process_chat`) can unpack
    uniformly.

    scene_prompt carries the optional body the model wrote inside
    `[SCENE: short cinematic description]` — when present, the caller uses
    it verbatim as the image-prompt seed instead of deriving from the bot's
    reply prose. None means the caller falls back to the prose-derived path
    (also the case for bare `[SCENE]`).
    """
    kwargs = {"system_prompt": system_prompt}
    if model:
        kwargs["model"] = model
    if context_images:
        kwargs["context_images"] = context_images
    # Backend-specific: only the LM Studio path uses character_name (for
    # the plain-prose self-name strip + roleplay format directive). Forward
    # it only when set AND when the active backend accepts it, so the
    # Groq / Ollama paths see no signature change.
    if character_name and _backend() == "lmstudio":
        kwargs["character_name"] = character_name
    result = await _mod().chat(messages, **kwargs)
    # Normalise to 6-tuple so callers don't have to know which backend
    # is active. LM Studio already returns 6; Groq / Ollama still return 4
    # — pad the cinematic-signal flag with False and the scene-prompt body
    # with None.
    if len(result) == 4:
        return (*result, False, None)
    if len(result) == 5:
        return (*result, None)
    return result


async def understand_image(
    image_bytes: bytes,
    mime_type: str,
    question: str = "Describe this image in detail.",
) -> Optional[str]:
    return await _mod().understand_image(image_bytes, mime_type, question)


async def extract_memories(exchange: str, bot_name: str) -> list:
    return await _mod().extract_memories(exchange, bot_name)


async def enhance_image_prompt(
    raw_prompt: str,
    character_context: str = "",
    subject_references: dict = None,
    subject_supplements: dict = None,
    reference_images: list = None,
    reference_image_labels: list = None,
    n_subjects_override: int = None,
    scene_only: bool = False,
) -> str:
    """Forward to the active backend's enhancer.

    `scene_only` is a Qwen-edit-pipeline-only flag honoured by the LM Studio
    backend; for Groq / Ollama it's silently dropped (their enhancers don't
    accept the kwarg) so non-LMStudio rigs keep working unchanged.
    """
    extra: dict = {}
    if scene_only and _backend() == "lmstudio":
        extra["scene_only"] = True
    return await _mod().enhance_image_prompt(
        raw_prompt,
        character_context=character_context,
        subject_references=subject_references,
        subject_supplements=subject_supplements,
        reference_images=reference_images,
        reference_image_labels=reference_image_labels,
        n_subjects_override=n_subjects_override,
        **extra,
    )


async def generate_image_comment(
    image_prompt: str,
    bot_name: str,
    character_background: str,
    user_request: str = "",
    history: list = None,
) -> str:
    return await _mod().generate_image_comment(
        image_prompt, bot_name, character_background,
        user_request=user_request, history=history,
    )


async def generate_suggestions(
    topic: str,
    bot_name: str,
    character_background: str,
    count: int = 3,
    guiding_prompt: str = "",
    language_sample: str = "",
) -> list:
    return await _mod().generate_suggestions(
        topic, bot_name, character_background,
        count=count, guiding_prompt=guiding_prompt,
        language_sample=language_sample,
    )
