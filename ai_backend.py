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
) -> tuple[str, Optional[str]]:
    kwargs = {"system_prompt": system_prompt}
    if model:
        kwargs["model"] = model
    if context_images:
        kwargs["context_images"] = context_images
    return await _mod().chat(messages, **kwargs)


async def understand_image(
    image_bytes: bytes,
    mime_type: str,
    question: str = "Describe this image in detail.",
) -> Optional[str]:
    return await _mod().understand_image(image_bytes, mime_type, question)


async def extract_memories(exchange: str, bot_name: str) -> list:
    return await _mod().extract_memories(exchange, bot_name)


async def enhance_image_prompt(raw_prompt: str, character_context: str = "") -> str:
    return await _mod().enhance_image_prompt(raw_prompt, character_context=character_context)


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
) -> list:
    return await _mod().generate_suggestions(
        topic, bot_name, character_background,
        count=count, guiding_prompt=guiding_prompt,
    )
