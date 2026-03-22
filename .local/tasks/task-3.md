---
title: Fix text model fallback & self-ref detection bugs
---
# Fix Text Model Fallback & Self-Ref Detection Bugs

## What & Why

Three related bugs affect image generation quality and reliability:

**Bug 1 — Corrupted image when text model falls back:**  
`chat()` injects base64 image data (multimodal `image_url` parts) into the last user message before entering the model retry loop. When every vision model fails and a text-only model is tried next, the multimodal content is still in the message — text models cannot process `image_url` parts and produce garbage output. Result: the generated image looks completely wrong (wrong hair length, wrong style, wrong scene).

**Bug 2 — Bot refuses "想看妳照片" (and similar) without routing to image gen:**  
The current `response_declines_image()` matches a hardcoded list of exact phrases. Any novel phrasing (e.g. "抱歉，我無法提供") passes right through as the final reply, and image generation never happens.

**Bug 3 — Enhancement skipped when LLM writes marker in third person:**  
When the vision model generates an `[IMAGE:]` marker, it describes the character in third person (e.g. "slender teenage girl with mint-green hair"). The `_is_self_ref` check runs only against that marker text — not the original user message — so it doesn't match self-reference patterns. `has_visual_refs` stays False, enhancement is skipped, and the prompt reaches Cloudflare without detailed hairstyle/eye-color/complexion traits.

## Done looks like

- When falling back to a text model while `context_images` were provided, image_url parts are stripped; the text model receives clean text-only messages.
- Refusal detection is context-gated: the heuristic only fires when `user_wants_image()` confirms the user clearly asked for an image. If that condition is met AND the bot's response has no `[IMAGE:]` marker AND the response contains any negation/inability word (無法, 不能, 沒辦法, can't, cannot, unable, not able, etc.) → treat as decline and route to image generation. The existing phrase list is kept as a fast first check; the new heuristic runs as a second pass.
- When the user's original message is self-referential, character visual refs are injected and enhancement runs even if the LLM's marker prompt uses third-person language.
- All fixes apply to both `groq_ai.py` and `ollama_ai.py` where applicable.

## Out of scope

- Replacing the phrase-list check entirely (kept as fast first pass).
- Improving text-only prompt quality when vision models are unavailable.
- Changing the vision model fallback order.

## Tasks

1. **Strip image_url parts when falling back to text models** — In the `chat()` retry loop in `groq_ai.py` and `ollama_ai.py`, detect when the current `attempt_model` is not in `VISION_MODELS`. When that's the case, rebuild the message list replacing any multimodal content blocks with their plain-text equivalent (extracting only the `"text"` part), so the text model gets a valid text-only conversation.

2. **Context-gated negation heuristic for refusal detection** — In `groq_ai.py` and `ollama_ai.py`, after the existing phrase-list check, add a second-pass heuristic: if `user_wants_image(messages)` returns a non-None result AND the response contains no `[IMAGE:]` marker AND the response matches a broad negation/inability regex (`無法|不能|沒辦法|can't|cannot|unable|not able|no way|做不到|辦不到`), treat it as a decline. Do not require an apology word — intent + negation is sufficient.

3. **Check self-ref on original user message, not just marker prompt** — In `bot.py`, when an `[IMAGE:]` marker is found, also run `is_self_referential_image()` against the original user message text. If either the marker prompt or the user message is self-referential, set `_is_self_ref=True` and collect character thumbnails so enhancement runs with full visual grounding.

## Relevant files

- `groq_ai.py:83-95,97-142,319-341,480-554`
- `ollama_ai.py:16-90,152-165,280-390`
- `bot.py:580-650`