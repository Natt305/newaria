# Fix Image Prompt Double-Rewriting

## What & Why

The image generation pipeline currently rewrites the prompt up to 3 times, causing the final prompt sent to Cloudflare to drift significantly from the user's original intent:

1. The character LLM writes a clean English `[IMAGE: ...]` prompt from the user request — already well-crafted with subject, style, lighting, and mood.
2. KB enrichment appends appearance references — useful, keep this.
3. `enhance_image_prompt` runs a second full LLM rewrite on top — this is the problem. It re-interprets the already-polished Stage 1 prompt and can silently change the subject, art style, or key details.

Additionally, in the fallback path (when the LLM declines to generate and `user_wants_image()` extracts raw user text), `enhance_image_prompt` is called once inside `groq_ai.chat()` and then again in `bot.py` — so fallback prompts are rewritten twice.

## Done looks like

- Image prompts that came from the `[IMAGE: ...]` marker are passed to Cloudflare with only KB enrichment applied — no second LLM rewrite.
- The second `enhance_image_prompt` call is only triggered when character appearance context needs to be injected (self-referential requests like "selfie") or when the prompt came via the fallback path without prior enhancement.
- The fallback path does not double-enhance.
- Logged output shows clearly which path was taken (marker vs fallback, enhanced vs not).

## Out of scope

- Changing the Cloudflare model or resolution (separate concern).
- Changing the character system prompt or `[IMAGE: ...]` format.

## Tasks

1. **Track prompt origin in `groq_ai.chat()`** — Change the return signature of `chat()` to include a boolean flag indicating whether the image prompt came from the `[IMAGE:]` marker (`True` = already enhanced) or from the fallback `user_wants_image()` path (`False` = raw). Remove the `enhance_image_prompt` call from inside the fallback path in `chat()` so all enhancement is handled uniformly by the caller.

2. **Fix the caller in `bot.py`** — Update `process_chat()` to use the new flag. Only call `enhance_image_prompt` when the prompt is raw (fallback path) or when character appearance context needs injecting (self-referential). For marker-sourced prompts with no character context, skip the LLM rewrite and go straight to Cloudflare with the KB-enriched prompt.

## Relevant files

- `groq_ai.py:390-497`
- `bot.py:480-524`
- `bot.py:230-258`
