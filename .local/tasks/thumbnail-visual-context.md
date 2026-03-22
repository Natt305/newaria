# Thumbnail Visual Context for Chat

## What & Why
Instead of passing only text descriptions when the bot references character or KB images, generate small thumbnails at save-time and pass them alongside the description at chat-time. The model sees the actual image (better accuracy) at a fraction of the cost of the full file. Images are only sent when genuinely relevant — not on every message.

## Done looks like
- When a character image or KB image is saved, a 256×256 JPEG thumbnail is automatically generated and stored alongside the original
- Existing images without thumbnails get thumbnails generated on first bot startup (one-time migration)
- Character thumbnails are only passed to the AI when the user's message triggers an appearance-related query (e.g. asking what the bot looks like, self-referential image generation)
- KB image thumbnails are only passed when an existing match fires — title name in user text, or visual subject overlap from an uploaded image — capped at 2–3 thumbnails maximum
- The text description is always kept and still injected into the system prompt alongside any thumbnail passed
- Both Groq and Ollama backends receive and handle the thumbnail images correctly

## Out of scope
- Changing the full original images (they remain untouched)
- Passing thumbnails for text-only KB entries
- Any UI for managing thumbnails (they are internal only)

## Tasks

1. **Thumbnail generation in `database.py`** — On every `add_character_image`, `add_image_entry`, and `add_image_to_entry` call, use Pillow to generate a 256×256 JPEG thumbnail and save it alongside the original. Store the thumb filename in the JSON metadata. Add `get_character_image_thumb(index)` and `get_kb_image_thumb(entry_id, image_index)` retrieval helpers. On bot startup, run a one-time migration to generate missing thumbnails for any existing images.

2. **Add `context_images` to AI backends** — Add an optional `context_images: list[tuple[bytes, str]]` parameter to `chat()` in `groq_ai.py` and `ollama_ai.py`. When provided, inject the images as multimodal content (image_url entries) in the final user message alongside the text. Forward the parameter through `ai_backend.py`.

3. **Selective thumbnail loading in `bot.py`** — Add an appearance-trigger detector function (regex-based, bilingual, reusing and extending the existing `is_self_referential_image` patterns). In the chat handler: load character thumbnails only when the trigger fires or the image prompt is self-referential; load KB thumbnails only when `_enrich_image_prompt_with_kb` or `build_visual_kb_context` returns a match, capped at 2 thumbnails. Pass the collected thumbnails to `groq_ai.chat()` via `context_images`.

## Relevant files
- `database.py:119-178,440-528`
- `groq_ai.py:172-218,294-357`
- `ollama_ai.py:113-178,192-232`
- `ai_backend.py`
- `bot.py:42-139,228-285,370-470`
