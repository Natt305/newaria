---
title: Pass thumbnails to image prompt rewriter
---
# Pass thumbnails to image prompt rewriter

## What & Why
`enhance_image_prompt` currently rewrites the image generation prompt using only text descriptions of the character/subject. It never actually sees the reference photos, so subtle visual traits like eye color can still be wrong even when descriptions are available.

This task passes the stored 256×256 thumbnails directly into the prompt rewriting step, so the vision model can literally observe hair color, eye color, hairstyle, and other details when crafting the final generation prompt.

## Done looks like
- When a self-referential image is requested ("draw yourself"), character thumbnails are passed to the prompt rewriter and the resulting prompt reflects the actual visual appearance from the photos.
- When a KB subject is referenced by name in an image request, that subject's KB thumbnails are also passed to the prompt rewriter.
- All existing fallback behavior (text-only when no thumbnails exist) is preserved.

## Out of scope
- Changing how thumbnails are stored or generated.
- Changing the main conversational `chat()` call — only `enhance_image_prompt` is affected.

## Tasks
1. Add a `reference_images` parameter (list of `(bytes, mime_type)` tuples) to `enhance_image_prompt` in `groq_ai.py`. When provided, inject the images into the user message of the rewrite call and ensure a vision-capable model is selected.
2. In `bot.py` `process_chat`, when `_is_self_ref` is True, collect the character thumbnails and pass them as `reference_images` to `enhance_image_prompt`. For KB subject matches, collect and pass that subject's KB thumbnail similarly.

## Relevant files
- `groq_ai.py:344-407`
- `bot.py:498-595`
- `database.py:221` (get_character_image_thumb)
- `database.py:551` (get_kb_image_thumb)