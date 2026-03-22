---
title: Chat pipeline debloat (simple msg trimming, image ctx cache, KB title index)
---
# Chat Pipeline Debloat

## What & Why
Three targeted optimizations to reduce latency for simple dialogue by eliminating redundant work done on every message.

## Done looks like
- Short/chitchat messages (greetings, single sentences with no questions) send a trimmed system prompt — only KB entries actually relevant to the message are included, rather than dumping all 30 entries every time.
- `build_character_images_context()` reads from an in-memory cache instead of querying the DB on every message. Cache is invalidated when character images are added or removed (`/addcharimage`, `/removecharimage`).
- The per-message scan that loads up to 200 KB entries to find image title matches in user text is replaced by a pre-built in-memory lookup (title → entry_id dict) that rebuilds only when the KB changes.

## Out of scope
- Changing AI models or temperature (separate optimization #2 and #3).
- Modifying memory retrieval logic.
- Changing how the system prompt is structured for complex/long messages.

## Tasks

1. **Simple-message KB trimming** — Add an `is_simple_message(text)` classifier (short text, no question words in Chinese or English, no recall triggers). Pass a `simple=True` flag through `build_knowledge_context`; when simple, skip injecting the background (non-relevant) KB entries and only include starred relevant ones. This keeps the system prompt small for chitchat while preserving full context for real questions.

2. **Cache character image context** — Add a module-level cache variable for the result of `build_character_images_context()`. Expose an `_invalidate_char_images_ctx()` helper and call it from every command that adds or removes character images (`/addcharimage`, `/removecharimage`, and the gallery remove button in `views.py`).

3. **Pre-built KB image title index** — Replace the `get_all_entries(limit=200)` scan in `process_chat` with an in-memory dict `{ lowercase_title: entry_id }` built from image-only entries. Rebuild the index lazily on first use and invalidate it (reset to `None`) whenever a KB entry is added, updated, or deleted (`/remember`, `/saveimage`, `/forget`, `/setdesc` commands).

## Relevant files
- `bot.py:42-52`
- `bot.py:123-182`
- `bot.py:378-453`
- `bot.py:975-1100`
- `database.py:649`
- `views.py:1109-1116`