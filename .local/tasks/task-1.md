---
title: Configurable AI backend (Ollama + Groq)
---
# Configurable AI Backend (Ollama + Groq)

## What & Why
Add Ollama as an alternative AI backend so the bot can run against a local model (gemma3:12b by default) instead of Groq. The active backend and model names should be selectable via `tokens.txt` config, with Groq remaining the default so nothing breaks for users who don't change anything.

## Done looks like
- `tokens.txt` has new config options: `AI_BACKEND`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_VISION_MODEL`
- Setting `AI_BACKEND=ollama` routes all AI calls to a local Ollama instance
- Default Ollama model is `gemma3:12b` (handles both text and vision)
- Setting `AI_BACKEND=groq` (or leaving it unset) keeps existing Groq behaviour unchanged
- Startup logs clearly show which backend and model are active
- All bot features (chat, image understanding, memory extraction, suggestions, image prompt enhancement) work with either backend

## Out of scope
- Cloudflare image generation is unaffected — it stays as-is
- No UI for switching backends at runtime (config file only)
- No support for other backends (OpenAI, Gemini, etc.) in this task

## Tasks
1. **Create `ollama_ai.py`** — Implement all functions from `groq_ai.py`'s public interface (`chat`, `understand_image`, `extract_memories`, `generate_suggestions`, `enhance_image_prompt`, `generate_image_comment`, `is_self_referential_image`, `is_recall_request`, `response_declines_image`, `user_wants_image`) using Ollama's OpenAI-compatible API endpoint (`/v1/chat/completions`). Default model is `gemma3:12b`, vision model is also `gemma3:12b`.

2. **Create `ai_backend.py`** — A thin routing module that reads `AI_BACKEND` from environment and delegates every public function call to either `groq_ai` or `ollama_ai`. This module exposes the same interface so callers need no logic changes.

3. **Update `bot.py` and `views.py`** — Change `import groq_ai` to `import ai_backend as groq_ai` in both files (one-line change each). No other changes needed since the interface is identical.

4. **Update `tokens.txt` and `launcher.py`** — Add the four new config options to `tokens.txt` with comments and example values. Update `launcher.py` to log the active backend and model name on startup.

## Relevant files
- `groq_ai.py`
- `ollama_ai.py`
- `bot.py:1-20`
- `views.py:1-15`
- `tokens.txt`
- `launcher.py`