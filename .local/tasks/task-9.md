---
title: Rewriter must discard raw prompt appearance words — no duplicate descriptions, enforce bang gap
---
---
title: Rewriter must discard raw prompt appearance words — no duplicate descriptions
---

Small targeted change to groq_ai.py and ollama_ai.py system prompt only.

## Problem

When the LLM writes an [IMAGE:] marker (e.g. "silver-haired girl with amber eyes playing guitar"),
that text becomes the raw_prompt fed into enhance_image_prompt. The rewriter then keeps the
original appearance words at the START of the output and appends corrected descriptions at the end:

  "long silver hair and amber eyes playing guitar... [then later] ...long straight silver-mint hair"

Flux reads left-to-right. The first occurrence of each trait dominates. The corrected description
at the end is too late to override "silver hair" or "amber eyes" from the raw input.

## Fix — one new rule in the system prompt (both files)

Add the following rule immediately BEFORE the HAIR COLOR rule in the Rules section:

  "- DISCARD raw prompt appearance words: the raw prompt may contain wrong or approximate
  color/style descriptions written without access to reference photos. Do NOT copy or repeat
  any hair color, eye color, skin tone, or hairstyle words from the raw prompt into your output.
  Rewrite ALL physical appearance from scratch using the reference photos and character context.
  The final output must contain exactly ONE description of each trait — yours, not the raw prompt's.\n"

This makes it explicit that the rewriter must replace, not append.

## Files
- groq_ai.py (add rule before HAIR COLOR rule, ~line 460)
- ollama_ai.py (add rule before HAIR COLOR rule, ~line 553)