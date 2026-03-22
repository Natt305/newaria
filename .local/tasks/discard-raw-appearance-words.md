---
title: Rewriter must discard raw prompt appearance words — no duplicate descriptions, enforce bang gap
---

Small targeted change to groq_ai.py and ollama_ai.py system prompt only.

## Problem 1 — Duplicate appearance descriptions

When the LLM writes an [IMAGE:] marker (e.g. "silver-haired girl with amber eyes playing guitar"),
that text becomes the raw_prompt fed into enhance_image_prompt. The rewriter then keeps the
original appearance words at the START of the output and appends corrected descriptions at the end:

  "long silver hair and amber eyes playing guitar... [then later] ...long straight silver-mint hair"

Flux reads left-to-right. The first occurrence of each trait dominates. The corrected description
at the end is too late to override "silver hair" or "amber eyes" from the raw input.

## Problem 2 — Bang asymmetry silently dropped

The BANG ASYMMETRY rule exists but the output produces "straight-across bangs" with NO mention
of the gap on the right side. The rewriter describes the bang STYLE correctly but omits the
asymmetric gap entirely.

Observed output: "straight-across bangs, no ahoge, completely loose and flowing"
Expected output: "straight-across bangs with a small gap on the right side, no ahoge, completely loose and flowing"

The gap needs to be treated as a mandatory field — not optional — whenever bang style is described.

## Fix 1 — DISCARD rule (add immediately before HAIR COLOR rule in both files)

  "- DISCARD raw prompt appearance words: the raw prompt may contain wrong or approximate
  color/style descriptions written without reference photos. Do NOT copy or repeat any hair color,
  eye color, skin tone, or hairstyle words from the raw prompt into your output.
  Rewrite ALL physical appearance from scratch using the reference photos and character context.
  The final output must contain exactly ONE description of each trait — yours, not the raw prompt's.\n"

## Fix 2 — Strengthen the BANG ASYMMETRY rule (update existing rule in both files)

Update the BANG ASYMMETRY rule to make the gap MANDATORY when asymmetry is present:

  "- BANG ASYMMETRY (MANDATORY): if the reference or character context shows a gap or parting
  in the bangs on one specific side, you MUST include it in the output — it is not optional.
  State the side explicitly: 'straight-across bangs with a small gap on the right side'.
  Never write just 'straight-across bangs' if a gap is visible — that silently drops the detail.\n"

## Files
- groq_ai.py (DISCARD rule before HAIR COLOR ~line 460; update BANG ASYMMETRY rule ~line 446)
- ollama_ai.py (DISCARD rule before HAIR COLOR ~line 553; update BANG ASYMMETRY rule ~line 550)
