---
title: Prompt rewriter fixes — full log, eyelash rule, eye color, bang asymmetry
---

Small, targeted fixes to groq_ai.py and ollama_ai.py only.
No structural changes — only logging and system prompt instruction text.

## 1. Full prompt logging (groq_ai.py line 484, ollama_ai.py line 562)

Remove the [:120] slice so the full enhanced prompt is always printed.

groq_ai.py line 484:
  BEFORE: print(f"[Groq] Prompt enhanced: {enhanced[:120]}")
  AFTER:  print(f"[Groq] Prompt enhanced: {enhanced}")

ollama_ai.py line 562:
  BEFORE: print(f"[Ollama] Prompt enhanced: {enhanced[:120]}")
  AFTER:  print(f"[Ollama] Prompt enhanced: {enhanced}")

## 2. Eyelash rule (add in BOTH files, after the EYE COLOR rule)

Add this new rule immediately after the EYE COLOR bullet:

  "- EYELASHES: when the character has a light, pale, or cool color palette "
  "(near-white or silver hair, pale skin), explicitly include eyelash color in the "
  "prompt — e.g. 'soft light brown lashes', 'subtle warm lashes'. Do NOT default "
  "to dark or heavy lashes — that clashes with a pale color scheme.\n"

## 3. Eye color good-output example (add in BOTH files)

In the Good output example at the bottom of the system prompt, change:
  "pale soft honey-gold eyes"
to:
  "soft muted olive-hazel eyes with subtle greenish tint, low saturation"

This teaches the model what the target eye description looks like.

## 4. Bang asymmetry — standalone rule (add in BOTH files)

Add a dedicated rule BEFORE the HAIRSTYLE block (after the physical-details priority line):

  "- BANG ASYMMETRY (high priority): if the reference shows a gap or parting "
  "in the bangs on one specific side, state this explicitly and early — e.g. "
  "'straight-across bangs with a small gap on the right side'. Do not merge it "
  "into a longer hairstyle sentence where it gets lost.\n"

## Files
- groq_ai.py (~lines 444-466 of enhance_image_prompt system prompt, line 484 for logging)
- ollama_ai.py (~lines 535-565 of enhance_image_prompt system prompt, line 562 for logging)
