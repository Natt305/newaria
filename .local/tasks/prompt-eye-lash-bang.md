---
title: Fix image prompt rewriter — eye color, eyelash, and bang asymmetry rules
---

Three targeted fixes to the prompt rewriter system prompt in groq_ai.py and ollama_ai.py.
No new features, no structural changes — only the instruction text and the good-output example change.

## Problem

1. **Eye color** — The current instruction says "'Pale soft honey-gold' is very different from 'vivid amber'."
   The good-output example still uses "pale soft honey-gold eyes." Both steer the rewriter toward warm golden/amber.
   The actual character has soft **olive-hazel** eyes — muted, low-saturation, brownish-yellow with a subtle greenish tint.
   The model is producing vivid amber/gold which doesn't match at all.

2. **Eyelashes** — No instruction exists for eyelash color or weight.
   The image model defaults to dark heavy lashes (statistically the norm), but a pale/cool-colored character
   should have light, subtle lashes consistent with their color palette.
   Generated images show stark dark lashes that clash with near-white hair and pale skin.

3. **Bang asymmetry** — The gap/parting on one side of the bangs is listed as one sub-bullet inside a general
   HAIRSTYLE block and is getting ignored by the model. It needs to be a standalone priority rule so the
   rewriter always surfaces it at the top of the prompt, not buried mid-sentence.

## Changes

### groq_ai.py — `enhance_image_prompt` system prompt

1. **EYE COLOR rule** — Replace the current example line:
   ```
   "- EYE COLOR: describe hue AND saturation/brightness. 'Pale soft honey-gold' is
   very different from 'vivid amber' or 'dark orange-brown'. Match what you see.\n"
   ```
   With a richer instruction that explicitly names the olive-hazel axis:
   ```
   "- EYE COLOR: describe hue, saturation, AND any secondary color undertones.
   'Soft muted olive-hazel with subtle greenish undertones' is very different from
   'vivid amber' or 'warm golden'. If eyes have a cool or greenish tint, name it.
   Low-saturation and desaturated eye colors must be described that way — never
   default to vivid warm amber if the reference shows muted or olive-toned irises.\n"
   ```

2. **EYELASH rule** — Add a new rule immediately after the EYE COLOR rule:
   ```
   "- EYELASHES: for characters with a light or cool color palette (near-white/silver hair,
   pale skin), explicitly describe lash color — e.g. 'light warm-brown lashes', 'soft subtle
   lashes', NOT dark or heavy. Dark eyelashes clash with pale color schemes; match lash
   intensity to the character's overall lightness.\n"
   ```

3. **BANG ASYMMETRY rule** — Add a dedicated standalone rule BEFORE the HAIRSTYLE block:
   ```
   "- BANG ASYMMETRY (CRITICAL): if the reference shows a gap, split, or parting in the
   bangs on one specific side — state it explicitly and early in the prompt, e.g.
   'straight bangs parted slightly to the right, small gap on the right side'.
   Do not bury this in a longer hairstyle description — it must appear near the front.\n"
   ```

4. **Good-output example** — Update the eye color in the example from "pale soft honey-gold eyes"
   to "soft muted olive-hazel eyes with subtle greenish tint" to demonstrate the target style.

### ollama_ai.py — `enhance_image_prompt` system prompt

Apply equivalent changes to the ollama backend's system prompt:
- Same EYE COLOR expanded rule
- Same EYELASH rule
- Same BANG ASYMMETRY rule
- Update good-output example if it references eye color

## Files
- `groq_ai.py` (lines ~438-466 of enhance_image_prompt system prompt)
- `ollama_ai.py` (lines ~535-553 of enhance_image_prompt system prompt)
