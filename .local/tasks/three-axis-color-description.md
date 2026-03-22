---
title: Three-axis color description for eyes/lashes + revert eye priority ruling
---

Two targeted changes to groq_ai.py and ollama_ai.py only.

## 1. Revert priority ruling for eye color

In groq_ai.py, the image_note (PRIORITY RULING block) currently reads:

  "HAIR COLOR and SKIN TONE — use what you literally see in the photos."
  "EYE COLOR — use ONLY the text description above. Do NOT name eye color from the photos..."

Revert this back to the original shape where eye color is also read from photos,
BUT add explicit instruction on HOW to describe it (three-axis — see section 2):

  "COLORS (hair color, eye color, skin tone lightness/darkness/saturation) —
  use what you literally see in the photos. The photos are the final word.
  STRUCTURE (hairstyle shape, bang style, accessories, character identity) —
  use the text descriptions as the authority."

Keep the hair example line at the bottom of image_note unchanged.

## 2. Three-axis color description rule (add in BOTH files)

Replace the current EYE COLOR rule and EYELASHES rule with expanded versions
that require three separate description axes:

### EYE COLOR rule (replace existing):
  "- EYE COLOR (three-axis required): describe eye color using ALL THREE axes separately —
  never collapse them into a single color name like 'amber' or 'hazel':\n"
  "  * HUE FAMILY: the base color group, e.g. 'warm olive-brown', 'gray-green', 'khaki'\n"
  "  * SATURATION: explicitly state the chroma level — 'very low saturation', 'muted',
  'desaturated', 'soft' — or 'vivid' if truly vivid\n"
  "  * BRIGHTNESS/CONTRAST: e.g. 'medium brightness', 'slightly dark', 'light'\n"
  "  Example: 'very low-saturation warm olive-brown eyes, muted and soft, medium brightness'\n"
  "  NOT: 'amber eyes' or 'honey-gold eyes' (too vague — forces model to default to vivid warm amber)\n"

### EYELASHES rule (replace existing):
  "- EYELASHES (three-axis required): describe eyelash appearance using ALL THREE axes:\n"
  "  * HUE: e.g. 'warm gray-brown', 'pale brown', 'warm beige'\n"
  "  * DARKNESS/CONTRAST: e.g. 'very low contrast against skin', 'soft', 'not dark'\n"
  "  * WEIGHT: e.g. 'fine and light', 'delicate', 'subtle'\n"
  "  Example: 'pale warm-brown lashes, very low contrast, fine and delicate'\n"
  "  NOT: just 'light lashes' (still too vague — model may render dark)\n"

## 3. Update good-output example (BOTH files)

Change the eye/lash portion of the good-output example to demonstrate three-axis style:

  OLD: "soft muted olive-hazel eyes with subtle greenish tint, soft light brown lashes,"
  NEW: "very low-saturation warm olive-brown eyes with subtle greenish undertone, muted and soft,
  medium brightness, pale warm-gray lashes, very low contrast, fine and delicate,"

## Files
- groq_ai.py (image_note block ~lines 416-428, EYE COLOR rule ~line 462, EYELASHES rule ~line 466, good-output example ~line 475)
- ollama_ai.py (EYE COLOR rule ~line 555, EYELASHES rule ~line 560, good-output example ~line 563)
