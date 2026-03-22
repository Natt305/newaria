---
title: Increase rewriter word budget to 300, require detailed outfit description
---
---
title: Increase rewriter word budget to 300, require detailed outfit description
---

Small targeted change to groq_ai.py and ollama_ai.py system prompt only.

## Changes

### 1. Word budget (both files)

groq_ai.py: change "Aim for 20-70 words." to "Aim for 150-300 words."
ollama_ai.py: change "Aim for 20-60 words." to "Aim for 150-300 words."

### 2. Outfit/clothing rule (add in both files, after COMPLEXION/SKIN TONE rule)

Add a new mandatory rule:

  "- OUTFIT (mandatory, very detailed): describe the character's clothing in full detail —
  include garment types (jacket, blouse, skirt, dress, coat, etc.), colors of each piece,
  materials/textures if visible (e.g. leather, satin, knit, denim), distinctive features
  (buttons, zippers, lace, trim, patterns, logos), layering order, fit (oversized, fitted,
  loose), and any accessories (belt, tie, hat, gloves, bag, jewelry, etc.).
  Example: 'navy blue double-breasted sailor uniform, white rectangular collar with navy
  trim, short dark pleated skirt, black leather wrist cuffs, dark knee-high socks, white
  shirt underneath'.
  Never reduce outfit to a single vague word like 'uniform' or 'casual clothes'.\n"

## Files
- groq_ai.py (~line 441 for word budget; add outfit rule after COMPLEXION rule ~line 473)
- ollama_ai.py (~line 543 for word budget; add outfit rule after HAIR COLOR rule ~line 562)