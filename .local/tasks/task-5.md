---
title: Remove SDXL and DreamShaper support, restore simple Flux-only pipeline
---
---
title: Remove SDXL and DreamShaper support, restore simple Flux-only pipeline
---

Revert the multi-model image generation back to a single Flux-only implementation. The user no longer wants SDXL/DreamShaper support.

## Files to change

### cloudflare_ai.py
- Remove `_MODEL_PRESETS`, `_active_preset()`, `active_model_name()`, `uses_negative_prompt()`.
- Restore the original simple module-level constants: `MODEL`, `WIDTH`, `HEIGHT`, `NUM_STEPS`.
- Restore `_do_generate()` to build the Flux payload internally (prompt, width, height, num_steps).
- `generate_image(prompt)` — remove the `negative_prompt` parameter; build Flux payload directly.

### groq_ai.py
- Remove `_uses_sdxl()` helper.
- Revert `enhance_image_prompt` return type from `tuple` back to `str`.
- Remove the SDXL `if sdxl:` branching for output format / POSITIVE:/NEGATIVE: parsing.
- Restore original single-string return: `return enhanced.strip()` / `return raw_prompt`.

### ollama_ai.py
- Remove `_uses_sdxl()` helper.
- Same revert as groq_ai.py: return type back to `str`, remove SDXL branch.

### ai_backend.py
- Revert `enhance_image_prompt` return type annotation back to `str`.

### bot.py
- In `process_chat`: unpack `enhance_image_prompt` result back to a plain string (not tuple).
- Remove `_neg_prompt` variable.
- Remove `negative_prompt=_neg_prompt` arg from `cloudflare_ai.generate_image()` call.
- In `/generate` command: remove `negative_prompt=None` from `generate_image()` call.
- Remove `negative_prompt=None` from `ui.GenerateView(...)` instantiation.

### views.py
- Remove `negative_prompt` param from `GenerateView.__init__`.
- Remove `self.negative_prompt` storage.
- Revert `cloudflare_ai.generate_image(self.prompt)` — no negative_prompt arg.
- Revert `GenerateView(self.prompt, new_bytes, new_mime)` — no negative_prompt kwarg.

### launcher.py
- Remove the `else:` block that imports `cloudflare_ai` and logs model key/path/negative-prompt support.
- Keep the existing `[Warning] Cloudflare config is incomplete` line as-is.