---
title: Chat response progress bar
---
# Chat Response Progress Bar

  ## What & Why
  While the bot processes a reply, show a minimal 5-block progress bar with a fixed "思考中" label so the user gets clear, consistent feedback on every message — regardless of whether it ends as a text reply or an image.

  ## Done looks like
  - As soon as the bot starts processing, a message appears as a reply in this exact format (no other text):
    `⬜⬜⬜⬜⬜ 思考中`
  - The bar always has exactly 5 blocks, advancing one block at a time as each internal stage completes:
    `🟦⬜⬜⬜⬜ 思考中` → `🟦🟦⬜⬜⬜ 思考中` → `🟦🟦🟦⬜⬜ 思考中` → `🟦🟦🟦🟦⬜ 思考中` → `🟦🟦🟦🟦🟦 思考中`
  - The "思考中" label never changes regardless of what stage the bot is on
  - The bar is the same 5 stages for every path (text reply, image upload, or image generation) — no variable stage count
  - The bar message is deleted immediately before the final reply is sent (text or image)
  - If an error occurs, the bar message is deleted before the error reply is sent
  - Bar edit failures are silently ignored — they must never break the chat flow

  ## Out of scope
  - Variable stage counts per path
  - Different labels per stage
  - Progress bars for admin commands (those already have their own)

  ## Tasks
  1. **Add 5-stage progress bar to `process_chat`** — Send the initial `⬜⬜⬜⬜⬜ 思考中` reply at the very start and advance it (edit) at evenly-spaced checkpoints across the full pipeline (image analysis, KB context load, memory load, AI chat call, image generation or final prep). All edits wrapped in try/except.

  2. **Delete bar before every exit** — In every exit path of `process_chat` — text reply, image reply, and all error reply paths — delete the bar message first, then send the final reply so the bar never lingers.

  3. **SuggestionButton path** — `SuggestionButton.callback` calls `process_chat` directly, so it inherits the bar automatically with no extra changes.

  ## Relevant files
  - `bot.py:391-614`
  - `bot.py:936-996`