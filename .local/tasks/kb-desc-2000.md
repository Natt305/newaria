# Increase KB image description limit to 2000

## What & Why
The inline description appended to image generation prompts when a KB subject is matched is currently capped at 400 characters. Increasing it to 2000 ensures that even longer, more detailed descriptions are fully represented in the raw prompt that gets passed to the enhancement step.

## Done looks like
- When the bot generates an image featuring a named KB subject, up to 2000 characters of that subject's saved description is included in the prompt string.

## Out of scope
- Changes to character image descriptions (those are already untruncated).
- Any other prompt-building logic.

## Tasks
1. Change the description truncation limit from 400 to 2000 in the KB image enrichment function.

## Relevant files
- `bot.py:347`
