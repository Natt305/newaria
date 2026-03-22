---
title: Revert to flux-1-schnell with 4 steps
---
# Revert to flux-1-schnell, 4 steps

## What & Why

Revert cloudflare_ai.py back to flux-1-schnell (free tier, no Partner access needed).
Also reduce NUM_STEPS from 6 to 4.

## Done looks like

- MODEL = "@cf/black-forest-labs/flux-1-schnell"
- NUM_STEPS = 4
- Request body uses JSON (not multipart) — flux-1-schnell requires JSON
- Content-Type: application/json header restored
- Module/function docstrings updated

## Relevant files

- cloudflare_ai.py
