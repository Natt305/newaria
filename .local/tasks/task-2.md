---
title: Switch to flux-2-klein-4b with correct API format
---
# Switch to flux-2-klein-4b

## What & Why

Switch the image generation model from `flux-1-schnell` back to `flux-2-klein-4b` with the correct request format. The previous attempt failed because:
1. The field name must be `steps` (not `num_steps`) per the Cloudflare API docs
2. The model requires multipart/form-data (already implemented correctly)

## Done looks like

- `cloudflare_ai.py` uses model `@cf/black-forest-labs/flux-2-klein-4b`
- The multipart form sends `steps` (not `num_steps`) as the field name
- Steps remain at 6

## Out of scope

- Changing resolution or any other generation parameters

## Tasks

1. Update `cloudflare_ai.py` — set MODEL to `@cf/black-forest-labs/flux-2-klein-4b`, restore multipart/form-data request format, and rename the `num_steps` field to `steps`.

## Relevant files

- `cloudflare_ai.py`