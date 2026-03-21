"""
Cloudflare Workers AI client for image generation using Flux 1 Schnell.
Uses @cf/black-forest-labs/flux-1-schnell model.
"""
import os
import re
import asyncio
import base64
from typing import Optional, Tuple
import aiohttp

MODEL = "@cf/black-forest-labs/flux-1-schnell"

WIDTH = 512
HEIGHT = 512
NUM_STEPS = 6

# Words that Cloudflare's NSFW filter falsely flags even in innocent contexts.
# Each entry is (pattern, replacement). Applied before the first request.
_NSFW_FALSE_POSITIVES = [
    # Cucumbers and pickles — innocuous vegetable but triggers filter due to shape
    (re.compile(r"\bpickled\s+cucumber[s]?\b", re.I), "preserved vegetables"),
    (re.compile(r"\bcucumber[s]?\b", re.I), "fresh green vegetable"),
    (re.compile(r"\bpickle[s]?\b", re.I), "preserved vegetable"),
    (re.compile(r"\bgherkin[s]?\b", re.I), "small green vegetable"),
]

# If the first sanitised attempt still gets 400 NSFW, apply these additional
# removals (drop the whole phrase rather than substitute) for a second retry.
_NSFW_AGGRESSIVE = re.compile(
    r"\b(vegetable[s]?|jar of [a-z ]+vegetables?|preserved [a-z ]+vegetable[s]?)\b",
    re.I,
)


def _sanitize_prompt(prompt: str) -> str:
    """Replace known NSFW false-positive terms with safe alternatives."""
    for pattern, replacement in _NSFW_FALSE_POSITIVES:
        prompt = pattern.sub(replacement, prompt)
    return prompt.strip()


async def _do_generate(session: aiohttp.ClientSession, url: str, headers: dict, prompt: str) -> Optional[Tuple[bytes, str]]:
    """Single attempt: POST prompt to Cloudflare and parse the image response.
    Returns (bytes, mime) on success, ("NSFW", "") on NSFW rejection,
    ("API_KEY_ERROR", "") on auth failure, ("MODEL_ERROR", "") on server error,
    or None on other failure.
    """
    payload = {
        "prompt": prompt,
        "width": WIDTH,
        "height": HEIGHT,
        "num_steps": NUM_STEPS,
    }
    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
        content_type = resp.headers.get("Content-Type", "")
        print(f"[Cloudflare] Status: {resp.status} | Content-Type: {content_type}")

        if resp.status in (401, 403):
            body = await resp.text()
            print(f"[Cloudflare] Auth error body: {body[:300]}")
            return ("API_KEY_ERROR", "")

        if resp.status == 400:
            body = await resp.text()
            print(f"[Cloudflare] Bad request: {body[:300]}")
            if "NSFW" in body or "nsfw" in body.lower():
                return ("NSFW", "")
            return None

        if resp.status >= 500:
            body = await resp.text()
            print(f"[Cloudflare] Server error {resp.status}: {body[:300]}")
            return ("MODEL_ERROR", "")

        if resp.status != 200:
            body = await resp.text()
            print(f"[Cloudflare] Unexpected status {resp.status}: {body[:300]}")
            return None

        raw = await resp.read()

        if "image" in content_type:
            print(f"[Cloudflare] Got binary image, size: {len(raw)} bytes")
            return (raw, "image/png") if raw else None

        print(f"[Cloudflare] Non-image response ({len(raw)} bytes): {raw[:200]}")

        import json as _json
        try:
            data = _json.loads(raw)
        except Exception as parse_err:
            print(f"[Cloudflare] Could not parse response as JSON: {parse_err}")
            return None

        print(f"[Cloudflare] JSON keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

        if isinstance(data, dict) and not data.get("success", True):
            print(f"[Cloudflare] API errors: {data.get('errors', [])}")
            return ("MODEL_ERROR", "")

        result = data.get("result", data) if isinstance(data, dict) else data
        print(f"[Cloudflare] result type: {type(result)}, value preview: {str(result)[:200]}")

        if isinstance(result, dict):
            b64 = result.get("image") or result.get("images", [None])[0]
        elif isinstance(result, str):
            b64 = result
        elif isinstance(result, (bytes, bytearray)):
            return bytes(result), "image/png"
        else:
            b64 = None

        if b64:
            try:
                img_data = base64.b64decode(b64)
                print(f"[Cloudflare] Decoded base64 image: {len(img_data)} bytes")
                return img_data, "image/png"
            except Exception as decode_err:
                print(f"[Cloudflare] Base64 decode failed: {decode_err}")
                return None

        print(f"[Cloudflare] Could not extract image from response: {str(data)[:400]}")
        return None


async def generate_image(prompt: str) -> Optional[Tuple[bytes, str]]:
    """Generate an image using Cloudflare Workers AI Flux 1 Schnell model.

    Automatically sanitizes prompts that contain terms Cloudflare's NSFW filter
    incorrectly flags (e.g. "cucumber"). If the sanitised prompt is still rejected
    for NSFW, a second attempt drops the substituted terms entirely.
    """
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")

    if not api_token or not account_id:
        print("[Cloudflare] Missing API token or account ID")
        return None

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    # Pre-sanitize: replace known false-positive NSFW terms before first send
    clean_prompt = _sanitize_prompt(prompt)
    if clean_prompt != prompt:
        print(f"[Cloudflare] Prompt sanitized: {prompt[:80]!r} → {clean_prompt[:80]!r}")
    print(f"[Cloudflare] Generating image — prompt: {clean_prompt[:120]}")

    try:
        async with aiohttp.ClientSession() as session:
            result = await _do_generate(session, url, headers, clean_prompt)

            # If NSFW flag persists after first sanitization, retry more aggressively
            if isinstance(result, tuple) and result[0] == "NSFW":
                aggressive_prompt = _NSFW_AGGRESSIVE.sub("", clean_prompt).strip()
                aggressive_prompt = re.sub(r"\s{2,}", " ", aggressive_prompt)
                print(f"[Cloudflare] NSFW retry — aggressive prompt: {aggressive_prompt[:120]!r}")
                result = await _do_generate(session, url, headers, aggressive_prompt)
                if isinstance(result, tuple) and result[0] == "NSFW":
                    print("[Cloudflare] NSFW rejected even after aggressive cleanup — giving up")
                    return None

            return result

    except asyncio.TimeoutError:
        print("[Cloudflare] Request timed out after 120s")
        return None
    except Exception as e:
        print(f"[Cloudflare] Exception: {type(e).__name__}: {e}")
        return None
