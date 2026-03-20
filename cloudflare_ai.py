"""
Cloudflare Workers AI client for image generation using Flux 2.
Uses @cf/black-forest-labs/flux-2-dev model.
"""
import os
import asyncio
import base64
from typing import Optional, Tuple
import aiohttp

MODEL = "@cf/black-forest-labs/flux-1-schnell"


async def generate_image(prompt: str) -> Optional[Tuple[bytes, str]]:
    """Generate an image using Cloudflare Workers AI Flux 2 model."""
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")

    if not api_token or not account_id:
        print("[Cloudflare] Missing API token or account ID")
        return None

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{MODEL}"
    headers = {
        "Authorization": f"Bearer {api_token}",
    }

    print(f"[Cloudflare] Sending request to: {url}")

    headers["Content-Type"] = "application/json"

    payload = {
        "prompt": prompt,
        "width": 512,
        "height": 512,
        "num_steps": 4,
    }

    try:
        async with aiohttp.ClientSession() as session:
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
                    return None

                if resp.status >= 500:
                    body = await resp.text()
                    print(f"[Cloudflare] Server error {resp.status}: {body[:300]}")
                    return ("MODEL_ERROR", "")

                if resp.status != 200:
                    body = await resp.text()
                    print(f"[Cloudflare] Unexpected status {resp.status}: {body[:300]}")
                    return None

                if "image" in content_type:
                    img_data = await resp.read()
                    print(f"[Cloudflare] Got binary image, size: {len(img_data)} bytes")
                    if img_data:
                        return img_data, "image/png"
                    return None

                raw = await resp.read()
                print(f"[Cloudflare] Non-image response ({len(raw)} bytes): {raw[:200]}")

                try:
                    data = await resp.json(content_type=None)
                except Exception:
                    import json as _json
                    data = _json.loads(raw)

                print(f"[Cloudflare] JSON keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

                if not data.get("success", True):
                    print(f"[Cloudflare] API errors: {data.get('errors', [])}")
                    return ("MODEL_ERROR", "")

                result = data.get("result", data)
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
                    img_data = base64.b64decode(b64)
                    print(f"[Cloudflare] Decoded base64 image: {len(img_data)} bytes")
                    return img_data, "image/png"

                print(f"[Cloudflare] Could not extract image from response: {str(data)[:400]}")
                return None

    except asyncio.TimeoutError:
        print("[Cloudflare] Request timed out after 120s")
        return None
    except Exception as e:
        print(f"[Cloudflare] Exception: {type(e).__name__}: {e}")
        return None
