"""
Cloudflare Workers AI client for image generation.
Supports Flux 1 Schnell, SDXL base, SDXL Lightning, and DreamShaper 8 LCM.
The active model is selected via the CF_IMAGE_MODEL env var (default: flux).
"""
import os
import re
import asyncio
import base64
from typing import Optional, Tuple
import aiohttp

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------
# Each entry: (model_path, width, height, num_steps, guidance_scale or None,
#               step_param_name, supports_negative_prompt)
_MODEL_PRESETS = {
    "flux": (
        "@cf/black-forest-labs/flux-1-schnell",
        512, 512, 4, None, "num_steps", False,
    ),
    "sdxl": (
        "@cf/stabilityai/stable-diffusion-xl-base-1.0",
        768, 768, 20, 7.5, "num_inference_steps", True,
    ),
    "sdxl-lightning": (
        "@cf/bytedance/stable-diffusion-xl-lightning",
        768, 768, 8, 1.0, "num_inference_steps", True,
    ),
    "sdxl_lightning": (  # underscore alias
        "@cf/bytedance/stable-diffusion-xl-lightning",
        768, 768, 8, 1.0, "num_inference_steps", True,
    ),
    "dreamshaper": (
        "@cf/lykon/dreamshaper-8-lcm",
        512, 512, 8, 1.0, "num_inference_steps", True,
    ),
}


def _active_preset() -> tuple:
    """Return the preset tuple for the currently configured model."""
    key = os.environ.get("CF_IMAGE_MODEL", "flux").strip().lower()
    if key not in _MODEL_PRESETS:
        print(f"[Cloudflare] Unknown CF_IMAGE_MODEL={key!r}, falling back to flux")
        key = "flux"
    return _MODEL_PRESETS[key]


def active_model_name() -> str:
    """Return the full Cloudflare model path for the active preset."""
    return _active_preset()[0]


def uses_negative_prompt() -> bool:
    """Return True when the active model supports negative_prompt."""
    return _active_preset()[6]


# ---------------------------------------------------------------------------
# NSFW sanitization
# ---------------------------------------------------------------------------
# Words that Cloudflare's NSFW filter falsely flags even in innocent contexts.
# Each entry is (pattern, replacement). Applied before the first request.
_NSFW_FALSE_POSITIVES = [
    (re.compile(r"\bpickled\s+cucumber[s]?\b", re.I), "preserved vegetables"),
    (re.compile(r"\bcucumber[s]?\b", re.I), "fresh green vegetable"),
    (re.compile(r"\bpickle[s]?\b", re.I), "preserved vegetable"),
    (re.compile(r"\bgherkin[s]?\b", re.I), "small green vegetable"),
]

_NSFW_AGGRESSIVE = re.compile(
    r"\b(vegetable[s]?|jar of [a-z ]+vegetables?|preserved [a-z ]+vegetable[s]?)\b",
    re.I,
)


def _sanitize_prompt(prompt: str) -> str:
    """Replace known NSFW false-positive terms with safe alternatives."""
    for pattern, replacement in _NSFW_FALSE_POSITIVES:
        prompt = pattern.sub(replacement, prompt)
    return prompt.strip()


# ---------------------------------------------------------------------------
# Core request helper
# ---------------------------------------------------------------------------

async def _do_generate(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
) -> Optional[Tuple[bytes, str]]:
    """Single attempt: POST payload to Cloudflare and parse the image response.
    Returns (bytes, mime) on success, ("NSFW", "") on NSFW rejection,
    ("API_KEY_ERROR", "") on auth failure, ("MODEL_ERROR", "") on server error,
    or None on other failure.
    """
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_image(
    prompt: str,
    negative_prompt: Optional[str] = None,
) -> Optional[Tuple[bytes, str]]:
    """Generate an image using the configured Cloudflare Workers AI model.

    negative_prompt: optional comma-separated list of features to suppress.
    Only used when the active model supports it (SDXL-family); ignored for Flux.

    Automatically sanitizes prompts that contain terms Cloudflare's NSFW filter
    incorrectly flags. If the sanitised prompt is still rejected for NSFW,
    a second attempt drops the substituted terms entirely.
    """
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")

    if not api_token or not account_id:
        print("[Cloudflare] Missing API token or account ID")
        return None

    model_path, width, height, num_steps, guidance_scale, step_param, neg_supported = _active_preset()

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_path}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    clean_prompt = _sanitize_prompt(prompt)
    if clean_prompt != prompt:
        print(f"[Cloudflare] Prompt sanitized: {prompt[:80]!r} → {clean_prompt[:80]!r}")
    print(f"[Cloudflare] Generating image — model: {model_path} — prompt: {clean_prompt[:120]}")

    def _build_payload(p: str) -> dict:
        payload: dict = {
            "prompt": p,
            "width": width,
            "height": height,
            step_param: num_steps,
        }
        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale
        if neg_supported and negative_prompt:
            payload["negative_prompt"] = negative_prompt
            print(f"[Cloudflare] Negative prompt: {negative_prompt[:120]}")
        return payload

    try:
        async with aiohttp.ClientSession() as session:
            result = await _do_generate(session, url, headers, _build_payload(clean_prompt))

            if isinstance(result, tuple) and result[0] == "NSFW":
                aggressive_prompt = _NSFW_AGGRESSIVE.sub("", clean_prompt).strip()
                aggressive_prompt = re.sub(r"\s{2,}", " ", aggressive_prompt)
                print(f"[Cloudflare] NSFW retry — aggressive prompt: {aggressive_prompt[:120]!r}")
                result = await _do_generate(session, url, headers, _build_payload(aggressive_prompt))
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
