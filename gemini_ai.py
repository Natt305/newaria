import os
import asyncio
from typing import Optional, Tuple
from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

IMAGE_GEN_MODEL = "gemini-2.5-flash-preview-image-generation"
VISION_MODEL = "gemini-2.0-flash"

_client: Optional[genai.Client] = None


def get_client() -> Optional[genai.Client]:
    global _client
    if _client is None and GEMINI_API_KEY:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def understand_image(image_bytes: bytes, mime_type: str, question: str = "Describe this image in detail.") -> str:
    client = get_client()
    if not client:
        return "Gemini API key is not configured."

    try:
        loop = asyncio.get_event_loop()

        def _run():
            response = client.models.generate_content(
                model=VISION_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    question,
                ],
            )
            return response.text

        result = await loop.run_in_executor(None, _run)
        return result or "Could not analyze image."
    except Exception as e:
        print(f"[Gemini] Image understanding error: {e}")
        return f"I couldn't analyze that image: {str(e)}"


async def generate_image(prompt: str) -> Optional[Tuple[bytes, str]]:
    client = get_client()
    if not client:
        return None

    try:
        loop = asyncio.get_event_loop()

        def _run():
            response = client.models.generate_content(
                model=IMAGE_GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.data:
                        return part.inline_data.data, part.inline_data.mime_type or "image/png"
            return None

        result = await loop.run_in_executor(None, _run)
        return result
    except Exception as e:
        err_msg = str(e).lower()
        if "api key" in err_msg or "unauthorized" in err_msg or "403" in err_msg:
            print(f"[Gemini] Invalid API key: {e}")
            return ("API_KEY_ERROR", "")
        elif "model" in err_msg or "not found" in err_msg or "404" in err_msg:
            print(f"[Gemini] Model not available: {e}")
            return ("MODEL_ERROR", "")
        else:
            print(f"[Gemini] Image generation error: {e}")
            return None
