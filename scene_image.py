"""
Scene image runner for LM Studio RP-mode replies.

Single shared entry point (`run_scene_image`) used by all three triggers:
  - 🎬 button click on a bot message
  - LLM-emitted [SCENE] auto-trigger
  - Visual-intent re-route (when scene mode is on for the channel)
  - Legacy [IMAGE: ...] re-route (when scene mode is on for the channel)

Guarantees:
  - At most one image per (channel_id, target_message_id) — in-flight set
  - Per-channel cooldown (3 generations / 60s)
  - Image is attached IN-PLACE via bot_message.edit(attachments=[...]),
    never as a separate reply (kills the doubled-response feel)
  - Same temporary-message progress bar UX the legacy flow uses

Per-channel toggle persists in the existing settings store under the key
`scene_image:{channel_id}` (bool, default False).

Engine tiers, in order of fidelity:
  - Qwen (ComfyUI):   full multi-ref edit + cinematic suffix, up to 4 refs
  - FLUX (ComfyUI):   single-ref multiref + cinematic suffix
  - Cloudflare:       plain text-to-image with cinematic suffix
"""

from __future__ import annotations

import asyncio
import io
import os
import re
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Optional

import discord
from discord.ext import commands

import database
import image_dispatch
import ai_backend as groq_ai


CINEMATIC_SUFFIX = (
    ", cinematic composition, soft rim light, "
    "shallow depth of field, film grain"
)


# ── State ─────────────────────────────────────────────────────────────────────

_inflight: set[tuple[str, int]] = set()
_cooldown = commands.CooldownMapping.from_cooldown(
    3, 60.0, commands.BucketType.channel
)


# ── Per-channel toggle ────────────────────────────────────────────────────────

def _setting_key(channel_id) -> str:
    return f"scene_image:{channel_id}"


def is_scene_mode_on(channel_id) -> bool:
    return bool(database.get_setting(_setting_key(channel_id)))


def set_scene_mode(channel_id, enabled: bool) -> bool:
    return database.set_setting(_setting_key(channel_id), bool(enabled))


# ── Visual-intent reuse ───────────────────────────────────────────────────────

# Compact visual-intent regex used when scene mode is on but the LLM did not
# emit [SCENE] and there's no [IMAGE: ...] tag — covers EN/ZH "show me / I
# want to see" phrasing. Looser than lmstudio_ai.IMAGE_REQUEST_PATTERNS so
# RP-mode users don't have to phrase requests as image commands.
_VISUAL_INTENT_RE = re.compile(
    r"(?:show\s+me|let\s+me\s+see|i\s+want\s+to\s+see|"
    r"draw\s+me|paint\s+me|illustrate|"
    r"我想看|我要看|讓我看|給我看|看看妳|看看你|"
    r"幫我畫|畫一下|畫個|長什麼樣|長甚麼樣)",
    re.I,
)


def is_user_visual_intent(text: str) -> bool:
    """True if the user message reads like a visual request (loose detector)."""
    if not text:
        return False
    if _VISUAL_INTENT_RE.search(text):
        return True
    # Also defer to the lmstudio backend's own self-ref detector when present
    try:
        return bool(groq_ai.is_self_referential_image(text))
    except Exception:
        return False


# ── SCENE marker (re-exported so bot.py can strip server-side too) ────────────

SCENE_MARKER_RE = re.compile(r"\[\s*SCENE(?:\s*:\s*([^\]\n]+?))?\s*\]", re.I)


# ── Progress bar context manager (factored out of process_chat) ───────────────

@asynccontextmanager
async def progress_bar(
    target_message: discord.Message,
    name: str,
    formatter: Callable[[str, str], Optional[str]],
    *,
    enabled: bool = True,
):
    """Reusable progress UX matching the legacy [IMAGE:] flow.

    Yields an async `on_progress(tag)` callback (or None when disabled).
    Posts a temp progress reply, polls a queue, edits the message live with
    `formatter(tag, name)` output, deletes the message on exit.
    """
    if not enabled:
        yield None
        return

    progress_msg: Optional[discord.Message] = None
    try:
        progress_msg = await target_message.reply(
            formatter("STAGE:loading", name) or "正在生成圖像…",
            mention_author=False,
        )
    except discord.HTTPException:
        progress_msg = None

    if progress_msg is None:
        yield None
        return

    queue: asyncio.Queue[str] = asyncio.Queue()

    async def on_progress(tag: str) -> None:
        await queue.put(tag)

    async def poller() -> None:
        while True:
            tag = await queue.get()
            while not queue.empty():
                try:
                    tag = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            content = formatter(tag, name)
            if content:
                try:
                    await progress_msg.edit(content=content)
                except discord.HTTPException:
                    pass
            await asyncio.sleep(0.5)

    poller_task = asyncio.create_task(poller())
    try:
        yield on_progress
    finally:
        poller_task.cancel()
        try:
            await poller_task
        except asyncio.CancelledError:
            pass
        try:
            await progress_msg.delete()
        except discord.HTTPException:
            pass


# ── Reference-image assembly ──────────────────────────────────────────────────

def _gather_refs(seed_text: str) -> tuple[list, list, dict]:
    """Build (refs, subjects, appearances) from character + KB matches.

    Character (bot) photos go first (up to 2). Then any KB image entry whose
    title appears in `seed_text` contributes up to 2 photos each. Hard cap at
    4 refs total — Qwen v2 image1..image4 slots.
    """
    MAX_TOTAL = 4
    refs: list = []
    subjects: list = []
    appearances: dict = {}

    haystack = (seed_text or "").lower()

    char = database.get_character()
    bot_name = (char.get("name") or "").strip()
    looks = (char.get("looks") or "").strip()

    char_count = database.get_character_image_count()
    for i in range(1, min(char_count, 2) + 1):
        if len(refs) >= MAX_TOTAL:
            break
        full = database.get_character_image(i)
        if full:
            refs.append(full)
            subjects.append(bot_name or "self")
    if bot_name and looks:
        appearances[bot_name] = looks

    if len(refs) < MAX_TOTAL:
        try:
            kb_entries = database.get_image_entries()
        except Exception:
            kb_entries = []
        for entry in kb_entries:
            if len(refs) >= MAX_TOTAL:
                break
            title = (entry.get("title") or "").strip()
            if len(title) < 2 or title.lower() not in haystack:
                continue
            entry_id = entry.get("id")
            if entry_id is None:
                continue
            n_imgs = database.get_entry_image_count(entry_id)
            for idx in range(1, min(n_imgs, 2) + 1):
                if len(refs) >= MAX_TOTAL:
                    break
                full = database.get_kb_image_full(entry_id, idx)
                if full:
                    refs.append(full)
                    subjects.append(title)
            desc = (entry.get("appearance_description") or "").strip()
            if desc and title not in appearances:
                appearances[title] = desc

    return refs, subjects, appearances


# ── Prompt derivation ─────────────────────────────────────────────────────────

async def _derive_prompt(seed: str, looks: str, bot_name: str) -> str:
    """Run prompt enhancement against the active backend's enhancer.

    Falls back to the seed itself on any failure — scene-image generation is
    never blocked by enhancement issues.
    """
    char_context = ""
    if looks:
        char_context = (
            f"[Authoritative written appearance — {bot_name or 'character'}]\n"
            f"{looks}"
        )
    try:
        enriched = await groq_ai.enhance_image_prompt(
            seed,
            character_context=char_context,
        )
        if enriched and isinstance(enriched, str):
            return enriched.strip()
    except Exception as exc:
        print(
            f"[SceneImage] enhance_image_prompt failed: "
            f"{type(exc).__name__}: {exc} — falling back to seed"
        )
    return seed.strip()


# ── Public runner ─────────────────────────────────────────────────────────────

async def run_scene_image(
    *,
    bot_message: discord.Message,
    channel: discord.abc.Messageable,
    channel_id,
    trigger: str,
    hint_prompt: Optional[str] = None,
    seed_override: Optional[str] = None,
    acker: Optional[Callable[[str], Awaitable[None]]] = None,
) -> None:
    """Generate a scene image and edit it onto `bot_message` in place.

    Args:
        bot_message:    The bot's own message to attach the image onto. Must
                        already be sent (not a Webhook return). The image is
                        attached via `.edit(attachments=[...])`.
        channel:        For typing context.
        channel_id:     Stringified channel id (used for cooldown + dedup).
        trigger:        One of "button" | "scene_tag" | "intent_reroute"
                        | "image_tag_reroute" — diagnostic only.
        hint_prompt:    Optional seed text. For [IMAGE: ...] re-route this is
                        the marker body. For intent re-route it's the user
                        message text. For bare [SCENE] / button it's None
                        and we derive from `bot_message.content`. Combined
                        with the bot prose unless `seed_override` is set.
        seed_override:  When the model emitted `[SCENE: short cinematic
                        description]`, the body comes through here and is
                        used VERBATIM as the prompt seed — the bot prose is
                        skipped because the model already handed us the
                        polished prompt it had in mind. Mutually exclusive
                        with the prose-derive path; takes precedence over
                        `hint_prompt` and the bot-message text.
        acker:          Optional async ephemeral-acknowledger (button path).
    """
    key = (str(channel_id), bot_message.id)

    # In-flight dedup — every trigger consults the same set so a click
    # cannot race with a [SCENE] auto-trigger on the same target.
    if key in _inflight:
        if acker is not None:
            try:
                await acker("⏳ 場景圖片正在生成中，請稍候…")
            except Exception:
                pass
        return

    # Per-channel cooldown
    bucket = _cooldown.get_bucket(bot_message)
    if bucket is not None:
        retry_after = bucket.update_rate_limit()
        if retry_after:
            if acker is not None:
                try:
                    await acker(f"⏳ 此頻道生成過於頻繁，請 {retry_after:.0f} 秒後再試。")
                except Exception:
                    pass
            return

    if not image_dispatch.image_ready():
        if acker is not None:
            try:
                await acker("⚠️ 尚未設定圖像生成後端，無法生成場景圖片。")
            except Exception:
                pass
        return

    _inflight.add(key)
    try:
        char = database.get_character()
        bot_name = (char.get("name") or "").strip()
        looks = (char.get("looks") or "").strip()

        # 1. Prompt seed
        # When the model emitted `[SCENE: short cinematic description]`, the
        # body lands in `seed_override` and is used VERBATIM — the bot's
        # reply prose is intentionally skipped because the model already
        # handed us the polished image-prompt it had in mind, and mixing in
        # the surrounding prose would dilute that signal.
        # Otherwise: hint_prompt (if any) + bot_message.content, with a
        # generic fallback if both are empty.
        if seed_override and seed_override.strip():
            seed = seed_override.strip()
        else:
            seed_parts = []
            if hint_prompt:
                seed_parts.append(hint_prompt.strip())
            # Always append bot's own message text so derived scene matches the prose
            body = (bot_message.content or "").strip()
            if body:
                seed_parts.append(body)
            seed = "  ".join(p for p in seed_parts if p) or "cinematic scene"

        enriched = await _derive_prompt(seed, looks, bot_name)
        enriched = enriched.rstrip(",. \n") + CINEMATIC_SUFFIX

        # 2. Engine-tier ref assembly
        backend = os.environ.get("IMAGE_BACKEND", "cloudflare").lower()
        engine = os.environ.get("COMFYUI_ENGINE", "qwen").lower()

        gen_kwargs: dict = {}
        if backend == "comfyui" and engine == "qwen":
            refs, subjects, appearances = _gather_refs(seed)
            if refs:
                gen_kwargs["reference_images"] = refs
                gen_kwargs["reference_subjects"] = subjects
                gen_kwargs["subject_appearances"] = appearances
        elif backend == "comfyui":
            # FLUX — single-ref char only, no KB interleaving
            refs, subjects, _ = _gather_refs(seed)
            if refs:
                gen_kwargs["reference_image"] = refs[0]
                gen_kwargs["reference_images"] = refs[:1]
                gen_kwargs["reference_subjects"] = subjects[:1]
        elif backend in ("local_diffusers", "hf_spaces"):
            refs, _, _ = _gather_refs(seed)
            if refs:
                gen_kwargs["reference_image"] = refs[0]
        # cloudflare: text-only, no refs

        # 3. Run generation with the shared progress bar
        # `_format_diffuser_progress` lives in bot.py — lazy import to avoid
        # a circular module load at import time.
        try:
            from bot import _format_diffuser_progress as _fmt_progress
        except Exception:
            _fmt_progress = lambda tag, name="": None  # type: ignore

        progress_enabled = backend in ("local_diffusers", "comfyui")

        result = None
        async with progress_bar(
            bot_message,
            bot_name or "場景",
            _fmt_progress,
            enabled=progress_enabled,
        ) as on_progress:
            if on_progress is not None:
                gen_kwargs["on_progress"] = on_progress
            try:
                async with channel.typing():
                    result = await image_dispatch.generate_image(
                        enriched, **gen_kwargs
                    )
            except Exception as exc:
                print(
                    f"[SceneImage] generate_image raised "
                    f"{type(exc).__name__}: {exc} (trigger={trigger})"
                )
                result = None

        # 4. Attach in place
        if (
            result
            and isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], bytes)
        ):
            img_bytes, mime = result
            ext = mime.split("/")[-1] if isinstance(mime, str) and "/" in mime else "png"
            try:
                file = discord.File(io.BytesIO(img_bytes), filename=f"scene.{ext}")
                await bot_message.edit(attachments=[file])
                if acker is not None:
                    try:
                        await acker("🎬 場景已生成 ✓")
                    except Exception:
                        pass
                return
            except discord.HTTPException as exc:
                # Strict edit-in-place contract: scene mode must NEVER post a
                # separate image message. If the in-place edit fails (rare —
                # message deleted, lost permissions, etc.) we report the
                # failure and stop. This is the central no-doubled-images
                # guarantee — never weaken it with a `channel.send` fallback.
                print(
                    f"[SceneImage] edit(attachments=) failed: {exc} — "
                    f"NOT falling back to a new message (strict in-place rule)."
                )
                if acker is not None:
                    try:
                        await acker(
                            "⚠️ 無法將場景圖片附加到原訊息 "
                            f"(`{type(exc).__name__}`)，已取消以避免重複貼圖。"
                        )
                    except Exception:
                        pass
                return

        # 5. Failure
        print(f"[SceneImage] no image produced for trigger={trigger}")
        if acker is not None:
            try:
                await acker("⚠️ 場景圖片生成失敗，請稍後再試。")
            except Exception:
                pass
    finally:
        _inflight.discard(key)


# ── Button click handler (called from views.SceneImageButtonView) ─────────────

async def handle_button_click(interaction: discord.Interaction) -> None:
    """Persistent-view button callback — defers, then dispatches the runner."""
    msg = interaction.message
    channel = interaction.channel
    if msg is None or channel is None:
        try:
            await interaction.response.send_message(
                "⚠️ 無法辨識此訊息。", ephemeral=True
            )
        except discord.HTTPException:
            pass
        return

    # Defer ephemerally so the user sees an immediate ack
    try:
        await interaction.response.defer(ephemeral=True, thinking=False)
    except discord.HTTPException:
        pass

    async def _ack(text: str) -> None:
        try:
            await interaction.followup.send(text, ephemeral=True)
        except discord.HTTPException:
            pass

    await run_scene_image(
        bot_message=msg,
        channel=channel,
        channel_id=str(channel.id),
        trigger="button",
        hint_prompt=None,
        acker=_ack,
    )
