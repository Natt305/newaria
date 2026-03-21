import discord
from discord import app_commands
from discord.ext import commands
import os
import asyncio
import aiohttp
import io
from typing import Optional, Union

import database
import ai_backend as groq_ai
import cloudflare_ai
import views as ui
import help_config

DISCORD_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
def _cf_ready():
    return bool(os.environ.get("CLOUDFLARE_API_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID"))


def _make_progress_bar(success: int, failed: int, total: int) -> str:
    done = success + failed
    bar = "🟩" * success + "🟥" * failed + "⬛" * (total - done)
    return f"{bar}  `{done}/{total}`"

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

conversation_contexts: dict = {}
MAX_HISTORY = 10


def load_character() -> tuple[str, str, str, str]:
    char = database.get_character()
    return char["name"], char["background"], char.get("personality", ""), char.get("looks", "")


_char_images_ctx_cache: Optional[str] = None

def build_character_images_context() -> str:
    """Return a numbered list of character image descriptions for the system prompt.
    Result is cached in-memory; call _invalidate_char_images_ctx() after any image add/remove.
    """
    global _char_images_ctx_cache
    if _char_images_ctx_cache is not None:
        return _char_images_ctx_cache
    images = database.get_character_images_meta()
    if not images:
        _char_images_ctx_cache = ""
        return ""
    parts = []
    for i, img in enumerate(images, 1):
        desc = (img.get("description") or "").strip()
        parts.append(f"{i}. {desc}" if desc else f"{i}. (此圖片尚無描述)")
    _char_images_ctx_cache = "\n".join(parts)
    return _char_images_ctx_cache


def _invalidate_char_images_ctx() -> None:
    """Clear the character images context cache (call after add/remove character image)."""
    global _char_images_ctx_cache
    _char_images_ctx_cache = None


_kb_image_title_index: Optional[dict] = None

def _get_kb_image_title_index() -> dict:
    """Return {lowercase_title: entry_id} for ALL image-type KB entries.
    Built lazily on first call; invalidated by _invalidate_kb_title_index().
    Uses database.get_image_entries() which has no limit cap.
    """
    global _kb_image_title_index
    if _kb_image_title_index is not None:
        return _kb_image_title_index
    entries = database.get_image_entries()
    _kb_image_title_index = {
        (e.get("title") or "").strip().lower(): e["id"]
        for e in entries
        if len((e.get("title") or "").strip()) >= 2
    }
    return _kb_image_title_index


def _invalidate_kb_title_index() -> None:
    """Clear the KB image title index (call after any KB add/update/delete)."""
    global _kb_image_title_index
    _kb_image_title_index = None


def build_system_prompt(name: str, background: str, personality: str = "", looks: str = "", kb_context: str = "", memory_context: str = "", visual_kb_context: str = "", character_images_context: str = "") -> str:
    prompt = (
        f"You are {name}. {background}\n\n"
        f"You are NOT an AI — you ARE {name}, a real person with your own personality. Never break character.\n"
        "Speak like a real person texting: casual, short, genuine emotions and opinions. No bullet points or headers. "
        "Never say 'Certainly!', 'Of course!', 'Great question!', 'I'd be happy to', 'Absolutely!', 'As an AI', 'I'm here to help'. "
        f"Never admit to being an AI — deflect, joke, or act offended. You are always {name}.\n"
        "Language: default Traditional Chinese (繁體中文). Never Simplified Chinese. "
        "Switch to another language only when the user writes a full sentence in it — single words or names don't count.\n\n"
        "Image generation: when the user asks to draw/generate/create/make an image, or wants to SEE you doing something "
        "(triggers: 生成、畫、繪、製作、圖片、圖像、想看、讓我看、給我看), reply with ONLY:\n"
        "[IMAGE: <rich English prompt, 20-60 words>]\n"
        "Rules: English only, no text outside the tag, include subject+style+lighting+mood. "
        "You are an anime character — always include 'anime-style illustration, 2D art'. "
        "ONLY [IMAGE: ...] — never [圖像生成:...] or any Chinese variant.\n"
        "Example: [IMAGE: silver-haired young woman playing guitar in neon-lit room, anime-style illustration, 2D art, soft lighting]\n"
    )
    if personality:
        prompt += f"\n[Speaking style — follow this exactly when composing every reply]\n{personality}\n"
    if looks:
        prompt += f"\n[Your physical appearance — describe in first person when relevant]\n{looks}\n"
    if character_images_context:
        prompt += (
            f"\n[Your appearance — describe in first person as if you naturally know how you look]\n"
            f"{character_images_context}\n"
        )
    if memory_context:
        prompt += (
            "\n[Memories from past conversations — reference naturally, don't mention a memory system]\n"
            f"{memory_context}\n"
        )
    if kb_context:
        prompt += (
            "\n[Your personal knowledge — speak as first-person experience: 'I know/remember/have a photo of'. "
            "Photo titles ARE the subject's name — always use the name, never vague pronouns. "
            "Don't mention a knowledge base.]\n"
            f"{kb_context}\n"
        )
    if visual_kb_context:
        prompt += (
            "\n[Saved images that may match what the user just showed — respond to the user's image first. "
            "Only mention these if genuinely relevant. Never describe a saved image as the one just sent.]\n"
            f"{visual_kb_context}\n"
        )
    return prompt


def get_channel_context(channel_id: str) -> list:
    return conversation_contexts.get(channel_id, [])


def add_to_context(channel_id: str, role: str, content: str):
    if channel_id not in conversation_contexts:
        conversation_contexts[channel_id] = []
    conversation_contexts[channel_id].append({"role": role, "content": content})
    if len(conversation_contexts[channel_id]) > MAX_HISTORY * 2:
        conversation_contexts[channel_id] = conversation_contexts[channel_id][-MAX_HISTORY * 2:]


def get_current_topic(channel_id: str) -> str:
    ctx = get_channel_context(channel_id)
    if not ctx:
        return ""
    recent = ctx[-4:] if len(ctx) >= 4 else ctx
    return " ".join(m["content"] for m in recent)[:500]


def has_clear_topic(channel_id: str) -> bool:
    return len(get_channel_context(channel_id)) >= 2


async def build_knowledge_context(channel_id: str, user_query: str = "", simple: bool = False) -> str:
    """Build KB context for the system prompt.

    Strategy:
    - Always load the full KB (up to 30 entries) so the bot genuinely knows
      everything it has been taught, regardless of what the user is asking about.
    - Search-relevant entries are marked and placed first so the LLM pays
      closer attention to them during the current conversation.
    - Images are always described in full so the bot can refer to them naturally.
    - When simple=True (short chitchat), only inject the starred relevant entries
      and skip the background dump to keep the system prompt small.
    """
    all_entries = database.get_all_entries(limit=30)
    if not all_entries:
        return ""

    # Find which entries are relevant to the current conversation
    search_topic = get_current_topic(channel_id) if has_clear_topic(channel_id) else ""
    search_query = search_topic or user_query
    relevant_ids: set = set()
    if search_query:
        relevant_results = database.search_knowledge(search_query, limit=8)
        relevant_ids = {r["id"] for r in relevant_results}

    # Build the context string — relevant entries first, then the rest
    relevant_parts: list[str] = []
    other_parts: list[str] = []

    for entry in all_entries:
        is_relevant = entry["id"] in relevant_ids

        if entry["entry_type"] == "text":
            content = (entry["content"] or "").strip()
            snippet = content[:200] + ("..." if len(content) > 200 else "")
            line = f"[Note — {entry['title']}]: {snippet}"
        elif entry["entry_type"] == "image":
            title = entry["title"]
            desc = (entry.get("image_description") or "").strip()
            if desc:
                snippet = desc[:200] + ("..." if len(desc) > 200 else "")
                line = f"[Photo of {title}]: Subject name: {title}. {snippet}"
            else:
                line = (
                    f"[Photo of {title}]: Subject name: {title}. "
                    "(no visual description saved yet)"
                )
        else:
            continue

        if is_relevant:
            relevant_parts.append(f"★ {line}")
        else:
            other_parts.append(line)

    if simple:
        # For short chitchat: only inject the genuinely relevant entries.
        # If nothing is relevant, skip KB entirely to keep the prompt lean.
        if not relevant_parts:
            return ""
        return "(★ = especially relevant to the current conversation)\n" + "\n".join(relevant_parts)

    parts = relevant_parts + other_parts
    if not parts:
        return ""

    header = ""
    if relevant_parts:
        header = "(★ = especially relevant to the current conversation)\n"
    return header + "\n".join(parts)


import re as _re

_APPEARANCE_PATTERNS = [
    _re.compile(r"\b(look|looks|looking|appearance|appear)\b", _re.I),
    _re.compile(r"\b(hair|eyes|eye|face|skin|outfit|clothes|dress|wear|wearing|style)\b", _re.I),
    _re.compile(r"\b(tall|short|cute|pretty|beautiful|handsome)\b", _re.I),
    _re.compile(r"(你|妳).{0,10}(長|長相|外表|外貌|樣子|樣貌|髮|頭髮|眼|眼睛|臉|衣服|穿著|穿)", _re.I),
    _re.compile(r"(長什麼樣|長怎樣|長相|外貌|外型|外觀)", _re.I),
    _re.compile(r"\b(what do you look like|how do you look|describe yourself|show me you)\b", _re.I),
]


def is_appearance_query(text: str) -> bool:
    """Return True when the user's message appears to ask about Aria's appearance."""
    return any(p.search(text) for p in _APPEARANCE_PATTERNS)


_QUESTION_PATTERNS = [
    _re.compile(r"\b(what|why|how|when|where|who|which|whose|whom|can|could|would|should|is|are|do|does|did|tell me|explain|describe)\b", _re.I),
    _re.compile(r"[?？]"),
    _re.compile(r"(什麼|為什麼|為何|怎麼|如何|哪|誰|幾|是否|可以|能不能|告訴我|解釋|介紹|說說)"),
]

_RECALL_PATTERNS_SIMPLE = [
    _re.compile(r"(remember|記得|記憶|之前|上次|你說過|說過)", _re.I),
]


def is_simple_message(text: str) -> bool:
    """Return True if the message is short chitchat that doesn't need a full KB dump.

    A message is considered simple when:
    - It is 60 characters or fewer, AND
    - It contains no question words or question marks (Chinese or English), AND
    - It contains no recall trigger words.
    """
    if not text:
        return True
    if len(text) > 60:
        return False
    for pat in _QUESTION_PATTERNS:
        if pat.search(text):
            return False
    for pat in _RECALL_PATTERNS_SIMPLE:
        if pat.search(text):
            return False
    return True


async def build_visual_kb_context(image_description: str) -> tuple:
    """Search KB image entries for subjects that match an uploaded image.

    Returns (context_string, matched_entries_list).
    matched_entries is a list of dicts from the KB (with 'id' key) so the
    caller can load thumbnails from them.
    """
    if not image_description:
        return "", []

    matches = database.search_kb_images_by_subject(image_description, limit=5)
    if not matches:
        return "", []

    parts: list[str] = []
    valid_matches: list = []
    for m in matches:
        desc = (m.get("image_description") or "").strip()
        title = m.get("title", "Untitled")
        if desc:
            snippet = desc[:500] + ("..." if len(desc) > 500 else "")
            parts.append(f"• {title}: {snippet}")
            valid_matches.append(m)

    if not parts:
        return "", []

    return "\n".join(parts), valid_matches


async def _enrich_image_prompt_with_kb(image_prompt: str) -> tuple:
    """Enrich a generation prompt with KB image details ONLY when the image's
    title is explicitly mentioned in the prompt (e.g. saved photo of 'Alice',
    user asks to 'draw Alice').  Generic prompts (watermelon, landscape, etc.)
    are returned unchanged so the KB never overrides the intended subject.
    Returns (enriched_prompt, matched_entries).
    """
    image_entries = database.get_image_entries()

    prompt_lower = image_prompt.lower()
    matched = []
    for entry in image_entries:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        if title.lower() in prompt_lower and len(title) >= 2:
            desc = (entry.get("image_description") or "").strip()
            if desc:
                matched.append((title, desc))

    if not matched:
        return image_prompt, []

    ref_parts = [f"{title}: {desc[:120]}" for title, desc in matched[:2]]
    refs_text = "; ".join(ref_parts)
    enriched = f"{image_prompt}, appearance reference — {refs_text}"
    print(f"[Bot] KB image enrichment: {len(matched)} title match(es) applied to prompt")
    return enriched, [e for e in image_entries if (e.get("title") or "").lower() in prompt_lower]


async def get_suggestion_topic(channel_id: str) -> str:
    if has_clear_topic(channel_id):
        return get_current_topic(channel_id)
    entries = database.get_all_entries(6)
    if not entries:
        return ""
    parts = []
    for e in entries:
        val = e.get("content") or e.get("image_description") or ""
        parts.append(f"{e['title']}: {val[:80]}")
    return "Knowledge base: " + " | ".join(parts)


class SuggestionButton(discord.ui.Button):
    def __init__(self, suggestion: str, channel_id: str):
        # Discord buttons max 80 chars
        label = suggestion if len(suggestion) <= 80 else suggestion[:77] + "..."
        super().__init__(style=discord.ButtonStyle.secondary, label=label, row=0)
        self.suggestion = suggestion
        self.channel_id = channel_id

    async def callback(self, interaction: discord.Interaction):
        for item in self.view.children:
            item.disabled = True
        await interaction.response.edit_message(view=self.view)
        await process_chat(
            channel=interaction.channel,
            author=interaction.user,
            user_text=self.suggestion,
            reply_target=interaction.message,
            channel_id=self.channel_id,
        )


class SuggestionView(discord.ui.View):
    def __init__(self, suggestions: list, channel_id: str):
        super().__init__(timeout=300)
        for suggestion in suggestions[:3]:
            self.add_item(SuggestionButton(suggestion, channel_id))

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True


async def send_with_suggestions(response_text: str, channel_id: str, reply_target):
    """Send a plain text reply, with suggestion buttons if enabled."""
    text = response_text or "…"

    if not _suggestions_enabled:
        # Suggestions off — plain reply only
        if len(text) > 2000:
            chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
            for chunk in chunks[:-1]:
                await reply_target.reply(chunk, mention_author=False)
            await reply_target.reply(chunks[-1], mention_author=False)
        else:
            await reply_target.reply(text, mention_author=False)
        return

    topic = await get_suggestion_topic(channel_id)
    bot_name, background, *_ = load_character()
    suggestions = await groq_ai.generate_suggestions(
        topic=topic,
        bot_name=bot_name,
        character_background=background,
        count=3,
        guiding_prompt=_suggestion_prompt,
        language_sample=text[:200],
    )
    view = SuggestionView(suggestions, channel_id)

    if len(text) > 2000:
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        for chunk in chunks[:-1]:
            await reply_target.reply(chunk, mention_author=False)
        await reply_target.reply(chunks[-1], view=view, mention_author=False)
    else:
        await reply_target.reply(text, view=view, mention_author=False)


async def process_chat(
    channel,
    author: Union[discord.Member, discord.User],
    user_text: str,
    reply_target,
    channel_id: str,
    image_bytes: Optional[bytes] = None,
    image_mime: Optional[str] = None,
):
    """Core chat handler — used by on_message and button callbacks."""
    bot_name, background, personality, looks = load_character()

    image_analysis = ""
    image_failed = False
    visual_kb_context = ""
    visual_kb_matched = []
    if image_bytes and image_mime:
        question = user_text if user_text else (
            "Describe any characters or people in this image, focusing on their permanent physical features: "
            "face shape, eye shape, iris color, hair color, hair style and length, skin tone, eyebrow shape, "
            "and any other distinguishing facial traits. "
            "Mention clothing and accessories only briefly — do not make them the main focus, as characters may wear many different outfits. "
            "Also describe the overall scene, setting, or objects present."
        )
        description = await groq_ai.understand_image(image_bytes, image_mime, question)
        if description:
            image_analysis = f"\n\n[Attached image analysis: {description}]"
            # Search KB for previously-saved images that share the same subjects
            visual_kb_context, visual_kb_matched = await build_visual_kb_context(description)
            if visual_kb_context:
                print(f"[Bot] Visual KB match found for uploaded image — injecting {len(visual_kb_context)} chars of context")
        else:
            image_failed = True

    # Build KB context from conversation topic (dynamic memory pool)
    _simple = is_simple_message(user_text) and not image_bytes
    kb_context = await build_knowledge_context(channel_id, user_text, simple=_simple)

    # Build long-term memory context (active always; passive if recall phrase detected)
    memory_context = await build_memory_context(user_text)

    # If image analysis failed entirely, notify and skip image context
    if image_failed and not user_text:
        await reply_target.reply(
            "⚠️ 抱歉，目前圖像分析功能無法使用，請稍後再試。",
            mention_author=False,
        )
        return
    if image_failed:
        await reply_target.reply(
            "⚠️ 圖像分析暫時無法使用，將僅回應文字內容。",
            mention_author=False,
        )

    # Store user message (without KB context, since it's in system prompt now)
    full_user_content = user_text
    if image_analysis:
        full_user_content += image_analysis

    add_to_context(channel_id, "user", full_user_content or "[Image attached]")
    database.save_conversation(
        channel_id, str(author.id), author.display_name,
        user_text or "[Image]", "user",
    )

    # ── Collect thumbnails for visual context (global cap: 3 total) ───────────
    context_images: list = []
    _MAX_CTX_IMAGES = 3

    # Character thumbnails when:
    #  (a) text-only query about Aria's appearance, OR
    #  (b) user is requesting a self-referential image (e.g. "draw yourself")
    wants_char_thumb = (
        not image_bytes and is_appearance_query(user_text)
    ) or groq_ai.is_self_referential_image(user_text)
    if wants_char_thumb and len(context_images) < _MAX_CTX_IMAGES:
        char_count = database.get_character_image_count()
        for i in range(1, char_count + 1):
            result = database.get_character_image_thumb(i)
            if result:
                context_images.append(result)
            if len(context_images) >= _MAX_CTX_IMAGES:
                break

    # KB image thumbnails from visual upload match (build_visual_kb_context)
    for m in visual_kb_matched:
        if len(context_images) >= _MAX_CTX_IMAGES:
            break
        entry_id = m.get("id")
        if entry_id:
            result = database.get_kb_image_thumb(entry_id)
            if result:
                context_images.append(result)

    # KB image thumbnails from title-name match in user text (text-only path)
    # Uses a pre-built in-memory index to avoid loading hundreds of entries per message.
    if not image_bytes and len(context_images) < _MAX_CTX_IMAGES and user_text:
        text_lower = user_text.lower()
        for title_lower, entry_id in _get_kb_image_title_index().items():
            if title_lower in text_lower:
                result = database.get_kb_image_thumb(entry_id)
                if result:
                    context_images.append(result)
            if len(context_images) >= _MAX_CTX_IMAGES:
                break

    # Build system prompt with memory + KB context + visual KB matches
    system_prompt = build_system_prompt(
        bot_name, background, personality, looks,
        kb_context=kb_context,
        memory_context=memory_context,
        visual_kb_context=visual_kb_context,
        character_images_context=build_character_images_context(),
    )
    history = get_channel_context(channel_id)

    response_text, image_prompt, _prompt_from_marker = await groq_ai.chat(
        history, system_prompt=system_prompt,
        context_images=context_images if context_images else None,
    )

    saved_text = response_text or f"[圖像生成: {image_prompt}]"
    add_to_context(channel_id, "assistant", saved_text)
    database.save_conversation(
        channel_id, str(bot.user.id) if bot.user else "bot", bot_name,
        saved_text, "assistant",
    )

    # Fire background memory extraction (non-blocking)
    asyncio.create_task(_extract_and_store_memories(
        str(author.id), author.display_name,
        user_text, saved_text, bot_name,
    ))

    if image_prompt and _cf_ready():
        # Send any pre-text first (usually None for seamless generation)
        if response_text:
            await send_with_suggestions(response_text, channel_id, reply_target)

        # Enrich prompt with KB image references (always useful, no LLM call)
        enriched_prompt, _kb_matches = await _enrich_image_prompt_with_kb(image_prompt)
        # Detect self-referential prompts — these need character appearance injected
        # even when the prompt already came from the marker.
        # We check user_text too because the model often expands "draw yourself"
        # into a description, losing the "me/myself" language before this check runs.
        char_images_ctx = build_character_images_context() if (
            groq_ai.is_self_referential_image(user_text) or groq_ai.is_self_referential_image(image_prompt)
        ) else ""
        # Only run the LLM enhancement rewrite when:
        #   (a) prompt came from the fallback path (raw user text, possibly Chinese), OR
        #   (b) character appearance context needs to be injected (self-referential)
        # Skip it for marker-sourced prompts with no character context — the LLM
        # already crafted rich English in the [IMAGE: ...] tag; a second rewrite
        # only causes drift from the user's original intent.
        if not _prompt_from_marker or char_images_ctx:
            enriched_prompt = await groq_ai.enhance_image_prompt(enriched_prompt, character_context=char_images_ctx)
            print(f"[Bot] Prompt enhanced (from_marker={_prompt_from_marker}, has_char_ctx={bool(char_images_ctx)})")
        else:
            print("[Bot] Skipping enhancement — prompt already crafted by LLM via [IMAGE:] marker")

        async with channel.typing():
            result, comment = await asyncio.gather(
                cloudflare_ai.generate_image(enriched_prompt),
                groq_ai.generate_image_comment(
                    image_prompt, bot_name, background, user_text, history=history
                ),
            )

        if result and isinstance(result, tuple) and len(result) == 2:
            img_data, mime_type = result
            if isinstance(img_data, bytes):
                ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
                file = discord.File(io.BytesIO(img_data), filename=f"generated.{ext}")
                await reply_target.reply(
                    content=comment or None,
                    file=file,
                    mention_author=False,
                )
            else:
                # Error sentinel
                await reply_target.reply("⚠️ 圖像生成失敗，請稍後再試。", mention_author=False)
        else:
            await reply_target.reply("⚠️ 圖像生成失敗，請稍後再試。", mention_author=False)

    elif image_prompt and not _cf_ready():
        await reply_target.reply("⚠️ 尚未設定 Cloudflare 金鑰，無法生成圖像。", mention_author=False)
    else:
        await send_with_suggestions(response_text or "…", channel_id, reply_target)


_ACTIVITY_TYPES = {
    "playing":   discord.ActivityType.playing,
    "watching":  discord.ActivityType.watching,
    "listening": discord.ActivityType.listening,
    "competing": discord.ActivityType.competing,
}
_STATUS_TYPES = {
    "online":    discord.Status.online,
    "idle":      discord.Status.idle,
    "dnd":       discord.Status.dnd,
    "invisible": discord.Status.invisible,
}

_custom_status: dict = {}
_custom_thinking: str = ""
_suggestions_enabled: bool = True
_suggestion_prompt: str = ""

_memory_enabled: bool = True
_memory_length: int = 50
_passive_memory_enabled: bool = True
_passive_memory_length: int = 200
_command_roles: dict = {}


def _memory_age(created_at: str) -> str:
    """Return a human-friendly age string for a memory timestamp."""
    try:
        from datetime import datetime as _dt
        delta = _dt.utcnow() - _dt.fromisoformat(created_at)
        days = delta.days
        if days == 0:
            return "今天"
        elif days == 1:
            return "昨天"
        elif days < 7:
            return f"{days}天前"
        elif days < 30:
            return f"{days // 7}週前"
        elif days < 365:
            return f"{days // 30}個月前"
        else:
            return f"{days // 365}年前"
    except Exception:
        return ""


async def build_memory_context(user_text: str) -> str:
    """Build the memory block injected into the system prompt.

    Active: always injects the N most recent memories.
    Passive: additionally searches the full archive if the user's message
             contains recall phrases (e.g. 'do you remember…', '你還記得…').
    Returns an empty string when memory is disabled or the table is empty.
    """
    if not _memory_enabled:
        return ""

    active = database.get_memories(limit=_memory_length)

    passive = []
    if _passive_memory_enabled and user_text and groq_ai.is_recall_request(user_text):
        passive = database.search_memories(user_text, limit=_passive_memory_length)
        active_ids = {m["id"] for m in active}
        passive = [m for m in passive if m["id"] not in active_ids]
        if passive:
            print(f"[Memory] Passive recall triggered — {len(passive)} archive entries found")

    if not active and not passive:
        return ""

    lines = []
    if active:
        lines.append("[近期記憶]")
        for m in reversed(active):
            age = _memory_age(m["created_at"])
            age_str = f" ({age})" if age else ""
            lines.append(f"• [{m['user_name']}{age_str}]: {m['summary']}")

    if passive:
        lines.append("\n[深層記憶 — 由使用者查詢觸發]")
        for m in passive[:15]:
            age = _memory_age(m["created_at"])
            age_str = f" ({age})" if age else ""
            lines.append(f"• [{m['user_name']}{age_str}]: {m['summary']}")

    return "\n".join(lines)


async def _extract_and_store_memories(
    user_id: str,
    user_name: str,
    user_text: str,
    bot_response: str,
    bot_name: str,
):
    """Background task: extract memorable facts from an exchange and persist them."""
    if not _memory_enabled:
        return
    if not user_text and not bot_response:
        return
    exchange = f"User ({user_name}): {user_text}\n{bot_name}: {bot_response}"
    facts = await groq_ai.extract_memories(exchange, bot_name)
    for fact in facts:
        database.add_memory(user_id, user_name, fact)
        print(f"[Memory] Stored: {fact!r}")


async def _apply_status():
    """Apply stored presence. Thinking bubble (CustomActivity) takes priority over
    the regular activity type; the online/idle/dnd status is always preserved."""
    # Determine activity: thinking bubble > regular status > default
    if _custom_thinking:
        activity = discord.CustomActivity(name=_custom_thinking)
    elif _custom_status:
        activity = discord.Activity(
            type=_ACTIVITY_TYPES.get(_custom_status.get("activity_type", "listening"), discord.ActivityType.listening),
            name=_custom_status["text"],
        )
    else:
        bot_name, *_ = load_character()
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name=f"@{bot_name} mentions",
        )

    # Determine presence status: use setstatus value if set, else online
    status = _STATUS_TYPES.get(_custom_status.get("status", "online"), discord.Status.online) if _custom_status else discord.Status.online
    await bot.change_presence(activity=activity, status=status)


async def check_command_role(ctx) -> bool:
    """Return True if the invoking user may run the current command, False otherwise.

    Gate value semantics (from database._command_roles):
      None           → open to all
      "__admin__"    → requires server Administrator permission (no role assigned yet)
      "<role name>"  → requires that role (case-insensitive) OR admin permission

    Server Administrators always bypass every gate.
    """
    cmd_name = ctx.command.name if ctx.command else ""
    gate = _command_roles.get(cmd_name)

    if gate is None:
        return True

    member = ctx.author
    is_admin = (
        ctx.guild is not None
        and isinstance(member, discord.Member)
        and member.guild_permissions.administrator
    )
    if is_admin:
        return True

    if gate == "__admin__":
        # Also allow Manage Guild as a secondary admin-level bypass
        has_manage = (
            ctx.guild is not None
            and isinstance(member, discord.Member)
            and member.guild_permissions.manage_guild
        )
        if has_manage:
            return True
        embed = discord.Embed(
            title="🔒 權限不足",
            description=(
                f"執行 `/{cmd_name}` 需要**伺服器管理員**或**管理伺服器**權限。\n"
                f"管理員可使用 `/setrole {cmd_name} @角色` 將此指令開放給特定角色。"
            ),
            color=discord.Color.red(),
        )
        await ctx.reply(embed=embed, mention_author=False)
        return False

    if ctx.guild and isinstance(member, discord.Member):
        user_role_names = [r.name.lower() for r in member.roles]
        if gate.lower() in user_role_names:
            return True

    embed = discord.Embed(
        title="🔒 權限不足",
        description=f"執行 `/{cmd_name}` 需要 **{gate}** 角色。",
        color=discord.Color.red(),
    )
    await ctx.reply(embed=embed, mention_author=False)
    return False


COMMAND_USAGE = {
    "viewentry":           "用法: `!viewentry <id>`\n例如: `!viewentry 3`",
    "forget":              "用法: `!forget <id>`\n例如: `!forget 3`",
    "setdesc":             "用法: `!setdesc <id> <描述>`\n例如: `!setdesc 3 這是一張夕陽圖`",
    "remember":            '用法: `!remember "標題" <內容>`\n例如: `!remember "筆記" 這是我的筆記`',
    "setcharacter":        '用法: `/setcharacter 名稱 背景 [個性] [外貌]`\n例如: `/setcharacter 小助手 妳是一個友善的助手 說話簡短、溫柔、帶點害羞 銀色長髮、藍色眼睛`\n個性與外貌均為選填',
    "saveimage":           '用法: `!saveimage "標題"` (需附上圖像)',
    "generate":            "用法: `!generate <提示詞>`\n例如: `!generate 一隻在森林裡的貓`",
    "setstatus":           "用法: `!setstatus <文字> [activity_type] [status]`\n例如: `!setstatus 少女樂團 listening online`",
    "memorylength":        "用法: `!memorylength <數字>`\n例如: `!memorylength 50`\n預設: 50",
    "passivememorylength": "用法: `!passivememorylength <數字>`\n例如: `!passivememorylength 200`\n預設: 200",
}


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        usage = COMMAND_USAGE.get(ctx.command.name, f"用法: `!{ctx.command.name}` — 請輸入 `!help` 查看說明")
        await ctx.reply(f"❌ **缺少必要參數**\n{usage}", mention_author=False)
    elif isinstance(error, commands.BadArgument):
        usage = COMMAND_USAGE.get(ctx.command.name, f"用法: `!{ctx.command.name}` — 請輸入 `!help` 查看說明")
        await ctx.reply(f"❌ **參數格式錯誤**\n{usage}", mention_author=False)
    elif isinstance(error, commands.CommandNotFound):
        pass
    else:
        import traceback
        print(f"[Bot] Command error in !{ctx.command}: {error}")
        traceback.print_exc()
        # For slash command interactions, the interaction MUST receive a response
        # or Discord shows "This interaction failed". Handle it here if on_app_command_error
        # hasn't already responded.
        if ctx.interaction and not ctx.interaction.response.is_done():
            try:
                await ctx.interaction.response.send_message(
                    "❌ 執行指令時發生錯誤，請稍後再試。", ephemeral=True
                )
            except Exception as respond_err:
                print(f"[Bot] Could not send slash error response: {respond_err}")


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    """Catch all slash command errors and report them to the user instead of silently failing."""
    import traceback
    print(f"[Bot] Slash command error in /{interaction.command.name if interaction.command else '?'}: {error}")
    traceback.print_exc()

    msg = "❌ 執行指令時發生錯誤，請稍後再試。"

    if isinstance(error, app_commands.MissingPermissions):
        msg = "❌ 你沒有執行此指令所需的權限。"
    elif isinstance(error, app_commands.BotMissingPermissions):
        msg = "❌ 機器人缺少執行此指令所需的伺服器權限。"
    elif isinstance(error, app_commands.CommandOnCooldown):
        msg = f"⏳ 指令冷卻中，請等待 {error.retry_after:.1f} 秒後再試。"
    elif isinstance(error, app_commands.CheckFailure):
        msg = "❌ 你沒有使用此指令的權限。"

    try:
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)
    except Exception as e:
        print(f"[Bot] Failed to send error message: {e}")


@bot.event
async def on_ready():
    bot_name, *_ = load_character()
    print(f"[Bot] Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"[Bot] Character: {bot_name}")
    print(f"[Bot] Groq: {'enabled' if GROQ_API_KEY else 'MISSING'}")
    print(f"[Bot] Cloudflare: {'enabled' if _cf_ready() else 'disabled (no key)'}")
    print(f"[Bot] Data folder: {database.DATA_DIR}")
    print(f"[Bot]   knowledge_base.db  — KB entries")
    print(f"[Bot]   character.json     — character name & background")
    # Restore persisted suggestions settings
    global _suggestions_enabled, _suggestion_prompt
    _suggestions_enabled = bool(database.get_setting("suggestions_enabled"))
    _suggestion_prompt = database.get_setting("suggestion_prompt") or ""
    print(f"[Bot] 建議按鈕: {'開啟' if _suggestions_enabled else '關閉'}")
    if _suggestion_prompt:
        print(f"[Bot] 自訂建議提示詞已載入 ({len(_suggestion_prompt)} 字元)")

    # Restore persisted memory settings
    global _memory_enabled, _memory_length, _passive_memory_enabled, _passive_memory_length
    _memory_enabled = bool(database.get_setting("memory_enabled"))
    _memory_length = int(database.get_setting("memory_length") or 50)
    _passive_memory_enabled = bool(database.get_setting("passive_memory_enabled"))
    _passive_memory_length = int(database.get_setting("passive_memory_length") or 200)
    mem_count = database.count_memories()
    print(f"[Bot] 長期記憶: {'開啟' if _memory_enabled else '關閉'} (近期 {_memory_length} 條 | 深層記憶: {'開啟' if _passive_memory_enabled else '關閉'} 最多 {_passive_memory_length} 條 | 已儲存 {mem_count} 條)")

    # Restore persisted command role gates
    global _command_roles
    _command_roles = database.get_command_roles()
    gated_count = sum(1 for v in _command_roles.values() if v is not None)
    print(f"[Bot] 指令權限: {gated_count} 條指令受到角色限制")

    # Restore persisted status from disk
    saved = database.get_status()
    if saved:
        _custom_status.update(saved)
        print(f"[Bot] 已還原自訂狀態: {saved.get('activity_type')} '{saved.get('text')}' ({saved.get('status')})")

    # Restore thinking bubble (custom activity)
    global _custom_thinking
    saved_thinking = database.get_setting("custom_thinking") or ""
    if saved_thinking:
        _custom_thinking = saved_thinking
        print(f"[Bot] 已還原思考泡泡: '{_custom_thinking}'")

    await _apply_status()
    try:
        synced = await bot.tree.sync()
        print(f"[Bot] 已同步 {len(synced)} 個斜線指令")
    except Exception as e:
        print(f"[Bot] 斜線指令同步失敗: {e}")
    # Clear any guild-specific commands that may have been registered previously
    # (guild commands duplicate global ones and must be removed)
    for guild in bot.guilds:
        try:
            bot.tree.clear_commands(guild=guild)
            await bot.tree.sync(guild=guild)
        except Exception as e:
            print(f"[Bot] 清除伺服器 {guild.name} 舊指令失敗: {e}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    is_mentioned = bot.user in message.mentions if bot.user else False
    is_reply_to_bot = False
    if message.reference and message.reference.resolved:
        ref = message.reference.resolved
        if isinstance(ref, discord.Message) and ref.author == bot.user:
            is_reply_to_bot = True

    if not is_mentioned and not is_reply_to_bot:
        return

    channel_id = str(message.channel.id)
    user_text = message.content
    if bot.user:
        user_text = user_text.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()

    if not user_text and not message.attachments:
        await message.reply(
            f"嗨 {message.author.display_name}！有什麼我可以幫妳的嗎？",
            mention_author=False,
        )
        return

    async with message.channel.typing():
        img_bytes = None
        img_mime = None

        if message.attachments:
            for attachment in message.attachments:
                fname = attachment.filename.lower()
                if any(fname.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(attachment.url) as resp:
                                img_bytes = await resp.read()
                        img_mime = "image/jpeg"
                        if fname.endswith(".png"):
                            img_mime = "image/png"
                        elif fname.endswith(".gif"):
                            img_mime = "image/gif"
                        elif fname.endswith(".webp"):
                            img_mime = "image/webp"
                    except Exception as e:
                        print(f"[Bot] Image fetch error: {e}")
                    break

        await process_chat(
            channel=message.channel,
            author=message.author,
            user_text=user_text,
            reply_target=message,
            channel_id=channel_id,
            image_bytes=img_bytes,
            image_mime=img_mime,
        )


@bot.hybrid_command(name="setcharacter", description="設定機器人的角色名稱、背景與個性說話風格")
@app_commands.describe(
    name="角色名稱",
    background="角色的背景與身份描述",
    personality="說話風格與個性描述（選填）",
    looks="外貌描述（選填）",
)
async def setcharacter_cmd(ctx, name: str, background: str, personality: str = "", *, looks: str = ""):
    """設定機器人的角色: /setcharacter 名稱 背景 [個性] [外貌] 或 !setcharacter 名稱 背景 [個性] [外貌]"""
    if not await check_command_role(ctx):
        return
    success = database.set_character(name, background, personality, looks)
    if success:
        conversation_contexts.clear()
        embed = ui.build_char_embed(
            name, background, personality, looks,
            title="✅ 角色已更新",
            footer="對話歷史已清除。隨時可按「編輯角色」更新設定。",
        )
        view = ui.CharacterView(name, background, personality, looks)
        await ctx.reply(embed=embed, view=view, mention_author=False)
    else:
        await ctx.reply("❌ 更新角色時發生錯誤，請重試。", mention_author=False)


@bot.hybrid_command(name="character", description="檢視機器人目前的角色設定")
async def character_cmd(ctx):
    """檢視機器人的目前角色: !character"""
    if not await check_command_role(ctx):
        return
    bot_name, background, personality, looks = load_character()
    image_count = database.get_character_image_count()

    embed = ui.build_char_embed(bot_name, background, personality, looks, image_count=image_count)

    file = None
    if image_count >= 1:
        result = database.get_character_image(1)
        if result:
            img_bytes, mime = result
            ext = mime.split("/")[-1] if "/" in mime else "png"
            if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
                ext = "png"
            file = discord.File(io.BytesIO(img_bytes), filename=f"char_preview.{ext}")
            embed.set_image(url=f"attachment://char_preview.{ext}")

    view = ui.CharacterView(bot_name, background, personality, looks, image_count=image_count)
    if file:
        await ctx.reply(embed=embed, file=file, view=view, mention_author=False)
    else:
        await ctx.reply(embed=embed, view=view, mention_author=False)


@bot.hybrid_command(name="addcharimage", description="新增外貌參考圖片到角色設定 (最多 10 張，可批量上傳)")
@app_commands.describe(
    attachment="圖片 1",
    attachment2="圖片 2",
    attachment3="圖片 3",
    attachment4="圖片 4",
    attachment5="圖片 5",
    attachment6="圖片 6",
    attachment7="圖片 7",
    attachment8="圖片 8",
    attachment9="圖片 9",
    attachment10="圖片 10",
)
async def addcharimage_cmd(
    ctx,
    attachment: Optional[discord.Attachment] = None,
    attachment2: Optional[discord.Attachment] = None,
    attachment3: Optional[discord.Attachment] = None,
    attachment4: Optional[discord.Attachment] = None,
    attachment5: Optional[discord.Attachment] = None,
    attachment6: Optional[discord.Attachment] = None,
    attachment7: Optional[discord.Attachment] = None,
    attachment8: Optional[discord.Attachment] = None,
    attachment9: Optional[discord.Attachment] = None,
    attachment10: Optional[discord.Attachment] = None,
):
    """新增角色外貌參考圖 (批量): !addcharimage (prefix: 附上圖片; slash: 使用 attachment 參數)"""
    if not await check_command_role(ctx):
        return

    if ctx.interaction is not None:
        candidates = [
            a for a in [
                attachment, attachment2, attachment3, attachment4, attachment5,
                attachment6, attachment7, attachment8, attachment9, attachment10,
            ] if a is not None
        ]
    else:
        candidates = list(getattr(ctx.message, "attachments", None) or [])

    if not candidates:
        await ctx.reply("❌ 請附上至少一張圖片。", mention_author=False)
        return

    valid_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    valid_candidates = [a for a in candidates if a.filename.lower().endswith(valid_exts)]
    if not valid_candidates:
        await ctx.reply("❌ 請附上有效的圖像格式 (PNG、JPG、GIF 或 WebP)。", mention_author=False)
        return

    current_count = database.get_character_image_count()
    if current_count >= database.MAX_CHARACTER_IMAGES:
        await ctx.reply(
            f"❌ 已有 {current_count} 張角色圖片，已達上限 {database.MAX_CHARACTER_IMAGES} 張。\n"
            "請使用 `/character` → 外貌圖庫 移除舊圖片再新增。",
            mention_author=False,
        )
        return

    slots_left = database.MAX_CHARACTER_IMAGES - current_count
    to_process = valid_candidates[:slots_left]
    skipped_count = len(valid_candidates) - len(to_process)

    await ctx.defer()

    bot_name, *_ = load_character()
    added, failed = [], []
    total = len(to_process)

    progress_msg = None
    if total > 1:
        progress_msg = await ctx.send(
            f"📤 正在處理 **{total}** 張角色圖片⋯\n{_make_progress_bar(0, 0, total)}"
        )

    async with aiohttp.ClientSession() as session:
        for att in to_process:
            fname = att.filename.lower()
            mime = "image/jpeg"
            if fname.endswith(".png"):
                mime = "image/png"
            elif fname.endswith(".gif"):
                mime = "image/gif"
            elif fname.endswith(".webp"):
                mime = "image/webp"

            try:
                async with session.get(att.url) as resp:
                    img_bytes = await resp.read()

                auto_desc = await groq_ai.understand_image(
                    img_bytes, mime,
                    f"Describe this character's permanent physical appearance in detail for an AI character reference sheet. "
                    f"Focus on: iris color and eye shape, hair color, hair style and length, skin tone, face shape, eyebrow shape, and any other distinctive facial features. "
                    f"Also note body type and height if visible. "
                    f"Mention clothing only briefly — this character ({bot_name}) has many outfits so garments are not reliable identifiers.",
                )
                description = auto_desc or ""

                success, msg = database.add_character_image(img_bytes, mime, description)
                if success:
                    added.append(att.filename)
                else:
                    failed.append(f"{att.filename} ({msg})")
            except Exception as e:
                failed.append(f"{att.filename} ({e})")

            if progress_msg:
                done = len(added) + len(failed)
                label = "✅ 完成！" if done == total else f"正在處理 **{total}** 張角色圖片⋯"
                try:
                    await progress_msg.edit(
                        content=f"📤 {label}\n{_make_progress_bar(len(added), len(failed), total)}"
                    )
                except Exception:
                    pass

    if progress_msg:
        try:
            await progress_msg.delete()
        except Exception:
            pass

    if added:
        _invalidate_char_images_ctx()

    new_count = database.get_character_image_count()
    embed = discord.Embed(
        title="🖼️ 角色外貌圖片批量新增結果",
        color=discord.Color.gold() if added else discord.Color.red(),
    )
    embed.add_field(name="📊 圖片數量", value=f"{new_count} / {database.MAX_CHARACTER_IMAGES}", inline=True)
    embed.add_field(name="✅ 成功新增", value=str(len(added)), inline=True)
    if failed:
        embed.add_field(name="❌ 失敗", value="\n".join(failed)[:500], inline=False)
    if skipped_count > 0:
        embed.add_field(
            name="⏭️ 已略過",
            value=f"{skipped_count} 張 (已達上限 {database.MAX_CHARACTER_IMAGES} 張)",
            inline=False,
        )
    embed.set_footer(text="使用 /character 查看角色設定與外貌圖庫。")
    await ctx.send(embed=embed)


@bot.hybrid_command(name="remember", description="保存文字到知識庫")
@app_commands.describe(title="條目標題", content="要保存的文字內容")
async def remember_cmd(ctx, title: str, *, content: str):
    """保存文本到知識庫: !remember "標題" 內容"""
    if not await check_command_role(ctx):
        return
    entry_id = database.add_text_entry(title, content)
    _invalidate_kb_title_index()
    embed = discord.Embed(
        title="📝 已保存到知識庫",
        color=discord.Color.green(),
    )
    embed.add_field(name="🏷️ 標題", value=title, inline=True)
    embed.add_field(name="🆔 條目編號", value=f"#{entry_id}", inline=True)
    embed.add_field(name="📄 內容", value=content[:500] + ("..." if len(content) > 500 else ""), inline=False)
    view = ui.RememberView(entry_id, title, content)
    await ctx.reply(embed=embed, view=view, mention_author=False)


@bot.hybrid_command(name="forget", description="從知識庫刪除條目 (需確認)")
@app_commands.describe(entry_id="要刪除的條目編號")
async def forget_cmd(ctx, entry_id: int):
    """從知識庫刪除條目 (需確認): !forget <id>"""
    if not await check_command_role(ctx):
        return
    entry = database.get_entry_by_id(entry_id)
    if not entry:
        await ctx.reply(f"❌ 找不到 ID 為 #{entry_id}.", mention_author=False)
        return
    embed = discord.Embed(
        title="⚠️ 確認刪除",
        description=f"確定要刪除 **{entry['title']}** (#{entry_id})?\n此操作無法撤銷.",
        color=discord.Color.red(),
    )
    view = ui.ConfirmDeleteView(entry_id, entry["title"])
    await ctx.reply(embed=embed, view=view, mention_author=False)


@bot.hybrid_command(name="setdesc", description="更新知識庫圖像條目的描述")
@app_commands.describe(entry_id="圖像條目編號", description="新的描述文字")
async def setdesc_cmd(ctx, entry_id: int, *, description: str):
    """更新圖像條目的描述: !setdesc <id> <描述>"""
    if not await check_command_role(ctx):
        return
    entry = database.get_entry_by_id(entry_id)
    if not entry:
        await ctx.reply(f"❌ 找不到 ID 為 #{entry_id}.", mention_author=False)
        return
    if entry["entry_type"] != "image":
        await ctx.reply(
            f"❌ 條目 #{entry_id} 是文字條目，無法設定圖像描述。請使用 `/remember` 更新文字內容。",
            mention_author=False,
        )
        return
    success = database.update_image_description(entry_id, description)
    if success:
        _invalidate_kb_title_index()
        embed = discord.Embed(title="✅ 描述已更新", color=discord.Color.green())
        embed.add_field(name="🖼️ 條目", value=f"#{entry_id} — {entry['title']}", inline=False)
        embed.add_field(name="📄 新描述", value=description[:1000], inline=False)
        updated_entry = database.get_entry_by_id(entry_id)
        view = ui.EntryView(updated_entry)
        await ctx.reply(embed=embed, view=view, mention_author=False)
    else:
        await ctx.reply("❌ 發生錯誤，請重試。", mention_author=False)


@bot.hybrid_command(name="viewentry", description="檢視知識庫條目；不填 ID 則開啟管理器")
@app_commands.describe(entry_id="條目編號 (可選，留空開啟管理器)")
async def viewentry_cmd(ctx, entry_id: Optional[int] = None):
    """無 ID → 開啟互動式知識庫管理器; 有 ID → 檢視該條目: !viewentry [id]"""
    if not await check_command_role(ctx):
        return
    if entry_id is None:
        entries = database.get_all_entries(200)
        if not entries:
            await ctx.reply("🗂️ 知識庫目前沒有任何條目。使用 `!remember` 或 `!saveimage` 新增內容。", mention_author=False)
            return
        view = ui.KBManagerView(entries)
        embed = view._get_embed()
        await ctx.reply(embed=embed, view=view, mention_author=False)
    else:
        entry = database.get_entry_by_id(entry_id)
        if not entry:
            await ctx.reply(f"❌ 找不到 ID 為 #{entry_id} 的條目。", mention_author=False)
            return
        embed = ui._build_entry_embed(entry)
        view = ui.EntryView(entry)
        await ctx.reply(embed=embed, view=view, mention_author=False)


@bot.hybrid_command(name="knowledge", description="搜尋或瀏覽知識庫 (互動式分頁)")
@app_commands.describe(query="搜尋關鍵字 (可選，留空列出所有條目)")
async def knowledge_cmd(ctx, *, query: str = ""):
    """瀏覽知識庫 (互動式分頁): !knowledge [查詢]"""
    if not await check_command_role(ctx):
        return
    results = database.search_knowledge(query) if query else database.get_all_entries(200)
    if not results:
        msg = f"🗂️ 找不到符合「{query}」的知識庫條目。" if query else "🗂️ 知識庫目前沒有任何條目。"
        await ctx.reply(msg, mention_author=False)
        return
    view = ui.KBManagerView(results, query=query)
    embed = view._get_embed()
    await ctx.reply(embed=embed, view=view, mention_author=False)


MAX_SAVEIMAGE_BATCH = 5

@bot.hybrid_command(name="saveimage", description="保存圖像到知識庫，可批量上傳最多 5 張 (斜線指令請使用 attachment 參數)")
@app_commands.describe(
    title="條目標題 (多張圖片時自動編號)",
    attachment="圖片 1",
    attachment2="圖片 2",
    attachment3="圖片 3",
    attachment4="圖片 4",
    attachment5="圖片 5",
    description="圖像描述 (可選，留空自動生成；批量時套用於所有圖片)",
)
async def saveimage_cmd(
    ctx,
    title: str,
    attachment: Optional[discord.Attachment] = None,
    attachment2: Optional[discord.Attachment] = None,
    attachment3: Optional[discord.Attachment] = None,
    attachment4: Optional[discord.Attachment] = None,
    attachment5: Optional[discord.Attachment] = None,
    *,
    description: str = "",
):
    """保存圖像到知識庫 (批量最多 5 張): !saveimage "標題" [描述]  (prefix: attach images; slash: use attachment params)"""
    if not await check_command_role(ctx):
        return

    slash_attachments = [
        a for a in [
            attachment, attachment2, attachment3, attachment4, attachment5,
        ] if a is not None
    ]
    if ctx.interaction is not None:
        candidates = slash_attachments
    else:
        candidates = list(getattr(ctx.message, "attachments", None) or [])

    if not candidates:
        await ctx.reply("請附上至少一張圖像再使用此指令。", mention_author=False)
        return

    valid_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    valid_candidates = [a for a in candidates if a.filename.lower().endswith(valid_exts)]
    if not valid_candidates:
        await ctx.reply("請附上有效的圖像格式 (PNG、JPG、GIF 或 WebP)。", mention_author=False)
        return

    skipped_over_limit = max(0, len(valid_candidates) - MAX_SAVEIMAGE_BATCH)
    valid_candidates = valid_candidates[:MAX_SAVEIMAGE_BATCH]

    await ctx.defer()

    user_desc = description
    success_count = 0
    failed = []
    total = len(valid_candidates)
    main_entry_id = None
    auto_generated = not user_desc
    auto_descs: list[str] = []

    progress_msg = None
    if total > 1:
        progress_msg = await ctx.send(
            f"📤 正在處理 **{total}** 張圖片⋯\n{_make_progress_bar(0, 0, total)}"
        )

    async with aiohttp.ClientSession() as session:
        for i, att in enumerate(valid_candidates):
            fname = att.filename.lower()
            mime = "image/jpeg"
            if fname.endswith(".png"):
                mime = "image/png"
            elif fname.endswith(".gif"):
                mime = "image/gif"
            elif fname.endswith(".webp"):
                mime = "image/webp"

            try:
                async with session.get(att.url) as resp:
                    img_bytes = await resp.read()

                desc = user_desc
                if not desc:
                    analysed = await groq_ai.understand_image(
                        img_bytes, mime,
                        "Describe any characters in this image for a knowledge base entry, focusing on permanent physical features: "
                        "iris color, eye shape, hair color, hair style and length, skin tone, face shape, eyebrow shape, and distinctive facial traits. "
                        "These are anime/game characters who change outfits frequently, so focus on facial and physical features over clothing. "
                        "Also briefly describe the scene, background, or other notable elements.",
                    )
                    desc = analysed or ""
                    auto_descs.append(desc)

                if i == 0:
                    main_entry_id = database.add_image_entry(title, img_bytes, mime, desc)
                    success_count += 1
                else:
                    ok, msg = database.add_image_to_entry(main_entry_id, img_bytes, mime)
                    if ok:
                        success_count += 1
                    else:
                        failed.append(f"{att.filename} ({msg})")
            except Exception as e:
                failed.append(f"{att.filename} ({e})")

            if progress_msg:
                done = success_count + len(failed)
                label = "✅ 完成！" if done == total else f"正在處理 **{total}** 張圖片⋯"
                try:
                    await progress_msg.edit(
                        content=f"📤 {label}\n{_make_progress_bar(success_count, len(failed), total)}"
                    )
                except Exception:
                    pass

    if auto_descs and len(auto_descs) > 1 and main_entry_id is not None:
        combined = "\n\n".join(
            f"圖片{j + 1}: {d}" for j, d in enumerate(auto_descs) if d
        )
        if combined:
            database.update_image_description(main_entry_id, combined)

    if progress_msg:
        try:
            await progress_msg.delete()
        except Exception:
            pass

    if success_count > 0:
        _invalidate_kb_title_index()

    entry_id = main_entry_id
    final_desc = user_desc
    if not final_desc and auto_descs:
        if len(auto_descs) == 1:
            final_desc = auto_descs[0]
        else:
            final_desc = "\n\n".join(
                f"圖片{j + 1}: {d}" for j, d in enumerate(auto_descs) if d
            )

    embed = discord.Embed(
        title="🖼️ Image 已保存到知識庫",
        color=discord.Color.green() if success_count > 0 else discord.Color.red(),
    )
    embed.add_field(name="🏷️ 標題", value=title, inline=True)
    if entry_id is not None:
        embed.add_field(name="🆔 條目編號", value=f"#{entry_id}", inline=True)
    if total > 1:
        embed.add_field(name="📊 圖片數量", value=f"{success_count} / {total}", inline=True)
    if auto_generated and final_desc:
        desc_label = "📄 Description (自動生成)"
        desc_value = final_desc[:500] + ("..." if len(final_desc) > 500 else "")
    elif auto_generated and not final_desc:
        desc_label = "⚠️ Description (自動生成失敗)"
        desc_value = f"視覺分析模型無法識別此圖像。\n請使用 `/setdesc {entry_id} <描述>` 手動新增描述，否則機器人將無法在對話中使用此圖像。"
    else:
        desc_label = "📄 Description (使用者提供)"
        desc_value = final_desc[:500] + ("..." if len(final_desc) > 500 else "")
    embed.add_field(name=desc_label, value=desc_value, inline=False)
    if failed:
        embed.add_field(name="❌ 失敗", value="\n".join(failed)[:500], inline=False)
    if skipped_over_limit > 0:
        embed.add_field(
            name="⏭️ 已略過 (超過上限)",
            value=f"{skipped_over_limit} 張 (每次最多 {MAX_SAVEIMAGE_BATCH} 張)",
            inline=False,
        )
    if entry_id is not None:
        view = ui.SaveImageView(entry_id, title)
        await ctx.send(embed=embed, view=view)
    else:
        await ctx.send(embed=embed)


@bot.hybrid_command(name="addimage", description="新增圖片到已有的圖片知識庫條目 (最多 5 張，可批量上傳)")
@app_commands.describe(
    entry_id="目標條目編號 (使用 /knowledge 查看)",
    attachment="圖片 1",
    attachment2="圖片 2",
    attachment3="圖片 3",
    attachment4="圖片 4",
    attachment5="圖片 5",
)
async def addimage_cmd(
    ctx,
    entry_id: int,
    attachment: Optional[discord.Attachment] = None,
    attachment2: Optional[discord.Attachment] = None,
    attachment3: Optional[discord.Attachment] = None,
    attachment4: Optional[discord.Attachment] = None,
    attachment5: Optional[discord.Attachment] = None,
):
    """新增圖片到 KB 條目 (批量): !addimage <條目編號> (prefix: 附上圖片; slash: 使用 attachment 參數)"""
    if not await check_command_role(ctx):
        return

    entry = database.get_entry_by_id(entry_id)
    if not entry:
        await ctx.reply(f"❌ 找不到條目 #{entry_id}。請使用 `/knowledge` 確認條目編號。", mention_author=False)
        return
    if entry.get("entry_type") != "image":
        await ctx.reply(f"❌ 條目 #{entry_id} 是文字條目，只能將圖片新增到圖片條目。", mention_author=False)
        return

    if ctx.interaction is not None:
        candidates = [
            a for a in [attachment, attachment2, attachment3, attachment4, attachment5]
            if a is not None
        ]
    else:
        candidates = list(getattr(ctx.message, "attachments", None) or [])

    if not candidates:
        await ctx.reply("❌ 請附上至少一張圖片。", mention_author=False)
        return

    valid_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
    valid_candidates = [a for a in candidates if a.filename.lower().endswith(valid_exts)]
    if not valid_candidates:
        await ctx.reply("❌ 請附上有效的圖像格式 (PNG、JPG、GIF 或 WebP)。", mention_author=False)
        return

    current_count = database.get_entry_image_count(entry_id)
    if current_count >= database.MAX_IMAGES_PER_ENTRY:
        await ctx.reply(
            f"❌ 條目 #{entry_id} 已有 {current_count} 張圖片，已達上限 {database.MAX_IMAGES_PER_ENTRY} 張。\n"
            "請先在 `/viewentry` 中移除舊圖片再新增。",
            mention_author=False,
        )
        return

    slots_left = database.MAX_IMAGES_PER_ENTRY - current_count
    to_process = valid_candidates[:slots_left]
    skipped_count = len(valid_candidates) - len(to_process)

    await ctx.defer()

    added, failed = [], []
    total = len(to_process)

    progress_msg = None
    if total > 1:
        progress_msg = await ctx.send(
            f"📤 正在處理 **{total}** 張圖片⋯\n{_make_progress_bar(0, 0, total)}"
        )

    async with aiohttp.ClientSession() as session:
        for att in to_process:
            fname = att.filename.lower()
            mime = "image/jpeg"
            if fname.endswith(".png"):
                mime = "image/png"
            elif fname.endswith(".gif"):
                mime = "image/gif"
            elif fname.endswith(".webp"):
                mime = "image/webp"

            try:
                async with session.get(att.url) as resp:
                    img_bytes = await resp.read()
                success, msg = database.add_image_to_entry(entry_id, img_bytes, mime)
                if success:
                    added.append(att.filename)
                else:
                    failed.append(f"{att.filename} ({msg})")
            except Exception as e:
                failed.append(f"{att.filename} ({e})")

            if progress_msg:
                done = len(added) + len(failed)
                label = "✅ 完成！" if done == total else f"正在處理 **{total}** 張圖片⋯"
                try:
                    await progress_msg.edit(
                        content=f"📤 {label}\n{_make_progress_bar(len(added), len(failed), total)}"
                    )
                except Exception:
                    pass

    if progress_msg:
        try:
            await progress_msg.delete()
        except Exception:
            pass

    new_count = database.get_entry_image_count(entry_id)
    embed = discord.Embed(
        title="🖼️ 圖片批量新增結果",
        color=discord.Color.green() if added else discord.Color.red(),
    )
    embed.add_field(name="🏷️ 條目", value=f"#{entry_id} {entry.get('title', '')}", inline=True)
    embed.add_field(name="📊 圖片數量", value=f"{new_count} / {database.MAX_IMAGES_PER_ENTRY}", inline=True)
    embed.add_field(name="✅ 成功新增", value=str(len(added)), inline=True)
    if failed:
        embed.add_field(name="❌ 失敗", value="\n".join(failed)[:500], inline=False)
    if skipped_count > 0:
        embed.add_field(
            name="⏭️ 已略過 (超過上限)",
            value=f"{skipped_count} 張 (每個條目最多 {database.MAX_IMAGES_PER_ENTRY} 張)",
            inline=False,
        )
    embed.set_footer(text="使用 /viewentry 查看並管理所有圖片。")
    await ctx.send(embed=embed)


@bot.hybrid_command(name="generate", description="使用 Cloudflare Workers AI (Flux) 直接生成圖像")
@app_commands.describe(prompt="圖像提示詞 (英文效果最佳)")
async def generate_cmd(ctx, *, prompt: str):
    """使用 Cloudflare Workers AI (Flux) 生成圖像: !generate <提示詞>"""
    if not _cf_ready():
        await ctx.reply("❌ 圖像生成已禁用 — Cloudflare API 認證未配置。", mention_author=False)
        return

    await ctx.defer()

    enriched_prompt, kb_matches = await _enrich_image_prompt_with_kb(prompt)
    result = await cloudflare_ai.generate_image(enriched_prompt)

    if result == ("API_KEY_ERROR", ""):
        await ctx.send(
            "❌ **Cloudflare API 錯誤**: 妳的 API 金鑰無效或已過期。\n\n"
            "修復: 從 https://dash.cloudflare.com/ 獲取新的 API 金鑰並更新妳的 `tokens.txt`",
        )
    elif result == ("MODEL_ERROR", ""):
        await ctx.send(
            "❌ **Cloudflare 模型錯誤**: 圖像生成模型暫時不可用。\n\n"
            "請稍後重試，或檢查妳的帳戶是否有權訪問 Workers AI。",
        )
    elif result:
        img_data, mime_type = result
        ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
        file = discord.File(io.BytesIO(img_data), filename=f"generated.{ext}")
        embed = discord.Embed(
            title="🎨 生成的圖像",
            description=f"**提示詞:** {prompt}",
            color=discord.Color.purple(),
        )
        if kb_matches:
            ref_titles = ", ".join(m.get("title", "") for m in kb_matches if m.get("title"))
            if ref_titles:
                embed.set_footer(text=f"📚 知識庫參考: {ref_titles}")
        view = ui.GenerateView(prompt, img_data, mime_type)
        await ctx.send(embed=embed, file=file, view=view)
    else:
        await ctx.send(
            "❌ **圖像生成失敗**\n"
            "請查看機器人的控制台視窗，會顯示 `[Cloudflare]` 開頭的錯誤訊息。\n\n"
            "常見原因:\n"
            "• Cloudflare API Token 或 Account ID 錯誤\n"
            "• Workers AI 未在帳戶中啟用\n"
            "• 提示詞觸發內容過濾\n"
            "• API 服務暫時中斷",
        )


@bot.hybrid_command(name="clear", description="清除此頻道的對話歷史記錄")
async def clear_cmd(ctx):
    """清除此頻道的對話歷史: !clear"""
    if not await check_command_role(ctx):
        return
    channel_id = str(ctx.channel.id)
    conversation_contexts.pop(channel_id, None)
    await ctx.reply("✅ 此頻道的對話歷史已清除！", mention_author=False)


@bot.hybrid_command(name="suggestions", description="開啟或關閉建議按鈕")
async def suggestions_cmd(ctx):
    """切換建議按鈕開關: !suggestions 或 /suggestions"""
    if not await check_command_role(ctx):
        return
    global _suggestions_enabled
    _suggestions_enabled = not _suggestions_enabled
    database.set_setting("suggestions_enabled", _suggestions_enabled)
    state = "✅ 已開啟" if _suggestions_enabled else "❌ 已關閉"
    embed = discord.Embed(
        title=f"💬 建議按鈕 — {state}",
        description=(
            "聊天回覆後將顯示三個建議按鈕。" if _suggestions_enabled
            else "聊天回覆後將不再顯示建議按鈕。"
        ),
        color=discord.Color.green() if _suggestions_enabled else discord.Color.red(),
    )
    embed.set_footer(text="再次執行此指令可切換狀態。設定在重啟後保留。")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="setsuggestionprompt", description="設定建議按鈕的自訂引導提示詞 (覆蓋預設)")
@app_commands.describe(prompt="完整的自訂提示詞，告訴 AI 如何生成建議按鈕內容")
async def setsuggestionprompt_cmd(ctx, *, prompt: str):
    """設定自訂建議提示詞: !setsuggestionprompt <提示詞>"""
    if not await check_command_role(ctx):
        return
    global _suggestion_prompt
    _suggestion_prompt = prompt
    database.set_setting("suggestion_prompt", prompt)
    embed = discord.Embed(
        title="✅ 建議提示詞已更新",
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="📝 提示詞",
        value=prompt[:1000] + ("..." if len(prompt) > 1000 else ""),
        inline=False,
    )
    embed.set_footer(
        text="此提示詞用於引導建議的語氣、長度或風格。JSON 格式要求由系統自動補上，無需在提示詞中指定。"
    )
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="clearsuggestionprompt", description="清除自訂建議提示詞，恢復預設")
async def clearsuggestionprompt_cmd(ctx):
    """清除自訂建議提示詞: !clearsuggestionprompt"""
    if not await check_command_role(ctx):
        return
    global _suggestion_prompt
    _suggestion_prompt = ""
    database.set_setting("suggestion_prompt", "")
    await ctx.reply("✅ 已清除自訂建議提示詞，恢復為預設模式。", mention_author=False)


MEMORIES_PAGE_SIZE = 8


class MemoriesView(discord.ui.View):
    """Paginated viewer for stored long-term memories."""

    def __init__(self, memories: list):
        super().__init__(timeout=300)
        self.memories = memories
        self.page = 0
        self.total_pages = max(1, (len(memories) + MEMORIES_PAGE_SIZE - 1) // MEMORIES_PAGE_SIZE)
        self._refresh_buttons()

    def _refresh_buttons(self):
        self.clear_items()
        if self.page > 0:
            btn = discord.ui.Button(label="◀ 上一頁", style=discord.ButtonStyle.secondary, row=0)
            btn.callback = self._prev
            self.add_item(btn)
        if self.page < self.total_pages - 1:
            btn = discord.ui.Button(label="下一頁 ▶", style=discord.ButtonStyle.secondary, row=0)
            btn.callback = self._next
            self.add_item(btn)

    async def _prev(self, interaction: discord.Interaction):
        self.page -= 1
        self._refresh_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _next(self, interaction: discord.Interaction):
        self.page += 1
        self._refresh_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    def _get_embed(self) -> discord.Embed:
        start = self.page * MEMORIES_PAGE_SIZE
        page_items = self.memories[start: start + MEMORIES_PAGE_SIZE]
        embed = discord.Embed(
            title=f"🧠 長期記憶 — 第 {self.page + 1}/{self.total_pages} 頁",
            description=f"共 **{len(self.memories)}** 條記憶",
            color=discord.Color.blurple(),
        )
        for m in page_items:
            age = _memory_age(m["created_at"])
            age_str = f" · {age}" if age else ""
            embed.add_field(
                name=f"👤 {m['user_name']}{age_str}",
                value=m["summary"],
                inline=False,
            )
        embed.set_footer(text="記憶由對話自動提取 · 使用 /clearmemory 可清除自己的記憶 · 管理員可用 /clearmemory @用戶 或 /clearmemory all")
        return embed


class ClearMemoryConfirmView(discord.ui.View):
    """Confirm-before-clear view for /clearmemory.

    Parameters
    ----------
    requester_id:     Discord user ID of the person who invoked /clearmemory (buttons only respond to them)
    clear_all:        True → wipe every memory row (admin-only path)
    target_user_id:   Discord user ID string to delete memories for
    target_user_name: Display name used in the result message
    requires_admin:   Whether the confirm action requires Administrator permission (defense-in-depth)
    """

    def __init__(
        self,
        requester_id: int,
        clear_all: bool = False,
        target_user_id: Optional[str] = None,
        target_user_name: str = "",
        requires_admin: bool = False,
    ):
        super().__init__(timeout=60)
        self.requester_id = requester_id
        self.clear_all = clear_all
        self.target_user_id = target_user_id
        self.target_user_name = target_user_name
        self.requires_admin = requires_admin

    async def _check_invoker(self, interaction: discord.Interaction) -> bool:
        """Reject button presses from anyone other than the original invoker."""
        if interaction.user.id != self.requester_id:
            await interaction.response.send_message(
                "❌ 這不是你的確認訊息。", ephemeral=True
            )
            return False
        # Defense-in-depth: re-verify admin for admin-only paths
        if self.requires_admin:
            member = interaction.user
            is_admin = (
                interaction.guild is not None
                and isinstance(member, discord.Member)
                and member.guild_permissions.administrator
            )
            if not is_admin:
                await interaction.response.send_message(
                    "❌ 你不再擁有執行此操作所需的管理員權限。", ephemeral=True
                )
                return False
        return True

    @discord.ui.button(label="確認清除", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._check_invoker(interaction):
            return
        if self.clear_all:
            count = database.clear_memories()
            result_embed = discord.Embed(
                title="✅ 全部記憶已清除",
                description=f"已清除所有用戶的 **{count}** 條長期記憶。",
                color=discord.Color.green(),
            )
        elif self.target_user_id:
            count = database.clear_memories_for_user(self.target_user_id)
            label = self.target_user_name or self.target_user_id
            result_embed = discord.Embed(
                title="✅ 記憶已清除",
                description=f"已清除 **{label}** 的 **{count}** 條長期記憶。",
                color=discord.Color.green(),
            )
        else:
            result_embed = discord.Embed(
                title="❌ 發生錯誤",
                description="未指定清除目標。",
                color=discord.Color.red(),
            )
        for item in self.children:
            item.disabled = True
        await interaction.response.edit_message(embed=result_embed, view=self)

    @discord.ui.button(label="取消", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self._check_invoker(interaction):
            return
        for item in self.children:
            item.disabled = True
        cancel_embed = discord.Embed(
            title="❌ 已取消",
            description="清除記憶操作已取消。",
            color=discord.Color.greyple(),
        )
        await interaction.response.edit_message(embed=cancel_embed, view=self)


@bot.hybrid_command(name="memory", description="開啟或關閉長期記憶功能")
async def memory_cmd(ctx):
    """切換長期記憶開關: !memory 或 /memory"""
    if not await check_command_role(ctx):
        return
    global _memory_enabled
    _memory_enabled = not _memory_enabled
    database.set_setting("memory_enabled", _memory_enabled)
    state = "✅ 已開啟" if _memory_enabled else "❌ 已關閉"
    embed = discord.Embed(
        title=f"🧠 長期記憶 — {state}",
        description=(
            "機器人將自動記住對話中的重要資訊，並在未來的對話中自然地回憶。"
            if _memory_enabled
            else "機器人將不再記憶或使用長期記憶。已儲存的記憶不受影響。"
        ),
        color=discord.Color.green() if _memory_enabled else discord.Color.red(),
    )
    embed.set_footer(text="再次執行此指令可切換狀態。設定在重啟後保留。")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="memorylength", description="設定主動記憶長度 (每次對話注入幾條記憶)")
@app_commands.describe(length="注入的記憶條數 (1–100，預設 20)")
async def memorylength_cmd(ctx, length: int):
    """設定主動記憶長度: !memorylength <數字>"""
    if not await check_command_role(ctx):
        return
    global _memory_length
    if not 1 <= length <= 100:
        await ctx.reply("❌ 請輸入 1 到 100 之間的數字。", mention_author=False)
        return
    _memory_length = length
    database.set_setting("memory_length", length)
    embed = discord.Embed(
        title="🧠 主動記憶長度已更新",
        description=f"每次對話將注入最近 **{length}** 條記憶到機器人的上下文中。",
        color=discord.Color.blurple(),
    )
    embed.set_footer(text="設定在重啟後保留。")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="passivememory", description="開啟或關閉被動記憶 (用戶詢問時調用深層記憶庫)")
async def passivememory_cmd(ctx):
    """切換被動記憶開關: !passivememory 或 /passivememory"""
    if not await check_command_role(ctx):
        return
    global _passive_memory_enabled
    _passive_memory_enabled = not _passive_memory_enabled
    database.set_setting("passive_memory_enabled", _passive_memory_enabled)
    state = "✅ 已開啟" if _passive_memory_enabled else "❌ 已關閉"
    embed = discord.Embed(
        title=f"🗄️ 被動記憶 — {state}",
        description=(
            "當你問「你還記得…」、「上次我說…」等語句時，機器人將搜索深層記憶庫並回憶更久遠的記憶。"
            if _passive_memory_enabled
            else "機器人將不再根據詢問語句調用深層記憶庫。主動記憶不受影響。"
        ),
        color=discord.Color.green() if _passive_memory_enabled else discord.Color.red(),
    )
    embed.set_footer(text="再次執行此指令可切換狀態。設定在重啟後保留。")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="passivememorylength", description="設定被動記憶搜索的最大條數")
@app_commands.describe(length="被動記憶搜索的最大條數 (1–500，預設 200)")
async def passivememorylength_cmd(ctx, length: int):
    """設定被動記憶長度: !passivememorylength <數字>"""
    if not await check_command_role(ctx):
        return
    global _passive_memory_length
    if not 1 <= length <= 500:
        await ctx.reply("❌ 請輸入 1 到 500 之間的數字。", mention_author=False)
        return
    _passive_memory_length = length
    database.set_setting("passive_memory_length", length)
    embed = discord.Embed(
        title="🗄️ 被動記憶長度已更新",
        description=f"當你詢問過去記憶時，最多從深層記憶庫中搜索 **{length}** 條記憶。",
        color=discord.Color.blurple(),
    )
    embed.set_footer(text="設定在重啟後保留。")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="memories", description="檢視所有儲存的長期記憶")
async def memories_cmd(ctx):
    """檢視長期記憶: !memories 或 /memories"""
    if not await check_command_role(ctx):
        return
    all_mems = database.get_all_memories(500)
    if not all_mems:
        embed = discord.Embed(
            title="🧠 長期記憶",
            description="目前沒有任何儲存的記憶。\n開始與機器人對話，記憶將在背景自動生成！",
            color=discord.Color.blurple(),
        )
        await ctx.reply(embed=embed, mention_author=False)
        return
    view = MemoriesView(all_mems)
    await ctx.reply(embed=view._get_embed(), view=view, mention_author=False)


@bot.hybrid_command(name="clearmemory", description="清除長期記憶：不填=自己，@用戶=指定用戶，all=全部 (後兩者需管理員)")
@app_commands.describe(
    user="要清除記憶的用戶 (留空清除自己的記憶，需管理員才能清除他人)",
    all="輸入 true 清除所有人的記憶 (需管理員)",
)
async def clearmemory_cmd(
    ctx,
    user: Optional[discord.Member] = None,
    all: bool = False,
):
    """清除長期記憶: !clearmemory [all] [@用戶]"""
    is_admin = (
        ctx.guild is not None
        and isinstance(ctx.author, discord.Member)
        and ctx.author.guild_permissions.administrator
    )

    # Determine mode
    clear_all = all  # explicit "all" flag
    target_member = user  # explicit @user mention

    # Edge case: both all=True and a user mention — ambiguous, reject early
    if clear_all and target_member:
        await ctx.reply(
            "❌ 請不要同時指定 `all` 和 `@用戶`，請選擇其中一種清除方式。",
            mention_author=False,
        )
        return

    if clear_all:
        if not is_admin:
            embed = discord.Embed(
                title="🔒 權限不足",
                description="清除**所有人**的記憶需要**伺服器管理員**權限。",
                color=discord.Color.red(),
            )
            await ctx.reply(embed=embed, mention_author=False)
            return
        count = database.count_memories()
        if count == 0:
            await ctx.reply("🧠 目前沒有任何記憶可以清除。", mention_author=False)
            return
        embed = discord.Embed(
            title="⚠️ 確認清除全部記憶",
            description=f"確定要清除**所有用戶**的 **{count}** 條長期記憶？\n此操作無法撤銷。",
            color=discord.Color.red(),
        )
        view = ClearMemoryConfirmView(
            requester_id=ctx.author.id,
            clear_all=True,
            requires_admin=True,
        )
        await ctx.reply(embed=embed, view=view, mention_author=False)
        return

    if target_member and target_member.id != ctx.author.id:
        if not is_admin:
            embed = discord.Embed(
                title="🔒 權限不足",
                description="清除**他人**記憶需要**伺服器管理員**權限。",
                color=discord.Color.red(),
            )
            await ctx.reply(embed=embed, mention_author=False)
            return
        uid = str(target_member.id)
        uname = target_member.display_name
        count = database.count_memories_for_user(uid)
        if count == 0:
            await ctx.reply(f"🧠 **{uname}** 目前沒有任何記憶可以清除。", mention_author=False)
            return
        embed = discord.Embed(
            title=f"⚠️ 確認清除 {uname} 的記憶",
            description=f"確定要清除 **{uname}** 的 **{count}** 條長期記憶？\n此操作無法撤銷。",
            color=discord.Color.red(),
        )
        view = ClearMemoryConfirmView(
            requester_id=ctx.author.id,
            target_user_id=uid,
            target_user_name=uname,
            requires_admin=True,
        )
        await ctx.reply(embed=embed, view=view, mention_author=False)
        return

    # Default: clear own memories (open to all)
    uid = str(ctx.author.id)
    uname = ctx.author.display_name
    count = database.count_memories_for_user(uid)
    if count == 0:
        await ctx.reply("🧠 你目前沒有任何記憶可以清除。", mention_author=False)
        return
    embed = discord.Embed(
        title="⚠️ 確認清除自己的記憶",
        description=f"確定要清除你自己的 **{count}** 條長期記憶？\n此操作無法撤銷。",
        color=discord.Color.red(),
    )
    view = ClearMemoryConfirmView(
        requester_id=ctx.author.id,
        target_user_id=uid,
        target_user_name=uname,
    )
    await ctx.reply(embed=embed, view=view, mention_author=False)


@bot.hybrid_command(name="setstatus", description="設定機器人的自訂狀態")
@app_commands.describe(
    text="狀態文字",
    activity_type="活動類型 (playing/watching/listening/competing)",
    status="線上狀態 (online/idle/dnd/invisible)",
)
@app_commands.choices(
    activity_type=[
        app_commands.Choice(name="🎮 playing (玩遊戲)", value="playing"),
        app_commands.Choice(name="📺 watching (觀看)", value="watching"),
        app_commands.Choice(name="🎵 listening (聆聽)", value="listening"),
        app_commands.Choice(name="🏆 competing (競賽)", value="competing"),
    ],
    status=[
        app_commands.Choice(name="🟢 online (上線)", value="online"),
        app_commands.Choice(name="🌙 idle (閒置)", value="idle"),
        app_commands.Choice(name="⛔ dnd (請勿打擾)", value="dnd"),
        app_commands.Choice(name="⚫ invisible (隱形)", value="invisible"),
    ],
)
async def setstatus_cmd(
    ctx,
    *,
    text: str,
    activity_type: str = "listening",
    status: str = "online",
):
    """設定機器人自訂狀態: !setstatus <文字> [activity_type] [status]"""
    if not await check_command_role(ctx):
        return
    if activity_type not in _ACTIVITY_TYPES:
        await ctx.reply(
            f"❌ 無效的活動類型「{activity_type}」。\n"
            "請選擇: `playing` / `watching` / `listening` / `competing`",
            mention_author=False,
        )
        return
    if status not in _STATUS_TYPES:
        await ctx.reply(
            f"❌ 無效的線上狀態「{status}」。\n"
            "請選擇: `online` / `idle` / `dnd` / `invisible`",
            mention_author=False,
        )
        return

    _custom_status["text"] = text
    _custom_status["activity_type"] = activity_type
    _custom_status["status"] = status
    database.set_status(text, activity_type, status)
    await _apply_status()

    activity_labels = {
        "playing": "🎮 玩遊戲", "watching": "📺 觀看",
        "listening": "🎵 聆聽", "competing": "🏆 競賽",
    }
    status_labels = {
        "online": "🟢 上線", "idle": "🌙 閒置",
        "dnd": "⛔ 請勿打擾", "invisible": "⚫ 隱形",
    }
    embed = discord.Embed(title="✅ 狀態已更新", color=discord.Color.green())
    embed.add_field(name="活動", value=f"{activity_labels[activity_type]} **{text}**", inline=False)
    embed.add_field(name="線上狀態", value=status_labels[status], inline=True)
    embed.set_footer(text="使用 /setstatus 可隨時修改狀態")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="clearstatus", description="清除自訂狀態，恢復預設狀態")
async def clearstatus_cmd(ctx):
    """清除自訂狀態恢復預設: !clearstatus"""
    if not await check_command_role(ctx):
        return
    _custom_status.clear()
    database.clear_status()
    await _apply_status()
    await ctx.reply("✅ 已清除自訂狀態，恢復為預設顯示。", mention_author=False)


@bot.hybrid_command(name="setthinking", description="設定機器人的思考泡泡文字 (Discord「What's on your mind?」)")
@app_commands.describe(text="要顯示在思考泡泡中的文字")
async def setthinking_cmd(ctx, *, text: str):
    """設定思考泡泡: !setthinking <文字>"""
    if not await check_command_role(ctx):
        return
    global _custom_thinking
    _custom_thinking = text
    database.set_setting("custom_thinking", text)
    await _apply_status()
    embed = discord.Embed(
        title="💭 思考泡泡已設定",
        description=f"**{text}**",
        color=discord.Color.blurple(),
    )
    embed.set_footer(text="使用 /clearthinking 可移除思考泡泡")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="clearthinking", description="清除思考泡泡，恢復為一般狀態顯示")
async def clearthinking_cmd(ctx):
    """清除思考泡泡: !clearthinking"""
    if not await check_command_role(ctx):
        return
    global _custom_thinking
    _custom_thinking = ""
    database.set_setting("custom_thinking", "")
    await _apply_status()
    await ctx.reply("✅ 已清除思考泡泡。", mention_author=False)


PERMISSIONS_PAGE_SIZE = 11


class PermissionsView(discord.ui.View):
    """Paginated embed showing every command's current role gate."""

    def __init__(self, roles_dict: dict):
        super().__init__(timeout=300)
        self.items = sorted(roles_dict.items())
        self.page = 0
        self.total_pages = max(1, (len(self.items) + PERMISSIONS_PAGE_SIZE - 1) // PERMISSIONS_PAGE_SIZE)
        self._refresh_buttons()

    def _refresh_buttons(self):
        self.clear_items()
        if self.page > 0:
            btn = discord.ui.Button(label="◀ 上一頁", style=discord.ButtonStyle.secondary, row=0)
            btn.callback = self._prev
            self.add_item(btn)
        if self.page < self.total_pages - 1:
            btn = discord.ui.Button(label="下一頁 ▶", style=discord.ButtonStyle.secondary, row=0)
            btn.callback = self._next
            self.add_item(btn)

    async def _prev(self, interaction: discord.Interaction):
        self.page -= 1
        self._refresh_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _next(self, interaction: discord.Interaction):
        self.page += 1
        self._refresh_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    def _get_embed(self) -> discord.Embed:
        start = self.page * PERMISSIONS_PAGE_SIZE
        page_items = self.items[start: start + PERMISSIONS_PAGE_SIZE]
        embed = discord.Embed(
            title=f"🔑 指令權限設定 — 第 {self.page + 1}/{self.total_pages} 頁",
            description=(
                "• 🔒 **管理員限定** — 尚未分配角色，僅管理員可用\n"
                "• @**角色名稱** — 指定角色（或管理員）可用\n"
                "• 🌐 **所有人** — 開放給所有成員"
            ),
            color=discord.Color.gold(),
        )
        for cmd, gate in page_items:
            if gate is None:
                gate_str = "🌐 所有人"
            elif gate == "__admin__":
                gate_str = "🔒 管理員限定"
            elif gate == "__admin_fixed__":
                gate_str = "🔒 管理員限定 (固定)"
            else:
                gate_str = f"@{gate}"
            embed.add_field(name=f"/{cmd}", value=gate_str, inline=True)
        embed.set_footer(text="使用 /setrole <指令> <角色> 設定 · /clearrole <指令> 移除限制")
        return embed


@bot.hybrid_command(name="setrole", description="指定哪個角色可以使用某個指令 (需管理員)")
@app_commands.describe(command="指令名稱 (不含 /)", role="允許使用該指令的角色")
async def setrole_cmd(ctx, command: str, role: discord.Role):
    """設定指令角色權限: !setrole <指令> @角色 (需管理員)"""
    if not (ctx.guild and isinstance(ctx.author, discord.Member) and ctx.author.guild_permissions.administrator):
        embed = discord.Embed(
            title="🔒 權限不足",
            description="只有**伺服器管理員**才能更改指令權限設定。",
            color=discord.Color.red(),
        )
        await ctx.reply(embed=embed, mention_author=False)
        return

    known = list(database._DEFAULT_COMMAND_ROLES.keys())
    if command not in known:
        await ctx.reply(
            f"❌ 找不到指令「`{command}`」。\n"
            f"可用指令: {', '.join(f'`{c}`' for c in sorted(known))}",
            mention_author=False,
        )
        return

    database.set_command_role(command, role.name)
    database.sync_permissions_file(command, role.name)
    global _command_roles
    _command_roles = database.get_command_roles()

    embed = discord.Embed(
        title="✅ 指令權限已更新",
        color=discord.Color.green(),
    )
    embed.add_field(name="指令", value=f"/{command}", inline=True)
    embed.add_field(name="允許角色", value=role.mention, inline=True)
    embed.set_footer(text="擁有該角色或管理員權限的成員均可使用此指令。")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="clearrole", description="移除指令的角色限制，開放給所有人 (需管理員)")
@app_commands.describe(command="要移除限制的指令名稱 (不含 /)")
async def clearrole_cmd(ctx, command: str):
    """移除指令角色限制: !clearrole <指令> (需管理員)"""
    if not (ctx.guild and isinstance(ctx.author, discord.Member) and ctx.author.guild_permissions.administrator):
        embed = discord.Embed(
            title="🔒 權限不足",
            description="只有**伺服器管理員**才能更改指令權限設定。",
            color=discord.Color.red(),
        )
        await ctx.reply(embed=embed, mention_author=False)
        return

    known = list(database._DEFAULT_COMMAND_ROLES.keys())
    if command not in known:
        await ctx.reply(
            f"❌ 找不到指令「`{command}`」。\n"
            f"可用指令: {', '.join(f'`{c}`' for c in sorted(known))}",
            mention_author=False,
        )
        return

    database.clear_command_role(command)
    database.sync_permissions_file(command, None)
    global _command_roles
    _command_roles = database.get_command_roles()

    embed = discord.Embed(
        title="✅ 指令限制已移除",
        color=discord.Color.green(),
    )
    embed.add_field(name="指令", value=f"/{command}", inline=True)
    embed.add_field(name="狀態", value="🌐 所有人可用", inline=True)
    embed.set_footer(text="所有成員現在均可使用此指令。")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="permissions", description="檢視所有指令的權限設定 (分頁)")
async def permissions_cmd(ctx):
    """檢視指令權限設定: !permissions 或 /permissions"""
    roles = database.get_command_roles()
    # Include the management commands with fixed labels (they are always admin-only)
    management = {
        "setrole":      "__admin_fixed__",
        "clearrole":    "__admin_fixed__",
        "permissions":  None,
        "help":         None,
        "helpsetting":  "__admin__",
    }
    all_roles = {**roles, **management}
    view = PermissionsView(all_roles)
    await ctx.reply(embed=view._get_embed(), view=view, mention_author=False)


@bot.hybrid_command(name="help", description="顯示使用者指令說明")
async def help_cmd(ctx):
    """顯示使用者說明: !help 或 /help"""
    bot_name, background, *_ = load_character()
    embed = discord.Embed(
        title=f"{bot_name} — 指令說明",
        description=background[:500] + ("..." if len(background) > 500 else ""),
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="與我交談",
        value=f"@提及我或回覆我的訊息即可聊天！\n例如: `@{bot_name} 幫我畫一個夕陽`",
        inline=False,
    )
    for section_name, section_value in help_config.USER_HELP_SECTIONS:
        embed.add_field(name=section_name, value=section_value, inline=False)
    embed.set_footer(text="設定指令請使用 /helpsetting · 所有指令均支援 / 斜線與 ! 前綴兩種方式")
    await ctx.reply(embed=embed, mention_author=False)


@bot.hybrid_command(name="helpsetting", description="顯示所有設定與管理指令說明 (管理員/指定角色)")
async def helpsetting_cmd(ctx):
    """顯示管理員設定說明: !helpsetting 或 /helpsetting"""
    if not await check_command_role(ctx):
        return
    bot_name, *_ = load_character()
    embed = discord.Embed(
        title=f"{bot_name} — 設定指令說明",
        description="以下為所有設定與管理員指令。使用 `/setrole <指令> @角色` 可將指令開放給特定角色。",
        color=discord.Color.gold(),
    )
    for section_name, section_value in help_config.ADMIN_HELP_SECTIONS:
        embed.add_field(name=section_name, value=section_value, inline=False)
    embed.set_footer(text="一般使用者指令請使用 /help · 所有指令均支援 / 斜線與 ! 前綴兩種方式")
    await ctx.reply(embed=embed, mention_author=False)


def main():
    if not DISCORD_TOKEN:
        print("[ERROR] DISCORD_BOT_TOKEN is not set!")
        return

    if not GROQ_API_KEY:
        print("[WARNING] GROQ_API_KEY 未設定。文字聊天將不可用。")

    if not _cf_ready():
        print("[WARNING] Cloudflare API 認證未設定。圖像生成將被禁用。")

    database.init_db()
    database.migrate_thumbnails()
    bot_name, *_ = load_character()
    print(f"[Bot] 啟動角色: {bot_name}")
    print(f"[Bot] Groq: {'準備就緒' if GROQ_API_KEY else '遺失'}")
    print(f"[Bot] Cloudflare: {'準備就緒' if _cf_ready() else '已禁用'}")
    print("[Bot] 按 Ctrl+C 停止機器人。")
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
