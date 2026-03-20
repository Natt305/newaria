"""
Interactive Discord UI components for bot commands.
Views and Modals used by command responses — NOT used for regular chat.
"""
import io
import discord
import database
import cloudflare_ai
import groq_ai

PAGE_SIZE = 5


# ─────────────────────────────────────────────────────────────────────────────
# Modals
# ─────────────────────────────────────────────────────────────────────────────

class EditDescriptionModal(discord.ui.Modal, title="編輯描述"):
    description = discord.ui.TextInput(
        label="描述",
        style=discord.TextStyle.paragraph,
        placeholder="輸入新的描述 for this image entry...",
        max_length=2000,
    )

    def __init__(self, entry_id: int, entry_title: str):
        super().__init__()
        self.entry_id = entry_id
        self.entry_title = entry_title

    async def on_submit(self, interaction: discord.Interaction):
        success = database.update_image_description(self.entry_id, str(self.description))
        if success:
            await interaction.response.send_message(
                f"✅ Description 已更新 **{self.entry_title}** (#{self.entry_id})!",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "❌ 更新失敗. The entry may no longer exist.", ephemeral=True
            )


class EditTextModal(discord.ui.Modal, title="編輯內容"):
    content = discord.ui.TextInput(
        label="內容",
        style=discord.TextStyle.paragraph,
        placeholder="輸入更新的內容...",
        max_length=2000,
    )

    def __init__(self, entry_id: int, entry_title: str, current_content: str = ""):
        super().__init__()
        self.entry_id = entry_id
        self.entry_title = entry_title
        self.content.default = current_content[:2000]

    async def on_submit(self, interaction: discord.Interaction):
        success = database.update_text_content(self.entry_id, str(self.content))
        if success:
            await interaction.response.send_message(
                f"✅ Content 已更新 **{self.entry_title}** (#{self.entry_id})!",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "❌ 更新失敗. The entry may no longer exist.", ephemeral=True
            )


class EditCharacterModal(discord.ui.Modal, title="編輯角色"):
    name = discord.ui.TextInput(
        label="名稱",
        placeholder="例如: 少女樂團機器人",
        max_length=50,
    )
    background = discord.ui.TextInput(
        label="個性 / 背景",
        style=discord.TextStyle.paragraph,
        placeholder="描述角色的個性、語氣和背景...",
        max_length=1000,
    )

    def __init__(self, current_name: str = "", current_background: str = ""):
        super().__init__()
        self.name.default = current_name
        self.background.default = current_background[:1000]

    async def on_submit(self, interaction: discord.Interaction):
        # Import here to avoid circular dep; conversation_contexts lives in bot
        import bot as bot_module
        success = database.set_character(str(self.name), str(self.background))
        if success:
            bot_module.conversation_contexts.clear()
            embed = discord.Embed(
                title="""✅ 角色已更新""",
                color=discord.Color.gold(),
            )
            embed.add_field(name="🏷️ 名稱", value=str(self.name), inline=True)
            embed.add_field(
                name="📖 背景",
                value=str(self.background)[:1000],
                inline=False,
            )
            embed.set_footer(text="對話歷史已清除以套用新角色。")
            await interaction.response.send_message(embed=embed, ephemeral=True)
        else:
            await interaction.response.send_message(
                "❌ 發生錯誤，請重試。", ephemeral=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# 確認 刪除
# ─────────────────────────────────────────────────────────────────────────────

class 確認刪除View(discord.ui.View):
    """Shown by !forget — requires explicit confirmation before deleting."""

    def __init__(self, entry_id: int, entry_title: str):
        super().__init__(timeout=60)
        self.entry_id = entry_id
        self.entry_title = entry_title

    def _disable_all(self):
        for item in self.children:
            item.disabled = True

    @discord.ui.button(label="刪除", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        success = database.delete_entry(self.entry_id)
        self._disable_all()
        if success:
            await interaction.response.edit_message(
                content=f"🗑️ **{self.entry_title}** (#{self.entry_id}) 已刪除。",
                embed=None,
                view=self,
            )
        else:
            await interaction.response.edit_message(
                content=f"❌ 找不到條目 #{self.entry_id}，可能已被刪除。",
                embed=None,
                view=self,
            )

    @discord.ui.button(label="取消", style=discord.ButtonStyle.secondary, emoji="❌")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        self._disable_all()
        await interaction.response.edit_message(
            content="已取消刪除。", embed=None, view=self
        )

    async def on_timeout(self):
        self._disable_all()


# ─────────────────────────────────────────────────────────────────────────────
# Entry View  (!viewentry)
# ─────────────────────────────────────────────────────────────────────────────

def _build_entry_embed(entry: dict) -> discord.Embed:
    icon = "🖼️" if entry["entry_type"] == "image" else "📝"
    color = discord.Color.teal() if entry["entry_type"] == "image" else discord.Color.blue()
    embed = discord.Embed(
        title=f"{icon} #{entry['id']} — {entry['title']}",
        color=color,
    )
    embed.add_field(name="Type", value=entry["entry_type"].capitalize(), inline=True)
    embed.add_field(name="Created", value=entry["created_at"][:10], inline=True)
    if entry.get("tags"):
        embed.add_field(name="🏷️ Tags", value=entry["tags"], inline=True)
    if entry["entry_type"] == "image":
        desc = entry.get("image_description") or "No description set."
        for i, chunk in enumerate([desc[j:j+1024] for j in range(0, min(len(desc), 2048), 1024)]):
            embed.add_field(
                name="📄 Description" if i == 0 else "📄 Description (cont.)",
                value=chunk,
                inline=False,
            )
    else:
        content = entry.get("content") or "No content."
        for i, chunk in enumerate([content[j:j+1024] for j in range(0, min(len(content), 2048), 1024)]):
            embed.add_field(
                name="📄 Content" if i == 0 else "📄 Content (cont.)",
                value=chunk,
                inline=False,
            )
    return embed


class EntryView(discord.ui.View):
    """Shown by !viewentry — edit or delete the entry."""

    def __init__(self, entry: dict):
        super().__init__(timeout=120)
        self.entry = entry
        self.entry_id = entry["id"]
        self.entry_title = entry["title"]
        self.entry_type = entry["entry_type"]

        if self.entry_type == "image":
            self.add_item(self._make_view_image_btn())
            self.add_item(self._make_edit_desc_btn())
        else:
            self.add_item(self._make_edit_text_btn())
        self.add_item(self._make_delete_btn())

    def _make_view_image_btn(self):
        btn = discord.ui.Button(
            label="查看圖片",
            style=discord.ButtonStyle.secondary,
            emoji="🖼️",
        )

        entry_id = self.entry_id
        entry_title = self.entry_title

        async def callback(interaction: discord.Interaction):
            # Re-fetch from DB to guarantee we have the raw bytes
            row = database.get_entry_by_id(entry_id)
            if not row:
                await interaction.response.send_message(
                    "❌ 找不到此條目，可能已被刪除。", ephemeral=True
                )
                return

            img_data = row.get("image_data")
            if not img_data:
                await interaction.response.send_message(
                    "⚠️ 此條目沒有儲存的圖像資料。", ephemeral=True
                )
                return

            mime = row.get("image_mime") or "image/png"
            ext = mime.split("/")[-1] if "/" in mime else "png"
            # Sanitise: Discord only accepts common image extensions
            if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
                ext = "png"

            filename = f"{entry_title[:40]}.{ext}"
            file = discord.File(io.BytesIO(img_data), filename=filename)
            await interaction.response.send_message(file=file, ephemeral=True)

        btn.callback = callback
        return btn

    def _make_edit_desc_btn(self):
        btn = discord.ui.Button(
            label="編輯描述",
            style=discord.ButtonStyle.primary,
            emoji="✏️",
        )

        async def callback(interaction: discord.Interaction):
            modal = EditDescriptionModal(self.entry_id, self.entry_title)
            await interaction.response.send_modal(modal)

        btn.callback = callback
        return btn

    def _make_edit_text_btn(self):
        btn = discord.ui.Button(
            label="編輯內容",
            style=discord.ButtonStyle.primary,
            emoji="✏️",
        )
        current = self.entry.get("content", "")

        async def callback(interaction: discord.Interaction):
            modal = EditTextModal(self.entry_id, self.entry_title, current)
            await interaction.response.send_modal(modal)

        btn.callback = callback
        return btn

    def _make_delete_btn(self):
        btn = discord.ui.Button(
            label="刪除",
            style=discord.ButtonStyle.danger,
            emoji="🗑️",
        )

        async def callback(interaction: discord.Interaction):
            confirm_view = 確認刪除View(self.entry_id, self.entry_title)
            embed = discord.Embed(
                title="⚠️ 確認刪除",
                description=f"確定要刪除 **{self.entry_title}** (#{self.entry_id})?\n此操作無法撤銷。",
                color=discord.Color.red(),
            )
            await interaction.response.send_message(embed=embed, view=confirm_view, ephemeral=True)

        btn.callback = callback
        return btn


# ─────────────────────────────────────────────────────────────────────────────
# KB Manager View  (!viewentry / !knowledge)
# 4 entries per page; each row: 👁️ 標題(view)  ✏️ 編輯  🗑️ 刪除
# Row 0: ◀️ 上一頁 | 🗂️ 第X/Y頁(disabled) | ▶️ 下一頁
# ─────────────────────────────────────────────────────────────────────────────

KB_PAGE_SIZE = 4


def _build_kb_embed(entries: list, page: int, total_pages: int, total_count: int, query: str) -> discord.Embed:
    header = f"🔍 搜尋：{query}" if query else "📋 所有條目"
    embed = discord.Embed(
        title=f"🗂️ 知識庫管理 — {header}",
        color=discord.Color.blurple(),
    )
    if not entries:
        embed.description = "❌ 沒有找到任何條目。"
        return embed

    lines = []
    for entry in entries:
        icon = "🖼️" if entry["entry_type"] == "image" else "📝"
        val = entry.get("content") or entry.get("image_description") or ""
        preview = val[:80].replace("\n", " ") + ("…" if len(val) > 80 else "")
        lines.append(f"{icon} **#{entry['id']}** {entry['title']}\n　　`{preview}`")

    embed.description = "\n\n".join(lines)
    embed.set_footer(text=f"第 {page + 1} / {total_pages} 頁　共 {total_count} 個條目　👁️查看  ✏️編輯  🗑️刪除")
    return embed


class KBManagerView(discord.ui.View):
    """Full interactive KB manager with per-entry view/edit/delete buttons."""

    def __init__(self, all_entries: list, query: str = "", page: int = 0):
        super().__init__(timeout=180)
        self.all_entries = all_entries
        self.query = query
        self.page = page
        self.total_pages = max(1, (len(all_entries) + KB_PAGE_SIZE - 1) // KB_PAGE_SIZE)
        self._build_buttons()

    def _page_entries(self) -> list:
        start = self.page * KB_PAGE_SIZE
        return self.all_entries[start:start + KB_PAGE_SIZE]

    def _get_embed(self) -> discord.Embed:
        return _build_kb_embed(
            self._page_entries(), self.page, self.total_pages, len(self.all_entries), self.query
        )

    def _build_buttons(self):
        self.clear_items()

        # ── Row 0: navigation ──────────────────────────────────────────
        prev = discord.ui.Button(
            emoji="◀️", label="上一頁",
            style=discord.ButtonStyle.secondary,
            disabled=self.page == 0, row=0,
        )
        prev.callback = self._prev
        self.add_item(prev)

        counter = discord.ui.Button(
            label=f"第 {self.page + 1} / {self.total_pages} 頁",
            style=discord.ButtonStyle.secondary,
            disabled=True, row=0,
        )
        self.add_item(counter)

        nxt = discord.ui.Button(
            emoji="▶️", label="下一頁",
            style=discord.ButtonStyle.secondary,
            disabled=self.page >= self.total_pages - 1, row=0,
        )
        nxt.callback = self._next
        self.add_item(nxt)

        # ── Rows 1-4: one row per entry ────────────────────────────────
        for i, entry in enumerate(self._page_entries()):
            row = i + 1
            icon = "🖼️" if entry["entry_type"] == "image" else "📝"
            short_title = entry["title"][:22]

            view_btn = discord.ui.Button(
                emoji=icon, label=short_title,
                style=discord.ButtonStyle.primary, row=row,
            )
            view_btn.callback = self._make_view_cb(entry)
            self.add_item(view_btn)

            edit_btn = discord.ui.Button(
                emoji="✏️", label="編輯",
                style=discord.ButtonStyle.secondary, row=row,
            )
            edit_btn.callback = self._make_edit_cb(entry)
            self.add_item(edit_btn)

            del_btn = discord.ui.Button(
                emoji="🗑️", label="刪除",
                style=discord.ButtonStyle.danger, row=row,
            )
            del_btn.callback = self._make_delete_cb(entry)
            self.add_item(del_btn)

    # ── Callbacks ──────────────────────────────────────────────────────

    def _make_view_cb(self, entry: dict):
        async def callback(interaction: discord.Interaction):
            full = database.get_entry_by_id(entry["id"])
            if not full:
                await interaction.response.send_message("❌ 條目已不存在。", ephemeral=True)
                return
            embed = _build_entry_embed(full)
            view = EntryView(full)
            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        return callback

    def _make_edit_cb(self, entry: dict):
        async def callback(interaction: discord.Interaction):
            full = database.get_entry_by_id(entry["id"])
            if not full:
                await interaction.response.send_message("❌ 條目已不存在。", ephemeral=True)
                return
            if full["entry_type"] == "image":
                modal = EditDescriptionModal(full["id"], full["title"])
            else:
                modal = EditTextModal(full["id"], full["title"], full.get("content", ""))
            await interaction.response.send_modal(modal)
        return callback

    def _make_delete_cb(self, entry: dict):
        async def callback(interaction: discord.Interaction):
            confirm_view = 確認刪除View(entry["id"], entry["title"])
            embed = discord.Embed(
                title="⚠️ 確認刪除",
                description=f"確定要刪除 **{entry['title']}** (#{entry['id']})?\n此操作無法撤銷。",
                color=discord.Color.red(),
            )
            await interaction.response.send_message(embed=embed, view=confirm_view, ephemeral=True)
        return callback

    async def _prev(self, interaction: discord.Interaction):
        self.page = max(0, self.page - 1)
        self._build_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _next(self, interaction: discord.Interaction):
        self.page = min(self.total_pages - 1, self.page + 1)
        self._build_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True


# Keep KnowledgeView as an alias so !knowledge still works
KnowledgeView = KBManagerView
_build_knowledge_embed = _build_kb_embed


# ─────────────────────────────────────────────────────────────────────────────
# Save Image View  (!saveimage)
# ─────────────────────────────────────────────────────────────────────────────

class SaveImageView(discord.ui.View):
    """Shown after !saveimage — lets the user edit the auto-generated description."""

    def __init__(self, entry_id: int, entry_title: str):
        super().__init__(timeout=120)
        self.entry_id = entry_id
        self.entry_title = entry_title

    @discord.ui.button(label="編輯描述", style=discord.ButtonStyle.primary, emoji="✏️")
    async def edit_desc(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = EditDescriptionModal(self.entry_id, self.entry_title)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="刪除", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def delete_entry(self, interaction: discord.Interaction, button: discord.ui.Button):
        confirm_view = 確認刪除View(self.entry_id, self.entry_title)
        embed = discord.Embed(
            title="⚠️ 確認刪除",
            description=f"刪除 **{self.entry_title}** (#{self.entry_id}) from the knowledge base?",
            color=discord.Color.red(),
        )
        await interaction.response.send_message(embed=embed, view=confirm_view, ephemeral=True)


# ─────────────────────────────────────────────────────────────────────────────
# Remember View  (!remember)
# ─────────────────────────────────────────────────────────────────────────────

class RememberView(discord.ui.View):
    """Shown after !remember — lets the user edit or immediately forget the entry."""

    def __init__(self, entry_id: int, entry_title: str, content: str):
        super().__init__(timeout=120)
        self.entry_id = entry_id
        self.entry_title = entry_title
        self.content = content

    @discord.ui.button(label="編輯", style=discord.ButtonStyle.primary, emoji="✏️")
    async def edit_entry(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = EditTextModal(self.entry_id, self.entry_title, self.content)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="刪除", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def delete_entry(self, interaction: discord.Interaction, button: discord.ui.Button):
        confirm_view = 確認刪除View(self.entry_id, self.entry_title)
        embed = discord.Embed(
            title="⚠️ 確認刪除",
            description=f"刪除 **{self.entry_title}** (#{self.entry_id}) from the knowledge base?",
            color=discord.Color.red(),
        )
        await interaction.response.send_message(embed=embed, view=confirm_view, ephemeral=True)


# ─────────────────────────────────────────────────────────────────────────────
# Character View  (!character)
# ─────────────────────────────────────────────────────────────────────────────

class CharacterView(discord.ui.View):
    """Shown by !character — lets the user edit the bot's character via a modal."""

    def __init__(self, bot_name: str, background: str):
        super().__init__(timeout=120)
        self.bot_name = bot_name
        self.background = background

    @discord.ui.button(label="編輯角色", style=discord.ButtonStyle.primary, emoji="✏️")
    async def edit_character(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = EditCharacterModal(self.bot_name, self.background)
        await interaction.response.send_modal(modal)


# ─────────────────────────────────────────────────────────────────────────────
# Generate View  (!generate)
# ─────────────────────────────────────────────────────────────────────────────

class GenerateView(discord.ui.View):
    """Shown after !generate — lets the user regenerate or save to KB."""

    def __init__(self, prompt: str, img_bytes: bytes, mime_type: str):
        super().__init__(timeout=120)
        self.prompt = prompt
        self.img_bytes = img_bytes
        self.mime_type = mime_type

    @discord.ui.button(label="重新生成", style=discord.ButtonStyle.primary, emoji="🔄")
    async def regenerate(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        result = await cloudflare_ai.generate_image(self.prompt)
        if result and result not in [("API_KEY_ERROR", ""), ("MODEL_ERROR", "")]:
            new_bytes, new_mime = result
            self.img_bytes = new_bytes
            self.mime_type = new_mime
            ext = new_mime.split("/")[-1] if "/" in new_mime else "png"
            file = discord.File(io.BytesIO(new_bytes), filename=f"generated.{ext}")
            embed = discord.Embed(
                title="🎨 生成的圖像",
                description=f"**提示詞:** {self.prompt}",
                color=discord.Color.purple(),
            )
            await interaction.followup.send(embed=embed, file=file, view=GenerateView(self.prompt, new_bytes, new_mime))
        else:
            await interaction.followup.send(
                "❌ 重新生成失敗，請嘗試修改提示詞後再試！", ephemeral=True
            )

    @discord.ui.button(label="保存到知識庫", style=discord.ButtonStyle.success, emoji="💾")
    async def save_to_kb(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = SaveGeneratedModal(self.prompt, self.img_bytes, self.mime_type)
        await interaction.response.send_modal(modal)


class SaveGeneratedModal(discord.ui.Modal, title="保存生成的圖像"):
    title_input = discord.ui.TextInput(
        label="標題",
        placeholder="為這個圖像命名...",
        max_length=100,
    )
    description_input = discord.ui.TextInput(
        label="Description (optional)",
        style=discord.TextStyle.paragraph,
        placeholder="Leave blank to auto-generate a description with Llama 4...",
        required=False,
        max_length=2000,
    )

    def __init__(self, prompt: str, img_bytes: bytes, mime_type: str):
        super().__init__()
        self.prompt = prompt
        self.img_bytes = img_bytes
        self.mime_type = mime_type
        self.title_input.default = prompt[:100]

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(thinking=True)
        title = str(self.title_input)
        desc = str(self.description_input).strip()
        if not desc:
            desc = await groq_ai.understand_image(
                self.img_bytes, self.mime_type,
                "Describe this generated image in detail for a knowledge base entry.",
            )
        entry_id = database.add_image_entry(title, self.img_bytes, self.mime_type, desc)
        embed = discord.Embed(
            title="💾 已保存到知識庫",
            color=discord.Color.green(),
        )
        embed.add_field(name="標題", value=title, inline=True)
        embed.add_field(name="條目編號", value=f"#{entry_id}", inline=True)
        embed.add_field(name="描述", value=desc[:500] + ("..." if len(desc) > 500 else ""), inline=False)
        await interaction.followup.send(embed=embed)


# Alias so bot.py can reference it by a plain ASCII name
ConfirmDeleteView = 確認刪除View
