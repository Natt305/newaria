"""
Interactive Discord UI components for bot commands.
Views and Modals used by command responses — NOT used for regular chat.
"""
import io
from typing import Optional
import discord
import database
import cloudflare_ai
import ai_backend as groq_ai

PAGE_SIZE = 5


def _has_saveimage_permission(interaction: discord.Interaction) -> bool:
    """Return True if the interaction user may save to the knowledge base."""
    gate = database.get_command_roles().get("saveimage")
    member = interaction.user

    # No gate set — open to everyone
    if gate is None:
        return True

    # Must be in a guild with a real Member object to check roles/perms
    if not interaction.guild or not isinstance(member, discord.Member):
        return False

    # Server admins and Manage Guild always bypass
    if member.guild_permissions.administrator or member.guild_permissions.manage_guild:
        return True

    # Admin-only gate — already failed above
    if gate == "__admin__":
        return False

    # Role name gate — check if user has the role (case-insensitive)
    return any(r.name.lower() == gate.lower() for r in member.roles)


# ─────────────────────────────────────────────────────────────────────────────
# Modals
# ─────────────────────────────────────────────────────────────────────────────

class EditDescriptionModal(discord.ui.Modal, title="編輯描述"):
    description = discord.ui.TextInput(
        label="描述",
        style=discord.TextStyle.paragraph,
        placeholder="輸入新的圖像描述...",
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
                f"✅ 描述已更新：**{self.entry_title}** (#{self.entry_id})",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "❌ 更新失敗，條目可能已不存在。", ephemeral=True
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
                f"✅ 內容已更新：**{self.entry_title}** (#{self.entry_id})",
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "❌ 更新失敗，條目可能已不存在。", ephemeral=True
            )


class EditCharacterModal(discord.ui.Modal, title="編輯角色"):
    name = discord.ui.TextInput(
        label="名稱",
        placeholder="例如: 少女樂團機器人",
        max_length=50,
    )
    background = discord.ui.TextInput(
        label="背景",
        style=discord.TextStyle.paragraph,
        placeholder="描述角色的身份與背景...",
        max_length=4000,
        required=False,
    )
    personality = discord.ui.TextInput(
        label="個性 / 說話風格",
        style=discord.TextStyle.paragraph,
        placeholder="描述說話語氣、習慣用語、個性特徵...",
        max_length=4000,
        required=False,
    )
    looks = discord.ui.TextInput(
        label="外貌描述",
        style=discord.TextStyle.paragraph,
        placeholder="描述角色的外表、髮色、眼睛、服裝風格等...",
        max_length=4000,
        required=False,
    )

    def __init__(self, current_name: str = "", current_background: str = "", current_personality: str = "", current_looks: str = ""):
        super().__init__()
        self.name.default = current_name[:50]
        self.background.default = current_background[:4000]
        self.personality.default = current_personality[:4000]
        self.looks.default = current_looks[:4000]

    async def on_submit(self, interaction: discord.Interaction):
        import bot as bot_module
        new_name = str(self.name)
        new_bg = str(self.background)
        new_personality = str(self.personality)
        new_looks = str(self.looks)
        success = database.set_character(new_name, new_bg, new_personality, new_looks)
        if success:
            bot_module.conversation_contexts.clear()
            image_count = database.get_character_image_count()
            new_embed = build_char_embed(
                new_name, new_bg, new_personality, new_looks,
                tab="background", bg_page=0,
                title="✅ 角色已更新",
                footer="對話歷史已清除以套用新角色。點擊下方按鈕可繼續編輯。",
                image_count=image_count,
            )
            new_view = CharacterView(new_name, new_bg, new_personality, new_looks, image_count=image_count)
            await interaction.response.edit_message(embed=new_embed, view=new_view)
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
            import bot as _bot_module
            _bot_module._invalidate_kb_title_index()
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

_CONTENT_PAGE_SIZE = 1024


def _content_chunks(entry: dict) -> list:
    """Split entry content/description into 1024-char pages for pagination."""
    if entry["entry_type"] == "image":
        text = entry.get("image_description") or "（尚未設定描述）"
    else:
        text = entry.get("content") or "（尚未設定內容）"
    return [text[i:i + _CONTENT_PAGE_SIZE] for i in range(0, max(1, len(text)), _CONTENT_PAGE_SIZE)]


_ENTRY_TYPE_LABELS = {"image": "圖像", "text": "文字"}


def _build_entry_embed(entry: dict, content_page: int = 0) -> discord.Embed:
    icon = "🖼️" if entry["entry_type"] == "image" else "📝"
    color = discord.Color.teal() if entry["entry_type"] == "image" else discord.Color.blue()
    embed = discord.Embed(
        title=f"{icon} #{entry['id']} — {entry['title']}",
        color=color,
    )
    embed.add_field(name="類型", value=_ENTRY_TYPE_LABELS.get(entry["entry_type"], entry["entry_type"]), inline=True)
    embed.add_field(name="建立時間", value=entry["created_at"][:10], inline=True)
    if entry.get("tags"):
        embed.add_field(name="🏷️ 標籤", value=entry["tags"], inline=True)

    chunks = _content_chunks(entry)
    total = len(chunks)
    page = max(0, min(content_page, total - 1))

    field_name = "📄 描述" if entry["entry_type"] == "image" else "📄 內容"
    if total > 1:
        field_name += f"　第 {page + 1} / {total} 頁"

    embed.add_field(name=field_name, value=chunks[page], inline=False)
    return embed


class EntryView(discord.ui.View):
    """Shown by !viewentry — edit or delete the entry.

    When the content/description exceeds 1024 characters, ◀ / ▶ buttons are
    added so the user can page through it without leaving the view.
    """

    def __init__(self, entry: dict, parent_kb_view=None):
        super().__init__(timeout=300)
        self.entry = entry
        self.entry_id = entry["id"]
        self.entry_title = entry["title"]
        self.entry_type = entry["entry_type"]
        self.parent_kb_view = parent_kb_view
        self.content_page = 0
        self.content_total = len(_content_chunks(entry))
        self._build_buttons()

    # ── embed ─────────────────────────────────────────────────────────────────

    def _get_embed(self) -> discord.Embed:
        return _build_entry_embed(self.entry, self.content_page)

    # ── button layout ─────────────────────────────────────────────────────────

    def _build_buttons(self):
        self.clear_items()

        # Row 0: content pagination (only shown when content > 1 page)
        if self.content_total > 1:
            prev_btn = discord.ui.Button(
                emoji="◀️", label="上一頁",
                style=discord.ButtonStyle.secondary,
                disabled=self.content_page == 0, row=0,
            )
            prev_btn.callback = self._prev_page

            counter_btn = discord.ui.Button(
                label=f"第 {self.content_page + 1} / {self.content_total} 頁",
                style=discord.ButtonStyle.secondary,
                disabled=True, row=0,
            )

            next_btn = discord.ui.Button(
                emoji="▶️", label="下一頁",
                style=discord.ButtonStyle.secondary,
                disabled=self.content_page >= self.content_total - 1, row=0,
            )
            next_btn.callback = self._next_page

            self.add_item(prev_btn)
            self.add_item(counter_btn)
            self.add_item(next_btn)

        # Row 1: action buttons
        action_row = 1 if self.content_total > 1 else 0
        if self.entry_type == "image":
            view_btn = self._make_view_image_btn(row=action_row)
            edit_btn = self._make_edit_desc_btn(row=action_row)
            self.add_item(view_btn)
            self.add_item(edit_btn)
        else:
            self.add_item(self._make_edit_text_btn(row=action_row))
        self.add_item(self._make_delete_btn(row=action_row))

        # Row 2: back button (optional)
        if self.parent_kb_view is not None:
            back_row = action_row + 1
            back_btn = discord.ui.Button(
                label="← 返回列表",
                style=discord.ButtonStyle.secondary,
                emoji="📋",
                row=back_row,
            )
            back_btn.callback = self._back
            self.add_item(back_btn)

    # ── navigation callbacks ──────────────────────────────────────────────────

    async def _prev_page(self, interaction: discord.Interaction):
        self.content_page = max(0, self.content_page - 1)
        self._build_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _next_page(self, interaction: discord.Interaction):
        self.content_page = min(self.content_total - 1, self.content_page + 1)
        self._build_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _back(self, interaction: discord.Interaction):
        if self.parent_kb_view is None:
            await interaction.response.defer()
            return
        self.parent_kb_view._build_buttons()
        await interaction.response.edit_message(
            embed=self.parent_kb_view._get_embed(),
            attachments=[],
            view=self.parent_kb_view,
        )

    def _make_view_image_btn(self, row: int = 0):
        btn = discord.ui.Button(
            label="查看圖片",
            style=discord.ButtonStyle.secondary,
            emoji="🖼️",
            row=row,
        )

        entry_id = self.entry_id
        entry_title = self.entry_title

        async def callback(interaction: discord.Interaction):
            try:
                image_count = database.get_entry_image_count(entry_id)
                if image_count == 0:
                    await interaction.response.send_message(
                        "⚠️ 此條目沒有儲存的圖像資料。", ephemeral=True
                    )
                    return
                can_remove = (
                    interaction.guild is not None
                    and isinstance(interaction.user, discord.Member)
                    and interaction.user.guild_permissions.administrator
                )
                gallery = ImageGalleryView(entry_id, entry_title, image_count, can_remove)
                await gallery.send_first(interaction)
            except discord.HTTPException as e:
                print(f"[View] 查看圖片 HTTP error for entry #{entry_id}: {e.status} {e.text}")
                msg = "❌ 圖片太大，無法在 Discord 上顯示（超過 8MB 上限）。" if e.status == 413 else f"❌ Discord 錯誤: {e.text}"
                try:
                    if interaction.response.is_done():
                        await interaction.followup.send(msg, ephemeral=True)
                    else:
                        await interaction.response.send_message(msg, ephemeral=True)
                except Exception:
                    pass
            except Exception as e:
                print(f"[View] 查看圖片 error for entry #{entry_id}: {type(e).__name__}: {e}")
                try:
                    if interaction.response.is_done():
                        await interaction.followup.send(f"❌ 讀取圖片時發生錯誤: {type(e).__name__}", ephemeral=True)
                    else:
                        await interaction.response.send_message(f"❌ 讀取圖片時發生錯誤: {type(e).__name__}", ephemeral=True)
                except Exception:
                    pass

        btn.callback = callback
        return btn

    def _make_edit_desc_btn(self, row: int = 0):
        btn = discord.ui.Button(
            label="編輯描述",
            style=discord.ButtonStyle.primary,
            emoji="✏️",
            row=row,
        )

        async def callback(interaction: discord.Interaction):
            modal = EditDescriptionModal(self.entry_id, self.entry_title)
            await interaction.response.send_modal(modal)

        btn.callback = callback
        return btn

    def _make_edit_text_btn(self, row: int = 0):
        btn = discord.ui.Button(
            label="編輯內容",
            style=discord.ButtonStyle.primary,
            emoji="✏️",
            row=row,
        )
        current = self.entry.get("content", "")

        async def callback(interaction: discord.Interaction):
            modal = EditTextModal(self.entry_id, self.entry_title, current)
            await interaction.response.send_modal(modal)

        btn.callback = callback
        return btn

    def _make_delete_btn(self, row: int = 0):
        btn = discord.ui.Button(
            label="刪除",
            style=discord.ButtonStyle.danger,
            emoji="🗑️",
            row=row,
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


# ─────────────────────────────────────────────────────────────────────────────
# Image Gallery View  (shown when clicking 查看圖片 in EntryView)
# ─────────────────────────────────────────────────────────────────────────────

class ImageGalleryView(discord.ui.View):
    """Cycling image viewer for multi-image KB entries."""

    def __init__(self, entry_id: int, entry_title: str, image_count: int, can_remove: bool, current_index: int = 1):
        super().__init__(timeout=180)
        self.entry_id = entry_id
        self.entry_title = entry_title
        self.image_count = image_count
        self.current_index = current_index
        self.can_remove = can_remove
        self._build_buttons()

    def _build_buttons(self):
        self.clear_items()

        prev_btn = discord.ui.Button(
            emoji="◀️",
            style=discord.ButtonStyle.secondary,
            disabled=(self.image_count <= 1 or self.current_index <= 1),
            row=0,
        )
        prev_btn.callback = self._prev
        self.add_item(prev_btn)

        counter_btn = discord.ui.Button(
            label=f"{self.current_index} / {self.image_count}",
            style=discord.ButtonStyle.secondary,
            disabled=True,
            row=0,
        )
        self.add_item(counter_btn)

        next_btn = discord.ui.Button(
            emoji="▶️",
            style=discord.ButtonStyle.secondary,
            disabled=(self.image_count <= 1 or self.current_index >= self.image_count),
            row=0,
        )
        next_btn.callback = self._next
        self.add_item(next_btn)

        if self.can_remove:
            remove_btn = discord.ui.Button(
                emoji="🗑️",
                label="移除此圖",
                style=discord.ButtonStyle.danger,
                row=1,
            )
            remove_btn.callback = self._remove
            self.add_item(remove_btn)

    def _make_file(self) -> Optional[discord.File]:
        result = database.get_entry_image(self.entry_id, self.current_index)
        if not result:
            return None
        img_data, mime = result
        ext = mime.split("/")[-1] if "/" in mime else "png"
        if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
            ext = "png"
        return discord.File(io.BytesIO(img_data), filename=f"image_{self.current_index}.{ext}")

    def _content(self) -> str:
        return f"🖼️ **{self.entry_title}** — 圖片 {self.current_index} / {self.image_count}"

    async def send_first(self, interaction: discord.Interaction):
        try:
            file = self._make_file()
        except Exception as e:
            print(f"[View] ImageGalleryView._make_file error: {e}")
            await interaction.response.send_message("❌ 讀取圖片資料時發生錯誤。", ephemeral=True)
            return
        if not file:
            await interaction.response.send_message("❌ 無法讀取此圖片。", ephemeral=True)
            return
        try:
            await interaction.response.send_message(
                content=self._content(), file=file, view=self, ephemeral=True
            )
        except discord.HTTPException as e:
            print(f"[View] ImageGalleryView.send_first HTTP error: {e.status} {e.text}")
            if e.status == 413:
                await interaction.followup.send("❌ 圖片太大，無法在 Discord 上顯示（超過 8MB 上限）。", ephemeral=True)
            else:
                await interaction.followup.send(f"❌ Discord 錯誤: {e.text}", ephemeral=True)

    async def _prev(self, interaction: discord.Interaction):
        self.current_index = max(1, self.current_index - 1)
        self._build_buttons()
        file = self._make_file()
        try:
            await interaction.response.defer()
            if not file:
                await interaction.edit_original_response(content="❌ 無法讀取此圖片。", attachments=[], view=self)
                return
            await interaction.edit_original_response(content=self._content(), attachments=[file], view=self)
        except Exception as e:
            print(f"[View] ImageGalleryView._prev error: {e}")
            try:
                await interaction.followup.send("❌ 切換圖片時發生錯誤，請稍後再試。", ephemeral=True)
            except Exception:
                pass

    async def _next(self, interaction: discord.Interaction):
        self.current_index = min(self.image_count, self.current_index + 1)
        self._build_buttons()
        file = self._make_file()
        try:
            await interaction.response.defer()
            if not file:
                await interaction.edit_original_response(content="❌ 無法讀取此圖片。", attachments=[], view=self)
                return
            await interaction.edit_original_response(content=self._content(), attachments=[file], view=self)
        except Exception as e:
            print(f"[View] ImageGalleryView._next error: {e}")
            try:
                await interaction.followup.send("❌ 切換圖片時發生錯誤，請稍後再試。", ephemeral=True)
            except Exception:
                pass

    async def _remove(self, interaction: discord.Interaction):
        success, msg = database.remove_image_from_entry(self.entry_id, self.current_index)
        if not success:
            await interaction.response.send_message(f"❌ {msg}", ephemeral=True)
            return
        self.image_count -= 1
        if self.image_count == 0:
            await interaction.response.edit_message(
                content=f"🗑️ 已移除最後一張圖片。**{self.entry_title}** 目前沒有任何圖片。",
                attachments=[],
                view=None,
            )
            return
        self.current_index = min(self.current_index, self.image_count)
        self._build_buttons()
        file = self._make_file()
        try:
            await interaction.response.defer()
            if not file:
                await interaction.edit_original_response(
                    content=f"🗑️ 已移除圖片。剩餘 {self.image_count} 張。", attachments=[], view=self
                )
                return
            await interaction.edit_original_response(
                content=self._content(), attachments=[file], view=self
            )
        except Exception as e:
            print(f"[View] ImageGalleryView._remove error: {e}")
            try:
                await interaction.followup.send("❌ 操作時發生錯誤，請稍後再試。", ephemeral=True)
            except Exception:
                pass

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True


# ─────────────────────────────────────────────────────────────────────────────
# KB Manager View  (!viewentry / !knowledge)
# 4 entries per page; each row: 👁️ 標題(view)  ✏️ 編輯  🗑️ 刪除
# Row 0: ◀️ 上一頁 | 🗂️ 第X/Y頁(disabled) | ▶️ 下一頁
# ─────────────────────────────────────────────────────────────────────────────

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
        parent = self
        async def callback(interaction: discord.Interaction):
            full = database.get_entry_by_id(entry["id"])
            if not full:
                await interaction.response.send_message("❌ 條目已不存在。", ephemeral=True)
                return
            embed = _build_entry_embed(full)
            view = EntryView(full, parent_kb_view=parent)
            await interaction.response.edit_message(embed=embed, attachments=[], view=view)
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
        super().__init__(timeout=300)
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
            description=f"確定要從知識庫刪除 **{self.entry_title}** (#{self.entry_id})？\n此操作無法撤銷。",
            color=discord.Color.red(),
        )
        await interaction.response.send_message(embed=embed, view=confirm_view, ephemeral=True)


# ─────────────────────────────────────────────────────────────────────────────
# Remember View  (!remember)
# ─────────────────────────────────────────────────────────────────────────────

class RememberView(discord.ui.View):
    """Shown after !remember — lets the user edit or immediately forget the entry."""

    def __init__(self, entry_id: int, entry_title: str, content: str):
        super().__init__(timeout=300)
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
            description=f"確定要從知識庫刪除 **{self.entry_title}** (#{self.entry_id})？\n此操作無法撤銷。",
            color=discord.Color.red(),
        )
        await interaction.response.send_message(embed=embed, view=confirm_view, ephemeral=True)


# ─────────────────────────────────────────────────────────────────────────────
# Character View  (!character)
# ─────────────────────────────────────────────────────────────────────────────

class CharacterGalleryView(discord.ui.View):
    """Cycling viewer for character reference images."""

    def __init__(self, image_count: int, can_remove: bool, current_index: int = 1):
        super().__init__(timeout=180)
        self.image_count = image_count
        self.current_index = current_index
        self.can_remove = can_remove
        self._build_buttons()

    def _build_buttons(self):
        self.clear_items()

        prev_btn = discord.ui.Button(
            emoji="◀️",
            style=discord.ButtonStyle.secondary,
            disabled=(self.image_count <= 1 or self.current_index <= 1),
            row=0,
        )
        prev_btn.callback = self._prev
        self.add_item(prev_btn)

        counter_btn = discord.ui.Button(
            label=f"{self.current_index} / {self.image_count}",
            style=discord.ButtonStyle.secondary,
            disabled=True,
            row=0,
        )
        self.add_item(counter_btn)

        next_btn = discord.ui.Button(
            emoji="▶️",
            style=discord.ButtonStyle.secondary,
            disabled=(self.image_count <= 1 or self.current_index >= self.image_count),
            row=0,
        )
        next_btn.callback = self._next
        self.add_item(next_btn)

        if self.can_remove:
            remove_btn = discord.ui.Button(
                emoji="🗑️",
                label="移除此圖",
                style=discord.ButtonStyle.danger,
                row=1,
            )
            remove_btn.callback = self._remove
            self.add_item(remove_btn)

    def _make_file(self) -> Optional[discord.File]:
        result = database.get_character_image(self.current_index)
        if not result:
            return None
        img_data, mime = result
        ext = mime.split("/")[-1] if "/" in mime else "png"
        if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
            ext = "png"
        return discord.File(io.BytesIO(img_data), filename=f"char_{self.current_index}.{ext}")

    def _content(self) -> str:
        images = database.get_character_images_meta()
        desc = ""
        if 1 <= self.current_index <= len(images):
            desc = (images[self.current_index - 1].get("description") or "").strip()
        label = f"🖼️ 外貌圖庫 — 圖片 {self.current_index} / {self.image_count}"
        if desc:
            label += f"\n> {desc[:200]}"
        return label

    async def send_first(self, interaction: discord.Interaction):
        file = self._make_file()
        if not file:
            await interaction.response.send_message("❌ 無法讀取此圖片。", ephemeral=True)
            return
        await interaction.response.send_message(
            content=self._content(), file=file, view=self, ephemeral=True
        )

    async def _prev(self, interaction: discord.Interaction):
        self.current_index = max(1, self.current_index - 1)
        self._build_buttons()
        file = self._make_file()
        try:
            await interaction.response.defer()
            if not file:
                await interaction.edit_original_response(content="❌ 無法讀取此圖片。", attachments=[], view=self)
                return
            await interaction.edit_original_response(content=self._content(), attachments=[file], view=self)
        except Exception as e:
            print(f"[View] CharacterGalleryView._prev error: {e}")
            try:
                await interaction.followup.send("❌ 切換圖片時發生錯誤，請稍後再試。", ephemeral=True)
            except Exception:
                pass

    async def _next(self, interaction: discord.Interaction):
        self.current_index = min(self.image_count, self.current_index + 1)
        self._build_buttons()
        file = self._make_file()
        try:
            await interaction.response.defer()
            if not file:
                await interaction.edit_original_response(content="❌ 無法讀取此圖片。", attachments=[], view=self)
                return
            await interaction.edit_original_response(content=self._content(), attachments=[file], view=self)
        except Exception as e:
            print(f"[View] CharacterGalleryView._next error: {e}")
            try:
                await interaction.followup.send("❌ 切換圖片時發生錯誤，請稍後再試。", ephemeral=True)
            except Exception:
                pass

    async def _remove(self, interaction: discord.Interaction):
        success, msg = database.remove_character_image(self.current_index)
        if not success:
            await interaction.response.send_message(f"❌ {msg}", ephemeral=True)
            return
        import bot as _bot_module
        _bot_module._invalidate_char_images_ctx()
        self.image_count -= 1
        if self.image_count == 0:
            await interaction.response.edit_message(
                content="🗑️ 已移除最後一張角色圖片。目前沒有外貌參考圖。",
                attachments=[],
                view=None,
            )
            return
        self.current_index = min(self.current_index, self.image_count)
        self._build_buttons()
        file = self._make_file()
        try:
            await interaction.response.defer()
            if not file:
                await interaction.edit_original_response(
                    content=f"🗑️ 已移除圖片。剩餘 {self.image_count} 張。", attachments=[], view=self
                )
                return
            await interaction.edit_original_response(
                content=self._content(), attachments=[file], view=self
            )
        except Exception as e:
            print(f"[View] CharacterGalleryView._remove error: {e}")
            try:
                await interaction.followup.send("❌ 操作時發生錯誤，請稍後再試。", ephemeral=True)
            except Exception:
                pass

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True


def _bg_chunks(background: str) -> list:
    """Split character background into 1024-char pages for pagination."""
    text = background or "（未設定背景）"
    return [text[i:i + _CONTENT_PAGE_SIZE] for i in range(0, max(1, len(text)), _CONTENT_PAGE_SIZE)]


def _personality_chunks(personality: str) -> list:
    """Split personality into 1024-char pages for pagination."""
    text = personality or "（未設定個性）"
    return [text[i:i + _CONTENT_PAGE_SIZE] for i in range(0, max(1, len(text)), _CONTENT_PAGE_SIZE)]


def _looks_chunks(looks: str) -> list:
    """Split looks into 1024-char pages for pagination."""
    text = looks or "（未設定外貌）"
    return [text[i:i + _CONTENT_PAGE_SIZE] for i in range(0, max(1, len(text)), _CONTENT_PAGE_SIZE)]


def build_char_embed(
    bot_name: str,
    background: str,
    personality: str = "",
    looks: str = "",
    tab: str = "background",
    bg_page: int = 0,
    image_count: int = 0,
    title: str = "🎭 目前角色",
    color: discord.Color = discord.Color.gold(),
    footer: str = "點擊下方按鈕可編輯角色設定或瀏覽外貌圖庫。",
) -> discord.Embed:
    """Build the character profile embed. tab='background', 'personality', or 'looks'."""
    embed = discord.Embed(title=title, color=color)
    embed.add_field(name="🏷️ 名稱", value=bot_name, inline=True)
    if image_count:
        embed.add_field(name="🖼️ 外貌圖片", value=f"{image_count} 張", inline=True)

    if tab == "personality":
        chunks = _personality_chunks(personality)
        total = len(chunks)
        page = max(0, min(bg_page, total - 1))
        field_name = "💬 個性 / 說話風格"
        if total > 1:
            field_name += f"　第 {page + 1} / {total} 頁"
        embed.add_field(name=field_name, value=chunks[page], inline=False)
    elif tab == "looks":
        chunks = _looks_chunks(looks)
        total = len(chunks)
        page = max(0, min(bg_page, total - 1))
        field_name = "🎨 外貌描述"
        if total > 1:
            field_name += f"　第 {page + 1} / {total} 頁"
        embed.add_field(name=field_name, value=chunks[page], inline=False)
    else:
        chunks = _bg_chunks(background)
        total = len(chunks)
        page = max(0, min(bg_page, total - 1))
        field_name = "📖 背景"
        if total > 1:
            field_name += f"　第 {page + 1} / {total} 頁"
        embed.add_field(name=field_name, value=chunks[page], inline=False)

    if footer:
        embed.set_footer(text=footer)
    return embed


class CharacterView(discord.ui.View):
    """Shown by /character — lets the user browse background, personality, and looks tabs,
    paginate long text, edit all fields via modal, and open the image gallery.

    Layout:
      Row 0: 📖 背景 | 💬 個性 | 🎨 外貌  (tab selector)
      Row 1: ◀ 上一頁 | 第 x/y 頁 | 下一頁  (only when > 1 page)
      Row 2: ✏️ 編輯角色 | 🖼️ 外貌圖庫
    """

    def __init__(self, bot_name: str, background: str, personality: str = "", looks: str = "", image_count: int = 0):
        super().__init__(timeout=300)
        self.bot_name = bot_name
        self.background = background
        self.personality = personality
        self.looks = looks
        self.image_count = image_count
        self.tab = "background"
        self.bg_page = 0
        self._build_buttons()

    def _current_chunks(self) -> list:
        if self.tab == "personality":
            return _personality_chunks(self.personality)
        if self.tab == "looks":
            return _looks_chunks(self.looks)
        return _bg_chunks(self.background)

    def _get_embed(self) -> discord.Embed:
        return build_char_embed(
            self.bot_name, self.background, self.personality, self.looks,
            tab=self.tab, bg_page=self.bg_page, image_count=self.image_count,
        )

    def _build_buttons(self):
        self.clear_items()
        chunks = self._current_chunks()
        total = len(chunks)

        # Row 0: tab selector
        bg_tab = discord.ui.Button(
            label="📖 背景",
            style=discord.ButtonStyle.primary if self.tab == "background" else discord.ButtonStyle.secondary,
            row=0,
        )
        bg_tab.callback = self._switch_background
        self.add_item(bg_tab)

        pers_tab = discord.ui.Button(
            label="💬 個性",
            style=discord.ButtonStyle.primary if self.tab == "personality" else discord.ButtonStyle.secondary,
            row=0,
        )
        pers_tab.callback = self._switch_personality
        self.add_item(pers_tab)

        looks_tab = discord.ui.Button(
            label="🎨 外貌",
            style=discord.ButtonStyle.primary if self.tab == "looks" else discord.ButtonStyle.secondary,
            row=0,
        )
        looks_tab.callback = self._switch_looks
        self.add_item(looks_tab)

        # Row 1: pagination (only when > 1 page)
        if total > 1:
            prev_btn = discord.ui.Button(
                emoji="◀️", label="上一頁",
                style=discord.ButtonStyle.secondary,
                disabled=self.bg_page == 0, row=1,
            )
            prev_btn.callback = self._prev_page

            counter_btn = discord.ui.Button(
                label=f"第 {self.bg_page + 1} / {total} 頁",
                style=discord.ButtonStyle.secondary,
                disabled=True, row=1,
            )

            next_btn = discord.ui.Button(
                emoji="▶️", label="下一頁",
                style=discord.ButtonStyle.secondary,
                disabled=self.bg_page >= total - 1, row=1,
            )
            next_btn.callback = self._next_page

            self.add_item(prev_btn)
            self.add_item(counter_btn)
            self.add_item(next_btn)

        # Row 2: action buttons
        edit_btn = discord.ui.Button(
            label="編輯角色", style=discord.ButtonStyle.primary, emoji="✏️", row=2,
        )
        edit_btn.callback = self._edit_character
        self.add_item(edit_btn)

        if self.image_count >= 1:
            gallery_btn = discord.ui.Button(
                label="外貌圖庫", style=discord.ButtonStyle.secondary, emoji="🖼️", row=2,
            )
            gallery_btn.callback = self._open_gallery
            self.add_item(gallery_btn)

    async def _switch_background(self, interaction: discord.Interaction):
        self.tab = "background"
        self.bg_page = 0
        self._build_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _switch_personality(self, interaction: discord.Interaction):
        self.tab = "personality"
        self.bg_page = 0
        self._build_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _switch_looks(self, interaction: discord.Interaction):
        self.tab = "looks"
        self.bg_page = 0
        self._build_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _prev_page(self, interaction: discord.Interaction):
        self.bg_page = max(0, self.bg_page - 1)
        self._build_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _next_page(self, interaction: discord.Interaction):
        total = len(self._current_chunks())
        self.bg_page = min(total - 1, self.bg_page + 1)
        self._build_buttons()
        await interaction.response.edit_message(embed=self._get_embed(), view=self)

    async def _edit_character(self, interaction: discord.Interaction):
        try:
            modal = EditCharacterModal(self.bot_name, self.background, self.personality, self.looks)
            await interaction.response.send_modal(modal)
        except Exception as e:
            print(f"[View] EditCharacter modal error: {type(e).__name__}: {e}")
            try:
                if interaction.response.is_done():
                    await interaction.followup.send(f"❌ 開啟編輯器時發生錯誤: {type(e).__name__}", ephemeral=True)
                else:
                    await interaction.response.send_message(f"❌ 開啟編輯器時發生錯誤: {type(e).__name__}", ephemeral=True)
            except Exception:
                pass

    async def _open_gallery(self, interaction: discord.Interaction):
        can_remove = (
            interaction.guild is not None
            and isinstance(interaction.user, discord.Member)
            and interaction.user.guild_permissions.administrator
        )
        gallery = CharacterGalleryView(self.image_count, can_remove)
        await gallery.send_first(interaction)


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
        if not _has_saveimage_permission(interaction):
            await interaction.response.send_message(
                "🔒 你沒有權限將圖像保存到知識庫。", ephemeral=True
            )
            return
        modal = SaveGeneratedModal(self.prompt, self.img_bytes, self.mime_type)
        await interaction.response.send_modal(modal)


class SaveGeneratedModal(discord.ui.Modal, title="保存生成的圖像"):
    title_input = discord.ui.TextInput(
        label="標題",
        placeholder="為這個圖像命名...",
        max_length=100,
    )
    description_input = discord.ui.TextInput(
        label="描述（選填）",
        style=discord.TextStyle.paragraph,
        placeholder="留空則自動生成描述...",
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
        if not _has_saveimage_permission(interaction):
            await interaction.response.send_message(
                "🔒 你沒有權限將圖像保存到知識庫。", ephemeral=True
            )
            return
        await interaction.response.defer(thinking=True)
        title = str(self.title_input)
        desc = str(self.description_input).strip()
        if not desc:
            desc = await groq_ai.understand_image(
                self.img_bytes, self.mime_type,
                "Describe this generated image in detail for a knowledge base entry.",
            )
        entry_id = database.add_image_entry(title, self.img_bytes, self.mime_type, desc)
        import bot as _bot_module
        _bot_module._invalidate_kb_title_index()
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
