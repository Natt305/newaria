# ─────────────────────────────────────────────────────────────────────────────
# help_config.py — Edit this file to update /help and /helpsetting text.
#
# USER_HELP_SECTIONS  → shown by /help        (all users, open to everyone)
# ADMIN_HELP_SECTIONS → shown by /helpsetting  (admin / designated role)
#
# Each entry is a (section_name, section_content) tuple.
# The "與我交談" opening field is added dynamically in bot.py using the
# current character name — you don't need to edit it here.
# ─────────────────────────────────────────────────────────────────────────────

from typing import List, Tuple

USER_HELP_SECTIONS: List[Tuple[str, str]] = [
    (
        "🎨 圖像生成",
        "`/generate` 或 `!generate <提示詞>` — 直接生成圖像\n"
        "例如: `/generate 一隻在森林裡的貓`",
    ),
    (
        "🧠 長期記憶",
        "`/memories` — 檢視所有儲存的記憶 (分頁顯示)\n"
        "`/clearmemory` — 清除自己的記憶 (需確認，所有人可用)",
    ),
    (
        "🎭 角色 & 知識庫",
        "`/character` 或 `!character` — 檢視目前角色\n"
        "`/knowledge` 或 `!knowledge [查詢]` — 搜尋/列出知識庫條目\n"
        "`/viewentry` 或 `!viewentry [id]` — 檢視/管理知識庫條目",
    ),
    (
        "ℹ️ 說明",
        "`/help` 或 `!help` — 顯示此說明",
    ),
]

ADMIN_HELP_SECTIONS: List[Tuple[str, str]] = [
    (
        "🎭 角色設定",
        '`/setcharacter` 或 `!setcharacter "名稱" <背景>` — 設定角色名稱與背景\n'
        "`/character` 或 `!character` — 檢視目前角色",
    ),
    (
        "📚 知識庫",
        '`/remember` 或 `!remember "標題" 內容` — 保存文字\n'
        '`/saveimage` 或 `!saveimage "標題" [描述]` — 保存圖像\n'
        "`/setdesc` 或 `!setdesc <id> <描述>` — 更新圖像條目描述\n"
        "`/viewentry` 或 `!viewentry [id]` — 檢視/管理條目\n"
        "`/knowledge` 或 `!knowledge [查詢]` — 搜尋/列出條目\n"
        "`/forget` 或 `!forget <id>` — 刪除條目",
    ),
    (
        "🧠 長期記憶",
        "`/memory` — 開啟或關閉主動記憶 (每次對話自動注入)\n"
        "`/memorylength <數字>` — 設定每次注入的記憶條數 (預設 20)\n"
        "`/passivememory` — 開啟或關閉被動記憶 (詢問時搜索深層記憶)\n"
        "`/passivememorylength <數字>` — 設定深層記憶搜索條數 (預設 50)\n"
        "`/memories` — 檢視所有儲存的記憶 (分頁顯示)\n"
        "`/clearmemory` — 清除自己的記憶 (所有人可用)\n"
        "`/clearmemory @用戶` — 清除指定用戶的記憶 (需管理員)\n"
        "`/clearmemory all:True` — 清除所有人的記憶 (需管理員)",
    ),
    (
        "💬 建議按鈕",
        "`/suggestions` — 開啟或關閉建議按鈕\n"
        "`/setsuggestionprompt <提示詞>` — 自訂建議生成提示詞\n"
        "`/clearsuggestionprompt` — 恢復預設建議提示詞",
    ),
    (
        "🤖 機器人狀態",
        "`/setstatus <文字> [類型] [狀態]` — 設定機器人自訂狀態\n"
        "`/clearstatus` — 清除自訂狀態，恢復預設\n"
        "`/setthinking <文字>` — 設定思考泡泡 (Discord「What's on your mind?」)\n"
        "`/clearthinking` — 清除思考泡泡\n"
        "`/clear` 或 `!clear` — 清除頻道對話歷史",
    ),
    (
        "🔑 權限管理 (需管理員)",
        "`/permissions` — 檢視所有指令的目前權限設定\n"
        "`/setrole <指令> <@角色>` — 將指令開放給指定角色\n"
        "`/clearrole <指令>` — 移除指令的角色限制，開放給所有人",
    ),
]
