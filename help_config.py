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
        "`/generate <提示詞>` 或 `!generate <提示詞>`\n"
        "使用 Cloudflare Flux AI 直接生成圖像。\n"
        "例如: `/generate 夕陽下的貓咪在草原上`",
    ),
    (
        "🧠 清除記憶",
        "`/clearmemory` — 清除你自己的長期記憶 (需二次確認)",
    ),
    (
        "ℹ️ 說明",
        "`/help` 或 `!help` — 顯示此說明頁面",
    ),
]

ADMIN_HELP_SECTIONS: List[Tuple[str, str]] = [
    (
        "🎭 角色設定",
        '`/setcharacter "名稱" <背景> [個性] [外貌]` 或 `!setcharacter "名稱" <背景> [個性] [外貌]`\n'
        "　設定角色名稱、背景、個性說話風格與外貌描述。\n"
        "`/character` 或 `!character`\n"
        "　查看目前角色設定；可在圖庫中新增/移除外貌圖。\n"
        "`/addcharimage [attachment 1–10]` 或 `!addcharimage (附上圖片)`\n"
        "　批量新增角色外貌參考圖，最多 10 張；AI 自動分析外貌並生成描述。",
    ),
    (
        "📚 知識庫 — 文字",
        '`/remember "標題" <內容>` 或 `!remember "標題" <內容>`\n'
        "　將文字資料存入知識庫，供機器人對話時參考。\n"
        "`/forget <id>` 或 `!forget <id>`\n"
        "　刪除指定條目 (需二次確認)。\n"
        "`/viewentry [id]` 或 `!viewentry [id]`\n"
        "　查看條目詳情；不填 id 則開啟互動式管理器。\n"
        "`/knowledge [查詢]` 或 `!knowledge [查詢]`\n"
        "　搜尋或分頁瀏覽所有條目。",
    ),
    (
        "🖼️ 知識庫 — 圖像",
        '`/saveimage "標題" [描述] [attachment 1–5]` 或 `!saveimage "標題" [描述]` (附上圖片)\n'
        "　批量新增圖像條目，最多 5 張；未填描述時 AI 自動分析。\n"
        "`/addimage <id> [attachment 1–5]` 或 `!addimage <id>` (附上圖片)\n"
        "　將最多 5 張圖片追加到現有圖像條目。\n"
        "`/editdesc <id>`　開啟互動視窗，編輯圖像條目的使用者描述 (背景設定/劇情)。\n"
        "`/editappearance <id>`　開啟互動視窗，編輯圖像條目的外貌描述 (Bot 專用，用於圖像生成)。",
    ),
    (
        "🧠 長期記憶",
        "`/memory` — 開啟或關閉主動記憶 (每次對話自動注入最近記憶)\n"
        "`/memorylength <數字>` — 設定主動記憶注入條數 (預設 50)\n"
        "`/passivememory` — 開啟或關閉被動記憶 (用戶詢問時搜索深層記憶庫)\n"
        "`/passivememorylength <數字>` — 設定深層搜索條數 (預設 200)\n"
        "`/memories` — 分頁瀏覽所有記憶\n"
        "`/clearmemory` — 清除自己的記憶\n"
        "`/clearmemory @用戶` — 清除指定用戶的記憶 (需管理員)\n"
        "`/clearmemory all:True` — 清除所有人的記憶 (需管理員)",
    ),
    (
        "🎭 敘事模式",
        "`/narrative` 或 `!narrative` — 切換此頻道的 Narrative Mode\n"
        "　開啟後：對話加粗引號、氛圍描寫、微表情細節、感官文字。\n"
        "　圖像生成同步加入電影構圖與戲劇光影。\n"
        "　再次輸入即可關閉，設定僅限當前頻道。",
    ),
    (
        "💬 建議按鈕",
        "`/suggestions` — 開啟或關閉回覆下方的建議按鈕\n"
        "`/setsuggestionprompt <提示詞>` — 自訂建議生成提示詞 (覆蓋預設)\n"
        "`/clearsuggestionprompt` — 恢復預設建議提示詞",
    ),
    (
        "🤖 機器人狀態",
        "`/setstatus <文字> [類型] [狀態]` — 設定機器人的自訂 Discord 狀態\n"
        "　類型: `playing` / `watching` / `listening` / `streaming`\n"
        "　狀態: `online` / `idle` / `dnd`\n"
        "`/clearstatus` — 清除自訂狀態，恢復預設\n"
        "`/setthinking <文字>` — 設定思考泡泡 (Discord「What's on your mind?」)\n"
        "`/clearthinking` — 清除思考泡泡\n"
        "`/clear` 或 `!clear` — 清除此頻道的對話歷史記錄",
    ),
    (
        "🔑 權限管理 (需管理員)",
        "`/permissions` — 分頁檢視所有指令的目前權限設定\n"
        "`/setrole <指令> <@角色>` — 將指定指令的使用權開放給某個角色\n"
        "`/clearrole <指令>` — 移除指令的角色限制，恢復為所有人可用",
    ),
]
