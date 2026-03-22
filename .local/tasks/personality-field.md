---
title: Add personality field to /setcharacter and /character tab navigation
---
# Add personality field + /character tab navigation

## What & Why

Add a separate `personality` field that describes HOW the bot speaks (tone,
vocabulary, speech patterns), distinct from `background` (who the character is).
The /character command gets tab navigation buttons to switch between background
and personality views.

## Done looks like

- `/setcharacter name background [personality]` — personality is optional
- `/character` shows "📖 背景" and "💬 個性" tab buttons; clicking toggles the embed content
- Each tab has its own pagination (◀ / ▶) when content exceeds 1024 chars
- Personality is injected into the system prompt as a "how to speak" section
- EditCharacterModal gains a personality TextInput field
- Backward-compatible: old character JSONs without personality load fine (personality defaults to empty)

## Implementation

### database.py

1. Add `"personality": ""` to `DEFAULT_CHARACTER`
2. Update `get_character()` to return `personality` key (default `""` for old data)
3. Update `set_character(name, background, personality="")` to save all three fields

### bot.py

1. `load_character()` → return `(name, background, personality)` 3-tuple
2. `build_system_prompt(name, background, personality="", ...)`:
   - After the existing background/identity block, if personality is non-empty, add:
     `"Speaking style and personality: {personality}\n"`
3. `/setcharacter` command:
   - Add `personality: str = ""` parameter (optional, with app_commands.describe)
   - Pass personality to `database.set_character` and `ui.CharacterView`
4. `/character` command: pass personality to `CharacterView`
5. All call sites of `load_character()` — unpack 3-tuple (use `bot_name, background, personality`)
6. All call sites of `build_system_prompt()` — pass personality
7. Update the `setcharacter` help string

### views.py

1. `build_char_embed(bot_name, background, personality, tab="background", ...)`:
   - Add `personality` and `tab` params
   - When `tab == "personality"`: show personality field instead of background field
   - Rename field header: "📖 背景" or "💬 個性" accordingly
   - Pagination chunks the active tab's text

2. `CharacterView.__init__` gains `personality` param and `tab = "background"` state
3. `CharacterView._build_buttons`:
   - Row 0 (always): tab buttons "📖 背景" and "💬 個性" (highlighted = current tab)
   - Row 1 (when active tab text > 1024): pagination ◀ / ▶
   - Row 2 (or lowest free row): "✏️ 編輯角色" and "🖼️ 外貌圖庫"
4. Tab switch resets page to 0, updates embed and buttons
5. `EditCharacterModal` gains a `personality` TextInput (paragraph style, max 4000 chars)
   - On submit: save all three fields, refresh embed showing background tab

## Relevant files

- database.py
- bot.py
- views.py
