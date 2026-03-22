# Chat Response Progress Indicator

## What & Why
When a user mentions the bot, the only feedback is Discord's native "typing..." indicator, which is subtle and times out. Add a visible progress message that gets sent immediately as a reply and updates through each processing stage so users can see the bot is working — especially during slower paths (image analysis → AI generation → image rendering).

## Done looks like
- When the bot starts processing any mention/reply, a message like `⏳ 思考中⋯` appears immediately as a reply to the user
- The message updates automatically as work progresses:
  - If an image was attached: first shows `🔍 分析圖片中⋯`
  - During AI text generation: `⏳ 思考中⋯`
  - When image generation is triggered: updates to `🎨 生成圖像中⋯`
- For text replies: the progress message is edited in-place to become the final response (no extra ping, same message ID; suggestion buttons attach here too)
- For image replies: the progress message is deleted, and the image is posted as a fresh reply
- If any stage fails, the progress message is edited to show the error instead of leaving a stale "thinking" message
- The suggestion button click path also shows the progress indicator

## Out of scope
- Progress bars for admin/command responses (`!saveimage`, `!addcharimage`, `!generate` already have their own indicators)
- Percentage or ETA estimates
- Streaming text output (the bot still waits for the complete response before sending)

## Tasks
1. **Add `thinking_msg` to `process_chat`** — At the very beginning of `process_chat`, send a short "thinking" reply and hold a reference to it. Update its text at each major stage (image analysis, AI call, image generation). Use a try/except so failures in the indicator never abort the main flow.

2. **Wire text replies through thinking_msg** — Refactor `send_with_suggestions` to accept an optional `thinking_msg`; when provided, edit it in-place with the final text + view instead of sending a new reply. This avoids a second Discord ping.

3. **Handle image-generation path** — Before the Cloudflare generate call, update the thinking message to `🎨 生成圖像中⋯`. Because Discord can't edit a message to attach a file, delete the thinking message just before sending the image reply.

4. **Handle error paths** — Wherever `process_chat` would send an error reply (`reply_target.reply("⚠️ ...")`), edit the thinking message instead so it becomes the error notification and no stale indicator is left.

5. **SuggestionButton path** — `SuggestionButton.callback` also calls `process_chat`; it disables its buttons and then calls into the same pipeline, so it will inherit the indicator naturally once `process_chat` manages it internally.

## Relevant files
- `bot.py:391-614`
- `bot.py:936-996`
