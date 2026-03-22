---
title: Strip <think>...</think> blocks from AI responses
---
---
title: Strip <think>...</think> blocks from AI responses
---
# Strip reasoning think-blocks from responses

## What & Why

Reasoning models (DeepSeek-R1, QwQ, etc.) output their chain-of-thought inside
`<think>...</think>` tags before the actual reply. The bot is currently sending
these verbatim to Discord, making the bot "think out loud."

The fix is a single regex strip applied immediately after the raw text is captured
from the model response, before any other processing.

## Done looks like

- `<think>...</think>` blocks (including multiline content) are removed from all
  AI responses before they leave the AI module
- The actual reply text after the think block is preserved unchanged
- Fix applies to both groq_ai.py and ollama_ai.py

## Implementation

### groq_ai.py
After line 456:
```python
text = (response.choices[0].message.content or "").strip()
```
Add immediately after:
```python
text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
```

### ollama_ai.py
Two places extract response content (lines 257 and 424). Apply the same strip
after each one.

Add a module-level compiled regex near the top of each file for efficiency:
```python
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
```

Then use `_THINK_RE.sub("", text).strip()` at each extraction point.

## Relevant files

- groq_ai.py (line 456)
- ollama_ai.py (lines 257, 424)