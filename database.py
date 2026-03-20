import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

# ── Data folder (all persistent files live here — easy to back up / transfer) ─
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "knowledge_base.db")
CHARACTER_JSON = os.path.join(DATA_DIR, "character.json")
STATUS_JSON = os.path.join(DATA_DIR, "status.json")
SETTINGS_JSON = os.path.join(DATA_DIR, "settings.json")

DEFAULT_CHARACTER = {
    "name": "少女樂團機器人",
    "background": (
        "妳是少女樂團機器人，一個友善且知識豐富的Discord AI助手。"
        "擁有溫暖、機智且充滿趣味的個性。"
        "妳熱愛學習新事物並幫助人們。"
        "隨著用戶不斷教妳，妳的知識庫會逐漸擴展。"
        "妳充滿好奇心、感同身受，時刻準備投入有意義的對話。"
        "妳可以分析圖像、生成藝術作品，並記住用戶與妳分享的信息。"
    ),
}


# ── Character JSON (human-editable, primary source of truth) ─────────────────

def get_character() -> Dict[str, str]:
    """Load character from data/character.json, falling back to defaults."""
    if os.path.exists(CHARACTER_JSON):
        try:
            with open(CHARACTER_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "name" in data and "background" in data:
                return data
        except Exception as e:
            print(f"[DB] Warning: could not read character.json: {e}")
    return dict(DEFAULT_CHARACTER)


def set_character(name: str, background: str) -> bool:
    """Save character to data/character.json."""
    try:
        data = {"name": name, "background": background}
        with open(CHARACTER_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DB] Character saved to {CHARACTER_JSON}")
        return True
    except Exception as e:
        print(f"[DB] Error saving character.json: {e}")
        return False


def _ensure_character_json():
    """Create character.json with defaults if it doesn't exist yet."""
    if not os.path.exists(CHARACTER_JSON):
        set_character(DEFAULT_CHARACTER["name"], DEFAULT_CHARACTER["background"])
        print(f"[DB] Created default character.json at {CHARACTER_JSON}")


# ── Custom Status JSON (data/status.json) ────────────────────────────────────

def get_status() -> dict:
    """Load persisted custom status from data/status.json. Returns {} if none set."""
    if os.path.exists(STATUS_JSON):
        try:
            with open(STATUS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "text" in data:
                return data
        except Exception as e:
            print(f"[DB] Warning: could not read status.json: {e}")
    return {}


def set_status(text: str, activity_type: str, status: str) -> bool:
    """Persist custom status to data/status.json."""
    try:
        data = {"text": text, "activity_type": activity_type, "status": status}
        with open(STATUS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DB] Status saved: {activity_type} '{text}' ({status})")
        return True
    except Exception as e:
        print(f"[DB] Error saving status.json: {e}")
        return False


def clear_status() -> bool:
    """Delete status.json to reset to default bot presence."""
    try:
        if os.path.exists(STATUS_JSON):
            os.remove(STATUS_JSON)
            print("[DB] status.json removed (reset to default)")
        return True
    except Exception as e:
        print(f"[DB] Error removing status.json: {e}")
        return False


# ── General Settings JSON (data/settings.json) ───────────────────────────────

# Default role gate for every gated command.
# None  → open to all
# "__admin__" → requires Discord server Administrator permission (until owner sets a real role)
_DEFAULT_COMMAND_ROLES: Dict[str, Optional[str]] = {
    "setcharacter":          "__admin__",
    "character":             "__admin__",
    "remember":              "__admin__",
    "forget":                "__admin__",
    "setdesc":               "__admin__",
    "viewentry":             "__admin__",
    "knowledge":             "__admin__",
    "saveimage":             "__admin__",
    "clear":                 "__admin__",
    "suggestions":           "__admin__",
    "setsuggestionprompt":   "__admin__",
    "clearsuggestionprompt": "__admin__",
    "memory":                "__admin__",
    "memorylength":          "__admin__",
    "passivememory":         "__admin__",
    "passivememorylength":   "__admin__",
    "memories":              "__admin__",
    "clearmemory":           None,
    "setstatus":             "__admin__",
    "clearstatus":           "__admin__",
    "helpsetting":           "__admin__",
    "generate":              None,
    "help":                  None,
}

_SETTINGS_DEFAULTS = {
    "suggestions_enabled": True,
    "suggestion_prompt": "",
    "memory_enabled": True,
    "memory_length": 20,
    "passive_memory_enabled": True,
    "passive_memory_length": 50,
    "command_roles": dict(_DEFAULT_COMMAND_ROLES),
}


def _load_settings() -> dict:
    if os.path.exists(SETTINGS_JSON):
        try:
            with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
                return {**_SETTINGS_DEFAULTS, **json.load(f)}
        except Exception as e:
            print(f"[DB] Warning: could not read settings.json: {e}")
    return dict(_SETTINGS_DEFAULTS)


def _save_settings(data: dict) -> bool:
    try:
        with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[DB] Error saving settings.json: {e}")
        return False


def get_setting(key: str):
    """Get a single setting value by key."""
    return _load_settings().get(key, _SETTINGS_DEFAULTS.get(key))


def set_setting(key: str, value) -> bool:
    """Set a single setting value and persist to disk."""
    data = _load_settings()
    data[key] = value
    ok = _save_settings(data)
    if ok:
        print(f"[DB] Setting '{key}' = {repr(value)}")
    return ok


# ── Command Role Gates ────────────────────────────────────────────────────────

def get_command_roles() -> Dict[str, Optional[str]]:
    """Return the full command→gate mapping, merging saved overrides with defaults."""
    saved = _load_settings().get("command_roles", {})
    if not isinstance(saved, dict):
        saved = {}
    return {**_DEFAULT_COMMAND_ROLES, **saved}


def set_command_role(command_name: str, role_name: Optional[str]) -> bool:
    """Assign a role gate to a command.  Pass None to open the command to everyone."""
    data = _load_settings()
    roles = data.get("command_roles", {})
    if not isinstance(roles, dict):
        roles = {}
    roles[command_name] = role_name
    data["command_roles"] = roles
    ok = _save_settings(data)
    if ok:
        print(f"[DB] command_roles['{command_name}'] = {repr(role_name)}")
    return ok


def clear_command_role(command_name: str) -> bool:
    """Remove any role gate from a command (open to everyone)."""
    return set_command_role(command_name, None)


# ── SQLite (KB entries + conversation history) ────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_old_db():
    """Move old knowledge_base.db from the script root into data/ and extract character.json."""
    old_db = os.path.join(_SCRIPT_DIR, "knowledge_base.db")
    if not os.path.exists(old_db):
        return
    print(f"[DB] Migrating old database from {old_db} …")
    import shutil

    if not os.path.exists(DB_PATH):
        shutil.copy2(old_db, DB_PATH)
        print(f"[DB] Copied to {DB_PATH}")

    if not os.path.exists(CHARACTER_JSON):
        try:
            old_conn = sqlite3.connect(DB_PATH)
            old_conn.row_factory = sqlite3.Row
            row = old_conn.execute(
                "SELECT content FROM knowledge_entries WHERE entry_type = 'character' LIMIT 1"
            ).fetchone()
            old_conn.close()
            if row and row["content"]:
                data = json.loads(row["content"])
                if "name" in data and "background" in data:
                    set_character(data["name"], data["background"])
                    print(f"[DB] Extracted character data to {CHARACTER_JSON}")
        except Exception as e:
            print(f"[DB] Warning: could not extract character from old DB: {e}")

    # Remove the old character rows and the old DB file
    try:
        conn_tmp = sqlite3.connect(DB_PATH)
        conn_tmp.execute("DELETE FROM knowledge_entries WHERE entry_type = 'character'")
        conn_tmp.commit()
        conn_tmp.close()
    except Exception:
        pass

    try:
        os.remove(old_db)
        print(f"[DB] Removed old database at {old_db}")
    except Exception as e:
        print(f"[DB] Could not remove old DB: {e}")


def init_db():
    _migrate_old_db()
    _ensure_character_json()
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT,
            entry_type TEXT NOT NULL DEFAULT 'text',
            image_data BLOB,
            image_mime TEXT,
            image_description TEXT,
            tags TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            user_name TEXT NOT NULL,
            content TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_name TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Database : {DB_PATH}")
    print(f"[DB] Character: {CHARACTER_JSON}")
    print(f"[DB] Data dir : {DATA_DIR}")


def add_text_entry(title: str, content: str, tags: str = "") -> int:
    conn = get_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO knowledge_entries (title, content, entry_type, tags, created_at, updated_at) VALUES (?, ?, 'text', ?, ?, ?)",
        (title, content, tags, now, now),
    )
    entry_id = c.lastrowid
    conn.commit()
    conn.close()
    return entry_id


def add_image_entry(title: str, image_bytes: bytes, mime_type: str, description: str = "", tags: str = "") -> int:
    conn = get_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO knowledge_entries (title, entry_type, image_data, image_mime, image_description, tags, created_at, updated_at) VALUES (?, 'image', ?, ?, ?, ?, ?, ?)",
        (title, image_bytes, mime_type, description, tags, now, now),
    )
    entry_id = c.lastrowid
    conn.commit()
    conn.close()
    return entry_id


def search_kb_images_by_subject(description: str, limit: int = 6) -> List[Dict[str, Any]]:
    """Search only image entries whose stored description matches keywords from `description`.

    Used when a user sends an image — the vision description of the uploaded image
    is used to find KB images that share the same subjects (people, places, objects).
    Returns only entries that have a non-empty image_description.
    """
    conn = get_connection()
    c = conn.cursor()

    words = [w.strip() for w in description.split() if len(w.strip()) > 2][:15]
    if not words:
        conn.close()
        return []

    conditions = []
    params: List = []
    for word in words:
        q = f"%{word.lower()}%"
        conditions.append(
            "(LOWER(title) LIKE ? OR LOWER(image_description) LIKE ? OR LOWER(tags) LIKE ?)"
        )
        params.extend([q, q, q])
    params.append(limit)

    c.execute(f"""
        SELECT id, title, image_description, tags, created_at
        FROM knowledge_entries
        WHERE entry_type = 'image'
          AND image_description IS NOT NULL
          AND image_description != ''
          AND ({" OR ".join(conditions)})
        ORDER BY updated_at DESC
        LIMIT ?
    """, params)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_knowledge(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()

    # Split query into individual words and search for any of them.
    # Using the whole phrase as one LIKE pattern would almost never match.
    words = [w.strip() for w in query.split() if len(w.strip()) > 1][:12]
    if not words:
        conn.close()
        return []

    conditions = []
    params: List = []
    for word in words:
        q = f"%{word.lower()}%"
        conditions.append(
            "(LOWER(title) LIKE ? OR LOWER(content) LIKE ? OR LOWER(image_description) LIKE ? OR LOWER(tags) LIKE ?)"
        )
        params.extend([q, q, q, q])
    params.append(limit)

    c.execute(f"""
        SELECT id, title, content, entry_type, image_description, tags, created_at
        FROM knowledge_entries
        WHERE {" OR ".join(conditions)}
        ORDER BY updated_at DESC
        LIMIT ?
    """, params)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_entries(limit: int = 200) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, title, content, entry_type, image_description, tags, created_at
        FROM knowledge_entries
        ORDER BY updated_at DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_entry_by_id(entry_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM knowledge_entries WHERE id = ?", (entry_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def update_image_description(entry_id: int, description: str) -> bool:
    conn = get_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "UPDATE knowledge_entries SET image_description = ?, updated_at = ? WHERE id = ? AND entry_type = 'image'",
        (description, now, entry_id),
    )
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def update_text_content(entry_id: int, content: str) -> bool:
    conn = get_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "UPDATE knowledge_entries SET content = ?, updated_at = ? WHERE id = ? AND entry_type = 'text'",
        (content, now, entry_id),
    )
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def delete_entry(entry_id: int) -> bool:
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM knowledge_entries WHERE id = ?", (entry_id,))
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def save_conversation(channel_id: str, user_id: str, user_name: str, content: str, role: str):
    conn = get_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO conversation_history (channel_id, user_id, user_name, content, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (channel_id, user_id, user_name, content, role, now),
    )
    conn.commit()
    conn.close()


def get_recent_conversation(channel_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT user_name, content, role, created_at
        FROM conversation_history
        WHERE channel_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (channel_id, limit))
    rows = c.fetchall()
    conn.close()
    return list(reversed([dict(r) for r in rows]))


# ── Long-term Memory ──────────────────────────────────────────────────────────

def add_memory(user_id: str, user_name: str, summary: str) -> int:
    """Store a single memorable fact extracted from conversation."""
    conn = get_connection()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO memories (user_id, user_name, summary, created_at) VALUES (?, ?, ?, ?)",
        (user_id, user_name, summary, now),
    )
    entry_id = c.lastrowid
    conn.commit()
    conn.close()
    return entry_id


def get_memories(limit: int = 20) -> List[Dict[str, Any]]:
    """Get the most recent N memories across all users (active injection)."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, user_id, user_name, summary, created_at
        FROM memories
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_memories(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Keyword-search the full memory archive (passive recall)."""
    conn = get_connection()
    c = conn.cursor()
    words = [w.strip() for w in query.split() if len(w.strip()) > 1][:8]
    if not words:
        return get_memories(limit)
    conditions = " OR ".join(["LOWER(summary) LIKE ?" for _ in words])
    params = [f"%{w.lower()}%" for w in words] + [limit]
    c.execute(f"""
        SELECT id, user_id, user_name, summary, created_at
        FROM memories
        WHERE {conditions}
        ORDER BY id DESC
        LIMIT ?
    """, params)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_memories(limit: int = 500) -> List[Dict[str, Any]]:
    """Get all memories for display (newest first)."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, user_id, user_name, summary, created_at
        FROM memories
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_memories() -> int:
    """Return the total number of stored memories."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM memories")
    n = c.fetchone()[0]
    conn.close()
    return n


def count_memories_for_user(user_id: str) -> int:
    """Return the number of stored memories for a specific user."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,))
    n = c.fetchone()[0]
    conn.close()
    return n


def clear_memories() -> int:
    """Delete all memories. Returns the count deleted."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM memories")
    count = c.fetchone()[0]
    c.execute("DELETE FROM memories")
    conn.commit()
    conn.close()
    return count


def clear_memories_for_user(user_id: str) -> int:
    """Delete all memories belonging to a specific user. Returns the count deleted."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,))
    count = c.fetchone()[0]
    c.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
    return count
