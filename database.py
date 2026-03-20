import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "data")

CHARACTER_DIR  = os.path.join(DATA_DIR, "character")
MEMORIES_DIR   = os.path.join(DATA_DIR, "memories")
KNOWLEDGE_DIR  = os.path.join(DATA_DIR, "knowledge")
IMAGES_DIR     = os.path.join(KNOWLEDGE_DIR, "images")

CHARACTER_JSON = os.path.join(CHARACTER_DIR, "profile.json")
MEMORIES_JSON  = os.path.join(MEMORIES_DIR, "memories.json")
COUNTER_JSON   = os.path.join(KNOWLEDGE_DIR, "_counter.json")

STATUS_JSON    = os.path.join(DATA_DIR, "status.json")
SETTINGS_JSON  = os.path.join(DATA_DIR, "settings.json")
HISTORY_DB     = os.path.join(DATA_DIR, "history.db")

for _d in (CHARACTER_DIR, MEMORIES_DIR, KNOWLEDGE_DIR, IMAGES_DIR):
    os.makedirs(_d, exist_ok=True)

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


# ── ID counter (for knowledge entries) ───────────────────────────────────────

def _read_counter() -> int:
    if os.path.exists(COUNTER_JSON):
        try:
            with open(COUNTER_JSON, "r", encoding="utf-8") as f:
                return int(json.load(f).get("next_id", 1))
        except Exception:
            pass
    return 1


def _write_counter(next_id: int):
    with open(COUNTER_JSON, "w", encoding="utf-8") as f:
        json.dump({"next_id": next_id}, f)


def _next_id() -> int:
    n = _read_counter()
    _write_counter(n + 1)
    return n


# ── Character ─────────────────────────────────────────────────────────────────

def get_character() -> Dict[str, str]:
    if os.path.exists(CHARACTER_JSON):
        try:
            with open(CHARACTER_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "name" in data and "background" in data:
                return data
        except Exception as e:
            print(f"[DB] Warning: could not read character/profile.json: {e}")
    return dict(DEFAULT_CHARACTER)


def set_character(name: str, background: str) -> bool:
    try:
        data = {"name": name, "background": background}
        with open(CHARACTER_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DB] Character saved → {CHARACTER_JSON}")
        return True
    except Exception as e:
        print(f"[DB] Error saving character/profile.json: {e}")
        return False


def _ensure_character_json():
    if not os.path.exists(CHARACTER_JSON):
        set_character(DEFAULT_CHARACTER["name"], DEFAULT_CHARACTER["background"])
        print(f"[DB] Created default character/profile.json")


# ── Status ────────────────────────────────────────────────────────────────────

def get_status() -> dict:
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
    try:
        data = {"text": text, "activity_type": activity_type, "status": status}
        with open(STATUS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[DB] Error saving status.json: {e}")
        return False


def clear_status() -> bool:
    try:
        if os.path.exists(STATUS_JSON):
            os.remove(STATUS_JSON)
        return True
    except Exception as e:
        print(f"[DB] Error removing status.json: {e}")
        return False


# ── Settings ──────────────────────────────────────────────────────────────────

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
    "setthinking":           "__admin__",
    "clearthinking":         "__admin__",
    "helpsetting":           "__admin__",
    "generate":              None,
    "help":                  None,
}

_SETTINGS_DEFAULTS = {
    "suggestions_enabled":    True,
    "suggestion_prompt":      "",
    "memory_enabled":         True,
    "memory_length":          20,
    "passive_memory_enabled": True,
    "passive_memory_length":  50,
    "command_roles":          dict(_DEFAULT_COMMAND_ROLES),
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
    return _load_settings().get(key, _SETTINGS_DEFAULTS.get(key))


def set_setting(key: str, value) -> bool:
    data = _load_settings()
    data[key] = value
    ok = _save_settings(data)
    if ok:
        print(f"[DB] Setting '{key}' = {repr(value)}")
    return ok


def get_command_roles() -> Dict[str, Optional[str]]:
    saved = _load_settings().get("command_roles", {})
    if not isinstance(saved, dict):
        saved = {}
    return {**_DEFAULT_COMMAND_ROLES, **saved}


def set_command_role(command_name: str, role_name: Optional[str]) -> bool:
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
    return set_command_role(command_name, None)


# ── Knowledge base — file helpers ─────────────────────────────────────────────

def _text_entry_path(entry_id: int) -> str:
    return os.path.join(KNOWLEDGE_DIR, f"{entry_id}.json")


def _image_meta_path(entry_id: int) -> str:
    return os.path.join(IMAGES_DIR, f"{entry_id}.json")


def _image_file_path(entry_id: int, mime: str) -> str:
    ext = mime.split("/")[-1] if "/" in mime else "png"
    if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
        ext = "png"
    return os.path.join(IMAGES_DIR, f"{entry_id}.{ext}")


def _find_image_file(entry_id: int) -> Optional[str]:
    for ext in ("png", "jpg", "jpeg", "gif", "webp"):
        p = os.path.join(IMAGES_DIR, f"{entry_id}.{ext}")
        if os.path.exists(p):
            return p
    return None


# ── Knowledge base — CRUD ─────────────────────────────────────────────────────

def add_text_entry(title: str, content: str, tags: str = "") -> int:
    entry_id = _next_id()
    now = datetime.utcnow().isoformat()
    data = {
        "id": entry_id,
        "title": title,
        "content": content,
        "entry_type": "text",
        "tags": tags,
        "created_at": now,
        "updated_at": now,
    }
    with open(_text_entry_path(entry_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[DB] Text entry #{entry_id} saved → knowledge/{entry_id}.json")
    return entry_id


def add_image_entry(title: str, image_bytes: bytes, mime_type: str, description: str = "", tags: str = "") -> int:
    entry_id = _next_id()
    now = datetime.utcnow().isoformat()
    meta = {
        "id": entry_id,
        "title": title,
        "entry_type": "image",
        "image_description": description,
        "image_mime": mime_type,
        "tags": tags,
        "created_at": now,
        "updated_at": now,
    }
    img_path = _image_file_path(entry_id, mime_type)
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    with open(_image_meta_path(entry_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DB] Image entry #{entry_id} saved → knowledge/images/{entry_id}.*")
    return entry_id


def _load_all_knowledge_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for fname in os.listdir(KNOWLEDGE_DIR):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue
        try:
            with open(os.path.join(KNOWLEDGE_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            if "id" in data and "entry_type" in data:
                entries.append(data)
        except Exception:
            pass
    for fname in os.listdir(IMAGES_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(IMAGES_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            if "id" in data and data.get("entry_type") == "image":
                entries.append(data)
        except Exception:
            pass
    entries.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
    return entries


def get_all_entries(limit: int = 200) -> List[Dict[str, Any]]:
    entries = _load_all_knowledge_entries()
    return entries[:limit]


def get_entry_by_id(entry_id: int) -> Optional[Dict[str, Any]]:
    text_path = _text_entry_path(entry_id)
    if os.path.exists(text_path):
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    meta_path = _image_meta_path(entry_id)
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            img_path = _find_image_file(entry_id)
            if img_path:
                with open(img_path, "rb") as f:
                    meta["image_data"] = f.read()
                meta["image_mime"] = meta.get("image_mime", "image/png")
            return meta
        except Exception:
            return None

    return None


def search_knowledge(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    entries = _load_all_knowledge_entries()
    words = [w.strip().lower() for w in query.split() if len(w.strip()) > 1][:12]
    if not words:
        return entries[:limit]

    results = []
    for entry in entries:
        haystack = " ".join([
            (entry.get("title") or ""),
            (entry.get("content") or ""),
            (entry.get("image_description") or ""),
            (entry.get("tags") or ""),
        ]).lower()
        if any(w in haystack for w in words):
            results.append(entry)
        if len(results) >= limit:
            break
    return results


def search_kb_images_by_subject(description: str, limit: int = 6) -> List[Dict[str, Any]]:
    words = [w.strip().lower() for w in description.split() if len(w.strip()) > 2][:15]
    if not words:
        return []
    results = []
    for fname in os.listdir(IMAGES_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(IMAGES_DIR, fname), "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if not meta.get("image_description"):
            continue
        haystack = " ".join([
            (meta.get("title") or ""),
            (meta.get("image_description") or ""),
            (meta.get("tags") or ""),
        ]).lower()
        if any(w in haystack for w in words):
            results.append(meta)
        if len(results) >= limit:
            break
    return results


def update_image_description(entry_id: int, description: str) -> bool:
    meta_path = _image_meta_path(entry_id)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("entry_type") != "image":
            return False
        meta["image_description"] = description
        meta["updated_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[DB] Error updating image description #{entry_id}: {e}")
        return False


def update_text_content(entry_id: int, content: str) -> bool:
    text_path = _text_entry_path(entry_id)
    if not os.path.exists(text_path):
        return False
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("entry_type") != "text":
            return False
        data["content"] = content
        data["updated_at"] = datetime.utcnow().isoformat()
        with open(text_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[DB] Error updating text entry #{entry_id}: {e}")
        return False


def delete_entry(entry_id: int) -> bool:
    deleted = False
    text_path = _text_entry_path(entry_id)
    if os.path.exists(text_path):
        os.remove(text_path)
        deleted = True

    meta_path = _image_meta_path(entry_id)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        mime = meta.get("image_mime", "image/png")
        img_path = _find_image_file(entry_id)
        if img_path:
            os.remove(img_path)
        os.remove(meta_path)
        deleted = True

    return deleted


# ── Conversation history (SQLite — internal only) ─────────────────────────────

def _get_history_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    return conn


def _init_history_db():
    conn = _get_history_conn()
    conn.execute("""
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
    conn.commit()
    conn.close()


def save_conversation(channel_id: str, user_id: str, user_name: str, content: str, role: str):
    conn = _get_history_conn()
    conn.execute(
        "INSERT INTO conversation_history (channel_id, user_id, user_name, content, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (channel_id, user_id, user_name, content, role, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_conversation(channel_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    conn = _get_history_conn()
    rows = conn.execute("""
        SELECT user_name, content, role, created_at
        FROM conversation_history
        WHERE channel_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (channel_id, limit)).fetchall()
    conn.close()
    return list(reversed([dict(r) for r in rows]))


# ── Memories (plain JSON file) ────────────────────────────────────────────────

def _load_memories_list() -> List[Dict[str, Any]]:
    if os.path.exists(MEMORIES_JSON):
        try:
            with open(MEMORIES_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception as e:
            print(f"[DB] Warning: could not read memories.json: {e}")
    return []


def _save_memories_list(memories: List[Dict[str, Any]]):
    with open(MEMORIES_JSON, "w", encoding="utf-8") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)


def _next_memory_id(memories: List[Dict[str, Any]]) -> int:
    if not memories:
        return 1
    return max(m.get("id", 0) for m in memories) + 1


def add_memory(user_id: str, user_name: str, summary: str) -> int:
    memories = _load_memories_list()
    entry_id = _next_memory_id(memories)
    memories.append({
        "id": entry_id,
        "user_id": user_id,
        "user_name": user_name,
        "summary": summary,
        "created_at": datetime.utcnow().isoformat(),
    })
    _save_memories_list(memories)
    return entry_id


def get_memories(limit: int = 20) -> List[Dict[str, Any]]:
    memories = _load_memories_list()
    return list(reversed(memories))[:limit]


def search_memories(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    memories = _load_memories_list()
    words = [w.strip().lower() for w in query.split() if len(w.strip()) > 1][:8]
    if not words:
        return list(reversed(memories))[:limit]
    results = [
        m for m in reversed(memories)
        if any(w in m.get("summary", "").lower() for w in words)
    ]
    return results[:limit]


def get_all_memories(limit: int = 500) -> List[Dict[str, Any]]:
    memories = _load_memories_list()
    return list(reversed(memories))[:limit]


def count_memories() -> int:
    return len(_load_memories_list())


def count_memories_for_user(user_id: str) -> int:
    return sum(1 for m in _load_memories_list() if m.get("user_id") == user_id)


def clear_memories() -> int:
    memories = _load_memories_list()
    count = len(memories)
    _save_memories_list([])
    return count


def clear_memories_for_user(user_id: str) -> int:
    memories = _load_memories_list()
    kept = [m for m in memories if m.get("user_id") != user_id]
    count = len(memories) - len(kept)
    _save_memories_list(kept)
    return count


# ── Migration from old SQLite ─────────────────────────────────────────────────

def _migrate_old_sqlite():
    old_db_paths = [
        os.path.join(_SCRIPT_DIR, "knowledge_base.db"),
        os.path.join(DATA_DIR, "knowledge_base.db"),
    ]
    old_db = None
    for p in old_db_paths:
        if os.path.exists(p):
            old_db = p
            break
    if not old_db:
        return

    print(f"[DB] Migrating old SQLite database: {old_db}")
    import shutil

    try:
        conn = sqlite3.connect(old_db)
        conn.row_factory = sqlite3.Row

        # -- Character
        if not os.path.exists(CHARACTER_JSON):
            try:
                row = conn.execute(
                    "SELECT content FROM knowledge_entries WHERE entry_type = 'character' LIMIT 1"
                ).fetchone()
                if row and row["content"]:
                    data = json.loads(row["content"])
                    if "name" in data and "background" in data:
                        set_character(data["name"], data["background"])
                        print("[DB] Migrated character data")
            except Exception:
                pass

        # -- Knowledge entries
        try:
            rows = conn.execute(
                "SELECT * FROM knowledge_entries WHERE entry_type != 'character'"
            ).fetchall()
            existing_ids = {
                int(f.replace(".json", ""))
                for f in os.listdir(KNOWLEDGE_DIR)
                if f.endswith(".json") and not f.startswith("_")
            }
            max_id = _read_counter() - 1
            for row in rows:
                row = dict(row)
                old_id = row["id"]
                if row["entry_type"] == "text":
                    if not os.path.exists(_text_entry_path(old_id)):
                        data = {
                            "id": old_id,
                            "title": row.get("title", ""),
                            "content": row.get("content", ""),
                            "entry_type": "text",
                            "tags": row.get("tags", ""),
                            "created_at": row.get("created_at", datetime.utcnow().isoformat()),
                            "updated_at": row.get("updated_at", datetime.utcnow().isoformat()),
                        }
                        with open(_text_entry_path(old_id), "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        max_id = max(max_id, old_id)
                        print(f"[DB] Migrated text entry #{old_id}: {row.get('title', '')}")
                elif row["entry_type"] == "image":
                    if not os.path.exists(_image_meta_path(old_id)):
                        mime = row.get("image_mime") or "image/png"
                        meta = {
                            "id": old_id,
                            "title": row.get("title", ""),
                            "entry_type": "image",
                            "image_description": row.get("image_description", ""),
                            "image_mime": mime,
                            "tags": row.get("tags", ""),
                            "created_at": row.get("created_at", datetime.utcnow().isoformat()),
                            "updated_at": row.get("updated_at", datetime.utcnow().isoformat()),
                        }
                        with open(_image_meta_path(old_id), "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)
                        img_data = row.get("image_data")
                        if img_data:
                            img_path = _image_file_path(old_id, mime)
                            with open(img_path, "wb") as f:
                                f.write(img_data)
                        max_id = max(max_id, old_id)
                        print(f"[DB] Migrated image entry #{old_id}: {row.get('title', '')}")
            _write_counter(max_id + 1)
        except Exception as e:
            print(f"[DB] Warning during KB migration: {e}")

        # -- Memories
        if not os.path.exists(MEMORIES_JSON):
            try:
                rows = conn.execute(
                    "SELECT id, user_id, user_name, summary, created_at FROM memories ORDER BY id ASC"
                ).fetchall()
                if rows:
                    mem_list = [dict(r) for r in rows]
                    _save_memories_list(mem_list)
                    print(f"[DB] Migrated {len(mem_list)} memories")
            except Exception as e:
                print(f"[DB] Warning during memories migration: {e}")

        conn.close()

        # Rename old DB so migration doesn't run again
        archive = old_db + ".migrated"
        os.rename(old_db, archive)
        print(f"[DB] Old database archived as {archive}")

    except Exception as e:
        print(f"[DB] Migration error: {e}")


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db():
    _migrate_old_sqlite()
    _ensure_character_json()
    _init_history_db()
    if not os.path.exists(MEMORIES_JSON):
        _save_memories_list([])
    print(f"[DB] Character : {CHARACTER_JSON}")
    print(f"[DB] Memories  : {MEMORIES_JSON}")
    print(f"[DB] Knowledge : {KNOWLEDGE_DIR}/")
    print(f"[DB] History   : {HISTORY_DB}")
