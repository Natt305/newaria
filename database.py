import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_SCRIPT_DIR, "data")

CHARACTER_DIR        = os.path.join(DATA_DIR, "character")
CHARACTER_IMAGES_DIR = os.path.join(CHARACTER_DIR, "images")
MEMORIES_DIR         = os.path.join(DATA_DIR, "memories")
KNOWLEDGE_DIR        = os.path.join(DATA_DIR, "knowledge")
IMAGES_DIR           = os.path.join(KNOWLEDGE_DIR, "images")
USER_PROFILES_DIR    = os.path.join(DATA_DIR, "user_profiles")

CHARACTER_JSON        = os.path.join(CHARACTER_DIR, "profile.json")
CHARACTER_IMAGES_JSON = os.path.join(CHARACTER_DIR, "images.json")
MEMORIES_JSON         = os.path.join(MEMORIES_DIR, "memories.json")
COUNTER_JSON          = os.path.join(KNOWLEDGE_DIR, "_counter.json")

STATUS_JSON    = os.path.join(DATA_DIR, "status.json")
SETTINGS_JSON  = os.path.join(DATA_DIR, "settings.json")
HISTORY_DB     = os.path.join(DATA_DIR, "history.db")

MAX_CHARACTER_IMAGES = 10
MAX_USER_PROFILE_IMAGES = 5

for _d in (CHARACTER_DIR, CHARACTER_IMAGES_DIR, MEMORIES_DIR, KNOWLEDGE_DIR, IMAGES_DIR, USER_PROFILES_DIR):
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
    "personality": "",
    "looks": "",
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
                data.setdefault("personality", "")
                data.setdefault("looks", "")
                return data
        except Exception as e:
            print(f"[DB] Warning: could not read character/profile.json: {e}")
    return dict(DEFAULT_CHARACTER)


def set_character(name: str, background: str, personality: str = "", looks: str = "") -> bool:
    try:
        data = {"name": name, "background": background, "personality": personality, "looks": looks}
        with open(CHARACTER_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DB] Character saved → {CHARACTER_JSON}")
        return True
    except Exception as e:
        print(f"[DB] Error saving character/profile.json: {e}")
        return False


def _ensure_character_json():
    if not os.path.exists(CHARACTER_JSON):
        set_character(
            DEFAULT_CHARACTER["name"],
            DEFAULT_CHARACTER["background"],
            DEFAULT_CHARACTER["personality"],
            DEFAULT_CHARACTER["looks"],
        )


# ── Character images ──────────────────────────────────────────────────────────

def _read_char_images() -> list:
    if os.path.exists(CHARACTER_IMAGES_JSON):
        try:
            with open(CHARACTER_IMAGES_JSON, "r", encoding="utf-8") as f:
                return json.load(f).get("images", [])
        except Exception:
            pass
    return []


def _write_char_images(images: list):
    with open(CHARACTER_IMAGES_JSON, "w", encoding="utf-8") as f:
        json.dump({"images": images}, f, ensure_ascii=False, indent=2)


def get_character_image_count() -> int:
    return len(_read_char_images())


def get_character_images_meta() -> list:
    """Return list of dicts: [{filename, mime, description}, ...]"""
    return _read_char_images()


def get_character_image(index: int) -> Optional[tuple]:
    """1-indexed. Returns (bytes, mime) or None."""
    images = _read_char_images()
    if index < 1 or index > len(images):
        return None
    info = images[index - 1]
    path = os.path.join(CHARACTER_IMAGES_DIR, info["filename"])
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read(), info.get("mime", "image/png")


def _make_thumbnail(image_bytes: bytes) -> Optional[bytes]:
    """Generate a 512×512 JPEG thumbnail from image bytes using Pillow.
    Returns JPEG bytes on success, or None if Pillow is unavailable / image is corrupt."""
    try:
        from PIL import Image
        import io as _io
        img = Image.open(_io.BytesIO(image_bytes))
        img = img.convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=80, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"[DB] Thumbnail generation failed: {e}")
        return None


def add_character_image(img_bytes: bytes, mime: str, description: str = "") -> tuple:
    """Add a character reference image. Returns (success, message)."""
    images = _read_char_images()
    if len(images) >= MAX_CHARACTER_IMAGES:
        return False, f"已達上限 {MAX_CHARACTER_IMAGES} 張角色圖片。"
    ext = mime.split("/")[-1] if "/" in mime else "png"
    if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
        ext = "png"
    import time as _time
    ts = int(_time.time() * 1000)
    filename = f"char_{ts}.{ext}"
    while os.path.exists(os.path.join(CHARACTER_IMAGES_DIR, filename)):
        ts += 1
        filename = f"char_{ts}.{ext}"
    path = os.path.join(CHARACTER_IMAGES_DIR, filename)
    with open(path, "wb") as f:
        f.write(img_bytes)
    thumb_filename = None
    thumb_bytes = _make_thumbnail(img_bytes)
    if thumb_bytes:
        thumb_filename = f"char_{ts}_thumb.jpg"
        with open(os.path.join(CHARACTER_IMAGES_DIR, thumb_filename), "wb") as f:
            f.write(thumb_bytes)
    entry = {"filename": filename, "mime": mime, "description": description}
    if thumb_filename:
        entry["thumb"] = thumb_filename
    images.append(entry)
    _write_char_images(images)
    return True, f"已新增第 {len(images)} 張角色圖片。"


def remove_character_image(index: int) -> tuple:
    """1-indexed. Returns (success, message)."""
    images = _read_char_images()
    if index < 1 or index > len(images):
        return False, "索引超出範圍。"
    info = images.pop(index - 1)
    path = os.path.join(CHARACTER_IMAGES_DIR, info["filename"])
    if os.path.exists(path):
        os.remove(path)
    thumb = info.get("thumb")
    if thumb:
        thumb_path = os.path.join(CHARACTER_IMAGES_DIR, thumb)
        if os.path.exists(thumb_path):
            os.remove(thumb_path)
    _write_char_images(images)
    return True, "已移除圖片。"


def update_character_image_description(index: int, description: str) -> bool:
    """1-indexed. Returns True on success."""
    images = _read_char_images()
    if index < 1 or index > len(images):
        return False
    images[index - 1]["description"] = description
    _write_char_images(images)
    return True


def get_character_image_thumb(index: int) -> Optional[tuple]:
    """1-indexed. Returns (thumb_bytes, 'image/jpeg') or None."""
    images = _read_char_images()
    if index < 1 or index > len(images):
        return None
    info = images[index - 1]
    thumb = info.get("thumb")
    if not thumb:
        return None
    path = os.path.join(CHARACTER_IMAGES_DIR, thumb)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read(), "image/jpeg"


# ── Initiate opener ───────────────────────────────────────────────────────────

_INITIATE_IMAGE_PATH_BASE = os.path.join(CHARACTER_IMAGES_DIR, "initiate_opener")


def get_initiate_text() -> str:
    """Return the saved RP opener text (empty string if not set)."""
    return str(get_setting("initiate_text") or "")


def set_initiate_text(text: str) -> bool:
    """Persist the opener text."""
    return set_setting("initiate_text", text.strip())


def clear_initiate_text() -> bool:
    """Remove the opener text."""
    return set_setting("initiate_text", "")


def _clear_initiate_image_files() -> None:
    """Delete all initiate_opener.* files from the images directory."""
    for ext in ("png", "jpg", "jpeg", "gif", "webp"):
        path = f"{_INITIATE_IMAGE_PATH_BASE}.{ext}"
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def get_initiate_image() -> Optional[tuple]:
    """Return (bytes, mime) for the saved opener image, or None if not set."""
    mime = str(get_setting("initiate_image_mime") or "")
    if not mime:
        return None
    ext = mime.split("/")[-1] if "/" in mime else "png"
    if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
        ext = "png"
    path = f"{_INITIATE_IMAGE_PATH_BASE}.{ext}"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read(), mime
    return None


def set_initiate_image(img_bytes: bytes, mime: str) -> bool:
    """Save the opener image, replacing any previous one."""
    _clear_initiate_image_files()
    ext = mime.split("/")[-1] if "/" in mime else "png"
    if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
        ext = "png"
        mime = "image/png"
    path = f"{_INITIATE_IMAGE_PATH_BASE}.{ext}"
    try:
        with open(path, "wb") as f:
            f.write(img_bytes)
        set_setting("initiate_image_mime", mime)
        print(f"[DB] Initiate opener image saved → {path}")
        return True
    except Exception as e:
        print(f"[DB] Error saving initiate image: {e}")
        return False


def clear_initiate_image() -> bool:
    """Remove the saved opener image."""
    _clear_initiate_image_files()
    set_setting("initiate_image_mime", "")
    return True


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
    "editdesc":              "__admin__",
    "editappearance":        "__admin__",
    "viewentry":             "__admin__",
    "knowledge":             "__admin__",
    "saveimage":             "__admin__",
    "addimage":              "__admin__",
    "addcharimage":          "__admin__",
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
    "diagcomfyui":           "__admin__",
}

_SETTINGS_DEFAULTS = {
    "suggestions_enabled":    True,
    "suggestion_prompt":      "",
    "memory_enabled":         True,
    "memory_length":          50,
    "passive_memory_enabled": True,
    "passive_memory_length":  200,
    "command_roles":          dict(_DEFAULT_COMMAND_ROLES),
    "image_style":            "",
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


def delete_setting(key: str) -> bool:
    data = _load_settings()
    if key in data:
        del data[key]
        ok = _save_settings(data)
        if ok:
            print(f"[DB] Setting '{key}' deleted")
        return ok
    return True


def get_settings_by_prefix(prefix: str) -> dict:
    """Return all settings whose key starts with *prefix*."""
    return {k: v for k, v in _load_settings().items() if k.startswith(prefix)}


def get_image_style() -> str:
    """Return the stored image style suffix (empty string if not set)."""
    return str(get_setting("image_style") or "")


def set_image_style(style: str) -> bool:
    """Persist the image style suffix used for every generation."""
    return set_setting("image_style", style.strip())


def clear_image_style() -> bool:
    """Remove the image style suffix."""
    return set_setting("image_style", "")


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


def sync_permissions_file(command_name: str, role_value: Optional[str]) -> None:
    """Write a changed role back into permissions.txt, preserving all comments and formatting."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    perm_path = os.path.join(base_path, "permissions.txt")
    if not os.path.exists(perm_path):
        return

    if role_value is None:
        file_value = "everyone"
    elif role_value == "__admin__":
        file_value = "admin"
    else:
        file_value = role_value

    lines = []
    found = False
    try:
        with open(perm_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.partition("=")[0].strip().lower()
            if key == command_name:
                indent = len(line) - len(line.lstrip())
                padding = max(1, 23 - len(command_name))
                lines[i] = f"{' ' * indent}{command_name}{' ' * padding}= {file_value}\n"
                found = True
                break

        if not found:
            lines.append(f"{command_name:<23}= {file_value}\n")

        with open(perm_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        print(f"[DB] permissions.txt updated: {command_name} = {file_value}")
    except Exception as e:
        print(f"[DB] Could not sync permissions.txt: {e}")


MAX_IMAGES_PER_ENTRY = 5


# ── Knowledge base — file helpers ─────────────────────────────────────────────

def _text_entry_path(entry_id: int) -> str:
    return os.path.join(KNOWLEDGE_DIR, f"{entry_id}.json")


def _image_meta_path(entry_id: int) -> str:
    return os.path.join(IMAGES_DIR, f"{entry_id}.json")


def _safe_ext(mime: str) -> str:
    ext = mime.split("/")[-1] if "/" in mime else "png"
    return ext.lower() if ext.lower() in ("png", "jpg", "jpeg", "gif", "webp") else "png"


def _new_image_filename(entry_id: int, mime: str) -> str:
    import time
    return f"{entry_id}_{int(time.time() * 1000)}.{_safe_ext(mime)}"


def _find_legacy_image_file(entry_id: int) -> Optional[str]:
    """Find old single-image file: <id>.<ext>"""
    for ext in ("png", "jpg", "jpeg", "gif", "webp"):
        p = os.path.join(IMAGES_DIR, f"{entry_id}.{ext}")
        if os.path.exists(p):
            return p
    return None


def _get_meta_with_images(entry_id: int) -> Optional[Dict[str, Any]]:
    """Load image entry metadata, migrating legacy single-image format if needed."""
    meta_path = _image_meta_path(entry_id)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None

    if "images" not in meta:
        mime = meta.get("image_mime", "image/png")
        legacy_path = _find_legacy_image_file(entry_id)
        if legacy_path:
            new_filename = _new_image_filename(entry_id, mime)
            new_path = os.path.join(IMAGES_DIR, new_filename)
            import shutil
            shutil.move(legacy_path, new_path)
            meta["images"] = [{"filename": new_filename, "mime": mime}]
        else:
            meta["images"] = []
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


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


def add_image_entry(
    title: str,
    image_bytes: bytes,
    mime_type: str,
    description: str = "",
    tags: str = "",
    appearance_description: str = "",
    display_description: str = "",
    aliases: Optional[List[str]] = None,
) -> int:
    """Save a new image KB entry.

    appearance_description: bot-only Flux-ready text generated by the vision model.
    display_description: user-facing lore/background text written by the user.
    aliases: optional list of alternative names used by `scene_image._gather_refs`
             to recognise the subject when the seed text refers to it by a
             nickname / pronoun / short-form. Empty/missing keeps title-only
             matching — no migration needed for older entries.

    For backward compatibility, if only `description` is provided (old callers),
    it is stored as `appearance_description`.
    """
    entry_id = _next_id()
    now = datetime.utcnow().isoformat()
    filename = _new_image_filename(entry_id, mime_type)
    img_path = os.path.join(IMAGES_DIR, filename)
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    img_record = {"filename": filename, "mime": mime_type}
    thumb_bytes = _make_thumbnail(image_bytes)
    if thumb_bytes:
        stem = os.path.splitext(filename)[0]
        thumb_filename = f"{stem}_thumb.jpg"
        with open(os.path.join(IMAGES_DIR, thumb_filename), "wb") as f:
            f.write(thumb_bytes)
        img_record["thumb"] = thumb_filename
    # Resolve appearance_description: prefer explicit kwarg, else fall back to `description`
    final_appearance = appearance_description or description
    clean_aliases: List[str] = _normalize_aliases(aliases)
    meta = {
        "id": entry_id,
        "title": title,
        "entry_type": "image",
        "appearance_description": final_appearance,
        "display_description": display_description,
        "aliases": clean_aliases,
        "image_mime": mime_type,
        "tags": tags,
        "created_at": now,
        "updated_at": now,
        "images": [img_record],
    }
    with open(_image_meta_path(entry_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DB] Image entry #{entry_id} saved → knowledge/images/{filename}")
    return entry_id


def get_entry_image_count(entry_id: int) -> int:
    meta = _get_meta_with_images(entry_id)
    if not meta:
        return 0
    return len(meta.get("images", []))


def get_entry_image(entry_id: int, image_index: int) -> Optional[tuple]:
    """Return (bytes, mime) for the image at 1-based image_index, or None."""
    meta = _get_meta_with_images(entry_id)
    if not meta:
        return None
    images = meta.get("images", [])
    if image_index < 1 or image_index > len(images):
        return None
    img_info = images[image_index - 1]
    img_path = os.path.join(IMAGES_DIR, img_info["filename"])
    if not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as f:
        data = f.read()
    return data, img_info.get("mime", "image/png")


def get_kb_image_thumb(entry_id: int, image_index: int = 1) -> Optional[tuple]:
    """Return (thumb_bytes, 'image/jpeg') for a KB image entry, or None.
    image_index is 1-based. Defaults to the first image."""
    meta = _get_meta_with_images(entry_id)
    if not meta:
        return None
    images = meta.get("images", [])
    if image_index < 1 or image_index > len(images):
        return None
    img_info = images[image_index - 1]
    thumb = img_info.get("thumb")
    if not thumb:
        return None
    path = os.path.join(IMAGES_DIR, thumb)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read(), "image/jpeg"


def get_kb_image_full(entry_id: int, image_index: int = 1) -> Optional[tuple]:
    """Return (png_bytes, 'image/png') for a KB image entry at full resolution, or None.
    image_index is 1-based. Converts to PNG (handles WebP, JPEG, GIF, etc.) so the
    caller can pass directly to ComfyUI without further conversion."""
    meta = _get_meta_with_images(entry_id)
    if not meta:
        return None
    images = meta.get("images", [])
    if image_index < 1 or image_index > len(images):
        return None
    img_info = images[image_index - 1]
    filename = img_info.get("filename")
    if not filename:
        return None
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = f.read()
    try:
        from PIL import Image
        import io as _io
        img = Image.open(_io.BytesIO(raw)).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        buf = _io.BytesIO()
        bg.convert("RGB").save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    except Exception as e:
        print(f"[DB] Full image PNG conversion failed for entry #{entry_id} img {image_index}: {e}")
        return raw, img_info.get("mime", "image/png")


def add_image_to_entry(entry_id: int, image_bytes: bytes, mime_type: str) -> tuple:
    """Append an image to an existing image KB entry. Returns (success, message)."""
    meta = _get_meta_with_images(entry_id)
    if not meta:
        return False, "找不到此條目"
    if meta.get("entry_type") != "image":
        return False, "此條目不是圖片類型"
    images = meta.get("images", [])
    if len(images) >= MAX_IMAGES_PER_ENTRY:
        return False, f"已達到每個條目最多 {MAX_IMAGES_PER_ENTRY} 張圖片的上限"
    filename = _new_image_filename(entry_id, mime_type)
    img_path = os.path.join(IMAGES_DIR, filename)
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    img_record = {"filename": filename, "mime": mime_type}
    thumb_bytes = _make_thumbnail(image_bytes)
    if thumb_bytes:
        stem = os.path.splitext(filename)[0]
        thumb_filename = f"{stem}_thumb.jpg"
        with open(os.path.join(IMAGES_DIR, thumb_filename), "wb") as f:
            f.write(thumb_bytes)
        img_record["thumb"] = thumb_filename
    images.append(img_record)
    meta["images"] = images
    meta["updated_at"] = datetime.utcnow().isoformat()
    with open(_image_meta_path(entry_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DB] Added image to entry #{entry_id} ({len(images)}/{MAX_IMAGES_PER_ENTRY})")
    return True, f"已新增，目前 {len(images)} / {MAX_IMAGES_PER_ENTRY} 張"


def remove_image_from_entry(entry_id: int, image_index: int) -> tuple:
    """Remove image at 1-based image_index. Returns (success, message)."""
    meta = _get_meta_with_images(entry_id)
    if not meta:
        return False, "找不到此條目"
    images = meta.get("images", [])
    if image_index < 1 or image_index > len(images):
        return False, "無效的圖片編號"
    img_info = images.pop(image_index - 1)
    img_path = os.path.join(IMAGES_DIR, img_info["filename"])
    if os.path.exists(img_path):
        os.remove(img_path)
    thumb = img_info.get("thumb")
    if thumb:
        thumb_path = os.path.join(IMAGES_DIR, thumb)
        if os.path.exists(thumb_path):
            os.remove(thumb_path)
    meta["images"] = images
    meta["updated_at"] = datetime.utcnow().isoformat()
    with open(_image_meta_path(entry_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return True, f"已移除，剩餘 {len(images)} 張"


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


def get_image_entries() -> List[Dict[str, Any]]:
    """Return all image-type KB entries without any limit (for index building)."""
    return [e for e in _load_all_knowledge_entries() if e.get("entry_type") == "image"]


def get_text_entries() -> List[Dict[str, Any]]:
    """Return all text-type KB entries without any limit."""
    return [e for e in _load_all_knowledge_entries() if e.get("entry_type") == "text"]


def get_entry_by_id(entry_id: int) -> Optional[Dict[str, Any]]:
    text_path = _text_entry_path(entry_id)
    if os.path.exists(text_path):
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    meta = _get_meta_with_images(entry_id)
    return meta


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
            (entry.get("appearance_description") or ""),
            (entry.get("display_description") or ""),
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
        if not (meta.get("appearance_description") or meta.get("display_description")):
            continue
        haystack = " ".join([
            (meta.get("title") or ""),
            (meta.get("appearance_description") or ""),
            (meta.get("display_description") or ""),
            (meta.get("tags") or ""),
        ]).lower()
        match_count = sum(1 for w in words if w in haystack)
        if match_count >= 2:
            results.append((match_count, meta))
        if len(results) >= limit:
            break
    results.sort(key=lambda x: x[0], reverse=True)
    return [meta for _, meta in results]


def _normalize_aliases(value: Any) -> List[str]:
    """Coerce alias input (list, comma-separated string, None) into a clean list.

    Strips empties, dedupes case-insensitively while keeping the first-seen
    casing. Returns [] for falsy / unsupported inputs so older entries that
    simply lack the field continue to round-trip safely.
    """
    if not value:
        return []
    if isinstance(value, str):
        raw = [p.strip() for p in value.replace("\n", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw = []
        for p in value:
            if isinstance(p, str):
                raw.append(p.strip())
    else:
        return []
    seen: set = set()
    out: List[str] = []
    for p in raw:
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def update_aliases(entry_id: int, aliases: Any) -> bool:
    """Update the optional `aliases` list for an image KB entry.

    Accepts a List[str] or a comma-separated string. Empty input clears the
    list (entry falls back to title-only matching).
    """
    meta_path = _image_meta_path(entry_id)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("entry_type") != "image":
            return False
        meta["aliases"] = _normalize_aliases(aliases)
        meta["updated_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[DB] Error updating aliases #{entry_id}: {e}")
        return False


def update_appearance_description(entry_id: int, appearance_description: str) -> bool:
    """Update the hidden bot-only Flux-ready appearance description for an image entry."""
    meta_path = _image_meta_path(entry_id)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("entry_type") != "image":
            return False
        meta["appearance_description"] = appearance_description
        meta["updated_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[DB] Error updating appearance_description #{entry_id}: {e}")
        return False


def update_display_description(entry_id: int, display_description: str) -> bool:
    """Update the user-facing lore/background description for an image entry."""
    meta_path = _image_meta_path(entry_id)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("entry_type") != "image":
            return False
        meta["display_description"] = display_description
        meta["updated_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[DB] Error updating display_description #{entry_id}: {e}")
        return False


def update_image_description(entry_id: int, description: str) -> bool:
    """Backward-compat alias — updates display_description."""
    return update_display_description(entry_id, description)


def migrate_kb_descriptions() -> int:
    """One-time migration: move legacy `image_description` into `appearance_description`.

    For each image entry that still has `image_description` but no `appearance_description`,
    the value is moved to `appearance_description` and `display_description` is set to ''.
    Returns the number of entries migrated.
    """
    migrated = 0
    if not os.path.exists(IMAGES_DIR):
        return 0
    for fname in os.listdir(IMAGES_DIR):
        if not fname.endswith(".json"):
            continue
        meta_path = os.path.join(IMAGES_DIR, fname)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if meta.get("entry_type") != "image":
            continue
        changed = False
        if "appearance_description" not in meta:
            meta["appearance_description"] = meta.get("image_description", "")
            changed = True
        if "display_description" not in meta:
            meta["display_description"] = ""
            changed = True
        if changed:
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                migrated += 1
            except Exception as e:
                print(f"[DB] migrate_kb_descriptions: could not write {fname}: {e}")
    if migrated:
        print(f"[DB] migrate_kb_descriptions: migrated {migrated} image entry/entries.")
    return migrated


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
        try:
            meta = _get_meta_with_images(entry_id) or {}
            for img_info in meta.get("images", []):
                img_file = os.path.join(IMAGES_DIR, img_info["filename"])
                if os.path.exists(img_file):
                    os.remove(img_file)
            legacy = _find_legacy_image_file(entry_id)
            if legacy and os.path.exists(legacy):
                os.remove(legacy)
        except Exception:
            pass
        os.remove(meta_path)
        deleted = True

    return deleted


# ── Conversation history (SQLite — internal only) ─────────────────────────────

def _get_history_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(HISTORY_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS character_state (
            channel_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    try:
        conn.execute("ALTER TABLE character_state ADD COLUMN history_json TEXT")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE character_state ADD COLUMN turn_counter INTEGER DEFAULT 0")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            discord_id   TEXT PRIMARY KEY,
            discord_name TEXT NOT NULL DEFAULT '',
            background   TEXT NOT NULL DEFAULT '',
            looks        TEXT NOT NULL DEFAULT '',
            updated_at   TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS player_state (
            channel_id   TEXT NOT NULL,
            discord_id   TEXT NOT NULL,
            state_json   TEXT NOT NULL,
            updated_at   TEXT NOT NULL,
            history_json TEXT,
            turn_counter INTEGER DEFAULT 0,
            PRIMARY KEY (channel_id, discord_id)
        )
    """)
    conn.commit()
    conn.close()


# ── Character appearance state (SQLite) ───────────────────────────────────────

def get_character_state(channel_id: str) -> Optional[Dict[str, Any]]:
    """Return the persisted appearance state dict for a channel, or None."""
    conn = _get_history_conn()
    row = conn.execute(
        "SELECT state_json FROM character_state WHERE channel_id = ?",
        (channel_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    try:
        return json.loads(row["state_json"])
    except Exception as e:
        print(f"[DB] Could not parse character_state JSON for channel {channel_id}: {e}")
        return None


def set_character_state(channel_id: str, state_dict: Dict[str, Any]) -> None:
    """Upsert the appearance state dict for a channel."""
    try:
        conn = _get_history_conn()
        conn.execute(
            """
            INSERT INTO character_state (channel_id, state_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (channel_id, json.dumps(state_dict, ensure_ascii=False), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Failed to save character_state for channel {channel_id}: {e}")


def delete_character_state(channel_id: Optional[str] = None) -> None:
    """Delete appearance state for one channel, or ALL channels when None."""
    try:
        conn = _get_history_conn()
        if channel_id is None:
            conn.execute("DELETE FROM character_state")
        else:
            conn.execute("DELETE FROM character_state WHERE channel_id = ?", (channel_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Failed to delete character_state (channel={channel_id}): {e}")


# ── Player appearance state (SQLite) ──────────────────────────────────────────

def get_player_state(channel_id: str, discord_id: str) -> Optional[Dict[str, Any]]:
    """Return the persisted player appearance state dict, or None."""
    conn = _get_history_conn()
    row = conn.execute(
        "SELECT state_json FROM player_state WHERE channel_id = ? AND discord_id = ?",
        (channel_id, discord_id),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    try:
        return json.loads(row["state_json"])
    except Exception as e:
        print(f"[DB] Could not parse player_state JSON for channel {channel_id} / user {discord_id}: {e}")
        return None


def set_player_state(channel_id: str, discord_id: str, state_dict: Dict[str, Any]) -> None:
    """Upsert the player appearance state dict."""
    try:
        conn = _get_history_conn()
        conn.execute(
            """
            INSERT INTO player_state (channel_id, discord_id, state_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(channel_id, discord_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = excluded.updated_at
            """,
            (channel_id, discord_id, json.dumps(state_dict, ensure_ascii=False), datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Failed to save player_state for channel {channel_id} / user {discord_id}: {e}")


def delete_player_state(channel_id: Optional[str] = None, discord_id: Optional[str] = None) -> None:
    """Delete player appearance state.

    - Both None: delete all rows.
    - Only channel_id: delete all players in that channel.
    - Both set: delete that specific player in that channel.
    """
    try:
        conn = _get_history_conn()
        if channel_id is None and discord_id is None:
            conn.execute("DELETE FROM player_state")
        elif discord_id is None:
            conn.execute("DELETE FROM player_state WHERE channel_id = ?", (channel_id,))
        else:
            conn.execute(
                "DELETE FROM player_state WHERE channel_id = ? AND discord_id = ?",
                (channel_id, discord_id),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Failed to delete player_state (channel={channel_id}, discord_id={discord_id}): {e}")


def get_player_history(channel_id: str, discord_id: str) -> List[Dict[str, Any]]:
    """Return the persisted player appearance history list, or []."""
    try:
        conn = _get_history_conn()
        row = conn.execute(
            "SELECT history_json FROM player_state WHERE channel_id = ? AND discord_id = ?",
            (channel_id, discord_id),
        ).fetchone()
        conn.close()
        if row is None or row["history_json"] is None:
            return []
        return json.loads(row["history_json"])
    except Exception as e:
        print(f"[DB] Could not load player history for channel {channel_id} / user {discord_id}: {e}")
        return []


def set_player_history(channel_id: str, discord_id: str, history: List[Dict[str, Any]]) -> None:
    """Persist the player appearance history list."""
    try:
        conn = _get_history_conn()
        conn.execute(
            "UPDATE player_state SET history_json = ? WHERE channel_id = ? AND discord_id = ?",
            (json.dumps(history, ensure_ascii=False), channel_id, discord_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Failed to save player history for channel {channel_id} / user {discord_id}: {e}")


def get_player_turn_counter(channel_id: str, discord_id: str) -> int:
    """Return the persisted turn counter for a player in a channel, or 0."""
    try:
        conn = _get_history_conn()
        row = conn.execute(
            "SELECT turn_counter FROM player_state WHERE channel_id = ? AND discord_id = ?",
            (channel_id, discord_id),
        ).fetchone()
        conn.close()
        if row is None or row["turn_counter"] is None:
            return 0
        return int(row["turn_counter"])
    except Exception as e:
        print(f"[DB] Could not load player turn_counter for channel {channel_id} / user {discord_id}: {e}")
        return 0


def set_player_turn_counter(channel_id: str, discord_id: str, counter: int) -> None:
    """Persist the turn counter for a player in a channel."""
    try:
        conn = _get_history_conn()
        conn.execute(
            "UPDATE player_state SET turn_counter = ? WHERE channel_id = ? AND discord_id = ?",
            (counter, channel_id, discord_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Failed to save player turn_counter for channel {channel_id} / user {discord_id}: {e}")


# ── User profiles ─────────────────────────────────────────────────────────────

def _user_images_dir(discord_id: str) -> str:
    d = os.path.join(USER_PROFILES_DIR, discord_id, "images")
    os.makedirs(d, exist_ok=True)
    return d


def _user_images_json(discord_id: str) -> str:
    return os.path.join(USER_PROFILES_DIR, discord_id, "images.json")


def _read_user_images(discord_id: str) -> list:
    path = _user_images_json(discord_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("images", [])
        except Exception:
            pass
    return []


def _write_user_images(discord_id: str, images: list):
    path = _user_images_json(discord_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"images": images}, f, ensure_ascii=False, indent=2)


def get_user_profile(discord_id: str) -> Optional[Dict[str, Any]]:
    """Return the user profile dict or None if not found."""
    try:
        conn = _get_history_conn()
        row = conn.execute(
            "SELECT discord_id, discord_name, background, looks, updated_at FROM user_profiles WHERE discord_id = ?",
            (discord_id,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return dict(row)
    except Exception as e:
        print(f"[DB] get_user_profile error: {e}")
        return None


def set_user_profile(discord_id: str, discord_name: str, background: str, looks: str) -> bool:
    """Upsert a user profile. Returns True on success."""
    try:
        now = datetime.utcnow().isoformat()
        conn = _get_history_conn()
        conn.execute(
            """
            INSERT INTO user_profiles (discord_id, discord_name, background, looks, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(discord_id) DO UPDATE SET
                discord_name = excluded.discord_name,
                background   = excluded.background,
                looks        = excluded.looks,
                updated_at   = excluded.updated_at
            """,
            (discord_id, discord_name, background, looks, now),
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] set_user_profile error: {e}")
        return False


def get_user_profile_image_count(discord_id: str) -> int:
    return len(_read_user_images(discord_id))


def get_user_profile_images_meta(discord_id: str) -> list:
    return _read_user_images(discord_id)


def get_user_profile_image(discord_id: str, index: int) -> Optional[tuple]:
    """1-indexed. Returns (bytes, mime) or None."""
    images = _read_user_images(discord_id)
    if index < 1 or index > len(images):
        return None
    info = images[index - 1]
    path = os.path.join(_user_images_dir(discord_id), info["filename"])
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read(), info.get("mime", "image/png")


def get_user_profile_image_thumb(discord_id: str, index: int) -> Optional[tuple]:
    """1-indexed. Returns (thumb_bytes, 'image/jpeg') or None."""
    images = _read_user_images(discord_id)
    if index < 1 or index > len(images):
        return None
    info = images[index - 1]
    thumb = info.get("thumb")
    if not thumb:
        return None
    path = os.path.join(_user_images_dir(discord_id), thumb)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read(), "image/jpeg"


def add_user_profile_image(discord_id: str, img_bytes: bytes, mime: str, description: str = "") -> tuple:
    """Add a user profile reference image. Returns (success, message)."""
    images = _read_user_images(discord_id)
    if len(images) >= MAX_USER_PROFILE_IMAGES:
        return False, f"已達上限 {MAX_USER_PROFILE_IMAGES} 張個人參考圖片。"
    ext = mime.split("/")[-1] if "/" in mime else "png"
    if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
        ext = "png"
    import time as _time
    ts = int(_time.time() * 1000)
    images_dir = _user_images_dir(discord_id)
    filename = f"user_{ts}.{ext}"
    while os.path.exists(os.path.join(images_dir, filename)):
        ts += 1
        filename = f"user_{ts}.{ext}"
    path = os.path.join(images_dir, filename)
    with open(path, "wb") as f:
        f.write(img_bytes)
    thumb_filename = None
    thumb_bytes = _make_thumbnail(img_bytes)
    if thumb_bytes:
        thumb_filename = f"user_{ts}_thumb.jpg"
        with open(os.path.join(images_dir, thumb_filename), "wb") as f:
            f.write(thumb_bytes)
    entry = {"filename": filename, "mime": mime, "description": description}
    if thumb_filename:
        entry["thumb"] = thumb_filename
    images.append(entry)
    _write_user_images(discord_id, images)
    return True, f"已新增第 {len(images)} 張個人參考圖片。"


def remove_user_profile_image(discord_id: str, index: int) -> tuple:
    """1-indexed. Returns (success, message)."""
    images = _read_user_images(discord_id)
    if index < 1 or index > len(images):
        return False, "索引超出範圍。"
    info = images.pop(index - 1)
    images_dir = _user_images_dir(discord_id)
    path = os.path.join(images_dir, info["filename"])
    if os.path.exists(path):
        os.remove(path)
    thumb = info.get("thumb")
    if thumb:
        thumb_path = os.path.join(images_dir, thumb)
        if os.path.exists(thumb_path):
            os.remove(thumb_path)
    _write_user_images(discord_id, images)
    return True, "已移除圖片。"


def get_character_history(channel_id: str) -> List[Dict[str, Any]]:
    """Return the persisted appearance history list for a channel, or []."""
    try:
        conn = _get_history_conn()
        row = conn.execute(
            "SELECT history_json FROM character_state WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
        conn.close()
        if row is None or row["history_json"] is None:
            return []
        return json.loads(row["history_json"])
    except Exception as e:
        print(f"[DB] Could not load character history for channel {channel_id}: {e}")
        return []


def set_character_history(channel_id: str, history: List[Dict[str, Any]]) -> None:
    """Persist the appearance history list for a channel.

    Requires a character_state row to already exist for the channel (i.e. call
    set_character_state first). This is always the case because history is only
    written after a state change, which itself writes the state row.
    """
    try:
        conn = _get_history_conn()
        conn.execute(
            """
            UPDATE character_state SET history_json = ?
            WHERE channel_id = ?
            """,
            (json.dumps(history, ensure_ascii=False), channel_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Failed to save character history for channel {channel_id}: {e}")


def get_character_turn_counter(channel_id: str) -> int:
    """Return the persisted turn counter for a channel, or 0 if none stored."""
    try:
        conn = _get_history_conn()
        row = conn.execute(
            "SELECT turn_counter FROM character_state WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
        conn.close()
        if row is None or row["turn_counter"] is None:
            return 0
        return int(row["turn_counter"])
    except Exception as e:
        print(f"[DB] Could not load turn_counter for channel {channel_id}: {e}")
        return 0


def set_character_turn_counter(channel_id: str, counter: int) -> None:
    """Persist the turn counter for a channel.

    Requires a character_state row to already exist for the channel (i.e. call
    set_character_state first). This is always the case because the turn counter
    is only incremented during update_state, which writes the state row first.
    """
    try:
        conn = _get_history_conn()
        conn.execute(
            "UPDATE character_state SET turn_counter = ? WHERE channel_id = ?",
            (counter, channel_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Failed to save turn_counter for channel {channel_id}: {e}")


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
                            img_filename = _new_image_filename(old_id, mime)
                            img_path = os.path.join(IMAGES_DIR, img_filename)
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

def migrate_thumbnails():
    """Regenerate thumbnails for all images at the current target resolution.
    Safe to call on every startup — always regenerates to pick up resolution changes."""
    generated = 0

    # Character images
    images = _read_char_images()
    changed = False
    for info in images:
        src_path = os.path.join(CHARACTER_IMAGES_DIR, info["filename"])
        if not os.path.exists(src_path):
            continue
        with open(src_path, "rb") as f:
            raw = f.read()
        thumb_bytes = _make_thumbnail(raw)
        if not thumb_bytes:
            continue
        stem = os.path.splitext(info["filename"])[0]
        thumb_filename = f"{stem}_thumb.jpg"
        with open(os.path.join(CHARACTER_IMAGES_DIR, thumb_filename), "wb") as f:
            f.write(thumb_bytes)
        info["thumb"] = thumb_filename
        changed = True
        generated += 1
    if changed:
        _write_char_images(images)

    # KB images
    for fname in os.listdir(IMAGES_DIR):
        if not fname.endswith(".json"):
            continue
        meta_path = os.path.join(IMAGES_DIR, fname)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        img_list = meta.get("images", [])
        changed = False
        for img_info in img_list:
            src_path = os.path.join(IMAGES_DIR, img_info["filename"])
            if not os.path.exists(src_path):
                continue
            with open(src_path, "rb") as f:
                raw = f.read()
            thumb_bytes = _make_thumbnail(raw)
            if not thumb_bytes:
                continue
            stem = os.path.splitext(img_info["filename"])[0]
            thumb_filename = f"{stem}_thumb.jpg"
            with open(os.path.join(IMAGES_DIR, thumb_filename), "wb") as f:
                f.write(thumb_bytes)
            img_info["thumb"] = thumb_filename
            changed = True
            generated += 1
        if changed:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    if generated:
        print(f"[DB] migrate_thumbnails: regenerated {generated} thumbnail(s) at 512x512")


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
