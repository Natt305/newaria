"""Engine-scope ComfyUI custom-node packs before launch.

Invoked by `start.bat` immediately before it boots ComfyUI. Reads
`COMFYUI_ENGINE` and `COMFYUI_PATH` from `tokens.txt` (or the env),
parses `comfyui_engine_packs.txt`, and renames pack folders inside
`<COMFYUI_PATH>/custom_nodes/` so ComfyUI only imports the packs the
active engine needs. Disabling = renaming `<pack>` to
`<pack>.disabled`; re-enabling = stripping the suffix.

Safety:
  * Only ever renames folders directly inside `custom_nodes/`.
  * Never deletes anything.
  * Never touches `<COMFYUI_PATH>/models/` or any other directory.
  * Skips packs whose folders don't exist (silent no-op).
  * Refuses to run if COMFYUI_PATH/custom_nodes/ does not exist.

Always exits 0 — failure to toggle a pack must not block the bot
from starting. All actions are printed so the user sees them in the
start.bat console window before ComfyUI launches.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS_FILE = os.path.join(REPO_ROOT, "tokens.txt")
MANIFEST_FILE = os.path.join(REPO_ROOT, "comfyui_engine_packs.txt")

VALID_ENGINES = ("qwen", "flux")
SHARED_TAGS = ("both", "shared")


def _is_safe_pack_name(name: str) -> bool:
    """Reject manifest entries that could escape `custom_nodes/`. We only
    allow plain folder names — no path separators, no drive prefixes, no
    `.` / `..`, no trailing `.disabled` (the toggling code adds that
    suffix itself, and accepting it would let a typo silently no-op or
    target the wrong directory)."""
    if not name or name in (".", ".."):
        return False
    if name != os.path.basename(name):
        return False
    if "/" in name or "\\" in name or os.sep in name:
        return False
    if os.path.isabs(name):
        return False
    if name.endswith(".disabled"):
        return False
    return True


def _is_inside(path: str, parent: str) -> bool:
    """Defense-in-depth check that `path` resolves to something underneath
    `parent`. Used after `os.path.join` to make absolutely sure no rename
    target escapes the `custom_nodes/` scope, even if `_is_safe_pack_name`
    is somehow bypassed."""
    try:
        path_abs = os.path.realpath(path)
        parent_abs = os.path.realpath(parent)
    except OSError:
        return False
    if not parent_abs.endswith(os.sep):
        parent_abs_pref = parent_abs + os.sep
    else:
        parent_abs_pref = parent_abs
    return path_abs == parent_abs or path_abs.startswith(parent_abs_pref)


def _load_tokens(path: str) -> Dict[str, str]:
    """Parse tokens.txt the same way launcher.load_tokens_file does, but
    return the values instead of mutating os.environ. Env vars already
    set in the parent shell win, matching launcher.py's behavior."""
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key and value:
                    out[key] = value
    except OSError as exc:
        print(f"[scope-packs] WARN: could not read {path}: {exc}")
    return out


def _parse_manifest(path: str) -> Dict[str, Set[str]]:
    """Return {engine_tag: {pack_folder_name, ...}} from the manifest.
    Tags `both`/`shared` are recorded but ignored downstream — listed
    only so the user can document shared packs explicitly without
    triggering a rename."""
    by_engine: Dict[str, Set[str]] = {tag: set() for tag in (*VALID_ENGINES, *SHARED_TAGS)}
    if not os.path.exists(path):
        print(f"[scope-packs] No manifest at {path} — nothing to toggle.")
        return by_engine
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # strip inline `#` comments
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                tag, _, name = line.partition(":")
                tag = tag.strip().lower()
                name = name.strip()
                if not tag or not name:
                    continue
                if tag not in by_engine:
                    print(f"[scope-packs] WARN: unknown engine tag `{tag}` "
                          f"on line {lineno}; expected one of "
                          f"{', '.join((*VALID_ENGINES, *SHARED_TAGS))}.")
                    continue
                if not _is_safe_pack_name(name):
                    print(f"[scope-packs] WARN: rejected unsafe pack name "
                          f"`{name}` on line {lineno} (must be a plain "
                          f"folder name with no path separators, no "
                          f"`.`/`..`, no `.disabled` suffix).")
                    continue
                by_engine[tag].add(name)
    except OSError as exc:
        print(f"[scope-packs] WARN: could not read {path}: {exc}")
    return by_engine


def _toggle(custom_nodes_dir: str, manifest: Dict[str, Set[str]],
            active_engine: str) -> Tuple[List[str], List[str], List[str]]:
    """Apply renames. Returns (disabled_now, enabled_now, missing).

    For every pack listed under a non-matching, non-shared engine:
      if `<pack>` exists -> rename to `<pack>.disabled` (record as disabled_now)
      if it's already `.disabled` -> no-op
    For every pack listed under the matching engine:
      if `<pack>.disabled` exists -> strip suffix (record as enabled_now)
      if `<pack>` already exists -> no-op
    `missing` collects manifest entries whose folder doesn't exist in
    either form so the user can spot typos in the manifest."""
    disabled_now: List[str] = []
    enabled_now: List[str] = []
    missing: List[str] = []

    # Targets to disable: union of every non-matching engine's packs,
    # minus anything also listed for the active engine or as shared.
    keep_enabled: Set[str] = set(manifest.get(active_engine, set()))
    for tag in SHARED_TAGS:
        keep_enabled |= manifest.get(tag, set())

    targets_to_disable: Set[str] = set()
    for tag in VALID_ENGINES:
        if tag == active_engine:
            continue
        for name in manifest.get(tag, set()):
            if name not in keep_enabled:
                targets_to_disable.add(name)

    targets_to_enable: Set[str] = set(manifest.get(active_engine, set()))

    # Disable pass
    for name in sorted(targets_to_disable):
        live = os.path.join(custom_nodes_dir, name)
        gone = os.path.join(custom_nodes_dir, name + ".disabled")
        # Defense-in-depth: even though manifest names are validated up
        # front, double-check the joined paths land inside custom_nodes/.
        if not (_is_inside(live, custom_nodes_dir)
                and _is_inside(gone, custom_nodes_dir)):
            print(f"[scope-packs] WARN: refusing to rename `{name}` — "
                  f"target path escapes custom_nodes/.")
            continue
        if os.path.isdir(live):
            try:
                # If a stale .disabled exists alongside, leave the live one
                # in place rather than overwriting (defensive — shouldn't
                # normally happen, but better than a crash).
                if os.path.exists(gone):
                    print(f"[scope-packs] SKIP disable {name}: both "
                          f"`{name}` and `{name}.disabled` exist; leaving as-is.")
                    continue
                os.rename(live, gone)
                disabled_now.append(name)
            except OSError as exc:
                print(f"[scope-packs] WARN: could not disable {name}: {exc}")
        elif not os.path.isdir(gone):
            missing.append(name)
        # else: already disabled — silent no-op

    # Enable pass
    for name in sorted(targets_to_enable):
        live = os.path.join(custom_nodes_dir, name)
        gone = os.path.join(custom_nodes_dir, name + ".disabled")
        if not (_is_inside(live, custom_nodes_dir)
                and _is_inside(gone, custom_nodes_dir)):
            print(f"[scope-packs] WARN: refusing to rename `{name}` — "
                  f"target path escapes custom_nodes/.")
            continue
        if os.path.isdir(gone):
            try:
                if os.path.exists(live):
                    print(f"[scope-packs] SKIP enable {name}: both "
                          f"`{name}` and `{name}.disabled` exist; leaving as-is.")
                    continue
                os.rename(gone, live)
                enabled_now.append(name)
            except OSError as exc:
                print(f"[scope-packs] WARN: could not enable {name}: {exc}")
        elif not os.path.isdir(live):
            missing.append(name)
        # else: already enabled — silent no-op

    return disabled_now, enabled_now, missing


def main() -> int:
    tokens = _load_tokens(TOKENS_FILE)
    # env wins, then tokens.txt, then defaults
    engine = (os.environ.get("COMFYUI_ENGINE")
              or tokens.get("COMFYUI_ENGINE")
              or "qwen").strip().lower()
    if engine not in VALID_ENGINES:
        print(f"[scope-packs] Unknown COMFYUI_ENGINE=`{engine}` — defaulting to qwen.")
        engine = "qwen"

    comfy_path = (os.environ.get("COMFYUI_PATH")
                  or tokens.get("COMFYUI_PATH")
                  or "").strip()
    if not comfy_path:
        print("[scope-packs] COMFYUI_PATH is not set — skipping engine-scoped pack toggling.")
        return 0

    custom_nodes_dir = os.path.join(comfy_path, "custom_nodes")
    if not os.path.isdir(custom_nodes_dir):
        print(f"[scope-packs] {custom_nodes_dir} does not exist — skipping.")
        return 0

    manifest = _parse_manifest(MANIFEST_FILE)
    total_listed = sum(len(manifest.get(tag, set())) for tag in VALID_ENGINES)
    if total_listed == 0:
        print(f"[scope-packs] Manifest has no engine-scoped entries — nothing to toggle.")
        return 0

    print(f"[scope-packs] Engine = {engine}  |  custom_nodes = {custom_nodes_dir}")
    disabled_now, enabled_now, missing = _toggle(custom_nodes_dir, manifest, engine)

    if disabled_now:
        print(f"[scope-packs] Disabled {len(disabled_now)} pack(s) for engine={engine}:")
        for name in disabled_now:
            print(f"[scope-packs]   - {name}  ->  {name}.disabled")
    if enabled_now:
        print(f"[scope-packs] Re-enabled {len(enabled_now)} pack(s) for engine={engine}:")
        for name in enabled_now:
            print(f"[scope-packs]   + {name}.disabled  ->  {name}")
    if not disabled_now and not enabled_now:
        print(f"[scope-packs] No changes needed — pack state already matches engine={engine}.")

    if missing:
        # de-dup while preserving order
        seen: Set[str] = set()
        uniq = [n for n in missing if not (n in seen or seen.add(n))]
        print(f"[scope-packs] Note: {len(uniq)} manifest entr{'y' if len(uniq) == 1 else 'ies'} "
              f"not found in custom_nodes/ (typo or pack not installed): "
              f"{', '.join(uniq)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
