from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "Stemsplat"
APP_SLUG = "stemsplat"
LEGACY_MODELS_DIR = Path.home() / "Library" / "Application Support" / "stems"


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        for candidate in (os.environ.get("RESOURCEPATH"), getattr(sys, "_MEIPASS", None)):
            if candidate:
                return Path(candidate).expanduser().resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _app_support_root() -> Path:
    override = os.environ.get("STEMSPLAT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / APP_SLUG).resolve()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata).expanduser() if appdata else (Path.home() / "AppData" / "Roaming")
        return (base / APP_SLUG).resolve()
    return (Path.home() / ".local" / "share" / APP_SLUG).resolve()


RESOURCE_DIR = _resource_root()
APP_SUPPORT_DIR = _app_support_root()
CONFIG_DIR = RESOURCE_DIR / "configs"
WEB_DIR = RESOURCE_DIR / "web"
MODEL_DIR = APP_SUPPORT_DIR / "models"
LOG_DIR = APP_SUPPORT_DIR / "logs"
RUNTIME_DIR = APP_SUPPORT_DIR / ".runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
WORK_DIR = RUNTIME_DIR / "work"
ARTWORK_DIR = RUNTIME_DIR / "artwork"
INTERMEDIATE_CACHE_DIR = APP_SUPPORT_DIR / "intermediate_cache"
PREVIOUS_FILES_DIR = APP_SUPPORT_DIR / "previous_files"
OUTPUT_ROOT = (Path.home() / "Downloads").expanduser()
SETTINGS_PATH = APP_SUPPORT_DIR / "settings.json"
ETA_HISTORY_PATH = APP_SUPPORT_DIR / "eta_history.json"
PREVIOUS_FILES_INDEX_PATH = APP_SUPPORT_DIR / "previous_files.json"


def ensure_app_dirs() -> None:
    for path in (
        APP_SUPPORT_DIR,
        MODEL_DIR,
        LOG_DIR,
        RUNTIME_DIR,
        UPLOAD_DIR,
        WORK_DIR,
        ARTWORK_DIR,
        INTERMEDIATE_CACHE_DIR,
        PREVIOUS_FILES_DIR,
        OUTPUT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def model_search_dirs() -> list[Path]:
    candidates = [
        MODEL_DIR,
        RESOURCE_DIR / "models",
        RESOURCE_DIR / "Models",
        LEGACY_MODELS_DIR,
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered
