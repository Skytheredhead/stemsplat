from __future__ import annotations

import base64
import contextlib
import json
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
TERMINAL_TASK_STATUSES = frozenset({"done", "error", "stopped", "interrupted"})
RECOVERABLE_TASK_STATUSES = frozenset({"ready", "queued", "running"})


class StateConflictError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items() if str(key) != "stop_event"}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return None


def _encode_cursor(updated_at: float, item_id: str) -> str:
    raw = json.dumps([updated_at, item_id], separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None) -> tuple[float, str] | None:
    if not cursor:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        value = json.loads(raw)
        if not isinstance(value, list) or len(value) != 2:
            return None
        return float(value[0]), str(value[1])
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


@dataclass(frozen=True)
class Page:
    items: list[dict[str, Any]]
    next_cursor: str | None


class StateStore:
    def __init__(self, path: Path, *, clock=time.time) -> None:
        self.path = path
        self.clock = clock
        self._migration_lock = threading.RLock()
        self._progress_lock = threading.RLock()
        self._progress_state: dict[str, tuple[float, str, str]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def migrate(self) -> None:
        with self._migration_lock, self._connect() as connection:
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    queue_position INTEGER,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS tasks_status_updated_idx
                    ON tasks(status, updated_at DESC, id DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS tasks_queue_position_idx
                    ON tasks(queue_position) WHERE queue_position IS NOT NULL;
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    task_id TEXT,
                    payload_json TEXT NOT NULL,
                    finished_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS history_finished_idx
                    ON history(finished_at DESC, id DESC);
                INSERT INTO metadata(key, value) VALUES('schema_version', '1')
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value;
                COMMIT;
                """
            )

    def close(self) -> None:
        return

    def get_metadata(self, key: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
            return str(row["value"]) if row is not None else None

    def set_metadata(self, key: str, value: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def load_settings(self) -> tuple[dict[str, Any], int] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT version, payload_json FROM settings WHERE singleton = 1").fetchone()
        if row is None:
            return None
        payload = json.loads(str(row["payload_json"]))
        return (payload if isinstance(payload, dict) else {}, int(row["version"]))

    def save_settings(self, payload: dict[str, Any], *, expected_version: int | None = None) -> int:
        now = self.clock()
        encoded = json.dumps(_json_safe(payload), separators=(",", ":"), sort_keys=True)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT version FROM settings WHERE singleton = 1").fetchone()
                current_version = int(row["version"]) if row is not None else 0
                if expected_version is not None and expected_version != current_version:
                    raise StateConflictError("settings changed on another client")
                next_version = current_version + 1
                connection.execute(
                    """
                    INSERT INTO settings(singleton, version, payload_json, updated_at)
                    VALUES(1, ?, ?, ?)
                    ON CONFLICT(singleton) DO UPDATE SET
                        version=excluded.version,
                        payload_json=excluded.payload_json,
                        updated_at=excluded.updated_at
                    """,
                    (next_version, encoded, now),
                )
                connection.commit()
                return next_version
            except Exception:
                connection.rollback()
                raise

    def save_task(
        self,
        payload: dict[str, Any],
        *,
        queue_position: int | None = None,
        force: bool = False,
    ) -> bool:
        task_id = str(payload.get("id") or "")
        if not task_id:
            raise ValueError("task id is required")
        now = self.clock()
        status = str(payload.get("status") or "ready")
        stage = str(payload.get("stage") or "")
        with self._progress_lock:
            previous = self._progress_state.get(task_id)
            if (
                not force
                and previous is not None
                and previous[1] == status
                and previous[2] == stage
                and now - previous[0] < 1.0
            ):
                return False
            self._progress_state[task_id] = (now, status, stage)
        encoded = json.dumps(_json_safe(payload), separators=(",", ":"), sort_keys=True)
        created_at = float(payload.get("created_at") or now)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO tasks(id, status, queue_position, payload_json, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    queue_position=excluded.queue_position,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (task_id, status, queue_position, encoded, created_at, now),
            )
        return True

    def save_queue(self, task_ids: Iterable[str]) -> None:
        ordered = list(dict.fromkeys(str(item) for item in task_ids if str(item)))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("UPDATE tasks SET queue_position = NULL")
                for position, task_id in enumerate(ordered):
                    connection.execute(
                        "UPDATE tasks SET queue_position = ?, updated_at = ? WHERE id = ?",
                        (position, self.clock(), task_id),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def delete_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT payload_json FROM tasks WHERE id = ?", (task_id,)).fetchone()
                connection.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return json.loads(str(row["payload_json"])) if row is not None else None

    def recover_tasks(self) -> list[dict[str, Any]]:
        recovered: list[dict[str, Any]] = []
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                rows = connection.execute(
                    "SELECT id, status, payload_json FROM tasks ORDER BY created_at ASC, id ASC"
                ).fetchall()
                for row in rows:
                    payload = json.loads(str(row["payload_json"]))
                    status = str(row["status"])
                    if status in {"queued", "running"}:
                        payload["status"] = "interrupted"
                        payload["stage"] = "Interrupted by app restart"
                        payload["error"] = "Processing was interrupted. Retry when ready."
                        payload["retryable"] = True
                        payload["finished_at"] = self.clock()
                        payload["version"] = int(payload.get("version") or 0) + 1
                        encoded = json.dumps(_json_safe(payload), separators=(",", ":"), sort_keys=True)
                        connection.execute(
                            "UPDATE tasks SET status = 'interrupted', queue_position = NULL, "
                            "payload_json = ?, updated_at = ? WHERE id = ?",
                            (encoded, self.clock(), str(row["id"])),
                        )
                    recovered.append(payload)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return recovered

    def list_tasks(self, *, status: str | None = None, cursor: str | None = None, limit: int = 50) -> Page:
        return self._list_page("tasks", status=status, cursor=cursor, limit=limit)

    def save_history(self, payload: dict[str, Any]) -> None:
        entry_id = str(payload.get("id") or "")
        if not entry_id:
            raise ValueError("history id is required")
        finished_at = float(payload.get("finished_at") or self.clock())
        encoded = json.dumps(_json_safe(payload), separators=(",", ":"), sort_keys=True)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO history(id, task_id, payload_json, finished_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    task_id=excluded.task_id,
                    payload_json=excluded.payload_json,
                    finished_at=excluded.finished_at,
                    updated_at=excluded.updated_at
                """,
                (entry_id, str(payload.get("task_id") or ""), encoded, finished_at, self.clock()),
            )

    def list_history(self, *, cursor: str | None = None, limit: int = 50) -> Page:
        return self._list_page("history", cursor=cursor, limit=limit)

    def delete_history(self, entry_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute("SELECT payload_json FROM history WHERE id = ?", (entry_id,)).fetchone()
                connection.execute("DELETE FROM history WHERE id = ?", (entry_id,))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return json.loads(str(row["payload_json"])) if row is not None else None

    def _list_page(
        self,
        table: str,
        *,
        status: str | None = None,
        cursor: str | None,
        limit: int,
    ) -> Page:
        if table not in {"tasks", "history"}:
            raise ValueError("invalid table")
        bounded_limit = max(1, min(100, int(limit)))
        decoded = _decode_cursor(cursor)
        parameters: list[Any] = []
        if table == "tasks":
            if status and decoded is not None:
                query = (
                    "SELECT id, payload_json, updated_at FROM tasks WHERE status = ? AND "
                    "(updated_at < ? OR (updated_at = ? AND id < ?)) "
                    "ORDER BY updated_at DESC, id DESC LIMIT ?"
                )
                parameters.extend([status, decoded[0], decoded[0], decoded[1]])
            elif status:
                query = (
                    "SELECT id, payload_json, updated_at FROM tasks WHERE status = ? "
                    "ORDER BY updated_at DESC, id DESC LIMIT ?"
                )
                parameters.append(status)
            elif decoded is not None:
                query = (
                    "SELECT id, payload_json, updated_at FROM tasks WHERE "
                    "(updated_at < ? OR (updated_at = ? AND id < ?)) "
                    "ORDER BY updated_at DESC, id DESC LIMIT ?"
                )
                parameters.extend([decoded[0], decoded[0], decoded[1]])
            else:
                query = "SELECT id, payload_json, updated_at FROM tasks ORDER BY updated_at DESC, id DESC LIMIT ?"
        elif decoded is not None:
            query = (
                "SELECT id, payload_json, updated_at FROM history WHERE "
                "(updated_at < ? OR (updated_at = ? AND id < ?)) "
                "ORDER BY updated_at DESC, id DESC LIMIT ?"
            )
            parameters.extend([decoded[0], decoded[0], decoded[1]])
        else:
            query = "SELECT id, payload_json, updated_at FROM history ORDER BY updated_at DESC, id DESC LIMIT ?"
        parameters.append(bounded_limit + 1)
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        has_more = len(rows) > bounded_limit
        selected = rows[:bounded_limit]
        items = [json.loads(str(row["payload_json"])) for row in selected]
        next_cursor = None
        if has_more and selected:
            last = selected[-1]
            next_cursor = _encode_cursor(float(last["updated_at"]), str(last["id"]))
        return Page(items, next_cursor)

    def import_legacy_once(
        self,
        *,
        settings_path: Path,
        history_path: Path,
        eta_path: Path,
    ) -> dict[str, int]:
        if self.get_metadata("legacy_import_complete") == "1":
            return {"settings": 0, "history": 0, "eta": 0}
        counts = {"settings": 0, "history": 0, "eta": 0}
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.clock()))
        settings = self._read_legacy_json(settings_path, dict, stamp)
        history = self._read_legacy_json(history_path, list, stamp)
        eta = self._read_legacy_json(eta_path, dict, stamp)
        if isinstance(settings, dict):
            self.save_settings(settings)
            counts["settings"] = 1
        if isinstance(history, list):
            for item in history:
                if isinstance(item, dict) and item.get("id"):
                    self.save_history(item)
                    counts["history"] += 1
        if isinstance(eta, dict):
            self.set_metadata("legacy_eta_json", json.dumps(_json_safe(eta), separators=(",", ":"), sort_keys=True))
            counts["eta"] = 1
        self.set_metadata("legacy_import_complete", "1")
        self.set_metadata("legacy_imported_at", str(self.clock()))
        return counts

    @staticmethod
    def _read_legacy_json(path: Path, expected_type: type, stamp: str) -> Any | None:
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, expected_type):
            return None
        backup = path.with_name(f"{path.name}.v0.4.2.bak.{stamp}")
        with contextlib.suppress(OSError):
            shutil.copy2(path, backup)
        return payload
