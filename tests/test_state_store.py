from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stemsplat.state import StateConflictError, StateStore


class MutableClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class StateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.clock = MutableClock()
        self.store = StateStore(self.root / "state.sqlite3", clock=self.clock)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_settings_are_versioned_partial_update_safe(self) -> None:
        version = self.store.save_settings({"format": "wav"})
        self.assertEqual(version, 1)
        with self.assertRaises(StateConflictError):
            self.store.save_settings({"format": "flac"}, expected_version=0)
        version = self.store.save_settings({"format": "flac"}, expected_version=1)
        self.assertEqual(self.store.load_settings(), ({"format": "flac"}, version))

    def test_running_and_queued_tasks_recover_as_interrupted(self) -> None:
        self.store.save_task({"id": "run", "status": "running", "stage": "Separating", "created_at": 1}, force=True)
        self.store.save_task({"id": "ready", "status": "ready", "stage": "Ready", "created_at": 2}, force=True)
        recovered = {item["id"]: item for item in self.store.recover_tasks()}
        self.assertEqual(recovered["run"]["status"], "interrupted")
        self.assertTrue(recovered["run"]["retryable"])
        self.assertEqual(recovered["ready"]["status"], "ready")

    def test_progress_is_throttled_except_stage_transitions(self) -> None:
        task = {"id": "one", "status": "running", "stage": "Decode", "created_at": 1}
        self.assertTrue(self.store.save_task(task))
        self.assertFalse(self.store.save_task({**task, "pct": 2}))
        self.assertTrue(self.store.save_task({**task, "stage": "Separate", "pct": 3}))
        self.clock.value += 1.1
        self.assertTrue(self.store.save_task({**task, "stage": "Separate", "pct": 4}))

    def test_task_and_history_pages_are_cursor_paginated(self) -> None:
        for index in range(4):
            self.clock.value += 1
            self.store.save_task(
                {"id": f"task-{index}", "status": "ready", "stage": "Ready", "created_at": index},
                force=True,
            )
            self.store.save_history({"id": f"history-{index}", "finished_at": self.clock.value})
        first = self.store.list_tasks(limit=2)
        second = self.store.list_tasks(limit=2, cursor=first.next_cursor)
        self.assertEqual(len(first.items), 2)
        self.assertEqual(len(second.items), 2)
        self.assertFalse({item["id"] for item in first.items} & {item["id"] for item in second.items})
        self.assertEqual(len(self.store.list_history(limit=2).items), 2)

    def test_legacy_import_is_once_only_and_preserves_backups(self) -> None:
        settings = self.root / "settings.json"
        history = self.root / "previous_files.json"
        eta = self.root / "eta_history.json"
        settings.write_text(json.dumps({"output_format": "wav"}), encoding="utf-8")
        history.write_text(json.dumps([{"id": "old", "finished_at": 2}]), encoding="utf-8")
        eta.write_text(json.dumps({"version": 1}), encoding="utf-8")
        first = self.store.import_legacy_once(settings_path=settings, history_path=history, eta_path=eta)
        second = self.store.import_legacy_once(settings_path=settings, history_path=history, eta_path=eta)
        self.assertEqual(first, {"settings": 1, "history": 1, "eta": 1})
        self.assertEqual(second, {"settings": 0, "history": 0, "eta": 0})
        self.assertTrue(list(self.root.glob("settings.json.v0.4.2.bak.*")))


if __name__ == "__main__":
    unittest.main()
