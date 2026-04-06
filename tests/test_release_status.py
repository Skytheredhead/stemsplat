from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

os.environ.setdefault("STEMSPLAT_DISABLE_BACKGROUND_THREADS", "1")

import main


class ReleaseStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.tempdir.name) / "settings.json"
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(main, "SETTINGS_PATH", self.settings_path))
        self.stack.enter_context(mock.patch.object(main, "_require_local_request", lambda request, message=None: None))
        self.client = TestClient(main.app)
        with main.compat_settings_lock:
            self.original_settings = dict(main._compat_settings)
            main._compat_settings.clear()
            main._compat_settings.update(main._normalize_settings_payload({}))
        with main.release_download_lock:
            self.original_release_download_state = dict(main.release_download_state)
            self.original_release_download_thread = main.release_download_thread
            main.release_download_state.clear()
            main.release_download_state.update(
                {
                    "status": "idle",
                    "pct": 0,
                    "step": "",
                    "current_asset": "",
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "download_rate_bytes_per_sec": 0.0,
                    "eta_seconds": None,
                    "started_at": None,
                    "error": "",
                    "path": "",
                    "filename": "",
                    "version": "",
                    "release_name": "",
                }
            )
            main.release_download_thread = None

    def tearDown(self) -> None:
        self.client.close()
        with main.compat_settings_lock:
            main._compat_settings.clear()
            main._compat_settings.update(self.original_settings)
        with main.release_download_lock:
            main.release_download_state.clear()
            main.release_download_state.update(self.original_release_download_state)
            main.release_download_thread = self.original_release_download_thread
        self.stack.close()
        self.tempdir.cleanup()

    def test_release_status_refreshes_cached_release_metadata(self) -> None:
        fake_release = {
            "version": "v0.5.0",
            "name": "v0.5.0 release",
            "url": "https://example.com/releases/v0.5.0",
            "notes": "- better popup behavior\n- release fixes",
        }
        with mock.patch.object(main, "_fetch_latest_release", return_value=fake_release):
            with mock.patch.object(main.time, "time", return_value=1234.0):
                status = main._release_status_payload(refresh=True)

        self.assertTrue(status["update_available"])
        self.assertEqual(status["latest_version"], "v0.5.0")
        self.assertEqual(status["latest_name"], "v0.5.0 release")
        self.assertEqual(status["latest_url"], "https://example.com/releases/v0.5.0")
        self.assertEqual(status["notes"], "- better popup behavior\n- release fixes")
        self.assertEqual(status["last_checked_at"], 1234.0)

        saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["update_latest_version"], "v0.5.0")
        self.assertEqual(saved["update_latest_name"], "v0.5.0 release")
        self.assertEqual(saved["update_latest_url"], "https://example.com/releases/v0.5.0")
        self.assertEqual(saved["update_latest_notes"], "- better popup behavior\n- release fixes")
        self.assertEqual(saved["update_last_checked_at"], 1234.0)

    def test_release_status_endpoint_can_force_refresh_even_with_recent_cache(self) -> None:
        main._set_compat_settings(
            {
                "update_last_checked_at": main.time.time(),
                "update_latest_version": "v0.4.0",
                "update_latest_name": "v0.4.0 release",
                "update_latest_url": "https://example.com/releases/v0.4.0",
            }
        )
        fake_release = {
            "version": "v0.5.0",
            "name": "v0.5.0 release",
            "url": "https://example.com/releases/v0.5.0",
            "notes": "- better popup behavior",
            "assets": [],
        }
        with mock.patch.object(main, "_fetch_latest_release", return_value=fake_release) as fetch_latest:
            response = self.client.get("/api/release_status?refresh=1")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["latest_version"], "v0.5.0")
        self.assertTrue(payload["update_available"])
        fetch_latest.assert_called_once()

    def test_ack_release_status_marks_latest_version_as_notified(self) -> None:
        latest_version = "v0.5.0"
        main._set_compat_settings(
            {
                "update_last_checked_at": main.time.time(),
                "update_latest_version": latest_version,
                "update_latest_name": "v0.5.0 release",
            }
        )

        response = self.client.post("/api/release_status/ack")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["latest_version"], latest_version)
        self.assertEqual(payload["last_notified_version"], latest_version)
        self.assertEqual(main._compat_settings["update_last_notified_version"], latest_version)

    def test_skip_release_status_marks_latest_version_as_skipped(self) -> None:
        latest_version = "v0.5.0"
        main._set_compat_settings(
            {
                "update_last_checked_at": main.time.time(),
                "update_latest_version": latest_version,
                "update_latest_name": "v0.5.0 release",
            }
        )

        response = self.client.post("/api/release_status/skip")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["latest_version"], latest_version)
        self.assertEqual(payload["last_notified_version"], latest_version)
        self.assertEqual(payload["skipped_version"], latest_version)
        self.assertEqual(main._compat_settings["update_skipped_version"], latest_version)

    def test_release_download_saves_best_asset_to_downloads_and_marks_notified(self) -> None:
        output_root = Path(self.tempdir.name) / "Downloads"
        release = {
            "version": "v0.5.0",
            "name": "v0.5.0 release",
            "url": "https://example.com/releases/v0.5.0",
            "notes": "",
            "assets": [
                {"name": "Stemsplat-v0.5.0.dmg", "url": "https://example.com/Stemsplat-v0.5.0.dmg", "size": 90},
                {"name": "Stemsplat-v0.5.0.zip", "url": "https://example.com/Stemsplat-v0.5.0.zip", "size": 100},
            ],
        }
        calls: list[tuple[str, Path, str]] = []

        class _ImmediateThread:
            def __init__(self, *, target, args=(), daemon=None, **_kwargs) -> None:
                self._target = target
                self._args = args
                self._alive = False

            def start(self) -> None:
                self._alive = True
                try:
                    self._target(*self._args)
                finally:
                    self._alive = False

            def is_alive(self) -> bool:
                return self._alive

        def _fake_download(url: str, dest: Path, *, user_agent: str, progress_cb=None) -> Path:
            calls.append((url, dest, user_agent))
            dest.parent.mkdir(parents=True, exist_ok=True)
            if progress_cb is not None:
                progress_cb(
                    {
                        "pct": 100,
                        "downloaded_bytes": len(b"zip-bytes"),
                        "total_bytes": len(b"zip-bytes"),
                    }
                )
            dest.write_bytes(b"zip-bytes")
            return dest

        with mock.patch.object(main, "_fetch_latest_release", return_value=release):
            with mock.patch.object(main, "OUTPUT_ROOT", output_root):
                with mock.patch.object(main, "download_url_to_path", side_effect=_fake_download):
                    with mock.patch.object(main.threading, "Thread", _ImmediateThread):
                        response = self.client.post("/api/release_download")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "done")
        self.assertEqual(payload["version"], "v0.5.0")
        self.assertEqual(payload["filename"], "Stemsplat-v0.5.0.zip")
        self.assertEqual(Path(payload["path"]), output_root / "Stemsplat-v0.5.0.zip")
        self.assertEqual(main._compat_settings["update_last_notified_version"], "v0.5.0")
        self.assertEqual(main._compat_settings["update_latest_version"], "v0.5.0")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "https://example.com/Stemsplat-v0.5.0.zip")
        self.assertEqual(calls[0][1], output_root / "Stemsplat-v0.5.0.zip")
        self.assertIn("stemsplat/", calls[0][2].lower())

    def test_release_download_status_reports_current_state(self) -> None:
        main._set_release_download_state(status="downloading", pct=42, current_asset="Stemsplat.zip")

        response = self.client.get("/api/release_download_status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "downloading")
        self.assertEqual(payload["pct"], 42)
        self.assertEqual(payload["current_asset"], "Stemsplat.zip")


if __name__ == "__main__":
    unittest.main()
