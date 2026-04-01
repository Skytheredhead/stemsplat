from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
import zipfile

os.environ.setdefault("STEMSPLAT_DISABLE_BACKGROUND_THREADS", "1")

import main


class RuntimeEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        with main.tasks_lock:
            main.tasks.clear()
        while True:
            try:
                main.task_queue.get_nowait()
            except queue.Empty:
                break
            else:
                main.task_queue.task_done()
        main._resume_queue_processing()

    def tearDown(self) -> None:
        with main.tasks_lock:
            main.tasks.clear()
        while True:
            try:
                main.task_queue.get_nowait()
            except queue.Empty:
                break
            else:
                main.task_queue.task_done()
        main._resume_queue_processing()

    def _build_task(self, task_id: str, source_path: Path) -> dict[str, object]:
        return main._build_task_payload(
            task_id=task_id,
            original_name=source_path.name,
            source_path=source_path,
            source_dir=str(source_path.parent),
            mode="vocals",
            output_format="wav",
            video_handling="audio_only",
            auto_start=False,
        )

    def test_load_runtime_stats_migrates_legacy_eta_history_and_caps_entries(self) -> None:
        entries = [
            {"audio_seconds": float(index + 1), "elapsed_seconds": float(index + 2)}
            for index in range(35)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            stats_path = Path(tmpdir) / "eta_history.json"
            stats_path.write_text(json.dumps({"vocals": entries, "invalid": "skip"}), encoding="utf-8")
            with mock.patch.object(main, "ETA_HISTORY_PATH", stats_path):
                stats = main._load_runtime_stats()

        self.assertEqual(stats["version"], main.RUNTIME_STATS_VERSION)
        migrated = stats["stage_samples"]["model:vocals"]
        self.assertEqual(len(migrated), main.RUNTIME_STATS_STAGE_LIMIT)
        self.assertEqual(migrated[0]["audio_seconds"], 6.0)
        self.assertEqual(migrated[-1]["elapsed_seconds"], 36.0)

    def test_predict_model_runtime_prefers_nearby_samples(self) -> None:
        stats = {
            "version": main.RUNTIME_STATS_VERSION,
            "stage_samples": {
                "model:vocals": [
                    {"audio_seconds": 60.0, "elapsed_seconds": 62.0, "recorded_at": 1.0},
                    {"audio_seconds": 600.0, "elapsed_seconds": 610.0, "recorded_at": 2.0},
                ]
            },
            "task_samples": {},
        }
        with mock.patch.object(main, "runtime_stats", stats):
            predicted = main._predict_model_runtime_seconds("vocals", 70.0)

        self.assertIsNotNone(predicted)
        assert predicted is not None
        self.assertGreater(predicted, 50.0)
        self.assertLess(predicted, 160.0)

    def test_model_stage_prediction_is_conservative_above_median(self) -> None:
        stats = {
            "version": main.RUNTIME_STATS_VERSION,
            "stage_samples": {
                "model:vocals": [
                    {"audio_seconds": 180.0, "elapsed_seconds": 90.0, "recorded_at": 1.0},
                    {"audio_seconds": 180.0, "elapsed_seconds": 120.0, "recorded_at": 2.0},
                    {"audio_seconds": 180.0, "elapsed_seconds": 210.0, "recorded_at": 3.0},
                ]
            },
            "task_samples": {},
        }
        with mock.patch.object(main, "runtime_stats", stats):
            prediction, _count, basis = main._predict_model_stage_runtime("vocals", 180.0)

        self.assertEqual(basis, "history")
        self.assertGreater(prediction, 120.0)
        self.assertLess(prediction, 210.0)

    def test_export_prediction_varies_by_output_count(self) -> None:
        stats = {
            "version": main.RUNTIME_STATS_VERSION,
            "stage_samples": {
                "export:wav:1": [
                    {"audio_seconds": 120.0, "elapsed_seconds": 6.0, "recorded_at": 1.0},
                ],
                "export:wav:3": [
                    {"audio_seconds": 120.0, "elapsed_seconds": 18.0, "recorded_at": 2.0},
                ],
            },
            "task_samples": {},
        }
        with mock.patch.object(main, "runtime_stats", stats):
            one_output, _, _ = main._predict_export_runtime_seconds("wav", 1, 120.0)
            three_outputs, _, _ = main._predict_export_runtime_seconds("wav", 3, 120.0)

        self.assertGreater(three_outputs, one_output)

    def test_update_task_runtime_view_uses_runtime_plan_for_percent_and_eta(self) -> None:
        stats = main._blank_runtime_stats()
        with mock.patch.object(main, "runtime_stats", stats):
            task = {
                "mode": "both_separate",
                "output_format": "wav",
                "audio_seconds": 180.0,
                "status": "running",
                "stage": "Running vocals model",
                "pct": 0,
                "eta_seconds": None,
                "eta_state": None,
                "eta_finish_at": None,
                "eta_stage": None,
            }
            main._refresh_runtime_plan(task, audio_seconds=180.0)
            plan = task["runtime_plan"]["stages"]
            lookup = {stage["stage_key"]: stage for stage in plan}
            now = 500.0
            lookup["load_models"]["started_at"] = 10.0
            lookup["load_models"]["completed_at"] = 12.5
            lookup["load_models"]["predicted_seconds"] = 2.5
            lookup["prepare_audio"]["started_at"] = 12.5
            lookup["prepare_audio"]["completed_at"] = 18.0
            lookup["prepare_audio"]["predicted_seconds"] = 5.5
            lookup["vocals"]["started_at"] = now - 20.0
            lookup["vocals"]["live_fraction"] = 0.5

            main._update_task_runtime_view(task, now=now)

        predicted_total = float(task["predicted_total_seconds"])
        self.assertGreater(predicted_total, 0.0)
        self.assertIsInstance(task["eta_seconds"], int)
        self.assertGreater(task["eta_seconds"], 0)
        self.assertLess(task["pct"], 100)
        self.assertLess(task["pct"], 90)

    def test_live_progress_recalibrates_current_stage(self) -> None:
        task = {
            "status": "running",
            "stage": "Running bs-roformer 6s model",
        }
        stage = {
            "stage_key": "bs_roformer_6s",
            "predicted_seconds": 200.0,
            "started_at": 100.0,
            "completed_at": None,
            "live_fraction": 0.2,
            "supports_live_fraction": True,
            "prediction_basis": "fallback",
            "prediction_samples": 0,
        }

        predicted_seconds, fraction, eta_state = main._effective_runtime_stage_progress(task, stage, 106.0)

        self.assertLess(predicted_seconds, 200.0)
        self.assertAlmostEqual(fraction, 0.2, places=3)
        self.assertEqual(eta_state, "calibrating")

    def test_finalize_written_outputs_zips_multi_stem_exports_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            vocals = out_dir / "song - vocals.wav"
            drums = out_dir / "song - drums.wav"
            vocals.write_bytes(b"vocals")
            drums.write_bytes(b"drums")
            task = {
                "id": "task-1",
                "original_name": "song.wav",
                "delivery": "folder",
                "multi_stem_export_snapshot": "zip",
            }

            with mock.patch.dict(main.tasks, {"task-1": task}, clear=True):
                finalized = main._finalize_written_outputs("task-1", out_dir, [vocals, drums])

            self.assertEqual(len(finalized), 1)
            archive_path = finalized[0]
            self.assertEqual(archive_path.suffix, ".zip")
            self.assertTrue(archive_path.exists())
            self.assertFalse(vocals.exists())
            self.assertFalse(drums.exists())
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(sorted(archive.namelist()), ["song - drums.wav", "song - vocals.wav"])

    def test_request_task_stop_pauses_queue_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "song.wav"
            source_path.write_bytes(b"data")
            task = self._build_task("task-1", source_path)
            task["status"] = "running"
            task["stage"] = "Running vocals model"
            with main.tasks_lock:
                main.tasks["task-1"] = task

            main._request_task_stop("task-1")

        self.assertTrue(main._queue_processing_paused())
        self.assertTrue(task["stop_event"].is_set())
        self.assertEqual(task["stage"], "Stopping")

    def test_request_task_stop_hard_terminates_running_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "song.wav"
            source_path.write_bytes(b"data")
            task = self._build_task("task-1", source_path)
            task["status"] = "running"
            task["stage"] = "Running vocals model"
            with main.tasks_lock:
                main.tasks["task-1"] = task

            with mock.patch.object(main, "_terminate_task_runtime") as terminate_runtime:
                main._request_task_stop("task-1")

        terminate_runtime.assert_called_once_with("task-1")
        self.assertTrue(task["stop_event"].is_set())
        self.assertEqual(task["stage"], "Stopping")

    def test_progress_update_stops_single_chunk_model_when_cancel_requested_mid_chunk(self) -> None:
        class StopDuringForwardModel(main.torch.nn.Module):
            def __init__(self, stop_event: threading.Event) -> None:
                super().__init__()
                self._stop_event = stop_event
                self._anchor = main.torch.nn.Parameter(main.torch.zeros(1))

            def forward(self, batch: main.torch.Tensor) -> main.torch.Tensor:
                self._stop_event.set()
                return batch.unsqueeze(1)

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "song.wav"
            source_path.write_bytes(b"data")
            task = self._build_task("task-1", source_path)
            task["status"] = "running"
            task["stage"] = "Running vocals model"
            with main.tasks_lock:
                main.tasks["task-1"] = task

            waveform = main.torch.zeros((2, 1024), dtype=main.torch.float32)
            model = StopDuringForwardModel(task["stop_event"])

            with self.assertRaises(main.TaskStopped):
                main._run_model_chunks(
                    model,
                    waveform,
                    segment=4096,
                    overlap=1,
                    progress_cb=lambda frac: main._set_task_progress(
                        "task-1",
                        "Running vocals model",
                        main._map_fraction(main.MODEL_PROGRESS_START_PCT, main.SINGLE_MODEL_PROGRESS_END_PCT, frac),
                    ),
                    stop_check=lambda: main._stop_check("task-1"),
                )

    def test_worker_waits_for_resume_and_retry_can_take_front_of_queue(self) -> None:
        main._pause_queue_processing()
        main._queue_task("queued-later")

        result: list[str] = []

        def _read_next() -> None:
            result.append(main._next_task_id_for_worker())

        worker = threading.Thread(target=_read_next, daemon=True)
        worker.start()
        time.sleep(0.15)
        self.assertEqual(result, [])

        main._queue_task("retry-front", front=True)
        main._resume_queue_processing()

        worker.join(timeout=1.0)
        self.assertEqual(result, ["retry-front"])

        main.task_queue.task_done()
        self.assertEqual(main.task_queue.get_nowait(), "queued-later")
        main.task_queue.task_done()

    def test_restart_task_rejects_retry_while_original_is_still_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "song.wav"
            source_path.write_bytes(b"data")
            task = self._build_task("task-1", source_path)
            task["status"] = "running"
            task["stage"] = "Running vocals model"
            with main.tasks_lock:
                main.tasks["task-1"] = task

            with self.assertRaises(main.AppError) as ctx:
                main._restart_task_payload("task-1")

        self.assertEqual(ctx.exception.message, "Task is still processing.")


if __name__ == "__main__":
    unittest.main()
