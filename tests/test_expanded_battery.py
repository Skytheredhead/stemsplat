from __future__ import annotations

import asyncio
import contextlib
import errno
import math
import os
import queue
import shutil
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import URLError

import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

os.environ.setdefault("STEMSPLAT_DISABLE_BACKGROUND_THREADS", "1")

import downloader
import main


def _drain_task_queue() -> None:
    while True:
        try:
            main.task_queue.get_nowait()
        except queue.Empty:
            break
        else:
            main.task_queue.task_done()


def _wait_for(predicate, *, timeout: float = 8.0, interval: float = 0.02, message: str = "condition not met") -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(message)


def _tone(seconds: float, *, sample_rate: int = 44_100, channels: int = 2, frequency: float = 220.0) -> np.ndarray:
    total_frames = max(1, int(round(seconds * sample_rate)))
    timeline = np.linspace(0.0, seconds, total_frames, endpoint=False, dtype=np.float32)
    waveform = 0.35 * np.sin(2.0 * np.pi * frequency * timeline, dtype=np.float32)
    if channels == 1:
        return waveform[:, None]
    parts = []
    for channel in range(channels):
        phase = channel * 0.15
        parts.append((0.35 * np.sin((2.0 * np.pi * frequency * timeline) + phase)).astype(np.float32))
    return np.stack(parts, axis=1)


def _write_audio_fixture(path: Path, *, seconds: float = 1.0, sample_rate: int = 44_100, channels: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, _tone(seconds, sample_rate=sample_rate, channels=channels), sample_rate)
    return path


def _transcode(src: Path, dest: Path, *args: str) -> Path:
    ffmpeg = main._ensure_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), *args, str(dest)]
    main._run_interruptible_subprocess(cmd, stop_check=lambda: None)
    return dest


def _media_duration_seconds(path: Path) -> float:
    ffprobe = main._ffprobe_path()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = main.subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float((result.stdout or "0").strip() or 0.0)


class _FakeModelManager:
    missing_models: set[str] = set()

    def __init__(self) -> None:
        self.cache: dict[str, str] = {}

    def get(self, name: str) -> str:
        if name in type(self).missing_models:
            filename = main.MODEL_SPECS[name].filename if name in main.MODEL_SPECS else name
            raise main.AppError(main.ErrorCode.MODEL_MISSING, f"Missing model file: {filename}")
        self.cache.setdefault(name, name)
        return self.cache[name]


class _FakeProcessingController:
    def __init__(self) -> None:
        self.progress_steps = 8
        self.step_sleep_seconds = 0.0
        self.observed_inputs: list[tuple[str, tuple[int, ...]]] = []
        self.after_progress = None

    def _output_count(self, model_key: str) -> int:
        if model_key == "deux":
            return 2
        if model_key == "mel_band_karaoke":
            return 2
        labels = main.MODE_OUTPUT_LABELS.get(model_key)
        if labels:
            return len(labels)
        return 1

    def _fake_output(self, model_key: str, waveform: torch.Tensor) -> torch.Tensor:
        base = waveform.detach().clone().float()
        outputs = []
        for index in range(self._output_count(model_key)):
            scale = max(0.2, 1.0 - (index * 0.08))
            outputs.append(torch.clamp(base * scale, -1.0, 1.0))
        return torch.stack(outputs, dim=0)

    def _run(self, model_key: str, waveform: torch.Tensor, progress_cb, stop_check) -> torch.Tensor:
        self.observed_inputs.append((model_key, tuple(int(value) for value in waveform.shape)))
        for step in range(self.progress_steps):
            stop_check()
            frac = (step + 1) / self.progress_steps
            progress_cb(frac)
            if self.after_progress is not None:
                self.after_progress(model_key, frac)
            if self.step_sleep_seconds > 0:
                time.sleep(self.step_sleep_seconds)
        return self._fake_output(model_key, waveform)

    def run_chunks(self, model, waveform, segment, overlap, progress_cb, stop_check) -> torch.Tensor:
        return self._run(str(model), waveform, progress_cb, stop_check)

    def run_for_spec(self, model_key, model, waveform, progress_cb, stop_check) -> torch.Tensor:
        return self._run(str(model_key), waveform, progress_cb, stop_check)


class _InterruptingResponse:
    def __init__(self, first_chunk: bytes, second_error: Exception) -> None:
        self._first_chunk = first_chunk
        self._second_error = second_error
        self._reads = 0
        self.headers = {"Content-Length": str(len(first_chunk) * 2)}

    def __enter__(self) -> "_InterruptingResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads == 1:
            return self._first_chunk
        if self._reads == 2:
            raise self._second_error
        return b""


class ExpandedBatteryBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.app_home = self.root / "app_home"
        self.model_dir = self.app_home / "models"
        self.output_root = self.root / "outputs"
        self.runtime_dir = self.app_home / ".runtime"
        self.upload_dir = self.runtime_dir / "uploads"
        self.work_dir = self.runtime_dir / "work"
        self.artwork_dir = self.runtime_dir / "artwork"
        self.previous_files_dir = self.app_home / "previous_files"
        self.intermediate_cache_dir = self.app_home / "intermediate_cache"
        self.log_dir = self.app_home / "logs"
        self.settings_path = self.app_home / "settings.json"
        self.eta_history_path = self.app_home / "eta_history.json"
        self.previous_files_index_path = self.app_home / "previous_files.json"
        for directory in (
            self.model_dir,
            self.output_root,
            self.runtime_dir,
            self.upload_dir,
            self.work_dir,
            self.artwork_dir,
            self.previous_files_dir,
            self.intermediate_cache_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        for spec in main.MODEL_SPECS.values():
            (self.model_dir / spec.filename).write_bytes(b"fake-model")

        self.controller = _FakeProcessingController()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(mock.patch.object(main, "MODEL_DIR", self.model_dir))
        self.stack.enter_context(mock.patch.object(main, "MODEL_SEARCH_DIRS", [self.model_dir]))
        self.stack.enter_context(mock.patch.object(main, "OUTPUT_ROOT", self.output_root))
        self.stack.enter_context(mock.patch.object(main, "RUNTIME_DIR", self.runtime_dir))
        self.stack.enter_context(mock.patch.object(main, "UPLOAD_DIR", self.upload_dir))
        self.stack.enter_context(mock.patch.object(main, "WORK_DIR", self.work_dir))
        self.stack.enter_context(mock.patch.object(main, "ARTWORK_DIR", self.artwork_dir))
        self.stack.enter_context(mock.patch.object(main, "PREVIOUS_FILES_DIR", self.previous_files_dir))
        self.stack.enter_context(mock.patch.object(main, "INTERMEDIATE_CACHE_DIR", self.intermediate_cache_dir))
        self.stack.enter_context(mock.patch.object(main, "LOG_DIR", self.log_dir))
        self.stack.enter_context(mock.patch.object(main, "SETTINGS_PATH", self.settings_path))
        self.stack.enter_context(mock.patch.object(main, "ETA_HISTORY_PATH", self.eta_history_path))
        self.stack.enter_context(mock.patch.object(main, "PREVIOUS_FILES_INDEX_PATH", self.previous_files_index_path))
        self.stack.enter_context(mock.patch.object(main, "ModelManager", _FakeModelManager))
        self.stack.enter_context(mock.patch.object(main, "_run_model_chunks", self.controller.run_chunks))
        self.stack.enter_context(mock.patch.object(main, "_run_model_for_spec", self.controller.run_for_spec))
        self.stack.enter_context(mock.patch.object(main, "_safe_mps_empty_cache", lambda: None))
        self.stack.enter_context(mock.patch.object(main, "_is_remote_client", lambda request: False))
        self.stack.enter_context(mock.patch.object(main, "_require_local_request", lambda request, message=None: None))

        _FakeModelManager.missing_models = set()
        with main.tasks_lock:
            main.tasks.clear()
        with main.task_runtime_lock:
            main.task_runtimes.clear()
        _drain_task_queue()
        main._model_manager = None
        main.model_download_thread = None
        main._resume_queue_processing()

        with main.compat_settings_lock:
            main._compat_settings.clear()
            main._compat_settings.update(
                main._normalize_settings_payload(
                    {
                        "output_root": str(self.output_root),
                        "output_format": "wav",
                        "multi_stem_export": "separate",
                        "model_prompt_state": main.MODEL_PROMPT_COMPLETE,
                    }
                )
            )

        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.output_root.chmod(stat.S_IRWXU)
        self.stack.close()
        self.tempdir.cleanup()

    def make_audio(self, name: str, *, seconds: float = 1.0, sample_rate: int = 44_100, channels: int = 2) -> Path:
        return _write_audio_fixture(self.root / "fixtures" / name, seconds=seconds, sample_rate=sample_rate, channels=channels)

    def upload_task(self, path: Path, *, stems: str = "vocals", output_format: str = "wav", multi_stem_export: str = "separate") -> dict[str, object]:
        media_type = "audio/wav"
        if path.suffix.lower() == ".mp3":
            media_type = "audio/mpeg"
        elif path.suffix.lower() == ".flac":
            media_type = "audio/flac"
        with path.open("rb") as handle:
            response = self.client.post(
                "/upload",
                files={"file": (path.name, handle, media_type)},
                data={
                    "stems": stems,
                    "output_format": output_format,
                    "multi_stem_export": multi_stem_export,
                    "video_handling": "audio_only",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        with main.tasks_lock:
            main.tasks[str(payload["task_id"])]["delivery"] = "folder"
        return payload

    def import_task(self, path: Path, *, stems: str = "vocals") -> dict[str, object]:
        response = self.client.post(
            "/api/import_paths",
            json={
                "paths": [str(path)],
                "stems": stems,
                "output_format": "wav",
                "multi_stem_export": "separate",
                "video_handling": "audio_only",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        tasks = response.json()["tasks"]
        self.assertEqual(len(tasks), 1)
        return tasks[0]

    def start_task(self, task_id: str, *, output_root: Path | None = None, output_format: str = "wav", multi_stem_export: str = "separate") -> dict[str, object]:
        body: dict[str, object] = {
            "output_format": output_format,
            "multi_stem_export": multi_stem_export,
            "video_handling": "audio_only",
        }
        if output_root is not None:
            body["output_root"] = str(output_root)
        response = self.client.post(f"/start/{task_id}", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def run_task_async(self, task_id: str) -> threading.Thread:
        thread = threading.Thread(target=main._process_task, args=(task_id,), daemon=True)
        thread.start()
        return thread

    def run_task_to_completion(self, task_id: str, *, timeout: float = 8.0) -> dict[str, object]:
        worker = self.run_task_async(task_id)
        worker.join(timeout=timeout)
        self.assertFalse(worker.is_alive(), f"task {task_id} did not finish in time")
        return main._require_task(task_id)


class ExpandedBatterySmokeTests(ExpandedBatteryBase):
    def test_api_smoke_endpoints_respond(self) -> None:
        self.assertEqual(self.client.get("/").status_code, 200)
        runtime = self.client.get("/api/runtime_status")
        models = self.client.get("/api/models_status")
        downloads = self.client.get("/api/model_download_status")
        self.assertEqual(runtime.status_code, 200, runtime.text)
        self.assertEqual(models.status_code, 200, models.text)
        self.assertEqual(downloads.status_code, 200, downloads.text)
        self.assertIn("ffmpeg_available", runtime.json())
        self.assertIn("models", models.json())
        self.assertIn("status", downloads.json())


class ExpandedBatteryProcessingTests(ExpandedBatteryBase):
    def test_cancel_mid_processing_stops_cleanly_and_cleans_outputs(self) -> None:
        self.controller.progress_steps = 14
        self.controller.step_sleep_seconds = 0.03
        payload = self.upload_task(self.make_audio("cancel_me.wav", seconds=3.0))
        task_id = str(payload["task_id"])
        self.start_task(task_id, output_root=self.output_root)
        worker = self.run_task_async(task_id)
        _wait_for(
            lambda: 10 <= int(main._require_task(task_id).get("pct") or 0) <= 50,
            message="task never reached cancellable progress window",
        )
        stop_response = self.client.post(f"/stop/{task_id}")
        self.assertEqual(stop_response.status_code, 200, stop_response.text)
        worker.join(timeout=8.0)
        self.assertFalse(worker.is_alive(), "cancelled task did not stop")
        task = main._require_task(task_id)
        self.assertEqual(task["status"], "stopped")
        self.assertEqual(task["stage"], "Stopped")
        self.assertFalse(task["outputs"])
        self.assertEqual(list(self.output_root.glob("*")), [])
        self.assertEqual(task["eta_seconds"], None)

    def test_output_overwrite_uses_unique_suffixes(self) -> None:
        source = self.make_audio("collision.wav", seconds=1.5)
        first = self.upload_task(source)
        second = self.upload_task(source)
        first_id = str(first["task_id"])
        second_id = str(second["task_id"])
        self.start_task(first_id, output_root=self.output_root)
        self.start_task(second_id, output_root=self.output_root)
        first_task = self.run_task_to_completion(first_id)
        second_task = self.run_task_to_completion(second_id)
        self.assertEqual(first_task["status"], "done")
        self.assertEqual(second_task["status"], "done")
        first_output = Path(str(first_task["out_dir"])) / str(first_task["outputs"][0])
        second_output = Path(str(second_task["out_dir"])) / str(second_task["outputs"][0])
        self.assertTrue(first_output.exists())
        self.assertTrue(second_output.exists())
        self.assertNotEqual(first_output.name, second_output.name)
        self.assertIn("(2)", second_output.name)

    def test_read_only_output_directory_errors(self) -> None:
        read_only_root = self.root / "read_only_output"
        read_only_root.mkdir(parents=True, exist_ok=True)
        read_only_root.chmod(0o555)
        self.addCleanup(lambda: read_only_root.exists() and read_only_root.chmod(0o755))
        payload = self.upload_task(self.make_audio("read_only.wav"))
        task_id = str(payload["task_id"])
        self.start_task(task_id, output_root=read_only_root)
        task = self.run_task_to_completion(task_id)
        self.assertEqual(task["status"], "error")
        self.assertIn("permission", str(task["error"]).lower())

    def test_deleted_source_after_queueing_errors_cleanly(self) -> None:
        payload = self.upload_task(self.make_audio("queued_delete.wav"))
        task_id = str(payload["task_id"])
        self.start_task(task_id, output_root=self.output_root)
        source_path = Path(str(main._require_task(task_id)["source_path"]))
        source_path.unlink()
        task = self.run_task_to_completion(task_id)
        self.assertEqual(task["status"], "error")
        self.assertIn("uploaded source file is missing", str(task["error"]).lower())

    def test_deleted_output_root_mid_processing_does_not_succeed(self) -> None:
        payload = self.upload_task(self.make_audio("delete_output_root.wav", seconds=2.5))
        task_id = str(payload["task_id"])
        self.controller.progress_steps = 10
        self.controller.step_sleep_seconds = 0.02
        deleted = {"done": False}

        def _delete_output_root(_model_key: str, frac: float) -> None:
            task = main._require_task(task_id)
            out_dir = str(task.get("out_dir") or "")
            if deleted["done"] or not out_dir or frac < 0.3:
                return
            path = Path(out_dir)
            if path.exists():
                shutil.rmtree(path)
                deleted["done"] = True

        self.controller.after_progress = _delete_output_root
        self.start_task(task_id, output_root=self.output_root)
        task = self.run_task_to_completion(task_id)
        self.assertEqual(task["status"], "error")
        self.assertIn("finalize", str(task["error"]).lower())

    def test_disk_full_during_publish_errors_without_hanging(self) -> None:
        payload = self.upload_task(self.make_audio("disk_full.wav"))
        task_id = str(payload["task_id"])
        self.start_task(task_id, output_root=self.output_root)
        with mock.patch.object(
            main,
            "_publish_staged_outputs",
            side_effect=main.AppError(main.ErrorCode.SEPARATION_FAILED, f"Could not finalize export: {errno.ENOSPC}"),
        ):
            task = self.run_task_to_completion(task_id)
        self.assertEqual(task["status"], "error")
        self.assertIn("could not finalize export", str(task["error"]).lower())

    def test_unicode_filename_survives_processing_and_zip(self) -> None:
        source = self.make_audio("Beyonce_日本語🔥_test.wav", seconds=1.2)
        payload = self.upload_task(source, stems="vocals,instrumental", multi_stem_export="zip")
        task_id = str(payload["task_id"])
        self.start_task(task_id, output_root=self.output_root, multi_stem_export="zip")
        task = self.run_task_to_completion(task_id)
        self.assertEqual(task["status"], "done")
        self.assertEqual(len(task["outputs"]), 1)
        archive_path = Path(str(task["out_dir"])) / str(task["outputs"][0])
        self.assertTrue(archive_path.exists())
        self.assertEqual(archive_path.suffix, ".zip")

    def test_drumsep_6s_runs_drums_model_before_drum_split(self) -> None:
        source = self.make_audio("drumsplit6.wav", seconds=1.0)
        payload = self.upload_task(source, stems="drumsep_6s", multi_stem_export="separate")
        task_id = str(payload["task_id"])
        self.start_task(task_id, output_root=self.output_root, multi_stem_export="separate")
        task = self.run_task_to_completion(task_id)

        self.assertEqual(task["status"], "done")
        observed_models = [model_key for model_key, _shape in self.controller.observed_inputs]
        self.assertEqual(observed_models[:2], ["htdemucs_ft_drums", "drumsep_6s"])
        self.assertEqual(sorted(task["outputs"]), ["drumsplit6 - crash.wav", "drumsplit6 - hh.wav", "drumsplit6 - kick.wav", "drumsplit6 - ride.wav", "drumsplit6 - snare.wav", "drumsplit6 - toms.wav"])

    def test_drumsep_4s_runs_drums_model_before_drum_split(self) -> None:
        source = self.make_audio("drumsplit4.wav", seconds=1.0)
        payload = self.upload_task(source, stems="drumsep_4s", multi_stem_export="separate")
        task_id = str(payload["task_id"])
        self.start_task(task_id, output_root=self.output_root, multi_stem_export="separate")
        task = self.run_task_to_completion(task_id)

        self.assertEqual(task["status"], "done")
        observed_models = [model_key for model_key, _shape in self.controller.observed_inputs]
        self.assertEqual(observed_models[:2], ["htdemucs_ft_drums", "drumsep_4s"])
        self.assertEqual(sorted(task["outputs"]), ["drumsplit4 - cymbals.wav", "drumsplit4 - kick.wav", "drumsplit4 - snare.wav", "drumsplit4 - toms.wav"])

    def test_deep_path_import_survives_processing(self) -> None:
        nested_root = self.root / "deep"
        for index in range(12):
            nested_root /= f"layer_{index:02d}_very_long_folder_name"
        source = _write_audio_fixture(nested_root / "deep_source.wav", seconds=1.1)
        payload = self.import_task(source)
        task_id = str(payload["task_id"])
        self.start_task(task_id, output_root=self.output_root)
        task = self.run_task_to_completion(task_id)
        self.assertEqual(task["status"], "done")


class ExpandedBatteryAudioTests(ExpandedBatteryBase):
    def test_tiny_valid_audio_completes(self) -> None:
        payload = self.upload_task(self.make_audio("tiny.wav", seconds=0.6))
        task = self.run_task_to_completion(str(payload["task_id"]))
        self.assertEqual(task["status"], "done")
        output_path = Path(str(task["out_dir"])) / str(task["outputs"][0])
        self.assertTrue(output_path.exists())

    def test_high_sample_rate_audio_resamples_without_duration_drift(self) -> None:
        source = self.make_audio("high_rate.wav", seconds=1.8, sample_rate=96_000)
        payload = self.upload_task(source)
        task = self.run_task_to_completion(str(payload["task_id"]))
        self.assertEqual(task["status"], "done")
        output_path = Path(str(task["out_dir"])) / str(task["outputs"][0])
        info = sf.info(str(output_path))
        self.assertEqual(info.samplerate, 44_100)
        self.assertLess(abs(_media_duration_seconds(source) - _media_duration_seconds(output_path)), 0.12)

    def test_variable_bitrate_mp3_keeps_duration_reasonable(self) -> None:
        wav_path = self.make_audio("vbr_source.wav", seconds=2.2)
        mp3_path = self.root / "fixtures" / "vbr_source.mp3"
        _transcode(wav_path, mp3_path, "-c:a", "libmp3lame", "-q:a", "4")
        payload = self.upload_task(mp3_path)
        task = self.run_task_to_completion(str(payload["task_id"]))
        self.assertEqual(task["status"], "done")
        output_path = Path(str(task["out_dir"])) / str(task["outputs"][0])
        self.assertLess(abs(_media_duration_seconds(mp3_path) - _media_duration_seconds(output_path)), 0.15)

    def test_corrupted_audio_fails_cleanly(self) -> None:
        source = self.make_audio("healthy.wav", seconds=1.0)
        damaged = self.root / "fixtures" / "damaged.wav"
        raw = source.read_bytes()
        damaged.write_bytes(raw[:32])
        payload = self.upload_task(damaged)
        task = self.run_task_to_completion(str(payload["task_id"]))
        self.assertEqual(task["status"], "error")
        self.assertTrue(
            "could not decode audio" in str(task["error"]).lower()
            or "could not read wav" in str(task["error"]).lower()
        )

    def test_reverse_extension_mismatch_is_content_accepted(self) -> None:
        source = self.make_audio("mismatch.wav", seconds=1.0)
        disguised = self.root / "fixtures" / "mismatch.txt"
        disguised.write_bytes(source.read_bytes())
        with disguised.open("rb") as handle:
            response = self.client.post(
                "/upload",
                files={"file": (disguised.name, handle, "audio/wav")},
                data={"stems": "vocals", "output_format": "wav", "multi_stem_export": "separate"},
            )
        self.assertEqual(response.status_code, 200, response.text)

    def test_mono_input_round_trips_as_mono_output(self) -> None:
        mono = self.make_audio("mono.wav", seconds=1.0, channels=1)
        payload = self.upload_task(mono)
        task = self.run_task_to_completion(str(payload["task_id"]))
        self.assertEqual(task["status"], "done")
        output_path = Path(str(task["out_dir"])) / str(task["outputs"][0])
        info = sf.info(str(output_path))
        self.assertEqual(info.channels, 1)

    def test_multichannel_input_is_not_silently_clamped(self) -> None:
        surround = self.make_audio("surround.wav", seconds=1.0, channels=6)
        source_info = main._probe_source(surround)
        self.assertGreaterEqual(source_info.channels, 6)


class ExpandedBatteryModelFaultTests(ExpandedBatteryBase):
    def test_missing_model_surfaces_actionable_error(self) -> None:
        _FakeModelManager.missing_models = {"vocals"}
        payload = self.upload_task(self.make_audio("missing_model.wav"))
        task = self.run_task_to_completion(str(payload["task_id"]))
        self.assertEqual(task["status"], "error")
        self.assertIn("missing model file", str(task["error"]).lower())

    def test_corrupted_model_file_message_is_actionable(self) -> None:
        broken_model = self.root / "broken.ckpt"
        broken_model.write_bytes(b"not-a-real-torch-checkpoint")
        with self.assertRaises(main.AppError) as ctx:
            main._torch_load_compat(broken_model)
        self.assertEqual(ctx.exception.code, main.ErrorCode.SEPARATION_FAILED)
        self.assertIn("remove it and download it again", ctx.exception.message.lower())

    def test_interrupted_model_download_cleans_partial_file(self) -> None:
        target_root = self.root / "download_target"
        payload = b"x" * 1024
        with mock.patch.object(
            downloader,
            "FILES",
            [{"url": "https://example.test/vocals.ckpt", "subdir": "models", "filename": "vocals.ckpt", "tag": "vocals"}],
        ), mock.patch.object(
            downloader,
            "get_remote_file_metadata",
            return_value=downloader.RemoteFileMetadata(size=len(payload) * 2, sha256=None),
        ), mock.patch.object(
            downloader,
            "urlopen",
            return_value=_InterruptingResponse(payload, URLError("connection reset")),
        ):
            with self.assertRaises(downloader.ModelDownloadError):
                downloader.download_to(target_root)
        self.assertFalse((target_root / "models" / "vocals.ckpt.part").exists())
        self.assertFalse((target_root / "models" / "vocals.ckpt").exists())

    def test_model_download_state_not_marked_done_after_interruption(self) -> None:
        for spec in main.MODEL_SPECS.values():
            target = self.model_dir / spec.filename
            if target.exists():
                target.unlink()
        with mock.patch.object(
            main,
            "download_to",
            side_effect=downloader.ModelDownloadError(
                "network-reset",
                "network dropped mid-download",
                retryable=False,
                tag="vocals",
                filename="vocals.ckpt",
            ),
        ):
            main._run_model_download(["vocals"])
        status = main._public_model_download_status()
        self.assertEqual(status["status"], "error")
        self.assertNotEqual(status["status"], "done")
        self.assertIn("network dropped", str(status["error"]).lower())


class ExpandedBatteryRecoveryTests(ExpandedBatteryBase):
    def test_rehydrate_after_restart_drops_stale_running_tasks(self) -> None:
        payload = self.upload_task(self.make_audio("restart.wav", seconds=1.0))
        task_id = str(payload["task_id"])
        with main.tasks_lock:
            task = main.tasks[task_id]
            task["status"] = "running"
            task["stage"] = "Running vocals model"
            task["pct"] = 42
            task["version"] += 1
        with main.tasks_lock:
            main.tasks.clear()
        response = self.client.post("/rehydrate_tasks", json={"tasks": [{"id": task_id}]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["tasks"], [])

    def test_startup_cleanup_removes_stale_runtime_entries(self) -> None:
        stale = self.work_dir / "stale_runtime"
        stale.mkdir(parents=True, exist_ok=True)
        old_time = time.time() - (main.RUNTIME_CLEANUP_MAX_AGE_SEC + 60)
        os.utime(stale, (old_time, old_time))
        asyncio.run(main._startup_cleanup())
        self.assertFalse(stale.exists())


if __name__ == "__main__":
    unittest.main()
