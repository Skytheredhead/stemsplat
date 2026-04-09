from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

os.environ.setdefault("STEMSPLAT_DISABLE_BACKGROUND_THREADS", "1")

import main


class EditorWaveformTests(unittest.TestCase):
    def test_editor_waveform_payload_from_mono_tracks_extrema(self) -> None:
        mono = torch.tensor([-1.0, -0.5, 0.25, 0.75, -0.2, 0.1, 0.9, -0.8], dtype=torch.float32)

        payload = main._editor_waveform_payload_from_mono(mono, points=640)

        self.assertEqual(payload["point_count"], 8)
        self.assertEqual(payload["mins"][:4], [-1.0, -0.5, 0.25, 0.75])
        self.assertEqual(payload["maxs"][:4], [-1.0, -0.5, 0.25, 0.75])
        self.assertEqual(payload["points"][:4], [1.0, 0.5, 0.25, 0.75])

    def test_cached_or_resampled_editor_source_waveform_payload_reuses_higher_resolution_cache(self) -> None:
        mono = torch.linspace(-1.0, 1.0, 2048, dtype=torch.float32)
        high_res_payload = main._editor_waveform_payload_from_mono(mono, points=2048)

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(main, "WORK_DIR", Path(tmpdir)):
                task_id = "wave-cache-test"
                high_res_path = main._editor_source_waveform_cache_path(task_id, 2048)
                main._write_json_atomic(high_res_path, high_res_payload)

                resampled = main._cached_or_resampled_editor_source_waveform_payload(task_id, 1024)

                self.assertIsNotNone(resampled)
                assert resampled is not None
                self.assertEqual(resampled["point_count"], 1024)
                self.assertTrue(main._editor_source_waveform_cache_path(task_id, 1024).exists())


if __name__ == "__main__":
    unittest.main()
