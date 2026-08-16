from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import quality


class QualityHarnessTests(unittest.TestCase):
    checkpoint_hash = "a" * 64
    config_hash = "b" * 64
    device_id = "Apple M2 Pro:00000000"

    def test_identical_audio_passes_reference_safe_compare(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            audio = np.linspace(-0.4, 0.4, 44_100, dtype=np.float32)[:, None]
            stereo = np.repeat(audio, 2, axis=1)
            sf.write(reference / "vocals.wav", stereo, 44_100, subtype="FLOAT")
            sf.write(candidate / "vocals.wav", stereo, 44_100, subtype="FLOAT")
            baseline_path = root / "baseline.json"
            baseline_payload = {
                "audio": [quality.asdict(quality.measure_audio(reference / "vocals.wav", reference))],
                "checkpoint_hashes": [self.checkpoint_hash],
                "config_hashes": [self.config_hash],
                "device_id": self.device_id,
            }
            baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
            output = root / "compare.json"
            args = quality.argparse.Namespace(
                baseline=str(baseline_path),
                reference_dir=str(reference),
                candidate_dir=str(candidate),
                output=str(output),
                checkpoint_sha256=[self.checkpoint_hash],
                config_sha256=[self.config_hash],
                device_id=self.device_id,
            )
            self.assertEqual(quality.command_compare(args), 0)
            self.assertTrue(json.loads(output.read_text(encoding="utf-8"))["passed"])

    def test_structural_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            sf.write(reference / "stem.wav", np.ones((100, 2), dtype=np.float32) * 0.1, 44_100)
            sf.write(candidate / "stem.wav", np.ones((99, 2), dtype=np.float32) * 0.1, 44_100)
            baseline = {
                "audio": [quality.asdict(quality.measure_audio(reference / "stem.wav", reference))],
                "checkpoint_hashes": [self.checkpoint_hash],
                "config_hashes": [self.config_hash],
                "device_id": self.device_id,
            }
            baseline_path = root / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            output = root / "compare.json"
            args = quality.argparse.Namespace(
                baseline=str(baseline_path),
                reference_dir=str(reference),
                candidate_dir=str(candidate),
                output=str(output),
                checkpoint_sha256=[self.checkpoint_hash],
                config_sha256=[self.config_hash],
                device_id=self.device_id,
            )
            self.assertEqual(quality.command_compare(args), 2)

    def test_public_metrics_require_every_target_and_enforce_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "public.json"
            payload = {
                "metric": "museval_bsseval_v4",
                "source_tool": "museval",
                "museval_version": "0.4.1",
                "corpora": ["MUSDB18-HQ", "Slakh2100-tiny"],
                "failed_tracks": [],
                "targets": {
                    target: {"baseline_sdr_db": [5.0, 6.0], "candidate_sdr_db": [5.0, 6.0]}
                    for target in quality.REQUIRED_PUBLIC_TARGETS
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            report, failures = quality._validate_public_metrics(path)
            self.assertFalse(failures)
            self.assertTrue(report["passed"])

            payload["targets"]["vocals"]["candidate_sdr_db"] = [4.0, 5.0]
            path.write_text(json.dumps(payload), encoding="utf-8")
            report, failures = quality._validate_public_metrics(path)
            self.assertFalse(report["passed"])
            self.assertIn("public SDR gate failed for vocals", failures)

    def test_benchmark_comparison_requires_wall_rss_and_mps_gates(self) -> None:
        baseline = {
            "command": "benchmark",
            "role": "baseline",
            "mode": "vocals",
            "device_id": self.device_id,
            "summary": {
                kind: {"trials": 3, "median_wall_seconds": 10.0, "median_peak_rss_bytes": 1000, "peak_mps_bytes": 2000}
                for kind in ("cold", "warm")
            },
        }
        candidate = {
            "command": "benchmark",
            "role": "candidate",
            "mode": "vocals",
            "device_id": self.device_id,
            "summary": {
                kind: {"trials": 3, "median_wall_seconds": 10.4, "median_peak_rss_bytes": 1050, "peak_mps_bytes": 2100}
                for kind in ("cold", "warm")
            },
        }
        report, failures = quality._benchmark_regressions([baseline, candidate])
        self.assertFalse(failures)
        self.assertTrue(report["modes"][f"vocals@{self.device_id}"]["trials"]["cold"]["passed"])

        candidate["summary"]["warm"]["median_wall_seconds"] = 11.0
        _, failures = quality._benchmark_regressions([baseline, candidate])
        self.assertIn("vocals warm performance regression gate failed", failures)


if __name__ == "__main__":
    unittest.main()
