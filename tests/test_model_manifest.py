from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from stemsplat.model_manifest import ModelManifest, ModelManifestError, file_sha256

ROOT = Path(__file__).resolve().parents[1]


class ModelManifestTests(unittest.TestCase):
    def test_every_shipped_yaml_config_is_safe_loadable(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        for config_path in sorted(config_dir.glob("*.yaml")):
            with self.subTest(config=config_path.name):
                parsed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                self.assertIsInstance(parsed, dict)

    def test_repository_manifest_is_valid_and_blocks_unreviewed_downloads(self) -> None:
        manifest = ModelManifest(ROOT / "models" / "manifest.json")
        self.assertEqual(len(manifest.by_tag), 13)
        for tag, artifact in manifest.by_tag.items():
            manifest.verify_config(ROOT / "configs", tag)
            self.assertFalse(artifact.auto_download)
            with self.assertRaises(ModelManifestError):
                manifest.descriptor(tag)

    def test_exact_checkpoint_and_config_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root = root / "models"
            config_root = root / "configs"
            model_root.mkdir()
            config_root.mkdir()
            checkpoint = model_root / "model.ckpt"
            config = config_root / "model.yaml"
            checkpoint.write_bytes(b"checkpoint")
            config.write_bytes(b"config")
            revision = "1" * 40
            payload = {
                "schema_version": 1,
                "models": [
                    {
                        "tag": "model",
                        "revision": revision,
                        "url": f"https://example.invalid/repository/resolve/{revision}/model.ckpt",
                        "filename": checkpoint.name,
                        "byte_length": checkpoint.stat().st_size,
                        "sha256": file_sha256(checkpoint),
                        "config_filename": config.name,
                        "config_sha256": file_sha256(config),
                        "loader_type": "roformer-state-dict",
                        "source_project": "https://example.invalid/repository",
                        "spdx_license": "MIT",
                        "attribution": "Test fixture",
                        "auto_download": True,
                        "blocked_reason": None,
                    }
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = ModelManifest(path)
            self.assertEqual(manifest.verify_checkpoint(model_root, "model"), checkpoint)
            self.assertEqual(manifest.verify_config(config_root, "model"), config)
            checkpoint.write_bytes(b"tampered")
            with self.assertRaises(ModelManifestError):
                manifest.verify_checkpoint(model_root, "model")


if __name__ == "__main__":
    unittest.main()
