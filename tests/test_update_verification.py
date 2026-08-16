from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from stemsplat.updates import UpdateVerificationError, parse_and_verify_manifest, verify_release_asset
from tools.sign_update import sign_update


class UpdateVerificationTests(unittest.TestCase):
    def test_signed_manifest_and_asset_verify(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            asset = root / "stemsplat-v0.4.3.zip"
            asset.write_bytes(b"signed-release")
            import hashlib

            payload = {
                "schema_version": 1,
                "version": "v0.4.3",
                "asset_name": asset.name,
                "download_url": "https://github.com/Skytheredhead/stemsplat/releases/download/v0.4.3/stemsplat-v0.4.3.zip",
                "byte_length": asset.stat().st_size,
                "sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "issued_at": "2026-08-13T00:00:00Z",
            }
            manifest_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            private_key = Ed25519PrivateKey.generate()
            public_path = root / "public-key.pem"
            public_path.write_bytes(
                private_key.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            manifest = parse_and_verify_manifest(manifest_bytes, private_key.sign(manifest_bytes), public_path)
            verify_release_asset(asset, manifest)

    def test_tampered_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            private_key = Ed25519PrivateKey.generate()
            public_path = root / "public.pem"
            public_path.write_bytes(
                private_key.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            original = b'{"schema_version":1}'
            with self.assertRaises(UpdateVerificationError):
                parse_and_verify_manifest(original + b" ", private_key.sign(original), public_path)

    def test_release_signer_uses_single_app_version_and_matching_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            asset = root / "Stemsplat-0.4.3-arm64.zip"
            asset.write_bytes(b"release-asset")
            private_key = Ed25519PrivateKey.generate()
            private_path = root / "private.pem"
            public_path = root / "public.pem"
            private_path.write_bytes(
                private_key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
            public_path.write_bytes(
                private_key.public_key().public_bytes(
                    serialization.Encoding.PEM,
                    serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            manifest_path, signature_path = sign_update(
                asset,
                "https://github.com/Skytheredhead/stemsplat/releases/download/v0.4.3/Stemsplat-0.4.3-arm64.zip",
                private_path,
                public_path,
                root / "out",
            )
            manifest = parse_and_verify_manifest(manifest_path.read_bytes(), signature_path.read_bytes(), public_path)
            self.assertEqual(manifest.version, "v0.4.3")
            verify_release_asset(asset, manifest)


if __name__ == "__main__":
    unittest.main()
