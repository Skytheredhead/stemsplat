#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stemsplat.atomic import atomic_write_bytes  # noqa: E402
from stemsplat.version import __version__  # noqa: E402


def sign_update(asset: Path, download_url: str, private_key_path: Path, public_key_path: Path, output_dir: Path) -> tuple[Path, Path]:
    if not asset.is_file():
        raise ValueError(f"release asset is missing: {asset}")
    if not download_url.startswith("https://github.com/") or "/releases/download/" not in download_url:
        raise ValueError("update URL must be an HTTPS GitHub release asset")
    password_text = os.environ.get("STEMSPLAT_UPDATE_KEY_PASSWORD")
    password = password_text.encode("utf-8") if password_text else None
    private_key = serialization.load_pem_private_key(private_key_path.read_bytes(), password=password)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("update private key must be Ed25519")
    public_key = serialization.load_pem_public_key(public_key_path.read_bytes())
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("update public key must be Ed25519")
    derived = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    expected = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if derived != expected:
        raise ValueError("update private key does not match the packaged public key")

    hasher = hashlib.sha256()
    with asset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    payload = {
        "schema_version": 1,
        "version": f"v{__version__}",
        "asset_name": asset.name,
        "download_url": download_url,
        "byte_length": asset.stat().st_size,
        "sha256": digest,
        "issued_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    manifest_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature_bytes = base64.b64encode(private_key.sign(manifest_bytes)) + b"\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "stemsplat-update.json"
    signature_path = output_dir / "stemsplat-update.json.sig"
    atomic_write_bytes(manifest_path, manifest_bytes)
    atomic_write_bytes(signature_path, signature_bytes)
    return manifest_path, signature_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create signed Stemsplat update metadata")
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--download-url", required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, default=Path("updates/public-key.pem"))
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    manifest, signature = sign_update(
        args.asset.resolve(),
        args.download_url,
        args.private_key.resolve(),
        args.public_key.resolve(),
        args.output_dir.resolve(),
    )
    print(f"created {manifest} and {signature}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
