from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class UpdateVerificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class UpdateManifest:
    version: str
    asset_name: str
    download_url: str
    byte_length: int
    sha256: str
    issued_at: str


def parse_and_verify_manifest(
    manifest_bytes: bytes,
    signature_bytes: bytes,
    public_key_path: Path,
) -> UpdateManifest:
    if not public_key_path.is_file():
        raise UpdateVerificationError("the release signing public key is not configured")
    if len(manifest_bytes) > 128 * 1024:
        raise UpdateVerificationError("update metadata is too large")
    try:
        public_key = serialization.load_pem_public_key(public_key_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise UpdateVerificationError("the release signing public key is invalid") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise UpdateVerificationError("the release signing key must be Ed25519")
    signature = _decode_signature(signature_bytes)
    try:
        public_key.verify(signature, manifest_bytes)
    except InvalidSignature as exc:
        raise UpdateVerificationError("update metadata signature verification failed") from exc
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UpdateVerificationError("signed update metadata is invalid JSON") from exc
    if not isinstance(payload, dict) or int(payload.get("schema_version") or 0) != 1:
        raise UpdateVerificationError("signed update metadata has an unsupported schema")
    version = str(payload.get("version") or "").strip()
    asset_name = Path(str(payload.get("asset_name") or "")).name
    download_url = str(payload.get("download_url") or "").strip()
    sha256 = str(payload.get("sha256") or "").strip().lower()
    issued_at = str(payload.get("issued_at") or "").strip()
    try:
        byte_length = int(payload.get("byte_length") or 0)
    except (TypeError, ValueError) as exc:
        raise UpdateVerificationError("signed update byte length is invalid") from exc
    if not version or not asset_name or asset_name != str(payload.get("asset_name") or ""):
        raise UpdateVerificationError("signed update identity is incomplete")
    if not download_url.startswith("https://github.com/") or "/releases/download/" not in download_url:
        raise UpdateVerificationError("signed update URL is not an HTTPS GitHub release asset")
    if byte_length <= 0:
        raise UpdateVerificationError("signed update byte length must be positive")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise UpdateVerificationError("signed update SHA-256 is invalid")
    if not issued_at:
        raise UpdateVerificationError("signed update issue time is missing")
    return UpdateManifest(version, asset_name, download_url, byte_length, sha256, issued_at)


def verify_release_asset(path: Path, manifest: UpdateManifest) -> None:
    if not path.is_file():
        raise UpdateVerificationError("downloaded release asset is missing")
    if path.stat().st_size != manifest.byte_length:
        raise UpdateVerificationError("downloaded release asset has the wrong byte length")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest.sha256:
        raise UpdateVerificationError("downloaded release asset failed SHA-256 verification")


def _decode_signature(raw: bytes) -> bytes:
    # A binary Ed25519 signature can legitimately begin or end with bytes that
    # bytes.strip() treats as whitespace. Preserve exact 64-byte signatures.
    if len(raw) == 64:
        return raw
    stripped = raw.strip()
    try:
        decoded = base64.b64decode(stripped, validate=True)
    except ValueError as exc:
        raise UpdateVerificationError("update metadata signature is invalid") from exc
    if len(decoded) != 64:
        raise UpdateVerificationError("update metadata signature has the wrong length")
    return decoded
