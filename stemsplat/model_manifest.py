from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_FIELDS = frozenset(
    {
        "tag",
        "revision",
        "url",
        "filename",
        "byte_length",
        "sha256",
        "config_filename",
        "config_sha256",
        "loader_type",
        "source_project",
        "spdx_license",
        "attribution",
        "auto_download",
    }
)


class ModelManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelArtifact:
    tag: str
    revision: str | None
    url: str | None
    filename: str
    byte_length: int | None
    sha256: str | None
    config_filename: str | None
    config_sha256: str | None
    loader_type: str
    source_project: str
    spdx_license: str | None
    attribution: str
    auto_download: bool
    blocked_reason: str | None

    @property
    def release_eligible(self) -> bool:
        return bool(
            self.auto_download
            and self.revision
            and self.url
            and self.byte_length
            and self.sha256
            and (not self.config_filename or self.config_sha256)
            and self.spdx_license
            and not self.blocked_reason
        )


class ModelManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._artifacts = self._load(path)
        self.by_tag = {artifact.tag: artifact for artifact in self._artifacts}
        self.by_filename = {artifact.filename: artifact for artifact in self._artifacts}

    @staticmethod
    def _load(path: Path) -> tuple[ModelArtifact, ...]:
        if not path.is_file():
            raise ModelManifestError("model manifest is missing")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelManifestError("model manifest is invalid") from exc
        if not isinstance(payload, dict) or int(payload.get("schema_version") or 0) != 1:
            raise ModelManifestError("unsupported model manifest schema")
        models = payload.get("models")
        if not isinstance(models, list) or not models:
            raise ModelManifestError("model manifest has no entries")
        result: list[ModelArtifact] = []
        tags: set[str] = set()
        filenames: set[str] = set()
        for raw in models:
            if not isinstance(raw, dict):
                raise ModelManifestError("model manifest entry is invalid")
            missing = REQUIRED_FIELDS - set(raw)
            if missing:
                raise ModelManifestError(f"model manifest entry is missing: {', '.join(sorted(missing))}")
            tag = str(raw["tag"])
            filename = Path(str(raw["filename"])).name
            if not tag or not filename or tag in tags or filename in filenames:
                raise ModelManifestError("model manifest tags and filenames must be unique")
            tags.add(tag)
            filenames.add(filename)
            artifact = ModelArtifact(
                tag=tag,
                revision=_optional_text(raw.get("revision")),
                url=_optional_text(raw.get("url")),
                filename=filename,
                byte_length=_optional_positive_int(raw.get("byte_length")),
                sha256=_optional_hash(raw.get("sha256")),
                config_filename=_optional_text(raw.get("config_filename")),
                config_sha256=_optional_hash(raw.get("config_sha256")),
                loader_type=str(raw.get("loader_type") or ""),
                source_project=str(raw.get("source_project") or ""),
                spdx_license=_optional_text(raw.get("spdx_license")),
                attribution=str(raw.get("attribution") or ""),
                auto_download=bool(raw.get("auto_download")),
                blocked_reason=_optional_text(raw.get("blocked_reason")),
            )
            _validate_artifact(artifact)
            result.append(artifact)
        return tuple(result)

    def descriptor(self, tag: str) -> dict[str, Any]:
        artifact = self.by_tag.get(tag)
        if artifact is None:
            raise ModelManifestError(f"model {tag!r} is not in the manifest")
        if not artifact.release_eligible:
            reason = artifact.blocked_reason or "model rights or immutable hashes are not documented"
            raise ModelManifestError(f"automatic model download is blocked: {reason}")
        return {
            "tag": artifact.tag,
            "url": artifact.url,
            "filename": artifact.filename,
            "subdir": "models",
            "expected_size": artifact.byte_length,
            "sha256": artifact.sha256,
        }

    def verify_config(self, config_root: Path, tag: str) -> Path | None:
        artifact = self.by_tag.get(tag)
        if artifact is None:
            raise ModelManifestError(f"model {tag!r} is not in the manifest")
        if artifact.config_filename is None:
            return None
        path = config_root / artifact.config_filename
        _verify_file(path, artifact.config_sha256, None, label=f"config for {tag}")
        return path

    def verify_checkpoint(self, model_root: Path, tag: str) -> Path:
        artifact = self._require(tag)
        path = model_root / artifact.filename
        _verify_file(path, artifact.sha256, artifact.byte_length, label=f"checkpoint for {tag}")
        return path

    def _require(self, tag: str) -> ModelArtifact:
        artifact = self.by_tag.get(tag)
        if artifact is None:
            raise ModelManifestError(f"model {tag!r} is not in the manifest")
        if artifact.sha256 is None or artifact.byte_length is None:
            raise ModelManifestError(f"model {tag!r} is unsupported until its exact artifact hash is recorded")
        if not artifact.spdx_license:
            raise ModelManifestError(f"model {tag!r} is unsupported until its license is recorded")
        return artifact


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, sha256: str | None, byte_length: int | None, *, label: str) -> None:
    if not path.is_file():
        raise ModelManifestError(f"{label} is missing")
    if sha256 is None:
        raise ModelManifestError(f"{label} has no approved SHA-256")
    if byte_length is not None and path.stat().st_size != byte_length:
        raise ModelManifestError(f"{label} has the wrong byte length")
    if file_sha256(path) != sha256:
        raise ModelManifestError(f"{label} failed SHA-256 verification")


def _validate_artifact(artifact: ModelArtifact) -> None:
    if artifact.loader_type not in {"roformer-state-dict", "mdx-state-dict", "demucs-pickle"}:
        raise ModelManifestError(f"model {artifact.tag!r} has an unsupported loader type")
    if artifact.auto_download:
        if not artifact.release_eligible:
            raise ModelManifestError(f"auto-download model {artifact.tag!r} is missing release metadata")
        assert artifact.url is not None
        assert artifact.revision is not None
        parsed = urlparse(artifact.url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ModelManifestError(f"model {artifact.tag!r} must use HTTPS")
        if artifact.revision not in artifact.url:
            raise ModelManifestError(f"model {artifact.tag!r} URL is not pinned to its revision")
    if artifact.revision is not None and not REVISION_RE.fullmatch(artifact.revision):
        raise ModelManifestError(f"model {artifact.tag!r} revision is not a full commit hash")
    if artifact.config_filename and artifact.config_sha256 is None:
        raise ModelManifestError(f"model {artifact.tag!r} config has no SHA-256")
    if not artifact.source_project or not artifact.attribution:
        raise ModelManifestError(f"model {artifact.tag!r} attribution is incomplete")


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_hash(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if not SHA256_RE.fullmatch(text):
        raise ModelManifestError("manifest SHA-256 must contain exactly 64 hexadecimal characters")
    return text


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number <= 0:
        raise ModelManifestError("manifest byte length must be positive")
    return number
