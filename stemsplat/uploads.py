from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import unicodedata
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Iterable, Protocol

MIB = 1024 * 1024
GIB = 1024 * MIB
SUPPORTED_SUFFIXES = frozenset(
    {
        ".wav",
        ".wave",
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".oga",
        ".opus",
        ".aif",
        ".aiff",
        ".alac",
        ".mp4",
        ".m4v",
        ".mov",
        ".webm",
        ".mkv",
        ".avi",
    }
)


class UploadReadable(Protocol):
    filename: str | None
    content_type: str | None

    async def read(self, size: int = -1) -> bytes: ...

    async def close(self) -> None: ...


class UploadAdmissionError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True)
class UploadPolicy:
    max_file_bytes: int = 2 * GIB
    max_staged_bytes: int = 10 * GIB
    max_global_uploads: int = 4
    max_session_uploads: int = 2
    max_session_tasks: int = 20
    max_display_name_chars: int = 180
    chunk_bytes: int = 8 * MIB
    ffprobe_timeout_seconds: float = 30.0
    max_audio_channels: int = 2


@dataclass(frozen=True)
class StagedUpload:
    display_name: str
    path: Path
    size_bytes: int
    probe: dict[str, object]


def normalize_display_name(raw_name: str | None, *, limit: int = 180) -> str:
    raw = str(raw_name or "")
    if "/" in raw or "\\" in raw:
        raise UploadAdmissionError(415, "invalid-name", "The file name must not contain a path.")
    candidate = Path(raw).name
    normalized = unicodedata.normalize("NFC", candidate).strip()
    if not normalized:
        raise UploadAdmissionError(415, "invalid-name", "The file name is empty.")
    if len(normalized) > limit:
        raise UploadAdmissionError(415, "invalid-name", f"The file name must be {limit} characters or fewer.")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise UploadAdmissionError(415, "invalid-name", "The file name contains control characters.")
    if normalized in {".", ".."}:
        raise UploadAdmissionError(415, "invalid-name", "The file name is invalid.")
    return normalized


def _signature_kind(sample: bytes) -> str | None:
    if len(sample) < 4:
        return None
    if sample.startswith(b"RIFF") and len(sample) >= 12 and sample[8:12] == b"WAVE":
        return "wav"
    if sample.startswith(b"FORM") and len(sample) >= 12 and sample[8:12] in {b"AIFF", b"AIFC"}:
        return "aiff"
    if sample.startswith(b"fLaC"):
        return "flac"
    if sample.startswith(b"OggS"):
        return "ogg"
    if sample.startswith(b"\x1aE\xdf\xa3"):
        return "matroska"
    if sample.startswith(b"ID3"):
        return "mp3"
    if len(sample) >= 2 and sample[0] == 0xFF and (sample[1] & 0xE0) == 0xE0:
        return "aac" if (sample[1] & 0xF6) == 0xF0 else "mp3"
    if len(sample) >= 12 and sample[4:8] == b"ftyp":
        return "mp4"
    if sample.startswith(b"\x00\x00\x01\xba") or sample.startswith(b"\x00\x00\x01\xb3"):
        return "mpeg"
    if sample.startswith(b"\x00\x00\x01") or sample.startswith(b"RIFF"):
        return "avi"
    return None


KIND_SUFFIXES = {
    "wav": {".wav", ".wave"},
    "aiff": {".aif", ".aiff"},
    "flac": {".flac"},
    "ogg": {".ogg", ".oga", ".opus"},
    "matroska": {".webm", ".mkv"},
    "mp3": {".mp3"},
    "aac": {".aac"},
    "mp4": {".m4a", ".alac", ".mp4", ".m4v", ".mov"},
    "mpeg": {".mp4", ".m4v", ".mov"},
    "avi": {".avi"},
}


def validate_signature(path: Path, display_name: str, content_type: str | None = None) -> str:
    suffix = Path(display_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UploadAdmissionError(415, "unsupported-extension", "This file extension is not supported.")
    mime = str(content_type or "").lower().split(";", 1)[0].strip()
    if mime and not (
        mime.startswith("audio/")
        or mime.startswith("video/")
        or mime in {"application/octet-stream", "application/ogg"}
    ):
        raise UploadAdmissionError(415, "unsupported-mime", "The upload MIME type is not supported.")
    try:
        with path.open("rb") as handle:
            sample = handle.read(64)
    except OSError as exc:
        raise UploadAdmissionError(415, "unreadable", "The staged upload could not be read.") from exc
    kind = _signature_kind(sample)
    if kind is None or suffix not in KIND_SUFFIXES.get(kind, set()):
        raise UploadAdmissionError(415, "signature-mismatch", "The file content does not match its extension.")
    return kind


def probe_media(
    path: Path,
    *,
    ffprobe: str,
    timeout_seconds: float = 30.0,
    max_audio_channels: int = 2,
) -> dict[str, object]:
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,channels,sample_rate:format=duration,format_name",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UploadAdmissionError(415, "probe-failed", "The media file could not be validated.") from exc
    if result.returncode != 0:
        raise UploadAdmissionError(415, "malformed-media", "The media file is malformed or truncated.")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise UploadAdmissionError(415, "probe-failed", "The media validator returned invalid data.") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    audio_streams = [
        item
        for item in (streams if isinstance(streams, list) else [])
        if isinstance(item, dict) and item.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise UploadAdmissionError(415, "no-audio", "No supported audio stream was found.")
    try:
        channels = max(int(item.get("channels") or 0) for item in audio_streams)
        sample_rates = [int(item.get("sample_rate") or 0) for item in audio_streams]
        duration = float((payload.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError) as exc:
        raise UploadAdmissionError(415, "malformed-media", "The media file has invalid stream metadata.") from exc
    if channels <= 0 or channels > max_audio_channels:
        raise UploadAdmissionError(415, "unsupported-channels", "Only mono and stereo input are supported.")
    if duration <= 0 or max(sample_rates, default=0) <= 0:
        raise UploadAdmissionError(415, "malformed-media", "The media file has no valid duration or sample rate.")
    return {
        "duration": duration,
        "channels": channels,
        "codec": str(audio_streams[0].get("codec_name") or ""),
        "format": str((payload.get("format") or {}).get("format_name") or ""),
    }


def staged_size(root: Path) -> int:
    total = 0
    if not root.exists():
        return total
    for child in root.rglob("*"):
        with contextlib.suppress(OSError):
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
    return total


class UploadAdmissionController:
    def __init__(self, root: Path, *, policy: UploadPolicy | None = None) -> None:
        self.root = root
        self.policy = policy or UploadPolicy()
        self._lock = threading.RLock()
        self._global_active = 0
        self._session_active: dict[str, int] = {}
        self._reserved_bytes = 0

    @asynccontextmanager
    async def slot(self, session_id: str, *, outstanding_tasks: int) -> AsyncIterator[Callable[[int], None]]:
        normalized_session = session_id or "local"
        with self._lock:
            if outstanding_tasks >= self.policy.max_session_tasks:
                raise UploadAdmissionError(429, "task-limit", "Too many outstanding LAN tasks.")
            if self._global_active >= self.policy.max_global_uploads:
                raise UploadAdmissionError(429, "global-upload-limit", "Too many uploads are in progress.")
            if self._session_active.get(normalized_session, 0) >= self.policy.max_session_uploads:
                raise UploadAdmissionError(429, "session-upload-limit", "Too many uploads are in progress for this device.")
            self._global_active += 1
            self._session_active[normalized_session] = self._session_active.get(normalized_session, 0) + 1

        local_reserved = 0

        def reserve(amount: int) -> None:
            nonlocal local_reserved
            if amount == 0:
                return
            with self._lock:
                if amount < 0:
                    released = min(local_reserved, -amount)
                    local_reserved -= released
                    self._reserved_bytes = max(0, self._reserved_bytes - released)
                    return
                current_staged = staged_size(self.root)
                projected = current_staged + self._reserved_bytes + amount
                if projected > self.policy.max_staged_bytes:
                    raise UploadAdmissionError(413, "staging-limit", "Staged upload storage is full.")
                local_reserved += amount
                self._reserved_bytes += amount

        try:
            yield reserve
        finally:
            with self._lock:
                self._reserved_bytes = max(0, self._reserved_bytes - local_reserved)
                self._global_active = max(0, self._global_active - 1)
                count = self._session_active.get(normalized_session, 1) - 1
                if count <= 0:
                    self._session_active.pop(normalized_session, None)
                else:
                    self._session_active[normalized_session] = count


async def stage_upload(
    upload: UploadReadable,
    destination_root: Path,
    *,
    controller: UploadAdmissionController,
    session_id: str,
    outstanding_tasks: int,
    ffprobe: str,
) -> StagedUpload:
    policy = controller.policy
    display_name = normalize_display_name(upload.filename, limit=policy.max_display_name_chars)
    suffix = Path(display_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UploadAdmissionError(415, "unsupported-extension", "This file extension is not supported.")
    mime = str(upload.content_type or "").lower().split(";", 1)[0].strip()
    if mime and not (mime.startswith("audio/") or mime.startswith("video/") or mime in {"application/octet-stream", "application/ogg"}):
        raise UploadAdmissionError(415, "unsupported-mime", "The upload MIME type is not supported.")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{uuid.uuid4().hex}{suffix}"
    written = 0
    try:
        async with controller.slot(session_id, outstanding_tasks=outstanding_tasks) as reserve:
            with destination.open("xb") as handle:
                while True:
                    chunk = await upload.read(policy.chunk_bytes)
                    if not chunk:
                        break
                    if written + len(chunk) > policy.max_file_bytes:
                        raise UploadAdmissionError(413, "file-too-large", "The upload exceeds the 2 GiB file limit.")
                    reserve(len(chunk))
                    try:
                        handle.write(chunk)
                    finally:
                        reserve(-len(chunk))
                    written += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if written <= 0:
                raise UploadAdmissionError(415, "empty-upload", "The upload is empty.")
            validate_signature(destination, display_name, upload.content_type)
            probe = probe_media(
                destination,
                ffprobe=ffprobe,
                timeout_seconds=policy.ffprobe_timeout_seconds,
                max_audio_channels=policy.max_audio_channels,
            )
            return StagedUpload(display_name, destination, written, probe)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        with contextlib.suppress(Exception):
            await upload.close()


def stage_local_batch(
    paths: Iterable[Path],
    destination_root: Path,
    *,
    policy: UploadPolicy,
    ffprobe: str,
) -> list[StagedUpload]:
    sources = [path.expanduser().resolve() for path in paths]
    if not sources:
        return []
    names = [normalize_display_name(path.name, limit=policy.max_display_name_chars) for path in sources]
    folded = [unicodedata.normalize("NFC", name).casefold() for name in names]
    if len(set(folded)) != len(folded):
        raise UploadAdmissionError(415, "duplicate-name", "The batch contains duplicate normalized file names.")

    destination_root.mkdir(parents=True, exist_ok=True)
    temporary_root = destination_root / f".batch-{uuid.uuid4().hex}"
    temporary_root.mkdir(mode=0o700)
    staged: list[StagedUpload] = []
    committed: list[StagedUpload] = []
    try:
        existing_bytes = staged_size(destination_root)
        batch_bytes = 0
        for source, display_name in zip(sources, names):
            if not source.is_file() or source.is_symlink():
                raise UploadAdmissionError(415, "missing-file", "One of the selected files is unavailable.")
            size = source.stat().st_size
            if size <= 0:
                raise UploadAdmissionError(415, "empty-upload", "One of the selected files is empty.")
            if size > policy.max_file_bytes:
                raise UploadAdmissionError(413, "file-too-large", "A selected file exceeds the 2 GiB limit.")
            batch_bytes += size
            if existing_bytes + batch_bytes > policy.max_staged_bytes:
                raise UploadAdmissionError(413, "staging-limit", "The selected batch exceeds staged storage.")
            suffix = Path(display_name).suffix.lower()
            temporary_path = temporary_root / f"{uuid.uuid4().hex}{suffix}"
            shutil.copyfile(source, temporary_path)
            with temporary_path.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            validate_signature(temporary_path, display_name)
            probe = probe_media(
                temporary_path,
                ffprobe=ffprobe,
                timeout_seconds=policy.ffprobe_timeout_seconds,
                max_audio_channels=policy.max_audio_channels,
            )
            staged.append(StagedUpload(display_name, temporary_path, size, probe))
        for item in staged:
            final_path = destination_root / item.path.name
            os.replace(item.path, final_path)
            committed.append(StagedUpload(item.display_name, final_path, item.size_bytes, item.probe))
        return committed
    except BaseException:
        for item in staged:
            item.path.unlink(missing_ok=True)
        for item in committed:
            item.path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
