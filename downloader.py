from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:  # noqa: SIM105 - allow offline environments
    import certifi
except Exception:  # pragma: no cover - optional dependency
    certifi = None

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
REQUEST_TIMEOUT = 60
MIN_FREE_SPACE_BUFFER_BYTES = 64 * 1024 * 1024
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where()) if certifi else ssl.create_default_context()
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

FILES = [
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true",
        "subdir": "models",
        "filename": "mel_band_roformer_vocals_becruily.ckpt",
        "tag": "vocals",
    },
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true",
        "subdir": "models",
        "filename": "mel_band_roformer_instrumental_becruily.ckpt",
        "tag": "instrumental",
    },
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt?download=true",
        "subdir": "models",
        "filename": "becruily_deux.ckpt",
        "tag": "deux",
    },
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt?download=true",
        "subdir": "models",
        "filename": "becruily_guitar.ckpt",
        "tag": "guitar",
    },
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt?download=true",
        "subdir": "models",
        "filename": "mel_band_roformer_karaoke_becruily.ckpt",
        "tag": "mel_band_karaoke",
    },
    {
        "url": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt?download=true",
        "subdir": "models",
        "filename": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        "tag": "denoise",
    },
    {
        "url": "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt?download=true",
        "subdir": "models",
        "filename": "BS-Rofo-SW-Fixed.ckpt",
        "tag": "bs_roformer_6s",
    },
    {
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
        "subdir": "models",
        "filename": "f7e0c4bc-ba3fe64a.th",
        "tag": "htdemucs_ft_drums",
    },
    {
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
        "subdir": "models",
        "filename": "d12395a8-e57c48e6.th",
        "tag": "htdemucs_ft_bass",
    },
    {
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th",
        "subdir": "models",
        "filename": "92cfc3b6-ef3bcb9c.th",
        "tag": "htdemucs_ft_other",
    },
    {
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
        "subdir": "models",
        "filename": "5c90dfd2-34c22ccb.th",
        "tag": "htdemucs_6s",
    },
    {
        "url": "https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
        "subdir": "models",
        "filename": "aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
        "tag": "drumsep_6s",
    },
    {
        "url": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.5/model_drumsep.th",
        "subdir": "models",
        "filename": "model_drumsep.th",
        "tag": "drumsep_4s",
    },
]


@dataclass(frozen=True)
class RemoteFileMetadata:
    size: int | None = None
    sha256: str | None = None


class ModelDownloadError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool,
        tag: str = "",
        filename: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.tag = tag
        self.filename = filename


def _clean_header_value(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().strip('"').strip("'")
    return cleaned or None


def _normalize_sha256(value: str | None) -> str | None:
    cleaned = _clean_header_value(value)
    if cleaned is None:
        return None
    candidate = cleaned.lower()
    return candidate if HEX_SHA256_RE.fullmatch(candidate) else None


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_from_headers(headers: Mapping[str, str | None]) -> RemoteFileMetadata:
    content_length = headers.get("Content-Length")
    size = int(content_length) if content_length and str(content_length).isdigit() else None
    sha256 = None
    for header_name in ("X-Linked-ETag", "ETag", "X-Xet-Hash"):
        sha256 = _normalize_sha256(headers.get(header_name))
        if sha256 is not None:
            break
    return RemoteFileMetadata(size=size, sha256=sha256)


def _merge_remote_metadata(primary: RemoteFileMetadata, fallback: RemoteFileMetadata) -> RemoteFileMetadata:
    return RemoteFileMetadata(
        size=primary.size if primary.size is not None else fallback.size,
        sha256=primary.sha256 if primary.sha256 is not None else fallback.sha256,
    )


def _format_bytes(value: int) -> str:
    size = float(max(0, value))
    units = ("B", "KB", "MB", "GB", "TB")
    unit = units[0]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"


def _download_error(
    code: str,
    message: str,
    *,
    retryable: bool,
    item: Mapping[str, Any] | None = None,
) -> ModelDownloadError:
    return ModelDownloadError(
        code,
        message,
        retryable=retryable,
        tag=str(item.get("tag") or "") if item is not None else "",
        filename=str(item.get("filename") or "") if item is not None else "",
    )


def _validate_item(item: Mapping[str, Any], *, base_dir: Path) -> tuple[str, Path]:
    url = str(item.get("url") or "").strip()
    subdir = str(item.get("subdir") or "").strip()
    filename = str(item.get("filename") or "").strip()
    if not url or not subdir or not filename:
        raise _download_error("invalid-config", "model download entry is missing required fields", retryable=False, item=item)
    if not url.startswith("https://"):
        raise _download_error("insecure-url", f"refusing insecure model url for {filename}", retryable=False, item=item)

    filename_path = Path(filename)
    subdir_path = Path(subdir)
    if filename_path.name != filename or filename in {".", ".."}:
        raise _download_error("invalid-filename", f"invalid download filename for {filename}", retryable=False, item=item)
    if subdir_path.is_absolute() or any(part == ".." for part in subdir_path.parts):
        raise _download_error("invalid-subdir", f"invalid destination folder for {filename}", retryable=False, item=item)

    dest = (base_dir / subdir_path / filename_path).resolve()
    if os.path.commonpath([str(base_dir), str(dest)]) != str(base_dir):
        raise _download_error("path-escape", f"download path escapes models directory for {filename}", retryable=False, item=item)
    return url, dest


def _ensure_regular_file_or_missing(path: Path, *, item: Mapping[str, Any], label: str) -> None:
    if path.is_symlink():
        raise _download_error("unsafe-path", f"{label} for {item['filename']} is a symlink", retryable=False, item=item)
    if not path.exists():
        return
    if not path.is_file():
        raise _download_error("invalid-path", f"{label} for {item['filename']} is not a file", retryable=False, item=item)


def _cleanup_partial_file(tmp: Path, *, item: Mapping[str, Any]) -> None:
    if not tmp.exists():
        return
    _ensure_regular_file_or_missing(tmp, item=item, label="partial download")
    try:
        tmp.unlink()
    except OSError as exc:
        raise _map_os_error(exc, item=item, action="remove the partial download", retryable=False) from exc


def _ensure_destination_parent(dest: Path, *, item: Mapping[str, Any]) -> None:
    if dest.parent.exists() and not dest.parent.is_dir():
        raise _download_error("invalid-destination", f"destination folder for {item['filename']} is not a directory", retryable=False, item=item)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _map_os_error(exc, item=item, action="create the destination folder", retryable=False) from exc


def _ensure_disk_space(directory: Path, *, required_bytes: int | None, item: Mapping[str, Any]) -> None:
    if required_bytes is None or required_bytes <= 0:
        return
    try:
        free_bytes = shutil.disk_usage(directory).free
    except OSError as exc:
        raise _map_os_error(exc, item=item, action="inspect free disk space", retryable=False) from exc
    minimum_needed = required_bytes + MIN_FREE_SPACE_BUFFER_BYTES
    if free_bytes < minimum_needed:
        raise _download_error(
            "disk-full",
            f"not enough free disk space for {item['filename']} (need {_format_bytes(minimum_needed)}, have {_format_bytes(free_bytes)})",
            retryable=False,
            item=item,
        )


def _map_os_error(exc: OSError, *, item: Mapping[str, Any], action: str, retryable: bool | None = None) -> ModelDownloadError:
    if exc.errno in {errno.ENOSPC, errno.EDQUOT}:
        return _download_error("disk-full", f"ran out of disk space while trying to {action} for {item['filename']}", retryable=False, item=item)
    if exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
        return _download_error("permission-denied", f"permission error while trying to {action} for {item['filename']}", retryable=False, item=item)
    should_retry = retryable if retryable is not None else False
    return _download_error("filesystem-error", f"could not {action} for {item['filename']}: {exc}", retryable=should_retry, item=item)


def _map_url_error(exc: Exception, *, item: Mapping[str, Any], action: str) -> ModelDownloadError:
    if isinstance(exc, HTTPError):
        if exc.code in {401, 403, 404, 410}:
            return _download_error("remote-missing", f"remote file unavailable while trying to {action} for {item['filename']} (HTTP {exc.code})", retryable=False, item=item)
        if exc.code == 429 or exc.code >= 500:
            return _download_error("server-busy", f"server error while trying to {action} for {item['filename']} (HTTP {exc.code})", retryable=True, item=item)
        return _download_error("http-error", f"download failed while trying to {action} for {item['filename']} (HTTP {exc.code})", retryable=False, item=item)
    if isinstance(exc, URLError):
        return _download_error("network-error", f"network error while trying to {action} for {item['filename']}: {exc.reason}", retryable=True, item=item)
    if isinstance(exc, TimeoutError):
        return _download_error("timeout", f"timed out while trying to {action} for {item['filename']}", retryable=True, item=item)
    if isinstance(exc, ssl.SSLError):
        return _download_error("ssl-error", f"secure connection failed while trying to {action} for {item['filename']}: {exc}", retryable=True, item=item)
    return _download_error("download-error", f"unexpected error while trying to {action} for {item['filename']}: {exc}", retryable=True, item=item)


def _matches_expected_file(path: Path, *, expected_size: int | None, expected_sha256: str | None) -> bool:
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    if expected_sha256 is not None:
        return _sha256_for_file(path) == expected_sha256
    return expected_size is not None


def get_remote_file_metadata(url: str) -> RemoteFileMetadata:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "app-downloader"})
        with urlopen(req, timeout=REQUEST_TIMEOUT, context=SSL_CONTEXT) as r:
            return _metadata_from_headers(r.headers)
    except Exception:
        return RemoteFileMetadata()


def _emit_progress(
    progress_cb: Callable[[dict[str, Any]], None] | None,
    *,
    tag: str,
    filename: str,
    current_index: int,
    total_files: int,
    downloaded_bytes: int,
    total_bytes: int,
) -> None:
    if progress_cb is None:
        return
    pct = 0
    if total_bytes > 0:
        pct = int(max(0, min(100, round((downloaded_bytes / total_bytes) * 100))))
    elif total_files > 0:
        pct = int(max(0, min(100, round((current_index / total_files) * 100))))
    progress_cb(
        {
            "tag": tag,
            "filename": filename,
            "current_index": current_index,
            "total_files": total_files,
            "downloaded_bytes": downloaded_bytes,
            "total_bytes": total_bytes,
            "pct": pct,
        }
    )


def download_to(
    base_dir: Path,
    selected: list[str] | None = None,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    base_dir = base_dir.resolve()
    available_by_tag = {str(item.get("tag") or ""): item for item in FILES if item.get("tag")}
    if selected:
        selected_items = [available_by_tag[tag] for tag in dict.fromkeys(str(tag) for tag in selected) if tag in available_by_tag]
    else:
        selected_items = list(FILES)
    if not selected_items:
        _emit_progress(
            progress_cb,
            tag="",
            filename="",
            current_index=0,
            total_files=0,
            downloaded_bytes=0,
            total_bytes=0,
        )
        return

    total_files = len(selected_items)
    total_bytes = 0
    cached_metadata: dict[str, RemoteFileMetadata] = {}
    completed_bytes = 0
    for item in selected_items:
        tag = str(item.get("tag") or "")
        url, _dest = _validate_item(item, base_dir=base_dir)
        remote_sha256 = _normalize_sha256(str(item.get("sha256") or "")) if item.get("sha256") else None
        if item.get("sha256") and remote_sha256 is None:
            raise _download_error("invalid-config", f"invalid sha256 configured for {item['filename']}", retryable=False, item=item)
        remote_metadata = get_remote_file_metadata(url)
        if remote_sha256 is not None:
            remote_metadata = RemoteFileMetadata(size=remote_metadata.size, sha256=remote_sha256)
        cached_metadata[tag] = remote_metadata
        remote_len = remote_metadata.size
        if isinstance(remote_len, int) and remote_len > 0:
            total_bytes += remote_len

    for index, item in enumerate(selected_items, start=1):
        tag = str(item.get("tag") or "")
        url, dest = _validate_item(item, base_dir=base_dir)
        _ensure_destination_parent(dest, item=item)
        tmp = dest.with_suffix(dest.suffix + ".part")
        _ensure_regular_file_or_missing(dest, item=item, label="destination file")

        remote_metadata = cached_metadata.get(tag, RemoteFileMetadata())
        remote_len = remote_metadata.size
        expected_sha256 = remote_metadata.sha256
        if _matches_expected_file(dest, expected_size=remote_len, expected_sha256=expected_sha256):
            completed_bytes += remote_len if isinstance(remote_len, int) and remote_len > 0 else dest.stat().st_size
            _emit_progress(
                progress_cb,
                tag=tag,
                filename=item["filename"],
                current_index=index,
                total_files=total_files,
                downloaded_bytes=completed_bytes,
                total_bytes=total_bytes,
            )
            continue

        _cleanup_partial_file(tmp, item=item)
        _ensure_disk_space(dest.parent, required_bytes=remote_len, item=item)

        bytes_written = 0
        digest = hashlib.sha256()
        try:
            req = Request(url, headers={"User-Agent": "app-downloader"})
            with urlopen(req, timeout=REQUEST_TIMEOUT, context=SSL_CONTEXT) as r:
                response_metadata = _metadata_from_headers(r.headers)
                merged_metadata = _merge_remote_metadata(remote_metadata, response_metadata)
                remote_len = merged_metadata.size
                expected_sha256 = merged_metadata.sha256
                _ensure_disk_space(dest.parent, required_bytes=remote_len, item=item)

                try:
                    with open(tmp, "wb") as f:
                        while True:
                            chunk = r.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            f.write(chunk)
                            digest.update(chunk)
                            bytes_written += len(chunk)
                            _emit_progress(
                                progress_cb,
                                tag=tag,
                                filename=item["filename"],
                                current_index=index,
                                total_files=total_files,
                                downloaded_bytes=completed_bytes + bytes_written,
                                total_bytes=total_bytes,
                            )
                except OSError as exc:
                    raise _map_os_error(exc, item=item, action="write the downloaded model") from exc
        except ModelDownloadError:
            _cleanup_partial_file(tmp, item=item)
            raise
        except (HTTPError, URLError, TimeoutError, ssl.SSLError) as exc:
            _cleanup_partial_file(tmp, item=item)
            raise _map_url_error(exc, item=item, action="download the model") from exc
        except OSError as exc:
            _cleanup_partial_file(tmp, item=item)
            raise _map_os_error(exc, item=item, action="download the model", retryable=True) from exc
        except Exception as exc:
            _cleanup_partial_file(tmp, item=item)
            raise _map_url_error(exc, item=item, action="download the model") from exc

        if bytes_written <= 0:
            _cleanup_partial_file(tmp, item=item)
            raise _download_error("empty-download", f"downloaded zero bytes for {item['filename']}", retryable=True, item=item)
        if remote_len is not None and bytes_written != remote_len:
            _cleanup_partial_file(tmp, item=item)
            raise _download_error(
                "size-mismatch",
                f"download size mismatch for {item['filename']} (expected {remote_len} bytes, got {bytes_written})",
                retryable=True,
                item=item,
            )

        downloaded_sha256 = digest.hexdigest()
        if expected_sha256 is not None and downloaded_sha256 != expected_sha256:
            _cleanup_partial_file(tmp, item=item)
            raise _download_error("checksum-mismatch", f"checksum mismatch for {item['filename']}", retryable=True, item=item)
        try:
            os.replace(tmp, dest)
        except OSError as exc:
            _cleanup_partial_file(tmp, item=item)
            raise _map_os_error(exc, item=item, action="finalize the downloaded model") from exc
        completed_bytes += remote_len if isinstance(remote_len, int) and remote_len > 0 else bytes_written
        _emit_progress(
            progress_cb,
            tag=tag,
            filename=item["filename"],
            current_index=index,
            total_files=total_files,
            downloaded_bytes=completed_bytes,
            total_bytes=total_bytes,
        )


def download_url_to_path(
    url: str,
    dest: Path,
    *,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
    user_agent: str = "app-downloader",
    force_redownload: bool = False,
) -> Path:
    dest = Path(dest).expanduser().resolve()
    item = {
        "url": str(url or "").strip(),
        "filename": dest.name,
        "tag": dest.stem or dest.name,
    }
    url = str(item["url"])
    if not url:
        raise _download_error("invalid-config", "download url is missing", retryable=False, item=item)
    if not url.startswith("https://"):
        raise _download_error("insecure-url", f"refusing insecure download url for {item['filename']}", retryable=False, item=item)
    if not dest.name:
        raise _download_error("invalid-filename", "download destination filename is missing", retryable=False, item=item)

    _ensure_destination_parent(dest, item=item)
    tmp = dest.with_suffix(dest.suffix + ".part")
    _ensure_regular_file_or_missing(dest, item=item, label="destination file")

    remote_metadata = get_remote_file_metadata(url)
    remote_len = remote_metadata.size
    expected_sha256 = remote_metadata.sha256
    if not force_redownload and _matches_expected_file(dest, expected_size=remote_len, expected_sha256=expected_sha256):
        existing_size = remote_len if isinstance(remote_len, int) and remote_len > 0 else dest.stat().st_size
        _emit_progress(
            progress_cb,
            tag=str(item["tag"]),
            filename=str(item["filename"]),
            current_index=1,
            total_files=1,
            downloaded_bytes=existing_size,
            total_bytes=existing_size,
        )
        return dest

    _cleanup_partial_file(tmp, item=item)
    _ensure_disk_space(dest.parent, required_bytes=remote_len, item=item)

    bytes_written = 0
    digest = hashlib.sha256()
    try:
        req = Request(url, headers={"User-Agent": user_agent})
        with urlopen(req, timeout=REQUEST_TIMEOUT, context=SSL_CONTEXT) as response:
            response_metadata = _metadata_from_headers(response.headers)
            merged_metadata = _merge_remote_metadata(remote_metadata, response_metadata)
            remote_len = merged_metadata.size
            expected_sha256 = merged_metadata.sha256
            _ensure_disk_space(dest.parent, required_bytes=remote_len, item=item)
            try:
                with open(tmp, "wb") as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        bytes_written += len(chunk)
                        _emit_progress(
                            progress_cb,
                            tag=str(item["tag"]),
                            filename=str(item["filename"]),
                            current_index=1,
                            total_files=1,
                            downloaded_bytes=bytes_written,
                            total_bytes=remote_len or 0,
                        )
            except OSError as exc:
                raise _map_os_error(exc, item=item, action="write the downloaded file") from exc
    except ModelDownloadError:
        _cleanup_partial_file(tmp, item=item)
        raise
    except (HTTPError, URLError, TimeoutError, ssl.SSLError) as exc:
        _cleanup_partial_file(tmp, item=item)
        raise _map_url_error(exc, item=item, action="download the file") from exc
    except OSError as exc:
        _cleanup_partial_file(tmp, item=item)
        raise _map_os_error(exc, item=item, action="download the file", retryable=True) from exc
    except Exception as exc:
        _cleanup_partial_file(tmp, item=item)
        raise _map_url_error(exc, item=item, action="download the file") from exc

    if bytes_written <= 0:
        _cleanup_partial_file(tmp, item=item)
        raise _download_error("empty-download", f"downloaded zero bytes for {item['filename']}", retryable=True, item=item)
    if remote_len is not None and bytes_written != remote_len:
        _cleanup_partial_file(tmp, item=item)
        raise _download_error(
            "size-mismatch",
            f"download size mismatch for {item['filename']} (expected {remote_len} bytes, got {bytes_written})",
            retryable=True,
            item=item,
        )

    downloaded_sha256 = digest.hexdigest()
    if expected_sha256 is not None and downloaded_sha256 != expected_sha256:
        _cleanup_partial_file(tmp, item=item)
        raise _download_error("checksum-mismatch", f"checksum mismatch for {item['filename']}", retryable=True, item=item)
    try:
        os.replace(tmp, dest)
    except OSError as exc:
        _cleanup_partial_file(tmp, item=item)
        raise _map_os_error(exc, item=item, action="finalize the downloaded file") from exc

    _emit_progress(
        progress_cb,
        tag=str(item["tag"]),
        filename=str(item["filename"]),
        current_index=1,
        total_files=1,
        downloaded_bytes=remote_len if isinstance(remote_len, int) and remote_len > 0 else bytes_written,
        total_bytes=remote_len if isinstance(remote_len, int) and remote_len > 0 else bytes_written,
    )
    return dest
