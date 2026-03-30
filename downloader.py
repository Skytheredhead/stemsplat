from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

try:  # noqa: SIM105 - allow offline environments
    import certifi
except Exception:  # pragma: no cover - optional dependency
    certifi = None

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where()) if certifi else ssl.create_default_context()

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
]


def try_head_content_length(url: str) -> int | None:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "app-downloader"})
        with urlopen(req, timeout=60, context=SSL_CONTEXT) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else None
    except Exception:
        return None


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
    wanted = set(selected or [])
    selected_items = []
    for item in FILES:
        tag = item.get("tag")
        if wanted and tag and tag not in wanted:
            continue
        selected_items.append(item)

    total_files = len(selected_items)
    total_bytes = 0
    cached_lengths: dict[str, int | None] = {}
    completed_bytes = 0
    for item in selected_items:
        tag = str(item.get("tag") or "")
        remote_len = try_head_content_length(item["url"])
        cached_lengths[tag] = remote_len
        if isinstance(remote_len, int) and remote_len > 0:
            total_bytes += remote_len

    for index, item in enumerate(selected_items, start=1):
        tag = str(item.get("tag") or "")
        dest = base_dir / item["subdir"] / item["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        remote_len = cached_lengths.get(tag)
        if dest.exists() and remote_len is not None and dest.stat().st_size == remote_len:
            completed_bytes += remote_len
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

        req = Request(item["url"], headers={"User-Agent": "app-downloader"})
        with urlopen(req, timeout=60, context=SSL_CONTEXT) as r:
            if tmp.exists():
                tmp.unlink()
            bytes_written = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
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

        os.replace(tmp, dest)
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
