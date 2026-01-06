from __future__ import annotations

import os
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

import certifi

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

FILES = [
    # Vocals
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true",
        "subdir": "models",
        "filename": "mel_band_roformer_vocals_becruily.ckpt",
        "tag": "vocals",
    },
    # Instrumental
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true",
        "subdir": "models",
        "filename": "mel_band_roformer_instrumental_becruily.ckpt",
        "tag": "instrumental",
    },
    # Deux (new)
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt?download=true",
        "subdir": "models",
        "filename": "becruily_deux.ckpt",
        "tag": "deux",
    },
    # Additional legacy/support models (always fetched)
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt?download=true",
        "subdir": "models",
        "filename": "mel_band_roformer_karaoke_becruily.ckpt",
        "tag": None,
    },
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt?download=true",
        "subdir": "models",
        "filename": "becruily_guitar.ckpt",
        "tag": None,
    },
    {
        "url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_bass.onnx?download=true",
        "subdir": "models",
        "filename": "kuielab_a_bass.onnx",
        "tag": None,
    },
    {
        "url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_drums.onnx?download=true",
        "subdir": "models",
        "filename": "kuielab_a_drums.onnx",
        "tag": None,
    },
    {
        "url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_other.onnx?download=true",
        "subdir": "models",
        "filename": "kuielab_a_other.onnx",
        "tag": None,
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


def download_to(base_dir: Path, selected: list[str] | None = None) -> None:
    base_dir = base_dir.resolve()
    wanted = set(selected or [])
    for item in FILES:
        tag = item.get("tag")
        if wanted and tag and tag not in wanted:
            continue
        dest = base_dir / item["subdir"] / item["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        remote_len = try_head_content_length(item["url"])
        if dest.exists() and remote_len is not None and dest.stat().st_size == remote_len:
            continue

        req = Request(item["url"], headers={"User-Agent": "app-downloader"})
        with urlopen(req, timeout=60, context=SSL_CONTEXT) as r:
            if tmp.exists():
                tmp.unlink()
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)

        os.replace(tmp, dest)
