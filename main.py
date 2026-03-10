from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import logging
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf
import torch
import yaml
from app_paths import (
    CONFIG_DIR,
    LOG_DIR,
    MODEL_DIR,
    OUTPUT_ROOT,
    RESOURCE_DIR,
    RUNTIME_DIR,
    SETTINGS_PATH,
    UPLOAD_DIR,
    WEB_DIR,
    WORK_DIR,
    ensure_app_dirs,
    model_search_dirs,
)
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

try:
    from mutagen import File as MutagenFile
except Exception:  # pragma: no cover - optional cleanup dependency
    MutagenFile = None

BASE_DIR = RESOURCE_DIR

try:  # pragma: no cover - import failure is surfaced at runtime
    from split.mel_band_roformer import MelBandRoformer
except Exception as exc:  # pragma: no cover - keep app importable for syntax checks
    MelBandRoformer = None  # type: ignore[assignment]
    _model_import_error = exc
else:
    _model_import_error = None


class ErrorCode(str, Enum):
    MODEL_IMPORT_FAILED = "E001"
    MODEL_MISSING = "E002"
    CONFIG_MISSING = "E003"
    FFMPEG_MISSING = "E004"
    AUDIO_DECODE_FAILED = "E005"
    AUDIO_LOAD_FAILED = "E006"
    SEPARATION_FAILED = "E007"
    TASK_NOT_FOUND = "E008"
    INVALID_REQUEST = "E009"


@dataclass
class AppError(Exception):
    code: ErrorCode
    message: str

    def to_http(self, status: int = 400) -> HTTPException:
        return HTTPException(status_code=status, detail={"code": self.code, "message": self.message})


class TaskStopped(Exception):
    pass


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    config: str
    segment: int


@dataclass(frozen=True)
class SourceInfo:
    suffix: str
    codec: str | None
    bit_rate: int | None
    channels: int
    has_cover: bool


@dataclass(frozen=True)
class ExportPlan:
    suffix: str
    audio_args: list[str]
    supports_cover: bool


LOG_PATH = LOG_DIR / "main_stemsplat.log"
MODEL_SEARCH_DIRS = model_search_dirs()
COMPAT_SETTINGS_DEFAULTS = {
    "output_format": "same_as_input",
    "output_root": str(OUTPUT_ROOT),
    "structure_mode": "flat",
}

MODEL_SPECS: dict[str, ModelSpec] = {
    "vocals": ModelSpec(
        filename="mel_band_roformer_vocals_becruily.ckpt",
        config="Mel Band Roformer Vocals Config.yaml",
        segment=352_800,
    ),
    "instrumental": ModelSpec(
        filename="mel_band_roformer_instrumental_becruily.ckpt",
        config="Mel Band Roformer Instrumental Config.yaml",
        segment=352_800,
    ),
    "deux": ModelSpec(
        filename="becruily_deux.ckpt",
        config="config_deux_becruily.yaml",
        segment=573_300,
    ),
}

MODEL_ALIAS_MAP = {
    "mel_band_roformer_vocals_becruily.ckpt": ["Mel Band Roformer Vocals.ckpt"],
    "mel_band_roformer_instrumental_becruily.ckpt": ["Mel Band Roformer Instrumental.ckpt"],
    "becruily_deux.ckpt": ["Mel Band Roformer Deux.ckpt"],
}

MODEL_URLS = {
    "vocals": "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true",
    "instrumental": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true",
    "deux": "https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt?download=true",
}

MODE_CHOICES = {"vocals", "instrumental", "both_deux", "both_separate"}
OUTPUT_FORMAT_CHOICES = {"same_as_input", "mp3_320", "mp3_128", "wav", "m4a", "flac"}
TERMINAL_STATUSES = {"done", "error", "stopped"}
SUPPORTED_AUDIO_SUFFIXES = {
    ".wav",
    ".wave",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".aif",
    ".aiff",
    ".alac",
    ".ogg",
    ".opus",
}
OVERLAP_RATIO = 0.12
UPLOAD_CHUNK_SIZE = 1024 * 1024
RUNTIME_CLEANUP_MAX_AGE_SEC = 24 * 60 * 60


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    stem = path.stem
    suffix = path.suffix
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _safe_stem(name: str) -> str:
    raw = Path(name).stem or "split"
    cleaned = re.sub(r"[^\w .-]+", "_", raw, flags=re.ASCII).strip(" ._")
    return cleaned or "split"


def _locate_case_insensitive(path: Path) -> Path | None:
    if path.exists():
        return path
    parent = path.parent
    if not parent.exists():
        return None
    target = path.name.lower()
    for candidate in parent.iterdir():
        if candidate.name.lower() == target:
            return candidate
    return None


compat_settings_lock = threading.RLock()


def _load_compat_settings() -> dict[str, str]:
    settings = dict(COMPAT_SETTINGS_DEFAULTS)
    if not SETTINGS_PATH.exists():
        return settings
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key, value in data.items():
                if key in settings and isinstance(value, str):
                    settings[key] = value
    except Exception:
        logging.getLogger("stemsplat").debug("failed to load settings from %s", SETTINGS_PATH, exc_info=True)
    settings["output_root"] = str(OUTPUT_ROOT)
    settings["structure_mode"] = "flat"
    return settings


def _save_compat_settings(settings: dict[str, str]) -> None:
    payload = dict(COMPAT_SETTINGS_DEFAULTS)
    payload.update(settings)
    payload["output_root"] = str(OUTPUT_ROOT)
    payload["structure_mode"] = "flat"
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compat_settings_payload() -> dict[str, str]:
    with compat_settings_lock:
        return dict(_compat_settings)


def _set_compat_settings(patch: dict[str, str]) -> dict[str, str]:
    with compat_settings_lock:
        updated = dict(_compat_settings)
        updated.update(patch)
        _save_compat_settings(updated)
        _compat_settings.clear()
        _compat_settings.update(updated)
        return dict(_compat_settings)


def _mode_to_stems(mode: str) -> list[str]:
    if mode == "vocals":
        return ["vocals"]
    if mode == "instrumental":
        return ["instrumental"]
    if mode == "both_deux":
        return ["deux"]
    if mode == "both_separate":
        return ["vocals", "instrumental"]
    return []


def _stems_to_mode(stems_raw: str) -> str:
    stems = [item.strip().lower() for item in stems_raw.split(",") if item.strip()]
    if stems == ["vocals"]:
        return "vocals"
    if stems == ["instrumental"]:
        return "instrumental"
    if stems == ["deux"]:
        return "both_deux"
    if stems == ["vocals", "instrumental"] or stems == ["instrumental", "vocals"]:
        return "both_separate"
    raise AppError(ErrorCode.INVALID_REQUEST, "Invalid stem selection.")


_compat_settings = _load_compat_settings()


ensure_app_dirs()
file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
    handlers=[stream_handler, file_handler],
)
logger = logging.getLogger("stemsplat")
logging.getLogger("python_multipart").setLevel(logging.INFO)
for uvicorn_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    uvicorn_logger = logging.getLogger(uvicorn_name)
    uvicorn_logger.setLevel(logging.INFO)
    if file_handler not in uvicorn_logger.handlers:
        uvicorn_logger.addHandler(file_handler)

for path in (MODEL_DIR, RUNTIME_DIR, UPLOAD_DIR, WORK_DIR, OUTPUT_ROOT):
    _ensure_dir(path)


def _close_installer_ui(port: int = 6060) -> None:
    url = f"http://localhost:{port}/installer_shutdown"
    try:
        with urllib.request.urlopen(url, timeout=1):
            logger.debug("closed installer ui on %s", url)
    except Exception:
        logger.debug("installer ui not reachable at %s", url)


def _cleanup_old_runtime_entries(path: Path, max_age_seconds: int = RUNTIME_CLEANUP_MAX_AGE_SEC) -> None:
    cutoff = time.time() - max_age_seconds
    for candidate in path.iterdir():
        try:
            if candidate.stat().st_mtime >= cutoff:
                continue
            _cleanup_path(candidate)
        except Exception:
            logger.debug("failed to cleanup runtime entry %s", candidate, exc_info=True)


def _required_models_for_mode(mode: str) -> list[str]:
    if mode == "vocals":
        return ["vocals"]
    if mode == "instrumental":
        return ["instrumental"]
    if mode == "both_deux":
        return ["deux"]
    if mode == "both_separate":
        return ["vocals", "instrumental"]
    return []


def _model_file_exists(filename: str) -> bool:
    search_names = [filename, *MODEL_ALIAS_MAP.get(filename, [])]
    for base_dir in MODEL_SEARCH_DIRS:
        for search_name in search_names:
            if _locate_case_insensitive(base_dir / search_name):
                return True
    return False


def _find_missing_models_for_mode(mode: str) -> list[str]:
    missing: list[str] = []
    for key in _required_models_for_mode(mode):
        if not _model_file_exists(MODEL_SPECS[key].filename):
            missing.append(key)
    return missing


def _validate_models_for_mode(mode: str) -> None:
    missing = _find_missing_models_for_mode(mode)
    if missing:
        joined = ", ".join(missing)
        raise AppError(ErrorCode.MODEL_MISSING, f"Missing required model files: {joined}.")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", port))
            return True
        except OSError:
            return False


def _ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    candidates: list[str] = []
    if path:
        candidates.append(path)

    try:  # pragma: no cover - optional fallback dependency
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            candidates.append(bundled)
    except Exception:
        logger.debug("imageio-ffmpeg unavailable; relying on PATH")

    for candidate in candidates:
        try:
            subprocess.run(
                [candidate, "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return candidate
        except Exception:
            logger.debug("ffmpeg candidate failed: %s", candidate, exc_info=True)

    raise AppError(ErrorCode.FFMPEG_MISSING, "ffmpeg not found; install ffmpeg or add imageio-ffmpeg.")


def _ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if path:
        return path
    ffmpeg_path = _ensure_ffmpeg()
    probe_candidate = str(Path(ffmpeg_path).with_name("ffprobe"))
    if Path(probe_candidate).exists():
        return probe_candidate
    raise AppError(ErrorCode.FFMPEG_MISSING, "ffprobe not found; install ffmpeg.")


def _fallback_source_info(path: Path) -> SourceInfo:
    channels = 2
    bit_rate: int | None = None
    has_cover = False
    codec: str | None = None

    if MutagenFile is not None:
        try:
            audio = MutagenFile(path)
            if audio is not None:
                info = getattr(audio, "info", None)
                if info is not None:
                    raw_channels = getattr(info, "channels", None)
                    raw_bit_rate = getattr(info, "bitrate", None)
                    raw_codec = getattr(info, "codec", None) or getattr(info, "codec_description", None)
                    if raw_channels:
                        channels = max(1, min(2, int(raw_channels)))
                    if raw_bit_rate:
                        bit_rate = int(raw_bit_rate)
                    if raw_codec:
                        codec = str(raw_codec).lower()

                tags = getattr(audio, "tags", None)
                if tags is not None:
                    has_cover = any(
                        str(key).lower() in {"apic", "covr", "metadata_block_picture"}
                        for key in tags.keys()
                    )
        except Exception:
            logger.debug("mutagen probe failed for %s", path, exc_info=True)

    return SourceInfo(
        suffix=path.suffix.lower(),
        codec=codec,
        bit_rate=bit_rate,
        channels=channels,
        has_cover=has_cover,
    )


def _probe_source(path: Path) -> SourceInfo:
    fallback = _fallback_source_info(path)
    try:
        cmd = [
            _ffprobe_path(),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,bit_rate,channels:format=bit_rate",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout or "{}")
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return fallback

    streams = data.get("streams") or []
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    has_cover = any(stream.get("codec_type") == "video" for stream in streams)
    bit_rate = audio_stream.get("bit_rate") or (data.get("format") or {}).get("bit_rate")
    with contextlib.suppress(Exception):
        bit_rate = int(bit_rate) if bit_rate else None
    return SourceInfo(
        suffix=path.suffix.lower(),
        codec=audio_stream.get("codec_name") or fallback.codec,
        bit_rate=bit_rate if isinstance(bit_rate, int) else fallback.bit_rate,
        channels=max(1, min(2, int(audio_stream.get("channels") or fallback.channels or 2))),
        has_cover=has_cover or fallback.has_cover,
    )


def _clamp_kbps(bit_rate: int | None, fallback: int) -> int:
    if not bit_rate:
        return fallback
    return max(96, min(320, int(round(bit_rate / 1000.0))))


def _resolve_export_plan(source_info: SourceInfo, selection: str) -> ExportPlan:
    if selection == "mp3_320":
        return ExportPlan(".mp3", ["-c:a", "libmp3lame", "-b:a", "320k", "-id3v2_version", "3"], True)
    if selection == "mp3_128":
        return ExportPlan(".mp3", ["-c:a", "libmp3lame", "-b:a", "128k", "-id3v2_version", "3"], True)
    if selection == "wav":
        return ExportPlan(".wav", ["-c:a", "pcm_s16le"], False)
    if selection == "m4a":
        return ExportPlan(".m4a", ["-c:a", "aac", "-b:a", "256k"], True)
    if selection == "flac":
        return ExportPlan(".flac", ["-c:a", "flac"], True)

    codec = (source_info.codec or "").lower()
    suffix = source_info.suffix
    if suffix == ".mp3" or codec == "mp3":
        kbps = _clamp_kbps(source_info.bit_rate, 320)
        return ExportPlan(".mp3", ["-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-id3v2_version", "3"], True)
    if suffix in {".wav", ".wave"}:
        return ExportPlan(".wav", ["-c:a", "pcm_s16le"], False)
    if suffix in {".aif", ".aiff"}:
        return ExportPlan(".aiff", ["-c:a", "pcm_s16be"], False)
    if suffix == ".flac" or codec == "flac":
        return ExportPlan(".flac", ["-c:a", "flac"], True)
    if suffix == ".aac":
        kbps = _clamp_kbps(source_info.bit_rate, 256)
        return ExportPlan(".aac", ["-c:a", "aac", "-b:a", f"{kbps}k"], False)
    if suffix == ".ogg":
        kbps = _clamp_kbps(source_info.bit_rate, 192)
        if codec == "opus":
            return ExportPlan(".ogg", ["-c:a", "libopus", "-b:a", f"{kbps}k"], False)
        return ExportPlan(".ogg", ["-c:a", "libvorbis", "-q:a", "6"], False)
    if suffix == ".opus" or codec == "opus":
        kbps = _clamp_kbps(source_info.bit_rate, 192)
        return ExportPlan(".opus", ["-c:a", "libopus", "-b:a", f"{kbps}k"], False)
    if suffix == ".m4a" or codec == "aac":
        kbps = _clamp_kbps(source_info.bit_rate, 256)
        return ExportPlan(".m4a", ["-c:a", "aac", "-b:a", f"{kbps}k"], True)
    if suffix == ".alac" or codec == "alac":
        return ExportPlan(".m4a", ["-c:a", "alac"], True)
    if codec.startswith("pcm_"):
        return ExportPlan(".wav", ["-c:a", "pcm_s16le"], False)
    kbps = _clamp_kbps(source_info.bit_rate, 256)
    return ExportPlan(".m4a", ["-c:a", "aac", "-b:a", f"{kbps}k"], True)


def _strip_title_metadata(path: Path) -> None:
    if MutagenFile is None:
        return
    try:
        audio = MutagenFile(path)
        if audio is None or audio.tags is None:
            return
        tags = audio.tags
        with contextlib.suppress(Exception):
            if hasattr(tags, "delall"):
                for key in ("TIT2", "title", "\xa9nam"):
                    tags.delall(key)
        for key in list(tags.keys()):
            lower = str(key).lower()
            if lower in {"tit2", "title", "\xa9nam"}:
                with contextlib.suppress(Exception):
                    del tags[key]
        audio.save()
    except Exception:
        logger.debug("failed to strip title metadata from %s", path, exc_info=True)


def _export_stem(
    stem_wav: Path,
    source_path: Path,
    dest_path: Path,
    plan: ExportPlan,
    has_cover: bool,
) -> Path:
    ffmpeg_path = _ensure_ffmpeg()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = ensure_unique_path(dest_path)

    def _build_command(include_cover: bool) -> list[str]:
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(stem_wav),
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-map_metadata",
            "1",
        ]
        if include_cover:
            cmd.extend(["-map", "1:v?"])
        cmd.extend(plan.audio_args)
        if include_cover:
            cmd.extend(["-c:v", "copy", "-disposition:v", "attached_pic"])
        cmd.extend(["-metadata", "title=", str(candidate)])
        return cmd

    attempts = [False]
    if plan.supports_cover and has_cover:
        attempts = [True, False]

    last_error: Exception | None = None
    for include_cover in attempts:
        try:
            subprocess.run(
                _build_command(include_cover),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            _strip_title_metadata(candidate)
            return candidate
        except Exception as exc:
            last_error = exc
            _cleanup_path(candidate)
            logger.warning(
                "export failed for %s with include_cover=%s: %s",
                candidate,
                include_cover,
                exc,
            )
    raise AppError(ErrorCode.SEPARATION_FAILED, f"Export failed: {last_error}") from last_error


def _decode_audio_to_wav(source_path: Path, work_dir: Path, channels: int) -> Path:
    ffmpeg_path = _ensure_ffmpeg()
    decoded_path = work_dir / "input.wav"
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ar",
        "44100",
        "-ac",
        str(max(1, min(2, channels))),
        "-vn",
        str(decoded_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise AppError(ErrorCode.FFMPEG_MISSING, "ffmpeg not found.") from exc
    except Exception as exc:
        raise AppError(ErrorCode.AUDIO_DECODE_FAILED, f"Could not decode audio: {exc}") from exc
    if not decoded_path.exists():
        raise AppError(ErrorCode.AUDIO_DECODE_FAILED, "Decoded WAV was not created.")
    return decoded_path


def _load_waveform(wav_path: Path) -> torch.Tensor:
    try:
        data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AppError(ErrorCode.AUDIO_LOAD_FAILED, f"Could not read WAV: {exc}") from exc
    if sample_rate != 44100:
        raise AppError(ErrorCode.AUDIO_LOAD_FAILED, f"Expected 44.1 kHz audio, got {sample_rate}.")
    waveform = torch.from_numpy(data.T.copy())
    if waveform.ndim != 2:
        raise AppError(ErrorCode.AUDIO_LOAD_FAILED, "Decoded audio has an invalid shape.")
    if waveform.shape[0] == 0:
        raise AppError(ErrorCode.AUDIO_LOAD_FAILED, "Decoded audio is empty.")
    return waveform


def _write_temp_wave(out_dir: Path, name: str, tensor: torch.Tensor) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / name
    data = tensor.detach().cpu()
    if not torch.isfinite(data).all():
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = torch.clamp(data.float(), -1.0, 1.0)
    sf.write(candidate, data.T.contiguous().numpy(), 44100, subtype="PCM_16")
    return candidate


def _normalize_prediction(pred: torch.Tensor, input_channels: int, chunk_len: int) -> torch.Tensor:
    if pred.dim() == 2:
        if pred.shape[0] in (1, input_channels):
            pred = pred.unsqueeze(0)
        else:
            pred = pred[:, None, :]
    elif pred.dim() == 3 and pred.shape[1] not in (1, input_channels):
        pred = pred.permute(1, 0, 2)
    pred = pred[..., :chunk_len]
    if pred.shape[1] == 1 and input_channels == 2:
        pred = pred.repeat(1, 2, 1)
    return pred


def _prepare_model_input(waveform: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
    original_channels = waveform.shape[0]
    if original_channels == 1:
        waveform = waveform.repeat(2, 1)
    elif original_channels > 2:
        waveform = waveform[:2]
        original_channels = 2
    return waveform.to(device=device, dtype=torch.float32), original_channels


def _restore_output_channels(tensor: torch.Tensor, original_channels: int) -> torch.Tensor:
    if original_channels <= 1 and tensor.shape[0] > 1:
        return tensor.mean(dim=0, keepdim=True)
    return tensor[: max(1, original_channels)]


def _map_fraction(start_pct: int, end_pct: int, fraction: float) -> int:
    fraction = max(0.0, min(1.0, fraction))
    return int(round(start_pct + ((end_pct - start_pct) * fraction)))


def _cleanup_path(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(Exception):
            path.unlink()


def _safe_mps_empty_cache() -> None:
    if getattr(torch, "mps", None) and hasattr(torch.mps, "empty_cache"):
        with contextlib.suppress(Exception):
            torch.mps.empty_cache()


def _load_roformer_model(model_path: Path, config_path: Path, device: torch.device) -> torch.nn.Module:
    if MelBandRoformer is None:
        raise AppError(ErrorCode.MODEL_IMPORT_FAILED, f"Roformer import failed: {_model_import_error}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.unsafe_load(handle)
    model = MelBandRoformer(**(cfg.get("model") or {}))
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


def select_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning("MPS unavailable; falling back to CPU")
    return torch.device("cpu")


class ModelManager:
    def __init__(self) -> None:
        self.device = select_device()
        self.cache: dict[str, torch.nn.Module] = {}

    def _resolve_model_path(self, filename: str) -> Path:
        search_names = [filename, *MODEL_ALIAS_MAP.get(filename, [])]
        for base_dir in MODEL_SEARCH_DIRS:
            for search_name in search_names:
                match = _locate_case_insensitive(base_dir / search_name)
                if match:
                    return match
        raise AppError(ErrorCode.MODEL_MISSING, f"Missing model file: {filename}")

    def _resolve_config_path(self, filename: str) -> Path:
        found = _locate_case_insensitive(CONFIG_DIR / filename)
        if found:
            return found
        raise AppError(ErrorCode.CONFIG_MISSING, f"Missing config file: {filename}")

    def get(self, name: str) -> torch.nn.Module:
        if name in self.cache:
            return self.cache[name]
        if name not in MODEL_SPECS:
            raise AppError(ErrorCode.INVALID_REQUEST, f"Unknown model: {name}")
        spec = MODEL_SPECS[name]
        model_path = self._resolve_model_path(spec.filename)
        config_path = self._resolve_config_path(spec.config)
        model = _load_roformer_model(model_path, config_path, self.device)
        self.cache[name] = model
        return model


_model_manager: ModelManager | None = None
_model_manager_lock = threading.Lock()


def _get_model_manager() -> ModelManager:
    global _model_manager
    with _model_manager_lock:
        if _model_manager is None:
            _model_manager = ModelManager()
        return _model_manager


def _run_model_chunks(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    segment: int,
    progress_cb: Callable[[float], None],
    stop_check: Callable[[], None],
) -> torch.Tensor:
    working, original_channels = _prepare_model_input(waveform, next(model.parameters()).device)
    input_channels = working.shape[0]
    length = working.shape[1]
    step = max(1, int(segment * (1.0 - OVERLAP_RATIO)))
    acc: torch.Tensor | None = None
    counts = torch.zeros((1, length), device=working.device, dtype=working.dtype)

    progress_cb(0.0)
    with torch.no_grad():
        for start in range(0, length, step):
            stop_check()
            end = min(start + segment, length)
            chunk = working[:, start:end]
            if chunk.shape[1] < segment:
                padded = torch.zeros((input_channels, segment), device=working.device, dtype=working.dtype)
                padded[:, : chunk.shape[1]] = chunk
            else:
                padded = chunk
            pred = model(padded.unsqueeze(0))[0]
            pred = _normalize_prediction(pred, input_channels, chunk.shape[1])
            if acc is None:
                acc = torch.zeros(
                    (pred.shape[0], pred.shape[1], length),
                    device=working.device,
                    dtype=pred.dtype,
                )
            acc[:, :, start : start + pred.shape[-1]] += pred
            counts[:, start : start + pred.shape[-1]] += 1
            progress_cb(end / max(1, length))

    if acc is None:
        raise AppError(ErrorCode.SEPARATION_FAILED, "Model produced no output.")

    denom = counts.clamp_min(1).unsqueeze(0)
    restored = acc / denom
    outputs = []
    for index in range(restored.shape[0]):
        outputs.append(_restore_output_channels(restored[index], original_channels))
    return torch.stack(outputs, dim=0)


app = FastAPI()
tasks_lock = threading.RLock()
tasks: dict[str, dict[str, Any]] = {}
task_queue: queue.Queue[str] = queue.Queue()


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "name": task["original_name"],
        "mode": task["mode"],
        "output_format": task["output_format"],
        "status": task["status"],
        "stage": task["stage"],
        "pct": task["pct"],
        "eta_seconds": task["eta_seconds"],
        "out_dir": task["out_dir"],
        "outputs": list(task["outputs"]),
        "error": task["error"],
        "version": task["version"],
    }


def _require_task(task_id: str) -> dict[str, Any]:
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
        return task


def _estimate_eta(task: dict[str, Any], pct: int) -> int | None:
    started_at = task.get("started_at")
    if started_at is None or pct < 3 or pct >= 100:
        return 0 if pct >= 100 else None
    elapsed = max(1.0, time.time() - started_at)
    estimate = int((elapsed / (pct / 100.0)) - elapsed)
    previous = task.get("eta_seconds")
    if isinstance(previous, int) and previous > 0:
        estimate = int((previous * 0.6) + (estimate * 0.4))
    return max(0, estimate)


def _set_task_progress(task_id: str, stage: str, pct: int) -> None:
    with tasks_lock:
        task = tasks[task_id]
        if task["started_at"] is None:
            task["started_at"] = time.time()
        if task["status"] not in TERMINAL_STATUSES:
            task["status"] = "running"
        task["stage"] = stage
        task["pct"] = max(0, min(100, int(pct)))
        task["eta_seconds"] = _estimate_eta(task, task["pct"])
        task["version"] += 1


def _mark_task_done(task_id: str, out_dir: Path, outputs: list[str]) -> None:
    with tasks_lock:
        task = tasks[task_id]
        task["status"] = "done"
        task["stage"] = "Done"
        task["pct"] = 100
        task["eta_seconds"] = 0
        task["out_dir"] = str(out_dir)
        task["outputs"] = list(outputs)
        task["error"] = None
        task["finished_at"] = time.time()
        task["version"] += 1


def _mark_task_error(task_id: str, message: str) -> None:
    with tasks_lock:
        task = tasks[task_id]
        task["status"] = "error"
        task["stage"] = "Error"
        task["pct"] = max(0, int(task.get("pct", 0)))
        task["eta_seconds"] = None
        task["error"] = message
        task["finished_at"] = time.time()
        task["version"] += 1


def _mark_task_stopped(task_id: str) -> None:
    with tasks_lock:
        task = tasks[task_id]
        task["status"] = "stopped"
        task["stage"] = "Stopped"
        task["eta_seconds"] = None
        task["finished_at"] = time.time()
        task["version"] += 1


def _request_task_stop(task_id: str) -> None:
    with tasks_lock:
        task = tasks[task_id]
        task["stop_event"].set()
        if task["status"] == "queued":
            task["status"] = "stopped"
            task["stage"] = "Stopped"
            task["eta_seconds"] = None
        elif task["status"] == "running":
            task["stage"] = "Stopping"
            task["eta_seconds"] = None
        task["version"] += 1


def _stop_check(task_id: str) -> None:
    with tasks_lock:
        if tasks[task_id]["stop_event"].is_set():
            raise TaskStopped()


def _build_task_payload(
    *,
    task_id: str,
    original_name: str,
    source_path: Path,
    mode: str,
    output_format: str,
    auto_start: bool = True,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "original_name": original_name,
        "source_path": str(source_path),
        "mode": mode,
        "output_format": output_format,
        "status": "queued" if auto_start else "ready",
        "stage": "Waiting in queue" if auto_start else "Ready",
        "pct": 0,
        "eta_seconds": None,
        "out_dir": None,
        "outputs": [],
        "error": None,
        "version": 0,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "stop_event": threading.Event(),
    }


def _create_output_dir(filename: str) -> Path:
    _ = filename
    return _ensure_dir(OUTPUT_ROOT)


def _validate_mode_and_output_format(mode: str, output_format: str) -> None:
    if mode not in MODE_CHOICES:
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid split mode.")
    if output_format not in OUTPUT_FORMAT_CHOICES:
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid output format.")
    _validate_models_for_mode(mode)


async def _store_uploaded_file(file: UploadFile) -> tuple[str, Path]:
    original_name = Path(file.filename or "upload").name
    suffix = Path(original_name).suffix.lower()
    content_type = (file.content_type or "").lower()
    if suffix and suffix not in SUPPORTED_AUDIO_SUFFIXES:
        raise AppError(ErrorCode.INVALID_REQUEST, f"Unsupported file type: {suffix}")
    if not suffix and not content_type.startswith("audio/"):
        raise AppError(ErrorCode.INVALID_REQUEST, "Unsupported file type. Add a supported audio file.")

    task_id = str(uuid.uuid4())
    stored_name = f"{task_id}_{original_name}"
    source_path = UPLOAD_DIR / stored_name
    bytes_written = 0
    try:
        with source_path.open("wb") as handle:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    except Exception as exc:
        _cleanup_path(source_path)
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not save upload: {exc}") from exc
    finally:
        with contextlib.suppress(Exception):
            await file.close()

    if bytes_written <= 0:
        _cleanup_path(source_path)
        raise AppError(ErrorCode.INVALID_REQUEST, "Uploaded file is empty.")
    return original_name, source_path


def _register_task(
    *,
    original_name: str,
    source_path: Path,
    mode: str,
    output_format: str,
    auto_start: bool,
) -> dict[str, Any]:
    task_id = str(uuid.uuid4())
    payload = _build_task_payload(
        task_id=task_id,
        original_name=original_name,
        source_path=source_path,
        mode=mode,
        output_format=output_format,
        auto_start=auto_start,
    )
    with tasks_lock:
        tasks[task_id] = payload
    if auto_start:
        task_queue.put(task_id)
    return payload


def _enqueue_task(task_id: str) -> dict[str, Any]:
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
        if task["status"] in {"queued", "running"}:
            return task
        if task["status"] in TERMINAL_STATUSES:
            return task
        task["status"] = "queued"
        task["stage"] = "Waiting in queue"
        task["eta_seconds"] = None
        task["version"] += 1
    task_queue.put(task_id)
    return task


def _compat_stage(task: dict[str, Any]) -> str:
    status = str(task.get("status") or "").lower()
    if status == "ready":
        return "ready"
    if status == "queued":
        return "queued"
    if status == "done":
        return "done"
    if status == "stopped":
        return "stopped"
    if status == "error":
        return "error"
    return str(task.get("stage") or "queued")


def _compat_public_task(task: dict[str, Any]) -> dict[str, Any]:
    public = _public_task(task)
    stage = _compat_stage(public)
    pct = -1 if public["status"] == "error" else public["pct"]
    return {
        "task_id": public["id"],
        "id": public["id"],
        "name": public["name"],
        "stage": stage,
        "pct": pct,
        "stems": _mode_to_stems(public["mode"]),
        "out_dir": public["out_dir"],
        "error": public["error"],
        "outputs": public["outputs"],
    }


def _process_task(task_id: str) -> None:
    work_dir: Path | None = None
    output_dir: Path | None = None
    manager: ModelManager | None = None
    written_outputs: list[Path] = []
    try:
        task = _require_task(task_id)
        if task["stop_event"].is_set():
            raise TaskStopped()

        source_path = Path(task["source_path"])
        if not source_path.exists():
            raise AppError(ErrorCode.INVALID_REQUEST, "Uploaded source file is missing.")

        _set_task_progress(task_id, "Loading models", 1)
        manager = ModelManager()
        source_info = _probe_source(source_path)
        output_dir = _create_output_dir(task["original_name"])
        with tasks_lock:
            tasks[task_id]["out_dir"] = str(output_dir)
            tasks[task_id]["version"] += 1

        work_dir = Path(tempfile.mkdtemp(prefix=f"stemsplat_{task_id[:8]}_", dir=str(WORK_DIR)))
        _set_task_progress(task_id, "Preparing audio", 4)
        decoded_path = _decode_audio_to_wav(source_path, work_dir, source_info.channels)
        _stop_check(task_id)
        waveform = _load_waveform(decoded_path)

        temp_outputs: list[tuple[str, Path]] = []
        mode = task["mode"]

        if mode == "vocals":
            vocals_model = manager.get("vocals")
            vocals_pred = _run_model_chunks(
                vocals_model,
                waveform,
                MODEL_SPECS["vocals"].segment,
                progress_cb=lambda frac: _set_task_progress(task_id, "Running vocals model", _map_fraction(6, 88, frac)),
                stop_check=lambda: _stop_check(task_id),
            )
            temp_outputs.append(("vocals", _write_temp_wave(work_dir, "vocals.wav", vocals_pred[0])))
        elif mode == "instrumental":
            instrumental_model = manager.get("instrumental")
            instrumental_pred = _run_model_chunks(
                instrumental_model,
                waveform,
                MODEL_SPECS["instrumental"].segment,
                progress_cb=lambda frac: _set_task_progress(task_id, "Running instrumental model", _map_fraction(6, 88, frac)),
                stop_check=lambda: _stop_check(task_id),
            )
            temp_outputs.append(("instrumental", _write_temp_wave(work_dir, "instrumental.wav", instrumental_pred[0])))
        elif mode == "both_deux":
            deux_model = manager.get("deux")
            pair_pred = _run_model_chunks(
                deux_model,
                waveform,
                MODEL_SPECS["deux"].segment,
                progress_cb=lambda frac: _set_task_progress(task_id, "Running deux model", _map_fraction(6, 90, frac)),
                stop_check=lambda: _stop_check(task_id),
            )
            vocals_tensor = pair_pred[0]
            instrumental_tensor = (
                pair_pred[1]
                if pair_pred.shape[0] > 1
                else waveform[: vocals_tensor.shape[0], : vocals_tensor.shape[1]] - vocals_tensor
            )
            temp_outputs.append(("vocals", _write_temp_wave(work_dir, "vocals.wav", vocals_tensor)))
            temp_outputs.append(("instrumental", _write_temp_wave(work_dir, "instrumental.wav", instrumental_tensor)))
        elif mode == "both_separate":
            vocals_model = manager.get("vocals")
            instrumental_model = manager.get("instrumental")
            vocals_pred = _run_model_chunks(
                vocals_model,
                waveform,
                MODEL_SPECS["vocals"].segment,
                progress_cb=lambda frac: _set_task_progress(task_id, "Running vocals model", _map_fraction(6, 48, frac)),
                stop_check=lambda: _stop_check(task_id),
            )
            temp_outputs.append(("vocals", _write_temp_wave(work_dir, "vocals.wav", vocals_pred[0])))
            instrumental_pred = _run_model_chunks(
                instrumental_model,
                waveform,
                MODEL_SPECS["instrumental"].segment,
                progress_cb=lambda frac: _set_task_progress(task_id, "Running instrumental model", _map_fraction(50, 90, frac)),
                stop_check=lambda: _stop_check(task_id),
            )
            temp_outputs.append(("instrumental", _write_temp_wave(work_dir, "instrumental.wav", instrumental_pred[0])))
        else:
            raise AppError(ErrorCode.INVALID_REQUEST, "Invalid split mode.")

        _stop_check(task_id)
        export_plan = _resolve_export_plan(source_info, task["output_format"])
        exported_files: list[str] = []
        total_exports = len(temp_outputs)
        for index, (label, temp_path) in enumerate(temp_outputs, start=1):
            start_pct = _map_fraction(92, 98, (index - 1) / max(1, total_exports))
            _set_task_progress(task_id, f"Exporting {label}", start_pct)
            final_path = output_dir / f"{_safe_stem(task['original_name'])} - {label}{export_plan.suffix}"
            exported = _export_stem(temp_path, source_path, final_path, export_plan, source_info.has_cover)
            written_outputs.append(exported)
            exported_files.append(exported.name)
            _set_task_progress(task_id, f"Exporting {label}", _map_fraction(92, 99, index / max(1, total_exports)))

        _mark_task_done(task_id, output_dir, exported_files)
    except TaskStopped:
        for output_path in written_outputs:
            _cleanup_path(output_path)
        _mark_task_stopped(task_id)
    except AppError as exc:
        for output_path in written_outputs:
            _cleanup_path(output_path)
        _mark_task_error(task_id, f"{exc.code}: {exc.message}")
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("task %s crashed", task_id)
        for output_path in written_outputs:
            _cleanup_path(output_path)
        _mark_task_error(task_id, f"{ErrorCode.SEPARATION_FAILED}: {exc}")
    finally:
        if manager is not None:
            del manager
        _cleanup_path(work_dir)
        gc.collect()
        _safe_mps_empty_cache()


def _task_worker() -> None:
    while True:
        task_id = task_queue.get()
        try:
            _process_task(task_id)
        finally:
            task_queue.task_done()


threading.Thread(target=_task_worker, daemon=True).start()


@app.on_event("startup")
async def _startup_cleanup() -> None:
    _close_installer_ui()
    _cleanup_old_runtime_entries(WORK_DIR)
    _cleanup_old_runtime_entries(UPLOAD_DIR)


@app.exception_handler(AppError)
async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    logger.error("app error for %s %s: %s %s", request.method, request.url.path, exc.code, exc.message)
    return JSONResponse(status_code=400, content={"code": exc.code, "message": exc.message})


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    started = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - started) * 1000
    logger.info("%s %s -> %s in %.1fms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/api/models_status")
async def models_status() -> dict[str, Any]:
    missing = sorted({item for mode in MODE_CHOICES for item in _find_missing_models_for_mode(mode)})
    return {"missing": missing, "models_dir": str(MODEL_DIR)}


@app.post("/api/open_models_folder")
async def open_models_folder() -> dict[str, str]:
    _ensure_dir(MODEL_DIR)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(MODEL_DIR)])
        elif os.name == "nt":
            subprocess.Popen(["explorer", str(MODEL_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(MODEL_DIR)])
    except Exception as exc:
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not open models folder: {exc}").to_http(500) from exc
    return {"status": "opened", "path": str(MODEL_DIR)}


@app.post("/api/tasks")
async def create_task(
    file: UploadFile = File(...),
    mode: str = Form("vocals"),
    output_format: str = Form("same_as_input"),
):
    try:
        _validate_mode_and_output_format(mode, output_format)
        original_name, source_path = await _store_uploaded_file(file)
        payload = _register_task(
            original_name=original_name,
            source_path=source_path,
            mode=mode,
            output_format=output_format,
            auto_start=True,
        )
        return _public_task(payload)
    except AppError as exc:
        raise exc.to_http()


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str):
    _require_task(task_id)

    async def _event_stream():
        last_version = -1
        last_ping_at = time.time()
        while True:
            with tasks_lock:
                task = tasks.get(task_id)
                if task is None:
                    break
                snapshot = _public_task(task)
            if snapshot["version"] != last_version:
                yield f"data: {json.dumps(snapshot)}\n\n"
                last_version = snapshot["version"]
                last_ping_at = time.time()
                if snapshot["status"] in TERMINAL_STATUSES:
                    break
            elif time.time() - last_ping_at >= 10:
                yield ": ping\n\n"
                last_ping_at = time.time()
            await asyncio.sleep(0.35)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    _require_task(task_id)
    _request_task_stop(task_id)
    return _public_task(_require_task(task_id))


@app.post("/api/tasks/{task_id}/retry")
async def retry_task(task_id: str):
    old_task = _require_task(task_id)
    source_path = Path(old_task["source_path"])
    if not source_path.exists():
        raise AppError(ErrorCode.INVALID_REQUEST, "Original upload is missing; re-add the file.").to_http()
    _validate_models_for_mode(old_task["mode"])

    payload = _register_task(
        original_name=old_task["original_name"],
        source_path=source_path,
        mode=old_task["mode"],
        output_format=old_task["output_format"],
        auto_start=True,
    )
    return _public_task(payload)


@app.post("/api/tasks/{task_id}/reveal")
async def reveal_output(task_id: str):
    task = _require_task(task_id)
    out_dir = task.get("out_dir")
    if not out_dir:
        raise AppError(ErrorCode.INVALID_REQUEST, "Output is not ready.").to_http(409)
    out_path = Path(out_dir)
    if not out_path.exists():
        raise AppError(ErrorCode.INVALID_REQUEST, "Output location is missing.").to_http(404)
    outputs = [out_path / str(name) for name in (task.get("outputs") or [])]
    existing_outputs = [path for path in outputs if path.exists()]
    if outputs and not existing_outputs:
        raise AppError(ErrorCode.INVALID_REQUEST, "Output files are missing.").to_http(404)
    single_output = existing_outputs[0] if len(existing_outputs) == 1 else None

    try:
        if sys.platform.startswith("darwin"):
            if single_output is not None:
                subprocess.Popen(["open", "-R", str(single_output)])
            else:
                subprocess.Popen(["open", str(out_path)])
        elif sys.platform.startswith("win"):
            if single_output is not None:
                subprocess.Popen(["explorer", "/select,", str(single_output)])
            else:
                os.startfile(str(out_path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(single_output.parent if single_output is not None else out_path)])
    except Exception as exc:
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not reveal output: {exc}").to_http(500) from exc

    return {"status": "opened", "path": str(single_output or out_path)}


@app.get("/settings")
async def get_settings() -> dict[str, str]:
    return _compat_settings_payload()


@app.post("/settings")
async def update_settings(request: Request) -> dict[str, str]:
    body = await request.json()
    if not isinstance(body, dict):
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid settings payload.").to_http()
    patch: dict[str, str] = {}
    output_format = body.get("output_format")
    if isinstance(output_format, str):
        if output_format not in OUTPUT_FORMAT_CHOICES:
            raise AppError(ErrorCode.INVALID_REQUEST, "Invalid output format.").to_http()
        patch["output_format"] = output_format
    return _set_compat_settings(patch)


@app.post("/upload")
async def compat_upload(
    file: UploadFile = File(...),
    stems: str = Form("vocals"),
    output_format: str | None = Form(None),
):
    try:
        mode = _stems_to_mode(stems)
        resolved_output_format = output_format or _compat_settings_payload()["output_format"]
        _validate_mode_and_output_format(mode, resolved_output_format)
        original_name, source_path = await _store_uploaded_file(file)
        payload = _register_task(
            original_name=original_name,
            source_path=source_path,
            mode=mode,
            output_format=resolved_output_format,
            auto_start=False,
        )
        return {"task_id": payload["id"], "stems": _mode_to_stems(mode)}
    except AppError as exc:
        raise exc.to_http()


@app.post("/start/{task_id}")
async def compat_start(task_id: str) -> dict[str, Any]:
    try:
        task = _enqueue_task(task_id)
        return _compat_public_task(task)
    except AppError as exc:
        raise exc.to_http(404 if exc.code == ErrorCode.TASK_NOT_FOUND else 400)


@app.get("/progress/{task_id}")
async def compat_progress(task_id: str):
    _require_task(task_id)

    async def _event_stream():
        last_version = -1
        last_ping_at = time.time()
        while True:
            with tasks_lock:
                task = tasks.get(task_id)
                if task is None:
                    break
                snapshot = _compat_public_task(task)
            if snapshot["pct"] != -1 and snapshot["stage"] == "error":
                snapshot["pct"] = -1
            if task["version"] != last_version:
                yield f"data: {json.dumps(snapshot)}\n\n"
                last_version = task["version"]
                last_ping_at = time.time()
                if task["status"] in TERMINAL_STATUSES:
                    break
            elif time.time() - last_ping_at >= 10:
                yield ": ping\n\n"
                last_ping_at = time.time()
            await asyncio.sleep(0.35)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/stop/{task_id}")
async def compat_stop(task_id: str) -> dict[str, Any]:
    _require_task(task_id)
    _request_task_stop(task_id)
    return _compat_public_task(_require_task(task_id))


@app.post("/rerun/{task_id}")
async def compat_rerun(task_id: str) -> dict[str, Any]:
    old_task = _require_task(task_id)
    source_path = Path(old_task["source_path"])
    if not source_path.exists():
        raise AppError(ErrorCode.INVALID_REQUEST, "Original upload is missing; re-add the file.").to_http()
    _validate_models_for_mode(old_task["mode"])
    payload = _register_task(
        original_name=old_task["original_name"],
        source_path=source_path,
        mode=old_task["mode"],
        output_format=old_task["output_format"],
        auto_start=True,
    )
    return {"task_id": payload["id"], "stems": _mode_to_stems(payload["mode"])}


@app.post("/reveal/{task_id}")
async def compat_reveal(task_id: str):
    return await reveal_output(task_id)


@app.post("/clear_all_uploads")
async def compat_clear_all_uploads() -> dict[str, str]:
    for root in (UPLOAD_DIR, WORK_DIR):
        for candidate in root.iterdir():
            _cleanup_path(candidate)
    return {"status": "cleared"}


@app.post("/rehydrate_tasks")
async def compat_rehydrate_tasks(request: Request) -> dict[str, list[dict[str, Any]]]:
    body = await request.json()
    if not isinstance(body, dict):
        return {"tasks": []}
    requested = body.get("tasks")
    if not isinstance(requested, list):
        return {"tasks": []}
    hydrated: list[dict[str, Any]] = []
    with tasks_lock:
        for item in requested:
            if not isinstance(item, dict):
                continue
            task_id = item.get("id")
            if not isinstance(task_id, str):
                continue
            task = tasks.get(task_id)
            if task is not None:
                hydrated.append(_compat_public_task(task))
    return {"tasks": hydrated}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse(INDEX_HTML)


@app.get("/favicon.ico")
async def favicon():
    icon_path = WEB_DIR / "favicon.ico"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/x-icon")
    raise AppError(ErrorCode.INVALID_REQUEST, "favicon missing").to_http(404)


@app.api_route("/shutdown", methods=["POST", "GET"])
async def shutdown():
    logger.warning("shutdown requested; exiting process")
    _close_installer_ui()

    def _exit_soon() -> None:
        time.sleep(0.25)
        os._exit(0)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return {"status": "shutting down"}


def cli_main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the stemsplat server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    _close_installer_ui()
    if not _port_available(args.port):
        raise SystemExit(f"Port {args.port} is already in use.")
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>stemsplat</title>
  <link rel="icon" type="image/x-icon" href="/favicon.ico">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-1: #0F2027;
      --bg-2: #2C5364;
      --text: #E7ECEF;
      --muted: #B8C4CC;
      --accent: #8ED8FF;
      --accent-strong: #b5ffd8;
      --accent-soft: #96c5d6;
      --icon: #9BB6C2;
      --card: rgba(23, 35, 41, 0.55);
      --card-done: rgba(40, 66, 77, 0.7);
      --border: rgba(255,255,255,0.06);
      --danger: #f57a6d;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      font-family: "Nunito Sans", sans-serif;
      color: var(--text);
      text-transform: lowercase;
      background: linear-gradient(135deg, #0B1A1F 0%, var(--bg-1) 35%, var(--bg-2) 100%);
      padding: 56px 0;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120"><filter id="n"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="3" stitchTiles="stitch"/></filter><rect width="120" height="120" filter="url(%23n)" opacity="0.035"/></svg>') repeat;
      opacity: .18;
    }

    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        repeating-linear-gradient(0deg, rgba(255,255,255,0.022) 0 1px, transparent 1px 2px),
        repeating-linear-gradient(90deg, rgba(0,0,0,0.024) 0 1px, transparent 1px 2px);
      opacity: .14;
      mix-blend-mode: soft-light;
    }

    @keyframes fadeUpIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
      opacity: 0;
      animation: fadeUpIn 0.55s ease forwards;
    }

    .delay-1 { animation-delay: 0.04s; }
    .delay-2 { animation-delay: 0.12s; }
    .delay-3 { animation-delay: 0.2s; }
    .delay-4 { animation-delay: 0.28s; }

    @keyframes overlayBlurIn {
      from {
        backdrop-filter: blur(0px);
        -webkit-backdrop-filter: blur(0px);
        background-color: rgba(4, 10, 13, 0);
      }
      to {
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        background-color: rgba(4, 10, 13, 0.56);
      }
    }

    @keyframes overlayBlurOut {
      from {
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        background-color: rgba(4, 10, 13, 0.56);
      }
      to {
        backdrop-filter: blur(0px);
        -webkit-backdrop-filter: blur(0px);
        background-color: rgba(4, 10, 13, 0);
      }
    }

    @keyframes settingsCardIn {
      from { opacity: 0; transform: translateY(8px) scale(0.98); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes settingsCardOut {
      from { opacity: 1; transform: translateY(0) scale(1); }
      to { opacity: 0; transform: translateY(6px) scale(0.98); }
    }

    button, input, select {
      font: inherit;
    }

    .shell {
      width: min(980px, calc(100vw - 28px));
      margin: auto;
      padding: 0;
    }

    .shade {
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: rgba(0,0,0,0.24);
      z-index: -1;
    }

    .title {
      margin: 0 0 26px;
      text-align: center;
      font-size: clamp(2.7rem, 5vw, 4.6rem);
      font-weight: 300;
      letter-spacing: 0.5px;
      text-shadow: 0 8px 40px rgba(0,0,0,.45);
    }

    .close-button {
      position: fixed;
      top: 14px;
      left: 14px;
      width: 40px;
      height: 40px;
      display: grid;
      place-items: center;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.14);
      color: var(--text);
      font-size: 18px;
      margin: 0;
      box-shadow: 0 10px 30px rgba(0,0,0,.25);
      backdrop-filter: blur(10px) saturate(120%);
      -webkit-backdrop-filter: blur(10px) saturate(120%);
      z-index: 20;
    }

    .close-button:hover {
      background: rgba(255,255,255,0.22);
    }

    .controls {
      display: flex;
      gap: 18px;
      align-items: stretch;
    }

    .glass {
      background: var(--card);
      backdrop-filter: blur(12px) saturate(110%);
      -webkit-backdrop-filter: blur(12px) saturate(110%);
      border: 1px solid var(--border);
      box-shadow: 0 8px 26px rgba(0,0,0,.28);
    }

    .glass-light {
      background: rgba(255,255,255,0.12);
      backdrop-filter: blur(10px) saturate(110%);
      -webkit-backdrop-filter: blur(10px) saturate(110%);
      border: 1px solid rgba(255,255,255,0.18);
      box-shadow: 0 8px 26px rgba(0,0,0,.28);
    }

    .dropzone {
      flex: 1 1 auto;
      min-height: 292px;
      padding: 26px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      gap: 16px;
      border-radius: 28px;
      text-align: center;
      cursor: pointer;
      transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }

    .dropzone:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 32px rgba(0,0,0,.32);
    }

    .dropzone.dragging {
      transform: translateY(-2px);
      border-color: rgba(255,255,255,.16);
    }

    .dropzone svg {
      width: 54px;
      height: 54px;
      color: var(--icon);
    }

    .dropzone h3 {
      margin: 0;
      font-size: 1.25rem;
      font-weight: 400;
      letter-spacing: 0;
    }

    .dropzone p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 1rem;
    }

    .dropzone .note {
      font-size: 0.88rem;
    }

    .hidden-input {
      display: none;
    }

    .controls-side {
      width: 270px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .split-card {
      position: relative;
      border-radius: 28px;
      padding: 20px 18px 18px;
      color: var(--text);
    }

    .split-head {
      display: flex;
      justify-content: flex-end;
      align-items: flex-start;
      margin-bottom: 10px;
    }

    .icon-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 34px;
      height: 34px;
      padding: 0;
      border-radius: 12px;
      border: 0;
      background: transparent;
      color: var(--text);
    }

    .icon-button:hover {
      background: rgba(255,255,255,0.08);
    }

    .icon-button svg {
      width: 18px;
      height: 18px;
    }

    .modes {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .mode-card {
      position: relative;
      display: flex;
      gap: 10px;
      align-items: flex-start;
      cursor: pointer;
      padding: 2px 0;
    }

    .mode-card input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }

    .mode-card:hover {
      opacity: 0.92;
    }

    .mode-card .checkbox {
      width: 18px;
      height: 18px;
      border-radius: 6px;
      border: 1.5px solid rgba(255,255,255,.35);
      margin-top: 2px;
      display: grid;
      place-items: center;
      flex-shrink: 0;
    }

    .mode-card.active .checkbox {
      background: linear-gradient(135deg, var(--accent-soft), #5fa3b5);
      border-color: transparent;
    }

    .mode-card.active .checkbox::after {
      content: "";
      width: 6px;
      height: 10px;
      border: 2px solid #0F2027;
      border-top: 0;
      border-left: 0;
      transform: rotate(45deg);
    }

    .mode-card strong {
      display: block;
      font-size: 0.95rem;
      margin-bottom: 0;
    }

    .mode-card span {
      display: block;
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.45;
    }

    .small-note {
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
    }

    .small-note strong {
      color: var(--text);
      font-weight: 700;
    }

    .warning {
      display: none;
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      background: rgba(245,122,109,0.12);
      border: 1px solid rgba(245,122,109,0.16);
      color: #ffd7d2;
      font-size: 0.85rem;
      line-height: 1.45;
    }

    .warning.show {
      display: flex;
    }

    .warning-text {
      flex: 1 1 280px;
    }

    .warning-action {
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.08);
      color: inherit;
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
    }

    .warning-action:hover {
      background: rgba(255,255,255,0.14);
    }

    .start-button {
      border-radius: 28px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
      color: white;
      text-align: center;
      min-height: 88px;
    }

    .start-button .start-icon {
      font-size: 1.15rem;
      line-height: 1;
    }

    .start-button .start-text {
      font-size: 0.92rem;
      font-weight: 700;
    }

    .queue-panel {
      margin-top: 18px;
      padding: 0;
      background: transparent;
      border: 0;
      box-shadow: none;
      backdrop-filter: none;
    }

    .queue-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }

    .queue-head h2 {
      margin: 0;
      font-size: 1rem;
      font-weight: 600;
      letter-spacing: 0;
    }

    .queue-head p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.5;
    }

    .queue-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .queue-item {
      padding: 16px 18px;
      border-radius: 22px;
      background: var(--card);
      border: 1px solid var(--border);
      box-shadow: 0 8px 26px rgba(0,0,0,.28);
      backdrop-filter: blur(12px) saturate(110%);
      -webkit-backdrop-filter: blur(12px) saturate(110%);
    }

    .queue-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
    }

    .queue-main {
      flex: 1 1 auto;
      min-width: 0;
    }

    .queue-name {
      margin: 0;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      word-break: break-word;
    }

    .queue-subline {
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      background: rgba(255,255,255,0.1);
      color: var(--muted);
      margin-top: 10px;
    }

    .status-badge.status-running,
    .status-badge.status-queued,
    .status-badge.status-uploading {
      color: var(--accent-strong);
      border: 1px solid rgba(181,255,216,0.18);
    }

    .status-badge.status-done {
      color: #dfffe5;
      background: rgba(106, 189, 128, 0.12);
    }

    .status-badge.status-error,
    .status-badge.status-stopped {
      color: #ffd7d2;
      background: rgba(245,122,109,0.12);
    }

    .queue-stage {
      margin-top: 12px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.4;
    }

    .queue-stage.progress-only {
      justify-content: flex-end;
    }

    .queue-stage strong {
      color: var(--text);
      font-weight: 700;
    }

    .progress-shell {
      margin-top: 10px;
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      border-radius: inherit;
      width: 100%;
      transform: scaleX(0);
      transform-origin: left center;
      transition: transform 0.52s cubic-bezier(.22,.61,.36,1);
      will-change: transform;
      background: linear-gradient(90deg, #76cfba 0%, #baf7d8 55%, #e0fff3 100%);
      box-shadow: inset 0 0 16px rgba(255,255,255,0.22);
    }

    .queue-actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      min-width: 160px;
    }

    .button {
      border-radius: 14px;
      padding: 10px 14px;
      font-weight: 700;
    }

    .button:hover { transform: translateY(-1px); }
    .button:disabled { cursor: not-allowed; opacity: 0.5; transform: none; }

    .button.primary {
      background: linear-gradient(135deg, #89dbc2, #baffde);
      color: #081116;
      box-shadow: 0 14px 34px rgba(132, 213, 191, 0.22);
    }

    .button.secondary {
      background: rgba(255,255,255,0.07);
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.08);
    }

    .button.ghost {
      background: transparent;
      color: var(--muted);
      border: 1px solid rgba(255,255,255,0.08);
    }

    .button.danger {
      background: rgba(245,122,109,0.12);
      color: #ffd7d2;
      border: 1px solid rgba(245,122,109,0.16);
    }

    .empty {
      padding: 34px 20px;
      text-align: center;
      color: var(--muted);
      border-radius: 20px;
      border: 1px dashed rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.02);
    }

    .modal-shell {
      position: fixed;
      inset: 0;
      display: none;
      place-items: center;
      background: rgba(4, 10, 13, 0.56);
      backdrop-filter: blur(10px);
      padding: 24px;
    }

    .modal-shell.open {
      display: grid;
      animation: overlayBlurIn .28s ease both;
    }

    .modal-shell.closing {
      display: grid;
      animation: overlayBlurOut .18s ease both;
    }

    .modal {
      width: min(460px, 100%);
      padding: 24px;
    }

    .settings-card-in {
      animation: settingsCardIn .28s cubic-bezier(.2,.7,.2,1) both;
    }

    .settings-card-out {
      animation: settingsCardOut .18s ease both;
    }

    .modal h3 {
      margin: 0;
      font-size: 1.35rem;
      letter-spacing: -0.04em;
    }

    .modal p {
      color: var(--muted);
      line-height: 1.5;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 18px;
    }

    .field label {
      font-size: 0.88rem;
      color: var(--muted);
      text-transform: lowercase;
      letter-spacing: 0.08em;
    }

    .field select {
      width: 100%;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      background: rgba(255,255,255,0.05);
      color: var(--text);
      padding: 14px;
    }

    .modal-actions {
      margin-top: 22px;
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }

    @media (max-width: 900px) {
      .controls {
        flex-direction: column;
      }
      .controls-side {
        width: 100%;
      }
      .queue-row {
        flex-direction: column;
      }
      .queue-actions {
        width: 100%;
        justify-content: flex-start;
      }
    }

    @media (max-width: 620px) {
      .shell {
        width: min(100vw, calc(100vw - 18px));
      }
      body {
        padding: 40px 0;
      }
      .button {
        width: 100%;
        justify-content: center;
      }
    }
  </style>
</head>
<body>
  <div class="shade"></div>
  <button id="close-button" class="close-button" type="button" aria-label="Quit">×</button>

  <main class="shell">
    <h1 class="title fade-in delay-1">stemsplat</h1>

    <section class="controls">
      <label id="dropzone" class="dropzone glass fade-in delay-2" for="file-input" role="button" tabindex="0">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 15V9m0 0l3 3m-3-3L9 12m3 9a9 9 0 110-18 9 9 0 010 18z"></path>
        </svg>
        <div>
          <h3>drop songs here</h3>
          <p>or click to choose files</p>
        </div>
        <input id="file-input" class="hidden-input" type="file" accept="audio/*" multiple>
      </label>

      <div class="controls-side">
        <section class="split-card glass fade-in delay-3">
          <div class="split-head">
            <button id="settings-button" class="icon-button" type="button" aria-label="Open settings">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33A1.65 1.65 0 0 0 9 3.09V3a2 2 0 1 1 4 0v.09c0 .65.38 1.24.97 1.51a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06c-.47.47-.61 1.18-.33 1.82.27.6.86.97 1.51.97H21a2 2 0 1 1 0 4h-.09c-.65 0-1.24.38-1.51.97Z"></path>
              </svg>
            </button>
          </div>

          <div class="modes" id="mode-picker">
            <label class="mode-card active" data-mode="vocals">
              <input type="radio" name="split-mode" value="vocals" checked>
              <div class="checkbox"></div>
              <div>
                <strong>vocals</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="instrumental">
              <input type="radio" name="split-mode" value="instrumental">
              <div class="checkbox"></div>
              <div>
                <strong>instrumental</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="both_deux">
              <input type="radio" name="split-mode" value="both_deux">
              <div class="checkbox"></div>
              <div>
                <strong>both (deux)</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="both_separate">
              <input type="radio" name="split-mode" value="both_separate">
              <div class="checkbox"></div>
              <div>
                <strong>both (separate)</strong>
              </div>
            </label>
          </div>

        </section>

        <button id="start-button" class="start-button glass-light fade-in delay-4" type="button" disabled>start</button>
        <div id="models-warning" class="warning">
          <span id="models-warning-text" class="warning-text"></span>
          <button id="models-folder-button" class="warning-action" type="button" hidden>open models folder</button>
        </div>
      </div>
    </section>

    <section id="queue-panel" class="queue-panel fade-in delay-4">
      <div id="empty-state" class="empty">nothing queued yet.</div>
      <div id="queue" class="queue-list"></div>
    </section>
  </main>

  <div id="settings-modal" class="modal-shell" aria-hidden="true">
    <div class="modal glass" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <h3 id="settings-title">settings</h3>
      <p>Choose the format for exported stems.</p>
      <div class="field">
        <label for="output-format">output format</label>
        <select id="output-format">
          <option value="same_as_input">same as input</option>
          <option value="mp3_320">320kb mp3</option>
          <option value="mp3_128">128kb mp3</option>
          <option value="wav">wav</option>
          <option value="m4a">m4a</option>
          <option value="flac">flac</option>
        </select>
      </div>
      <div class="modal-actions">
        <button id="settings-cancel" class="button ghost" type="button">cancel</button>
        <button id="settings-save" class="button primary" type="button">save</button>
      </div>
    </div>
  </div>

  <script>
    const MODE_LABELS = {
      vocals: 'vocals',
      instrumental: 'instrumental',
      both_deux: 'both (deux)',
      both_separate: 'both (separate)',
    };

    const OUTPUT_LABELS = {
      same_as_input: 'same as input',
      mp3_320: '320kb mp3',
      mp3_128: '128kb mp3',
      wav: 'wav',
      m4a: 'm4a',
      flac: 'flac',
    };

    const queueEl = document.getElementById('queue');
    const queuePanelEl = document.getElementById('queue-panel');
    const emptyStateEl = document.getElementById('empty-state');
    const startButton = document.getElementById('start-button');
    const fileInput = document.getElementById('file-input');
    const dropzone = document.getElementById('dropzone');
    const settingsButton = document.getElementById('settings-button');
    const settingsModal = document.getElementById('settings-modal');
    const settingsCancel = document.getElementById('settings-cancel');
    const settingsSave = document.getElementById('settings-save');
    const outputFormatSelect = document.getElementById('output-format');
    const closeButton = document.getElementById('close-button');
    const modelsWarning = document.getElementById('models-warning');
    const modelsWarningText = document.getElementById('models-warning-text');
    const modelsFolderButton = document.getElementById('models-folder-button');
    const settingsCard = settingsModal.querySelector('.modal');

    const settings = {
      output_format: localStorage.getItem('stemsplat.output_format') || 'same_as_input',
    };

    let selectedMode = 'vocals';
    let tasks = [];
    let startBusy = false;
    let settingsCloseTimer = null;
    let missingModels = [];
    let modelsDir = '';

    function missingForMode(mode) {
      if (mode === 'vocals') return missingModels.includes('vocals') ? ['vocals'] : [];
      if (mode === 'instrumental') return missingModels.includes('instrumental') ? ['instrumental'] : [];
      if (mode === 'both_deux') return missingModels.includes('deux') ? ['deux'] : [];
      if (mode === 'both_separate') {
        return ['vocals', 'instrumental'].filter((name) => missingModels.includes(name));
      }
      return [];
    }

    function pendingMissingModels() {
      const missing = new Set();
      tasks
        .filter((task) => task.status === 'pending')
        .forEach((task) => missingForMode(task.mode).forEach((name) => missing.add(name)));
      return Array.from(missing);
    }

    function updateOutputSummary() {
      outputFormatSelect.value = settings.output_format;
    }

    function setMode(mode) {
      selectedMode = mode;
      document.querySelectorAll('.mode-card').forEach((card) => {
        const active = card.dataset.mode === mode;
        card.classList.toggle('active', active);
        const input = card.querySelector('input');
        if (input) input.checked = active;
      });
      showModelsWarning(missingModels);
    }

    function humanEta(seconds) {
      if (!Number.isFinite(seconds) || seconds <= 0) return '';
      const mins = Math.floor(seconds / 60);
      const secs = seconds % 60;
      if (mins >= 60) {
        const hours = Math.floor(mins / 60);
        const remMins = mins % 60;
        return `${hours}h ${remMins}m remaining`;
      }
      if (mins > 0) return `${mins}m ${secs}s remaining`;
      return `${secs}s remaining`;
    }

    function statusLabel(status) {
      if (status === 'queued') return 'queued';
      if (status === 'running') return 'running';
      if (status === 'done') return 'done';
      if (status === 'error') return 'error';
      if (status === 'stopped') return 'stopped';
      if (status === 'uploading') return 'uploading';
      return 'ready';
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function isTerminal(task) {
      return ['done', 'error', 'stopped'].includes(task.status);
    }

    function makeLocalTask(file) {
      return {
        localId: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        id: null,
        file,
        name: file.name,
        mode: selectedMode,
        output_format: settings.output_format,
        status: 'pending',
        stage: 'ready',
        pct: 0,
        eta_seconds: null,
        out_dir: null,
        outputs: [],
        error: null,
        eventSource: null,
        abortController: null,
        removed: false,
      };
    }

    function updateStartButton() {
      const pendingCount = tasks.filter((task) => task.status === 'pending').length;
      startButton.disabled = startBusy || pendingCount === 0 || pendingMissingModels().length > 0;
      startButton.textContent = startBusy ? 'starting...' : 'start';
    }

    function updateQueueSummary() {
      const queueSummaryEl = document.getElementById('queue-summary');
      if (!queueSummaryEl) {
        return;
      }
      if (tasks.length === 0) {
        queueSummaryEl.textContent = 'add songs, then press start.';
        return;
      }
      const pending = tasks.filter((task) => task.status === 'pending').length;
      const active = tasks.filter((task) => ['queued', 'running', 'uploading'].includes(task.status)).length;
      const done = tasks.filter((task) => task.status === 'done').length;
      queueSummaryEl.textContent = `${tasks.length} song${tasks.length === 1 ? '' : 's'} • ${pending} waiting • ${active} working • ${done} done`;
    }

    function renderQueue() {
      queueEl.innerHTML = '';
      if (queuePanelEl) {
        queuePanelEl.style.display = tasks.length === 0 ? 'none' : 'block';
      }
      emptyStateEl.style.display = 'none';

      tasks.forEach((task) => {
        const item = document.createElement('article');
        item.className = 'queue-item';

        const pct = Math.max(0, Math.min(100, task.pct || 0));
        const eta = humanEta(task.eta_seconds);
        const stageText = task.error ? task.error : '';
        const modeLabel = MODE_LABELS[task.mode] || task.mode;
        const outputLabel = task.output_format === 'same_as_input'
          ? ''
          : (OUTPUT_LABELS[task.output_format] || task.output_format);
        const queueMeta = outputLabel ? `${modeLabel} • ${outputLabel}` : modeLabel;
        const progressSide = `${pct}%${eta ? ` • ${eta}` : ''}`;
        const badgeText = statusLabel(task.status);
        const showStatusBadge = ['error', 'stopped'].includes(task.status)
          || (task.status === 'done' && stageText.trim().toLowerCase() !== badgeText);
        const statusBadgeHtml = showStatusBadge
          ? `<div class="status-badge status-${task.status}">${badgeText}</div>`
          : '';
        const stageHtml = stageText
          ? `
              <div class="queue-stage">
                <strong>${escapeHtml(stageText)}</strong>
                <span>${escapeHtml(progressSide)}</span>
              </div>
            `
          : `
              <div class="queue-stage progress-only">
                <span>${escapeHtml(progressSide)}</span>
              </div>
            `;

        item.innerHTML = `
          <div class="queue-row">
            <div class="queue-main">
              <p class="queue-name">${escapeHtml(task.name)}</p>
              <div class="queue-subline">${escapeHtml(queueMeta)}</div>
              ${statusBadgeHtml}
              ${stageHtml}
              <div class="progress-shell">
                <div class="progress-fill" style="transform:scaleX(${pct / 100})"></div>
              </div>
            </div>
            <div class="queue-actions" data-actions="${task.localId}">
            </div>
          </div>
        `;

        const actions = item.querySelector('.queue-actions');
        if (task.status === 'pending') {
          const remove = document.createElement('button');
          remove.className = 'button ghost';
          remove.textContent = 'remove';
          remove.addEventListener('click', () => {
            task.removed = true;
            tasks = tasks.filter((entry) => entry.localId !== task.localId);
            renderQueue();
          });
          actions.appendChild(remove);
        } else {
          if (!task.id) {
            const remove = document.createElement('button');
            remove.className = 'button ghost';
            remove.textContent = 'remove';
            remove.addEventListener('click', () => {
              task.removed = true;
              if (task.abortController) {
                task.abortController.abort();
                task.abortController = null;
              }
              tasks = tasks.filter((entry) => entry.localId !== task.localId);
              renderQueue();
            });
            actions.appendChild(remove);
          }

          if (['queued', 'running', 'uploading'].includes(task.status) && task.id) {
            const stop = document.createElement('button');
            stop.className = 'button danger';
            stop.textContent = 'stop';
            stop.disabled = task.status === 'uploading';
            stop.addEventListener('click', async () => {
              try {
                await fetch(`/api/tasks/${task.id}/stop`, { method: 'POST' });
              } catch (error) {
                console.error(error);
              }
            });
            actions.appendChild(stop);
          }

          if (task.status === 'done' && task.id) {
            const reveal = document.createElement('button');
            reveal.className = 'button secondary';
            reveal.textContent = (task.outputs || []).length === 1 ? 'show song' : 'show files';
            reveal.addEventListener('click', async () => {
              try {
                await fetch(`/api/tasks/${task.id}/reveal`, { method: 'POST' });
              } catch (error) {
                console.error(error);
              }
            });
            actions.appendChild(reveal);
          }

          if (task.id && isTerminal(task)) {
            const retry = document.createElement('button');
            retry.className = 'button ghost';
            retry.textContent = 'retry';
            retry.addEventListener('click', async () => {
              try {
                const res = await fetch(`/api/tasks/${task.id}/retry`, { method: 'POST' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.message || data.detail?.message || 'Retry failed');
                if (task.eventSource) task.eventSource.close();
                Object.assign(task, {
                  id: data.id,
                  status: data.status,
                  stage: data.stage,
                  pct: data.pct,
                  eta_seconds: data.eta_seconds,
                  out_dir: data.out_dir,
                  outputs: data.outputs,
                  error: data.error,
                });
                subscribeToTask(task);
                renderQueue();
              } catch (error) {
                console.error(error);
              }
            });
            actions.appendChild(retry);
          }
        }

        queueEl.appendChild(item);
      });

      showModelsWarning(missingModels);
      updateQueueSummary();
      updateStartButton();
    }

    function subscribeToTask(task) {
      if (!task.id) return;
      if (task.eventSource) task.eventSource.close();
      const source = new EventSource(`/api/tasks/${task.id}/events`);
      task.eventSource = source;
      source.onmessage = (event) => {
        const data = JSON.parse(event.data);
        Object.assign(task, {
          id: data.id,
          status: data.status,
          stage: data.stage,
          pct: data.pct,
          eta_seconds: data.eta_seconds,
          out_dir: data.out_dir,
          outputs: data.outputs || [],
          error: data.error,
        });
        renderQueue();
        if (isTerminal(task)) {
          source.close();
          task.eventSource = null;
        }
      };
      source.onerror = () => {
        if (isTerminal(task)) {
          source.close();
          task.eventSource = null;
        }
      };
    }

    async function addFiles(fileList) {
      const incoming = Array.from(fileList || []).filter(Boolean);
      fileInput.value = '';
      if (incoming.length === 0) return;
      incoming.forEach((file) => tasks.push(makeLocalTask(file)));
      renderQueue();
    }

    async function startPending() {
      const pending = tasks.filter((task) => task.status === 'pending' && task.file);
      if (pending.length === 0) return;
      if (pendingMissingModels().length > 0) {
        renderQueue();
        return;
      }
      startBusy = true;
      updateStartButton();
      try {
        for (const task of pending) {
          if (task.removed || !tasks.includes(task)) {
            continue;
          }
          try {
            task.status = 'uploading';
            task.stage = 'uploading';
            task.pct = 0;
            task.error = null;
            renderQueue();

            const body = new FormData();
            body.append('file', task.file);
            body.append('mode', task.mode);
            body.append('output_format', task.output_format);
            task.abortController = new AbortController();

            const res = await fetch('/api/tasks', { method: 'POST', body, signal: task.abortController.signal });
            const data = await res.json();
            task.abortController = null;
            if (!res.ok) {
              throw new Error(data.message || data.detail?.message || 'Upload failed');
            }
            if (task.removed || !tasks.includes(task)) {
              if (data.id) {
                fetch(`/api/tasks/${data.id}/stop`, { method: 'POST' }).catch(() => {});
              }
              continue;
            }

            task.file = null;
            task.id = data.id;
            task.status = data.status;
            task.stage = data.stage;
            task.pct = data.pct;
            task.eta_seconds = data.eta_seconds;
            task.out_dir = data.out_dir;
            task.outputs = data.outputs || [];
            task.error = data.error;
            subscribeToTask(task);
            renderQueue();
          } catch (error) {
            task.abortController = null;
            if (error?.name === 'AbortError' || task.removed) {
              continue;
            }
            task.status = 'error';
            task.stage = 'error';
            task.error = error?.message || 'Upload failed';
            renderQueue();
          }
        }
      } finally {
        startBusy = false;
        updateStartButton();
      }
    }

    function openSettings() {
      if (settingsCloseTimer) {
        clearTimeout(settingsCloseTimer);
        settingsCloseTimer = null;
      }
      settingsModal.classList.add('open');
      settingsModal.classList.remove('closing');
      settingsModal.setAttribute('aria-hidden', 'false');
      if (settingsCard) {
        settingsCard.classList.remove('settings-card-out');
        void settingsCard.offsetWidth;
        settingsCard.classList.add('settings-card-in');
      }
      outputFormatSelect.value = settings.output_format;
    }

    function closeSettings() {
      if (!settingsModal.classList.contains('open')) return;
      settingsModal.classList.add('closing');
      settingsModal.setAttribute('aria-hidden', 'true');
      if (settingsCard) {
        settingsCard.classList.remove('settings-card-in');
        settingsCard.classList.add('settings-card-out');
      }
      settingsCloseTimer = window.setTimeout(() => {
        settingsModal.classList.remove('open', 'closing');
        if (settingsCard) {
          settingsCard.classList.remove('settings-card-out');
        }
        settingsCloseTimer = null;
      }, 180);
    }

    function showModelsWarning(missing, nextModelsDir = modelsDir) {
      missingModels = Array.isArray(missing) ? [...missing] : [];
      modelsDir = nextModelsDir || modelsDir || '';
      const relevantMissing = pendingMissingModels().length > 0 ? pendingMissingModels() : missingForMode(selectedMode);
      if (relevantMissing.length === 0) {
        modelsWarning.classList.remove('show');
        modelsWarningText.textContent = '';
        modelsFolderButton.hidden = true;
        updateStartButton();
        return;
      }
      modelsWarning.classList.add('show');
      modelsWarningText.textContent = modelsDir
        ? `missing models: ${relevantMissing.join(', ')}. add them to ${modelsDir} before starting.`
        : `missing models: ${relevantMissing.join(', ')}. add them to the models folder before starting.`;
      modelsFolderButton.hidden = !modelsDir;
      updateStartButton();
    }

    async function loadModelsWarning() {
      try {
        const res = await fetch('/api/models_status');
        if (!res.ok) return;
        const data = await res.json();
        showModelsWarning(data.missing || [], data.models_dir || '');
      } catch (error) {
        console.error(error);
      }
    }

    fileInput.addEventListener('change', (event) => addFiles(event.target.files));
    dropzone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        fileInput.click();
      }
    });
    dropzone.addEventListener('dragover', (event) => {
      event.preventDefault();
      dropzone.classList.add('dragging');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
    dropzone.addEventListener('drop', (event) => {
      event.preventDefault();
      dropzone.classList.remove('dragging');
      addFiles(event.dataTransfer.files);
    });

    document.querySelectorAll('.mode-card').forEach((card) => {
      card.addEventListener('click', () => setMode(card.dataset.mode));
    });

    modelsFolderButton.addEventListener('click', async () => {
      try {
        await fetch('/api/open_models_folder', { method: 'POST' });
      } catch (error) {
        console.error(error);
      }
    });
    document.querySelectorAll('.mode-card input').forEach((input) => {
      input.addEventListener('change', (event) => setMode(event.target.value));
    });

    startButton.addEventListener('click', startPending);

    settingsButton.addEventListener('click', openSettings);
    settingsCancel.addEventListener('click', closeSettings);
    settingsSave.addEventListener('click', () => {
      settings.output_format = outputFormatSelect.value;
      localStorage.setItem('stemsplat.output_format', settings.output_format);
      updateOutputSummary();
      closeSettings();
    });

    settingsModal.addEventListener('click', (event) => {
      if (event.target === settingsModal) closeSettings();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && settingsModal.classList.contains('open')) {
        closeSettings();
      }
    });

    closeButton.addEventListener('click', async () => {
      try {
        await fetch('/shutdown', { method: 'POST', keepalive: true });
      } catch (error) {
        console.error(error);
      }
      setTimeout(() => {
        try { window.close(); } catch (_) {}
        window.location.replace('about:blank');
      }, 250);
    });

    updateOutputSummary();
    setMode(selectedMode);
    renderQueue();
    loadModelsWarning();
  </script>
</body>
</html>
"""


if __name__ == "__main__":  # pragma: no cover
    cli_main()
