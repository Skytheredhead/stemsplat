"""Unified entrypoint for stemsplat.

This module consolidates the server, CLI, and model handling logic into a
single location while focusing on Apple Silicon (Metal) execution. Detailed
error codes are emitted for every failure so issues can be diagnosed quickly.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
import argparse
import asyncio
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
import zipfile

import torch
import soundfile as sf
import torchaudio
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

# Optional heavy imports guarded for clarity
try:  # noqa: SIM105 - deliberate broad guard with explicit error codes
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ort = None

# “split” is a package; main() lives in split/split.py
try:  # noqa: SIM105 - deliberate broad guard with explicit error codes
    from split.split import load_model as _load_roformer
    from split.split import main as split_main
except Exception as _exc:  # pragma: no cover - capture import issues for error reporting
    _load_roformer = None  # type: ignore
    split_main = None  # type: ignore
    _import_error = _exc


class ErrorCode(str, Enum):
    TORCH_MISSING = "E001"
    MPS_UNAVAILABLE = "E002"
    SPLIT_IMPORT_FAILED = "E003"
    MODEL_MISSING = "E004"
    CONFIG_MISSING = "E005"
    ONNX_RUNTIME_MISSING = "E006"
    UPLOAD_FAILED = "E007"
    AUDIO_LOAD_FAILED = "E008"
    RESAMPLE_FAILED = "E009"
    FFMPEG_MISSING = "E010"
    SEPARATION_FAILED = "E011"
    ZIP_FAILED = "E012"
    TASK_NOT_FOUND = "E013"
    INVALID_REQUEST = "E014"
    RERUN_PREREQ_MISSING = "E015"


@dataclass
class AppError(Exception):
    code: ErrorCode
    message: str

    def to_http(self, status: int = 400) -> HTTPException:
        return HTTPException(status_code=status, detail={"code": self.code, "message": self.message})


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"
SEGMENT = 352_800
OVERLAP = 12

MODEL_URLS = [
    (
        "Mel Band Roformer Vocals.ckpt",
        "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true",
    ),
    (
        "Mel Band Roformer Instrumental.ckpt",
        "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true",
    ),
    (
        "mel_band_roformer_karaoke_becruily.ckpt",
        "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt?download=true",
    ),
    (
        "becruily_guitar.ckpt",
        "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt?download=true",
    ),
    (
        "kuielab_a_bass.onnx",
        "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_bass.onnx?download=true",
    ),
    (
        "kuielab_a_drums.onnx",
        "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_drums.onnx?download=true",
    ),
    (
        "kuielab_a_other.onnx",
        "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_other.onnx?download=true",
    ),
]


# ── Device handling ─────────────────────────────────────────────────────────

def _ensure_torch() -> None:
    if not hasattr(torch, "backends"):
        raise AppError(ErrorCode.TORCH_MISSING, "PyTorch is unavailable; install torch with Metal support.")


def select_device() -> torch.device:
    """Prefer Metal (MPS) on Apple Silicon; fall back to CPU with explicit warning."""
    _ensure_torch()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    raise AppError(ErrorCode.MPS_UNAVAILABLE, "Apple Metal (mps) device is unavailable; ensure torch==2.x with mps support.")


# ── Model management ────────────────────────────────────────────────────────


def _load_optional_roformer(path: Path, config_path: Optional[Path], device: torch.device):
    if _load_roformer is None:
        raise AppError(ErrorCode.SPLIT_IMPORT_FAILED, f"split package unavailable: {_import_error}")
    return _load_roformer(str(path), str(config_path or ""), device)


class StemModel:
    def __init__(self, path: Path | None, device: torch.device, config_path: Path | None = None):
        self.device = device
        self.kind = "none"
        self.session = None
        self.net = None
        self.path = path
        self.config_path = config_path
        self.expects_waveform = False
        if path is None:
            return
        if path.suffix.lower() == ".ckpt":
            self.net = _load_optional_roformer(path, config_path, device)
            self.net.eval()
            self.kind = "roformer"
            self.expects_waveform = True
            return
        if path.suffix.lower() == ".onnx":
            if ort is None:
                raise AppError(ErrorCode.ONNX_RUNTIME_MISSING, "onnxruntime is required for ONNX models on Apple Silicon (CPU mode).")
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.kind = "onnx"
            return
        self.net = torch.jit.load(str(path), map_location=device)
        self.net.eval()
        self.kind = "torchscript"

    def __call__(self, mag: "torch.Tensor") -> "torch.Tensor":
        if self.kind == "onnx" and self.session is not None:
            inp_name = self.session.get_inputs()[0].name
            out = self.session.run(None, {inp_name: mag.detach().cpu().numpy()})[0]
            return torch.from_numpy(out).to(mag.device)
        if self.net is None:
            raise AppError(ErrorCode.MODEL_MISSING, "Model session not initialized.")
        with torch.no_grad():
            return self.net(mag)


class ModelManager:
    def __init__(self):
        self.device = select_device()
        self.model_info = {
            "vocals": ("Mel Band Roformer Vocals.ckpt", "Mel Band Roformer Vocals Config.yaml"),
            "instrumental": ("Mel Band Roformer Instrumental.ckpt", "Mel Band Roformer Instrumental Config.yaml"),
            "drums": ("kuielab_a_drums.onnx", None),
            "bass": ("kuielab_a_bass.onnx", None),
            "other": ("kuielab_a_other.onnx", None),
            "karaoke": ("mel_band_roformer_karaoke_becruily.ckpt", None),
            "guitar": ("becruily_guitar.ckpt", "config_guitar_becruily.yaml"),
        }
        self.refresh_models()

    def _resolve_path(self, filename: str) -> Path:
        preferred = MODEL_DIR / filename
        if preferred.exists():
            return preferred
        fallback = Path.home() / "Library/Application Support/stems" / filename
        if fallback.exists():
            return fallback
        raise AppError(ErrorCode.MODEL_MISSING, f"Model file missing: {filename}")

    def _resolve_config(self, filename: Optional[str]) -> Optional[Path]:
        if not filename:
            return None
        cfg = CONFIG_DIR / filename
        if cfg.exists():
            return cfg
        raise AppError(ErrorCode.CONFIG_MISSING, f"Config file missing: {filename}")

    def refresh_models(self) -> None:
        for name, (model_fname, cfg_fname) in self.model_info.items():
            model_path = self._resolve_path(model_fname)
            cfg_path = self._resolve_config(cfg_fname)
            setattr(self, name, StemModel(model_path, self.device, cfg_path))
            if cfg_fname:
                setattr(self, f"{name}_config", cfg_path)

    # Separation helpers
    def split_vocals(
        self,
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[float], None]] = None,
        delay: float = 0.0,
    ):
        if isinstance(waveform, torch.Tensor):
            working = waveform.to(self.device)
        else:
            working = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        length = working.shape[1]
        step = max(1, segment - overlap)
        voc_acc = torch.zeros_like(working, device=self.device)
        inst_acc = torch.zeros_like(working, device=self.device)
        counts = torch.zeros((1, length), device=self.device)
        if progress_cb:
            progress_cb(0.0)
        for start in range(0, length, step):
            end = min(start + segment, length)
            seg = working[:, start:end]
            if seg.shape[1] < segment:
                padded = torch.zeros((working.shape[0], segment), device=self.device)
                padded[:, : seg.shape[1]] = seg
            else:
                padded = seg
            with torch.no_grad():
                pred = self.vocals(padded.unsqueeze(0))[0]
            if pred.dim() == 2:
                pred = pred[:, None, :]
            elif pred.dim() == 3 and pred.shape[1] not in (1, working.shape[0]):
                pred = pred.permute(1, 0, 2)
            pred = pred[..., : seg.shape[1]]
            voc_seg = pred[0]
            inst_seg = pred[1] if pred.shape[0] > 1 else seg - voc_seg
            sl = voc_seg.shape[1]
            voc_acc[:, start : start + sl] += voc_seg
            inst_acc[:, start : start + sl] += inst_seg
            counts[:, start : start + sl] += 1
            if progress_cb:
                progress_cb(end / length)
            if delay:
                time.sleep(delay)
        denom = counts.clamp_min(1)
        return voc_acc / denom, inst_acc / denom

    def split_pair_with_model(
        self,
        model: StemModel,
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[float], None]] = None,
        delay: float = 0.0,
    ):
        if isinstance(waveform, torch.Tensor):
            working = waveform.to(self.device)
        else:
            working = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        length = working.shape[1]
        step = max(1, segment - overlap)
        out0 = torch.zeros_like(working, device=self.device)
        out1 = torch.zeros_like(working, device=self.device)
        counts = torch.zeros((1, length), device=self.device)
        if progress_cb:
            progress_cb(0.0)

        if model.expects_waveform:
            for start in range(0, length, step):
                end = min(start + segment, length)
                seg = working[:, start:end]
                if seg.shape[1] < segment:
                    padded = torch.zeros((working.shape[0], segment), device=self.device)
                    padded[:, : seg.shape[1]] = seg
                else:
                    padded = seg
                with torch.no_grad():
                    pred = model(padded.unsqueeze(0))[0]
                if pred.dim() == 2:
                    pred = pred[:, None, :]
                elif pred.dim() == 3 and pred.shape[1] not in (1, working.shape[0]):
                    pred = pred.permute(1, 0, 2)
                pred = pred[..., : seg.shape[1]]
                stem0 = pred[0]
                stem1 = pred[1] if pred.shape[0] > 1 else seg - stem0
                sl = stem0.shape[1]
                out0[:, start : start + sl] += stem0
                out1[:, start : start + sl] += stem1
                counts[:, start : start + sl] += 1
                if progress_cb:
                    progress_cb(end / length)
                if delay:
                    time.sleep(delay)
            denom = counts.clamp_min(1)
            return out0 / denom, out1 / denom

        # STFT mask path
        n_fft = 2048
        hop = 441
        win = torch.hann_window(n_fft, device=self.device)
        for start in range(0, length, step):
            end = min(start + segment, length)
            seg = working[:, start:end]
            if seg.shape[1] < n_fft:
                out0[:, start:end] += seg
                if progress_cb:
                    progress_cb(end / length)
                continue
            spec = torch.stft(seg, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
            mag = spec.abs().unsqueeze(0)
            mask = model(mag)[0]
            stem0_spec = spec * mask
            stem0 = torch.istft(
                stem0_spec,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=win,
                length=seg.shape[1],
            )
            out0[:, start : start + stem0.shape[1]] += stem0
            out1[:, start : start + stem0.shape[1]] += seg[:, : stem0.shape[1]] - stem0
            if progress_cb:
                progress_cb(end / length)
            if delay:
                time.sleep(delay)
        return out0, out1

    def split_instrumental(
        self,
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[float], None]] = None,
        delay: float = 0.0,
    ):
        if isinstance(waveform, torch.Tensor):
            working = waveform.to(self.device)
        else:
            working = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        length = working.shape[1]
        step = max(1, segment - overlap)
        n_fft = 2048
        hop = 441
        win = torch.hann_window(n_fft, device=self.device)
        drums = torch.zeros_like(working, device=self.device)
        bass = torch.zeros_like(working, device=self.device)
        other = torch.zeros_like(working, device=self.device)
        karaoke = torch.zeros_like(working, device=self.device)
        guitar = torch.zeros_like(working, device=self.device)
        if progress_cb:
            progress_cb(0.0)
        for start in range(0, length, step):
            end = min(start + segment, length)
            seg = working[:, start:end]
            if seg.shape[1] < n_fft:
                for tensor in (drums, bass, other, karaoke, guitar):
                    tensor[:, start:end] += seg * 0
                if progress_cb:
                    progress_cb(end / length)
                continue
            spec = torch.stft(seg, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
            mag = spec.abs().unsqueeze(0)
            masks = {
                "drums": self.drums(mag)[0],
                "bass": self.bass(mag)[0],
                "other": self.other(mag)[0],
                "karaoke": self.karaoke(mag)[0],
                "guitar": self.guitar(mag)[0],
            }
            stems = {
                "drums": drums,
                "bass": bass,
                "other": other,
                "karaoke": karaoke,
                "guitar": guitar,
            }
            for name, mask in masks.items():
                spec_masked = spec * mask
                stem = torch.istft(
                    spec_masked,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=n_fft,
                    window=win,
                    length=seg.shape[1],
                )
                stems[name][:, start : start + stem.shape[1]] += stem
            if progress_cb:
                progress_cb(end / length)
            if delay:
                time.sleep(delay)
        return drums, bass, other, karaoke, guitar


# ── Task orchestration ─────────────────────────────────────────────────────-

app = FastAPI()
progress: dict[str, dict[str, int | str]] = {}
errors: dict[str, str] = {}
tasks: dict[str, dict] = {}
controls: dict[str, dict[str, threading.Event]] = {}
process_queue: queue.Queue[Callable[[], None]] = queue.Queue()

def _worker() -> None:
    while True:
        fn = process_queue.get()
        try:
            fn()
        finally:
            process_queue.task_done()

threading.Thread(target=_worker, daemon=True).start()

download_lock = threading.Lock()
downloading = False


def _download_models() -> None:
    global downloading
    with download_lock:
        if downloading:
            return
        downloading = True
    MODEL_DIR.mkdir(exist_ok=True)
    for name, url in MODEL_URLS:
        dest = MODEL_DIR / name
        try:
            import urllib.request

            with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    out.write(chunk)
        except Exception as exc:  # pragma: no cover - network errors
            logging.error("model download failed: %s", exc)
    downloading = False


@app.post("/download_models")
async def download_models():
    thread = threading.Thread(target=_download_models, daemon=True)
    thread.start()
    return {"status": "started"}


# ── Separation helpers ─────────────────────────────────────────────────────-

def _prepare_waveform(audio_path: Path) -> torch.Tensor:
    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception as exc:
        raise AppError(ErrorCode.AUDIO_LOAD_FAILED, f"Failed to load audio: {exc}") from exc
    if sr != 44100:
        try:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        except Exception as exc:
            raise AppError(ErrorCode.RESAMPLE_FAILED, f"Resample failed: {exc}") from exc
    return waveform

def _ensure_ffmpeg() -> None:
    import subprocess

    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise AppError(ErrorCode.FFMPEG_MISSING, "ffmpeg not found; install ffmpeg and ensure it is on PATH.") from exc
    except Exception as exc:
        raise AppError(ErrorCode.FFMPEG_MISSING, f"ffmpeg check failed: {exc}") from exc


def _convert_to_wav(audio_path: Path, *, remove_source: bool = False) -> Path:
    """Convert ``audio_path`` to WAV in-place and optionally drop the original copy."""

    _ensure_ffmpeg()

    if audio_path.suffix.lower() in {".wav", ".wave"}:
        return audio_path

    wav_path = audio_path.with_suffix(".wav")
    cmd = ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "44100", "-ac", "2", "-vn", str(wav_path)]
    try:
        import subprocess

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise AppError(ErrorCode.FFMPEG_MISSING, "ffmpeg not found; install ffmpeg and ensure it is on PATH.") from exc
    except Exception as exc:
        raise AppError(ErrorCode.UPLOAD_FAILED, f"Conversion to WAV failed: {exc}") from exc

    if remove_source and wav_path != audio_path:
        try:
            audio_path.unlink()
        except Exception:
            logging.warning("failed to remove source %s after conversion", audio_path)
    return wav_path


def _stage_audio_copy(src: Path, dest_dir: Path, *, remove_original_copy: bool = True) -> Path:
    """Place a copy of ``src`` into ``dest_dir`` and normalize it to WAV."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = dest_dir / src.name
    if staged != src:
        shutil.copy2(src, staged)
    return _convert_to_wav(staged, remove_source=remove_original_copy)


def convert_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    """Convert all supported audio files in ``input_dir`` to WAV under ``output_dir``."""

    _ensure_ffmpeg()

    if not input_dir.exists() or not input_dir.is_dir():
        raise AppError(ErrorCode.INVALID_REQUEST, f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted: list[Path] = []
    supported = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".alac", ".opus", ".m4a"}

    for src_path in input_dir.rglob("*"):
        if src_path.suffix.lower() not in supported:
            continue
        relative_subpath = src_path.relative_to(input_dir)
        staged_dest = output_dir / relative_subpath
        staged_dest.parent.mkdir(parents=True, exist_ok=True)
        if staged_dest != src_path:
            shutil.copy2(src_path, staged_dest)
        wav_path = _convert_to_wav(staged_dest, remove_source=True)
        converted.append(wav_path)
    return converted


def _write_wave(out_dir: Path, fname: str, tensor: torch.Tensor, sr: int = 44100) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / fname, tensor.T.cpu().numpy(), sr)


def _separate_waveform(
    manager: "ModelManager",
    audio_path: Path,
    stem_list: list[str],
    cb: Callable[[str, int], None],
    out_dir: Path,
) -> list[str]:
    waveform = _prepare_waveform(audio_path)
    cb("load_audio.done", 2)

    temp_dir = Path(tempfile.gettempdir())
    stems_out: list[str] = []

    need_vocal_pass = ("vocals" in stem_list) or any(s in stem_list for s in ["drums", "bass", "other", "guitar"])
    need_instrumental_model = "instrumental" in stem_list

    inst_path: Optional[Path] = None

    if need_vocal_pass:
        def _cb(frac: float) -> None:
            cb("vocals", 1 + int(frac * 99))

        cb("split_vocals.start", 2)
        voc, inst = manager.split_vocals(waveform, SEGMENT, OVERLAP, progress_cb=_cb)
        cb("split_vocals.done", 50)
        if "vocals" in stem_list:
            fname = f"{audio_path.stem} - vocals.wav"
            _write_wave(out_dir, fname, voc)
            stems_out.append(fname)
            cb("write.vocals", 52)
        need_inst = any(s in stem_list for s in ["drums", "bass", "other", "guitar"])
        if need_inst:
            inst_path = temp_dir / f"{uuid.uuid4()}_inst.wav"
            _write_wave(temp_dir, inst_path.name, inst)
        del voc, inst

    if need_instrumental_model:
        cb("instrumental.start", 52)
        inst_model = manager.instrumental
        if inst_path is None:
            inst_path = temp_dir / f"{uuid.uuid4()}_inst.wav"
            waveform_np = waveform.cpu().numpy()
            _write_wave(temp_dir, inst_path.name, torch.from_numpy(waveform_np))
        inst_wave = _prepare_waveform(inst_path)
        inst_pred = inst_model(inst_wave)
        fname = f"{audio_path.stem} - instrumental.wav"
        _write_wave(out_dir, fname, inst_pred)
        stems_out.append(fname)
        cb("instrumental.done", 75)

    for stem_name in ["drums", "bass", "other", "guitar"]:
        if stem_name not in stem_list:
            continue
        cb(f"{stem_name}.start", 75)
        inst_model = getattr(manager, stem_name)
        if inst_path is None:
            inst_path = temp_dir / f"{uuid.uuid4()}_inst.wav"
            waveform_np = waveform.cpu().numpy()
            _write_wave(temp_dir, inst_path.name, torch.from_numpy(waveform_np))
        inst_wave = _prepare_waveform(inst_path)
        pred = inst_model(inst_wave)
        fname = f"{audio_path.stem} - {stem_name}.wav"
        _write_wave(out_dir, fname, pred)
        stems_out.append(fname)
        cb(f"{stem_name}.done", min(98, 75 + len(stems_out)))

    cb("done", 100)
    return stems_out


# ── API endpoints ─────────────────────────────────────────────────────────--

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), stems: str = Form("vocals")):
    task_id = str(uuid.uuid4())
    pause_evt = threading.Event()
    stop_evt = threading.Event()
    controls[task_id] = {"pause": pause_evt, "stop": stop_evt}
    stem_list = [s for s in stems.split(",") if s]
    try:
        path = Path("uploads") / file.filename
        path.parent.mkdir(exist_ok=True)
        with path.open("wb") as f:
            f.write(await file.read())
    except Exception as exc:
        raise AppError(ErrorCode.UPLOAD_FAILED, f"Failed to persist upload: {exc}").to_http()

    conv_dir = Path("uploads_converted")
    conv_dir.mkdir(exist_ok=True)
    conv_path = _stage_audio_copy(path, conv_dir)

    out_dir = conv_dir / f"{Path(file.filename).stem}—stems"
    expected = [f"{Path(file.filename).stem} - {s}.wav" for s in stem_list]

    def cb(stage: str, pct: int):
        while pause_evt.is_set():
            progress[task_id] = {"stage": "paused", "pct": pct}
            time.sleep(0.5)
        if stop_evt.is_set():
            progress[task_id] = {"stage": "stopped", "pct": 0}
            raise AppError(ErrorCode.INVALID_REQUEST, "Task stopped by user")
        progress[task_id] = {"stage": stage, "pct": pct}

    def run() -> None:
        try:
            manager = ModelManager()
            cb("preparing", 0)
            cb("prepare.complete", 1)
            audio_path = conv_path
            stems_out = _separate_waveform(manager, audio_path, stem_list, cb, out_dir)

            zip_path = conv_dir / f"{audio_path.stem}—stems.zip"
            cb("zip.start", 95)
            try:
                with zipfile.ZipFile(zip_path, "w") as zf:
                    for name in stems_out:
                        fp = out_dir / name
                        if fp.exists():
                            zf.write(fp, arcname=name)
            except Exception as exc:
                raise AppError(ErrorCode.ZIP_FAILED, f"Failed to create zip: {exc}") from exc
            cb("zip.done", 98)
            tasks[task_id]["zip"] = zip_path
            tasks[task_id]["stems"] = stems_out
            cb("finalizing", 99)
            progress[task_id] = {"stage": "done", "pct": 100}
        except AppError as exc:
            progress[task_id] = {"stage": "error", "pct": -1}
            errors[task_id] = json.dumps({"code": exc.code, "message": exc.message})
        except Exception as exc:  # pragma: no cover - safety net
            logging.exception("processing failed")
            progress[task_id] = {"stage": "error", "pct": -1}
            errors[task_id] = json.dumps({"code": ErrorCode.SEPARATION_FAILED, "message": str(exc)})

    process_queue.put(run)
    progress[task_id] = {"stage": "queued", "pct": 0}
    tasks[task_id] = {
        "dir": out_dir,
        "stems": expected,
        "controls": controls[task_id],
        "conv_src": str(conv_path),
        "orig_src": str(path),
        "stem_list": stem_list,
    }
    return {"task_id": task_id, "stems": expected}


@app.get("/progress/{task_id}")
async def progress_stream(task_id: str):
    async def event_generator():
        last = None
        while True:
            info = progress.get(task_id, {"stage": "queued", "pct": 0})
            current = (info["stage"], info["pct"])
            if current != last:
                if info.get("stage") == "stopped":
                    yield {"event": "message", "data": json.dumps({"stage": "stopped", "pct": 0})}
                elif info.get("pct", 0) < 0:
                    yield {"event": "error", "data": errors.get(task_id, "processing failed")}
                else:
                    yield {"event": "message", "data": json.dumps({"stage": info["stage"], "pct": info["pct"]})}
                last = current
                if info.get("stage") == "stopped" or info.get("pct", 0) >= 100 or info.get("pct", 0) < 0:
                    break
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.post("/rerun/{task_id}")
async def rerun(task_id: str):
    old = tasks.get(task_id)
    if not old:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id").to_http(404)
    conv_src = Path(old.get("conv_src", ""))
    stem_list = old.get("stem_list") or []
    if not conv_src.exists() or not stem_list:
        raise AppError(ErrorCode.RERUN_PREREQ_MISSING, "Missing source or stems for rerun").to_http(409)

    new_id = str(uuid.uuid4())
    pause_evt = threading.Event()
    stop_evt = threading.Event()
    controls[new_id] = {"pause": pause_evt, "stop": stop_evt}
    conv_dir = conv_src.parent
    out_dir = conv_dir / f"{conv_src.stem}—stems"
    expected = [f"{conv_src.stem} - {s}.wav" for s in stem_list]

    def cb(stage: str, pct: int):
        while pause_evt.is_set():
            progress[new_id] = {"stage": "paused", "pct": pct}
            time.sleep(0.5)
        if stop_evt.is_set():
            progress[new_id] = {"stage": "stopped", "pct": 0}
            raise AppError(ErrorCode.INVALID_REQUEST, "Task stopped by user")
        progress[new_id] = {"stage": stage, "pct": pct}

    def run_again() -> None:
        try:
            manager = ModelManager()
            cb("preparing", 0)
            cb("prepare.complete", 1)
            audio_path = _convert_to_wav(conv_src)
            stems_out = _separate_waveform(manager, audio_path, stem_list, cb, out_dir)

            zip_path = conv_dir / f"{audio_path.stem}—stems.zip"
            cb("zip.start", 95)
            with zipfile.ZipFile(zip_path, "w") as zf:
                for name in stems_out:
                    fp = out_dir / name
                    if fp.exists():
                        zf.write(fp, arcname=name)
            cb("zip.done", 98)
            tasks[new_id] = {**old, "zip": zip_path, "controls": controls[new_id]}
            cb("finalizing", 99)
            progress[new_id] = {"stage": "done", "pct": 100}
        except AppError as exc:
            progress[new_id] = {"stage": "error", "pct": -1}
            errors[new_id] = json.dumps({"code": exc.code, "message": exc.message})
        except Exception as exc:  # pragma: no cover
            logging.exception("rerun failed")
            progress[new_id] = {"stage": "error", "pct": -1}
            errors[new_id] = json.dumps({"code": ErrorCode.SEPARATION_FAILED, "message": str(exc)})

    process_queue.put(run_again)
    progress[new_id] = {"stage": "queued", "pct": 0}
    tasks[new_id] = {
        "dir": out_dir,
        "stems": expected,
        "controls": controls[new_id],
        "conv_src": str(conv_src),
        "orig_src": old.get("orig_src"),
        "stem_list": stem_list,
    }
    return {"task_id": new_id, "stems": expected}


@app.get("/download/{task_id}")
async def download(task_id: str):
    info = tasks.get(task_id)
    if not info:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id").to_http(404)
    zip_path = info.get("zip")
    if zip_path and Path(zip_path).exists():
        return FileResponse(path=zip_path, media_type="application/zip", filename=Path(zip_path).name)
    raise AppError(ErrorCode.INVALID_REQUEST, "Files not ready").to_http(409)


@app.post("/clear_all_uploads")
async def clear_all_uploads():
    dirs = ["uploads", "uploads_converted"]
    for d in dirs:
        dir_path = BASE_DIR / d
        if dir_path.exists():
            for entry in dir_path.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
    return {"status": "cleared"}


@app.post("/pause/{task_id}")
async def pause_task(task_id: str):
    ctrl = controls.get(task_id)
    if not ctrl:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task").to_http(404)
    ctrl["pause"].set()
    return {"status": "paused"}


@app.post("/resume/{task_id}")
async def resume_task(task_id: str):
    ctrl = controls.get(task_id)
    if not ctrl:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task").to_http(404)
    ctrl["pause"].clear()
    return {"status": "resumed"}


@app.post("/stop/{task_id}")
async def stop_task(task_id: str):
    ctrl = controls.get(task_id)
    if not ctrl:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task").to_http(404)
    ctrl["stop"].set()
    return {"status": "stopped"}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "web" / "index.html"
    return HTMLResponse(html_path.read_text())


# ── CLI entrypoint ─────────────────────────────────────────────────────────-


def _process_local_file(path: Path, stem_list: list[str]) -> list[str]:
    conv_dir = Path("uploads_converted")
    conv_path = _stage_audio_copy(path, conv_dir)
    out_dir = conv_dir / f"{conv_path.stem}—stems"

    def cb(stage: str, pct: int) -> None:
        logging.info("%s: %s%%", stage, pct)

    manager = ModelManager()
    stems_out = _separate_waveform(manager, conv_path, stem_list, cb, out_dir)
    return [str(out_dir / stem) for stem in stems_out]

def cli_main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run stemsplat server or process a file")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI server with uvicorn")
    parser.add_argument("files", nargs="*", help="Optional list of audio files to process locally")
    parser.add_argument("--stems", default="vocals", help="Comma-separated stems when processing locally")
    args = parser.parse_args(argv)

    if args.serve:
        import uvicorn

        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
        return

    if not args.files:
        parser.error("No files provided and --serve not set")

    stem_list = [s for s in args.stems.split(",") if s]
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            raise AppError(ErrorCode.UPLOAD_FAILED, f"File not found: {file_path}")
        outputs = _process_local_file(path, stem_list)
        for out in outputs:
            print(out)


if __name__ == "__main__":  # pragma: no cover
    cli_main()
