from pathlib import Path
from typing import Callable, Optional
import tempfile
import subprocess
import os
from array import array
import torch
import torchaudio

from .models import ModelManager

SEGMENT_STAGE_A = 352800
SEGMENT_STAGE_B = 4000
OVERLAP = 8


def _convert_to_wav(path: Path, sample_rate: int) -> Path:
    """Convert an audio file to WAV using ffmpeg and return the new path."""
    fd, tmp_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    tmp = Path(tmp_path)
    try:
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-i',
                str(path),
                '-ar',
                str(sample_rate),
                '-ac',
                '2',
                '-sample_fmt',
                's16',
                str(tmp),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError('ffmpeg not found') from exc
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError('ffmpeg failed to convert audio') from exc
    return tmp


def _load_waveform(path: Path, sample_rate: int):
    """Load audio returning (waveform, sample_rate) with ffmpeg fallback."""
    try:
        return torchaudio.load(path)
    except Exception:
        try:
            result = subprocess.run(
                [
                    'ffmpeg',
                    '-y',
                    '-i',
                    str(path),
                    '-f',
                    's16le',
                    '-ac',
                    '2',
                    '-ar',
                    str(sample_rate),
                    'pipe:1',
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError('ffmpeg not found') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('ffmpeg failed to decode audio') from exc
        data = torch.frombuffer(result.stdout, dtype=torch.int16)
        if data.numel() == 0:
            waveform = torch.zeros(2, 0, dtype=torch.float32)
        else:
            waveform = data.view(-1, 2).t().to(torch.float32) / 32768.0
        return waveform, sample_rate


def _save_waveform(path: Path, waveform: torch.Tensor, sample_rate: int):
    """Save audio using torchaudio with an FFmpeg fallback."""
    try:
        torchaudio.save(path, waveform, sample_rate, encoding='PCM_S24LE')
        return
    except Exception:
        try:
            data = waveform.t().contiguous().cpu().numpy().astype('float32').tobytes()
        except Exception:
            arr = array('f', waveform.t().contiguous().cpu().reshape(-1).tolist())
            data = arr.tobytes()
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-y',
                    '-f', 'f32le',
                    '-ar', str(sample_rate),
                    '-ac', str(waveform.shape[0]),
                    '-i', 'pipe:0',
                    '-c:a', 'pcm_s24le',
                    str(path),
                ],
                check=True,
                input=data,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError('ffmpeg not found') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('ffmpeg failed to encode audio') from exc


def process_file(
    path: Path,
    manager: ModelManager,
    segment: int | None = None,
    outdir: str | None = None,
    progress_cb: Optional[Callable[[str, int], None]] = None,
    delay: float = 0.0,
):
    """Process a single audio file and save stems."""
    if progress_cb is None:
        progress_cb = lambda stage, pct: None
    sample_rate = 44100
    temp_path = None
    load_path = path
    progress_cb('preparing', 0)
    if path.suffix.lower() != '.wav':
        temp_path = _convert_to_wav(path, sample_rate)
        load_path = temp_path
    try:
        waveform, sr = _load_waveform(load_path, sample_rate)
    except Exception as exc:
        msg = (
            f'failed to load {path.name}: {exc}. '\
            'Ensure ffmpeg and torchaudio are installed with WAV support.'
        )
        raise RuntimeError(msg) from exc
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    progress_cb('preparing', 10)
    out_dir = Path(outdir or path.parent) / f"{path.stem}—stems"
    out_dir.mkdir(exist_ok=True)

    # Stage A - Vocals (10%-50%)
    stage_start, stage_end = 10, 50
    progress_cb('vocals', stage_start)
    def vocals_cb(pct):
        total = stage_start + int((stage_end - stage_start) * pct / 100)
        progress_cb('vocals', total)
    vocals, instrumental = manager.split_vocals(
        waveform,
        segment or SEGMENT_STAGE_A,
        OVERLAP,
        progress_cb=vocals_cb,
        delay=delay,
    )
    progress_cb('vocals', stage_end)
    _save_waveform(out_dir / f"{path.stem}—Vocals.wav", vocals, sample_rate)
    _save_waveform(out_dir / f"{path.stem}—Instrumental.wav", instrumental, sample_rate)

    # Stage B - Stems (50%-100%)
    stage_start, stage_end = 50, 100
    progress_cb('stems', stage_start)
    def stems_cb(pct):
        total = stage_start + int((stage_end - stage_start) * pct / 100)
        progress_cb('stems', total)
    drums, bass, other, karaoke, guitar = manager.split_instrumental(
        instrumental,
        SEGMENT_STAGE_B,
        OVERLAP,
        progress_cb=stems_cb,
        delay=delay,
    )
    progress_cb('stems', stage_end)
    _save_waveform(out_dir / f"{path.stem}—Drums.wav", drums, sample_rate)
    _save_waveform(out_dir / f"{path.stem}—Bass.wav", bass, sample_rate)
    _save_waveform(out_dir / f"{path.stem}—Other.wav", other, sample_rate)
    _save_waveform(out_dir / f"{path.stem}—Karaoke.wav", karaoke, sample_rate)
    _save_waveform(out_dir / f"{path.stem}—Guitar.wav", guitar, sample_rate)
    progress_cb('done', 100)
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)
