from pathlib import Path
from typing import Callable, Optional
import tempfile
import subprocess
import os
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


def process_file(
    path: Path,
    manager: ModelManager,
    segment: int | None = None,
    outdir: str | None = None,
    progress_cb: Optional[Callable[[int], None]] = None,
):
    """Process a single audio file and save stems."""
    if progress_cb is None:
        progress_cb = lambda x: None
    sample_rate = 44100
    temp_path = None
    load_path = path
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
    progress_cb(10)
    out_dir = Path(outdir or path.parent) / f"{path.stem}—stems"
    out_dir.mkdir(exist_ok=True)

    # Stage A
    vocals, instrumental = manager.split_vocals(waveform, segment or SEGMENT_STAGE_A, OVERLAP)
    progress_cb(40)
    torchaudio.save(out_dir / f"{path.stem}—Vocals.wav", vocals, sample_rate, encoding='PCM_S24LE')
    torchaudio.save(out_dir / f"{path.stem}—Instrumental.wav", instrumental, sample_rate, encoding='PCM_S24LE')

    # Stage B
    drums, bass, other, karaoke, guitar = manager.split_instrumental(instrumental, SEGMENT_STAGE_B, OVERLAP)
    progress_cb(70)
    torchaudio.save(out_dir / f"{path.stem}—Drums.wav", drums, sample_rate, encoding='PCM_S24LE')
    torchaudio.save(out_dir / f"{path.stem}—Bass.wav", bass, sample_rate, encoding='PCM_S24LE')
    torchaudio.save(out_dir / f"{path.stem}—Other.wav", other, sample_rate, encoding='PCM_S24LE')
    torchaudio.save(out_dir / f"{path.stem}—Karaoke.wav", karaoke, sample_rate, encoding='PCM_S24LE')
    torchaudio.save(out_dir / f"{path.stem}—Guitar.wav", guitar, sample_rate, encoding='PCM_S24LE')
    progress_cb(100)
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)
