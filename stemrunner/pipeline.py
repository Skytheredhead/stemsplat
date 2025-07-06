from pathlib import Path
from typing import Callable, Optional
import tempfile
import subprocess
import os
from array import array
import torch
import torchaudio

from split.split import main as split_main

SEGMENT = 352800
OVERLAP = 18


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
    ckpt: Path,
    outdir: str | None = None,
    progress_cb: Optional[Callable[[str, int], None]] = None,
):
    """Run the Roformer splitter and output only the vocal stem."""
    if progress_cb is None:
        progress_cb = lambda stage, pct: None

    out_dir = Path(outdir or path.parent) / f"{path.stem}—stems"
    tmp = None
    wav_path = path
    if path.suffix.lower() != ".wav":
        tmp = _convert_to_wav(path, 44100)
        wav_path = tmp
    args = [
        "--ckpt", str(ckpt),
        "--config", str(Path(__file__).resolve().parents[1] / "configs" / "Mel Band Roformer Vocals Config.yaml"),
        "--wav", str(wav_path),
        "--out", str(out_dir),
        "--segment", str(SEGMENT),
        "--overlap", str(OVERLAP),
        "--device", "mps" if torch.backends.mps.is_available() else "cpu",
        "--vocals-only",
    ]

    def cb(frac: float):
        progress_cb("vocals", int(frac * 100))

    progress_cb("preparing", 0)
    split_main(args, progress_cb=cb)
    if tmp is not None:
        Path(tmp).unlink(missing_ok=True)
    progress_cb("done", 100)
