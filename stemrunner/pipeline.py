from pathlib import Path
from typing import Callable, Optional
import tempfile
import subprocess
import os
from array import array

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
    """Placeholder load helper (overridden in tests)."""
    raise RuntimeError('not implemented')


def _save_waveform(path: Path, waveform, sample_rate: int):
    """Placeholder save helper (overridden in tests)."""
    raise RuntimeError('not implemented')


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
    device = "cpu"
    try:
        import torch
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = 'mps'
    except Exception:
        pass

    args = [
        "--ckpt", str(ckpt),
        "--config", str(Path(__file__).resolve().parents[1] / "configs" / "Mel Band Roformer Vocals Config.yaml"),
        "--wav", str(wav_path),
        "--out", str(out_dir),
        "--segment", str(SEGMENT),
        "--overlap", str(OVERLAP),
        "--device", device,
        "--vocals-only",
    ]

    def cb(frac: float):
        progress_cb("vocals", int(frac * 100))

    progress_cb("preparing", 0)
    split_main(args, progress_cb=cb)
    if tmp is not None:
        Path(tmp).unlink(missing_ok=True)
    progress_cb("done", 100)
