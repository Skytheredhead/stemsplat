from pathlib import Path
from typing import Callable, Optional
from array import array

from split.split import main as split_main

SEGMENT = 352800
OVERLAP = 18




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
    wav_path = path
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
    progress_cb("done", 100)
