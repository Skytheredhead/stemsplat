"""Minimal placeholder splitter module."""

try:
    from packaging import version  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - packaging optional
    from distutils.version import LooseVersion as _LooseVersion  # type: ignore

    class _CompatVersion:
        @staticmethod
        def parse(v):
            return _LooseVersion(v)

    version = _CompatVersion()  # type: ignore

import argparse
import torchaudio
import soundfile as _sf
from pathlib import Path as _Path


def _noop(frac: float) -> None:
    pass


def main(argv=None, progress_cb=None):
    """Copy input audio to vocals.wav so the pipeline can proceed."""
    progress_cb = progress_cb or _noop

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ckpt")
    p.add_argument("--config")
    p.add_argument("--wav", required=True)
    p.add_argument("--out", default="stems_out")
    p.add_argument("--segment", type=int, default=352_800)
    p.add_argument("--overlap", type=int, default=18)
    p.add_argument("--device", default="cpu")
    p.add_argument("--vocals-only", action="store_true")
    args, _unknown = p.parse_known_args(argv)

    wav, sr = torchaudio.load(args.wav)
    out_dir = _Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_cb(0.0)
    _sf.write(out_dir / "vocals.wav", wav.T, sr)
    progress_cb(1.0)
    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
