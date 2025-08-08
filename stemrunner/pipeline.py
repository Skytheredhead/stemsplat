try:
    from packaging import version  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - packaging optional
    from distutils.version import LooseVersion as _LooseVersion  # type: ignore

    class _CompatVersion:
        @staticmethod
        def parse(v):
            return _LooseVersion(v)

    version = _CompatVersion()  # type: ignore

from pathlib import Path
from typing import Callable, Optional

try:
    # “split” is a package; main() lives in split/split.py
    from split.split import main as split_main
except Exception as _exc:  # pragma: no cover - missing heavy splitter
    split_main = None
    _import_error = _exc

SEGMENT = 352_800
OVERLAP = 12


def process_file(
    path: Path,
    ckpt: Path,
    outdir: str | None = None,
    progress_cb: Optional[Callable[[str, int], None]] = None,
):
    """Run the vocal splitter on ``path`` and write stems to ``outdir``."""

    if split_main is None:
        raise RuntimeError(f"split.main could not be imported:\n{_import_error}")

    if progress_cb is None:
        progress_cb = lambda stage, pct: None  # noqa: E731

    out_root = Path(outdir) if outdir else path.parent
    out_dir = out_root / f"{path.stem}—stems"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    try:
        import torch  # local import so dependency is optional
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        elif hasattr(torch, "cuda") and torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass

    args = [
        "--ckpt", str(ckpt),
        "--config",
        str(Path(__file__).resolve().parents[1] / "configs" / "Mel Band Roformer Vocals Config.yaml"),
        "--wav", str(path),
        "--out", str(out_dir),
        "--segment", str(SEGMENT),
        "--overlap", str(OVERLAP),
        "--device", device,
        "--vocals-only",
    ]

    def _cb(frac: float) -> None:
        # Map 0‑100 into the “vocals” stage range (1‑100)
        progress_cb("vocals", 1 + int(frac * 99))

    progress_cb("preparing", 0)
    split_main(args, progress_cb=_cb)
    progress_cb("done", 100)
