"""Compatibility pipeline wrapper using the unified main.py implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from main import _process_local_file

SEGMENT = 352_800
OVERLAP = 12


def process_file(
    path: Path,
    ckpt: Path | None = None,
    outdir: str | None = None,
    progress_cb: Optional[Callable[[str, int], None]] = None,
):
    stems = ["vocals"]
    out_paths = _process_local_file(path, stems)
    if progress_cb:
        progress_cb("done", 100)
    return out_paths
