"""Lightweight torch stub used for tests.

If a real ``torch`` package is installed this file will proxy all
attributes to the real library. This allows the repository to include a
minimal stub for unit tests without breaking environments that have the
actual PyTorch installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_here = Path(__file__).resolve()

for p in list(sys.path)[1:]:  # skip current directory
    spec = importlib.util.find_spec("torch", [p])
    if spec and Path(getattr(spec, "origin", "")).resolve() != _here:
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(mod)
        globals().update(mod.__dict__)
        sys.modules[__name__] = mod
        break
else:
    float32 = float
    int16 = int

    def zeros(*shape, dtype=float32):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        if not shape:
            return 0.0
        out = [0.0] * shape[-1]
        for dim in reversed(shape[:-1]):
            out = [out.copy() for _ in range(dim)]
        return out

    def frombuffer(buf, dtype):
        return list(buf)

    def zeros_like(arr, dtype=None):
        if isinstance(arr[0], list):
            return [zeros_like(a, dtype) for a in arr]
        return [0.0 for _ in arr]

    class Device:
        def __init__(self, type):
            self.type = type

    class backends:
        class mps:
            @staticmethod
            def is_available():
                return False

    def device(type):
        return Device(type)
