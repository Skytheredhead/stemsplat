"""Lightweight torchaudio stub used for tests.

If the real ``torchaudio`` package is available this module proxies all
attributes to it so the application can use the full library when
installed. Otherwise a minimal implementation is provided for unit tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_here = Path(__file__).resolve()

for p in list(sys.path)[1:]:
    spec = importlib.util.find_spec("torchaudio", [p])
    if spec and Path(getattr(spec, "origin", "")).resolve() != _here:
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader
        spec.loader.exec_module(mod)
        globals().update(mod.__dict__)
        sys.modules[__name__] = mod
        break
else:
    import wave, struct

    __all__ = ["save", "load", "functional"]

    def _to_list(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        return x

    def save(path, waveform, sample_rate, encoding=None):
        arr = _to_list(waveform)
        if arr and not isinstance(arr[0], (list, tuple)):
            arr = [arr]
        n_channels = len(arr)
        n_samples = len(arr[0]) if n_channels else 0
        frames = bytearray()
        for i in range(n_samples):
            for ch in range(n_channels):
                val = int(max(-1.0, min(1.0, float(arr[ch][i]))) * 32767)
                frames += struct.pack('<h', val)
        with wave.open(str(path), 'wb') as w:
            w.setnchannels(n_channels)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(frames)

    def load(path):
        with wave.open(str(path), 'rb') as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            frames = w.readframes(w.getnframes())
        samples = struct.iter_unpack('<h', frames)
        data = [[0.0] * (len(frames) // 2 // nch) for _ in range(nch)]
        idx = 0
        for s in samples:
            data[idx % nch][idx // nch] = s[0] / 32768.0
            idx += 1
        return data, sr

    class functional:
        @staticmethod
        def resample(wav, sr, new_sr):
            return wav
