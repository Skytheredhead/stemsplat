from pathlib import Path
from typing import Callable, Optional
import time
import torch
import torchaudio

def _select_device(gpu: Optional[int]) -> torch.device:
    """Return best available device. Prefers CUDA when gpu provided,
    otherwise auto-detects CUDA, then Metal (MPS), else CPU."""
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    if gpu is None:
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return torch.device('mps')
    return torch.device('cpu')

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
CONFIGS_DIR = Path(__file__).resolve().parent.parent / 'configs'


class ModelManager:
    """Load and manage model checkpoints."""

    def __init__(self, gpu: Optional[int] = None):
        self.device = _select_device(gpu)
        self.model_info = {
            'vocals': ('Mel Band Roformer Vocals.ckpt', 'Mel Band Roformer Vocals Config.yaml'),
            'instrumental': ('Mel Band Roformer Instrumental.ckpt', 'Mel Band Roformer Instrumental Config.yaml'),
            'drums': ('kuielab_a_drums.onnx', None),
            'bass': ('kuielab_a_bass.onnx', None),
            'other': ('kuielab_a_other.onnx', None),
            'karaoke': ('mel_band_roformer_karaoke_becruily.ckpt', None),
            'guitar': ('becruily_guitar.ckpt', 'config_guitar_becruily.yaml'),
        }
        self.refresh_models()

    def refresh_models(self):
        """Reload any missing model checkpoints from disk."""
        for name, (model_fname, cfg_fname) in self.model_info.items():
            setattr(self, f"{name}_model", self._load_path(MODELS_DIR / model_fname))
            if cfg_fname:
                setattr(self, f"{name}_config", self._load_path(CONFIGS_DIR / cfg_fname))

    def missing_models(self):
        self.refresh_models()
        missing = []
        for name, (model_fname, cfg_fname) in self.model_info.items():
            if getattr(self, f"{name}_model") is None:
                missing.append(model_fname)
            elif cfg_fname and getattr(self, f"{name}_config") is None:
                missing.append(cfg_fname)
        return missing

    def _load_path(self, path: Path):
        """Return the path if it exists."""
        if path.exists():
            return path
        return None

    def split_vocals(
        self,
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[int], None]] = None,
        delay: float = 0.0,
    ):
        sr = 44100
        length = waveform.shape[1]
        chunk = max(1, length // 10)
        vocals = torch.empty_like(waveform)
        instrumental = torch.empty_like(waveform)
        for i in range(0, length, chunk):
            end = min(i + chunk, length)
            seg = waveform[:, i:end]
            v = torchaudio.functional.highpass_biquad(seg, sr, 1000)
            vocals[:, i:end] = v
            instrumental[:, i:end] = seg - v
            pct = int((end / length) * 100)
            if progress_cb:
                progress_cb(pct)
            if delay:
                time.sleep(delay)
        return vocals, instrumental

    def split_instrumental(
        self,
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[int], None]] = None,
        delay: float = 0.0,
    ):
        sr = 44100
        length = waveform.shape[1]
        chunk = max(1, length // 10)
        drums = torch.empty_like(waveform)
        bass = torch.empty_like(waveform)
        other = torch.empty_like(waveform)
        karaoke = torch.empty_like(waveform)
        guitar = torch.empty_like(waveform)
        for i in range(0, length, chunk):
            end = min(i + chunk, length)
            seg = waveform[:, i:end]
            d = torchaudio.functional.highpass_biquad(seg, sr, 1500)
            b = torchaudio.functional.lowpass_biquad(seg, sr, 250)
            o = seg - d - b
            k = seg.clone()
            g = torchaudio.functional.bandpass_biquad(seg, sr, 600, 0.707)
            drums[:, i:end] = d
            bass[:, i:end] = b
            other[:, i:end] = o
            karaoke[:, i:end] = k
            guitar[:, i:end] = g
            pct = int((end / length) * 100)
            if progress_cb:
                progress_cb(pct)
            if delay:
                time.sleep(delay)
        return drums, bass, other, karaoke, guitar
