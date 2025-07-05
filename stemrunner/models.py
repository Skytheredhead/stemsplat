from pathlib import Path
from typing import Optional
import torch

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
        self.model_files = {
            'vocals': 'mel_band_roformer_vocals_becruily.ckpt',
            'instrumental': 'mel_band_roformer_instrumental_becruily.ckpt',
            'drums': 'kuielab_a_drums.onnx',
            'bass': 'kuielab_a_bass.onnx',
            'other': 'kuielab_a_other.onnx',
            'karaoke': 'mel_band_roformer_karaoke_becruily.ckpt',
            'guitar': 'becruily_guitar.ckpt',
        }
        self.refresh_models()

    def refresh_models(self):
        """Reload any missing model checkpoints from disk."""
        for attr, fname in self.model_files.items():
            setattr(self, f"{attr}_model", self._load_model(fname))

    def missing_models(self):
        self.refresh_models()
        return [name for name in self.model_files if getattr(self, f"{name}_model") is None]

    def _load_model(self, name: str):
        """Return the path to a model checkpoint if it exists."""
        path = MODELS_DIR / name
        if path.exists():
            return path
        return None

    def split_vocals(self, waveform, segment: int, overlap: int):
        # Placeholder: implement real inference here
        vocals = waveform.clone()
        instrumental = waveform.clone()
        return vocals, instrumental

    def split_instrumental(self, waveform, segment: int, overlap: int):
        """Placeholder secondary split producing five stems."""
        drums = waveform.clone()
        bass = waveform.clone()
        other = waveform.clone()
        karaoke = waveform.clone()
        guitar = waveform.clone()
        return drums, bass, other, karaoke, guitar
