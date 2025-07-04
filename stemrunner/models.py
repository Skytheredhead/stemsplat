from pathlib import Path
from typing import Optional

import torch

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
CONFIGS_DIR = Path(__file__).resolve().parent.parent / 'configs'


class ModelManager:
    """Load and manage model checkpoints."""

    def __init__(self, gpu: Optional[int] = None):
        self.device = torch.device(f'cuda:{gpu}' if gpu is not None and torch.cuda.is_available() else 'cpu')
        self.vocals_model = self._load_model('mel_band_roformer_vocals_becruily.ckpt')
        self.instrumental_model = self._load_model('mel_band_roformer_instrumental_becruily.ckpt')
        self.drums_model = self._load_model('kuielab_a_drums.onnx')
        self.bass_model = self._load_model('kuielab_a_bass.onnx')
        self.other_model = self._load_model('kuielab_a_other.onnx')
        self.karaoke_model = self._load_model('mel_band_roformer_karaoke_becruily.ckpt')
        self.guitar_model = self._load_model('becruily_guitar.ckpt')

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
