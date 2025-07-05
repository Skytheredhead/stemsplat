from pathlib import Path
from typing import Callable, Optional
import time
import math
import yaml
import numpy as np
import torch
import torchaudio

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

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


class StemModel:
    """Wrapper that loads TorchScript, ONNX or generic checkpoint files."""

    def __init__(self, path: Path | None, device: torch.device):
        self.device = device
        self.kind = 'none'
        self.session = None
        self.net = None
        self.path = path
        if path is None:
            return
        if path.suffix.lower() == '.onnx' and ort is not None:
            try:
                providers = (
                    ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    if device.type == 'cuda'
                    else ['CPUExecutionProvider']
                )
                self.session = ort.InferenceSession(str(path), providers=providers)
                self.kind = 'onnx'
                return
            except Exception:
                self.session = None
                self.kind = 'file'
        else:
            try:
                self.net = torch.jit.load(str(path), map_location=device)
                self.net.eval()
                self.kind = 'torchscript'
                return
            except Exception:
                self.net = None
                try:
                    # fall back to loading generic checkpoint just to verify file
                    torch.load(str(path), map_location='cpu')
                    self.kind = 'file'
                except Exception:
                    self.kind = 'file'

    def __call__(self, mag: torch.Tensor) -> torch.Tensor:
        if self.kind == 'onnx' and self.session is not None:
            inp_name = self.session.get_inputs()[0].name
            out = self.session.run(None, {inp_name: mag.detach().cpu().numpy()})[0]
            return torch.from_numpy(out).to(mag.device)
        if self.kind == 'torchscript' and self.net is not None:
            with torch.no_grad():
                return self.net(mag)
        # fallback mask: all-pass
        return mag.new_ones(mag.shape)


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
            path = self._load_path(MODELS_DIR / model_fname)
            setattr(self, name, StemModel(path, self.device))
            if cfg_fname:
                setattr(self, f"{name}_config", self._load_path(CONFIGS_DIR / cfg_fname))

    def missing_models(self):
        """Return a list of model or config files that could not be found."""
        self.refresh_models()
        missing = []
        for name, (model_fname, cfg_fname) in self.model_info.items():
            model_obj = getattr(self, name)
            if model_obj.kind == 'none':
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
        progress_cb: Optional[Callable[[float], None]] = None,
        delay: float = 0.0,
    ):
        sr = 44100
        device = self.device
        waveform = waveform.to(device)
        length = waveform.shape[1]
        step = max(1, segment - overlap)
        vocals = torch.zeros_like(waveform, device=device)
        instrumental = torch.zeros_like(waveform, device=device)
        n_fft = 2048
        hop = 441
        win = torch.hann_window(n_fft, device=device)
        for start in range(0, length, step):
            end = min(start + segment, length)
            seg = waveform[:, start:end]
            if seg.shape[1] < n_fft:
                vocals[:, start:end] += seg
                instrumental[:, start:end] += seg * 0
                frac = end / length
                if progress_cb:
                    progress_cb(frac)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                if delay:
                    time.sleep(delay)
                continue
            spec = torch.stft(seg, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                               window=win, return_complex=True)
            mag = spec.abs().unsqueeze(0)
            mask = self.vocals(mag)[0]
            voc_spec = spec * mask
            voc = torch.istft(voc_spec, n_fft=n_fft, hop_length=hop,
                              win_length=n_fft, window=win, length=seg.shape[1])
            vocals[:, start:start + voc.shape[1]] += voc
            instrumental[:, start:start + voc.shape[1]] += seg[:, :voc.shape[1]] - voc
            frac = end / length
            if progress_cb:
                progress_cb(frac)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if delay:
                time.sleep(delay)
        return vocals.cpu(), instrumental.cpu()

    def split_instrumental(
        self,
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[float], None]] = None,
        delay: float = 0.0,
    ):
        sr = 44100
        device = self.device
        waveform = waveform.to(device)
        length = waveform.shape[1]
        step = max(1, segment - overlap)
        drums = torch.zeros_like(waveform, device=device)
        bass = torch.zeros_like(waveform, device=device)
        other = torch.zeros_like(waveform, device=device)
        karaoke = torch.zeros_like(waveform, device=device)
        guitar = torch.zeros_like(waveform, device=device)
        n_fft = 2048
        hop = 441
        win = torch.hann_window(n_fft, device=device)
        for start in range(0, length, step):
            end = min(start + segment, length)
            seg = waveform[:, start:end]
            if seg.shape[1] < n_fft:
                drums[:, start:end] += seg * 0
                bass[:, start:end] += seg * 0
                other[:, start:end] += seg * 0
                karaoke[:, start:end] += seg * 0
                guitar[:, start:end] += seg * 0
                frac = end / length
                if progress_cb:
                    progress_cb(frac)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                if delay:
                    time.sleep(delay)
                continue
            spec = torch.stft(seg, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                               window=win, return_complex=True)
            mag = spec.abs().unsqueeze(0)
            d_mask = self.drums(mag)[0]
            b_mask = self.bass(mag)[0]
            o_mask = self.other(mag)[0]
            k_mask = self.karaoke(mag)[0]
            g_mask = self.guitar(mag)[0]
            drums_spec = spec * d_mask
            bass_spec = spec * b_mask
            other_spec = spec * o_mask
            karaoke_spec = spec * k_mask
            guitar_spec = spec * g_mask
            drums_seg = torch.istft(drums_spec, n_fft=n_fft, hop_length=hop,
                                   win_length=n_fft, window=win, length=seg.shape[1])
            bass_seg = torch.istft(bass_spec, n_fft=n_fft, hop_length=hop,
                                  win_length=n_fft, window=win, length=seg.shape[1])
            other_seg = torch.istft(other_spec, n_fft=n_fft, hop_length=hop,
                                   win_length=n_fft, window=win, length=seg.shape[1])
            karaoke_seg = torch.istft(karaoke_spec, n_fft=n_fft, hop_length=hop,
                                     win_length=n_fft, window=win, length=seg.shape[1])
            guitar_seg = torch.istft(guitar_spec, n_fft=n_fft, hop_length=hop,
                                    win_length=n_fft, window=win, length=seg.shape[1])
            drums[:, start:start + drums_seg.shape[1]] += drums_seg
            bass[:, start:start + bass_seg.shape[1]] += bass_seg
            other[:, start:start + other_seg.shape[1]] += other_seg
            karaoke[:, start:start + karaoke_seg.shape[1]] += karaoke_seg
            guitar[:, start:start + guitar_seg.shape[1]] += guitar_seg
            frac = end / length
            if progress_cb:
                progress_cb(frac)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if delay:
                time.sleep(delay)
        return (
            drums.cpu(),
            bass.cpu(),
            other.cpu(),
            karaoke.cpu(),
            guitar.cpu(),
        )
