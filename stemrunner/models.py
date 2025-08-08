from pathlib import Path
from typing import Callable, Optional
import time
import math
import yaml
import numpy as np
# Remove project root from sys.path if local torch.py stub exists
import sys, pathlib as _pathlib
_PROJECT_ROOT = _pathlib.Path(__file__).resolve().parents[1]
if (_PROJECT_ROOT / "torch.py").exists():
    sys.path = [p for p in sys.path if p not in ("", str(_PROJECT_ROOT))]
import torch
import torchaudio

# ── Roformer loader ──────────────────────────────────────────────────────
# We may have just *removed* the project root from sys.path to dodge a stub
# torch.py.  That also hides the local “split/” package.  First try the
# import; if it fails, append the project root *after* torch is safely
# imported so it can’t shadow it, then retry.
try:
    from split.split import load_model as _load_roformer
except ModuleNotFoundError:
    import sys as _sys
    _ROOT = str(_PROJECT_ROOT)
    if _ROOT not in _sys.path:
        _sys.path.append(_ROOT)          # add to the tail → torch already loaded
    from split.split import load_model as _load_roformer

# Base project root for model/config resolution
ROOT_DIR = Path(__file__).resolve().parent.parent

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

def _select_device(gpu: Optional[int]) -> torch.device:
    """Return best available device. Prefers Metal (MPS), else CPU."""
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

MODELS_DIR = ROOT_DIR / 'models'
CONFIGS_DIR = ROOT_DIR / 'configs'



class StemModel:
    """Wrapper that loads TorchScript, ONNX, Roformer, or generic checkpoint files."""

    def __init__(self, path: Path | None, device: torch.device, config_path: Path | None = None):
        self.device = device
        self.kind = 'none'
        self.session = None
        self.net = None
        self.path = path
        self.config_path = config_path
        self.expects_waveform = False
        if path is None:
            return
        if path.suffix.lower() == '.ckpt':
            try:
                # build a real model from <ckpt,+yaml>
                self.net = _load_roformer(str(path), str(config_path or ''), device)
                self.net.eval()
                self.kind = 'roformer'          # new sentinel
                self.expects_waveform = True    # takes raw waveform, not mags
                return
            except Exception:
                self.net = None   # fall back to old logic below
        if path.suffix.lower() == '.onnx':
            if ort is None:
                # ONNX models require onnxruntime to run
                self.kind = 'none'
                return
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
                self.kind = 'none'
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

    def __call__(self, mag: 'torch.Tensor') -> 'torch.Tensor':
        if self.kind == 'onnx' and self.session is not None:
            inp_name = self.session.get_inputs()[0].name
            out = self.session.run(None, {inp_name: mag.detach().cpu().numpy()})[0]
            return torch.from_numpy(out).to(mag.device)
        if self.kind == 'torchscript' and self.net is not None:
            with torch.no_grad():
                return self.net(mag)
        if self.kind == 'roformer' and self.net is not None:
            with torch.no_grad():
                return self.net(mag)            # here *mag* is really the raw waveform
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
            cfg_path = CONFIGS_DIR / cfg_fname if cfg_fname else None
            setattr(self, name, StemModel(path, self.device, cfg_path))
            if cfg_fname:
                setattr(self, f"{name}_config", cfg_path)

    def missing_models(self):
        """Return model or config files that are unavailable or unusable."""
        self.refresh_models()
        missing = []
        for name, (model_fname, cfg_fname) in self.model_info.items():
            model_obj = getattr(self, name)
            # Treat generic checkpoint files as valid
            if model_obj.kind not in ('onnx', 'torchscript', 'file', 'roformer'):
                missing.append(model_fname)
            if cfg_fname and getattr(self, f"{name}_config") is None:
                missing.append(cfg_fname)
        return missing

    def _load_path(self, path: Path):
        """Return the path if it exists, else try fallback locations."""
        if path and path.exists():
            return path
        # fallback to user application support (for server uploads)
        alt = Path('models') / path.name
        if alt.exists():
            return alt
        home = Path.home() / 'Library/Application Support/stems' / path.name
        if home.exists():
            return home
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
        # ── Roformer chunked inference ───────────────────────────────
        if getattr(self.vocals, 'expects_waveform', False):
            if isinstance(waveform, np.ndarray):
                waveform = torch.tensor(waveform, dtype=torch.float32)
            waveform = waveform.to(device)
            length = waveform.shape[1]
            step = max(1, segment - overlap)
            vocals_acc = torch.zeros_like(waveform, device=device)
            inst_acc = torch.zeros_like(waveform, device=device)
            counts = torch.zeros((1, length), device=device)
            if progress_cb:
                progress_cb(0.0)
            for start in range(0, length, step):
                end = min(start + segment, length)
                seg = waveform[:, start:end]
                if seg.shape[1] < segment:
                    seg_model = torch.zeros((waveform.shape[0], segment), device=device)
                    seg_model[:, :seg.shape[1]] = seg
                else:
                    seg_model = seg
                with torch.no_grad():
                    pred = self.vocals(seg_model.unsqueeze(0))[0]
                if pred.dim() == 2:
                    pred = pred[:, None, :]
                elif pred.dim() == 3 and pred.shape[1] not in (1, waveform.shape[0]):
                    pred = pred.permute(1, 0, 2)
                pred = pred[..., :seg.shape[1]]
                vocals_seg = pred[0]
                if pred.shape[0] > 1:
                    inst_seg = pred[1]
                else:
                    inst_seg = seg - vocals_seg
                sl = vocals_seg.shape[1]
                vocals_acc[:, start:start+sl] += vocals_seg
                inst_acc[:,   start:start+sl] += inst_seg
                counts[:, start:start+sl]     += 1
                frac = end / length
                if progress_cb:
                    progress_cb(frac)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                if delay:
                    time.sleep(delay)
            denom = counts.clamp_min(1)
            vocals = vocals_acc / denom
            inst = inst_acc / denom
            return vocals.cpu(), inst.cpu()
        # ── existing STFT / mask path continues here ──
        # Ensure tensor input (server may pass NumPy array)
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)
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

    def split_pair_with_model(
        self,
        model: 'StemModel',
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[float], None]] = None,
        delay: float = 0.0,
    ):
        """Generic two-output separation using the provided model.
        Works with Roformer checkpoints (expects raw waveform) and
        mask-based models (expects STFT magnitude). Returns (out0, out1).
        """
        device = self.device
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        waveform = waveform.to(device)
        length = waveform.shape[1]
        step = max(1, segment - overlap)
        if progress_cb:
            progress_cb(0.0)

        # Roformer path: model takes raw waveform and outputs two stems
        if getattr(model, 'expects_waveform', False):
            out0_acc = torch.zeros_like(waveform, device=device)
            out1_acc = torch.zeros_like(waveform, device=device)
            counts = torch.zeros((1, length), device=device)
            for start in range(0, length, step):
                end = min(start + segment, length)
                seg = waveform[:, start:end]
                if seg.shape[1] < segment:
                    seg_model = torch.zeros((waveform.shape[0], segment), device=device)
                    seg_model[:, :seg.shape[1]] = seg
                else:
                    seg_model = seg
                with torch.no_grad():
                    pred = model(seg_model.unsqueeze(0))[0]
                if pred.dim() == 2:
                    pred = pred[:, None, :]
                elif pred.dim() == 3 and pred.shape[1] not in (1, waveform.shape[0]):
                    pred = pred.permute(1, 0, 2)
                pred = pred[..., :seg.shape[1]]
                stem0 = pred[0]
                if pred.shape[0] > 1:
                    stem1 = pred[1]
                else:
                    stem1 = seg - stem0
                sl = stem0.shape[1]
                out0_acc[:, start:start+sl] += stem0
                out1_acc[:, start:start+sl] += stem1
                counts[:, start:start+sl]    += 1
                if progress_cb:
                    progress_cb(end / length)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                if delay:
                    time.sleep(delay)
            denom = counts.clamp_min(1)
            out0 = out0_acc / denom
            out1 = out1_acc / denom
            return out0.cpu(), out1.cpu()

        # Mask-based path: fall back to STFT + mask application
        n_fft = 2048
        hop = 441
        win = torch.hann_window(n_fft, device=device)
        out0 = torch.zeros_like(waveform, device=device)
        out1 = torch.zeros_like(waveform, device=device)
        for start in range(0, length, step):
            end = min(start + segment, length)
            seg = waveform[:, start:end]
            if seg.shape[1] < n_fft:
                # not enough for STFT → pass-through
                out0[:, start:end] += seg
                out1[:, start:end] += seg * 0
                if progress_cb:
                    progress_cb(end / length)
                continue
            spec = torch.stft(seg, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                               window=win, return_complex=True)
            mag = spec.abs().unsqueeze(0)
            mask = model(mag)[0]
            stem0_spec = spec * mask
            stem0 = torch.istft(stem0_spec, n_fft=n_fft, hop_length=hop,
                                 win_length=n_fft, window=win, length=seg.shape[1])
            out0[:, start:start + stem0.shape[1]] += stem0
            out1[:, start:start + stem0.shape[1]] += seg[:, :stem0.shape[1]] - stem0
            if progress_cb:
                progress_cb(end / length)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            if delay:
                time.sleep(delay)
        return out0.cpu(), out1.cpu()

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
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)
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
