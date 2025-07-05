#!/usr/bin/env python3
"""
split.py – convenience edition
------------------------------
Runs a mel-band Roformer (or any UVR-style) model on a WAV, using
defaults that match the files shipped in the same directory:

  • Mel Band Roformer Vocals.ckpt
  • Mel Band Roformer Vocals Config.yaml

Example
-------
python split.py --wav /path/to/song.wav          # uses built‑in defaults
python split.py --ckpt other.ckpt --config other.yaml --wav song.wav
"""
import argparse, os, yaml, importlib
from pathlib import Path
import numpy as np
import torch, torchaudio, soundfile as sf
from tqdm import tqdm

# --------------------------------------------------------------------------
# Hard‑wired defaults so you don’t need to pass --ckpt / --config each run
# --------------------------------------------------------------------------
DEFAULT_CKPT = Path("/Users/skylarenns/Documents/split/Mel Band Roformer Vocals.ckpt")
DEFAULT_YAML = Path("/Users/skylarenns/Documents/split/Mel Band Roformer Vocals Config.yaml")

# --------------------------------------------------------------------------
def load_model(ckpt_path: str, yaml_path: str, device: torch.device):
    """Build model object from a UVR‑style YAML + CKPT pair."""
    with open(yaml_path, "r") as f:
        # Use the unsafe loader so PyYAML can handle tags like !!python/tuple
        cfg_root = yaml.unsafe_load(f)

    # If the YAML has no 'target', fall back to Mel‑Band Roformer default
    default_target = "mel_band_roformer.MelBandRoformer"

    def find_target(cfg):
        if "target" in cfg:
            target = cfg.pop("target")
            return target, cfg
        raise ValueError("No 'target' in YAML")

    try:
        target_path, kwargs = find_target(cfg_root)
    except ValueError:
        # No 'target' in YAML – assume it's a Mel‑Band Roformer config
        target_path, kwargs = default_target, {"model": cfg_root.get("model", {})}

    module_name, class_name = target_path.rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_name), class_name)

    model = ModelClass(**kwargs.get("model", {}))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


def overlap_add(dst: np.ndarray, seg: np.ndarray, start: int, fade: int):
    """Overlap-add helper for (stems, channels, time)."""
    end = start + seg.shape[-1]

    if fade:
        win   = np.ones(seg.shape[-1], dtype=np.float32)
        ramp  = np.linspace(0, 1, fade, dtype=np.float32)
        win[:fade]  = ramp
        win[-fade:] = ramp[::-1]
        dst[..., start:end] += seg * win          # win broadcasts
    else:
        dst[..., start:end] += seg


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default=str(DEFAULT_CKPT),   help=".ckpt weights")
    parser.add_argument("--config", default=str(DEFAULT_YAML),   help=".yaml model def")
    parser.add_argument("--wav",    required=True,               help="input WAV")
    parser.add_argument("--out",    default="stems_out",         help="output dir")
    parser.add_argument("--segment",type=int, default=352_800,   help="segment size")
    parser.add_argument("--overlap",type=int, default=12,        help="overlap percent")
    parser.add_argument(
        "--device",
        default=("mps" if torch.backends.mps.is_available()
                 else ("cuda" if torch.cuda.is_available() else "cpu")),
        help="compute device: 'mps' (Apple Metal), 'cuda', or 'cpu'"
    )
    args = parser.parse_args()

    # --- Load audio --------------------------------------------------------
    wav, sr = torchaudio.load(args.wav)
    if sr != 44_100:
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        sr = 44_100
    wav = wav.numpy()                              # (channels, T)

    n_channels = wav.shape[0]
    n_samples  = wav.shape[1]

    # allocate (stems, channels, time)
    stems = np.zeros((2, n_channels, n_samples), dtype=np.float32)

    seg_size  = args.segment
    fade_len  = int(seg_size * (args.overlap / 100))
    hop_size  = seg_size - fade_len

    # --- Model -------------------------------------------------------------
    device = torch.device(args.device)
    model  = load_model(args.ckpt, args.config, device)

    # --- Process -----------------------------------------------------------
    with torch.no_grad(), tqdm(total=n_samples, unit="sample") as bar:
        for start in range(0, n_samples, hop_size):
            end = min(start + seg_size, n_samples)
            seg = wav[:, start:end]
            if seg.shape[1] < seg_size:
                seg = np.pad(seg, ((0,0),(0,seg_size - seg.shape[1])))

            pred = model(torch.from_numpy(seg).unsqueeze(0).to(device))
            if isinstance(pred, dict) and "sources" in pred:
                pred = pred["sources"]
            pred = pred.squeeze(0).cpu().numpy()          # (stems?, channels?, time?) or (2, T)

            # ── reshape to (stems, channels, time) ─────────────────────────────
            if pred.ndim == 2:
                # (2, T)  or  (channels, T)
                if pred.shape[0] == 2:           # (2, T)
                    pred = pred[:, np.newaxis, :]
                else:                            # (channels, T)
                    pred = pred[np.newaxis, :, :]
            elif pred.ndim == 3 and pred.shape[1] not in (1, n_channels):
                # (channels, stems, T) -> swap axes
                pred = pred.transpose(1, 0, 2)

            # If still mono but input is stereo, copy L->R
            if pred.shape[1] == 1 and n_channels == 2:
                pred = np.repeat(pred, 2, axis=1)

            # Trim zero-padding from last chunk
            chunk_len = end - start
            if pred.shape[2] > chunk_len:
                pred = pred[:, :, :chunk_len]
            if pred.shape[0] == 1:
                vocals_seg = pred[0]                      # (channels, time)
                mix_seg    = seg[:, :chunk_len]           # original mixture
                inst_seg   = mix_seg - vocals_seg
                pred       = np.stack([vocals_seg, inst_seg], axis=0)

            overlap_add(stems, pred, start, fade_len)
            bar.update(min(hop_size, n_samples - start))

    # --- Normalise windowing ----------------------------------------------
    window_sum = np.ones(n_samples, dtype=np.float32)
    if fade_len:
        win = np.ones(seg_size, dtype=np.float32)
        ramp = np.linspace(0, 1, fade_len, dtype=np.float32)
        win[:fade_len] = ramp
        win[-fade_len:] = ramp[::-1]
        for start in range(0, n_samples, hop_size):
            end = min(start + seg_size, n_samples)
            window_sum[start:end] += win[:end - start]
    stems /= window_sum[None, None, :]

    # --- Write -------------------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "vocals.wav",       stems[0].T, sr)   # (samples, channels)
    sf.write(out_dir / "instrumental.wav", stems[1].T, sr)
    print(f"✓ Done – stems saved to “{out_dir}”")


if __name__ == "__main__":
    main()
