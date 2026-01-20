"""Unified entrypoint for stemsplat.

This module consolidates the server, CLI, and model handling logic into a
single location while focusing on Apple Silicon (Metal) execution. Detailed
error codes are emitted for every failure so issues can be diagnosed quickly.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable, Optional
import argparse
import asyncio
import http.server
import importlib
import importlib.util
import json
import logging
import math
import os
import queue
import shutil
import socket
import socketserver
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import webbrowser
import zipfile

INSTALL_DEPS = [
    "torch",
    "torchaudio",
    "fastapi",
    "uvicorn[standard]",
    "click",
    "sse-starlette",
    "python-multipart",
    "numpy",
    "pyyaml",
    "soundfile",
    "tqdm",
    "einops",
    "rotary_embedding_torch",
    "packaging",
    "beartype>=0.17",
    "librosa",
    "imageio-ffmpeg",
    "certifi",
]


def _in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def pip_path() -> Path:
    if os.name == "nt":
        return Path("venv") / "Scripts" / "pip"
    return Path("venv") / "bin" / "pip"


def python_path() -> Path:
    if os.name == "nt":
        return Path("venv") / "Scripts" / "python"
    return Path("venv") / "bin" / "python"


def _bootstrap_install() -> None:
    if not Path("venv").exists():
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
    subprocess.check_call([str(pip_path()), "install", "--upgrade", "pip"])
    for dep in INSTALL_DEPS:
        subprocess.check_call([str(pip_path()), "install", dep])
    subprocess.check_call([str(python_path()), str(Path(__file__).resolve()), "--install"])
    raise SystemExit(0)


if "--install" in sys.argv and not _in_venv():
    _bootstrap_install()

FILES = [
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true",
        "subdir": "models",
        "filename": "mel_band_roformer_vocals_becruily.ckpt",
        "tag": "vocals",
    },
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true",
        "subdir": "models",
        "filename": "mel_band_roformer_instrumental_becruily.ckpt",
        "tag": "instrumental",
    },
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt?download=true",
        "subdir": "models",
        "filename": "becruily_deux.ckpt",
        "tag": "deux",
    },
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt?download=true",
        "subdir": "models",
        "filename": "mel_band_roformer_karaoke_becruily.ckpt",
        "tag": None,
    },
    {
        "url": "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt?download=true",
        "subdir": "models",
        "filename": "becruily_guitar.ckpt",
        "tag": None,
    },
    {
        "url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_bass.onnx?download=true",
        "subdir": "models",
        "filename": "kuielab_a_bass.onnx",
        "tag": None,
    },
    {
        "url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_drums.onnx?download=true",
        "subdir": "models",
        "filename": "kuielab_a_drums.onnx",
        "tag": None,
    },
    {
        "url": "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/MDXNet/kuielab_a_other.onnx?download=true",
        "subdir": "models",
        "filename": "kuielab_a_other.onnx",
        "tag": None,
    },
]

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from beartype import beartype
from beartype.typing import Callable as BeartypeCallable
from beartype.typing import Optional as BeartypeOptional
from beartype.typing import Tuple as BeartypeTuple
from einops import pack, rearrange, reduce, repeat, unpack
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from librosa import filters
from packaging import version
from rotary_embedding_torch import RotaryEmbedding
from sse_starlette.sse import EventSourceResponse
from tqdm import tqdm
from torch import einsum, nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

def _optional_import(module_name: str):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)


certifi = _optional_import("certifi")
ort = _optional_import("onnxruntime")


if importlib.util.find_spec("torch.nn.attention"):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _sdpa_ctx(config):
        """Map old style flags to new SDPBackend list."""
        backends = []
        if config.enable_flash:
            backends.append(SDPBackend.FLASH_ATTENTION)
        if config.enable_math:
            backends.append(SDPBackend.MATH)
        if config.enable_mem_efficient:
            backends.append(SDPBackend.EFFICIENT_ATTENTION)
        try:
            return sdpa_kernel(backends, set_priority_order=True)
        except TypeError:
            return sdpa_kernel(backends)

else:  # pragma: no cover - fallback for older torch
    from torch.backends.cuda import sdp_kernel as sdpa_kernel  # type: ignore

    def _sdpa_ctx(config):
        return sdpa_kernel(**config._asdict())


FlashAttentionConfig = namedtuple(
    "FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.0,
        flash=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "in order to use flash attention, you must be using pytorch 2.0 or above"

        self.cpu_config = FlashAttentionConfig(False, True, False)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )
        config = self.cuda_config if is_cuda else self.cpu_config
        with _sdpa_ctx(config):
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
        return out

    def forward(self, q, k, v):
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        scale = q.shape[-1] ** -0.5

        if self.flash:
            return self.flash_attn(q, k, v)

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        return out


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        rotary_embed=None,
        flash=True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            flash=flash_attn,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BandSplit(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: BeartypeTuple[int, ...],
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))
            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
    dim_in,
    dim_out,
    dim_hidden=None,
    depth=1,
    activation=nn.Tanh,
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: BeartypeTuple[int, ...],
        depth,
        mlp_expansion_factor=4,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1),
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


class MelBandRoformer(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        num_bands=60,
        dim_head=64,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        flash_attn=True,
        dim_freqs_in=1025,
        sample_rate=44100,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: BeartypeOptional[BeartypeCallable] = None,
        mask_estimator_depth=1,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: BeartypeTuple[int, ...] = (
            4096,
            2048,
            1024,
            512,
            256,
        ),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: BeartypeCallable = torch.hann_window,
        match_input_audio_length=False,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Transformer(
                            depth=time_transformer_depth,
                            rotary_embed=time_rotary_embed,
                            **transformer_kwargs,
                        ),
                        Transformer(
                            depth=freq_transformer_depth,
                            rotary_embed=freq_rotary_embed,
                            **transformer_kwargs,
                        ),
                    ]
                )
            )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )

        _init_window = self.stft_window_fn(device="cpu")
        freqs = torch.stft(
            torch.randn(1, 4096),
            **self.stft_kwargs,
            window=_init_window,
            return_complex=True,
        ).shape[1]

        mel_filter_bank_numpy = filters.mel(
            sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands
        )

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        mel_filter_bank[0][0] = 1.0
        mel_filter_bank[-1, -1] = 1.0

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), (
            "all frequencies need to be covered by all bands for now"
        )

        repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")

        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        freqs_per_bands_with_complex = tuple(
            2 * f * self.audio_channels for f in num_freqs_per_band.tolist()
        )

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim, dim_inputs=freqs_per_bands_with_complex, depth=mask_estimator_depth
            )

            self.mask_estimators.append(mask_estimator)

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized,
        )

        self.match_input_audio_length = match_input_audio_length

    def forward(
        self,
        raw_audio,
        target=None,
        return_loss_breakdown=False,
    ):
        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        batch, channels, raw_audio_length = raw_audio.shape

        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (
            self.stereo and channels == 2
        ), (
            "stereo needs to be set to True if passing in audio signal that is stereo "
            "(channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")

        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(
            raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True
        )
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")
        stft_repr = rearrange(
            stft_repr, "b s f t c -> b (f s) t c"
        )

        batch_arange = torch.arange(batch, device=device)[..., None]

        x = stft_repr[batch_arange, self.freq_indices]

        x = rearrange(x, "b f t c -> b t (f c)")

        x = self.band_split(x)

        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            x = time_transformer(x)

            x, = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            x = freq_transformer(x)

            x, = unpack(x, ps, "* f d")

        num_stems = len(self.mask_estimators)

        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)

        masks = masks.type(stft_repr.dtype)

        scatter_indices = repeat(
            self.freq_indices, "f -> b n f t", b=batch, n=num_stems, t=stft_repr.shape[-1]
        )

        stft_repr_expanded_stems = repeat(stft_repr, "b 1 ... -> b n ...", n=num_stems)
        if stft_repr_expanded_stems.is_complex():
            real = torch.zeros_like(stft_repr_expanded_stems.real).scatter_add_(
                2, scatter_indices, masks.real
            )
            imag = torch.zeros_like(stft_repr_expanded_stems.imag).scatter_add_(
                2, scatter_indices, masks.imag
            )
            masks_summed = torch.complex(real, imag)
        else:
            masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(
                2, scatter_indices, masks
            )

        denom = repeat(self.num_bands_per_freq, "f -> (f r) 1", r=channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        stft_repr = stft_repr * masks_averaged

        stft_repr = rearrange(
            stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels
        )

        recon_audio = torch.istft(
            stft_repr,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=istft_length,
        )

        recon_audio = rearrange(
            recon_audio, "(b n s) t -> b n s t", b=batch, s=self.audio_channels, n=num_stems
        )

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")

        target = target[..., : recon_audio.shape[-1]]

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.0

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_audio_stft = torch.stft(recon_audio, **res_stft_kwargs)
            target_stft = torch.stft(target, **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(
                recon_audio_stft, target_stft
            )

        multi_stft_resolution_loss = (
            multi_stft_resolution_loss / len(self.multi_stft_resolutions_window_sizes)
        )

        total_loss = loss + multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        if not return_loss_breakdown:
            return total_loss

        return total_loss, loss, multi_stft_resolution_loss

class ErrorCode(str, Enum):
    TORCH_MISSING = "E001"
    MPS_UNAVAILABLE = "E002"
    SPLIT_IMPORT_FAILED = "E003"
    MODEL_MISSING = "E004"
    CONFIG_MISSING = "E005"
    ONNX_RUNTIME_MISSING = "E006"
    UPLOAD_FAILED = "E007"
    AUDIO_LOAD_FAILED = "E008"
    RESAMPLE_FAILED = "E009"
    FFMPEG_MISSING = "E010"
    SEPARATION_FAILED = "E011"
    ZIP_FAILED = "E012"
    TASK_NOT_FOUND = "E013"
    INVALID_REQUEST = "E014"
    RERUN_PREREQ_MISSING = "E015"


@dataclass
class AppError(Exception):
    code: ErrorCode
    message: str

    def to_http(self, status: int = 400) -> HTTPException:
        return HTTPException(status_code=status, detail={"code": self.code, "message": self.message})


class TaskStopped(Exception):
    """Signal that a task was stopped by the user."""


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"
UPLOAD_DIR = BASE_DIR / "uploads"
CONVERTED_DIR = BASE_DIR / "uploads_converted"
SEGMENT = 352_800
DEUX_SEGMENT = 573_300
OVERLAP = 12
LOG_PATH = BASE_DIR / "main_stemsplat.log"
LEGACY_LOG = BASE_DIR / "stemsplat.log"
if LEGACY_LOG.exists():
    try:
        LEGACY_LOG.unlink()
    except Exception:
        pass

MODEL_ALIAS_MAP = {
    "mel_band_roformer_vocals_becruily.ckpt": ["Mel Band Roformer Vocals.ckpt"],
    "mel_band_roformer_instrumental_becruily.ckpt": ["Mel Band Roformer Instrumental.ckpt"],
}

CONFIG_ALIAS_MAP: dict[str, list[str]] = {}

file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
    handlers=[
        stream_handler,
        file_handler,
    ],
)
logger = logging.getLogger("stemsplat")
for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    uvicorn_logger = logging.getLogger(name)
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.addHandler(file_handler)

# Silence extremely noisy multipart debug logs
logging.getLogger("python_multipart").setLevel(logging.INFO)

INSTALL_LOG_PATH = BASE_DIR / "install_stemsplat.log"
install_logger = logging.getLogger("stemsplat.install")
install_logger.setLevel(logging.DEBUG)
install_logger.addHandler(logging.FileHandler(INSTALL_LOG_PATH, encoding="utf-8"))

DEFAULT_OUTPUT_ROOT = (Path.home() / "Downloads" / "stemsplat").expanduser()
MODEL_FILE_MAP = {
    "vocals": ("mel_band_roformer_vocals_becruily.ckpt", "Mel Band Roformer Vocals Config.yaml"),
    "instrumental": ("mel_band_roformer_instrumental_becruily.ckpt", "Mel Band Roformer Instrumental Config.yaml"),
    "deux": ("becruily_deux.ckpt", "config_deux_becruily.yaml"),
    "drums": ("kuielab_a_drums.onnx", None),
    "bass": ("kuielab_a_bass.onnx", None),
    "other": ("kuielab_a_other.onnx", None),
    "karaoke": ("mel_band_roformer_karaoke_becruily.ckpt", None),
    "guitar": ("becruily_guitar.ckpt", "config_guitar_becruily.yaml"),
}

install_progress = {"pct": 0, "step": "starting", "models_missing": []}
choice_event = threading.Event()
shutdown_event = threading.Event()
INSTALL_PORT = 6060
MAIN_PORT = 8000
MODEL_URLS = [(item["filename"], item["url"]) for item in FILES]
TOTAL_BYTES = int(3.68 * 1024**3)


class InstallerHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.web_dir = BASE_DIR / "web"
        super().__init__(*args, directory=str(self.web_dir), **kwargs)

    def log_message(self, format, *args):  # noqa: A003 - match base signature
        install_logger.info("HTTP %s", format % args)

    def do_GET(self):
        if self.path == "/progress":
            install_logger.debug("progress requested: %s", install_progress)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(install_progress).encode())
            return

        if self.path in {"/", "/index.html", "/install.html"}:
            install_logger.debug("redirecting installer ui %s to main app", self.path)
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{MAIN_PORT}/")
            self.end_headers()
            return

        if self.path == "/installer_shutdown":
            install_logger.info("installer shutdown requested via http")
            shutdown_event.set()
            self.send_response(200)
            self.end_headers()
            return

        return super().do_GET()

    def do_POST(self):
        if self.path == "/download_models":
            install_logger.info("user requested model download")
            length = int(self.headers.get("Content-Length", 0))
            selection = None
            if length:
                try:
                    body = self.rfile.read(length)
                    parsed = json.loads(body.decode("utf-8"))
                    if isinstance(parsed, dict) and isinstance(parsed.get("models"), list):
                        selection = [str(x) for x in parsed["models"]]
                except Exception:
                    install_logger.debug("failed to parse selection body", exc_info=True)
            install_progress["choice"] = "download"
            install_progress["selection"] = selection
            choice_event.set()
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/skip_models":
            install_logger.info("user skipped model download")
            install_progress["choice"] = "skip"
            choice_event.set()
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/installer_shutdown":
            install_logger.info("installer shutdown requested via post")
            shutdown_event.set()
            self.send_response(200)
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()


def _installed():
    return Path("venv").exists()


def run_installer_ui():
    handler = InstallerHandler
    worker = threading.Thread(target=install, daemon=True)
    worker.start()
    threading.Thread(target=_launch_main_page, daemon=True).start()
    socketserver.TCPServer.allow_reuse_address = True

    if not _port_available(INSTALL_PORT):
        msg = f"Port {INSTALL_PORT} is already in use. Skipping installer UI."
        install_logger.warning(msg)
        print(msg)
        return

    try:
        with socketserver.TCPServer(("localhost", INSTALL_PORT), handler) as httpd:
            install_logger.info("installer ui listening on http://localhost:%s", INSTALL_PORT)
            server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            server_thread.start()
            install_logger.debug("waiting for installer routine to finish before shutting ui")
            try:
                while not shutdown_event.wait(0.5):
                    continue
                install_logger.info("installer routine finished; shutting down installer ui")
                httpd.shutdown()
                httpd.server_close()
                server_thread.join(timeout=2)
                install_logger.debug(
                    "installer ui closed; main app launch handled by installer page"
                )
            except KeyboardInterrupt:
                install_logger.info("installer interrupted; shutting down")
                httpd.shutdown()
                httpd.server_close()
    except OSError as exc:
        install_logger.warning("failed to bind install server on port %s: %s", INSTALL_PORT, exc)
        print(f"Port {INSTALL_PORT} is already in use. Skipping installer UI.")
        return


def _missing_required_models() -> list[str]:
    models_dir_candidates = [BASE_DIR / "Models", Path("Models")]
    models_dir = next((d for d in models_dir_candidates if d.exists()), models_dir_candidates[0])
    required_terms = ["instrumental", "vocals", "deux"]
    found = {term: False for term in required_terms}
    if models_dir.exists():
        for path in models_dir.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if not name.endswith(".cpkt"):
                continue
            for term in required_terms:
                if term in name:
                    found[term] = True
    return [term for term, present in found.items() if not present]


def _models_missing() -> bool:
    return len(_missing_required_models()) > 0


def _start_server():
    install_logger.info("starting main server with uvicorn on port %s", MAIN_PORT)
    if not _port_available(MAIN_PORT):
        install_logger.info("main server already running on port %s; opening browser", MAIN_PORT)
        install_progress["main_running"] = True
        install_logger.debug("main server already running; installer page will navigate")
        shutdown_event.set()
        return
    subprocess.Popen(
        [str(python_path()), "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(MAIN_PORT)]
    )
    install_logger.debug("main server started; installer page will navigate")


def _launch_main_page() -> None:
    main_url = f"http://localhost:{MAIN_PORT}/"
    for _ in range(40):
        try:
            with urllib.request.urlopen(main_url, timeout=1):
                webbrowser.open(main_url, new=0)
                return
        except Exception:
            time.sleep(0.5)
    webbrowser.open(main_url, new=0)


def _download_models(selection: Optional[list[str]] = None):
    try:
        install_progress["step"] = "downloading models"
        install_progress["pct"] = 5
        download_to(BASE_DIR, selection or [])
        install_progress["pct"] = 100
        install_progress["step"] = "done"
        _start_server()
    except Exception as exc:
        install_logger.exception("model download failed")
        install_progress["step"] = f"download failed: {exc}"
        install_progress["error"] = str(exc)
        install_progress["pct"] = -1


def install():
    install_logger.info("install routine starting")
    try:
        install_progress["step"] = "installing prerequisites"
        install_progress["pct"] = 1
        install_progress["models_missing"] = _missing_required_models()
        if _installed():
            _start_server()
        if _installed():
            install_logger.info("virtual environment already present")
        steps = []
        if not _installed():
            steps.append(("creating virtual environment", [sys.executable, "-m", "venv", "venv"]))
        steps.append(("upgrading pip", [str(pip_path()), "install", "--upgrade", "pip"]))
        steps.extend([(f"installing {pkg}", [str(pip_path()), "install", pkg]) for pkg in INSTALL_DEPS])

        total = max(1, len(steps))
        for i, (msg, cmd) in enumerate(steps, start=1):
            install_progress["step"] = msg
            install_progress["pct"] = int((i - 1) / total * 100)
            install_logger.info("running step %s/%s: %s", i, total, msg)
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as exc:
                install_logger.exception("error during %s", msg)
                install_progress["step"] = f"error during {msg}: {exc}"
                install_progress["pct"] = -1
                return

        if _models_missing():
            install_logger.info("models missing; skipping downloads per v0.1 flow")
            install_progress["step"] = "models missing; skipping downloads"
            install_progress["pct"] = 100
            _start_server()
            return

        install_progress["pct"] = 100
        install_progress["step"] = "done"
        install_logger.info("prerequisites satisfied; launching server")
        _start_server()
    except Exception:
        install_logger.exception("installer crashed")
        install_progress["step"] = "installation failed"
        install_progress["pct"] = -1
    finally:
        shutdown_event.set()

CHUNK_SIZE = 8 * 1024 * 1024
SSL_CONTEXT = (
    ssl.create_default_context(cafile=certifi.where())
    if certifi is not None
    else ssl.create_default_context()
)

def try_head_content_length(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "app-downloader"})
    try:
        with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as resp:
            cl = resp.headers.get("Content-Length")
    except Exception:
        return None
    return int(cl) if cl and cl.isdigit() else None


def download_to(base_dir: Path, selected: list[str] | None = None) -> None:
    base_dir = base_dir.resolve()
    wanted = set(selected or [])
    for item in FILES:
        tag = item.get("tag")
        if wanted and tag and tag not in wanted:
            continue
        dest = base_dir / item["subdir"] / item["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        remote_len = try_head_content_length(item["url"])
        if dest.exists() and remote_len is not None and dest.stat().st_size == remote_len:
            continue

        req = urllib.request.Request(item["url"], headers={"User-Agent": "app-downloader"})
        with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as resp:
            if tmp.exists():
                tmp.unlink()
            with open(tmp, "wb") as handle:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)

        os.replace(tmp, dest)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    stem, suffix = path.stem, path.suffix
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def _locate_case_insensitive(path: Path) -> Path | None:
    """Return a filesystem entry matching ``path`` regardless of case."""
    if path.exists():
        return path
    parent = path.parent
    if not parent.exists():
        return None
    target_lower = path.name.lower()
    for candidate in parent.iterdir():
        if candidate.name.lower() == target_lower:
            return candidate
    return None


@dataclass
class UserSettings:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    structure_mode: str = "flat"  # structured|flat

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "output_root": str(self.output_root),
            "structure_mode": self.structure_mode,
        }

    def resolve_output_dir(self, base_name: str) -> Path:
        root = self.output_root.expanduser()
        if not root.exists():
            raise AppError(ErrorCode.INVALID_REQUEST, "output root missing")
        target = root if self.structure_mode == "flat" else ensure_unique_dir(root / base_name)
        return _ensure_dir(target)

    def update(
        self,
        *,
        output_root: Optional[str] = None,
        structure_mode: Optional[str] = None,
    ) -> None:
        if output_root is not None:
            candidate = Path(output_root).expanduser()
            try:
                candidate.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise AppError(ErrorCode.INVALID_REQUEST, f"unable to create folder: {exc}")
            if not candidate.is_dir():
                raise AppError(ErrorCode.INVALID_REQUEST, "folder does not exist")
            self.output_root = candidate
        if structure_mode in {"structured", "flat"}:
            self.structure_mode = structure_mode


settings = UserSettings()
_ensure_dir(settings.output_root)


def resolve_output_plan(info: dict, *, structure_mode: Optional[str] = None) -> dict[str, Path | str | None]:
    base_name = Path(info.get("orig_name") or info.get("conv_src", "stems")).stem
    root = DEFAULT_OUTPUT_ROOT
    flat_root = root.parent if root.name == "stemsplat" else root
    _ensure_dir(flat_root)
    staging_dir = Path(tempfile.mkdtemp(prefix="stemsplat_stage_", dir=str(CONVERTED_DIR)))
    deliver_dir = flat_root
    zip_target = ensure_unique_path(flat_root / f"{base_name}.zip")
    return {"deliver_dir": deliver_dir, "staging_dir": staging_dir, "zip_target": zip_target, "structure_mode": "flat"}

# ── Device handling ─────────────────────────────────────────────────────────

def _ensure_torch() -> None:
    if not hasattr(torch, "backends"):
        raise AppError(ErrorCode.TORCH_MISSING, "PyTorch is unavailable; install torch with Metal support.")


def select_device() -> torch.device:
    """Prefer Metal (MPS) on Apple Silicon; fall back to CPU with explicit warning."""
    _ensure_torch()
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    raise AppError(ErrorCode.MPS_UNAVAILABLE, "Apple Metal (mps) device is unavailable; ensure torch==2.x with mps support.")


def _close_installer_ui(port: int = 6060) -> None:
    """Best-effort request to shut down the installer helper server."""

    url = f"http://localhost:{port}/installer_shutdown"
    logger.info("attempting to close installer ui on port %s", port)
    try:
        with urllib.request.urlopen(url, timeout=1) as resp:
            logger.debug("installer ui shutdown status=%s", getattr(resp, "status", "unknown"))
    except Exception as exc:
        logger.debug("installer ui not reachable on %s: %s", url, exc)


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", port))
            return True
        except OSError:
            return False


# ── Model management ────────────────────────────────────────────────────────

DEFAULT_CKPT = BASE_DIR / "models" / "Mel Band Roformer Vocals.ckpt"
DEFAULT_YAML = BASE_DIR / "configs" / "Mel Band Roformer Vocals Config.yaml"


def load_model(ckpt_path: str, yaml_path: str, device: torch.device):
    with open(yaml_path, "r") as handle:
        cfg_root = yaml.unsafe_load(handle)

    default_target = "mel_band_roformer.MelBandRoformer"

    def find_target(cfg):
        if "target" in cfg:
            target = cfg.pop("target")
            return target, cfg
        raise ValueError("No 'target' in YAML")

    try:
        target_path, kwargs = find_target(cfg_root)
    except ValueError:
        target_path, kwargs = default_target, {"model": cfg_root.get("model", {})}

    module_name, class_name = target_path.rsplit(".", 1)
    if module_name == "mel_band_roformer":
        ModelClass = MelBandRoformer
    else:
        ModelClass = getattr(importlib.import_module(module_name), class_name)

    model = ModelClass(**kwargs.get("model", {}))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


def overlap_add(dst: np.ndarray, seg: np.ndarray, start: int, fade: int):
    end = start + seg.shape[-1]
    fade = min(fade, seg.shape[-1] // 2)
    if fade > 0:
        win = np.ones(seg.shape[-1], dtype=np.float32)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        win[:fade] = ramp
        win[-fade:] = ramp[::-1]
        dst[..., start:end] += seg * win
    else:
        dst[..., start:end] += seg


def split_main(argv=None, progress_cb=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help=".ckpt weights")
    parser.add_argument("--config", default=str(DEFAULT_YAML), help=".yaml model def")
    parser.add_argument("--wav", required=True, help="input WAV")
    parser.add_argument("--out", default="stems_out", help="output dir")
    parser.add_argument("--segment", type=int, default=352_800, help="segment size")
    parser.add_argument("--overlap", type=int, default=18, help="overlap percent")
    parser.add_argument("--vocals-only", action="store_true", help="only save vocals")
    parser.add_argument(
        "--device",
        default=(
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        ),
        help="compute device: 'mps' (Apple Metal) or 'cpu' (fallback)",
    )
    args = parser.parse_args(argv)

    wav, sr = torchaudio.load(args.wav)
    if sr != 44_100:
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        sr = 44_100
    is_tensor = getattr(torch, "is_tensor", lambda _: False)
    if is_tensor(wav):
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        else:
            wav = wav.cpu().numpy() if hasattr(wav, "cpu") else wav.numpy()

    n_channels = wav.shape[0]
    n_samples = wav.shape[1]

    stems = np.zeros((2, n_channels, n_samples), dtype=np.float32)

    seg_size = args.segment
    fade_len = int(seg_size * (args.overlap / 100))
    hop_size = seg_size - fade_len

    device = torch.device(args.device)
    model = load_model(args.ckpt, args.config, device)

    if progress_cb:
        progress_cb(0.0)
    with torch.no_grad(), tqdm(total=n_samples, unit="sample") as bar:
        for start in range(0, n_samples, hop_size):
            end = min(start + seg_size, n_samples)
            seg = wav[:, start:end]
            if seg.shape[1] < seg_size:
                seg = np.pad(seg, ((0, 0), (0, seg_size - seg.shape[1])))

            pred = model(torch.from_numpy(seg).unsqueeze(0).to(device))
            if isinstance(pred, dict) and "sources" in pred:
                pred = pred["sources"]
            pred = pred.squeeze(0).cpu().numpy()

            if pred.ndim == 2:
                if pred.shape[0] == 2:
                    pred = pred[:, np.newaxis, :]
                else:
                    pred = pred[np.newaxis, :, :]
            elif pred.ndim == 3 and pred.shape[1] not in (1, n_channels):
                pred = pred.transpose(1, 0, 2)

            if pred.shape[1] == 1 and n_channels == 2:
                pred = np.repeat(pred, 2, axis=1)

            chunk_len = end - start
            if pred.shape[2] > chunk_len:
                pred = pred[:, :, :chunk_len]
            if pred.shape[0] == 1:
                vocals_seg = pred[0]
                mix_seg = seg[:, :chunk_len]
                inst_seg = mix_seg - vocals_seg
                pred = np.stack([vocals_seg, inst_seg], axis=0)

            overlap_add(stems, pred, start, fade_len)
            bar.update(min(hop_size, n_samples - start))
            if progress_cb:
                progress_cb(bar.n / bar.total)

    window_sum = np.ones(n_samples, dtype=np.float32)
    if fade_len:
        win = np.ones(seg_size, dtype=np.float32)
        ramp = np.linspace(0, 1, fade_len, dtype=np.float32)
        win[:fade_len] = ramp
        win[-fade_len:] = ramp[::-1]
        for start in range(0, n_samples, hop_size):
            end = min(start + seg_size, n_samples)
            window_sum[start:end] += win[: end - start]
    stems /= window_sum[None, None, :]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / "vocals.wav", stems[0].T, sr)
    if not args.vocals_only:
        sf.write(out_dir / "instrumental.wav", stems[1].T, sr)
    if progress_cb:
        progress_cb(1.0)
    print(f"✓ Done – stems saved to “{out_dir}”")


def _load_optional_roformer(path: Path, config_path: Optional[Path], device: torch.device):
    return load_model(str(path), str(config_path or ""), device)


class StemModel:
    def __init__(self, path: Path | None, device: torch.device, config_path: Path | None = None):
        self.device = device
        self.kind = "none"
        self.session = None
        self.net = None
        self.path = path
        self.config_path = config_path
        self.expects_waveform = False
        if path is None:
            return
        if path.suffix.lower() == ".ckpt":
            self.net = _load_optional_roformer(path, config_path, device)
            self.net.eval()
            self.kind = "roformer"
            self.expects_waveform = True
            return
        if path.suffix.lower() == ".onnx":
            if ort is None:
                raise AppError(ErrorCode.ONNX_RUNTIME_MISSING, "onnxruntime is required for ONNX models on Apple Silicon (CPU mode).")
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.kind = "onnx"
            return
        self.net = torch.jit.load(str(path), map_location=device)
        self.net.eval()
        self.kind = "torchscript"

    def _move_to(self, device: torch.device) -> None:
        """Move model to a different device (no-op for ONNX)."""
        if self.kind == "onnx":
            return
        if self.net is not None:
            self.net = self.net.to(device)
        self.device = device

    def __call__(self, mag: "torch.Tensor") -> "torch.Tensor":
        if self.kind == "onnx" and self.session is not None:
            inp_name = self.session.get_inputs()[0].name
            out = self.session.run(None, {inp_name: mag.detach().cpu().numpy()})[0]
            return torch.from_numpy(out).to(mag.device)
        if self.net is None:
            raise AppError(ErrorCode.MODEL_MISSING, "Model session not initialized.")
        with torch.no_grad():
            return self.net(mag)


class ModelManager:
    def __init__(self):
        self.device = select_device()
        self.model_info = dict(MODEL_FILE_MAP)
        self.model_cache: dict[str, StemModel] = {}

    def _resolve_path(self, filename: str) -> Path:
        search_names = [filename, *MODEL_ALIAS_MAP.get(filename, [])]
        search_dirs = [MODEL_DIR, Path.home() / "Library/Application Support/stems"]
        for name in search_names:
            for base in search_dirs:
                match = _locate_case_insensitive(base / name)
                if match:
                    return match
        raise AppError(ErrorCode.MODEL_MISSING, f"Model file missing: {filename}")

    def _resolve_config(self, filename: Optional[str]) -> Optional[Path]:
        if not filename:
            return None
        found = _locate_case_insensitive(CONFIG_DIR / filename)
        if found:
            return found
        raise AppError(ErrorCode.CONFIG_MISSING, f"Config file missing: {filename}")

    def _load_model(self, name: str) -> StemModel:
        if name in self.model_cache:
            return self.model_cache[name]
        if name not in self.model_info:
            raise AttributeError(name)
        model_fname, cfg_fname = self.model_info[name]
        model_path = self._resolve_path(model_fname)
        cfg_path = self._resolve_config(cfg_fname)
        model = StemModel(model_path, self.device, cfg_path)
        self.model_cache[name] = model
        if cfg_fname:
            setattr(self, f"{name}_config", cfg_path)
        return model

    def __getattr__(self, name: str) -> StemModel:
        if name in self.model_info:
            return self._load_model(name)
        raise AttributeError(name)

    # Separation helpers
    def split_vocals(
        self,
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[float], None]] = None,
        delay: float = 0.0,
    ):
        if isinstance(waveform, torch.Tensor):
            working = waveform.to(self.device)
        else:
            working = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        length = working.shape[1]
        step = max(1, segment - overlap)
        voc_acc = torch.zeros_like(working, device=self.device)
        inst_acc = torch.zeros_like(working, device=self.device)
        counts = torch.zeros((1, length), device=self.device)
        if progress_cb:
            progress_cb(0.0)
        for start in range(0, length, step):
            end = min(start + segment, length)
            seg = working[:, start:end]
            if seg.shape[1] < segment:
                padded = torch.zeros((working.shape[0], segment), device=self.device)
                padded[:, : seg.shape[1]] = seg
            else:
                padded = seg
            with torch.no_grad():
                pred = self.vocals(padded.unsqueeze(0))[0]
            if pred.dim() == 2:
                pred = pred[:, None, :]
            elif pred.dim() == 3 and pred.shape[1] not in (1, working.shape[0]):
                pred = pred.permute(1, 0, 2)
            pred = pred[..., : seg.shape[1]]
            voc_seg = pred[0]
            inst_seg = pred[1] if pred.shape[0] > 1 else seg - voc_seg
            sl = voc_seg.shape[1]
            voc_acc[:, start : start + sl] += voc_seg
            inst_acc[:, start : start + sl] += inst_seg
            counts[:, start : start + sl] += 1
            if progress_cb:
                progress_cb(end / length)
            if delay:
                time.sleep(delay)
        denom = counts.clamp_min(1)
        return voc_acc / denom, inst_acc / denom

    def split_pair_with_model(
        self,
        model: StemModel,
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[float], None]] = None,
        delay: float = 0.0,
    ):
        if isinstance(waveform, torch.Tensor):
            working = waveform.to(self.device)
        else:
            working = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        length = working.shape[1]
        step = max(1, segment - overlap)
        out0 = torch.zeros_like(working, device=self.device)
        out1 = torch.zeros_like(working, device=self.device)
        counts = torch.zeros((1, length), device=self.device)
        if progress_cb:
            progress_cb(0.0)

        if model.expects_waveform:
            for start in range(0, length, step):
                end = min(start + segment, length)
                seg = working[:, start:end]
                if seg.shape[1] < segment:
                    padded = torch.zeros((working.shape[0], segment), device=self.device)
                    padded[:, : seg.shape[1]] = seg
                else:
                    padded = seg
                with torch.no_grad():
                    pred = model(padded.unsqueeze(0))[0]
                if pred.dim() == 2:
                    pred = pred[:, None, :]
                elif pred.dim() == 3 and pred.shape[1] not in (1, working.shape[0]):
                    pred = pred.permute(1, 0, 2)
                pred = pred[..., : seg.shape[1]]
                stem0 = pred[0]
                stem1 = pred[1] if pred.shape[0] > 1 else seg - stem0
                sl = stem0.shape[1]
                out0[:, start : start + sl] += stem0
                out1[:, start : start + sl] += stem1
                counts[:, start : start + sl] += 1
                if progress_cb:
                    progress_cb(end / length)
                if delay:
                    time.sleep(delay)
            denom = counts.clamp_min(1)
            return out0 / denom, out1 / denom

        # STFT mask path
        n_fft = 2048
        hop = 441
        win = torch.hann_window(n_fft, device=self.device)
        for start in range(0, length, step):
            end = min(start + segment, length)
            seg = working[:, start:end]
            if seg.shape[1] < n_fft:
                out0[:, start:end] += seg
                if progress_cb:
                    progress_cb(end / length)
                continue
            spec = torch.stft(seg, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
            mag = spec.abs().unsqueeze(0)
            mask = model(mag)[0]
            stem0_spec = spec * mask
            stem0 = torch.istft(
                stem0_spec,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=win,
                length=seg.shape[1],
            )
            out0[:, start : start + stem0.shape[1]] += stem0
            out1[:, start : start + stem0.shape[1]] += seg[:, : stem0.shape[1]] - stem0
            if progress_cb:
                progress_cb(end / length)
            if delay:
                time.sleep(delay)
        return out0, out1

    def split_instrumental(
        self,
        waveform,
        segment: int,
        overlap: int,
        progress_cb: Optional[Callable[[float], None]] = None,
        delay: float = 0.0,
    ):
        if isinstance(waveform, torch.Tensor):
            working = waveform.to(self.device)
        else:
            working = torch.tensor(waveform, dtype=torch.float32, device=self.device)
        length = working.shape[1]
        step = max(1, segment - overlap)
        n_fft = 2048
        hop = 441
        win = torch.hann_window(n_fft, device=self.device)
        drums = torch.zeros_like(working, device=self.device)
        bass = torch.zeros_like(working, device=self.device)
        other = torch.zeros_like(working, device=self.device)
        karaoke = torch.zeros_like(working, device=self.device)
        guitar = torch.zeros_like(working, device=self.device)
        if progress_cb:
            progress_cb(0.0)
        for start in range(0, length, step):
            end = min(start + segment, length)
            seg = working[:, start:end]
            if seg.shape[1] < n_fft:
                for tensor in (drums, bass, other, karaoke, guitar):
                    tensor[:, start:end] += seg * 0
                if progress_cb:
                    progress_cb(end / length)
                continue
            spec = torch.stft(seg, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
            mag = spec.abs().unsqueeze(0)
            masks = {
                "drums": self.drums(mag)[0],
                "bass": self.bass(mag)[0],
                "other": self.other(mag)[0],
                "karaoke": self.karaoke(mag)[0],
                "guitar": self.guitar(mag)[0],
            }
            stems = {
                "drums": drums,
                "bass": bass,
                "other": other,
                "karaoke": karaoke,
                "guitar": guitar,
            }
            for name, mask in masks.items():
                spec_masked = spec * mask
                stem = torch.istft(
                    spec_masked,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=n_fft,
                    window=win,
                    length=seg.shape[1],
                )
                stems[name][:, start : start + stem.shape[1]] += stem
            if progress_cb:
                progress_cb(end / length)
            if delay:
                time.sleep(delay)
        return drums, bass, other, karaoke, guitar


# Ensure core helper methods remain bound to ModelManager
_REQUIRED_MANAGER_HELPERS = ("split_vocals", "split_pair_with_model", "split_instrumental")
_missing_helpers = [name for name in _REQUIRED_MANAGER_HELPERS if not hasattr(ModelManager, name)]
if _missing_helpers:
    raise AppError(ErrorCode.SPLIT_IMPORT_FAILED, f"ModelManager missing helpers: {', '.join(_missing_helpers)}")


# ── Task orchestration ─────────────────────────────────────────────────────-

app = FastAPI()
progress: dict[str, dict[str, int | str]] = {}
errors: dict[str, str] = {}
tasks: dict[str, dict] = {}
controls: dict[str, dict[str, threading.Event]] = {}
process_queue: queue.Queue[Callable[[], None]] = queue.Queue()
app_ready_event = threading.Event()
app_ready_state = {"ready": False, "checking": True, "models_missing": []}

def _worker() -> None:
    while True:
        fn = process_queue.get()
        try:
            fn()
        finally:
            process_queue.task_done()

threading.Thread(target=_worker, daemon=True).start()

download_lock = threading.Lock()
downloading = False


def _run_startup_checks() -> None:
    try:
        app_ready_state["checking"] = True
        app_ready_state["models_missing"] = _missing_required_models()
    finally:
        app_ready_state["checking"] = False
        app_ready_state["ready"] = True
        app_ready_event.set()


@app.on_event("startup")
async def _startup_cleanup() -> None:
    _close_installer_ui()
    threading.Thread(target=_run_startup_checks, daemon=True).start()


@app.exception_handler(AppError)
async def _handle_app_error(request: Request, exc: AppError):  # noqa: D401 - FastAPI signature
    """Return JSON-encoded error responses while logging the failure."""

    logger.error("app error for %s %s: %s %s", request.method, request.url.path, exc.code, exc.message)
    return JSONResponse(status_code=400, content={"code": exc.code, "message": exc.message})


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info("request %s %s from %s", request.method, request.url.path, request.client.host if request.client else "?")
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled error for %s %s", request.method, request.url.path)
        raise
    duration = (time.time() - start) * 1000
    logger.info("completed %s %s in %.1fms status=%s", request.method, request.url.path, duration, response.status_code)
    return response


def _download_models() -> None:
    global downloading
    with download_lock:
        if downloading:
            return
        downloading = True
    MODEL_DIR.mkdir(exist_ok=True)
    for name, url in MODEL_URLS:
        dest = MODEL_DIR / name
        try:
            import urllib.request

            with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    out.write(chunk)
                    logger.debug("downloaded chunk for %s", name)
        except Exception as exc:  # pragma: no cover - network errors
            logging.error("model download failed: %s", exc)
            logger.exception("model download failure for %s", name)
    downloading = False


@app.post("/download_models")
async def download_models():
    thread = threading.Thread(target=_download_models, daemon=True)
    thread.start()
    return {"status": "started"}


# ── Separation helpers ─────────────────────────────────────────────────────-

def _prepare_waveform(audio_path: Path) -> torch.Tensor:
    loaded_via_soundfile = False
    try:
        logger.info("loading audio %s", audio_path)
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception as exc:
        needs_fallback = isinstance(exc, ImportError) and ("torchcodec" in str(exc).lower() or "TorchCodec" in str(exc))
        if not needs_fallback:
            logger.exception("failed to load %s", audio_path)
            raise AppError(ErrorCode.AUDIO_LOAD_FAILED, f"Failed to load audio: {exc}") from exc

        logger.warning("torchcodec missing; retrying %s with soundfile backend", audio_path)
        try:
            torchaudio.set_audio_backend("soundfile")
            waveform, sr = torchaudio.load(str(audio_path))
            logger.info("loaded %s via torchaudio soundfile backend", audio_path)
        except Exception as sf_exc:
            logger.debug("torchaudio soundfile backend failed for %s: %s", audio_path, sf_exc)
            try:
                data, sr = sf.read(str(audio_path))
                waveform = torch.from_numpy(data)
                logger.info("loaded %s via soundfile direct read", audio_path)
                loaded_via_soundfile = True
            except Exception as final_exc:
                logger.exception("failed to load %s via all fallbacks", audio_path)
                raise AppError(ErrorCode.AUDIO_LOAD_FAILED, f"Failed to load audio: {final_exc}") from final_exc

    if loaded_via_soundfile and waveform.ndim == 2:
        waveform = waveform.transpose(0, 1)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if not torch.is_floating_point(waveform):
        waveform = waveform.float()
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
    if sr != 44100:
        try:
            logger.info("resampling %s from %s to 44100", audio_path, sr)
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        except Exception as exc:
            logger.exception("resample failed for %s", audio_path)
            raise AppError(ErrorCode.RESAMPLE_FAILED, f"Resample failed: {exc}") from exc
    logger.debug("loaded waveform shape=%s sr=44100", tuple(waveform.shape))
    return waveform

def _ensure_ffmpeg() -> str:
    """Return a usable ffmpeg executable path, downloading a bundled copy if needed."""

    import shutil
    import subprocess

    path = shutil.which("ffmpeg")
    candidates: list[str] = []
    if path:
        candidates.append(path)

    try:  # fallback to bundled binary if available
        import imageio_ffmpeg  # type: ignore

        try:
            bundled = imageio_ffmpeg.get_ffmpeg_exe()
            if bundled:
                candidates.append(bundled)
                logger.debug("using bundled ffmpeg at %s", bundled)
        except Exception:
            logger.debug("imageio-ffmpeg present but failed to provide binary", exc_info=True)
    except Exception:
        logger.debug("imageio-ffmpeg not available; relying on system ffmpeg")

    for candidate in candidates:
        try:
            subprocess.run([candidate, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return candidate
        except FileNotFoundError:
            logger.debug("ffmpeg candidate missing at %s", candidate)
        except Exception as exc:
            logger.debug("ffmpeg candidate at %s failed health check: %s", candidate, exc)

    logger.error("ffmpeg binary missing from PATH and bundled lookup failed; conversion cannot proceed")
    raise AppError(ErrorCode.FFMPEG_MISSING, "ffmpeg not found; install ffmpeg or ensure imageio-ffmpeg can download it.")


def _convert_to_wav(audio_path: Path, *, remove_source: bool = False) -> Path:
    """Convert ``audio_path`` to WAV in-place and optionally drop the original copy."""

    ffmpeg_path = _ensure_ffmpeg()

    if audio_path.suffix.lower() in {".wav", ".wave"}:
        logger.debug("%s already wav; skipping convert", audio_path)
        return audio_path

    wav_path = audio_path.with_suffix(".wav")
    logger.info("converting %s to wav at %s", audio_path, wav_path)
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-ar",
        "44100",
        "-ac",
        "2",
        "-vn",
        str(wav_path),
    ]
    conversion_ok = False
    try:
        import subprocess

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        conversion_ok = True
    except FileNotFoundError as exc:
        raise AppError(ErrorCode.FFMPEG_MISSING, "ffmpeg not found; install ffmpeg and ensure it is on PATH.") from exc
    except Exception as exc:
        logger.error("conversion to wav failed for %s: %s", audio_path, exc)
        try:
            if hasattr(exc, "stderr") and exc.stderr:
                logger.error("ffmpeg stderr: %s", exc.stderr)
        except Exception:
            pass
        raise AppError(ErrorCode.UPLOAD_FAILED, f"Conversion to WAV failed: {exc}") from exc

    if conversion_ok and not wav_path.exists():
        raise AppError(ErrorCode.UPLOAD_FAILED, f"WAV conversion did not create output for {audio_path}")

    if remove_source and conversion_ok and wav_path != audio_path:
        try:
            audio_path.unlink()
        except Exception as exc:
            logging.warning("failed to remove source %s after conversion: %s", audio_path, exc)
    elif remove_source and not conversion_ok and audio_path.exists():
        logger.debug("preserving original %s because conversion did not complete", audio_path)
    return wav_path


def _stage_audio_copy(src: Path, dest_dir: Path, *, remove_original_copy: bool = True) -> Path:
    """Place a copy of ``src`` into ``dest_dir`` and normalize it to WAV."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = dest_dir / src.name
    if staged != src:
        logger.debug("copying %s (%s bytes) to %s", src, src.stat().st_size if src.exists() else "n/a", staged)
        shutil.copy2(src, staged)
    try:
        wav_path = _convert_to_wav(staged, remove_source=remove_original_copy)
    except AppError:
        if staged.exists() and staged.suffix.lower() != ".wav":
            try:
                staged.unlink()
                logger.debug("cleaned failed staging copy %s", staged)
            except Exception as exc:
                logger.warning("failed to clean staging copy %s after error: %s", staged, exc)
        raise
    if staged.exists() and staged.suffix.lower() != ".wav" and staged != wav_path:
        try:
            staged.unlink()
            logger.debug("removed non-wav staging copy %s after conversion", staged)
        except Exception as exc:
            logger.warning("failed to remove staging copy %s: %s", staged, exc)
    logger.info("staged %s as %s", src, wav_path)
    return wav_path


def _queue_processing(task_id: str, out_dir: Path, stem_list: list[str]) -> None:
    pause_evt = controls.setdefault(task_id, {}).setdefault("pause", threading.Event())
    stop_evt = controls.setdefault(task_id, {}).setdefault("stop", threading.Event())

    def _mark_stopped() -> None:
        progress[task_id] = {
            "stage": "stopped",
            "pct": 0,
            "stems": tasks.get(task_id, {}).get("stems"),
            "out_dir": tasks.get(task_id, {}).get("dir"),
            "zip": tasks.get(task_id, {}).get("zip"),
        }
        tasks[task_id]["status"] = "stopped"

    def _raise_if_stopped() -> None:
        if stop_evt.is_set():
            _mark_stopped()
            raise TaskStopped()

    def cb(stage: str, pct: int):
        while pause_evt.is_set():
            progress[task_id] = {
                "stage": "paused",
                "pct": pct,
                "stems": tasks.get(task_id, {}).get("stems"),
                "out_dir": tasks.get(task_id, {}).get("dir"),
                "zip": tasks.get(task_id, {}).get("zip"),
            }
            time.sleep(0.5)
        _raise_if_stopped()
        logger.debug("task %s stage=%s pct=%s", task_id, stage, pct)
        progress[task_id] = {
            "stage": stage,
            "pct": pct,
            "stems": tasks.get(task_id, {}).get("stems"),
            "out_dir": tasks.get(task_id, {}).get("dir"),
            "zip": tasks.get(task_id, {}).get("zip"),
        }

    def run() -> None:
        try:
            _raise_if_stopped()
            while not app_ready_event.is_set():
                _raise_if_stopped()
                tasks[task_id]["status"] = "preparing"
                progress[task_id] = {
                    "stage": "preparing",
                    "pct": 0,
                    "stems": tasks.get(task_id, {}).get("stems"),
                    "out_dir": tasks.get(task_id, {}).get("dir"),
                    "zip": tasks.get(task_id, {}).get("zip"),
                }
                time.sleep(0.5)
            tasks[task_id]["status"] = "running"
            manager = ModelManager()
            cb("preparing", 0)
            cb("prepare.complete", 1)
            conv_src = tasks.get(task_id, {}).get("conv_src")
            if conv_src:
                audio_path = Path(conv_src)
                if not audio_path.exists():
                    raise AppError(ErrorCode.UPLOAD_FAILED, "Converted source missing; please re-upload.")
            else:
                orig_src = tasks.get(task_id, {}).get("orig_src")
                if not orig_src:
                    raise AppError(ErrorCode.UPLOAD_FAILED, "Missing source file for conversion")
                audio_path = _stage_audio_copy(Path(orig_src), CONVERTED_DIR)
                tasks[task_id]["conv_src"] = str(audio_path)
            plan_mode = tasks.get(task_id, {}).get("structure_mode", settings.structure_mode)
            staging_dir = Path(tasks.get(task_id, {}).get("staging_dir") or out_dir)
            deliver_dir = Path(tasks.get(task_id, {}).get("dir") or out_dir)
            zip_target = tasks.get(task_id, {}).get("zip_target")
            stems_out = _separate_waveform(manager, audio_path, stem_list, cb, staging_dir, stop_check=_raise_if_stopped)

            zip_path: Path | None = None
            if plan_mode != "structured" and zip_target is not None and len(stems_out) > 1:
                _raise_if_stopped()
                zip_path = ensure_unique_path(Path(zip_target) if zip_target else deliver_dir / f"{audio_path.stem}.zip")
                cb("zip.start", 99)
                try:
                    with zipfile.ZipFile(zip_path, "w") as zf:
                        for name in stems_out:
                            _raise_if_stopped()
                            fp = staging_dir / name
                            if fp.exists():
                                zf.write(fp, arcname=name)
                                logger.debug("added %s to %s", fp, zip_path)
                            else:
                                logger.warning("expected stem %s missing at %s", name, fp)
                except Exception as exc:
                    raise AppError(ErrorCode.ZIP_FAILED, f"Failed to create zip: {exc}") from exc
                cb("zip.done", 99)
                try:
                    if staging_dir.exists():
                        shutil.rmtree(staging_dir)
                except Exception:
                    logger.debug("failed to clean staging dir %s", staging_dir, exc_info=True)
                tasks[task_id]["zip"] = str(zip_path)
                deliver_dir = zip_path.parent
            else:
                tasks[task_id]["zip"] = None

            tasks[task_id]["stems"] = stems_out
            tasks[task_id]["dir"] = str(deliver_dir)
            cb("finalizing", 99)
            progress[task_id] = {
                "stage": "done",
                "pct": 100,
                "stems": stems_out,
                "out_dir": str(deliver_dir),
                "zip": tasks.get(task_id, {}).get("zip"),
            }
            tasks[task_id]["status"] = "done"
            logger.info("task %s completed; stems=%s; zip=%s", task_id, stems_out, zip_path)
        except TaskStopped:
            _mark_stopped()
        except AppError as exc:
            progress[task_id] = {
                "stage": "error",
                "pct": -1,
                "stems": tasks.get(task_id, {}).get("stems"),
                "out_dir": tasks.get(task_id, {}).get("dir"),
                "zip": tasks.get(task_id, {}).get("zip"),
            }
            errors[task_id] = json.dumps({"code": exc.code, "message": exc.message})
            tasks[task_id]["status"] = "error"
            logger.error("task %s failed with %s: %s", task_id, exc.code, exc.message)
            logger.debug(
                "task %s context dir=%s conv_src=%s stems=%s",
                task_id,
                out_dir,
                tasks.get(task_id, {}).get("conv_src"),
                stem_list,
            )
        except Exception as exc:  # pragma: no cover - safety net
            logging.exception("processing failed")
            progress[task_id] = {
                "stage": "error",
                "pct": -1,
                "stems": tasks.get(task_id, {}).get("stems"),
                "out_dir": tasks.get(task_id, {}).get("dir"),
                "zip": tasks.get(task_id, {}).get("zip"),
            }
            errors[task_id] = json.dumps({"code": ErrorCode.SEPARATION_FAILED, "message": str(exc)})
            tasks[task_id]["status"] = "error"
            logger.exception("task %s crashed", task_id)

    tasks[task_id]["status"] = "queued"
    progress[task_id] = {
        "stage": "queued",
        "pct": 0,
        "stems": tasks.get(task_id, {}).get("stems"),
        "out_dir": tasks.get(task_id, {}).get("dir"),
        "zip": tasks.get(task_id, {}).get("zip"),
    }
    process_queue.put(run)


def convert_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    """Convert all supported audio files in ``input_dir`` to WAV under ``output_dir``."""

    _ensure_ffmpeg()

    if not input_dir.exists() or not input_dir.is_dir():
        raise AppError(ErrorCode.INVALID_REQUEST, f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    converted: list[Path] = []
    supported = {".wav", ".mp3", ".aac", ".flac", ".ogg", ".alac", ".opus", ".m4a"}

    for src_path in input_dir.rglob("*"):
        if src_path.suffix.lower() not in supported:
            continue
        relative_subpath = src_path.relative_to(input_dir)
        staged_dest = output_dir / relative_subpath
        staged_dest.parent.mkdir(parents=True, exist_ok=True)
        if staged_dest != src_path:
            logger.debug("copying %s to %s", src_path, staged_dest)
            shutil.copy2(src_path, staged_dest)
        wav_path = _convert_to_wav(staged_dest, remove_source=True)
        converted.append(wav_path)
    return converted


def _write_wave(
    out_dir: Path,
    fname: str,
    tensor: torch.Tensor,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = ensure_unique_path(out_dir / fname)
    logger.info("writing stem %s to %s", candidate.name, candidate.parent)
    data = tensor.detach().cpu()
    if not torch.isfinite(data).all():
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = torch.clamp(data.float(), -1.0, 1.0)
    sf.write(candidate, data.T.contiguous().numpy(), 44100, subtype="PCM_16")
    return candidate


def _separate_waveform(
    manager: "ModelManager",
    audio_path: Path,
    stem_list: list[str],
    cb: Callable[[str, int], None],
    out_dir: Path,
    *,
    stop_check: Optional[Callable[[], None]] = None,
) -> list[str]:
    logger.info("starting separation for %s with stems %s", audio_path, stem_list)
    waveform = _prepare_waveform(audio_path)
    cb("load_audio.done", 1)
    if stop_check:
        stop_check()

    stems_out: list[str] = []
    remaining_stems = [s for s in stem_list if s != "deux"]
    processing_started_at = time.time()
    chunk_count = math.ceil(waveform.shape[1] / max(1, SEGMENT - OVERLAP))

    def ensure_stereo(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[0] >= 2:
            return x
        return x.repeat(2, 1)

    def ensure_mono(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.unsqueeze(0)
        if x.shape[0] == 1:
            return x
        return x.mean(dim=0, keepdim=True)

    if "deux" in stem_list:
        cb("deux.start", 2)
        deux_model = manager.deux

        def _cb(frac: float) -> None:
            cb("deux", 2 + int(frac * 94))

        deux_wave = ensure_stereo(waveform)
        if stop_check:
            stop_check()
        voc_d, inst_d = manager.split_pair_with_model(deux_model, deux_wave, DEUX_SEGMENT, OVERLAP, progress_cb=_cb)
        fname_v = f"{audio_path.stem} - vocals (deux).wav"
        fname_i = f"{audio_path.stem} - instrumental (deux).wav"
        if stop_check:
            stop_check()
        path_v = _write_wave(out_dir, fname_v, voc_d)
        cb("write.vocals_deux", 97)
        if stop_check:
            stop_check()
        path_i = _write_wave(out_dir, fname_i, inst_d)
        cb("write.instrumental_deux", 98)
        stems_out.extend([path_v.name, path_i.name])

    channel_count = waveform.shape[0]
    channel_specs: list[tuple[int, str, int, int]] = []
    if channel_count >= 2:
        channel_specs = [(0, "left", 2, 49), (1, "right", 51, 99)]
    else:
        # mono: use most of the range so the bar still progresses smoothly
        channel_specs = [(0, "left", 2, 99)]

    need_vocal_pass = ("vocals" in remaining_stems) or any(s in remaining_stems for s in ["drums", "bass", "other", "guitar"])
    need_instrumental_model = "instrumental" in remaining_stems

    if remaining_stems:
        channel_stems: dict[str, list[torch.Tensor]] = {}

        def map_channel_pct(local_pct: float, start: int, end: int) -> int:
            """
            Convert a channel-local percent or fraction into an overall percent.

            If ``local_pct`` is in [0, 1], treat it as a fraction; otherwise treat
            it as a percent in [0, 100].
            """
            span = max(1, end - start)
            if 0.0 <= local_pct <= 1.0:
                scaled = local_pct
            else:
                scaled = max(0.0, min(100.0, local_pct)) / 100
            return start + int(span * scaled)

        def process_channel(idx: int, label: str, start: int, end: int) -> None:
            if idx >= channel_count:
                return
            chan_wave = waveform[idx : idx + 1]
            chan_outputs: dict[str, torch.Tensor] = {}

            def ch(stage: str, local_pct: float) -> None:
                overall = map_channel_pct(local_pct, start, end)
                cb(f"{label}.{stage}", overall)

            ch("start", 0)
            inst_wave: Optional[torch.Tensor] = None
            vocals_wave: Optional[torch.Tensor] = None

            if need_vocal_pass:
                ch("split_vocals.start", 2)

                def _cb(frac: float) -> None:
                    ch("vocals", 5 + (frac * 70))

                if stop_check:
                    stop_check()
                voc, inst = manager.split_vocals(ensure_stereo(chan_wave), SEGMENT, OVERLAP, progress_cb=_cb)
                vocals_wave = voc
                inst_wave = inst
                ch("split_vocals.done", 78)
                if "vocals" in remaining_stems:
                    chan_outputs["vocals"] = voc[:1]
                    ch("write.vocals", 82)
            else:
                inst_wave = chan_wave

            def _run_stem_model(
                model: StemModel,
                wave: torch.Tensor,
                progress_cb: Optional[Callable[[float], None]] = None,
            ) -> torch.Tensor:
                """Normalize tensor shape/device for model expectations."""
                prefer_stereo = getattr(model.net, "stereo", True) if model.net is not None else True

                def _normalize_waveform_input(target_model: StemModel, tensor: torch.Tensor) -> torch.Tensor:
                    prepared = ensure_stereo(tensor) if prefer_stereo else ensure_mono(tensor)
                    if prepared.dim() == 2:
                        prepared = prepared.unsqueeze(0)
                    return prepared.to(target_model.device)

                def _call_model(prepared: torch.Tensor) -> torch.Tensor:
                    try:
                        with torch.no_grad():
                            pred_out = model(prepared)
                    except RuntimeError as exc:
                        if model.device.type == "mps" and ("MPSGaph" in str(exc) or "MPSGraph" in str(exc)):
                            logger.error("MPSGraph failed and CPU fallback is disabled; aborting task")
                            raise AppError(ErrorCode.MPS_UNAVAILABLE, "Metal execution failed; CPU fallback disabled.")
                        raise

                    pred_out = pred_out if pred_out is not None else prepared

                    if pred_out.dim() == 2:
                        pred_out = pred_out[:, None, :]
                    elif pred_out.dim() == 3 and pred_out.shape[1] not in (1, prepared.shape[1]):
                        pred_out = pred_out.permute(1, 0, 2)

                    pred_out = pred_out[..., : prepared.shape[-1]]
                    if pred_out.dim() == 3 and pred_out.shape[0] == 1:
                        pred_out = pred_out[0]
                    return pred_out

                if model.expects_waveform:
                    prepared = _normalize_waveform_input(model, wave)

                    length = prepared.shape[-1]
                    # The Roformer-based waveform models can exhaust memory on long tracks.
                    # Run them in chunks to keep attention buffer sizes manageable.
                    step = max(1, SEGMENT - OVERLAP)
                    if length <= SEGMENT:
                        if stop_check:
                            stop_check()
                        pred_full = _call_model(prepared)
                        if progress_cb:
                            progress_cb(1.0)
                        return pred_full

                    acc: Optional[torch.Tensor] = None
                    counts = torch.zeros((1, length), device=model.device, dtype=prepared.dtype)

                    for start_idx in range(0, length, step):
                        if stop_check:
                            stop_check()
                        end_idx = min(start_idx + SEGMENT, length)
                        seg = prepared[..., start_idx:end_idx]
                        if seg.shape[-1] < SEGMENT:
                            padded = torch.zeros(
                                (prepared.shape[0], prepared.shape[1], SEGMENT),
                                device=model.device,
                                dtype=prepared.dtype,
                            )
                            padded[..., : seg.shape[-1]] = seg
                        else:
                            padded = seg

                        pred_seg = _call_model(padded)
                        trimmed = pred_seg[..., : seg.shape[-1]]

                        if acc is None:
                            acc = torch.zeros(
                                (trimmed.shape[0], length),
                                device=model.device,
                                dtype=trimmed.dtype,
                            )

                        acc[:, start_idx : start_idx + trimmed.shape[-1]] += trimmed
                        counts[:, start_idx : start_idx + trimmed.shape[-1]] += 1

                        if progress_cb:
                            progress_cb(min(1.0, end_idx / length))

                    if acc is None:
                        raise AppError(ErrorCode.MODEL_MISSING, "Model failed to produce output.")

                    denom = counts.clamp_min(1)
                    return acc / denom
                prepared = ensure_stereo(wave).to(model.device)
                return model(prepared)

            if need_instrumental_model:
                ch("instrumental.start", 0.0)
                inst_model = manager.instrumental
                use_wave = inst_wave if inst_wave is not None else chan_wave

                def _inst_progress(frac: float) -> None:
                    # map chunk progress across the full channel span
                    ch("instrumental.progress", frac)

                if stop_check:
                    stop_check()
                inst_pred = _run_stem_model(inst_model, use_wave, progress_cb=_inst_progress)
                chan_outputs["instrumental"] = inst_pred[:1] if inst_pred.shape[0] > 1 else inst_pred
                ch("instrumental.done", 1.0)

            for stem_name in ["drums", "bass", "other", "guitar"]:
                if stem_name not in remaining_stems:
                    continue
                ch(f"{stem_name}.start", 90)
                inst_model = getattr(manager, stem_name)
                use_wave = inst_wave if inst_wave is not None else chan_wave
                if stop_check:
                    stop_check()
                if inst_model.expects_waveform:
                    pred = _run_stem_model(inst_model, use_wave)
                else:
                    pred = inst_model(ensure_stereo(use_wave))
                chan_outputs[stem_name] = pred[:1] if pred.shape[0] > 1 else pred
                ch(f"{stem_name}.done", 96)

            ch("channel.done", 100)
            for stem_name, tensor in chan_outputs.items():
                channel_stems.setdefault(stem_name, []).append(tensor)
            # free channel-specific references promptly
            del chan_outputs, inst_wave, vocals_wave

        for idx, label, start, end in channel_specs:
            process_channel(idx, label, start, end)

        for stem_name in remaining_stems:
            tensors = channel_stems.get(stem_name)
            if not tensors:
                continue
            combined = torch.cat(tensors, dim=0)
            fname = f"{audio_path.stem} - {stem_name}.wav"
            if stop_check:
                stop_check()
            written = _write_wave(out_dir, fname, combined)
            stems_out.append(written.name)
            cb(f"write.{stem_name}", 98)

    cb("merge.done", 99)
    return stems_out


# ── API endpoints ─────────────────────────────────────────────────────────--

@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), stems: str = Form("vocals")):
    task_id = str(uuid.uuid4())
    pause_evt = threading.Event()
    stop_evt = threading.Event()
    controls[task_id] = {"pause": pause_evt, "stop": stop_evt}
    stem_list = [s for s in stems.split(",") if s]
    logger.info("upload received task=%s filename=%s stems=%s", task_id, file.filename, stem_list)
    try:
        path = UPLOAD_DIR / file.filename
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("wb") as f:
            content = await file.read()
            f.write(content)
        logger.info("persisted upload %s (%s bytes) to %s", file.filename, len(content), path)
        logger.debug("persisted upload bytes head=%s tail=%s", content[:32], content[-32:])
    except Exception as exc:
        logger.exception("failed to persist upload %s", file.filename)
        raise AppError(ErrorCode.UPLOAD_FAILED, f"Failed to persist upload: {exc}").to_http()

    expected: list[str] = []
    for s in stem_list:
        if s == "deux":
            expected.append(f"{Path(file.filename).stem} - vocals (deux).wav")
            expected.append(f"{Path(file.filename).stem} - instrumental (deux).wav")
        else:
            expected.append(f"{Path(file.filename).stem} - {s}.wav")

    progress[task_id] = {"stage": "ready", "pct": 0}
    tasks[task_id] = {
        "dir": None,
        "stems": expected,
        "controls": controls[task_id],
        "conv_src": None,
        "orig_src": str(path),
        "orig_name": file.filename,
        "stem_list": stem_list,
        "status": "ready",
        "structure_mode": settings.structure_mode,
        "staging_dir": None,
        "zip_target": None,
    }
    return {"task_id": task_id, "stems": expected, "status": "ready"}


@app.post("/start/{task_id}")
async def start_task(task_id: str):
    info = tasks.get(task_id)
    if not info:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id").to_http(404)
    if info.get("status") in {"queued", "running", "preparing"}:
        return {"task_id": task_id, "status": info.get("status")}
    orig_src = Path(info.get("orig_src", ""))
    if not orig_src.exists():
        raise AppError(ErrorCode.UPLOAD_FAILED, "Source missing; please re-upload.").to_http()
    plan = resolve_output_plan(info, structure_mode=settings.structure_mode)
    out_dir = plan["staging_dir"] or plan["deliver_dir"]
    tasks[task_id]["dir"] = str(plan["deliver_dir"])
    tasks[task_id]["structure_mode"] = plan["structure_mode"]
    tasks[task_id]["staging_dir"] = str(plan["staging_dir"])
    tasks[task_id]["zip_target"] = str(plan["zip_target"]) if plan.get("zip_target") else None
    tasks[task_id]["status"] = "queued"
    stem_list = info.get("stem_list") or []
    logger.info("starting task %s on demand", task_id)
    _queue_processing(task_id, Path(out_dir), stem_list)
    return {"task_id": task_id, "status": "queued"}


@app.get("/progress/{task_id}")
async def progress_stream(task_id: str):
    async def event_generator():
        def _json_safe(val: object) -> object:
            return str(val) if isinstance(val, Path) else val

        last = None
        while True:
            info = progress.get(task_id, {"stage": "queued", "pct": 0})
            current = (info["stage"], info["pct"])
            if current != last:
                if info.get("stage") == "stopped":
                    payload = {
                        "stage": "stopped",
                        "pct": 0,
                        "stems": info.get("stems"),
                        "out_dir": _json_safe(info.get("out_dir")),
                        "zip": _json_safe(info.get("zip")),
                    }
                    yield {"event": "message", "data": json.dumps(payload)}
                elif info.get("pct", 0) < 0:
                    yield {"event": "error", "data": errors.get(task_id, "processing failed")}
                else:
                    payload = {
                        "stage": info["stage"],
                        "pct": info["pct"],
                        "stems": info.get("stems"),
                        "out_dir": _json_safe(info.get("out_dir")),
                        "zip": _json_safe(info.get("zip")),
                    }
                    yield {"event": "message", "data": json.dumps(payload)}
                last = current
                if info.get("stage") == "stopped" or info.get("pct", 0) >= 100 or info.get("pct", 0) < 0:
                    break
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@app.get("/ready")
async def ready_status():
    return {
        "ready": app_ready_event.is_set(),
        "checking": app_ready_state.get("checking", False),
        "models_missing": app_ready_state.get("models_missing", []),
    }


def _detect_output(out_dir: Path, stems: list[str], zip_hint: str | None = None) -> tuple[bool, str | None]:
    """
    Determine whether an export is complete based on output artifacts.

    Returns a tuple of (is_complete, zip_path_if_found).
    """

    def _all_stems_present() -> bool:
        if not stems:
            return False
        return all((out_dir / stem).exists() for stem in stems)

    zip_path: Path | None = None
    if zip_hint:
        zp = Path(zip_hint)
        if zp.exists():
            zip_path = zp
    if not out_dir.exists() and not zip_path:
        return False, None

    if out_dir.is_file() and out_dir.suffix.lower() == ".zip":
        zip_path = out_dir
        out_dir = out_dir.parent

    if _all_stems_present():
        return True, str(zip_path) if zip_path else None

    if zip_path and Path(zip_path).exists():
        return True, str(zip_path)

    zips = list(out_dir.glob("*.zip"))
    if len(zips) == 1 and zips[0].exists():
        return True, str(zips[0])

    return False, None


@app.post("/rehydrate_tasks")
async def rehydrate_tasks(request: Request):
    """
    Reconcile client-side tasks with on-disk outputs.

    Any task without completed outputs is omitted so abandoned/force-quit jobs
    do not reappear as queued entries.
    """

    payload: dict[str, Any] = await request.json()
    incoming = payload.get("tasks", [])
    if not isinstance(incoming, list):
        raise AppError(ErrorCode.INVALID_REQUEST, "tasks payload must be a list").to_http()

    refreshed: list[dict[str, Any]] = []
    for raw in incoming:
        if not isinstance(raw, dict):
            continue
        out_dir = raw.get("out_dir")
        stems = raw.get("stems") or []
        zip_hint = raw.get("zip")
        if not out_dir:
            continue
        complete, zip_path = _detect_output(Path(out_dir), stems, zip_hint)
        if not complete:
            continue
        refreshed.append(
            {
                "id": raw.get("id"),
                "name": raw.get("name"),
                "stage": "done",
                "pct": 100,
                "stems": stems,
                "out_dir": out_dir,
                "zip": zip_path,
            }
        )
    return {"tasks": refreshed}


@app.post("/rerun/{task_id}")
async def rerun(task_id: str):
    old = tasks.get(task_id)
    if not old:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id").to_http(404)
    conv_src = Path(old.get("conv_src", ""))
    stem_list = old.get("stem_list") or []
    logger.info("rerun requested for task %s", task_id)
    if not conv_src.exists() or not stem_list:
        raise AppError(ErrorCode.RERUN_PREREQ_MISSING, "Missing source or stems for rerun").to_http(409)

    new_id = str(uuid.uuid4())
    controls[new_id] = {"pause": threading.Event(), "stop": threading.Event()}
    plan = resolve_output_plan(old, structure_mode=settings.structure_mode)
    out_dir = plan["staging_dir"] or plan["deliver_dir"]
    expected: list[str] = []
    for s in stem_list:
        if s == "deux":
            expected.append(f"{conv_src.stem} - vocals (deux).wav")
            expected.append(f"{conv_src.stem} - instrumental (deux).wav")
        else:
            expected.append(f"{conv_src.stem} - {s}.wav")

    progress[new_id] = {"stage": "queued", "pct": 0}
    tasks[new_id] = {
        "dir": str(plan["deliver_dir"]),
        "stems": expected,
        "controls": controls[new_id],
        "conv_src": str(conv_src),
        "orig_src": old.get("orig_src"),
        "orig_name": old.get("orig_name"),
        "stem_list": stem_list,
        "zip": old.get("zip"),
        "status": "queued",
        "structure_mode": plan["structure_mode"],
        "staging_dir": str(plan["staging_dir"]),
        "zip_target": str(plan["zip_target"]) if plan.get("zip_target") else None,
    }
    _queue_processing(new_id, Path(out_dir), stem_list)
    return {"task_id": new_id, "stems": expected}


@app.get("/settings")
async def get_settings():
    return settings.as_dict()


@app.post("/settings")
async def update_settings(request: Request):
    payload = await request.json()
    output_root = payload.get("output_root")
    structure_mode = payload.get("structure_mode")
    try:
        settings.update(
            output_root=output_root,
            structure_mode=structure_mode,
        )
    except AppError as exc:
        settings.output_root = DEFAULT_OUTPUT_ROOT
        _ensure_dir(DEFAULT_OUTPUT_ROOT)
        logger.warning("invalid settings update, resetting output root to default: %s", exc.message)
        raise AppError(exc.code, "folder doesn't exist").to_http(400)
    _ensure_dir(settings.output_root)
    return settings.as_dict()


@app.post("/reveal/{task_id}")
async def reveal_output(task_id: str):
    info = tasks.get(task_id)
    if not info:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id").to_http(404)
    out_dir = info.get("dir")
    if not out_dir:
        raise AppError(ErrorCode.INVALID_REQUEST, "Output not ready").to_http(409)
    out_path = Path(out_dir)
    if not out_path.exists():
        raise AppError(ErrorCode.INVALID_REQUEST, "Output path missing").to_http(404)

    def _pick_target() -> Path:
        zip_path = info.get("zip")
        if zip_path:
            zp = Path(zip_path)
            if zp.exists():
                return zp
        stems = info.get("stems") or []
        for stem in stems:
            candidate = out_path / stem
            if candidate.exists():
                return candidate
        return out_path

    target = _pick_target()
    select_path = target if target.is_file() else target
    reveal_dir = target.parent if target.is_file() else target
    try:
        if sys.platform.startswith("darwin"):
            if select_path.is_file():
                subprocess.Popen(["open", "-R", str(select_path)])
            else:
                subprocess.Popen(["open", str(reveal_dir)])
        elif sys.platform.startswith("win"):
            if select_path.is_file():
                subprocess.Popen(["explorer", "/select,", str(select_path)])
            else:
                os.startfile(str(reveal_dir))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(reveal_dir)])
    except Exception as exc:
        logger.warning("failed to reveal folder %s: %s", reveal_dir, exc)
        raise AppError(ErrorCode.INVALID_REQUEST, "Unable to open folder").to_http(500)
    return {"status": "opened", "path": str(select_path if select_path.exists() else reveal_dir)}


def _model_exists(filename: str) -> bool:
    names: set[str] = {filename}
    names.update(MODEL_ALIAS_MAP.get(filename, []))
    for canon, alias_list in MODEL_ALIAS_MAP.items():
        if filename in alias_list:
            names.add(canon)
            names.update(alias_list)
    for name in names:
        if _locate_case_insensitive(MODEL_DIR / name):
            return True
        if _locate_case_insensitive(Path.home() / "Library/Application Support/stems" / name):
            return True
    return False


@app.get("/models_status")
async def models_status():
    missing = []
    for _, (fname, _) in MODEL_FILE_MAP.items():
        if not _model_exists(fname):
            missing.append(fname)
    return {"missing": missing}


@app.get("/download/{task_id}")
async def download(task_id: str):
    info = tasks.get(task_id)
    if not info:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id").to_http(404)
    zip_path = info.get("zip")
    if zip_path and Path(zip_path).exists():
        logger.info("serving download for task %s at %s", task_id, zip_path)
        return FileResponse(path=zip_path, media_type="application/zip", filename=Path(zip_path).name)
    out_dir = info.get("dir")
    stems = info.get("stems") or []
    if out_dir:
        out_path = Path(out_dir)
        if out_path.exists() and out_path.is_file():
            logger.info("serving single-file download for task %s at %s", task_id, out_path)
            return FileResponse(path=out_path, filename=out_path.name)
        if len(stems) == 1:
            candidate = out_path / stems[0]
            if candidate.exists():
                logger.info("serving single-stem download for task %s at %s", task_id, candidate)
                return FileResponse(path=candidate, filename=candidate.name)
    raise AppError(ErrorCode.INVALID_REQUEST, "Files not ready").to_http(409)


@app.post("/clear_all_uploads")
async def clear_all_uploads():
    # stop any running tasks
    for ctrl in controls.values():
        try:
            ctrl.get("pause", threading.Event()).set()
            ctrl.get("stop", threading.Event()).set()
        except Exception:
            logger.debug("failed to signal stop for control %s", ctrl, exc_info=True)

    # drain any queued items so they don't start after clearing
    try:
        while True:
            fn = process_queue.get_nowait()
            process_queue.task_done()
            logger.debug("discarded queued task %s during clear_all_uploads", fn)
    except queue.Empty:
        pass

    # reset in-memory state
    progress.clear()
    tasks.clear()
    errors.clear()
    controls.clear()

    dirs = [UPLOAD_DIR, CONVERTED_DIR]
    for dir_path in dirs:
        if dir_path.exists():
            for entry in dir_path.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
    logger.info("cleared upload directories and reset task state")
    return {"status": "cleared"}


@app.post("/pause/{task_id}")
async def pause_task(task_id: str):
    ctrl = controls.get(task_id)
    if not ctrl:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task").to_http(404)
    ctrl["pause"].set()
    logger.info("paused task %s", task_id)
    return {"status": "paused"}


@app.post("/resume/{task_id}")
async def resume_task(task_id: str):
    ctrl = controls.get(task_id)
    if not ctrl:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task").to_http(404)
    ctrl["pause"].clear()
    logger.info("resumed task %s", task_id)
    return {"status": "resumed"}


@app.post("/stop/{task_id}")
async def stop_task(task_id: str):
    ctrl = controls.get(task_id)
    if not ctrl:
        raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task").to_http(404)
    ctrl["stop"].set()
    logger.info("stopped task %s", task_id)
    return {"status": "stopped"}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "web" / "index.html"
    return HTMLResponse(html_path.read_text())

@app.get("/favicon.ico")
async def favicon():
    icon_path = BASE_DIR / "web" / "favicon.ico"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/x-icon")
    raise AppError(ErrorCode.INVALID_REQUEST, "favicon missing").to_http(404)


@app.api_route("/shutdown", methods=["POST", "GET"])
async def shutdown():
    logger.warning("shutdown requested via api; terminating process")
    try:
        _close_installer_ui()
    except Exception:
        logger.debug("installer ui shutdown during close failed", exc_info=True)

    def _stop():
        logger.debug("shutdown thread armed; sleeping briefly to flush logs")
        time.sleep(0.25)
        os._exit(0)

    threading.Thread(target=_stop, daemon=True).start()
    return {"status": "shutting down"}


# ── CLI entrypoint ─────────────────────────────────────────────────────────-


def _process_local_file(path: Path, stem_list: list[str]) -> list[str]:
    conv_dir = CONVERTED_DIR
    conv_path = _stage_audio_copy(path, conv_dir)
    out_dir = conv_dir / f"{conv_path.stem}—stems"

    def cb(stage: str, pct: int) -> None:
        logging.info("%s: %s%%", stage, pct)

    manager = ModelManager()
    stems_out = _separate_waveform(manager, conv_path, stem_list, cb, out_dir)
    return [str(out_dir / stem) for stem in stems_out]

def cli_main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if "--split" in argv:
        split_args = [arg for arg in argv if arg != "--split"]
        split_main(split_args)
        return
    parser = argparse.ArgumentParser(description="Run stemsplat server or process a file")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI server with uvicorn")
    parser.add_argument("--install", action="store_true", help="Run the installer UI")
    parser.add_argument("files", nargs="*", help="Optional list of audio files to process locally")
    parser.add_argument("--stems", default="vocals", help="Comma-separated stems when processing locally")
    args = parser.parse_args(argv)

    if args.install:
        run_installer_ui()
        return

    if args.serve:
        import uvicorn

        _close_installer_ui()
        if not _port_available(8000):
            logger.error("Port 8000 is already in use; aborting startup")
            raise SystemExit("Port 8000 is already in use. Please free the port and try again.")
        threading.Thread(target=_launch_main_page, daemon=True).start()
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
        return

    if not args.files:
        parser.error("No files provided and --serve not set")

    stem_list = [s for s in args.stems.split(",") if s]
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            raise AppError(ErrorCode.UPLOAD_FAILED, f"File not found: {file_path}")
        outputs = _process_local_file(path, stem_list)
        for out in outputs:
            print(out)


if __name__ == "__main__":  # pragma: no cover
    cli_main()
