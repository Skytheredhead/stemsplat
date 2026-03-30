from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import importlib
import inspect
import json
import logging
import math
import os
import platform
import queue
import re
import resource
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import webbrowser
import zipfile
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from pathlib import Path
from packaging import version
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf
import torch
import yaml
from beartype import beartype
from beartype.typing import Callable as BeartypeCallable, Optional as BeartypeOptional, Tuple as BeartypeTuple
from downloader import SSL_CONTEXT, download_to
from einops import pack, rearrange, reduce, repeat, unpack
from app_paths import (
    ARTWORK_DIR,
    CONFIG_DIR,
    INTERMEDIATE_CACHE_DIR,
    LOG_DIR,
    MODEL_DIR,
    OUTPUT_ROOT,
    RESOURCE_DIR,
    RUNTIME_DIR,
    ETA_HISTORY_PATH,
    SETTINGS_PATH,
    UPLOAD_DIR,
    WEB_DIR,
    WORK_DIR,
    ensure_app_dirs,
    model_search_dirs,
)
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from rotary_embedding_torch import RotaryEmbedding
from starlette.background import BackgroundTask
from torch import einsum, nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

try:
    from mutagen import File as MutagenFile
except Exception:  # pragma: no cover - optional cleanup dependency
    MutagenFile = None

BASE_DIR = RESOURCE_DIR

try:  # pragma: no cover - import failure is surfaced at runtime
    from split.mel_band_roformer import MelBandRoformer, _mel_filter_bank
except Exception as exc:  # pragma: no cover - keep app importable for syntax checks
    MelBandRoformer = None  # type: ignore[assignment]
    _mel_filter_bank = None  # type: ignore[assignment]
    _model_import_error = exc
else:
    _model_import_error = None


def _optional_import(module_name: str) -> Any | None:
    with contextlib.suppress(Exception):
        return importlib.import_module(module_name)
    return None


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


def _hz_to_mel(frequencies: np.ndarray | float, *, htk: bool = False) -> np.ndarray:
    frequencies = np.asanyarray(frequencies, dtype=np.float32)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    f_sp = 200.0 / 3
    mels = frequencies / f_sp

    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def _mel_to_hz(mels: np.ndarray | float, *, htk: bool = False) -> np.ndarray:
    mels = np.asanyarray(mels, dtype=np.float32)

    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_sp = 200.0 / 3
    freqs = f_sp * mels

    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def _mel_frequencies(
    n_mels: int, *, fmin: float, fmax: float, htk: bool = False
) -> np.ndarray:
    min_mel = _hz_to_mel(fmin, htk=htk)
    max_mel = _hz_to_mel(fmax, htk=htk)
    return _mel_to_hz(
        np.linspace(min_mel, max_mel, n_mels, dtype=np.float32), htk=htk
    )


def _local_mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    fmax = float(sample_rate) / 2 if fmax is None else fmax
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=np.float32)

    fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    mel_freqs = _mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=False)
    fdiff = np.diff(mel_freqs)
    ramps = np.subtract.outer(mel_freqs, fft_freqs)

    for index in range(n_mels):
        lower = -ramps[index] / fdiff[index]
        upper = ramps[index + 2] / fdiff[index + 1]
        weights[index] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_freqs[2 : n_mels + 2] - mel_freqs[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


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
        legacy_depth=False,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor
        mlp_depth = max(1, depth - 1) if legacy_depth else depth

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=mlp_depth),
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
        mlp_expansion_factor=4,
        freqs_per_bands: BeartypeOptional[BeartypeTuple[int, ...]] = None,
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

        band_count = len(freqs_per_bands) if freqs_per_bands is not None else num_bands

        if freqs_per_bands is not None:
            assert sum(freqs_per_bands) == freqs, "freqs_per_bands must sum to the STFT frequency bins"
            freqs_per_band = torch.zeros((band_count, freqs), dtype=torch.bool)
            start = 0
            for band_index, band_width in enumerate(freqs_per_bands):
                end = start + band_width
                freqs_per_band[band_index, start:end] = True
                start = end
        else:
            mel_filter_bank_fn = _mel_filter_bank or _local_mel_filter_bank

            mel_filter_bank_numpy = mel_filter_bank_fn(
                sample_rate=sample_rate,
                n_fft=stft_n_fft,
                n_mels=num_bands,
            )

            mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

            mel_filter_bank[0][0] = 1.0
            mel_filter_bank[-1, -1] = 1.0

            freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), (
            "all frequencies need to be covered by all bands for now"
        )

        repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=band_count)
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
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
                legacy_depth=freqs_per_bands is not None,
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
    MODEL_IMPORT_FAILED = "E001"
    MODEL_MISSING = "E002"
    CONFIG_MISSING = "E003"
    FFMPEG_MISSING = "E004"
    AUDIO_DECODE_FAILED = "E005"
    AUDIO_LOAD_FAILED = "E006"
    SEPARATION_FAILED = "E007"
    TASK_NOT_FOUND = "E008"
    INVALID_REQUEST = "E009"


@dataclass
class AppError(Exception):
    code: ErrorCode
    message: str

    def to_http(self, status: int = 400) -> HTTPException:
        return HTTPException(status_code=status, detail={"code": self.code, "message": self.message})


class TaskStopped(Exception):
    pass


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    config: str
    segment: int
    overlap: int = 2


@dataclass(frozen=True)
class SourceInfo:
    suffix: str
    codec: str | None
    bit_rate: int | None
    channels: int
    has_cover: bool
    has_video: bool


@dataclass(frozen=True)
class ExportPlan:
    suffix: str
    audio_args: list[str]
    supports_cover: bool


LOG_PATH = LOG_DIR / "main_stemsplat.log"
MODEL_SEARCH_DIRS = model_search_dirs()
APP_VERSION = "0.3.0"
GITHUB_REPO = "Skytheredhead/stemsplat"
GITHUB_LATEST_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
WATCHDOG_INTERVAL_SECONDS = 5.0
WATCHDOG_CONFIRM_SAMPLES = 3
TASK_STALL_TIMEOUT_SECONDS = 300.0
TASK_HARD_TIMEOUT_SECONDS = 7200.0
PROCESS_RSS_HEADROOM_RATIO = 0.82
PROCESS_RSS_LIMIT_FALLBACK_BYTES = 8 * 1024 * 1024 * 1024
MPS_MEMORY_HEADROOM_RATIO = 0.88
UPDATE_CHECK_INTERVAL_SEC = 12 * 60 * 60
LAN_AUTH_COOKIE_NAME = "stemsplat_lan_auth"
LAN_AUTH_TTL_CHOICES = {"15m", "1d", "1w", "never"}
LAN_AUTH_ALLOWED_PATHS = {"/", "/favicon.ico", "/api/lan_auth"}
TERMINAL_TASK_RETENTION_LIMIT = 100
MODEL_PROMPT_PENDING = "pending"
MODEL_PROMPT_DISMISSED = "dismissed"
MODEL_PROMPT_ACCEPTED = "accepted"
MODEL_PROMPT_COMPLETE = "complete"
PRESET_GAIN_DB_MIN = -18.0
PRESET_GAIN_DB_MAX = 18.0
PRESET_DEFAULT_OVERLAY_GAIN_DB = 3.0
PRESET_DEFAULT_BASE_GAIN_DB = -3.0
BOOST_HARMONIES_DEFAULT_BACKGROUND_GAIN_DB = PRESET_DEFAULT_OVERLAY_GAIN_DB
BOOST_HARMONIES_DEFAULT_BASE_GAIN_DB = PRESET_DEFAULT_BASE_GAIN_DB
BOOST_GUITAR_DEFAULT_GUITAR_GAIN_DB = PRESET_DEFAULT_OVERLAY_GAIN_DB
BOOST_GUITAR_DEFAULT_BASE_GAIN_DB = PRESET_DEFAULT_BASE_GAIN_DB
BOOST_HARMONIES_VOCALS_END_PCT = 44
BOOST_HARMONIES_BACKGROUND_START_PCT = 48
BOOST_HARMONIES_BACKGROUND_END_PCT = 92
BOOST_GUITAR_MODEL_END_PCT = 92
INTERMEDIATE_CACHE_RETENTION_SECONDS = 7 * 24 * 60 * 60
COMPAT_SETTINGS_DEFAULTS = {
    "output_format": "same_as_input",
    "output_root": str(OUTPUT_ROOT),
    "output_root_migrated_to_downloads": False,
    "output_same_as_input": False,
    "video_handling": "audio_only",
    "structure_mode": "flat",
    "model_prompt_state": MODEL_PROMPT_PENDING,
    "update_last_checked_at": 0.0,
    "update_latest_version": "",
    "update_latest_name": "",
    "update_latest_url": "",
    "update_latest_notes": "",
    "update_last_notified_version": "",
    "update_skipped_version": "",
    "lan_passcode_enabled": False,
    "lan_passcode": "",
    "lan_passcode_ttl": "1d",
    "boost_harmonies_background_vocals_gain_db": BOOST_HARMONIES_DEFAULT_BACKGROUND_GAIN_DB,
    "boost_harmonies_base_song_gain_db": BOOST_HARMONIES_DEFAULT_BASE_GAIN_DB,
    "boost_guitar_guitar_gain_db": BOOST_GUITAR_DEFAULT_GUITAR_GAIN_DB,
    "boost_guitar_base_song_gain_db": BOOST_GUITAR_DEFAULT_BASE_GAIN_DB,
}

MODEL_SPECS: dict[str, ModelSpec] = {
    "vocals": ModelSpec(
        filename="mel_band_roformer_vocals_becruily.ckpt",
        config="Mel Band Roformer Vocals Config.yaml",
        segment=352_800,
        overlap=2,
    ),
    "instrumental": ModelSpec(
        filename="mel_band_roformer_instrumental_becruily.ckpt",
        config="Mel Band Roformer Instrumental Config.yaml",
        segment=352_800,
        overlap=2,
    ),
    "deux": ModelSpec(
        filename="becruily_deux.ckpt",
        config="config_deux_becruily.yaml",
        segment=573_300,
        overlap=2,
    ),
    "guitar": ModelSpec(
        filename="becruily_guitar.ckpt",
        config="config_guitar_becruily.yaml",
        segment=485_100,
        overlap=2,
    ),
    "mel_band_karaoke": ModelSpec(
        filename="mel_band_roformer_karaoke_becruily.ckpt",
        config="config_karaoke_becruily.yaml",
        segment=485_100,
        overlap=8,
    ),
    "denoise": ModelSpec(
        filename="denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
        config="model_mel_band_roformer_denoise.yaml",
        segment=352_800,
        overlap=4,
    ),
    "bs_roformer_6s": ModelSpec(
        filename="BS-Rofo-SW-Fixed.ckpt",
        config="BS-Rofo-SW-Fixed.yaml",
        segment=588_800,
        overlap=2,
    ),
}

MODEL_ALIAS_MAP = {
    "mel_band_roformer_vocals_becruily.ckpt": ["Mel Band Roformer Vocals.ckpt"],
    "mel_band_roformer_instrumental_becruily.ckpt": ["Mel Band Roformer Instrumental.ckpt"],
    "becruily_deux.ckpt": ["Mel Band Roformer Deux.ckpt"],
    "becruily_guitar.ckpt": ["Mel Band Roformer Guitar.ckpt"],
    "mel_band_roformer_karaoke_becruily.ckpt": ["Mel Band Roformer Karaoke.ckpt"],
    "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt": ["Mel Band Roformer Denoise.ckpt"],
    "BS-Rofo-SW-Fixed.ckpt": ["BS-Rofo-SW-Fixed-v1.ckpt", "BS Rofo SW Fixed.ckpt"],
}

MODEL_URLS = {
    "vocals": "https://huggingface.co/becruily/mel-band-roformer-vocals/resolve/main/mel_band_roformer_vocals_becruily.ckpt?download=true",
    "instrumental": "https://huggingface.co/becruily/mel-band-roformer-instrumental/resolve/main/mel_band_roformer_instrumental_becruily.ckpt?download=true",
    "deux": "https://huggingface.co/becruily/mel-band-roformer-deux/resolve/main/becruily_deux.ckpt?download=true",
    "guitar": "https://huggingface.co/becruily/mel-band-roformer-guitar/resolve/main/becruily_guitar.ckpt?download=true",
    "mel_band_karaoke": "https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt?download=true",
    "denoise": "https://huggingface.co/jarredou/aufr33_MelBand_Denoise/resolve/main/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt?download=true",
    "bs_roformer_6s": "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt?download=true",
}
MODEL_DISPLAY_NAMES = {
    "vocals": "vocals",
    "instrumental": "instrumental",
    "deux": "both - deux model",
    "guitar": "mel-band guitar",
    "mel_band_karaoke": "mel-band karaoke",
    "denoise": "mel-band denoise",
    "bs_roformer_6s": "bs-roformer 6s",
}
MODE_TO_STEMS = {
    "vocals": ("vocals",),
    "instrumental": ("instrumental",),
    "both_deux": ("deux",),
    "both_separate": ("vocals", "instrumental"),
    "guitar": ("guitar",),
    "mel_band_karaoke": ("mel_band_karaoke",),
    "bs_roformer_6s": ("bs_roformer_6s",),
    "preset_boost_harmonies": ("boost_harmonies",),
    "preset_boost_guitar": ("boost_guitar",),
    "preset_denoise": ("denoise",),
}
MODE_REQUIRED_MODELS = {
    "vocals": ("vocals",),
    "instrumental": ("instrumental",),
    "both_deux": ("deux",),
    "both_separate": ("vocals", "instrumental"),
    "guitar": ("guitar",),
    "mel_band_karaoke": ("mel_band_karaoke",),
    "bs_roformer_6s": ("bs_roformer_6s",),
    "preset_boost_harmonies": ("vocals", "mel_band_karaoke"),
    "preset_boost_guitar": ("guitar",),
    "preset_denoise": ("denoise",),
}
MODE_OUTPUT_LABELS = {
    "vocals": ("vocals",),
    "instrumental": ("instrumental",),
    "both_deux": ("vocals", "instrumental"),
    "both_separate": ("vocals", "instrumental"),
    "guitar": ("guitar",),
    "mel_band_karaoke": ("vocals", "karaoke"),
    "bs_roformer_6s": ("bass", "drums", "other", "vocals", "guitar", "piano"),
    "preset_boost_harmonies": ("boost harmonies",),
    "preset_boost_guitar": ("boost guitar",),
    "preset_denoise": ("denoise",),
}
ETA_HISTORY_KEYS = tuple(MODEL_SPECS.keys())
ETA_HISTORY_LIMIT = 10
ETA_PREP_OVERHEAD_SECONDS = 5.0
ETA_EXPORT_PER_OUTPUT_SECONDS = 2.5
MODEL_PROGRESS_START_PCT = 6
SINGLE_MODEL_PROGRESS_END_PCT = 95
DEUX_MODEL_PROGRESS_END_PCT = 95
BOTH_SEPARATE_VOCALS_END_PCT = 48
BOTH_SEPARATE_INSTRUMENTAL_START_PCT = 50
BOTH_SEPARATE_INSTRUMENTAL_END_PCT = 95
EXPORT_PROGRESS_START_PCT = 96
EXPORT_PROGRESS_END_PCT = 99

MODE_CHOICES = set(MODE_TO_STEMS)
OUTPUT_FORMAT_CHOICES = {"same_as_input", "mp3_320", "mp3_128", "wav", "m4a", "flac"}
VIDEO_HANDLING_CHOICES = {"audio_only"}
TERMINAL_STATUSES = {"done", "error", "stopped"}
SUPPORTED_MEDIA_SUFFIXES = {
    ".wav",
    ".wave",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".aif",
    ".aiff",
    ".alac",
    ".ogg",
    ".opus",
    ".mp4",
    ".m4v",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
}
UPLOAD_CHUNK_SIZE = 1024 * 1024
RUNTIME_CLEANUP_MAX_AGE_SEC = 24 * 60 * 60


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.name}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    stem = path.stem
    suffix = path.suffix
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _safe_stem(name: str) -> str:
    raw = Path(name).stem or "split"
    cleaned = re.sub(r"[^\w .-]+", "_", raw, flags=re.ASCII).strip(" ._")
    return cleaned or "split"


def _locate_case_insensitive(path: Path) -> Path | None:
    if path.exists():
        return path
    parent = path.parent
    if not parent.exists():
        return None
    target = path.name.lower()
    for candidate in parent.iterdir():
        if candidate.name.lower() == target:
            return candidate
    return None


compat_settings_lock = threading.RLock()


def _coerce_gain_db(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    clamped = max(PRESET_GAIN_DB_MIN, min(PRESET_GAIN_DB_MAX, parsed))
    return round(clamped * 10.0) / 10.0


def _boost_harmonies_settings_payload(settings: dict[str, Any] | None = None) -> dict[str, float]:
    source = settings or {}
    return {
        "overlay_gain_db": _coerce_gain_db(
            source.get("boost_harmonies_background_vocals_gain_db"),
            BOOST_HARMONIES_DEFAULT_BACKGROUND_GAIN_DB,
        ),
        "base_song_gain_db": _coerce_gain_db(
            source.get("boost_harmonies_base_song_gain_db"),
            BOOST_HARMONIES_DEFAULT_BASE_GAIN_DB,
        ),
    }


def _boost_guitar_settings_payload(settings: dict[str, Any] | None = None) -> dict[str, float]:
    source = settings or {}
    return {
        "overlay_gain_db": _coerce_gain_db(
            source.get("boost_guitar_guitar_gain_db"),
            BOOST_GUITAR_DEFAULT_GUITAR_GAIN_DB,
        ),
        "base_song_gain_db": _coerce_gain_db(
            source.get("boost_guitar_base_song_gain_db"),
            BOOST_GUITAR_DEFAULT_BASE_GAIN_DB,
        ),
    }


def _preset_settings_payload_for_mode(mode: str, settings: dict[str, Any] | None = None) -> dict[str, float] | None:
    if mode == "preset_boost_harmonies":
        return _boost_harmonies_settings_payload(settings)
    if mode == "preset_boost_guitar":
        return _boost_guitar_settings_payload(settings)
    return None


def _normalize_settings_payload(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(COMPAT_SETTINGS_DEFAULTS)
    normalized.update(settings)
    output_root = Path(str(normalized.get("output_root") or OUTPUT_ROOT)).expanduser()
    legacy_documents_root = (Path.home() / "Documents").expanduser()
    migrated_to_downloads = bool(normalized.get("output_root_migrated_to_downloads"))
    if not migrated_to_downloads and output_root == legacy_documents_root:
        output_root = OUTPUT_ROOT
        migrated_to_downloads = True
    elif output_root == OUTPUT_ROOT:
        migrated_to_downloads = True
    normalized["output_root"] = str(output_root)
    normalized["output_root_migrated_to_downloads"] = migrated_to_downloads
    normalized["output_same_as_input"] = bool(normalized.get("output_same_as_input"))
    normalized["lan_passcode_enabled"] = bool(normalized.get("lan_passcode_enabled"))
    normalized["lan_passcode"] = str(normalized.get("lan_passcode") or "")[:24]
    if str(normalized.get("lan_passcode_ttl") or "") not in LAN_AUTH_TTL_CHOICES:
        normalized["lan_passcode_ttl"] = "1d"
    if str(normalized.get("video_handling") or "") not in VIDEO_HANDLING_CHOICES:
        normalized["video_handling"] = "audio_only"
    boost_harmonies_settings = _boost_harmonies_settings_payload(normalized)
    normalized["boost_harmonies_background_vocals_gain_db"] = boost_harmonies_settings["overlay_gain_db"]
    normalized["boost_harmonies_base_song_gain_db"] = boost_harmonies_settings["base_song_gain_db"]
    boost_guitar_settings = _boost_guitar_settings_payload(normalized)
    normalized["boost_guitar_guitar_gain_db"] = boost_guitar_settings["overlay_gain_db"]
    normalized["boost_guitar_base_song_gain_db"] = boost_guitar_settings["base_song_gain_db"]
    normalized["structure_mode"] = "flat"
    return normalized


def _load_compat_settings() -> dict[str, Any]:
    settings = dict(COMPAT_SETTINGS_DEFAULTS)
    if not SETTINGS_PATH.exists():
        return _normalize_settings_payload(settings)
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key, value in data.items():
                if key in settings:
                    settings[key] = value
    except Exception:
        logging.getLogger("stemsplat").debug("failed to load settings from %s", SETTINGS_PATH, exc_info=True)
    return _normalize_settings_payload(settings)


def _load_eta_history() -> dict[str, list[dict[str, float]]]:
    payload: dict[str, list[dict[str, float]]] = {key: [] for key in ETA_HISTORY_KEYS}
    if not ETA_HISTORY_PATH.exists():
        return payload
    try:
        data = json.loads(ETA_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        logging.getLogger("stemsplat").debug("failed to load eta history from %s", ETA_HISTORY_PATH, exc_info=True)
        return payload
    if not isinstance(data, dict):
        return payload
    for key in ETA_HISTORY_KEYS:
        raw_entries = data.get(key)
        if not isinstance(raw_entries, list):
            continue
        normalized_entries: list[dict[str, float]] = []
        for entry in raw_entries[-ETA_HISTORY_LIMIT:]:
            if not isinstance(entry, dict):
                continue
            try:
                audio_seconds = float(entry.get("audio_seconds") or 0.0)
                elapsed_seconds = float(entry.get("elapsed_seconds") or 0.0)
            except Exception:
                continue
            if audio_seconds <= 0 or elapsed_seconds <= 0:
                continue
            normalized_entries.append(
                {
                    "audio_seconds": audio_seconds,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
        payload[key] = normalized_entries[-ETA_HISTORY_LIMIT:]
    return payload


def _save_eta_history(history: dict[str, list[dict[str, float]]]) -> None:
    ETA_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _save_compat_settings(settings: dict[str, Any]) -> None:
    payload = _normalize_settings_payload(settings)
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compat_settings_payload() -> dict[str, Any]:
    with compat_settings_lock:
        return dict(_compat_settings)


def _set_compat_settings(patch: dict[str, Any]) -> dict[str, Any]:
    with compat_settings_lock:
        current = _normalize_settings_payload(dict(_compat_settings))
        updated = dict(current)
        updated.update(patch)
        updated = _normalize_settings_payload(updated)
        _save_compat_settings(updated)
        _compat_settings.clear()
        _compat_settings.update(updated)
        changed_security = any(
            current.get(key) != updated.get(key)
            for key in ("lan_passcode_enabled", "lan_passcode", "lan_passcode_ttl")
        )
        result = dict(_compat_settings)
    if changed_security:
        _reset_lan_auth_sessions()
    return result


def _current_output_root() -> Path:
    with compat_settings_lock:
        raw = str(_compat_settings.get("output_root") or OUTPUT_ROOT)
    return _ensure_dir(Path(raw).expanduser())


def _is_remote_client(request: Request | None) -> bool:
    if request is None or request.client is None:
        return False
    host = str(request.client.host or "").strip().lower()
    return host not in {"127.0.0.1", "::1", "localhost"}


def _should_use_mobile_ui(request: Request | None) -> bool:
    if not _is_remote_client(request):
        return False
    user_agent = str((request.headers.get("user-agent") if request else "") or "").lower()
    mobile_markers = (
        "iphone",
        "ipad",
        "ipod",
        "android",
        "mobile",
        "windows phone",
        "blackberry",
    )
    return any(marker in user_agent for marker in mobile_markers)


def _lan_auth_ttl_seconds(ttl_value: str | None) -> int | None:
    normalized = str(ttl_value or "").strip().lower()
    if normalized == "15m":
        return 15 * 60
    if normalized == "1d":
        return 24 * 60 * 60
    if normalized == "1w":
        return 7 * 24 * 60 * 60
    return None


def _reset_lan_auth_sessions() -> None:
    with lan_auth_lock:
        lan_auth_sessions.clear()


def _lan_passcode_required(request: Request | None) -> bool:
    if not _is_remote_client(request):
        return False
    settings = _compat_settings_payload()
    return bool(settings.get("lan_passcode_enabled")) and bool(str(settings.get("lan_passcode") or "").strip())


def _request_client_host(request: Request | None) -> str:
    if request is None or request.client is None:
        return ""
    return str(request.client.host or "").strip()


def _require_local_request(request: Request, message: str = "This action is only available on the host machine.") -> None:
    if _is_remote_client(request):
        raise AppError(ErrorCode.INVALID_REQUEST, message).to_http(403)


def _prune_lan_auth_sessions_locked(now: float | None = None) -> None:
    current = time.time() if now is None else now
    expired_tokens = [
        token
        for token, session in lan_auth_sessions.items()
        if isinstance(session.get("expires_at"), (int, float)) and float(session["expires_at"]) <= current
    ]
    for token in expired_tokens:
        lan_auth_sessions.pop(token, None)


def _request_has_valid_lan_session(request: Request) -> bool:
    token = request.cookies.get(LAN_AUTH_COOKIE_NAME)
    if not token:
        return False
    client_host = _request_client_host(request)
    now = time.time()
    with lan_auth_lock:
        _prune_lan_auth_sessions_locked(now)
        session = lan_auth_sessions.get(token)
        if not isinstance(session, dict):
            return False
        session_host = str(session.get("host") or "")
        if not session_host or session_host != client_host:
            lan_auth_sessions.pop(token, None)
            return False
        expires_at = session.get("expires_at")
        if isinstance(expires_at, (int, float)) and float(expires_at) <= now:
            lan_auth_sessions.pop(token, None)
            return False
        return True


def _load_lan_login_page() -> str:
    candidate = WEB_DIR / "lan_login.html"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return """<!DOCTYPE html><html><body style="background:#0f2027;color:#fff;font-family:sans-serif;display:grid;place-items:center;min-height:100vh;">lan login page missing</body></html>"""


def _lan_unauthorized_response(request: Request) -> HTMLResponse | JSONResponse:
    if request.method in {"GET", "HEAD"} and request.url.path == "/":
        return HTMLResponse(_load_lan_login_page())
    response = JSONResponse(status_code=401, content={"error": "lan passcode required", "requires_auth": True})
    response.delete_cookie(LAN_AUTH_COOKIE_NAME, path="/")
    return response


def _normalize_version_parts(raw: str) -> tuple[int, ...]:
    matches = [int(part) for part in re.findall(r"\d+", raw or "")]
    return tuple(matches or [0])


def _is_newer_version(candidate: str, current: str) -> bool:
    left = list(_normalize_version_parts(candidate))
    right = list(_normalize_version_parts(current))
    width = max(len(left), len(right))
    left.extend([0] * (width - len(left)))
    right.extend([0] * (width - len(right)))
    return tuple(left) > tuple(right)


def _fetch_latest_release() -> dict[str, Any]:
    req = urllib.request.Request(
        GITHUB_LATEST_RELEASE_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"stemsplat/{APP_VERSION} ({platform.system()})",
        },
    )
    with urllib.request.urlopen(req, timeout=5, context=SSL_CONTEXT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        "version": str(payload.get("tag_name") or "").strip(),
        "name": str(payload.get("name") or "").strip(),
        "url": str(payload.get("html_url") or "").strip(),
        "notes": str(payload.get("body") or "").strip(),
    }


def _release_status_payload(*, refresh: bool = False) -> dict[str, Any]:
    now = time.time()
    settings = _compat_settings_payload()
    last_checked = float(settings.get("update_last_checked_at") or 0.0)
    cached = {
        "current_version": APP_VERSION,
        "latest_version": str(settings.get("update_latest_version") or ""),
        "latest_name": str(settings.get("update_latest_name") or ""),
        "latest_url": str(settings.get("update_latest_url") or ""),
        "notes": str(settings.get("update_latest_notes") or ""),
        "last_checked_at": last_checked,
        "last_notified_version": str(settings.get("update_last_notified_version") or ""),
        "skipped_version": str(settings.get("update_skipped_version") or ""),
    }
    should_refresh = refresh or (now - last_checked >= UPDATE_CHECK_INTERVAL_SEC) or not cached["latest_version"]
    if should_refresh:
        try:
            latest = _fetch_latest_release()
        except Exception:
            logger.debug("release check failed", exc_info=True)
        else:
            cached.update(
                {
                    "latest_version": latest["version"],
                    "latest_name": latest["name"],
                    "latest_url": latest["url"],
                    "notes": latest["notes"],
                    "last_checked_at": now,
                }
            )
            _set_compat_settings(
                {
                    "update_last_checked_at": now,
                    "update_latest_version": latest["version"],
                    "update_latest_name": latest["name"],
                    "update_latest_url": latest["url"],
                    "update_latest_notes": latest["notes"],
                }
            )
    cached["update_available"] = bool(cached["latest_version"]) and _is_newer_version(
        cached["latest_version"], cached["current_version"]
    )
    return cached


def _mode_to_stems(mode: str) -> list[str]:
    return list(MODE_TO_STEMS.get(mode, ()))


def _stems_to_mode(stems_raw: str) -> str:
    stems = [item.strip().lower() for item in stems_raw.split(",") if item.strip()]
    for mode, expected in MODE_TO_STEMS.items():
        if stems == list(expected):
            return mode
    if sorted(stems) == ["instrumental", "vocals"]:
        return "both_separate"
    raise AppError(ErrorCode.INVALID_REQUEST, "Invalid stem selection.")


_compat_settings = _load_compat_settings()
eta_history_lock = threading.RLock()
eta_history = _load_eta_history()
lan_auth_lock = threading.RLock()
lan_auth_sessions: dict[str, dict[str, Any]] = {}


ensure_app_dirs()
stream_handler = logging.StreamHandler()
file_handler: logging.Handler | None
try:
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
except Exception:
    file_handler = None

handlers = [stream_handler]
if file_handler is not None:
    handlers.append(file_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("stemsplat")
if file_handler is None:
    logger.warning("file logging unavailable at %s; continuing with console logging only", LOG_PATH)
logging.getLogger("python_multipart").setLevel(logging.INFO)
for uvicorn_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    uvicorn_logger = logging.getLogger(uvicorn_name)
    uvicorn_logger.setLevel(logging.INFO)
    if file_handler is not None and file_handler not in uvicorn_logger.handlers:
        uvicorn_logger.addHandler(file_handler)

for path in (MODEL_DIR, RUNTIME_DIR, UPLOAD_DIR, WORK_DIR, OUTPUT_ROOT):
    _ensure_dir(path)


def _close_installer_ui(port: int = 6060) -> None:
    url = f"http://localhost:{port}/installer_shutdown"
    try:
        with urllib.request.urlopen(url, timeout=1):
            logger.debug("closed installer ui on %s", url)
    except Exception:
        logger.debug("installer ui not reachable at %s", url)


def _cleanup_old_runtime_entries(path: Path, max_age_seconds: int = RUNTIME_CLEANUP_MAX_AGE_SEC) -> None:
    cutoff = time.time() - max_age_seconds
    for candidate in path.iterdir():
        try:
            if candidate.stat().st_mtime >= cutoff:
                continue
            _cleanup_path(candidate)
        except Exception:
            logger.debug("failed to cleanup runtime entry %s", candidate, exc_info=True)


def _required_models_for_mode(mode: str) -> list[str]:
    return list(MODE_REQUIRED_MODELS.get(mode, ()))


def _model_file_exists(filename: str) -> bool:
    search_names = [filename, *MODEL_ALIAS_MAP.get(filename, [])]
    for base_dir in MODEL_SEARCH_DIRS:
        for search_name in search_names:
            if _locate_case_insensitive(base_dir / search_name):
                return True
    return False


def _find_missing_models_for_mode(mode: str) -> list[str]:
    missing: list[str] = []
    for key in _required_models_for_mode(mode):
        if not _model_file_exists(MODEL_SPECS[key].filename):
            missing.append(key)
    return missing


def _validate_models_for_mode(mode: str) -> None:
    missing = _find_missing_models_for_mode(mode)
    if missing:
        joined = ", ".join(missing)
        raise AppError(ErrorCode.MODEL_MISSING, f"Missing required model files: {joined}.")


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("localhost", port))
            return True
        except OSError:
            return False


def _ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    candidates: list[str] = []
    if path:
        candidates.append(path)

    try:  # pragma: no cover - optional fallback dependency
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            candidates.append(bundled)
    except Exception:
        logger.debug("imageio-ffmpeg unavailable; relying on PATH")

    for candidate in candidates:
        try:
            subprocess.run(
                [candidate, "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return candidate
        except Exception:
            logger.debug("ffmpeg candidate failed: %s", candidate, exc_info=True)

    raise AppError(ErrorCode.FFMPEG_MISSING, "ffmpeg not found; install ffmpeg or add imageio-ffmpeg.")


def _ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if path:
        return path
    ffmpeg_path = _ensure_ffmpeg()
    probe_candidate = str(Path(ffmpeg_path).with_name("ffprobe"))
    if Path(probe_candidate).exists():
        return probe_candidate
    raise AppError(ErrorCode.FFMPEG_MISSING, "ffprobe not found; install ffmpeg.")


def _fallback_source_info(path: Path) -> SourceInfo:
    channels = 2
    bit_rate: int | None = None
    has_cover = False
    codec: str | None = None

    if MutagenFile is not None:
        try:
            audio = MutagenFile(path)
            if audio is not None:
                info = getattr(audio, "info", None)
                if info is not None:
                    raw_channels = getattr(info, "channels", None)
                    raw_bit_rate = getattr(info, "bitrate", None)
                    raw_codec = getattr(info, "codec", None) or getattr(info, "codec_description", None)
                    if raw_channels:
                        channels = max(1, min(2, int(raw_channels)))
                    if raw_bit_rate:
                        bit_rate = int(raw_bit_rate)
                    if raw_codec:
                        codec = str(raw_codec).lower()

                tags = getattr(audio, "tags", None)
                if tags is not None:
                    has_cover = any(
                        str(key).lower() in {"apic", "covr", "metadata_block_picture"}
                        for key in tags.keys()
                    )
        except Exception:
            logger.debug("mutagen probe failed for %s", path, exc_info=True)

    return SourceInfo(
        suffix=path.suffix.lower(),
        codec=codec,
        bit_rate=bit_rate,
        channels=channels,
        has_cover=has_cover,
        has_video=False,
    )


def _probe_source(path: Path) -> SourceInfo:
    fallback = _fallback_source_info(path)
    try:
        cmd = [
            _ffprobe_path(),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,bit_rate,channels,disposition:format=bit_rate",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout or "{}")
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        return fallback

    streams = data.get("streams") or []
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    has_cover = any(bool((stream.get("disposition") or {}).get("attached_pic")) for stream in video_streams)
    has_video = any(not bool((stream.get("disposition") or {}).get("attached_pic")) for stream in video_streams)
    bit_rate = audio_stream.get("bit_rate") or (data.get("format") or {}).get("bit_rate")
    with contextlib.suppress(Exception):
        bit_rate = int(bit_rate) if bit_rate else None
    return SourceInfo(
        suffix=path.suffix.lower(),
        codec=audio_stream.get("codec_name") or fallback.codec,
        bit_rate=bit_rate if isinstance(bit_rate, int) else fallback.bit_rate,
        channels=max(1, min(2, int(audio_stream.get("channels") or fallback.channels or 2))),
        has_cover=has_cover or fallback.has_cover,
        has_video=has_video,
    )


def _clamp_kbps(bit_rate: int | None, fallback: int) -> int:
    if not bit_rate:
        return fallback
    return max(96, min(320, int(round(bit_rate / 1000.0))))


def _resolve_export_plan(source_info: SourceInfo, selection: str) -> ExportPlan:
    if selection == "mp3_320":
        return ExportPlan(".mp3", ["-c:a", "libmp3lame", "-b:a", "320k", "-id3v2_version", "3"], True)
    if selection == "mp3_128":
        return ExportPlan(".mp3", ["-c:a", "libmp3lame", "-b:a", "128k", "-id3v2_version", "3"], True)
    if selection == "wav":
        return ExportPlan(".wav", ["-c:a", "pcm_s16le"], False)
    if selection == "m4a":
        return ExportPlan(".m4a", ["-c:a", "aac", "-b:a", "256k"], True)
    if selection == "flac":
        return ExportPlan(".flac", ["-c:a", "flac"], True)

    codec = (source_info.codec or "").lower()
    suffix = source_info.suffix
    if suffix == ".mp3" or codec == "mp3":
        kbps = _clamp_kbps(source_info.bit_rate, 320)
        return ExportPlan(".mp3", ["-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-id3v2_version", "3"], True)
    if suffix in {".wav", ".wave"}:
        return ExportPlan(".wav", ["-c:a", "pcm_s16le"], False)
    if suffix in {".aif", ".aiff"}:
        return ExportPlan(".aiff", ["-c:a", "pcm_s16be"], False)
    if suffix == ".flac" or codec == "flac":
        return ExportPlan(".flac", ["-c:a", "flac"], True)
    if suffix == ".aac":
        kbps = _clamp_kbps(source_info.bit_rate, 256)
        return ExportPlan(".aac", ["-c:a", "aac", "-b:a", f"{kbps}k"], False)
    if suffix == ".ogg":
        kbps = _clamp_kbps(source_info.bit_rate, 192)
        if codec == "opus":
            return ExportPlan(".ogg", ["-c:a", "libopus", "-b:a", f"{kbps}k"], False)
        return ExportPlan(".ogg", ["-c:a", "libvorbis", "-q:a", "6"], False)
    if suffix == ".opus" or codec == "opus":
        kbps = _clamp_kbps(source_info.bit_rate, 192)
        return ExportPlan(".opus", ["-c:a", "libopus", "-b:a", f"{kbps}k"], False)
    if suffix == ".m4a" or codec == "aac":
        kbps = _clamp_kbps(source_info.bit_rate, 256)
        return ExportPlan(".m4a", ["-c:a", "aac", "-b:a", f"{kbps}k"], True)
    if suffix == ".alac" or codec == "alac":
        return ExportPlan(".m4a", ["-c:a", "alac"], True)
    if codec.startswith("pcm_"):
        return ExportPlan(".wav", ["-c:a", "pcm_s16le"], False)
    kbps = _clamp_kbps(source_info.bit_rate, 256)
    return ExportPlan(".m4a", ["-c:a", "aac", "-b:a", f"{kbps}k"], True)


def _strip_title_metadata(path: Path) -> None:
    if MutagenFile is None:
        return
    try:
        audio = MutagenFile(path)
        if audio is None or audio.tags is None:
            return
        tags = audio.tags
        with contextlib.suppress(Exception):
            if hasattr(tags, "delall"):
                for key in ("TIT2", "title", "\xa9nam"):
                    tags.delall(key)
        for key in list(tags.keys()):
            lower = str(key).lower()
            if lower in {"tit2", "title", "\xa9nam"}:
                with contextlib.suppress(Exception):
                    del tags[key]
        audio.save()
    except Exception:
        logger.debug("failed to strip title metadata from %s", path, exc_info=True)


def _export_stem(
    stem_wav: Path,
    source_path: Path,
    dest_path: Path,
    plan: ExportPlan,
    has_cover: bool,
) -> Path:
    ffmpeg_path = _ensure_ffmpeg()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = ensure_unique_path(dest_path)

    def _build_command(include_cover: bool) -> list[str]:
        cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(stem_wav),
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-map_metadata",
            "1",
        ]
        if include_cover:
            cmd.extend(["-map", "1:v?"])
        cmd.extend(plan.audio_args)
        if include_cover:
            cmd.extend(["-c:v", "copy", "-disposition:v", "attached_pic"])
        cmd.extend(["-metadata", "title=", str(candidate)])
        return cmd

    attempts = [False]
    if plan.supports_cover and has_cover:
        attempts = [True, False]

    last_error: Exception | None = None
    for include_cover in attempts:
        try:
            subprocess.run(
                _build_command(include_cover),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            _strip_title_metadata(candidate)
            return candidate
        except Exception as exc:
            last_error = exc
            _cleanup_path(candidate)
            logger.warning(
                "export failed for %s with include_cover=%s: %s",
                candidate,
                include_cover,
                exc,
            )
    raise AppError(ErrorCode.SEPARATION_FAILED, f"Export failed: {last_error}") from last_error


def _resolve_video_output(source_info: SourceInfo) -> tuple[str, list[str]]:
    suffix = source_info.suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        return suffix, ["-c:a", "aac", "-b:a", f"{_clamp_kbps(source_info.bit_rate, 256)}k"]
    if suffix == ".webm":
        return ".webm", ["-c:a", "libopus", "-b:a", f"{_clamp_kbps(source_info.bit_rate, 192)}k"]
    if suffix == ".avi":
        return ".avi", ["-c:a", "libmp3lame", "-b:a", f"{_clamp_kbps(source_info.bit_rate, 192)}k"]
    return ".mkv", ["-c:a", "aac", "-b:a", f"{_clamp_kbps(source_info.bit_rate, 256)}k"]


def _export_video_stem(
    stem_wav: Path,
    source_path: Path,
    dest_path: Path,
    source_info: SourceInfo,
) -> Path:
    ffmpeg_path = _ensure_ffmpeg()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = ensure_unique_path(dest_path)
    suffix, audio_args = _resolve_video_output(source_info)
    if candidate.suffix.lower() != suffix:
        candidate = ensure_unique_path(candidate.with_suffix(suffix))
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-i",
        str(stem_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map_metadata",
        "0",
        "-c:v",
        "copy",
        *audio_args,
        "-shortest",
        "-metadata",
        "title=",
        str(candidate),
    ]
    if suffix in {".mp4", ".m4v", ".mov"}:
        cmd.extend(["-movflags", "+faststart"])
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        _strip_title_metadata(candidate)
        return candidate
    except Exception as exc:
        _cleanup_path(candidate)
        raise AppError(ErrorCode.SEPARATION_FAILED, f"Video export failed: {exc}") from exc


def _decode_audio_to_wav(source_path: Path, work_dir: Path, channels: int) -> Path:
    ffmpeg_path = _ensure_ffmpeg()
    decoded_path = work_dir / "input.wav"
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ar",
        "44100",
        "-ac",
        str(max(1, min(2, channels))),
        "-vn",
        str(decoded_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise AppError(ErrorCode.FFMPEG_MISSING, "ffmpeg not found.") from exc
    except Exception as exc:
        raise AppError(ErrorCode.AUDIO_DECODE_FAILED, f"Could not decode audio: {exc}") from exc
    if not decoded_path.exists():
        raise AppError(ErrorCode.AUDIO_DECODE_FAILED, "Decoded WAV was not created.")
    return decoded_path


def _load_waveform(wav_path: Path) -> torch.Tensor:
    try:
        data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise AppError(ErrorCode.AUDIO_LOAD_FAILED, f"Could not read WAV: {exc}") from exc
    if sample_rate != 44100:
        raise AppError(ErrorCode.AUDIO_LOAD_FAILED, f"Expected 44.1 kHz audio, got {sample_rate}.")
    waveform = torch.from_numpy(data.T.copy())
    if waveform.ndim != 2:
        raise AppError(ErrorCode.AUDIO_LOAD_FAILED, "Decoded audio has an invalid shape.")
    if waveform.shape[0] == 0:
        raise AppError(ErrorCode.AUDIO_LOAD_FAILED, "Decoded audio is empty.")
    return waveform


def _write_temp_wave(out_dir: Path, name: str, tensor: torch.Tensor) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = out_dir / name
    data = tensor.detach().cpu()
    if not torch.isfinite(data).all():
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = torch.clamp(data.float(), -1.0, 1.0)
    sf.write(candidate, data.T.contiguous().numpy(), 44100, subtype="PCM_16")
    return candidate


def _normalize_prediction(pred: torch.Tensor, input_channels: int, chunk_len: int) -> torch.Tensor:
    if pred.dim() == 2:
        if pred.shape[0] in (1, input_channels):
            pred = pred.unsqueeze(0)
        else:
            pred = pred[:, None, :]
    elif pred.dim() == 3 and pred.shape[1] not in (1, input_channels):
        pred = pred.permute(1, 0, 2)
    pred = pred[..., :chunk_len]
    if pred.shape[1] == 1 and input_channels == 2:
        pred = pred.repeat(1, 2, 1)
    return pred


def _prepare_model_input(waveform: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int]:
    original_channels = waveform.shape[0]
    if original_channels == 1:
        waveform = waveform.repeat(2, 1)
    elif original_channels > 2:
        waveform = waveform[:2]
        original_channels = 2
    return waveform.to(device=device, dtype=torch.float32), original_channels


def _restore_output_channels(tensor: torch.Tensor, original_channels: int) -> torch.Tensor:
    if original_channels <= 1 and tensor.shape[0] > 1:
        return tensor.mean(dim=0, keepdim=True)
    return tensor[: max(1, original_channels)]


def _map_fraction(start_pct: int, end_pct: int, fraction: float) -> int:
    fraction = max(0.0, min(1.0, fraction))
    return int(round(start_pct + ((end_pct - start_pct) * fraction)))


def _cleanup_path(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        with contextlib.suppress(Exception):
            path.unlink()


def _safe_mps_empty_cache() -> None:
    if getattr(torch, "mps", None) and hasattr(torch.mps, "empty_cache"):
        with contextlib.suppress(Exception):
            torch.mps.empty_cache()


def _physical_memory_bytes() -> int:
    if sys.platform == "darwin":
        with contextlib.suppress(Exception):
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            return max(0, int((result.stdout or "").strip() or "0"))
    for key in ("SC_PHYS_PAGES", "SC_PAGE_SIZE"):
        if not hasattr(os, "sysconf") or key not in os.sysconf_names:
            break
    else:
        with contextlib.suppress(Exception):
            return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    return 0


def _process_rss_bytes() -> int:
    with contextlib.suppress(Exception):
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True,
            capture_output=True,
            text=True,
        )
        return int((result.stdout or "0").strip() or "0") * 1024
    with contextlib.suppress(Exception):
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if value > 0:
            if sys.platform == "darwin":
                return int(value)
            return int(value) * 1024
    return 0


def _process_rss_limit_bytes() -> int:
    physical = _physical_memory_bytes()
    if physical > 0:
        return max(2 * 1024 * 1024 * 1024, int(physical * PROCESS_RSS_HEADROOM_RATIO))
    return PROCESS_RSS_LIMIT_FALLBACK_BYTES


def _mps_memory_limit_bytes() -> int | None:
    if not getattr(torch, "mps", None):
        return None
    with contextlib.suppress(Exception):
        if hasattr(torch.mps, "recommended_max_memory"):
            recommended = int(torch.mps.recommended_max_memory())
            if recommended > 0:
                return int(recommended * MPS_MEMORY_HEADROOM_RATIO)
    return None


def _mps_allocated_bytes() -> int:
    if not getattr(torch, "mps", None):
        return 0
    with contextlib.suppress(Exception):
        if hasattr(torch.mps, "current_allocated_memory"):
            return int(torch.mps.current_allocated_memory())
    with contextlib.suppress(Exception):
        if hasattr(torch.mps, "driver_allocated_memory"):
            return int(torch.mps.driver_allocated_memory())
    return 0


def _load_roformer_model(model_path: Path, config_path: Path, device: torch.device) -> torch.nn.Module:
    if MelBandRoformer is None:
        raise AppError(ErrorCode.MODEL_IMPORT_FAILED, f"Roformer import failed: {_model_import_error}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.unsafe_load(handle)
    raw_model_cfg = dict(cfg.get("model") or {})
    valid_params = set(inspect.signature(MelBandRoformer.__init__).parameters)
    valid_params.discard("self")
    model_kwargs = {key: value for key, value in raw_model_cfg.items() if key in valid_params}
    ignored = sorted(set(raw_model_cfg) - set(model_kwargs))
    if ignored:
        logger.info("ignoring unsupported model config keys for %s: %s", config_path.name, ", ".join(ignored))
    model = MelBandRoformer(**model_kwargs)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


def select_device() -> torch.device:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning("MPS unavailable; falling back to CPU")
    return torch.device("cpu")


class ModelManager:
    def __init__(self) -> None:
        self.device = select_device()
        self.cache: dict[str, torch.nn.Module] = {}

    def _resolve_model_path(self, filename: str) -> Path:
        search_names = [filename, *MODEL_ALIAS_MAP.get(filename, [])]
        for base_dir in MODEL_SEARCH_DIRS:
            for search_name in search_names:
                match = _locate_case_insensitive(base_dir / search_name)
                if match:
                    return match
        raise AppError(ErrorCode.MODEL_MISSING, f"Missing model file: {filename}")

    def _resolve_config_path(self, filename: str) -> Path:
        found = _locate_case_insensitive(CONFIG_DIR / filename)
        if found:
            return found
        raise AppError(ErrorCode.CONFIG_MISSING, f"Missing config file: {filename}")

    def get(self, name: str) -> torch.nn.Module:
        if name in self.cache:
            return self.cache[name]
        if name not in MODEL_SPECS:
            raise AppError(ErrorCode.INVALID_REQUEST, f"Unknown model: {name}")
        spec = MODEL_SPECS[name]
        model_path = self._resolve_model_path(spec.filename)
        config_path = self._resolve_config_path(spec.config)
        model = _load_roformer_model(model_path, config_path, self.device)
        self.cache[name] = model
        return model


_model_manager: ModelManager | None = None
_model_manager_lock = threading.Lock()


def _get_model_manager() -> ModelManager:
    global _model_manager
    with _model_manager_lock:
        if _model_manager is None:
            _model_manager = ModelManager()
        return _model_manager


def _run_model_chunks(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    segment: int,
    overlap: int,
    progress_cb: Callable[[float], None],
    stop_check: Callable[[], None],
) -> torch.Tensor:
    working, original_channels = _prepare_model_input(waveform, next(model.parameters()).device)
    input_channels = working.shape[0]
    length = working.shape[1]
    overlap = max(1, int(overlap))
    step = max(1, segment // overlap)
    border = max(0, segment - step)
    padded = working

    # Each config carries its own overlap-add expectations instead of using one global ratio.
    if border > 0:
        if padded.shape[-1] > 1:
            remaining = border
            while remaining > 0:
                pad_amount = min(remaining, max(1, padded.shape[-1] - 1))
                padded = F.pad(padded.unsqueeze(0), (pad_amount, pad_amount), mode="reflect").squeeze(0)
                remaining -= pad_amount
        else:
            padded = F.pad(padded, (border, border))
    if padded.shape[-1] < segment:
        padded = F.pad(padded, (0, segment - padded.shape[-1]))

    padded_length = padded.shape[-1]
    starts = list(range(0, max(1, padded_length - segment + 1), step))
    last_start = max(0, padded_length - segment)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    acc: torch.Tensor | None = None
    counts = torch.zeros(padded_length, device=working.device, dtype=working.dtype)
    if len(starts) <= 1 or step >= segment:
        window = torch.ones(segment, device=working.device, dtype=working.dtype)
    else:
        window = torch.hann_window(segment, periodic=False, device=working.device, dtype=working.dtype).clamp_min(1e-6)
    window_view = window.view(1, 1, -1)

    progress_cb(0.0)
    with torch.no_grad():
        for index, start in enumerate(starts, start=1):
            stop_check()
            end = start + segment
            chunk = padded[:, start:end]
            pred = model(chunk.unsqueeze(0))[0]
            pred = _normalize_prediction(pred, input_channels, segment)
            pred_len = pred.shape[-1]
            if acc is None:
                acc = torch.zeros(
                    (pred.shape[0], pred.shape[1], padded_length),
                    device=working.device,
                    dtype=pred.dtype,
                )
            acc[:, :, start : start + pred_len] += pred * window_view[:, :, :pred_len]
            counts[start : start + pred_len] += window[:pred_len]
            progress_cb(index / max(1, len(starts)))

    if acc is None:
        raise AppError(ErrorCode.SEPARATION_FAILED, "Model produced no output.")

    denom = counts.clamp_min(1e-6).view(1, 1, -1)
    restored = acc / denom
    crop_start = border
    crop_end = crop_start + length
    restored = restored[:, :, crop_start:crop_end]
    outputs = []
    for index in range(restored.shape[0]):
        outputs.append(_restore_output_channels(restored[index], original_channels))
    return torch.stack(outputs, dim=0)


def _waveform_like(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    sliced = source[: reference.shape[0], : reference.shape[1]]
    return sliced.to(device=reference.device, dtype=reference.dtype)


def _residual_output(source: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    return _waveform_like(source, predicted) - predicted


def _db_to_gain(db_value: float) -> float:
    return float(10.0 ** (float(db_value) / 20.0))


def _boost_overlay_mix(
    source_waveform: torch.Tensor,
    overlay_tensor: torch.Tensor,
    *,
    base_song_gain_db: float,
    overlay_gain_db: float,
) -> torch.Tensor:
    base_song = _waveform_like(source_waveform, overlay_tensor)
    return (base_song * _db_to_gain(base_song_gain_db)) + (
        overlay_tensor * _db_to_gain(overlay_gain_db)
    )


def _temp_audio_name(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{cleaned or 'stem'}.wav"


def _append_named_output(
    outputs: list[tuple[str, Path]],
    work_dir: Path,
    label: str,
    tensor: torch.Tensor,
) -> None:
    outputs.append((label, _write_temp_wave(work_dir, _temp_audio_name(label), tensor)))


def _cache_intermediate_output(task_id: str, label: str, tensor: torch.Tensor) -> Path | None:
    cache_dir = _ensure_dir(INTERMEDIATE_CACHE_DIR / task_id)
    try:
        return _write_temp_wave(cache_dir, _temp_audio_name(label), tensor)
    except Exception:
        logger.warning("failed to cache intermediate output %s for task %s", label, task_id, exc_info=True)
        return None


def _cached_intermediate_output_path(task_id: str, label: str) -> Path | None:
    candidate = INTERMEDIATE_CACHE_DIR / task_id / _temp_audio_name(label)
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _preset_overlay_cache_label(mode: str) -> str | None:
    if mode == "preset_boost_harmonies":
        return "background_vocals"
    if mode == "preset_boost_guitar":
        return "guitar"
    return None


def _preset_output_label(mode: str) -> str | None:
    labels = MODE_OUTPUT_LABELS.get(mode)
    if labels:
        return str(labels[0])
    return None


def _task_can_adjust_preset(task: dict[str, Any]) -> bool:
    mode = str(task.get("mode") or "")
    cache_label = _preset_overlay_cache_label(mode)
    if cache_label is None:
        return False
    if str(task.get("status") or "").lower() != "done":
        return False
    source_path = Path(str(task.get("source_path") or "")).expanduser()
    if not source_path.exists() or not source_path.is_file():
        return False
    return _cached_intermediate_output_path(str(task.get("id") or ""), cache_label) is not None


def _export_cached_boost_harmonies_mix(task_id: str, preset_settings: dict[str, float]) -> dict[str, Any]:
    task = _require_task(task_id)
    mode = str(task.get("mode") or "")
    cache_label = _preset_overlay_cache_label(mode)
    output_label = _preset_output_label(mode)
    if cache_label is None or output_label is None:
        raise AppError(ErrorCode.INVALID_REQUEST, "This preset export is not available for this task.")
    if not _task_can_adjust_preset(task):
        raise AppError(
            ErrorCode.INVALID_REQUEST,
            "Cached preset files are unavailable. Run the preset again to create a new adjustable export.",
        )

    overlay_cache = _cached_intermediate_output_path(task_id, cache_label)
    if overlay_cache is None:
        raise AppError(
            ErrorCode.INVALID_REQUEST,
            "Cached preset overlay is unavailable. Run the preset again to recreate it.",
        )

    source_path = Path(str(task.get("source_path") or "")).expanduser()
    work_dir = Path(tempfile.mkdtemp(prefix=f"presetmix_{task_id[:8]}_", dir=str(WORK_DIR)))
    try:
        source_info = _probe_source(source_path)
        decoded_path = _decode_audio_to_wav(source_path, work_dir, source_info.channels)
        waveform = _load_waveform(decoded_path)
        overlay_tensor = _load_waveform(overlay_cache)
        boost_mix_tensor = _boost_overlay_mix(
            waveform,
            overlay_tensor,
            base_song_gain_db=preset_settings["base_song_gain_db"],
            overlay_gain_db=preset_settings["overlay_gain_db"],
        )
        temp_mix_path = _write_temp_wave(work_dir, _temp_audio_name(output_label), boost_mix_tensor)

        output_dir = Path(str(task.get("out_dir") or "")).expanduser()
        if not output_dir.exists() or not output_dir.is_dir():
            output_dir = _create_output_dir(task)

        export_plan = _resolve_export_plan(source_info, str(task.get("output_format") or _compat_settings_payload()["output_format"]))
        final_path = output_dir / f"{_safe_stem(str(task['original_name']))} - {output_label}{export_plan.suffix}"
        exported = _export_stem(temp_mix_path, source_path, final_path, export_plan, source_info.has_cover)

        with tasks_lock:
            current = tasks[task_id]
            current["out_dir"] = str(output_dir)
            current["preset_settings_snapshot"] = {
                "overlay_gain_db": preset_settings["overlay_gain_db"],
                "base_song_gain_db": preset_settings["base_song_gain_db"],
            }
            outputs = list(current.get("outputs") or [])
            if exported.name not in outputs:
                outputs.append(exported.name)
            current["outputs"] = outputs
            current["finished_at"] = time.time()
            current["version"] += 1
            snapshot = _public_task(current)
        return snapshot
    finally:
        _cleanup_path(work_dir)


def _expected_output_count(mode: str) -> int:
    return len(MODE_OUTPUT_LABELS.get(mode, ()))


def _stage_model_key(stage_text: str) -> str | None:
    stage = str(stage_text or "").lower()
    mapping = {
        "running vocals model": "vocals",
        "running instrumental model": "instrumental",
        "running deux model": "deux",
        "running guitar model": "guitar",
        "running mel-band karaoke model": "mel_band_karaoke",
        "running harmony background model": "mel_band_karaoke",
        "running denoise model": "denoise",
    }
    for prefix, model_key in mapping.items():
        if prefix in stage:
            return model_key
    return None


app = FastAPI()
app.state.runtime_status_provider = None
tasks_lock = threading.RLock()
tasks: dict[str, dict[str, Any]] = {}
task_queue: queue.Queue[str] = queue.Queue()
model_download_lock = threading.RLock()
model_download_state: dict[str, Any] = {
    "status": "idle",
    "pct": 0,
    "step": "",
    "current_model": "",
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "eta_seconds": None,
    "retry_count": 0,
    "retry_label": "0",
    "started_at": None,
    "error": "",
}
model_download_thread: threading.Thread | None = None
ffmpeg_status_lock = threading.RLock()
ffmpeg_status_cache: dict[str, Any] = {"checked_at": 0.0, "available": True, "message": ""}


def set_runtime_status_provider(provider: Callable[[], dict[str, Any]] | None) -> None:
    app.state.runtime_status_provider = provider


def _runtime_status_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "windowed": False,
        "preferred_port": 8000,
        "current_port": 8000,
        "client_url": "http://127.0.0.1:8000/",
        "lan_url": "",
        "lan_local_url": "",
        "lan_display": "",
        "lan_local_display": "",
        "network_name": "",
        "port_conflict": False,
        "show_port_notice": False,
        "kill_command": "",
    }
    provider = getattr(app.state, "runtime_status_provider", None)
    if callable(provider):
        try:
            raw = provider()
        except Exception:
            logger.debug("runtime status provider failed", exc_info=True)
        else:
            if isinstance(raw, dict):
                payload.update(raw)
    payload.update(_ffmpeg_runtime_payload())
    return payload


def _ffmpeg_runtime_payload(max_age_seconds: float = 10.0) -> dict[str, Any]:
    now = time.time()
    with ffmpeg_status_lock:
        checked_at = float(ffmpeg_status_cache.get("checked_at") or 0.0)
        if now - checked_at <= max_age_seconds:
            return {
                "ffmpeg_available": bool(ffmpeg_status_cache.get("available", True)),
                "ffmpeg_message": str(ffmpeg_status_cache.get("message") or ""),
            }
    available = True
    message = ""
    try:
        _ensure_ffmpeg()
    except AppError as exc:
        available = False
        message = exc.message
    with ffmpeg_status_lock:
        ffmpeg_status_cache.update({"checked_at": now, "available": available, "message": message})
    return {"ffmpeg_available": available, "ffmpeg_message": message}


def _settings_response_payload(request: Request | None = None) -> dict[str, Any]:
    payload = _compat_settings_payload()
    if _is_remote_client(request):
        payload["lan_passcode"] = ""
    payload["runtime"] = _runtime_status_payload()
    return payload


def _task_boost_harmonies_settings(task: dict[str, Any]) -> dict[str, float]:
    snapshot = task.get("preset_settings_snapshot")
    if isinstance(snapshot, dict):
        return {
            "overlay_gain_db": _coerce_gain_db(
                snapshot.get("overlay_gain_db", snapshot.get("background_vocals_gain_db")),
                BOOST_HARMONIES_DEFAULT_BACKGROUND_GAIN_DB,
            ),
            "base_song_gain_db": _coerce_gain_db(
                snapshot.get("base_song_gain_db"),
                BOOST_HARMONIES_DEFAULT_BASE_GAIN_DB,
            ),
        }
    return _boost_harmonies_settings_payload(_compat_settings_payload())


def _task_boost_guitar_settings(task: dict[str, Any]) -> dict[str, float]:
    snapshot = task.get("preset_settings_snapshot")
    if isinstance(snapshot, dict):
        return {
            "overlay_gain_db": _coerce_gain_db(
                snapshot.get("overlay_gain_db", snapshot.get("guitar_gain_db")),
                BOOST_GUITAR_DEFAULT_GUITAR_GAIN_DB,
            ),
            "base_song_gain_db": _coerce_gain_db(
                snapshot.get("base_song_gain_db"),
                BOOST_GUITAR_DEFAULT_BASE_GAIN_DB,
            ),
        }
    return _boost_guitar_settings_payload(_compat_settings_payload())


def _task_preset_settings(task: dict[str, Any]) -> dict[str, float] | None:
    mode = str(task.get("mode") or "")
    if mode == "preset_boost_harmonies":
        return _task_boost_harmonies_settings(task)
    if mode == "preset_boost_guitar":
        return _task_boost_guitar_settings(task)
    return None


def _set_model_download_state(**patch: Any) -> None:
    with model_download_lock:
        model_download_state.update(patch)


def _selected_missing_models(selection: list[str] | None = None) -> list[str]:
    available = {key for key in MODEL_SPECS}
    if selection:
        requested = [item for item in selection if item in available]
    else:
        requested = list(MODEL_SPECS)
    return [item for item in requested if not _model_file_exists(MODEL_SPECS[item].filename)]


def _public_model_download_status() -> dict[str, Any]:
    with model_download_lock:
        payload = dict(model_download_state)
    missing = sorted(_selected_missing_models())
    prompt_state = str(_compat_settings_payload().get("model_prompt_state") or MODEL_PROMPT_PENDING)
    payload.update(
        {
            "missing": missing,
            "models_dir": str(MODEL_DIR),
            "prompt_state": MODEL_PROMPT_COMPLETE if not missing else prompt_state,
        }
    )
    return payload


def _model_retry_label(retry_count: int) -> str:
    return "too many to count" if retry_count > 9999 else str(max(0, retry_count))


def _run_model_download(selection: list[str] | None = None) -> None:
    missing = _selected_missing_models(selection)
    if not missing:
        _set_model_download_state(
            status="done",
            pct=100,
            step="models ready",
            current_model="",
            downloaded_bytes=0,
            total_bytes=0,
            retry_count=0,
            retry_label="0",
            error="",
        )
        _set_compat_settings({"model_prompt_state": MODEL_PROMPT_COMPLETE})
        return

    _set_model_download_state(
        status="downloading",
        pct=1,
        step="preparing downloads",
        current_model="",
        downloaded_bytes=0,
        total_bytes=0,
        eta_seconds=None,
        retry_count=0,
        retry_label="0",
        started_at=time.time(),
        error="",
    )
    _set_compat_settings({"model_prompt_state": MODEL_PROMPT_ACCEPTED})

    def _progress(update: dict[str, Any]) -> None:
        tag = str(update.get("tag") or "")
        filename = str(update.get("filename") or "")
        pretty = MODEL_DISPLAY_NAMES.get(tag, tag.replace("_", " ")) if tag else filename
        downloaded_bytes = int(update.get("downloaded_bytes") or 0)
        total_bytes = int(update.get("total_bytes") or 0)
        with model_download_lock:
            started_at = model_download_state.get("started_at")
            retry_count = int(model_download_state.get("retry_count") or 0)
        eta_seconds: int | None = None
        if isinstance(started_at, (int, float)) and started_at and total_bytes > 0 and downloaded_bytes > 0:
            elapsed = max(1.0, time.time() - float(started_at))
            remaining = max(0, total_bytes - downloaded_bytes)
            eta_seconds = int(round(remaining / max(downloaded_bytes / elapsed, 1)))
        _set_model_download_state(
            status="downloading",
            pct=int(update.get("pct") or 0),
            step=pretty,
            current_model=pretty,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            eta_seconds=eta_seconds,
            retry_label=_model_retry_label(retry_count),
            error="",
        )

    retry_count = 0
    while True:
        missing = _selected_missing_models(selection)
        if not missing:
            break
        try:
            download_to(MODEL_DIR.parent, missing, progress_cb=_progress)
            break
        except Exception as exc:
            retry_count += 1
            retry_label = _model_retry_label(retry_count)
            logger.warning("model download attempt %s failed; retrying", retry_label, exc_info=True)
            _set_model_download_state(
                status="retrying",
                step="waiting to retry",
                current_model="",
                eta_seconds=5,
                retry_count=retry_count,
                retry_label=retry_label,
                error=f"network dropped, retrying in 5s... ({retry_label})",
            )
            time.sleep(5)

    _set_model_download_state(
        status="done",
        pct=100,
        step="models ready",
        current_model="",
        eta_seconds=0,
        retry_count=retry_count,
        retry_label=_model_retry_label(retry_count),
        error="",
    )
    _set_compat_settings({"model_prompt_state": MODEL_PROMPT_COMPLETE})


def _start_model_download(selection: list[str] | None = None) -> dict[str, Any]:
    global model_download_thread
    with model_download_lock:
        if model_download_thread is not None and model_download_thread.is_alive():
            return _public_model_download_status()
        model_download_thread = threading.Thread(target=_run_model_download, args=(selection,), daemon=True)
        model_download_thread.start()
    return _public_model_download_status()


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "name": task["original_name"],
        "mode": task["mode"],
        "output_format": task["output_format"],
        "video_handling": task["video_handling"],
        "delivery": str(task.get("delivery") or "folder"),
        "status": task["status"],
        "stage": task["stage"],
        "pct": task["pct"],
        "eta_seconds": task["eta_seconds"],
        "out_dir": task["out_dir"],
        "outputs": list(task["outputs"]),
        "error": task["error"],
        "version": task["version"],
        "preset_settings": _task_preset_settings(task),
        "can_adjust_preset": _task_can_adjust_preset(task),
    }


def _require_task(task_id: str) -> dict[str, Any]:
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
        return task


def _estimate_eta(task: dict[str, Any], pct: int) -> int | None:
    started_at = task.get("started_at")
    if started_at is None or pct < 1 or pct >= 100:
        return 0 if pct >= 100 else None
    now = time.time()
    elapsed = max(1.0, now - started_at)
    mode = str(task.get("mode") or "")
    stage_text = str(task.get("stage") or "").lower()
    stage_started_at = task.get("stage_started_at")
    stage_elapsed = max(0.0, now - float(stage_started_at or started_at))
    audio_seconds = float(task.get("audio_seconds") or 0.0)
    export_outputs = max(1, _expected_output_count(mode))
    export_budget = ETA_EXPORT_PER_OUTPUT_SECONDS * export_outputs
    raw_estimate: float | None = None

    if "exporting" in stage_text:
        export_span = max(1, EXPORT_PROGRESS_END_PCT - EXPORT_PROGRESS_START_PCT)
        export_progress = max(0.0, min(1.0, (pct - EXPORT_PROGRESS_START_PCT) / export_span))
        raw_estimate = max(1.0, export_budget * (1.0 - export_progress))
        return _stabilize_eta(task, raw_estimate, now=now, stage_text=stage_text)

    if mode == "both_separate" and audio_seconds > 0:
        vocals_total = _predict_model_runtime_seconds("vocals", audio_seconds)
        instrumental_total = _predict_model_runtime_seconds("instrumental", audio_seconds)
        if vocals_total is not None or instrumental_total is not None:
            vocals_total = float(vocals_total or 0.0)
            instrumental_total = float(instrumental_total or 0.0)
            if "instrumental" in stage_text and instrumental_total > 0:
                raw_estimate = max(0.0, instrumental_total - stage_elapsed) + export_budget
            elif "vocals" in stage_text and vocals_total > 0:
                raw_estimate = max(0.0, vocals_total - stage_elapsed) + instrumental_total + export_budget
            else:
                raw_estimate = max(
                    0.0,
                    ETA_PREP_OVERHEAD_SECONDS + vocals_total + instrumental_total + export_budget - elapsed,
                )
    elif mode == "preset_boost_harmonies" and audio_seconds > 0:
        vocals_total = _predict_model_runtime_seconds("vocals", audio_seconds)
        background_total = _predict_model_runtime_seconds("mel_band_karaoke", audio_seconds)
        if vocals_total is not None or background_total is not None:
            vocals_total = float(vocals_total or 0.0)
            background_total = float(background_total or 0.0)
            if "harmony background" in stage_text and background_total > 0:
                raw_estimate = max(0.0, background_total - stage_elapsed) + export_budget
            elif "vocals" in stage_text and vocals_total > 0:
                raw_estimate = max(0.0, vocals_total - stage_elapsed) + background_total + export_budget
            else:
                raw_estimate = max(
                    0.0,
                    ETA_PREP_OVERHEAD_SECONDS + vocals_total + background_total + export_budget - elapsed,
                )
    elif mode == "preset_boost_guitar" and audio_seconds > 0:
        guitar_total = _predict_model_runtime_seconds("guitar", audio_seconds)
        if guitar_total is not None:
            guitar_total = float(guitar_total)
            if "guitar" in stage_text and guitar_total > 0:
                raw_estimate = max(0.0, guitar_total - stage_elapsed) + export_budget
            else:
                raw_estimate = max(
                    0.0,
                    ETA_PREP_OVERHEAD_SECONDS + guitar_total + export_budget - elapsed,
                )

    if audio_seconds > 0:
        stage_model_key = _stage_model_key(stage_text)
        if stage_model_key is not None:
            stage_total = _predict_model_runtime_seconds(stage_model_key, audio_seconds)
            if stage_total is not None:
                stage_estimate = max(0.0, float(stage_total) - stage_elapsed) + export_budget
                raw_estimate = stage_estimate if raw_estimate is None else ((raw_estimate * 0.72) + (stage_estimate * 0.28))

    progress_fraction = _stage_progress_fraction(mode, stage_text, pct)
    if progress_fraction is not None and progress_fraction >= 0.08 and stage_elapsed >= 4.0:
        stage_progress_remaining = max(0.0, (stage_elapsed / progress_fraction) - stage_elapsed)
        if mode == "both_separate" and "running vocals model" in stage_text:
            next_stage_total = _predict_model_runtime_seconds("instrumental", audio_seconds) or stage_progress_remaining
            progress_estimate = stage_progress_remaining + float(next_stage_total) + export_budget
        elif mode == "preset_boost_harmonies" and "running vocals model" in stage_text:
            next_stage_total = _predict_model_runtime_seconds("mel_band_karaoke", audio_seconds) or stage_progress_remaining
            progress_estimate = stage_progress_remaining + float(next_stage_total) + export_budget
        else:
            progress_estimate = stage_progress_remaining + export_budget
        raw_estimate = progress_estimate if raw_estimate is None else ((raw_estimate * 0.7) + (progress_estimate * 0.3))

    predicted_total = task.get("predicted_total_seconds")
    if isinstance(predicted_total, (int, float)) and float(predicted_total) > 0:
        history_estimate = max(0.0, ETA_PREP_OVERHEAD_SECONDS + float(predicted_total) + export_budget - elapsed)
        raw_estimate = history_estimate if raw_estimate is None else ((raw_estimate * 0.78) + (history_estimate * 0.22))

    if raw_estimate is None and pct >= 8 and elapsed >= 6:
        raw_estimate = max(0.0, (elapsed / max(pct / 100.0, 0.01)) - elapsed)

    return _stabilize_eta(task, raw_estimate, now=now, stage_text=stage_text)


def _record_eta_sample(model_key: str, audio_seconds: float, elapsed_seconds: float) -> None:
    if model_key not in ETA_HISTORY_KEYS or audio_seconds <= 0 or elapsed_seconds <= 0:
        return
    with eta_history_lock:
        entries = list(eta_history.get(model_key) or [])
        entries.append(
            {
                "audio_seconds": float(audio_seconds),
                "elapsed_seconds": float(elapsed_seconds),
            }
        )
        eta_history[model_key] = entries[-ETA_HISTORY_LIMIT:]
        _save_eta_history(eta_history)


def _predict_model_runtime_seconds(model_key: str, audio_seconds: float) -> float | None:
    if model_key not in ETA_HISTORY_KEYS or audio_seconds <= 0:
        return None
    with eta_history_lock:
        entries = list(eta_history.get(model_key) or [])
    if not entries:
        return None
    ranked = sorted(
        entries,
        key=lambda entry: abs(float(entry.get("audio_seconds") or 0.0) - audio_seconds),
    )[: min(5, len(entries))]
    if not ranked:
        return None
    weighted_total = 0.0
    weight_sum = 0.0
    scaled_predictions: list[float] = []
    for entry in ranked:
        sample_audio = max(1.0, float(entry.get("audio_seconds") or 0.0))
        sample_elapsed = max(1.0, float(entry.get("elapsed_seconds") or 0.0))
        distance = abs(sample_audio - audio_seconds)
        weight = 1.0 / max(1.0, distance)
        ratio = sample_elapsed / sample_audio
        weighted_total += ratio * weight
        weight_sum += weight
        scaled_predictions.append(max(1.0, sample_elapsed * (audio_seconds / sample_audio)))
    if weight_sum <= 0:
        return None
    weighted_prediction = max(1.0, audio_seconds * (weighted_total / weight_sum))
    scaled_predictions.sort()
    median_prediction = scaled_predictions[len(scaled_predictions) // 2]
    return max(1.0, (median_prediction * 0.7) + (weighted_prediction * 0.3))


def _predict_task_runtime_seconds(mode: str, audio_seconds: float) -> float | None:
    if audio_seconds <= 0:
        return None
    if mode == "both_separate":
        vocals = _predict_model_runtime_seconds("vocals", audio_seconds)
        instrumental = _predict_model_runtime_seconds("instrumental", audio_seconds)
        if vocals is None and instrumental is None:
            return None
        return float(vocals or 0.0) + float(instrumental or 0.0)
    if mode == "preset_boost_harmonies":
        vocals = _predict_model_runtime_seconds("vocals", audio_seconds)
        background = _predict_model_runtime_seconds("mel_band_karaoke", audio_seconds)
        if vocals is None and background is None:
            return None
        return float(vocals or 0.0) + float(background or 0.0)
    if mode == "preset_boost_guitar":
        return _predict_model_runtime_seconds("guitar", audio_seconds)
    required = _required_models_for_mode(mode)
    if len(required) == 1:
        return _predict_model_runtime_seconds(required[0], audio_seconds)
    return None


def _stage_progress_fraction(mode: str, stage_text: str, pct: int) -> float | None:
    stage = str(stage_text or "").lower()
    clamped = max(0, min(100, int(pct)))
    stage_model = _stage_model_key(stage)
    if stage_model == "vocals":
        if mode == "both_separate":
            span = max(1, BOTH_SEPARATE_VOCALS_END_PCT - MODEL_PROGRESS_START_PCT)
            return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
        if mode == "preset_boost_harmonies":
            span = max(1, BOOST_HARMONIES_VOCALS_END_PCT - MODEL_PROGRESS_START_PCT)
            return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
        span = max(1, SINGLE_MODEL_PROGRESS_END_PCT - MODEL_PROGRESS_START_PCT)
        return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
    if stage_model == "instrumental":
        if mode == "both_separate":
            span = max(1, BOTH_SEPARATE_INSTRUMENTAL_END_PCT - BOTH_SEPARATE_INSTRUMENTAL_START_PCT)
            return max(0.0, min(1.0, (clamped - BOTH_SEPARATE_INSTRUMENTAL_START_PCT) / span))
        span = max(1, SINGLE_MODEL_PROGRESS_END_PCT - MODEL_PROGRESS_START_PCT)
        return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
    if stage_model in {"deux", "guitar", "mel_band_karaoke", "denoise"}:
        if mode == "preset_boost_harmonies" and stage_model == "mel_band_karaoke":
            span = max(1, BOOST_HARMONIES_BACKGROUND_END_PCT - BOOST_HARMONIES_BACKGROUND_START_PCT)
            return max(0.0, min(1.0, (clamped - BOOST_HARMONIES_BACKGROUND_START_PCT) / span))
        span = max(1, DEUX_MODEL_PROGRESS_END_PCT - MODEL_PROGRESS_START_PCT)
        return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
    if "exporting" in stage:
        span = max(1, EXPORT_PROGRESS_END_PCT - EXPORT_PROGRESS_START_PCT)
        return max(0.0, min(1.0, (clamped - EXPORT_PROGRESS_START_PCT) / span))
    return None


def _stabilize_eta(task: dict[str, Any], raw_seconds: float | None, *, now: float, stage_text: str) -> int | None:
    if raw_seconds is None:
        task["eta_finish_at"] = None
        task["eta_stage"] = None
        return None
    raw_seconds = max(0.0, float(raw_seconds))
    if raw_seconds <= 0:
        task["eta_finish_at"] = now
        task["eta_stage"] = stage_text
        return 0
    target_finish_at = now + raw_seconds
    previous_finish_at = task.get("eta_finish_at")
    previous_stage = str(task.get("eta_stage") or "")
    if not isinstance(previous_finish_at, (int, float)) or previous_stage != stage_text:
        finish_at = target_finish_at
    else:
        finish_at = float(previous_finish_at)
        drift = target_finish_at - finish_at
        if drift < -1.0:
            finish_at += drift * 0.68
        elif drift > 6.0:
            finish_at += drift * 0.22
        finish_at = max(now + 1.0, finish_at)
    task["eta_finish_at"] = finish_at
    task["eta_stage"] = stage_text
    return max(1, int(round(finish_at - now)))


def _set_task_progress(task_id: str, stage: str, pct: int) -> None:
    with tasks_lock:
        task = tasks[task_id]
        previous_pct = int(task.get("pct") or 0)
        previous_stage = str(task.get("stage") or "")
        now = time.time()
        if task["started_at"] is None:
            task["started_at"] = now
        if task["status"] not in TERMINAL_STATUSES:
            task["status"] = "running"
        if stage != previous_stage or task.get("stage_started_at") is None:
            task["stage_started_at"] = now
            task["eta_finish_at"] = None
            task["eta_stage"] = None
        task["stage"] = stage
        task["pct"] = max(0, min(100, int(pct)))
        task["eta_seconds"] = _estimate_eta(task, task["pct"])
        if task["pct"] != previous_pct or stage != previous_stage:
            task["last_progress_at"] = now
            task["last_progress_pct"] = task["pct"]
            task["last_progress_stage"] = stage
        task["version"] += 1


def _mark_task_done(task_id: str, out_dir: Path, outputs: list[str]) -> None:
    cleanup_snapshot: dict[str, Any] | None = None
    with tasks_lock:
        task = tasks[task_id]
        task["status"] = "done"
        task["stage"] = "Done"
        task["pct"] = 100
        task["eta_seconds"] = 0
        task["eta_finish_at"] = None
        task["eta_stage"] = None
        task["out_dir"] = str(out_dir)
        task["outputs"] = list(outputs)
        task["error"] = None
        task["guard_error"] = None
        task["finished_at"] = time.time()
        task["version"] += 1
        if bool(task.get("cleared")):
            cleanup_snapshot = dict(task)
    if cleanup_snapshot is not None:
        _forget_task(task_id, cleanup_snapshot)
        return
    _prune_terminal_tasks()


def _mark_task_error(task_id: str, message: str) -> None:
    cleanup_snapshot: dict[str, Any] | None = None
    with tasks_lock:
        task = tasks[task_id]
        task["status"] = "error"
        task["stage"] = "Error"
        task["pct"] = max(0, int(task.get("pct", 0)))
        task["eta_seconds"] = None
        task["eta_finish_at"] = None
        task["eta_stage"] = None
        task["error"] = message
        task["guard_error"] = None
        task["finished_at"] = time.time()
        task["version"] += 1
        if bool(task.get("cleared")):
            cleanup_snapshot = dict(task)
    if cleanup_snapshot is not None:
        _forget_task(task_id, cleanup_snapshot)
        return
    _prune_terminal_tasks()


def _task_output_paths(task: dict[str, Any]) -> tuple[Path, list[Path]]:
    out_dir = task.get("out_dir")
    if not out_dir:
        raise AppError(ErrorCode.INVALID_REQUEST, "Output is not ready.")
    out_path = Path(out_dir)
    if not out_path.exists():
        raise AppError(ErrorCode.INVALID_REQUEST, "file doesn't exist")
    outputs = [out_path / str(name) for name in (task.get("outputs") or [])]
    existing_outputs = [path for path in outputs if path.exists()]
    if outputs and not existing_outputs:
        raise AppError(ErrorCode.INVALID_REQUEST, "file doesn't exist")
    return out_path, existing_outputs


def _download_label(task: dict[str, Any]) -> str:
    return _safe_stem(str(task.get("original_name") or "stems"))


def _path_within(path: Path, root: Path) -> bool:
    with contextlib.suppress(Exception):
        return path.expanduser().resolve().is_relative_to(root.expanduser().resolve())
    return False


def _cleanup_task_runtime(task_id: str, task: dict[str, Any]) -> None:
    source_path_text = str(task.get("source_path") or "").strip()
    if source_path_text:
        source_path = Path(source_path_text).expanduser()
        if _path_within(source_path, UPLOAD_DIR):
            _cleanup_path(source_path)

    if ARTWORK_DIR.exists():
        for candidate in ARTWORK_DIR.glob(f"{task_id}.*"):
            _cleanup_path(candidate)

    if WORK_DIR.exists():
        prefix = f"stemsplat_{task_id[:8]}_"
        for candidate in WORK_DIR.iterdir():
            if candidate.name.startswith(prefix):
                _cleanup_path(candidate)

    downloads_dir = RUNTIME_DIR / "downloads"
    if downloads_dir.exists():
        for candidate in downloads_dir.glob(f"{task_id}_*"):
            _cleanup_path(candidate)

    out_dir_text = str(task.get("out_dir") or "").strip()
    if out_dir_text:
        out_path = Path(out_dir_text).expanduser()
        if _path_within(out_path, WORK_DIR) or _path_within(out_path, RUNTIME_DIR):
            _cleanup_path(out_path)


def _forget_task(task_id: str, task_snapshot: dict[str, Any] | None = None) -> None:
    snapshot = task_snapshot
    with tasks_lock:
        current = tasks.pop(task_id, None)
    if snapshot is None:
        snapshot = current
    if snapshot is not None:
        _cleanup_task_runtime(task_id, snapshot)


def _forget_cleared_terminal_tasks(task_ids: list[str]) -> None:
    removable: list[tuple[str, dict[str, Any]]] = []
    with tasks_lock:
        for task_id in task_ids:
            task = tasks.get(task_id)
            if task is None:
                continue
            if not bool(task.get("cleared")):
                continue
            if str(task.get("status") or "") not in TERMINAL_STATUSES:
                continue
            removable.append((task_id, dict(task)))
            tasks.pop(task_id, None)
    for task_id, snapshot in removable:
        _cleanup_task_runtime(task_id, snapshot)


def _prune_terminal_tasks(max_keep: int = TERMINAL_TASK_RETENTION_LIMIT) -> None:
    removable: list[tuple[str, dict[str, Any]]] = []
    with tasks_lock:
        terminal = [
            (task_id, dict(task))
            for task_id, task in tasks.items()
            if str(task.get("status") or "") in TERMINAL_STATUSES and not bool(task.get("cleared"))
        ]
        if len(terminal) <= max_keep:
            return
        terminal.sort(
            key=lambda item: float(item[1].get("finished_at") or item[1].get("created_at") or 0.0),
            reverse=True,
        )
        for task_id, snapshot in terminal[max_keep:]:
            tasks.pop(task_id, None)
            removable.append((task_id, snapshot))
    for task_id, snapshot in removable:
        _cleanup_task_runtime(task_id, snapshot)


def _mark_task_stopped(task_id: str) -> None:
    cleanup_snapshot: dict[str, Any] | None = None
    with tasks_lock:
        task = tasks[task_id]
        guard_error = str(task.get("guard_error") or "").strip()
        if guard_error:
            task["status"] = "error"
            task["stage"] = "Error"
            task["eta_seconds"] = None
            task["eta_finish_at"] = None
            task["eta_stage"] = None
            task["error"] = guard_error
            task["guard_error"] = None
            task["finished_at"] = time.time()
            task["version"] += 1
        else:
            task["status"] = "stopped"
            task["stage"] = "Stopped"
            task["eta_seconds"] = None
            task["eta_finish_at"] = None
            task["eta_stage"] = None
            task["finished_at"] = time.time()
            task["version"] += 1
        if bool(task.get("cleared")):
            cleanup_snapshot = dict(task)
    if cleanup_snapshot is not None:
        _forget_task(task_id, cleanup_snapshot)
        return
    _prune_terminal_tasks()


def _request_task_stop(task_id: str) -> None:
    should_forget = False
    should_prune = False
    with tasks_lock:
        task = tasks[task_id]
        task["stop_event"].set()
        if task["status"] == "queued":
            task["status"] = "stopped"
            task["stage"] = "Stopped"
            task["eta_seconds"] = None
            task["eta_finish_at"] = None
            task["eta_stage"] = None
            task["finished_at"] = time.time()
            should_forget = bool(task.get("cleared"))
            should_prune = not should_forget
        elif task["status"] == "running":
            task["stage"] = "Stopping"
            task["eta_seconds"] = None
            task["eta_finish_at"] = None
            task["eta_stage"] = None
        task["version"] += 1
    if should_forget:
        _forget_task(task_id)
        return
    if should_prune:
        _prune_terminal_tasks()


def _trip_task_guard(task_id: str, message: str) -> None:
    cleanup_snapshot: dict[str, Any] | None = None
    should_prune = False
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None or task["status"] in TERMINAL_STATUSES:
            return
        if task["stop_event"].is_set() and str(task.get("guard_error") or "").strip():
            return
        task["guard_error"] = message
        task["stop_event"].set()
        task["eta_seconds"] = None
        task["eta_finish_at"] = None
        task["eta_stage"] = None
        if task["status"] == "queued":
            task["status"] = "error"
            task["stage"] = "Error"
            task["error"] = message
            task["finished_at"] = time.time()
            task["guard_error"] = None
            if bool(task.get("cleared")):
                cleanup_snapshot = dict(task)
            else:
                should_prune = True
        else:
            task["stage"] = "Stopping"
        task["version"] += 1
    if cleanup_snapshot is not None:
        _forget_task(task_id, cleanup_snapshot)
        return
    if should_prune:
        _prune_terminal_tasks()


def _stop_all_tasks() -> list[str]:
    drained: list[str] = []
    while True:
        try:
            drained.append(task_queue.get_nowait())
        except queue.Empty:
            break
    for _task_id in drained:
        task_queue.task_done()

    task_ids: list[str] = []
    with tasks_lock:
        task_ids = list(tasks.keys())
        for task in tasks.values():
            task["stop_event"].set()
            task["cleared"] = True
            if task["status"] == "queued":
                task["status"] = "stopped"
                task["stage"] = "Stopped"
                task["eta_seconds"] = None
                task["eta_finish_at"] = None
                task["eta_stage"] = None
                task["finished_at"] = time.time()
            elif task["status"] == "running":
                task["stage"] = "Stopping"
                task["eta_seconds"] = None
                task["eta_finish_at"] = None
                task["eta_stage"] = None
            task["version"] += 1
    return task_ids


def _restart_task_payload(
    task_id: str,
    *,
    stems_raw: str | None = None,
    output_format: str | None = None,
    video_handling: str | None = None,
    output_root: str | None = None,
    output_same_as_input: bool | None = None,
    prioritize: bool = False,
) -> dict[str, Any]:
    old_task = _require_task(task_id)
    source_path = Path(old_task["source_path"])
    if not source_path.exists():
        raise AppError(ErrorCode.INVALID_REQUEST, "file doesn't exist")

    mode = old_task["mode"] if stems_raw is None else _stems_to_mode(stems_raw)
    resolved_output_format = output_format or str(old_task["output_format"])
    resolved_video_handling = _validate_video_handling(video_handling or str(old_task["video_handling"]))
    _validate_mode_and_output_format(mode, resolved_output_format)

    payload = _register_task(
        original_name=old_task["original_name"],
        source_path=source_path,
        source_dir=old_task.get("source_dir"),
        mode=mode,
        output_format=resolved_output_format,
        video_handling=resolved_video_handling,
        delivery=str(old_task.get("delivery") or "folder"),
        auto_start=False,
    )
    _apply_task_start_settings(
        payload,
        output_format=resolved_output_format,
        video_handling=resolved_video_handling,
        output_root=output_root,
        output_same_as_input=output_same_as_input,
    )
    _enqueue_task(payload["id"], front=prioritize)
    return payload


def _update_ready_task_selection(task_id: str, stems_raw: str) -> dict[str, Any]:
    mode = _stems_to_mode(stems_raw)
    _validate_models_for_mode(mode)
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
        if str(task.get("status") or "") != "ready":
            raise AppError(ErrorCode.INVALID_REQUEST, "Task has already started.")
        task["mode"] = mode
        task["version"] += 1
        return dict(task)


def _watchdog_cleanup() -> None:
    gc.collect()
    _safe_mps_empty_cache()


def _watchdog_loop() -> None:
    rss_limit = _process_rss_limit_bytes()
    mps_limit = _mps_memory_limit_bytes()
    hard_timeout_hits: dict[str, int] = {}
    stall_hits: dict[str, int] = {}
    rss_breach_hits = 0
    mps_breach_hits = 0
    while True:
        time.sleep(WATCHDOG_INTERVAL_SECONDS)
        now = time.time()
        running_ids: list[str] = []
        with tasks_lock:
            snapshot = [
                (
                    task_id,
                    str(task.get("status") or ""),
                    float(task.get("started_at") or 0.0),
                    float(task.get("last_progress_at") or task.get("started_at") or 0.0),
                    str(task.get("original_name") or "song"),
                )
                for task_id, task in tasks.items()
            ]
        active_set = {task_id for task_id, status, *_rest in snapshot if status == "running"}
        hard_timeout_hits = {task_id: count for task_id, count in hard_timeout_hits.items() if task_id in active_set}
        stall_hits = {task_id: count for task_id, count in stall_hits.items() if task_id in active_set}
        for task_id, status, started_at, last_progress_at, original_name in snapshot:
            if status != "running":
                continue
            running_ids.append(task_id)
            if started_at and (now - started_at) > TASK_HARD_TIMEOUT_SECONDS:
                hard_timeout_hits[task_id] = hard_timeout_hits.get(task_id, 0) + 1
                if hard_timeout_hits[task_id] >= WATCHDOG_CONFIRM_SAMPLES:
                    logger.error("watchdog stopping task %s after hard timeout", task_id)
                    _trip_task_guard(
                        task_id,
                        f"safety stop: {original_name} exceeded the maximum processing time.",
                    )
                continue
            hard_timeout_hits.pop(task_id, None)
            if last_progress_at and (now - last_progress_at) > TASK_STALL_TIMEOUT_SECONDS:
                stall_hits[task_id] = stall_hits.get(task_id, 0) + 1
                if stall_hits[task_id] >= WATCHDOG_CONFIRM_SAMPLES:
                    logger.error("watchdog stopping task %s after progress stall", task_id)
                    _trip_task_guard(
                        task_id,
                        f"safety stop: {original_name} stopped making progress.",
                    )
                continue
            stall_hits.pop(task_id, None)

        if not running_ids:
            rss_breach_hits = 0
            if mps_limit:
                mps_allocated = _mps_allocated_bytes()
                if mps_allocated and mps_allocated > mps_limit:
                    mps_breach_hits += 1
                    if mps_breach_hits >= WATCHDOG_CONFIRM_SAMPLES:
                        logger.warning(
                            "watchdog clearing idle gpu cache due to mps_allocated=%s over limit=%s",
                            mps_allocated,
                            mps_limit,
                        )
                        _watchdog_cleanup()
                        mps_breach_hits = 0
                else:
                    mps_breach_hits = 0
            continue

        rss_bytes = _process_rss_bytes()
        if rss_bytes and rss_bytes > rss_limit:
            rss_breach_hits += 1
            if rss_breach_hits >= WATCHDOG_CONFIRM_SAMPLES:
                logger.error("watchdog stopping active tasks due to rss=%s over limit=%s", rss_bytes, rss_limit)
                for task_id in running_ids:
                    _trip_task_guard(
                        task_id,
                        "safety stop: stemsplat hit its memory protection limit.",
                    )
                _watchdog_cleanup()
                rss_breach_hits = 0
                continue
        else:
            rss_breach_hits = 0

        mps_breach_hits = 0


def _stop_check(task_id: str) -> None:
    with tasks_lock:
        if tasks[task_id]["stop_event"].is_set():
            raise TaskStopped()


def _build_task_payload(
    *,
    task_id: str,
    original_name: str,
    source_path: Path,
    source_dir: str | None,
    mode: str,
    output_format: str,
    video_handling: str,
    delivery: str = "folder",
    auto_start: bool = True,
) -> dict[str, Any]:
    return {
        "id": task_id,
        "original_name": original_name,
        "source_path": str(source_path),
        "source_dir": source_dir,
        "mode": mode,
        "output_format": output_format,
        "video_handling": video_handling,
        "output_root_snapshot": None,
        "output_same_as_input_snapshot": None,
        "preset_settings_snapshot": None,
        "delivery": delivery,
        "status": "queued" if auto_start else "ready",
        "stage": "Waiting in queue" if auto_start else "Ready",
        "pct": 0,
        "eta_seconds": None,
        "out_dir": None,
        "outputs": [],
        "error": None,
        "version": 0,
        "created_at": time.time(),
        "started_at": None,
        "stage_started_at": None,
        "eta_finish_at": None,
        "eta_stage": None,
        "audio_seconds": None,
        "predicted_total_seconds": None,
        "last_progress_at": time.time(),
        "last_progress_pct": 0,
        "last_progress_stage": "Waiting in queue" if auto_start else "Ready",
        "finished_at": None,
        "guard_error": None,
        "cleared": False,
        "stop_event": threading.Event(),
    }


def _create_output_dir(task: dict[str, Any]) -> Path:
    if str(task.get("delivery") or "") == "browser_download":
        return _ensure_dir(WORK_DIR / "exports" / str(task["id"]))
    use_same_as_input = task.get("output_same_as_input_snapshot")
    if use_same_as_input is None:
        settings = _compat_settings_payload()
        use_same_as_input = bool(settings.get("output_same_as_input"))
    if bool(use_same_as_input):
        source_dir = task.get("source_dir")
        if isinstance(source_dir, str) and source_dir:
            candidate = Path(source_dir).expanduser()
            if candidate.exists() and candidate.is_dir():
                return _ensure_dir(candidate)
    output_root_snapshot = str(task.get("output_root_snapshot") or "").strip()
    if output_root_snapshot:
        return _ensure_dir(Path(output_root_snapshot).expanduser())
    return _current_output_root()


def _validate_mode_and_output_format(mode: str, output_format: str) -> None:
    if mode not in MODE_CHOICES:
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid split mode.")
    if output_format not in OUTPUT_FORMAT_CHOICES:
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid output format.")
    _validate_models_for_mode(mode)


def _validate_video_handling(video_handling: str) -> str:
    if video_handling not in VIDEO_HANDLING_CHOICES:
        return "audio_only"
    return "audio_only"


def _validate_media_type(name: str, content_type: str | None = None) -> str:
    suffix = Path(name).suffix.lower()
    kind = (content_type or "").lower()
    if suffix and suffix not in SUPPORTED_MEDIA_SUFFIXES:
        raise AppError(ErrorCode.INVALID_REQUEST, f"Unsupported file type: {suffix}")
    if not suffix and not (kind.startswith("audio/") or kind.startswith("video/")):
        raise AppError(ErrorCode.INVALID_REQUEST, "Unsupported file type. Add a supported audio or video file.")
    return suffix


async def _store_uploaded_file(file: UploadFile) -> tuple[str, Path]:
    original_name = Path(file.filename or "upload").name
    _validate_media_type(original_name, file.content_type or "")

    task_id = str(uuid.uuid4())
    stored_name = f"{task_id}_{original_name}"
    source_path = UPLOAD_DIR / stored_name
    bytes_written = 0
    try:
        with source_path.open("wb") as handle:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    except Exception as exc:
        _cleanup_path(source_path)
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not save upload: {exc}") from exc
    finally:
        with contextlib.suppress(Exception):
            await file.close()

    if bytes_written <= 0:
        _cleanup_path(source_path)
        raise AppError(ErrorCode.INVALID_REQUEST, "Uploaded file is empty.")
    return original_name, source_path


def _store_local_media_file(path: Path) -> tuple[str, Path]:
    source = path.expanduser()
    if not source.exists() or not source.is_file():
        raise AppError(ErrorCode.INVALID_REQUEST, f"file doesn't exist: {source}")
    original_name = source.name
    _validate_media_type(original_name)
    task_id = str(uuid.uuid4())
    stored_name = f"{task_id}_{original_name}"
    stored_path = UPLOAD_DIR / stored_name
    try:
        shutil.copy2(source, stored_path)
    except Exception as exc:
        _cleanup_path(stored_path)
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not import file: {exc}") from exc
    return original_name, stored_path


def _queue_task(task_id: str, *, front: bool = False) -> None:
    if not front:
        task_queue.put(task_id)
        return
    with task_queue.mutex:
        task_queue.queue.appendleft(task_id)
        task_queue.unfinished_tasks += 1
        task_queue.not_empty.notify()


def _apply_task_start_settings(
    task: dict[str, Any],
    *,
    output_format: str | None = None,
    video_handling: str | None = None,
    output_root: str | None = None,
    output_same_as_input: bool | None = None,
) -> None:
    next_output_format = str(output_format or task.get("output_format") or _compat_settings_payload()["output_format"])
    next_video_handling = _validate_video_handling(
        str(video_handling or task.get("video_handling") or _compat_settings_payload()["video_handling"])
    )
    mode = str(task.get("mode") or "vocals")
    _validate_mode_and_output_format(mode, next_output_format)
    task["output_format"] = next_output_format
    task["video_handling"] = next_video_handling
    if output_root is not None:
        task["output_root_snapshot"] = str(_ensure_dir(Path(output_root).expanduser()))
    elif not str(task.get("output_root_snapshot") or "").strip():
        task["output_root_snapshot"] = str(_current_output_root())
    if output_same_as_input is not None:
        task["output_same_as_input_snapshot"] = bool(output_same_as_input)
    elif task.get("output_same_as_input_snapshot") is None:
        task["output_same_as_input_snapshot"] = bool(_compat_settings_payload().get("output_same_as_input"))
    task["preset_settings_snapshot"] = _preset_settings_payload_for_mode(mode, _compat_settings_payload())


def _register_task(
    *,
    original_name: str,
    source_path: Path,
    source_dir: str | None,
    mode: str,
    output_format: str,
    video_handling: str,
    delivery: str = "folder",
    auto_start: bool,
    queue_front: bool = False,
) -> dict[str, Any]:
    task_id = str(uuid.uuid4())
    payload = _build_task_payload(
        task_id=task_id,
        original_name=original_name,
        source_path=source_path,
        source_dir=source_dir,
        mode=mode,
        output_format=output_format,
        video_handling=video_handling,
        delivery=delivery,
        auto_start=auto_start,
    )
    _apply_task_start_settings(payload)
    with tasks_lock:
        tasks[task_id] = payload
    threading.Thread(target=_extract_task_artwork, args=(task_id,), daemon=True).start()
    if auto_start:
        _queue_task(task_id, front=queue_front)
    return payload


def _enqueue_task(task_id: str, *, front: bool = False) -> dict[str, Any]:
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
        if task["status"] in {"queued", "running"}:
            return task
        if task["status"] in TERMINAL_STATUSES:
            return task
        task["status"] = "queued"
        task["stage"] = "Waiting in queue"
        task["eta_seconds"] = None
        task["eta_finish_at"] = None
        task["eta_stage"] = None
        task["started_at"] = None
        task["stage_started_at"] = None
        task["last_progress_at"] = time.time()
        task["last_progress_pct"] = 0
        task["last_progress_stage"] = task["stage"]
        task["guard_error"] = None
        task["version"] += 1
    _queue_task(task_id, front=front)
    return task


def _compat_stage(task: dict[str, Any]) -> str:
    status = str(task.get("status") or "").lower()
    if status == "ready":
        return "ready"
    if status == "queued":
        return "queued"
    if status == "done":
        return "done"
    if status == "stopped":
        return "stopped"
    if status == "error":
        return "error"
    return str(task.get("stage") or "queued")


def _compat_public_task(task: dict[str, Any]) -> dict[str, Any]:
    public = _public_task(task)
    stage = _compat_stage(public)
    pct = -1 if public["status"] == "error" else public["pct"]
    return {
        "task_id": public["id"],
        "id": public["id"],
        "name": public["name"],
        "mode": public["mode"],
        "stage": stage,
        "pct": pct,
        "eta_seconds": public["eta_seconds"],
        "stems": _mode_to_stems(public["mode"]),
        "video_handling": public["video_handling"],
        "out_dir": public["out_dir"],
        "delivery": public["delivery"],
        "error": public["error"],
        "outputs": public["outputs"],
        "preset_settings": public["preset_settings"],
        "can_adjust_preset": public["can_adjust_preset"],
        "artwork_url": f"/api/tasks/{public['id']}/artwork",
    }


def _extract_task_artwork(task_id: str) -> Path | None:
    task = _require_task(task_id)
    source_path = Path(str(task.get("source_path") or "")).expanduser()
    if not source_path.exists():
        return None
    cached_path = ARTWORK_DIR / f"{task_id}.jpg"
    if cached_path.exists() and cached_path.stat().st_size > 0:
        return cached_path

    embedded = _extract_embedded_artwork(source_path)
    if embedded is not None:
        payload, suffix = embedded
        cached_path = ARTWORK_DIR / f"{task_id}{suffix}"
        cached_path.write_bytes(payload)
        return cached_path

    temp_path = ARTWORK_DIR / f"{task_id}.tmp.jpg"
    if temp_path.exists():
        with contextlib.suppress(Exception):
            temp_path.unlink()
    try:
        subprocess.run(
            [
                _ffmpeg_path(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-map",
                "0:v:0",
                "-frames:v",
                "1",
                "-vf",
                "scale=320:-1:force_original_aspect_ratio=decrease",
                str(temp_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        with contextlib.suppress(Exception):
            temp_path.unlink()
        return None

    if not temp_path.exists() or temp_path.stat().st_size <= 0:
        with contextlib.suppress(Exception):
            temp_path.unlink()
        return None

    temp_path.replace(cached_path)
    return cached_path


def _extract_embedded_artwork(source_path: Path) -> tuple[bytes, str] | None:
    if MutagenFile is None:
        return None
    try:
        tagged = MutagenFile(source_path)
    except Exception:
        return None
    if tagged is None:
        return None
    tags = getattr(tagged, "tags", None)
    if not tags:
        return None

    def _resolve_suffix(data: bytes, mime: str | None = None) -> str:
        mime_text = (mime or "").lower()
        if "png" in mime_text:
            return ".png"
        if "jpeg" in mime_text or "jpg" in mime_text:
            return ".jpg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        return ".jpg"

    with contextlib.suppress(Exception):
        covers = tags.get("covr")
        if covers:
            raw = bytes(covers[0])
            if raw:
                return raw, _resolve_suffix(raw)

    values: list[Any]
    if hasattr(tags, "values"):
        values = list(tags.values())
    else:
        values = []
    for value in values:
        frames = value if isinstance(value, list) else [value]
        for frame in frames:
            data = getattr(frame, "data", None)
            if isinstance(data, (bytes, bytearray)) and data:
                return bytes(data), _resolve_suffix(bytes(data), getattr(frame, "mime", None))
    return None


def _process_task(task_id: str) -> None:
    work_dir: Path | None = None
    output_dir: Path | None = None
    manager: ModelManager | None = None
    written_outputs: list[Path] = []
    try:
        task = _require_task(task_id)
        if task["stop_event"].is_set():
            raise TaskStopped()

        source_path = Path(task["source_path"])
        if not source_path.exists():
            raise AppError(ErrorCode.INVALID_REQUEST, "Uploaded source file is missing.")

        _set_task_progress(task_id, "Loading models", 1)
        manager = ModelManager()
        source_info = _probe_source(source_path)
        output_dir = _create_output_dir(task)
        with tasks_lock:
            tasks[task_id]["out_dir"] = str(output_dir)
            tasks[task_id]["version"] += 1

        work_dir = Path(tempfile.mkdtemp(prefix=f"stemsplat_{task_id[:8]}_", dir=str(WORK_DIR)))
        _set_task_progress(task_id, "Preparing audio", 4)
        decoded_path = _decode_audio_to_wav(source_path, work_dir, source_info.channels)
        _stop_check(task_id)
        waveform = _load_waveform(decoded_path)
        mode = task["mode"]
        audio_seconds = waveform.shape[1] / 44100.0 if waveform.shape[1] > 0 else 0.0
        with tasks_lock:
            tasks[task_id]["audio_seconds"] = audio_seconds
            tasks[task_id]["predicted_total_seconds"] = _predict_task_runtime_seconds(mode, audio_seconds)
            tasks[task_id]["version"] += 1

        temp_outputs: list[tuple[str, Path]] = []

        if mode == "vocals":
            vocals_model = manager.get("vocals")
            vocals_started_at = time.time()
            vocals_pred = _run_model_chunks(
                vocals_model,
                waveform,
                MODEL_SPECS["vocals"].segment,
                MODEL_SPECS["vocals"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running vocals model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("vocals", audio_seconds, time.time() - vocals_started_at)
            _append_named_output(temp_outputs, work_dir, "vocals", vocals_pred[0])
        elif mode == "instrumental":
            instrumental_model = manager.get("instrumental")
            instrumental_started_at = time.time()
            instrumental_pred = _run_model_chunks(
                instrumental_model,
                waveform,
                MODEL_SPECS["instrumental"].segment,
                MODEL_SPECS["instrumental"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running instrumental model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("instrumental", audio_seconds, time.time() - instrumental_started_at)
            _append_named_output(temp_outputs, work_dir, "instrumental", instrumental_pred[0])
        elif mode == "both_deux":
            deux_model = manager.get("deux")
            deux_started_at = time.time()
            pair_pred = _run_model_chunks(
                deux_model,
                waveform,
                MODEL_SPECS["deux"].segment,
                MODEL_SPECS["deux"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running deux model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, DEUX_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("deux", audio_seconds, time.time() - deux_started_at)
            vocals_tensor = pair_pred[0]
            instrumental_tensor = (
                pair_pred[1]
                if pair_pred.shape[0] > 1
                else _residual_output(waveform, vocals_tensor)
            )
            _append_named_output(temp_outputs, work_dir, "vocals", vocals_tensor)
            _append_named_output(temp_outputs, work_dir, "instrumental", instrumental_tensor)
        elif mode == "both_separate":
            vocals_model = manager.get("vocals")
            instrumental_model = manager.get("instrumental")
            vocals_started_at = time.time()
            vocals_pred = _run_model_chunks(
                vocals_model,
                waveform,
                MODEL_SPECS["vocals"].segment,
                MODEL_SPECS["vocals"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running vocals model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, BOTH_SEPARATE_VOCALS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("vocals", audio_seconds, time.time() - vocals_started_at)
            instrumental_started_at = time.time()
            instrumental_pred = _run_model_chunks(
                instrumental_model,
                waveform,
                MODEL_SPECS["instrumental"].segment,
                MODEL_SPECS["instrumental"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running instrumental model",
                    _map_fraction(BOTH_SEPARATE_INSTRUMENTAL_START_PCT, BOTH_SEPARATE_INSTRUMENTAL_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("instrumental", audio_seconds, time.time() - instrumental_started_at)
            _append_named_output(temp_outputs, work_dir, "vocals", vocals_pred[0])
            _append_named_output(temp_outputs, work_dir, "instrumental", instrumental_pred[0])
        elif mode == "guitar":
            guitar_model = manager.get("guitar")
            guitar_started_at = time.time()
            guitar_pred = _run_model_chunks(
                guitar_model,
                waveform,
                MODEL_SPECS["guitar"].segment,
                MODEL_SPECS["guitar"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running guitar model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("guitar", audio_seconds, time.time() - guitar_started_at)
            _append_named_output(temp_outputs, work_dir, "guitar", guitar_pred[0])
        elif mode == "mel_band_karaoke":
            karaoke_model = manager.get("mel_band_karaoke")
            karaoke_started_at = time.time()
            karaoke_pred = _run_model_chunks(
                karaoke_model,
                waveform,
                MODEL_SPECS["mel_band_karaoke"].segment,
                MODEL_SPECS["mel_band_karaoke"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running mel-band karaoke model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("mel_band_karaoke", audio_seconds, time.time() - karaoke_started_at)
            vocals_tensor = karaoke_pred[0]
            karaoke_tensor = karaoke_pred[1] if karaoke_pred.shape[0] > 1 else _residual_output(waveform, vocals_tensor)
            _append_named_output(temp_outputs, work_dir, "vocals", vocals_tensor)
            _append_named_output(temp_outputs, work_dir, "karaoke", karaoke_tensor)
        elif mode == "preset_boost_harmonies":
            vocals_model = manager.get("vocals")
            harmony_model = manager.get("mel_band_karaoke")
            preset_settings = _task_boost_harmonies_settings(task)

            vocals_started_at = time.time()
            vocals_pred = _run_model_chunks(
                vocals_model,
                waveform,
                MODEL_SPECS["vocals"].segment,
                MODEL_SPECS["vocals"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running vocals model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, BOOST_HARMONIES_VOCALS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("vocals", audio_seconds, time.time() - vocals_started_at)
            vocals_tensor = vocals_pred[0]

            harmony_started_at = time.time()
            harmony_pred = _run_model_chunks(
                harmony_model,
                vocals_tensor,
                MODEL_SPECS["mel_band_karaoke"].segment,
                MODEL_SPECS["mel_band_karaoke"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running harmony background model",
                    _map_fraction(BOOST_HARMONIES_BACKGROUND_START_PCT, BOOST_HARMONIES_BACKGROUND_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("mel_band_karaoke", audio_seconds, time.time() - harmony_started_at)
            background_vocals_tensor = (
                harmony_pred[1]
                if harmony_pred.shape[0] > 1
                else _residual_output(vocals_tensor, harmony_pred[0])
            )

            _cache_intermediate_output(task_id, "vocals", vocals_tensor)
            _cache_intermediate_output(task_id, "background_vocals", background_vocals_tensor)

            _set_task_progress(task_id, "Mixing boost harmonies", 95)
            boost_mix_tensor = _boost_overlay_mix(
                waveform,
                background_vocals_tensor,
                base_song_gain_db=preset_settings["base_song_gain_db"],
                overlay_gain_db=preset_settings["overlay_gain_db"],
            )
            _append_named_output(temp_outputs, work_dir, "boost harmonies", boost_mix_tensor)
        elif mode == "preset_boost_guitar":
            guitar_model = manager.get("guitar")
            preset_settings = _task_boost_guitar_settings(task)

            guitar_started_at = time.time()
            guitar_pred = _run_model_chunks(
                guitar_model,
                waveform,
                MODEL_SPECS["guitar"].segment,
                MODEL_SPECS["guitar"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running guitar model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, BOOST_GUITAR_MODEL_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("guitar", audio_seconds, time.time() - guitar_started_at)
            guitar_tensor = guitar_pred[0]

            _cache_intermediate_output(task_id, "guitar", guitar_tensor)

            _set_task_progress(task_id, "Mixing boost guitar", 95)
            boost_mix_tensor = _boost_overlay_mix(
                waveform,
                guitar_tensor,
                base_song_gain_db=preset_settings["base_song_gain_db"],
                overlay_gain_db=preset_settings["overlay_gain_db"],
            )
            _append_named_output(temp_outputs, work_dir, "boost guitar", boost_mix_tensor)
        elif mode == "preset_denoise":
            denoise_model = manager.get("denoise")
            denoise_started_at = time.time()
            denoise_pred = _run_model_chunks(
                denoise_model,
                waveform,
                MODEL_SPECS["denoise"].segment,
                MODEL_SPECS["denoise"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running denoise model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("denoise", audio_seconds, time.time() - denoise_started_at)
            _append_named_output(temp_outputs, work_dir, "denoise", denoise_pred[0])
        else:
            raise AppError(ErrorCode.INVALID_REQUEST, "Invalid split mode.")

        _stop_check(task_id)
        export_video = False
        export_plan = None if export_video else _resolve_export_plan(source_info, task["output_format"])
        exported_files: list[str] = []
        total_exports = len(temp_outputs)
        for index, (label, temp_path) in enumerate(temp_outputs, start=1):
            start_pct = _map_fraction(EXPORT_PROGRESS_START_PCT, EXPORT_PROGRESS_END_PCT - 1, (index - 1) / max(1, total_exports))
            _set_task_progress(task_id, f"Exporting {label}", start_pct)
            export_label = f"{label} - deux" if mode == "both_deux" else label
            if export_video:
                video_suffix, _audio_args = _resolve_video_output(source_info)
                final_path = output_dir / f"{_safe_stem(task['original_name'])} - {export_label}{video_suffix}"
                exported = _export_video_stem(temp_path, source_path, final_path, source_info)
            else:
                assert export_plan is not None
                final_path = output_dir / f"{_safe_stem(task['original_name'])} - {export_label}{export_plan.suffix}"
                exported = _export_stem(temp_path, source_path, final_path, export_plan, source_info.has_cover)
            written_outputs.append(exported)
            exported_files.append(exported.name)
            _set_task_progress(
                task_id,
                f"Exporting {label}",
                _map_fraction(EXPORT_PROGRESS_START_PCT, EXPORT_PROGRESS_END_PCT, index / max(1, total_exports)),
            )

        _mark_task_done(task_id, output_dir, exported_files)
    except TaskStopped:
        for output_path in written_outputs:
            _cleanup_path(output_path)
        _mark_task_stopped(task_id)
    except AppError as exc:
        for output_path in written_outputs:
            _cleanup_path(output_path)
        _mark_task_error(task_id, f"{exc.code}: {exc.message}")
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("task %s crashed", task_id)
        for output_path in written_outputs:
            _cleanup_path(output_path)
        _mark_task_error(task_id, f"{ErrorCode.SEPARATION_FAILED}: {exc}")
    finally:
        if manager is not None:
            del manager
        _cleanup_path(work_dir)
        gc.collect()
        _safe_mps_empty_cache()


def _task_worker() -> None:
    while True:
        task_id = task_queue.get()
        try:
            _process_task(task_id)
        finally:
            task_queue.task_done()


threading.Thread(target=_task_worker, daemon=True).start()
threading.Thread(target=_watchdog_loop, daemon=True).start()


@app.on_event("startup")
async def _startup_cleanup() -> None:
    _close_installer_ui()
    _cleanup_old_runtime_entries(WORK_DIR)
    _cleanup_old_runtime_entries(UPLOAD_DIR)
    _cleanup_old_runtime_entries(INTERMEDIATE_CACHE_DIR, INTERMEDIATE_CACHE_RETENTION_SECONDS)
    if _selected_missing_models():
        _set_model_download_state(
            status="idle",
            pct=0,
            step="",
            current_model="",
            downloaded_bytes=0,
            total_bytes=0,
            eta_seconds=None,
            retry_count=0,
            retry_label="0",
            started_at=None,
            error="",
        )
    else:
        _set_model_download_state(
            status="done",
            pct=100,
            step="models ready",
            current_model="",
            downloaded_bytes=0,
            total_bytes=0,
            eta_seconds=0,
            retry_count=0,
            retry_label="0",
            started_at=None,
            error="",
        )
        _set_compat_settings({"model_prompt_state": MODEL_PROMPT_COMPLETE})


@app.exception_handler(AppError)
async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    logger.error("app error for %s %s: %s %s", request.method, request.url.path, exc.code, exc.message)
    return JSONResponse(status_code=400, content={"code": exc.code, "message": exc.message})


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    started = time.time()
    if _lan_passcode_required(request):
        path = request.url.path
        allowed = path in LAN_AUTH_ALLOWED_PATHS
        authorized = _request_has_valid_lan_session(request)
        if not authorized and not allowed:
            response = _lan_unauthorized_response(request)
        elif not authorized and path == "/" and request.method in {"GET", "HEAD"}:
            response = _lan_unauthorized_response(request)
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)
    elapsed_ms = (time.time() - started) * 1000
    logger.info("%s %s -> %s in %.1fms", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def _pick_directory_dialog() -> Path | None:
    if sys.platform != "darwin":
        raise AppError(ErrorCode.INVALID_REQUEST, "Folder picking is only supported on macOS in this build.")
    script = 'POSIX path of (choose folder with prompt "choose output folder")'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").lower()
        if "user canceled" in stderr or exc.returncode == 1:
            return None
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not choose folder: {exc.stderr or exc}") from exc
    chosen = (result.stdout or "").strip()
    return Path(chosen).expanduser() if chosen else None


def _open_path_in_finder(path: Path) -> None:
    target = _ensure_dir(path.expanduser())
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
        return
    if os.name == "nt":
        subprocess.Popen(["explorer", str(target)])
        return
    subprocess.Popen(["xdg-open", str(target)])


@app.get("/api/models_status")
async def models_status() -> dict[str, Any]:
    missing = sorted({item for mode in MODE_CHOICES for item in _find_missing_models_for_mode(mode)})
    return {"missing": missing, "models_dir": str(MODEL_DIR)}


@app.post("/api/open_models_folder")
async def open_models_folder(request: Request) -> dict[str, str]:
    _require_local_request(request, "LAN clients cannot control the host machine.")
    _ensure_dir(MODEL_DIR)
    try:
        _open_path_in_finder(MODEL_DIR)
    except Exception as exc:
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not open models folder: {exc}").to_http(500) from exc
    return {"status": "opened", "path": str(MODEL_DIR)}


@app.get("/api/release_status")
async def release_status() -> dict[str, Any]:
    return _release_status_payload()


@app.get("/api/runtime_status")
async def runtime_status() -> dict[str, Any]:
    return _runtime_status_payload()


@app.post("/api/lan_auth")
async def lan_auth(request: Request) -> JSONResponse:
    if not _lan_passcode_required(request):
        return JSONResponse({"ok": True, "required": False})
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    submitted = str(body.get("passcode") or "")
    settings = _compat_settings_payload()
    expected = str(settings.get("lan_passcode") or "")
    if not expected or not secrets.compare_digest(submitted, expected):
        return JSONResponse(status_code=401, content={"ok": False, "error": "incorrect passcode"})
    ttl_seconds = _lan_auth_ttl_seconds(str(settings.get("lan_passcode_ttl") or "1d"))
    token = secrets.token_urlsafe(32)
    expires_at = None if ttl_seconds is None else time.time() + ttl_seconds
    with lan_auth_lock:
        _prune_lan_auth_sessions_locked()
        lan_auth_sessions[token] = {
            "host": _request_client_host(request),
            "expires_at": expires_at,
            "created_at": time.time(),
        }
    response = JSONResponse(
        {
            "ok": True,
            "required": True,
            "ttl": str(settings.get("lan_passcode_ttl") or "1d"),
        }
    )
    response.set_cookie(
        key=LAN_AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=ttl_seconds,
        path="/",
    )
    return response


@app.post("/api/release_status/ack")
async def ack_release_status(request: Request) -> dict[str, Any]:
    _require_local_request(request, "LAN clients cannot change settings.")
    status = _release_status_payload()
    latest = str(status.get("latest_version") or "")
    if latest:
        _set_compat_settings({"update_last_notified_version": latest})
    return _release_status_payload()


@app.post("/api/release_status/skip")
async def skip_release_status(request: Request) -> dict[str, Any]:
    _require_local_request(request, "LAN clients cannot change settings.")
    status = _release_status_payload()
    latest = str(status.get("latest_version") or "")
    if latest:
        _set_compat_settings(
            {
                "update_last_notified_version": latest,
                "update_skipped_version": latest,
            }
        )
    return _release_status_payload()


@app.get("/api/model_download_status")
async def model_download_status() -> dict[str, Any]:
    return _public_model_download_status()


@app.post("/api/model_downloads/start")
async def start_model_download(request: Request) -> dict[str, Any]:
    _require_local_request(request, "LAN clients cannot change settings.")
    selection: list[str] | None = None
    with contextlib.suppress(Exception):
        body = await request.json()
        if isinstance(body, dict) and isinstance(body.get("models"), list):
            selection = [str(item) for item in body["models"]]
    return _start_model_download(selection)


@app.post("/api/model_download_prompt/dismiss")
async def dismiss_model_download_prompt(request: Request) -> dict[str, Any]:
    _require_local_request(request, "LAN clients cannot change settings.")
    _set_compat_settings({"model_prompt_state": MODEL_PROMPT_DISMISSED})
    return _public_model_download_status()


@app.post("/api/settings/output_root/pick")
async def pick_output_root(request: Request) -> dict[str, Any]:
    _require_local_request(request, "LAN clients cannot change settings.")
    chosen = _pick_directory_dialog()
    if chosen is None:
        return {"cancelled": True, "output_root": str(_current_output_root())}
    chosen = _ensure_dir(chosen)
    _set_compat_settings({"output_root": str(chosen)})
    return {"cancelled": False, "output_root": str(chosen)}


@app.post("/api/settings/output_root/open")
async def open_output_root(request: Request) -> dict[str, str]:
    _require_local_request(request, "LAN clients cannot control the host machine.")
    path = _current_output_root()
    try:
        _open_path_in_finder(path)
    except Exception as exc:
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not open output folder: {exc}").to_http(500) from exc
    return {"status": "opened", "path": str(path)}


@app.post("/api/import_paths")
async def import_paths(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid import payload.").to_http()
    raw_paths = body.get("paths")
    raw_stems = body.get("stems")
    output_format = str(body.get("output_format") or _compat_settings_payload()["output_format"])
    video_handling = _validate_video_handling(str(body.get("video_handling") or _compat_settings_payload()["video_handling"]))
    if not isinstance(raw_paths, list) or not raw_paths:
        raise AppError(ErrorCode.INVALID_REQUEST, "No files selected.").to_http()
    if not isinstance(raw_stems, str):
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid stem selection.").to_http()
    mode = _stems_to_mode(raw_stems)
    _validate_mode_and_output_format(mode, output_format)
    delivery = "browser_download" if _is_remote_client(request) else "folder"
    created: list[dict[str, Any]] = []
    for item in raw_paths:
        path_text = str(item or "").strip()
        if not path_text:
            continue
        source_original = Path(path_text).expanduser()
        original_name, source_path = _store_local_media_file(source_original)
        payload = _register_task(
            original_name=original_name,
            source_path=source_path,
            source_dir=str(source_original.parent),
            mode=mode,
            output_format=output_format,
            video_handling=video_handling,
            delivery=delivery,
            auto_start=False,
        )
        created.append(_compat_public_task(payload))
    return {"tasks": created}


@app.post("/api/tasks")
async def create_task(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("vocals"),
    output_format: str = Form("same_as_input"),
    video_handling: str = Form("audio_only"),
    source_dir: str | None = Form(None),
):
    try:
        _validate_mode_and_output_format(mode, output_format)
        video_handling = _validate_video_handling(video_handling)
        original_name, source_path = await _store_uploaded_file(file)
        payload = _register_task(
            original_name=original_name,
            source_path=source_path,
            source_dir=source_dir,
            mode=mode,
            output_format=output_format,
            video_handling=video_handling,
            delivery="browser_download" if _is_remote_client(request) else "folder",
            auto_start=True,
        )
        return _public_task(payload)
    except AppError as exc:
        raise exc.to_http()


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str):
    _require_task(task_id)

    async def _event_stream():
        last_version = -1
        last_ping_at = time.time()
        while True:
            with tasks_lock:
                task = tasks.get(task_id)
                if task is None:
                    break
                snapshot = _public_task(task)
            if snapshot["version"] != last_version:
                yield f"data: {json.dumps(snapshot)}\n\n"
                last_version = snapshot["version"]
                last_ping_at = time.time()
                if snapshot["status"] in TERMINAL_STATUSES:
                    break
            elif time.time() - last_ping_at >= 10:
                yield ": ping\n\n"
                last_ping_at = time.time()
            await asyncio.sleep(0.35)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    return _public_task(_require_task(task_id))


@app.get("/api/tasks/{task_id}/artwork")
async def task_artwork(task_id: str):
    _require_task(task_id)
    try:
        artwork_path = _extract_task_artwork(task_id)
    except AppError:
        artwork_path = None
    if artwork_path is None:
        raise HTTPException(status_code=404, detail="artwork not found")
    media_type = "image/png" if artwork_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(
        artwork_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    _require_task(task_id)
    _request_task_stop(task_id)
    return _public_task(_require_task(task_id))


@app.post("/api/tasks/{task_id}/retry")
async def retry_task(task_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    stems_raw = body.get("stems")
    output_format = body.get("output_format")
    video_handling = body.get("video_handling")
    output_root = body.get("output_root")
    prioritize = bool(body.get("prioritize"))
    output_same_as_input = body.get("output_same_as_input")
    try:
        payload = _restart_task_payload(
            task_id,
            stems_raw=stems_raw if isinstance(stems_raw, str) and stems_raw.strip() else None,
            output_format=str(output_format) if isinstance(output_format, str) and output_format.strip() else None,
            video_handling=str(video_handling) if isinstance(video_handling, str) and video_handling.strip() else None,
            output_root=str(output_root) if isinstance(output_root, str) and output_root.strip() else None,
            output_same_as_input=bool(output_same_as_input) if "output_same_as_input" in body else None,
            prioritize=prioritize,
        )
    except AppError as exc:
        raise exc.to_http(404 if exc.message == "file doesn't exist" else 400)
    return _public_task(payload)


@app.post("/api/tasks/{task_id}/selection")
async def update_task_selection(task_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    stems_raw = body.get("stems")
    if not isinstance(stems_raw, str) or not stems_raw.strip():
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid stem selection.").to_http()
    try:
        task = _update_ready_task_selection(task_id, stems_raw.strip())
    except AppError as exc:
        status = 404 if exc.code == ErrorCode.TASK_NOT_FOUND else 400
        raise exc.to_http(status)
    return _public_task(task)


@app.post("/api/tasks/{task_id}/preset_mix")
async def export_task_preset_mix(task_id: str, request: Request):
    _require_local_request(request, "LAN clients cannot adjust completed presets.")
    task = _require_task(task_id)
    mode = str(task.get("mode") or "")
    if mode not in {"preset_boost_harmonies", "preset_boost_guitar"}:
        raise AppError(ErrorCode.INVALID_REQUEST, "This task does not support preset gain adjustments.").to_http()
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    current_settings = _task_preset_settings(task) or {
        "overlay_gain_db": PRESET_DEFAULT_OVERLAY_GAIN_DB,
        "base_song_gain_db": PRESET_DEFAULT_BASE_GAIN_DB,
    }
    preset_settings = {
        "overlay_gain_db": _coerce_gain_db(
            body.get(
                "overlay_gain_db",
                body.get("background_vocals_gain_db", body.get("guitar_gain_db")),
            ),
            current_settings["overlay_gain_db"],
        ),
        "base_song_gain_db": _coerce_gain_db(
            body.get("base_song_gain_db"),
            current_settings["base_song_gain_db"],
        ),
    }
    try:
        return _export_cached_boost_harmonies_mix(task_id, preset_settings)
    except AppError as exc:
        status = 404 if exc.code == ErrorCode.TASK_NOT_FOUND else 400
        raise exc.to_http(status)


@app.post("/api/tasks/{task_id}/reveal")
async def reveal_output(task_id: str, request: Request):
    _require_local_request(request, "LAN clients cannot control the host machine.")
    task = _require_task(task_id)
    try:
        out_path, existing_outputs = _task_output_paths(task)
    except AppError as exc:
        status = 404 if exc.message == "file doesn't exist" else 409
        raise exc.to_http(status) from exc
    single_output = existing_outputs[0] if len(existing_outputs) == 1 else None

    try:
        if sys.platform.startswith("darwin"):
            if single_output is not None:
                subprocess.Popen(["open", "-R", str(single_output)])
            else:
                subprocess.Popen(["open", str(out_path)])
        elif sys.platform.startswith("win"):
            if single_output is not None:
                subprocess.Popen(["explorer", "/select,", str(single_output)])
            else:
                os.startfile(str(out_path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(single_output.parent if single_output is not None else out_path)])
    except Exception as exc:
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not reveal output: {exc}").to_http(500) from exc

    return {"status": "opened", "path": str(single_output or out_path)}


@app.get("/api/tasks/{task_id}/download")
async def download_output(task_id: str):
    task = _require_task(task_id)
    try:
        _out_path, existing_outputs = _task_output_paths(task)
    except AppError as exc:
        status = 404 if exc.message == "file doesn't exist" else 409
        raise exc.to_http(status) from exc
    if not existing_outputs:
        raise AppError(ErrorCode.INVALID_REQUEST, "file doesn't exist").to_http(404)
    if len(existing_outputs) == 1:
        output = existing_outputs[0]
        return FileResponse(output, filename=output.name)

    archive_dir = _ensure_dir(RUNTIME_DIR / "downloads")
    archive_name = f"{_download_label(task)}.zip"
    archive_path = archive_dir / f"{task_id}_{archive_name}"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for output in existing_outputs:
            zf.write(output, arcname=output.name)
    return FileResponse(
        archive_path,
        filename=archive_name,
        background=BackgroundTask(lambda: _cleanup_path(archive_path)),
    )


@app.get("/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    return _settings_response_payload(request)


@app.post("/settings")
async def update_settings(request: Request) -> dict[str, Any]:
    _require_local_request(request, "LAN clients cannot change settings.")
    body = await request.json()
    if not isinstance(body, dict):
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid settings payload.").to_http()
    patch: dict[str, Any] = {}
    output_format = body.get("output_format")
    if isinstance(output_format, str):
        if output_format not in OUTPUT_FORMAT_CHOICES:
            raise AppError(ErrorCode.INVALID_REQUEST, "Invalid output format.").to_http()
        patch["output_format"] = output_format
    output_root = body.get("output_root")
    if isinstance(output_root, str):
        try:
            resolved = _ensure_dir(Path(output_root).expanduser())
        except Exception as exc:
            raise AppError(ErrorCode.INVALID_REQUEST, f"Invalid output folder: {exc}").to_http() from exc
        patch["output_root"] = str(resolved)
    if "output_same_as_input" in body:
        patch["output_same_as_input"] = bool(body.get("output_same_as_input"))
    if "video_handling" in body:
        patch["video_handling"] = _validate_video_handling(str(body.get("video_handling") or "audio_only"))
    if "boost_harmonies_background_vocals_gain_db" in body:
        patch["boost_harmonies_background_vocals_gain_db"] = _coerce_gain_db(
            body.get("boost_harmonies_background_vocals_gain_db"),
            BOOST_HARMONIES_DEFAULT_BACKGROUND_GAIN_DB,
        )
    if "boost_harmonies_base_song_gain_db" in body:
        patch["boost_harmonies_base_song_gain_db"] = _coerce_gain_db(
            body.get("boost_harmonies_base_song_gain_db"),
            BOOST_HARMONIES_DEFAULT_BASE_GAIN_DB,
        )
    if "boost_guitar_guitar_gain_db" in body:
        patch["boost_guitar_guitar_gain_db"] = _coerce_gain_db(
            body.get("boost_guitar_guitar_gain_db"),
            BOOST_GUITAR_DEFAULT_GUITAR_GAIN_DB,
        )
    if "boost_guitar_base_song_gain_db" in body:
        patch["boost_guitar_base_song_gain_db"] = _coerce_gain_db(
            body.get("boost_guitar_base_song_gain_db"),
            BOOST_GUITAR_DEFAULT_BASE_GAIN_DB,
        )
    if not _is_remote_client(request):
        if "lan_passcode_enabled" in body:
            patch["lan_passcode_enabled"] = bool(body.get("lan_passcode_enabled"))
        if "lan_passcode" in body:
            patch["lan_passcode"] = str(body.get("lan_passcode") or "")
        if "lan_passcode_ttl" in body:
            ttl_value = str(body.get("lan_passcode_ttl") or "")
            if ttl_value not in LAN_AUTH_TTL_CHOICES:
                raise AppError(ErrorCode.INVALID_REQUEST, "Invalid LAN passcode ttl.").to_http()
            patch["lan_passcode_ttl"] = ttl_value
    _set_compat_settings(patch)
    return _settings_response_payload(request)


@app.post("/upload")
async def compat_upload(
    request: Request,
    file: UploadFile = File(...),
    stems: str = Form("vocals"),
    output_format: str | None = Form(None),
    video_handling: str | None = Form(None),
    source_dir: str | None = Form(None),
):
    try:
        mode = _stems_to_mode(stems)
        resolved_output_format = output_format or _compat_settings_payload()["output_format"]
        resolved_video_handling = _validate_video_handling(video_handling or _compat_settings_payload()["video_handling"])
        _validate_mode_and_output_format(mode, resolved_output_format)
        original_name, source_path = await _store_uploaded_file(file)
        payload = _register_task(
            original_name=original_name,
            source_path=source_path,
            source_dir=source_dir,
            mode=mode,
            output_format=resolved_output_format,
            video_handling=resolved_video_handling,
            delivery="browser_download" if _is_remote_client(request) else "folder",
            auto_start=False,
        )
        public = _compat_public_task(payload)
        return public
    except AppError as exc:
        raise exc.to_http()


@app.post("/start/{task_id}")
async def compat_start(task_id: str, request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    output_format = body.get("output_format")
    video_handling = body.get("video_handling")
    output_root = body.get("output_root")
    output_same_as_input = body.get("output_same_as_input")
    try:
        with tasks_lock:
            task = tasks.get(task_id)
            if task is None:
                raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
            if str(task.get("status") or "") == "ready":
                _apply_task_start_settings(
                    task,
                    output_format=str(output_format) if isinstance(output_format, str) and output_format.strip() else None,
                    video_handling=str(video_handling) if isinstance(video_handling, str) and video_handling.strip() else None,
                    output_root=str(output_root) if isinstance(output_root, str) and output_root.strip() else None,
                    output_same_as_input=bool(output_same_as_input) if "output_same_as_input" in body else None,
                )
        task = _enqueue_task(task_id)
        return _compat_public_task(task)
    except AppError as exc:
        raise exc.to_http(404 if exc.code == ErrorCode.TASK_NOT_FOUND else 400)


@app.get("/progress/{task_id}")
async def compat_progress(task_id: str):
    _require_task(task_id)

    async def _event_stream():
        last_version = -1
        last_ping_at = time.time()
        last_emit_at = 0.0
        while True:
            now = time.time()
            with tasks_lock:
                task = tasks.get(task_id)
                if task is None:
                    break
                snapshot = _compat_public_task(task)
                if task["status"] == "running" and 0 < int(task.get("pct") or 0) < 100:
                    snapshot["eta_seconds"] = _estimate_eta(task, int(task.get("pct") or 0))
            if snapshot["pct"] != -1 and snapshot["stage"] == "error":
                snapshot["pct"] = -1
            should_emit = task["version"] != last_version
            if not should_emit and task["status"] == "running" and 0 < int(snapshot["pct"]) < 100:
                should_emit = (now - last_emit_at) >= 1.0
            if should_emit:
                yield f"data: {json.dumps(snapshot)}\n\n"
                last_version = task["version"]
                last_ping_at = now
                last_emit_at = now
                if task["status"] in TERMINAL_STATUSES:
                    break
            elif now - last_ping_at >= 10:
                yield ": ping\n\n"
                last_ping_at = now
            await asyncio.sleep(0.35)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/stop/{task_id}")
async def compat_stop(task_id: str) -> dict[str, Any]:
    _require_task(task_id)
    _request_task_stop(task_id)
    return _compat_public_task(_require_task(task_id))


@app.post("/rerun/{task_id}")
async def compat_rerun(task_id: str, request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    stems_raw = body.get("stems")
    output_format = body.get("output_format")
    video_handling = body.get("video_handling")
    output_root = body.get("output_root")
    prioritize = bool(body.get("prioritize"))
    output_same_as_input = body.get("output_same_as_input")
    try:
        payload = _restart_task_payload(
            task_id,
            stems_raw=stems_raw if isinstance(stems_raw, str) and stems_raw.strip() else None,
            output_format=str(output_format) if isinstance(output_format, str) and output_format.strip() else None,
            video_handling=str(video_handling) if isinstance(video_handling, str) and video_handling.strip() else None,
            output_root=str(output_root) if isinstance(output_root, str) and output_root.strip() else None,
            output_same_as_input=bool(output_same_as_input) if "output_same_as_input" in body else None,
            prioritize=prioritize,
        )
    except AppError as exc:
        raise exc.to_http(404 if exc.message == "file doesn't exist" else 400)
    return _compat_public_task(payload)


@app.post("/reveal/{task_id}")
async def compat_reveal(task_id: str, request: Request):
    return await reveal_output(task_id, request)


@app.get("/download/{task_id}")
async def compat_download(task_id: str):
    return await download_output(task_id)


@app.post("/clear_all_uploads")
async def compat_clear_all_uploads() -> dict[str, str]:
    cleared_task_ids = _stop_all_tasks()
    _forget_cleared_terminal_tasks(cleared_task_ids)
    return {"status": "cleared"}


@app.post("/rehydrate_tasks")
async def compat_rehydrate_tasks(request: Request) -> dict[str, list[dict[str, Any]]]:
    body = await request.json()
    if not isinstance(body, dict):
        return {"tasks": []}
    requested = body.get("tasks")
    if not isinstance(requested, list):
        return {"tasks": []}
    hydrated: list[dict[str, Any]] = []
    with tasks_lock:
        for item in requested:
            if not isinstance(item, dict):
                continue
            task_id = item.get("id")
            if not isinstance(task_id, str):
                continue
            task = tasks.get(task_id)
            if task is not None:
                hydrated.append(_compat_public_task(task))
    return {"tasks": hydrated}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    candidate = "mobile.html" if _should_use_mobile_ui(request) else "index.html"
    index_path = WEB_DIR / candidate
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    fallback_path = WEB_DIR / "index.html"
    if fallback_path.exists():
        return HTMLResponse(fallback_path.read_text(encoding="utf-8"))
    return HTMLResponse(INDEX_HTML)

@app.get("/favicon.ico")
async def favicon():
    icon_path = BASE_DIR / "web" / "favicon.ico"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/x-icon")
    raise AppError(ErrorCode.INVALID_REQUEST, "favicon missing").to_http(404)

@app.get("/favicon.ico")
async def favicon():
    icon_path = WEB_DIR / "favicon.ico"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/x-icon")
    raise AppError(ErrorCode.INVALID_REQUEST, "favicon missing").to_http(404)


@app.api_route("/shutdown", methods=["POST", "GET"])
async def shutdown(request: Request):
    _require_local_request(request, "LAN clients cannot control the host machine.")
    logger.warning("shutdown requested; exiting process")
    _close_installer_ui()

    def _exit_soon() -> None:
        time.sleep(0.25)
        os._exit(0)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return {"status": "shutting down"}


def cli_main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the stemsplat server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    import uvicorn

    _close_installer_ui()
    if not _port_available(args.port):
        raise SystemExit(f"Port {args.port} is already in use.")
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>stemsplat</title>
  <link rel="icon" type="image/x-icon" href="/favicon.ico">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-1: #0F2027;
      --bg-2: #2C5364;
      --text: #E7ECEF;
      --muted: #B8C4CC;
      --accent: #8ED8FF;
      --accent-strong: #b5ffd8;
      --accent-soft: #96c5d6;
      --icon: #9BB6C2;
      --card: rgba(23, 35, 41, 0.55);
      --card-done: rgba(40, 66, 77, 0.7);
      --border: rgba(255,255,255,0.06);
      --danger: #f57a6d;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      font-family: "Nunito Sans", sans-serif;
      color: var(--text);
      text-transform: lowercase;
      background: linear-gradient(135deg, #0B1A1F 0%, var(--bg-1) 35%, var(--bg-2) 100%);
      padding: 56px 0;
    }

    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 120 120"><filter id="n"><feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="3" stitchTiles="stitch"/></filter><rect width="120" height="120" filter="url(%23n)" opacity="0.035"/></svg>') repeat;
      opacity: .18;
    }

    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background:
        repeating-linear-gradient(0deg, rgba(255,255,255,0.022) 0 1px, transparent 1px 2px),
        repeating-linear-gradient(90deg, rgba(0,0,0,0.024) 0 1px, transparent 1px 2px);
      opacity: .14;
      mix-blend-mode: soft-light;
    }

    @keyframes fadeUpIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
      opacity: 0;
      animation: fadeUpIn 0.55s ease forwards;
    }

    .delay-1 { animation-delay: 0.04s; }
    .delay-2 { animation-delay: 0.12s; }
    .delay-3 { animation-delay: 0.2s; }
    .delay-4 { animation-delay: 0.28s; }

    @keyframes overlayBlurIn {
      from {
        backdrop-filter: blur(0px);
        -webkit-backdrop-filter: blur(0px);
        background-color: rgba(4, 10, 13, 0);
      }
      to {
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        background-color: rgba(4, 10, 13, 0.56);
      }
    }

    @keyframes overlayBlurOut {
      from {
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        background-color: rgba(4, 10, 13, 0.56);
      }
      to {
        backdrop-filter: blur(0px);
        -webkit-backdrop-filter: blur(0px);
        background-color: rgba(4, 10, 13, 0);
      }
    }

    @keyframes settingsCardIn {
      from { opacity: 0; transform: translateY(8px) scale(0.98); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes settingsCardOut {
      from { opacity: 1; transform: translateY(0) scale(1); }
      to { opacity: 0; transform: translateY(6px) scale(0.98); }
    }

    button, input, select {
      font: inherit;
    }

    .shell {
      width: min(980px, calc(100vw - 28px));
      margin: auto;
      padding: 0;
    }

    .shade {
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: rgba(0,0,0,0.24);
      z-index: -1;
    }

    .title {
      margin: 0 0 26px;
      text-align: center;
      font-size: clamp(2.7rem, 5vw, 4.6rem);
      font-weight: 300;
      letter-spacing: 0.5px;
      text-shadow: 0 8px 40px rgba(0,0,0,.45);
    }

    .close-button {
      position: fixed;
      top: 14px;
      left: 14px;
      width: 40px;
      height: 40px;
      display: grid;
      place-items: center;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.14);
      color: var(--text);
      font-size: 18px;
      margin: 0;
      box-shadow: 0 10px 30px rgba(0,0,0,.25);
      backdrop-filter: blur(10px) saturate(120%);
      -webkit-backdrop-filter: blur(10px) saturate(120%);
      z-index: 20;
    }

    .close-button:hover {
      background: rgba(255,255,255,0.22);
    }

    .controls {
      display: flex;
      gap: 18px;
      align-items: stretch;
    }

    .glass {
      background: var(--card);
      backdrop-filter: blur(12px) saturate(110%);
      -webkit-backdrop-filter: blur(12px) saturate(110%);
      border: 1px solid var(--border);
      box-shadow: 0 8px 26px rgba(0,0,0,.28);
    }

    .glass-light {
      background: rgba(255,255,255,0.12);
      backdrop-filter: blur(10px) saturate(110%);
      -webkit-backdrop-filter: blur(10px) saturate(110%);
      border: 1px solid rgba(255,255,255,0.18);
      box-shadow: 0 8px 26px rgba(0,0,0,.28);
    }

    .dropzone {
      flex: 1 1 auto;
      min-height: 292px;
      padding: 26px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      gap: 16px;
      border-radius: 28px;
      text-align: center;
      cursor: pointer;
      transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }

    .dropzone:hover {
      transform: translateY(-2px);
      box-shadow: 0 12px 32px rgba(0,0,0,.32);
    }

    .dropzone.dragging {
      transform: translateY(-2px);
      border-color: rgba(255,255,255,.16);
    }

    .dropzone svg {
      width: 54px;
      height: 54px;
      color: var(--icon);
    }

    .dropzone h3 {
      margin: 0;
      font-size: 1.25rem;
      font-weight: 400;
      letter-spacing: 0;
    }

    .dropzone p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 1rem;
    }

    .dropzone .note {
      font-size: 0.88rem;
    }

    .hidden-input {
      display: none;
    }

    .controls-side {
      width: 270px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .split-card {
      position: relative;
      border-radius: 28px;
      padding: 20px 18px 18px;
      color: var(--text);
    }

    .split-head {
      display: flex;
      justify-content: flex-end;
      align-items: flex-start;
      margin-bottom: 10px;
    }

    .icon-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 34px;
      height: 34px;
      padding: 0;
      border-radius: 12px;
      border: 0;
      background: transparent;
      color: var(--text);
    }

    .icon-button:hover {
      background: rgba(255,255,255,0.08);
    }

    .icon-button svg {
      width: 18px;
      height: 18px;
    }

    .modes {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .mode-card {
      position: relative;
      display: flex;
      gap: 10px;
      align-items: flex-start;
      cursor: pointer;
      padding: 2px 0;
    }

    .mode-card input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }

    .mode-card:hover {
      opacity: 0.92;
    }

    .mode-card .checkbox {
      width: 18px;
      height: 18px;
      border-radius: 6px;
      border: 1.5px solid rgba(255,255,255,.35);
      margin-top: 2px;
      display: grid;
      place-items: center;
      flex-shrink: 0;
    }

    .mode-card.active .checkbox {
      background: linear-gradient(135deg, var(--accent-soft), #5fa3b5);
      border-color: transparent;
    }

    .mode-card.active .checkbox::after {
      content: "";
      width: 6px;
      height: 10px;
      border: 2px solid #0F2027;
      border-top: 0;
      border-left: 0;
      transform: rotate(45deg);
    }

    .mode-card strong {
      display: block;
      font-size: 0.95rem;
      margin-bottom: 0;
    }

    .mode-card span {
      display: block;
      color: var(--muted);
      font-size: 0.84rem;
      line-height: 1.45;
    }

    .small-note {
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.45;
    }

    .small-note strong {
      color: var(--text);
      font-weight: 700;
    }

    .warning {
      display: none;
      margin-top: 12px;
      padding: 10px 12px;
      border-radius: 14px;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      background: rgba(245,122,109,0.12);
      border: 1px solid rgba(245,122,109,0.16);
      color: #ffd7d2;
      font-size: 0.85rem;
      line-height: 1.45;
    }

    .warning.show {
      display: flex;
    }

    .warning-text {
      flex: 1 1 280px;
    }

    .warning-action {
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.08);
      color: inherit;
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      cursor: pointer;
    }

    .warning-action:hover {
      background: rgba(255,255,255,0.14);
    }

    .start-button {
      border-radius: 28px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 6px;
      color: white;
      text-align: center;
      min-height: 88px;
    }

    .start-button .start-icon {
      font-size: 1.15rem;
      line-height: 1;
    }

    .start-button .start-text {
      font-size: 0.92rem;
      font-weight: 700;
    }

    .queue-panel {
      margin-top: 18px;
      padding: 0;
      background: transparent;
      border: 0;
      box-shadow: none;
      backdrop-filter: none;
    }

    .queue-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 12px;
    }

    .queue-head h2 {
      margin: 0;
      font-size: 1rem;
      font-weight: 600;
      letter-spacing: 0;
    }

    .queue-head p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.5;
    }

    .queue-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .queue-item {
      padding: 16px 18px;
      border-radius: 22px;
      background: var(--card);
      border: 1px solid var(--border);
      box-shadow: 0 8px 26px rgba(0,0,0,.28);
      backdrop-filter: blur(12px) saturate(110%);
      -webkit-backdrop-filter: blur(12px) saturate(110%);
    }

    .queue-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
    }

    .queue-main {
      flex: 1 1 auto;
      min-width: 0;
    }

    .queue-name {
      margin: 0;
      font-size: 1rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      word-break: break-word;
    }

    .queue-subline {
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 700;
      background: rgba(255,255,255,0.1);
      color: var(--muted);
      margin-top: 10px;
    }

    .status-badge.status-running,
    .status-badge.status-queued,
    .status-badge.status-uploading {
      color: var(--accent-strong);
      border: 1px solid rgba(181,255,216,0.18);
    }

    .status-badge.status-done {
      color: #dfffe5;
      background: rgba(106, 189, 128, 0.12);
    }

    .status-badge.status-error,
    .status-badge.status-stopped {
      color: #ffd7d2;
      background: rgba(245,122,109,0.12);
    }

    .queue-stage {
      margin-top: 12px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.4;
    }

    .queue-stage.progress-only {
      justify-content: flex-end;
    }

    .queue-stage strong {
      color: var(--text);
      font-weight: 700;
    }

    .progress-shell {
      margin-top: 10px;
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
    }

    .progress-fill {
      height: 100%;
      border-radius: inherit;
      width: 100%;
      transform: scaleX(0);
      transform-origin: left center;
      transition: transform 0.52s cubic-bezier(.22,.61,.36,1);
      will-change: transform;
      background: linear-gradient(90deg, #76cfba 0%, #baf7d8 55%, #e0fff3 100%);
      box-shadow: inset 0 0 16px rgba(255,255,255,0.22);
    }

    .queue-actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      min-width: 160px;
    }

    .button {
      border-radius: 14px;
      padding: 10px 14px;
      font-weight: 700;
    }

    .button:hover { transform: translateY(-1px); }
    .button:disabled { cursor: not-allowed; opacity: 0.5; transform: none; }

    .button.primary {
      background: linear-gradient(135deg, #89dbc2, #baffde);
      color: #081116;
      box-shadow: 0 14px 34px rgba(132, 213, 191, 0.22);
    }

    .button.secondary {
      background: rgba(255,255,255,0.07);
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.08);
    }

    .button.ghost {
      background: transparent;
      color: var(--muted);
      border: 1px solid rgba(255,255,255,0.08);
    }

    .button.danger {
      background: rgba(245,122,109,0.12);
      color: #ffd7d2;
      border: 1px solid rgba(245,122,109,0.16);
    }

    .empty {
      padding: 34px 20px;
      text-align: center;
      color: var(--muted);
      border-radius: 20px;
      border: 1px dashed rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.02);
    }

    .modal-shell {
      position: fixed;
      inset: 0;
      display: none;
      place-items: center;
      background: rgba(4, 10, 13, 0.56);
      backdrop-filter: blur(10px);
      padding: 24px;
    }

    .modal-shell.open {
      display: grid;
      animation: overlayBlurIn .28s ease both;
    }

    .modal-shell.closing {
      display: grid;
      animation: overlayBlurOut .18s ease both;
    }

    .modal {
      width: min(460px, 100%);
      padding: 24px;
    }

    .settings-card-in {
      animation: settingsCardIn .28s cubic-bezier(.2,.7,.2,1) both;
    }

    .settings-card-out {
      animation: settingsCardOut .18s ease both;
    }

    .modal h3 {
      margin: 0;
      font-size: 1.35rem;
      letter-spacing: -0.04em;
    }

    .modal p {
      color: var(--muted);
      line-height: 1.5;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 18px;
    }

    .field label {
      font-size: 0.88rem;
      color: var(--muted);
      text-transform: lowercase;
      letter-spacing: 0.08em;
    }

    .field select {
      width: 100%;
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 16px;
      background: rgba(255,255,255,0.05);
      color: var(--text);
      padding: 14px;
    }

    .modal-actions {
      margin-top: 22px;
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }

    @media (max-width: 900px) {
      .controls {
        flex-direction: column;
      }
      .controls-side {
        width: 100%;
      }
      .queue-row {
        flex-direction: column;
      }
      .queue-actions {
        width: 100%;
        justify-content: flex-start;
      }
    }

    @media (max-width: 620px) {
      .shell {
        width: min(100vw, calc(100vw - 18px));
      }
      body {
        padding: 40px 0;
      }
      .button {
        width: 100%;
        justify-content: center;
      }
    }
  </style>
</head>
<body>
  <div class="shade"></div>
  <button id="close-button" class="close-button" type="button" aria-label="Quit">×</button>

  <main class="shell">
    <h1 class="title fade-in delay-1">stemsplat</h1>

    <section class="controls">
      <label id="dropzone" class="dropzone glass fade-in delay-2" for="file-input" role="button" tabindex="0">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M12 15V9m0 0l3 3m-3-3L9 12m3 9a9 9 0 110-18 9 9 0 010 18z"></path>
        </svg>
        <div>
          <h3>drop songs here</h3>
          <p>or click to choose files</p>
        </div>
        <input id="file-input" class="hidden-input" type="file" accept=".wav,.wave,.mp3,.m4a,.aac,.flac,.ogg,.oga,.aif,.aiff,.alac,.opus,audio/*" multiple>
      </label>

      <div class="controls-side">
        <section class="split-card glass fade-in delay-3">
          <div class="split-head">
            <button id="settings-button" class="icon-button" type="button" aria-label="Open settings">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33A1.65 1.65 0 0 0 9 3.09V3a2 2 0 1 1 4 0v.09c0 .65.38 1.24.97 1.51a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06c-.47.47-.61 1.18-.33 1.82.27.6.86.97 1.51.97H21a2 2 0 1 1 0 4h-.09c-.65 0-1.24.38-1.51.97Z"></path>
              </svg>
            </button>
          </div>

          <div class="modes" id="mode-picker">
            <label class="mode-card active" data-mode="vocals">
              <input type="radio" name="split-mode" value="vocals" checked>
              <div class="checkbox"></div>
              <div>
                <strong>vocals</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="instrumental">
              <input type="radio" name="split-mode" value="instrumental">
              <div class="checkbox"></div>
              <div>
                <strong>instrumental</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="both_deux">
              <input type="radio" name="split-mode" value="both_deux">
              <div class="checkbox"></div>
              <div>
                <strong>both (deux)</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="both_separate">
              <input type="radio" name="split-mode" value="both_separate">
              <div class="checkbox"></div>
              <div>
                <strong>both (separate)</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="guitar">
              <input type="radio" name="split-mode" value="guitar">
              <div class="checkbox"></div>
              <div>
                <strong>mel-band guitar</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="mel_band_karaoke">
              <input type="radio" name="split-mode" value="mel_band_karaoke">
              <div class="checkbox"></div>
              <div>
                <strong>mel-band karaoke</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="preset_denoise">
              <input type="radio" name="split-mode" value="preset_denoise">
              <div class="checkbox"></div>
              <div>
                <strong>denoise</strong>
              </div>
            </label>
          </div>

        </section>

        <button id="start-button" class="start-button glass-light fade-in delay-4" type="button" disabled>start</button>
        <div id="models-warning" class="warning">
          <span id="models-warning-text" class="warning-text"></span>
          <button id="models-folder-button" class="warning-action" type="button" hidden>open models folder</button>
        </div>
      </div>
    </section>

    <section id="queue-panel" class="queue-panel fade-in delay-4">
      <div id="empty-state" class="empty">nothing queued yet.</div>
      <div id="queue" class="queue-list"></div>
    </section>
  </main>

  <div id="settings-modal" class="modal-shell" aria-hidden="true">
    <div class="modal glass" role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <h3 id="settings-title">settings</h3>
      <p>Choose the format for exported stems.</p>
      <div class="field">
        <label for="output-format">output format</label>
        <select id="output-format">
          <option value="same_as_input">same as input</option>
          <option value="mp3_320">320kb mp3</option>
          <option value="mp3_128">128kb mp3</option>
          <option value="wav">wav</option>
          <option value="m4a">m4a</option>
          <option value="flac">flac</option>
        </select>
      </div>
      <div class="modal-actions">
        <button id="settings-cancel" class="button ghost" type="button">cancel</button>
        <button id="settings-save" class="button primary" type="button">save</button>
      </div>
    </div>
  </div>

  <script>
    const MODE_LABELS = {
      vocals: 'vocals',
      instrumental: 'instrumental',
      both_deux: 'both (deux)',
      both_separate: 'both (separate)',
      guitar: 'mel-band guitar',
      mel_band_karaoke: 'mel-band karaoke',
      preset_denoise: 'denoise',
    };

    const OUTPUT_LABELS = {
      same_as_input: 'same as input',
      mp3_320: '320kb mp3',
      mp3_128: '128kb mp3',
      wav: 'wav',
      m4a: 'm4a',
      flac: 'flac',
    };

    const queueEl = document.getElementById('queue');
    const queuePanelEl = document.getElementById('queue-panel');
    const emptyStateEl = document.getElementById('empty-state');
    const startButton = document.getElementById('start-button');
    const fileInput = document.getElementById('file-input');
    const dropzone = document.getElementById('dropzone');
    const settingsButton = document.getElementById('settings-button');
    const settingsModal = document.getElementById('settings-modal');
    const settingsCancel = document.getElementById('settings-cancel');
    const settingsSave = document.getElementById('settings-save');
    const outputFormatSelect = document.getElementById('output-format');
    const closeButton = document.getElementById('close-button');
    const modelsWarning = document.getElementById('models-warning');
    const modelsWarningText = document.getElementById('models-warning-text');
    const modelsFolderButton = document.getElementById('models-folder-button');
    const settingsCard = settingsModal.querySelector('.modal');

    const settings = {
      output_format: localStorage.getItem('stemsplat.output_format') || 'same_as_input',
    };

    let selectedMode = 'vocals';
    let tasks = [];
    let startBusy = false;
    let settingsCloseTimer = null;
    let missingModels = [];
    let modelsDir = '';

    function missingForMode(mode) {
      if (mode === 'vocals') return missingModels.includes('vocals') ? ['vocals'] : [];
      if (mode === 'instrumental') return missingModels.includes('instrumental') ? ['instrumental'] : [];
      if (mode === 'both_deux') return missingModels.includes('deux') ? ['deux'] : [];
      if (mode === 'both_separate') {
        return ['vocals', 'instrumental'].filter((name) => missingModels.includes(name));
      }
      if (mode === 'guitar') return missingModels.includes('guitar') ? ['guitar'] : [];
      if (mode === 'mel_band_karaoke') return missingModels.includes('mel_band_karaoke') ? ['mel_band_karaoke'] : [];
      if (mode === 'preset_denoise') return missingModels.includes('denoise') ? ['denoise'] : [];
      return [];
    }

    function pendingMissingModels() {
      const missing = new Set();
      tasks
        .filter((task) => task.status === 'pending')
        .forEach((task) => missingForMode(task.mode).forEach((name) => missing.add(name)));
      return Array.from(missing);
    }

    function updateOutputSummary() {
      outputFormatSelect.value = settings.output_format;
    }

    function setMode(mode) {
      selectedMode = mode;
      document.querySelectorAll('.mode-card').forEach((card) => {
        const active = card.dataset.mode === mode;
        card.classList.toggle('active', active);
        const input = card.querySelector('input');
        if (input) input.checked = active;
      });
      showModelsWarning(missingModels);
    }

    function humanEta(seconds) {
      if (!Number.isFinite(seconds) || seconds <= 0) return '';
      const mins = Math.floor(seconds / 60);
      const secs = seconds % 60;
      if (mins >= 60) {
        const hours = Math.floor(mins / 60);
        const remMins = mins % 60;
        return `${hours}h ${remMins}m remaining`;
      }
      if (mins > 0) return `${mins}m ${secs}s remaining`;
      return `${secs}s remaining`;
    }

    function statusLabel(status) {
      if (status === 'queued') return 'queued';
      if (status === 'running') return 'running';
      if (status === 'done') return 'done';
      if (status === 'error') return 'error';
      if (status === 'stopped') return 'stopped';
      if (status === 'uploading') return 'uploading';
      return 'ready';
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    function isTerminal(task) {
      return ['done', 'error', 'stopped'].includes(task.status);
    }

    function makeLocalTask(file) {
      return {
        localId: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        id: null,
        file,
        name: file.name,
        mode: selectedMode,
        output_format: settings.output_format,
        status: 'pending',
        stage: 'ready',
        pct: 0,
        eta_seconds: null,
        out_dir: null,
        outputs: [],
        error: null,
        eventSource: null,
        abortController: null,
        removed: false,
      };
    }

    function updateStartButton() {
      const pendingCount = tasks.filter((task) => task.status === 'pending').length;
      startButton.disabled = startBusy || pendingCount === 0 || pendingMissingModels().length > 0;
      startButton.textContent = startBusy ? 'starting...' : 'start';
    }

    function updateQueueSummary() {
      const queueSummaryEl = document.getElementById('queue-summary');
      if (!queueSummaryEl) {
        return;
      }
      if (tasks.length === 0) {
        queueSummaryEl.textContent = 'add songs, then press start.';
        return;
      }
      const pending = tasks.filter((task) => task.status === 'pending').length;
      const active = tasks.filter((task) => ['queued', 'running', 'uploading'].includes(task.status)).length;
      const done = tasks.filter((task) => task.status === 'done').length;
      queueSummaryEl.textContent = `${tasks.length} song${tasks.length === 1 ? '' : 's'} • ${pending} waiting • ${active} working • ${done} done`;
    }

    function renderQueue() {
      queueEl.innerHTML = '';
      if (queuePanelEl) {
        queuePanelEl.style.display = tasks.length === 0 ? 'none' : 'block';
      }
      emptyStateEl.style.display = 'none';

      tasks.forEach((task) => {
        const item = document.createElement('article');
        item.className = 'queue-item';

        const pct = Math.max(0, Math.min(100, task.pct || 0));
        const eta = humanEta(task.eta_seconds);
        const stageText = task.error ? task.error : '';
        const modeLabel = MODE_LABELS[task.mode] || task.mode;
        const outputLabel = task.output_format === 'same_as_input'
          ? ''
          : (OUTPUT_LABELS[task.output_format] || task.output_format);
        const queueMeta = outputLabel ? `${modeLabel} • ${outputLabel}` : modeLabel;
        const progressSide = `${pct}%${eta ? ` • ${eta}` : ''}`;
        const badgeText = statusLabel(task.status);
        const showStatusBadge = ['error', 'stopped'].includes(task.status)
          || (task.status === 'done' && stageText.trim().toLowerCase() !== badgeText);
        const statusBadgeHtml = showStatusBadge
          ? `<div class="status-badge status-${task.status}">${badgeText}</div>`
          : '';
        const stageHtml = stageText
          ? `
              <div class="queue-stage">
                <strong>${escapeHtml(stageText)}</strong>
                <span>${escapeHtml(progressSide)}</span>
              </div>
            `
          : `
              <div class="queue-stage progress-only">
                <span>${escapeHtml(progressSide)}</span>
              </div>
            `;

        item.innerHTML = `
          <div class="queue-row">
            <div class="queue-main">
              <p class="queue-name">${escapeHtml(task.name)}</p>
              <div class="queue-subline">${escapeHtml(queueMeta)}</div>
              ${statusBadgeHtml}
              ${stageHtml}
              <div class="progress-shell">
                <div class="progress-fill" style="transform:scaleX(${pct / 100})"></div>
              </div>
            </div>
            <div class="queue-actions" data-actions="${task.localId}">
            </div>
          </div>
        `;

        const actions = item.querySelector('.queue-actions');
        if (task.status === 'pending') {
          const remove = document.createElement('button');
          remove.className = 'button ghost';
          remove.textContent = 'remove';
          remove.addEventListener('click', () => {
            task.removed = true;
            tasks = tasks.filter((entry) => entry.localId !== task.localId);
            renderQueue();
          });
          actions.appendChild(remove);
        } else {
          if (!task.id) {
            const remove = document.createElement('button');
            remove.className = 'button ghost';
            remove.textContent = 'remove';
            remove.addEventListener('click', () => {
              task.removed = true;
              if (task.abortController) {
                task.abortController.abort();
                task.abortController = null;
              }
              tasks = tasks.filter((entry) => entry.localId !== task.localId);
              renderQueue();
            });
            actions.appendChild(remove);
          }

          if (['queued', 'running', 'uploading'].includes(task.status) && task.id) {
            const stop = document.createElement('button');
            stop.className = 'button danger';
            stop.textContent = 'stop';
            stop.disabled = task.status === 'uploading';
            stop.addEventListener('click', async () => {
              try {
                await fetch(`/api/tasks/${task.id}/stop`, { method: 'POST' });
              } catch (error) {
                console.error(error);
              }
            });
            actions.appendChild(stop);
          }

          if (task.status === 'done' && task.id) {
            const reveal = document.createElement('button');
            reveal.className = 'button secondary';
            reveal.textContent = (task.outputs || []).length === 1 ? 'show song' : 'show files';
            reveal.addEventListener('click', async () => {
              try {
                await fetch(`/api/tasks/${task.id}/reveal`, { method: 'POST' });
              } catch (error) {
                console.error(error);
              }
            });
            actions.appendChild(reveal);
          }

          if (task.id && isTerminal(task)) {
            const retry = document.createElement('button');
            retry.className = 'button ghost';
            retry.textContent = 'retry';
            retry.addEventListener('click', async () => {
              try {
                const res = await fetch(`/api/tasks/${task.id}/retry`, { method: 'POST' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.message || data.detail?.message || 'Retry failed');
                if (task.eventSource) task.eventSource.close();
                Object.assign(task, {
                  id: data.id,
                  status: data.status,
                  stage: data.stage,
                  pct: data.pct,
                  eta_seconds: data.eta_seconds,
                  out_dir: data.out_dir,
                  outputs: data.outputs,
                  error: data.error,
                });
                subscribeToTask(task);
                renderQueue();
              } catch (error) {
                console.error(error);
              }
            });
            actions.appendChild(retry);
          }
        }

        queueEl.appendChild(item);
      });

      showModelsWarning(missingModels);
      updateQueueSummary();
      updateStartButton();
    }

    function subscribeToTask(task) {
      if (!task.id) return;
      if (task.eventSource) task.eventSource.close();
      const source = new EventSource(`/api/tasks/${task.id}/events`);
      task.eventSource = source;
      source.onmessage = (event) => {
        const data = JSON.parse(event.data);
        Object.assign(task, {
          id: data.id,
          status: data.status,
          stage: data.stage,
          pct: data.pct,
          eta_seconds: data.eta_seconds,
          out_dir: data.out_dir,
          outputs: data.outputs || [],
          error: data.error,
        });
        renderQueue();
        if (isTerminal(task)) {
          source.close();
          task.eventSource = null;
        }
      };
      source.onerror = () => {
        if (isTerminal(task)) {
          source.close();
          task.eventSource = null;
        }
      };
    }

    async function addFiles(fileList) {
      const incoming = Array.from(fileList || []).filter(Boolean);
      fileInput.value = '';
      if (incoming.length === 0) return;
      incoming.forEach((file) => tasks.push(makeLocalTask(file)));
      renderQueue();
    }

    async function startPending() {
      const pending = tasks.filter((task) => task.status === 'pending' && task.file);
      if (pending.length === 0) return;
      if (pendingMissingModels().length > 0) {
        renderQueue();
        return;
      }
      startBusy = true;
      updateStartButton();
      try {
        for (const task of pending) {
          if (task.removed || !tasks.includes(task)) {
            continue;
          }
          try {
            task.status = 'uploading';
            task.stage = 'uploading';
            task.pct = 0;
            task.error = null;
            renderQueue();

            const body = new FormData();
            body.append('file', task.file);
            body.append('mode', task.mode);
            body.append('output_format', task.output_format);
            task.abortController = new AbortController();

            const res = await fetch('/api/tasks', { method: 'POST', body, signal: task.abortController.signal });
            const data = await res.json();
            task.abortController = null;
            if (!res.ok) {
              throw new Error(data.message || data.detail?.message || 'Upload failed');
            }
            if (task.removed || !tasks.includes(task)) {
              if (data.id) {
                fetch(`/api/tasks/${data.id}/stop`, { method: 'POST' }).catch(() => {});
              }
              continue;
            }

            task.file = null;
            task.id = data.id;
            task.status = data.status;
            task.stage = data.stage;
            task.pct = data.pct;
            task.eta_seconds = data.eta_seconds;
            task.out_dir = data.out_dir;
            task.outputs = data.outputs || [];
            task.error = data.error;
            subscribeToTask(task);
            renderQueue();
          } catch (error) {
            task.abortController = null;
            if (error?.name === 'AbortError' || task.removed) {
              continue;
            }
            task.status = 'error';
            task.stage = 'error';
            task.error = error?.message || 'Upload failed';
            renderQueue();
          }
        }
      } finally {
        startBusy = false;
        updateStartButton();
      }
    }

    function openSettings() {
      if (settingsCloseTimer) {
        clearTimeout(settingsCloseTimer);
        settingsCloseTimer = null;
      }
      settingsModal.classList.add('open');
      settingsModal.classList.remove('closing');
      settingsModal.setAttribute('aria-hidden', 'false');
      if (settingsCard) {
        settingsCard.classList.remove('settings-card-out');
        void settingsCard.offsetWidth;
        settingsCard.classList.add('settings-card-in');
      }
      outputFormatSelect.value = settings.output_format;
    }

    function closeSettings() {
      if (!settingsModal.classList.contains('open')) return;
      settingsModal.classList.add('closing');
      settingsModal.setAttribute('aria-hidden', 'true');
      if (settingsCard) {
        settingsCard.classList.remove('settings-card-in');
        settingsCard.classList.add('settings-card-out');
      }
      settingsCloseTimer = window.setTimeout(() => {
        settingsModal.classList.remove('open', 'closing');
        if (settingsCard) {
          settingsCard.classList.remove('settings-card-out');
        }
        settingsCloseTimer = null;
      }, 180);
    }

    function showModelsWarning(missing, nextModelsDir = modelsDir) {
      missingModels = Array.isArray(missing) ? [...missing] : [];
      modelsDir = nextModelsDir || modelsDir || '';
      const relevantMissing = pendingMissingModels().length > 0 ? pendingMissingModels() : missingForMode(selectedMode);
      if (relevantMissing.length === 0) {
        modelsWarning.classList.remove('show');
        modelsWarningText.textContent = '';
        modelsFolderButton.hidden = true;
        updateStartButton();
        return;
      }
      modelsWarning.classList.add('show');
      modelsWarningText.textContent = modelsDir
        ? `missing models: ${relevantMissing.join(', ')}. add them to ${modelsDir} before starting.`
        : `missing models: ${relevantMissing.join(', ')}. add them to the models folder before starting.`;
      modelsFolderButton.hidden = !modelsDir;
      updateStartButton();
    }

    async function loadModelsWarning() {
      try {
        const res = await fetch('/api/models_status');
        if (!res.ok) return;
        const data = await res.json();
        showModelsWarning(data.missing || [], data.models_dir || '');
      } catch (error) {
        console.error(error);
      }
    }

    fileInput.addEventListener('change', (event) => addFiles(event.target.files));
    dropzone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        fileInput.click();
      }
    });
    dropzone.addEventListener('dragover', (event) => {
      event.preventDefault();
      dropzone.classList.add('dragging');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging'));
    dropzone.addEventListener('drop', (event) => {
      event.preventDefault();
      dropzone.classList.remove('dragging');
      addFiles(event.dataTransfer.files);
    });

    document.querySelectorAll('.mode-card').forEach((card) => {
      card.addEventListener('click', () => setMode(card.dataset.mode));
    });

    modelsFolderButton.addEventListener('click', async () => {
      try {
        await fetch('/api/open_models_folder', { method: 'POST' });
      } catch (error) {
        console.error(error);
      }
    });
    document.querySelectorAll('.mode-card input').forEach((input) => {
      input.addEventListener('change', (event) => setMode(event.target.value));
    });

    startButton.addEventListener('click', startPending);

    settingsButton.addEventListener('click', openSettings);
    settingsCancel.addEventListener('click', closeSettings);
    settingsSave.addEventListener('click', () => {
      settings.output_format = outputFormatSelect.value;
      localStorage.setItem('stemsplat.output_format', settings.output_format);
      updateOutputSummary();
      closeSettings();
    });

    settingsModal.addEventListener('click', (event) => {
      if (event.target === settingsModal) closeSettings();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && settingsModal.classList.contains('open')) {
        closeSettings();
      }
    });

    closeButton.addEventListener('click', async () => {
      try {
        await fetch('/shutdown', { method: 'POST', keepalive: true });
      } catch (error) {
        console.error(error);
      }
      setTimeout(() => {
        try { window.close(); } catch (_) {}
        window.location.replace('about:blank');
      }, 250);
    });

    updateOutputSummary();
    setMode(selectedMode);
    renderQueue();
    loadModelsWarning();
  </script>
</body>
</html>
"""


if __name__ == "__main__":  # pragma: no cover
    cli_main()
