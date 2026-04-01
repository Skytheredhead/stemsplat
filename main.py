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
import signal
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
from downloader import ModelDownloadError, SSL_CONTEXT, download_to
from einops import pack, rearrange, reduce, repeat, unpack
from app_paths import (
    ARTWORK_DIR,
    CONFIG_DIR,
    INTERMEDIATE_CACHE_DIR,
    LOG_DIR,
    MODEL_DIR,
    OUTPUT_ROOT,
    PREVIOUS_FILES_DIR,
    PREVIOUS_FILES_INDEX_PATH,
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

try:  # pragma: no cover - optional runtime dependency
    from demucs.apply import TensorChunk, apply_model as demucs_apply_model
    from demucs.states import set_state as demucs_set_state
except Exception as exc:  # pragma: no cover - surfaced when fast models are used
    TensorChunk = None  # type: ignore[assignment]
    demucs_apply_model = None  # type: ignore[assignment]
    demucs_set_state = None  # type: ignore[assignment]
    _demucs_import_error = exc
else:
    _demucs_import_error = None

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

try:  # pragma: no cover - import failure is surfaced at runtime
    from split.bs_roformer import BSRoformer
except Exception as exc:  # pragma: no cover - keep app importable for syntax checks
    BSRoformer = None  # type: ignore[assignment]
    _bs_model_import_error = exc
else:
    _bs_model_import_error = None

try:  # pragma: no cover - import failure is surfaced at runtime
    from split.drumsep_mdx23c import (
        TFC_TDF_net,
        config_namespace,
        demix_mdx23c,
        load_not_compatible_weights,
    )
except Exception as exc:  # pragma: no cover - keep app importable for syntax checks
    TFC_TDF_net = None  # type: ignore[assignment]
    config_namespace = None  # type: ignore[assignment]
    demix_mdx23c = None  # type: ignore[assignment]
    load_not_compatible_weights = None  # type: ignore[assignment]
    _drumsep_import_error = exc
else:
    _drumsep_import_error = None


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
    kind: str = "roformer"
    target: str | None = None


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
APP_VERSION = "0.4.1"
DEFAULT_APP_PORT = 9876
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
PREVIOUS_FILES_RETENTION_CHOICES: dict[str, int] = {
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
    "3d": 3 * 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
    "2w": 14 * 24 * 60 * 60,
    "1mo": 30 * 24 * 60 * 60,
    "3mo": 90 * 24 * 60 * 60,
    "6mo": 180 * 24 * 60 * 60,
}
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
PREVIOUS_FILES_LIMIT_GB_MIN = 0.5
PREVIOUS_FILES_LIMIT_GB_MAX = 1024.0
PREVIOUS_FILES_WARN_GB_MIN = 0.1
PREVIOUS_FILES_WARN_GB_MAX = 1024.0
PREVIOUS_FILES_LIMIT_GB_DEFAULT = 10.0
PREVIOUS_FILES_WARN_GB_DEFAULT = 8.0
MULTI_STEM_EXPORT_CHOICES = {"zip", "separate"}
BOOST_HARMONIES_DEFAULT_BACKGROUND_GAIN_DB = PRESET_DEFAULT_OVERLAY_GAIN_DB
BOOST_HARMONIES_DEFAULT_BASE_GAIN_DB = PRESET_DEFAULT_BASE_GAIN_DB
BOOST_GUITAR_DEFAULT_GUITAR_GAIN_DB = PRESET_DEFAULT_OVERLAY_GAIN_DB
BOOST_GUITAR_DEFAULT_BASE_GAIN_DB = PRESET_DEFAULT_BASE_GAIN_DB
BOOST_HARMONIES_VOCALS_END_PCT = 44
BOOST_HARMONIES_BACKGROUND_START_PCT = 48
BOOST_HARMONIES_BACKGROUND_END_PCT = 92
BG_VOCAL_VOCALS_END_PCT = 44
BG_VOCAL_BACKGROUND_START_PCT = 48
BG_VOCAL_BACKGROUND_END_PCT = 92
BOOST_GUITAR_MODEL_END_PCT = 92
OTHER_FILTER_GUITAR_END_PCT = 44
OTHER_FILTER_OTHER_START_PCT = 48
OTHER_FILTER_OTHER_END_PCT = 92
ALL_STEMS_VOCALS_END_PCT = 20
ALL_STEMS_INSTRUMENTAL_START_PCT = 22
ALL_STEMS_INSTRUMENTAL_END_PCT = 36
ALL_STEMS_BACKGROUND_START_PCT = 38
ALL_STEMS_BACKGROUND_END_PCT = 52
ALL_STEMS_MULTI_START_PCT = 54
ALL_STEMS_MULTI_END_PCT = 76
ALL_STEMS_DRUMS_START_PCT = 78
ALL_STEMS_DRUMS_END_PCT = 92
INTERMEDIATE_CACHE_RETENTION_SECONDS = 7 * 24 * 60 * 60
COMPAT_SETTINGS_DEFAULTS = {
    "output_format": "same_as_input",
    "output_root": str(OUTPUT_ROOT),
    "output_root_migrated_to_downloads": False,
    "output_same_as_input": False,
    "multi_stem_export": "zip",
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
    "previous_files_retention": "1w",
    "previous_files_limit_gb": PREVIOUS_FILES_LIMIT_GB_DEFAULT,
    "previous_files_warn_gb": PREVIOUS_FILES_WARN_GB_DEFAULT,
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
        kind="bs_roformer",
    ),
    "htdemucs_ft_drums": ModelSpec(
        filename="f7e0c4bc-ba3fe64a.th",
        config="config_musdb18_htdemucs.yaml",
        segment=485_100,
        overlap=4,
        kind="demucs",
        target="drums",
    ),
    "htdemucs_ft_bass": ModelSpec(
        filename="d12395a8-e57c48e6.th",
        config="config_musdb18_htdemucs.yaml",
        segment=485_100,
        overlap=4,
        kind="demucs",
        target="bass",
    ),
    "htdemucs_ft_other": ModelSpec(
        filename="92cfc3b6-ef3bcb9c.th",
        config="config_musdb18_htdemucs.yaml",
        segment=485_100,
        overlap=4,
        kind="demucs",
        target="other",
    ),
    "htdemucs_6s": ModelSpec(
        filename="5c90dfd2-34c22ccb.th",
        config="config_htdemucs_6stems.yaml",
        segment=485_100,
        overlap=4,
        kind="demucs",
    ),
    "drumsep_6s": ModelSpec(
        filename="aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
        config="aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml",
        segment=130_560,
        overlap=4,
        kind="mdx23c",
    ),
    "drumsep_4s": ModelSpec(
        filename="model_drumsep.th",
        config="config_drumsep.yaml",
        segment=1_764_000,
        overlap=4,
        kind="demucs",
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
    "htdemucs_ft_drums": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
    "htdemucs_ft_bass": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
    "htdemucs_ft_other": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th",
    "htdemucs_6s": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
    "drumsep_6s": "https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt",
    "drumsep_4s": "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.5/model_drumsep.th",
}
MODEL_DISPLAY_NAMES = {
    "vocals": "vocals",
    "instrumental": "instrumental",
    "deux": "both",
    "guitar": "guitar",
    "mel_band_karaoke": "bg vocal",
    "denoise": "denoise",
    "bs_roformer_6s": "full mix",
    "htdemucs_ft_drums": "drums",
    "htdemucs_ft_bass": "bass",
    "htdemucs_ft_other": "other",
    "htdemucs_6s": "full mix faster",
    "drumsep_6s": "drum split - 6",
    "drumsep_4s": "drum split - 4",
}
MODE_TO_STEMS = {
    "vocals": ("vocals",),
    "instrumental": ("instrumental",),
    "both_deux": ("deux",),
    "both_separate": ("vocals", "instrumental"),
    "guitar": ("guitar",),
    "mel_band_karaoke": ("mel_band_karaoke",),
    "denoise": ("denoise",),
    "bs_roformer_6s": ("bs_roformer_6s",),
    "htdemucs_ft_drums": ("htdemucs_ft_drums",),
    "htdemucs_ft_bass": ("htdemucs_ft_bass",),
    "htdemucs_ft_other": ("htdemucs_ft_other",),
    "htdemucs_6s": ("htdemucs_6s",),
    "drumsep_6s": ("drumsep_6s",),
    "drumsep_4s": ("drumsep_4s",),
    "preset_all_stems": ("all_stems",),
    "preset_voc_instrum": ("voc_instrum",),
    "preset_boost_harmonies": ("boost_harmonies",),
    "preset_boost_guitar": ("boost_guitar",),
    "preset_denoise": ("preset_denoise",),
}
MODE_REQUIRED_MODELS = {
    "vocals": ("vocals",),
    "instrumental": ("instrumental",),
    "both_deux": ("deux",),
    "both_separate": ("vocals", "instrumental"),
    "guitar": ("guitar",),
    "mel_band_karaoke": ("vocals", "mel_band_karaoke"),
    "denoise": ("denoise",),
    "bs_roformer_6s": ("bs_roformer_6s",),
    "htdemucs_ft_drums": ("htdemucs_ft_drums",),
    "htdemucs_ft_bass": ("htdemucs_ft_bass",),
    "htdemucs_ft_other": ("guitar", "htdemucs_ft_other"),
    "htdemucs_6s": ("htdemucs_6s",),
    "drumsep_6s": ("drumsep_6s",),
    "drumsep_4s": ("drumsep_4s",),
    "preset_all_stems": ("vocals", "instrumental", "mel_band_karaoke", "bs_roformer_6s", "drumsep_6s"),
    "preset_voc_instrum": ("vocals", "instrumental"),
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
    "mel_band_karaoke": ("bg vocal",),
    "denoise": ("denoise",),
    "bs_roformer_6s": ("bass", "drums", "other", "vocals", "guitar", "piano"),
    "htdemucs_ft_drums": ("drums",),
    "htdemucs_ft_bass": ("bass",),
    "htdemucs_ft_other": ("other",),
    "htdemucs_6s": ("drums", "bass", "other", "vocals", "guitar", "piano"),
    "drumsep_6s": ("kick", "snare", "toms", "hh", "ride", "crash"),
    "drumsep_4s": ("kick", "snare", "cymbals", "toms"),
    "preset_all_stems": ("vocals", "background vocals", "bass", "drums", "other", "guitar", "piano", "kick", "snare", "toms", "hh", "ride", "crash"),
    "preset_voc_instrum": ("vocals", "instrumental"),
    "preset_boost_harmonies": ("boost harmonies",),
    "preset_boost_guitar": ("boost guitar",),
    "preset_denoise": ("denoise",),
}
ALL_STEMS_DRUM_OUTPUT_LABELS = frozenset(("kick", "snare", "toms", "hh", "ride", "crash"))
ETA_HISTORY_KEYS = tuple(MODEL_SPECS.keys())
RUNTIME_STATS_VERSION = 2
RUNTIME_STATS_STAGE_LIMIT = 30
RUNTIME_STATS_TASK_LIMIT = 30
ETA_FINISHING_THRESHOLD_SECONDS = 15.0
ETA_MIN_LIVE_FRACTION = 0.15
ETA_MIN_LIVE_SECONDS = 5.0
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


def _coerce_storage_gb(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    clamped = max(minimum, min(maximum, parsed))
    return round(clamped * 10.0) / 10.0


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
    if str(normalized.get("multi_stem_export") or "") not in MULTI_STEM_EXPORT_CHOICES:
        normalized["multi_stem_export"] = "zip"
    normalized["lan_passcode_enabled"] = bool(normalized.get("lan_passcode_enabled"))
    normalized["lan_passcode"] = str(normalized.get("lan_passcode") or "")[:24]
    if str(normalized.get("lan_passcode_ttl") or "") not in LAN_AUTH_TTL_CHOICES:
        normalized["lan_passcode_ttl"] = "1d"
    if str(normalized.get("previous_files_retention") or "") not in PREVIOUS_FILES_RETENTION_CHOICES:
        normalized["previous_files_retention"] = "1w"
    normalized["previous_files_limit_gb"] = _coerce_storage_gb(
        normalized.get("previous_files_limit_gb"),
        PREVIOUS_FILES_LIMIT_GB_DEFAULT,
        minimum=PREVIOUS_FILES_LIMIT_GB_MIN,
        maximum=PREVIOUS_FILES_LIMIT_GB_MAX,
    )
    normalized["previous_files_warn_gb"] = _coerce_storage_gb(
        normalized.get("previous_files_warn_gb"),
        PREVIOUS_FILES_WARN_GB_DEFAULT,
        minimum=PREVIOUS_FILES_WARN_GB_MIN,
        maximum=PREVIOUS_FILES_WARN_GB_MAX,
    )
    if normalized["previous_files_warn_gb"] > normalized["previous_files_limit_gb"]:
        normalized["previous_files_warn_gb"] = normalized["previous_files_limit_gb"]
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


def _normalize_runtime_sample(entry: Any) -> dict[str, float] | None:
    if not isinstance(entry, dict):
        return None
    try:
        audio_seconds = float(entry.get("audio_seconds") or 0.0)
        elapsed_seconds = float(entry.get("elapsed_seconds") or 0.0)
        recorded_at = float(entry.get("recorded_at") or 0.0)
    except Exception:
        return None
    if elapsed_seconds <= 0:
        return None
    return {
        "audio_seconds": max(0.0, audio_seconds),
        "elapsed_seconds": elapsed_seconds,
        "recorded_at": recorded_at if recorded_at > 0 else time.time(),
    }


def _blank_runtime_stats() -> dict[str, Any]:
    return {
        "version": RUNTIME_STATS_VERSION,
        "stage_samples": {},
        "task_samples": {},
    }


def _load_runtime_stats() -> dict[str, Any]:
    payload = _blank_runtime_stats()
    if not ETA_HISTORY_PATH.exists():
        return payload
    try:
        data = json.loads(ETA_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        logging.getLogger("stemsplat").debug("failed to load eta history from %s", ETA_HISTORY_PATH, exc_info=True)
        return payload
    if not isinstance(data, dict):
        return payload

    if int(data.get("version") or 0) >= RUNTIME_STATS_VERSION:
        raw_stage_samples = data.get("stage_samples")
        if isinstance(raw_stage_samples, dict):
            for key, raw_entries in raw_stage_samples.items():
                if not isinstance(key, str) or not isinstance(raw_entries, list):
                    continue
                entries = [
                    normalized
                    for normalized in (_normalize_runtime_sample(entry) for entry in raw_entries[-RUNTIME_STATS_STAGE_LIMIT:])
                    if normalized is not None
                ]
                if entries:
                    payload["stage_samples"][key] = entries[-RUNTIME_STATS_STAGE_LIMIT:]
        raw_task_samples = data.get("task_samples")
        if isinstance(raw_task_samples, dict):
            for key, raw_entries in raw_task_samples.items():
                if not isinstance(key, str) or not isinstance(raw_entries, list):
                    continue
                entries = [
                    normalized
                    for normalized in (_normalize_runtime_sample(entry) for entry in raw_entries[-RUNTIME_STATS_TASK_LIMIT:])
                    if normalized is not None
                ]
                if entries:
                    payload["task_samples"][key] = entries[-RUNTIME_STATS_TASK_LIMIT:]
        return payload

    # Backward compatibility for the previous per-model ETA history shape.
    for key in ETA_HISTORY_KEYS:
        raw_entries = data.get(key)
        if not isinstance(raw_entries, list):
            continue
        normalized_entries = [
            normalized
            for normalized in (_normalize_runtime_sample(entry) for entry in raw_entries[-RUNTIME_STATS_STAGE_LIMIT:])
            if normalized is not None
        ]
        if normalized_entries:
            payload["stage_samples"][f"model:{key}"] = normalized_entries[-RUNTIME_STATS_STAGE_LIMIT:]
    return payload


def _save_runtime_stats(stats: dict[str, Any]) -> None:
    ETA_HISTORY_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _normalize_previous_file_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    entry_id = str(entry.get("id") or "").strip()
    if not entry_id:
        return None
    original_name = str(entry.get("original_name") or "").strip()
    storage_dir = str(entry.get("storage_dir") or "").strip()
    source_path = str(entry.get("source_path") or "").strip()
    finished_at = entry.get("finished_at")
    if not original_name or not storage_dir or not source_path:
        return None
    if not isinstance(finished_at, (int, float)):
        return None
    outputs = entry.get("outputs")
    if not isinstance(outputs, list):
        outputs = []
    stems = entry.get("stems")
    if not isinstance(stems, list):
        stems = []
    normalized: dict[str, Any] = {
        "id": entry_id,
        "task_id": str(entry.get("task_id") or "").strip(),
        "original_name": original_name,
        "mode": str(entry.get("mode") or "").strip(),
        "stems": [str(item) for item in stems if str(item).strip()],
        "storage_dir": storage_dir,
        "source_path": source_path,
        "source_name": str(entry.get("source_name") or Path(source_path).name or original_name),
        "outputs": [str(item) for item in outputs if str(item).strip()],
        "finished_at": float(finished_at),
        "artwork_path": str(entry.get("artwork_path") or "").strip(),
        "output_format": str(entry.get("output_format") or "").strip(),
        "video_handling": str(entry.get("video_handling") or "audio_only").strip() or "audio_only",
        "preset_settings": entry.get("preset_settings") if isinstance(entry.get("preset_settings"), dict) else None,
        "total_bytes": max(0, int(entry.get("total_bytes") or 0)),
    }
    return normalized


def _load_previous_files_index() -> list[dict[str, Any]]:
    if not PREVIOUS_FILES_INDEX_PATH.exists():
        return []
    try:
        data = json.loads(PREVIOUS_FILES_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        logging.getLogger("stemsplat").debug(
            "failed to load previous files from %s",
            PREVIOUS_FILES_INDEX_PATH,
            exc_info=True,
        )
        return []
    if not isinstance(data, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in data:
        entry = _normalize_previous_file_entry(item)
        if entry is not None:
            normalized.append(entry)
    return normalized


def _save_previous_files_index(entries: list[dict[str, Any]]) -> None:
    PREVIOUS_FILES_INDEX_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


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
runtime_stats_lock = threading.RLock()
runtime_stats = _load_runtime_stats()
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


def _locate_model_file(filename: str) -> Path | None:
    search_names = [filename, *MODEL_ALIAS_MAP.get(filename, [])]
    for base_dir in MODEL_SEARCH_DIRS:
        for search_name in search_names:
            match = _locate_case_insensitive(base_dir / search_name)
            if match:
                return match
    return None


def _model_file_exists(filename: str) -> bool:
    return _locate_model_file(filename) is not None


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


def _run_interruptible_subprocess(
    cmd: list[str],
    *,
    stop_check: Callable[[], None] | None = None,
) -> None:
    if stop_check is None:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        return

    stop_check()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        while True:
            stop_check()
            if process.poll() is not None:
                break
            time.sleep(0.12)
        _stdout, stderr = process.communicate()
    except TaskStopped:
        with contextlib.suppress(Exception):
            process.terminate()
        with contextlib.suppress(Exception):
            process.wait(timeout=1.5)
        if process.poll() is None:
            with contextlib.suppress(Exception):
                process.kill()
        with contextlib.suppress(Exception):
            process.communicate(timeout=1.0)
        raise

    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, cmd, stderr=stderr)


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
    *,
    stop_check: Callable[[], None] | None = None,
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
            _run_interruptible_subprocess(
                _build_command(include_cover),
                stop_check=stop_check,
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
    *,
    stop_check: Callable[[], None] | None = None,
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
        _run_interruptible_subprocess(
            cmd,
            stop_check=stop_check,
        )
        _strip_title_metadata(candidate)
        return candidate
    except Exception as exc:
        _cleanup_path(candidate)
        raise AppError(ErrorCode.SEPARATION_FAILED, f"Video export failed: {exc}") from exc


def _decode_audio_to_wav(
    source_path: Path,
    work_dir: Path,
    channels: int,
    *,
    stop_check: Callable[[], None] | None = None,
) -> Path:
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
        _run_interruptible_subprocess(cmd, stop_check=stop_check)
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


def _normalize_audio_tensor(audio: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    mono = audio.mean(dim=0)
    mean = mono.mean()
    std = mono.std().clamp_min(1e-8)
    return (audio - mean) / std, (mean, std)


def _denormalize_audio_tensor(audio: torch.Tensor, norm_params: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    mean, std = norm_params
    return (audio * std) + mean


def _bs_windowing_array(window_size: int, fade_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if fade_size <= 0:
        return torch.ones(window_size, device=device, dtype=dtype)
    fadein = torch.linspace(0, 1, fade_size, device=device, dtype=dtype)
    fadeout = torch.linspace(1, 0, fade_size, device=device, dtype=dtype)
    window = torch.ones(window_size, device=device, dtype=dtype)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


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
    state = _torch_load_compat(model_path, map_location="cpu", weights_only=None)
    model.load_state_dict(state.get("state_dict", state), strict=False)
    return model.to(device).eval()


def _load_bs_roformer_model(model_path: Path, config_path: Path, device: torch.device) -> torch.nn.Module:
    if BSRoformer is None:
        raise AppError(ErrorCode.MODEL_IMPORT_FAILED, f"BS-Roformer import failed: {_bs_model_import_error}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.unsafe_load(handle)
    raw_model_cfg = dict(cfg.get("model") or {})
    valid_params = set(inspect.signature(BSRoformer.__init__).parameters)
    valid_params.discard("self")
    model_kwargs = {key: value for key, value in raw_model_cfg.items() if key in valid_params}
    ignored = sorted(set(raw_model_cfg) - set(model_kwargs))
    if ignored:
        logger.info(
            "ignoring unsupported bs-roformer config keys for %s: %s",
            config_path.name,
            ", ".join(ignored),
        )
    model = BSRoformer(**model_kwargs)
    state = _torch_load_compat(model_path, map_location="cpu", weights_only=None)
    missing, unexpected = model.load_state_dict(state.get("state_dict", state), strict=False)
    if missing or unexpected:
        raise AppError(
            ErrorCode.MODEL_IMPORT_FAILED,
            "BS-Roformer checkpoint does not match the expected architecture. "
            f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}.",
        )
    setattr(model, "_stemsplat_config", cfg)
    return model.to(device).eval()


def _load_model_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.unsafe_load(handle) or {}
    if not isinstance(data, dict):
        raise AppError(ErrorCode.CONFIG_MISSING, f"Invalid config structure: {config_path.name}")
    return data


def _prepare_torch_pickle_compat() -> None:
    alias_pairs = (
        ("numpy.core.multiarray", "numpy._core.multiarray"),
        ("numpy.core.numeric", "numpy._core.numeric"),
        ("numpy._core.multiarray", "numpy.core.multiarray"),
        ("numpy._core.numeric", "numpy.core.numeric"),
    )
    for alias_name, module_name in alias_pairs:
        if alias_name in sys.modules:
            continue
        with contextlib.suppress(Exception):
            sys.modules[alias_name] = importlib.import_module(module_name)


def _torch_load_compat(model_path: Path, *, map_location: str | torch.device = "cpu", weights_only: bool | None = False) -> Any:
    _prepare_torch_pickle_compat()
    try:
        kwargs: dict[str, Any] = {"map_location": map_location}
        if weights_only is not None:
            kwargs["weights_only"] = weights_only
        return torch.load(model_path, **kwargs)
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "numpy.core.multiarray" in message or "numpy.core.numeric" in message:
            raise AppError(
                ErrorCode.MODEL_IMPORT_FAILED,
                "Missing NumPy checkpoint compatibility modules in the app bundle.",
            ) from exc
        if "incorrect header check" in lowered or "pytorchstreamreader failed" in lowered or "invalid load key" in lowered:
            raise AppError(
                ErrorCode.SEPARATION_FAILED,
                f"Model file appears corrupted: {model_path.name}. Remove it and download it again.",
            ) from exc
        raise


def _instantiate_demucs_package(package: dict[str, Any]) -> torch.nn.Module:
    klass = package.get("klass")
    args = tuple(package.get("args") or ())
    kwargs = dict(package.get("kwargs") or {})
    state = package.get("state")
    if klass is None or state is None:
        raise AppError(ErrorCode.SEPARATION_FAILED, "Demucs checkpoint is missing model metadata.")
    valid_params = set(inspect.signature(klass).parameters)
    valid_params.discard("self")
    model_kwargs = {key: value for key, value in kwargs.items() if key in valid_params}
    model = klass(*args, **model_kwargs)
    if demucs_set_state is not None:
        demucs_set_state(model, state)
    else:
        model.load_state_dict(state)
    return model


def _load_demucs_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    if TensorChunk is None or demucs_apply_model is None:
        raise AppError(ErrorCode.MODEL_IMPORT_FAILED, f"Demucs import failed: {_demucs_import_error}")
    package = _torch_load_compat(model_path, map_location="cpu", weights_only=False)
    if isinstance(package, torch.nn.Module):
        model = package
    elif isinstance(package, dict):
        model = _instantiate_demucs_package(package)
    else:
        raise AppError(ErrorCode.SEPARATION_FAILED, f"Unsupported Demucs checkpoint format: {type(package)!r}")
    return model.to(device).eval()


def _load_mdx23c_model(model_path: Path, config_path: Path, device: torch.device) -> torch.nn.Module:
    if TFC_TDF_net is None or config_namespace is None or load_not_compatible_weights is None:
        raise AppError(ErrorCode.MODEL_IMPORT_FAILED, f"DrumSep import failed: {_drumsep_import_error}")
    cfg = config_namespace(_load_model_config(config_path))
    model = TFC_TDF_net(cfg)
    state = _torch_load_compat(model_path, map_location="cpu", weights_only=False)
    load_not_compatible_weights(model, state)
    setattr(model, "_stemsplat_config", cfg)
    return model.to(device).eval()


def _load_model_from_spec(spec: ModelSpec, model_path: Path, config_path: Path, device: torch.device) -> torch.nn.Module:
    if spec.kind == "roformer":
        return _load_roformer_model(model_path, config_path, device)
    if spec.kind == "bs_roformer":
        return _load_bs_roformer_model(model_path, config_path, device)
    if spec.kind == "demucs":
        return _load_demucs_model(model_path, device)
    if spec.kind == "mdx23c":
        return _load_mdx23c_model(model_path, config_path, device)
    raise AppError(ErrorCode.INVALID_REQUEST, f"Unknown model kind: {spec.kind}")


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
        model = _load_model_from_spec(spec, model_path, config_path, self.device)
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


def _run_bs_roformer_chunks(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    progress_cb: Callable[[float], None],
    stop_check: Callable[[], None],
) -> torch.Tensor:
    config = getattr(model, "_stemsplat_config", None) or {}
    inference_cfg = dict(config.get("inference") or {})
    audio_cfg = dict(config.get("audio") or {})

    working, original_channels = _prepare_model_input(waveform, next(model.parameters()).device)
    normalize = bool(inference_cfg.get("normalize", False))
    norm_params: tuple[torch.Tensor, torch.Tensor] | None = None
    if normalize:
        working, norm_params = _normalize_audio_tensor(working)

    chunk_size = int(inference_cfg.get("chunk_size") or audio_cfg.get("chunk_size") or working.shape[-1])
    num_overlap = max(1, int(inference_cfg.get("num_overlap") or 1))
    batch_size = max(1, int(inference_cfg.get("batch_size") or 1))
    num_instruments = int(getattr(model, "num_stems", 1))
    fade_size = max(0, chunk_size // 10)
    step = max(1, chunk_size // num_overlap)
    border = max(0, chunk_size - step)
    length_init = int(working.shape[-1])

    if length_init > 2 * border and border > 0:
        working = F.pad(working, (border, border), mode="reflect")

    total_samples = int(working.shape[-1])
    total_chunks = max(1, math.ceil(total_samples / step))
    result = torch.zeros(
        (num_instruments, working.shape[0], total_samples),
        device=working.device,
        dtype=torch.float32,
    )
    counter = torch.zeros_like(result)
    base_window = _bs_windowing_array(chunk_size, fade_size, working.device, torch.float32)

    progress_cb(0.0)
    batch_data: list[torch.Tensor] = []
    batch_locations: list[tuple[int, int]] = []
    processed_chunks = 0
    index = 0

    with torch.inference_mode():
        while index < total_samples:
            stop_check()
            part = working[:, index : index + chunk_size]
            chunk_len = int(part.shape[-1])
            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
            part = F.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0.0)
            batch_data.append(part)
            batch_locations.append((index, chunk_len))
            index += step

            if len(batch_data) < batch_size and index < total_samples:
                continue

            batch_tensor = torch.stack(batch_data, dim=0)
            predicted = model(batch_tensor)

            for batch_index, (start, seg_len) in enumerate(batch_locations):
                chunk_window = base_window.clone()
                if start == 0:
                    chunk_window[:fade_size] = 1.0
                if start + step >= total_samples:
                    chunk_window[-fade_size:] = 1.0
                piece = predicted[batch_index, ..., :seg_len].to(dtype=torch.float32)
                result[..., start : start + seg_len] += piece * chunk_window[:seg_len]
                counter[..., start : start + seg_len] += chunk_window[:seg_len]
                processed_chunks += 1
                progress_cb(processed_chunks / total_chunks)

            batch_data.clear()
            batch_locations.clear()

    estimated = result / counter.clamp_min(1e-8)

    if length_init > 2 * border and border > 0:
        estimated = estimated[..., border:-border]

    if norm_params is not None:
        estimated = _denormalize_audio_tensor(estimated, norm_params)

    outputs = []
    for stem_index in range(estimated.shape[0]):
        outputs.append(_restore_output_channels(estimated[stem_index], original_channels))
    return torch.stack(outputs, dim=0)


def _overlap_fraction(overlap_count: int) -> float:
    overlap_count = max(1, int(overlap_count))
    return max(0.0, min(0.95, 1.0 - (1.0 / float(overlap_count))))


def _run_demucs_chunks(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    segment: int,
    overlap: int,
    progress_cb: Callable[[float], None],
    stop_check: Callable[[], None],
) -> torch.Tensor:
    if TensorChunk is None or demucs_apply_model is None:
        raise AppError(ErrorCode.MODEL_IMPORT_FAILED, f"Demucs import failed: {_demucs_import_error}")
    working, original_channels = _prepare_model_input(waveform, next(model.parameters()).device)
    mix = working.unsqueeze(0)
    length = int(mix.shape[-1])
    segment_seconds = max(1.0 / 44100.0, float(segment) / 44100.0)
    model_segment = getattr(model, "segment", None)
    with contextlib.suppress(Exception):
        if model_segment is not None:
            segment_seconds = min(segment_seconds, float(model_segment))
    segment_length = max(1, int(round(segment_seconds * float(getattr(model, "samplerate", 44100)))))
    stride = max(1, int((1.0 - _overlap_fraction(overlap)) * segment_length))
    offsets = list(range(0, length, stride)) or [0]
    weight = torch.cat(
        [
            torch.arange(1, segment_length // 2 + 1, device=working.device),
            torch.arange(segment_length - segment_length // 2, 0, -1, device=working.device),
        ]
    )
    weight = weight / weight.max().clamp_min(1e-6)
    out = torch.zeros((1, len(getattr(model, "sources", [])), working.shape[0], length), device=working.device)
    sum_weight = torch.zeros(length, device=working.device)

    progress_cb(0.0)
    with torch.no_grad():
        for index, offset in enumerate(offsets, start=1):
            stop_check()
            chunk = TensorChunk(mix, offset, segment_length)
            chunk_out = demucs_apply_model(
                model,
                chunk,
                shifts=0,
                split=False,
                overlap=_overlap_fraction(overlap),
                device=working.device,
                segment=segment_seconds,
            )
            chunk_length = int(chunk_out.shape[-1])
            out[..., offset : offset + chunk_length] += weight[:chunk_length] * chunk_out[..., :chunk_length]
            sum_weight[offset : offset + chunk_length] += weight[:chunk_length]
            progress_cb(index / max(1, len(offsets)))

    out = out / sum_weight.clamp_min(1e-6)
    restored = out[0]
    outputs = []
    for index in range(restored.shape[0]):
        outputs.append(_restore_output_channels(restored[index], original_channels))
    return torch.stack(outputs, dim=0)


def _run_mdx23c_chunks(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    progress_cb: Callable[[float], None],
    stop_check: Callable[[], None],
) -> torch.Tensor:
    if demix_mdx23c is None:
        raise AppError(ErrorCode.MODEL_IMPORT_FAILED, f"DrumSep import failed: {_drumsep_import_error}")
    config = getattr(model, "_stemsplat_config", None)
    if config is None:
        raise AppError(ErrorCode.CONFIG_MISSING, "DrumSep config missing from loaded model.")
    working, original_channels = _prepare_model_input(waveform, next(model.parameters()).device)
    predicted = demix_mdx23c(config, model, working, next(model.parameters()).device, progress_cb, stop_check)
    outputs = []
    for index in range(predicted.shape[0]):
        outputs.append(_restore_output_channels(predicted[index], original_channels))
    return torch.stack(outputs, dim=0)


def _run_model_for_spec(
    model_key: str,
    model: torch.nn.Module,
    waveform: torch.Tensor,
    progress_cb: Callable[[float], None],
    stop_check: Callable[[], None],
) -> torch.Tensor:
    spec = MODEL_SPECS[model_key]
    if spec.kind == "roformer":
        return _run_model_chunks(model, waveform, spec.segment, spec.overlap, progress_cb, stop_check)
    if spec.kind == "bs_roformer":
        return _run_bs_roformer_chunks(model, waveform, progress_cb, stop_check)
    if spec.kind == "demucs":
        return _run_demucs_chunks(model, waveform, spec.segment, spec.overlap, progress_cb, stop_check)
    if spec.kind == "mdx23c":
        return _run_mdx23c_chunks(model, waveform, progress_cb, stop_check)
    raise AppError(ErrorCode.INVALID_REQUEST, f"Unsupported model kind: {spec.kind}")


def _extract_target_tensor(model_key: str, model: torch.nn.Module, prediction: torch.Tensor) -> torch.Tensor:
    spec = MODEL_SPECS[model_key]
    if not spec.target:
        return prediction[0]
    sources = [str(item) for item in getattr(model, "sources", [])]
    if spec.target in sources:
        return prediction[sources.index(spec.target)]
    expected_labels = MODE_OUTPUT_LABELS.get(model_key, ())
    if spec.target in expected_labels:
        return prediction[list(expected_labels).index(spec.target)]
    raise AppError(ErrorCode.SEPARATION_FAILED, f"Target source '{spec.target}' not found in model output.")


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
        decoded_path = _decode_audio_to_wav(
            source_path,
            work_dir,
            source_info.channels,
            stop_check=lambda: _stop_check(task_id),
        )
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
        "running background vocal model": "mel_band_karaoke",
        "running harmony background model": "mel_band_karaoke",
        "running denoise model": "denoise",
        "running bs-roformer 6s model": "bs_roformer_6s",
        "running full mix model": "bs_roformer_6s",
        "running htdemucs4 ft drums model": "htdemucs_ft_drums",
        "running htdemucs4 ft bass model": "htdemucs_ft_bass",
        "running htdemucs4 ft other model": "htdemucs_ft_other",
        "running htdemucs4 6 stem model": "htdemucs_6s",
        "running drumsep 6 stem model": "drumsep_6s",
        "running drum sep 4 stem model": "drumsep_4s",
    }
    for prefix, model_key in mapping.items():
        if prefix in stage:
            return model_key
    return None


def _runtime_stage_key_for_display(stage_text: str) -> str | None:
    stage = str(stage_text or "").lower()
    if "loading models" in stage:
        return "load_models"
    if "preparing audio" in stage:
        return "prepare_audio"
    if "mixing boost harmonies" in stage or "mixing boost guitar" in stage:
        return "mix_preset"
    if "exporting" in stage:
        return "export"
    return _stage_model_key(stage)


def _runtime_model_sequence(mode: str) -> list[str]:
    if mode in {"vocals", "instrumental", "deux", "guitar", "denoise", "bs_roformer_6s"}:
        return [mode]
    if mode == "mel_band_karaoke":
        return ["vocals", "mel_band_karaoke"]
    if mode in {"htdemucs_ft_drums", "htdemucs_ft_bass", "htdemucs_6s"}:
        return [mode]
    if mode == "htdemucs_ft_other":
        return ["guitar", "htdemucs_ft_other"]
    if mode in {"drumsep_6s", "drumsep_4s"}:
        return [mode]
    if mode == "both_deux":
        return ["deux"]
    if mode == "preset_all_stems":
        return ["vocals", "instrumental", "mel_band_karaoke", "bs_roformer_6s", "drumsep_6s"]
    if mode in {"both_separate", "preset_voc_instrum"}:
        return ["vocals", "instrumental"]
    if mode == "preset_boost_harmonies":
        return ["vocals", "mel_band_karaoke"]
    if mode == "preset_boost_guitar":
        return ["guitar"]
    if mode == "preset_denoise":
        return ["denoise"]
    return []


def _runtime_stage_samples(stats_key: str) -> list[dict[str, float]]:
    with runtime_stats_lock:
        return list(runtime_stats.get("stage_samples", {}).get(stats_key) or [])


def _runtime_task_samples(task_key: str) -> list[dict[str, float]]:
    with runtime_stats_lock:
        return list(runtime_stats.get("task_samples", {}).get(task_key) or [])


def _append_runtime_sample(bucket: str, stats_key: str, sample: dict[str, float]) -> None:
    normalized = _normalize_runtime_sample(sample)
    if normalized is None:
        return
    limit = RUNTIME_STATS_STAGE_LIMIT if bucket == "stage_samples" else RUNTIME_STATS_TASK_LIMIT
    with runtime_stats_lock:
        target = runtime_stats.setdefault(bucket, {})
        entries = list(target.get(stats_key) or [])
        entries.append(normalized)
        target[stats_key] = entries[-limit:]
        _save_runtime_stats(runtime_stats)


def _weighted_quantile(pairs: list[tuple[float, float]], quantile: float) -> float | None:
    if not pairs:
        return None
    sorted_pairs = sorted(((float(value), max(1e-6, float(weight))) for value, weight in pairs), key=lambda item: item[0])
    total_weight = sum(weight for _, weight in sorted_pairs)
    if total_weight <= 0:
        return None
    threshold = max(0.0, min(1.0, quantile)) * total_weight
    cumulative = 0.0
    for value, weight in sorted_pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return sorted_pairs[-1][0]


def _sample_weight(audio_seconds: float, sample: dict[str, float], *, index: int, total: int) -> float:
    sample_audio = max(0.0, float(sample.get("audio_seconds") or 0.0))
    if audio_seconds > 0 and sample_audio > 0:
        rel_distance = abs(sample_audio - audio_seconds) / max(audio_seconds, sample_audio, 1.0)
        audio_weight = 1.0 / max(0.12, 0.22 + (rel_distance * 1.8))
    else:
        audio_weight = 1.0
    recency_weight = 0.65 + (0.35 * ((index + 1) / max(1, total)))
    return audio_weight * recency_weight


def _scaled_sample_elapsed(sample: dict[str, float], audio_seconds: float, *, scale_by_audio: bool) -> float:
    elapsed_seconds = max(0.1, float(sample.get("elapsed_seconds") or 0.0))
    if not scale_by_audio or audio_seconds <= 0:
        return elapsed_seconds
    sample_audio = max(0.1, float(sample.get("audio_seconds") or 0.0))
    return max(0.1, elapsed_seconds * (audio_seconds / sample_audio))


def _predict_from_samples(
    samples: list[dict[str, float]],
    *,
    audio_seconds: float,
    scale_by_audio: bool,
    conservative: bool,
) -> tuple[float | None, int]:
    if not samples:
        return None, 0
    recent_samples = samples[-RUNTIME_STATS_STAGE_LIMIT:]
    weighted_predictions = [
        (
            _scaled_sample_elapsed(sample, audio_seconds, scale_by_audio=scale_by_audio),
            _sample_weight(audio_seconds, sample, index=index, total=len(recent_samples)),
        )
        for index, sample in enumerate(recent_samples)
    ]
    median = _weighted_quantile(weighted_predictions, 0.5)
    if median is None:
        return None, len(recent_samples)
    if not conservative:
        return max(0.1, median), len(recent_samples)
    p70 = _weighted_quantile(weighted_predictions, 0.7)
    if p70 is None:
        p70 = median
    return max(0.1, (median * 0.7) + (p70 * 0.3)), len(recent_samples)


def _fallback_model_runtime_seconds(model_key: str, audio_seconds: float) -> float:
    seconds = max(1.0, float(audio_seconds or 0.0))
    if model_key == "bs_roformer_6s":
        return max(20.0, seconds * 0.95)
    kind = MODEL_SPECS.get(model_key, ModelSpec("", "", 0, 0)).kind
    if kind == "demucs":
        return max(8.0, seconds * 0.18)
    if kind == "mdx23c":
        return max(10.0, seconds * 0.28)
    return max(12.0, seconds * 0.55)


def _predict_fixed_stage_runtime_seconds(stage_key: str, audio_seconds: float) -> tuple[float, int, str]:
    samples = _runtime_stage_samples(f"fixed:{stage_key}")
    prediction, count = _predict_from_samples(
        samples,
        audio_seconds=audio_seconds,
        scale_by_audio=False,
        conservative=False,
    )
    if prediction is not None:
        return max(0.5, prediction), count, "history"
    if stage_key == "load_models":
        return 2.5, 0, "fallback"
    if stage_key == "prepare_audio":
        return max(3.0, min(18.0, max(1.0, audio_seconds) * 0.03)), 0, "fallback"
    if stage_key == "mix_preset":
        return 3.0, 0, "fallback"
    return 1.0, 0, "fallback"


def _predict_model_runtime_seconds(model_key: str, audio_seconds: float) -> float | None:
    if model_key not in ETA_HISTORY_KEYS:
        return None
    samples = _runtime_stage_samples(f"model:{model_key}")
    prediction, _count = _predict_from_samples(
        samples,
        audio_seconds=audio_seconds,
        scale_by_audio=True,
        conservative=True,
    )
    if prediction is not None:
        return prediction
    return _fallback_model_runtime_seconds(model_key, audio_seconds)


def _predict_model_stage_runtime(model_key: str, audio_seconds: float) -> tuple[float, int, str]:
    samples = _runtime_stage_samples(f"model:{model_key}")
    prediction, count = _predict_from_samples(
        samples,
        audio_seconds=audio_seconds,
        scale_by_audio=True,
        conservative=True,
    )
    if prediction is not None:
        return prediction, count, "history"
    return _fallback_model_runtime_seconds(model_key, audio_seconds), 0, "fallback"


def _predict_export_runtime_seconds(output_format: str, output_count: int, audio_seconds: float) -> tuple[float, int, str]:
    stats_key = f"export:{output_format}:{max(1, output_count)}"
    samples = _runtime_stage_samples(stats_key)
    prediction, count = _predict_from_samples(
        samples,
        audio_seconds=audio_seconds,
        scale_by_audio=True,
        conservative=False,
    )
    if prediction is not None:
        return max(1.0, prediction), count, "history"
    fallback = max(2.0, max(1, output_count) * (1.5 + (max(1.0, audio_seconds) * 0.01)))
    return fallback, 0, "fallback"


def _predict_task_runtime_seconds(mode: str, audio_seconds: float) -> float | None:
    if audio_seconds <= 0:
        return None
    samples = _runtime_task_samples(f"task:{mode}")
    prediction, _count = _predict_from_samples(
        samples,
        audio_seconds=audio_seconds,
        scale_by_audio=True,
        conservative=True,
    )
    return prediction


def _build_runtime_plan(task: dict[str, Any], *, audio_seconds: float | None = None) -> dict[str, Any]:
    mode = str(task.get("mode") or "")
    output_format = str(task.get("output_format") or _compat_settings_payload()["output_format"])
    export_count = max(1, _expected_output_count(mode))
    audio_value = max(0.0, float(audio_seconds if audio_seconds is not None else (task.get("audio_seconds") or 0.0)))
    stages: list[dict[str, Any]] = []

    load_seconds, load_count, load_basis = _predict_fixed_stage_runtime_seconds("load_models", audio_value)
    stages.append(
        {
            "stage_key": "load_models",
            "stats_key": "fixed:load_models",
            "predicted_seconds": load_seconds,
            "started_at": None,
            "completed_at": None,
            "live_fraction": None,
            "supports_live_fraction": False,
            "prediction_basis": load_basis,
            "prediction_samples": load_count,
        }
    )

    prep_seconds, prep_count, prep_basis = _predict_fixed_stage_runtime_seconds("prepare_audio", audio_value)
    stages.append(
        {
            "stage_key": "prepare_audio",
            "stats_key": "fixed:prepare_audio",
            "predicted_seconds": prep_seconds,
            "started_at": None,
            "completed_at": None,
            "live_fraction": None,
            "supports_live_fraction": False,
            "prediction_basis": prep_basis,
            "prediction_samples": prep_count,
        }
    )

    for model_key in _runtime_model_sequence(mode):
        predicted_seconds, sample_count, basis = _predict_model_stage_runtime(model_key, audio_value)
        stages.append(
            {
                "stage_key": model_key,
                "stats_key": f"model:{model_key}",
                "predicted_seconds": predicted_seconds,
                "started_at": None,
                "completed_at": None,
                "live_fraction": None,
                "supports_live_fraction": True,
                "prediction_basis": basis,
                "prediction_samples": sample_count,
            }
        )

    if mode in {"preset_boost_harmonies", "preset_boost_guitar"}:
        mix_seconds, mix_count, mix_basis = _predict_fixed_stage_runtime_seconds("mix_preset", audio_value)
        stages.append(
            {
                "stage_key": "mix_preset",
                "stats_key": "fixed:mix_preset",
                "predicted_seconds": mix_seconds,
                "started_at": None,
                "completed_at": None,
                "live_fraction": None,
                "supports_live_fraction": False,
                "prediction_basis": mix_basis,
                "prediction_samples": mix_count,
            }
        )

    export_seconds, export_count_samples, export_basis = _predict_export_runtime_seconds(
        output_format,
        export_count,
        audio_value,
    )
    stages.append(
        {
            "stage_key": "export",
            "stats_key": f"export:{output_format}:{export_count}",
            "predicted_seconds": export_seconds,
            "started_at": None,
            "completed_at": None,
            "live_fraction": None,
            "supports_live_fraction": True,
            "prediction_basis": export_basis,
            "prediction_samples": export_count_samples,
        }
    )

    return {"stages": stages}


def _refresh_runtime_plan(task: dict[str, Any], *, audio_seconds: float | None = None) -> None:
    previous_plan = task.get("runtime_plan") if isinstance(task.get("runtime_plan"), dict) else {"stages": []}
    previous_by_key = {
        str(stage.get("stage_key") or ""): stage
        for stage in list(previous_plan.get("stages") or [])
        if isinstance(stage, dict)
    }
    next_plan = _build_runtime_plan(task, audio_seconds=audio_seconds)
    for stage in next_plan["stages"]:
        previous = previous_by_key.get(str(stage.get("stage_key") or ""))
        if previous is None:
            continue
        stage["started_at"] = previous.get("started_at")
        stage["completed_at"] = previous.get("completed_at")
        stage["live_fraction"] = previous.get("live_fraction")
        if previous.get("completed_at") and isinstance(previous.get("predicted_seconds"), (int, float)):
            stage["predicted_seconds"] = max(0.1, float(previous["predicted_seconds"]))
    task["runtime_plan"] = next_plan


def _runtime_plan_stage(task: dict[str, Any], stage_key: str) -> dict[str, Any] | None:
    runtime_plan = task.get("runtime_plan")
    if not isinstance(runtime_plan, dict):
        return None
    for stage in list(runtime_plan.get("stages") or []):
        if isinstance(stage, dict) and str(stage.get("stage_key") or "") == stage_key:
            return stage
    return None


def _complete_runtime_stage(task: dict[str, Any], stage_key: str | None, *, now: float, record_sample: bool = True) -> None:
    if not stage_key:
        return
    stage = _runtime_plan_stage(task, stage_key)
    if stage is None or stage.get("completed_at") is not None:
        return
    started_at = stage.get("started_at")
    if not isinstance(started_at, (int, float)):
        return
    elapsed_seconds = max(0.1, now - float(started_at))
    stage["completed_at"] = now
    stage["live_fraction"] = 1.0 if stage.get("supports_live_fraction") else None
    stage["predicted_seconds"] = elapsed_seconds
    if record_sample and not str(stage.get("stats_key") or "").startswith("model:"):
        _append_runtime_sample(
            "stage_samples",
            str(stage.get("stats_key") or ""),
            {
                "audio_seconds": float(task.get("audio_seconds") or 0.0),
                "elapsed_seconds": elapsed_seconds,
                "recorded_at": now,
            },
        )


def _stage_live_fraction_from_legacy_pct(task: dict[str, Any], stage_text: str, pct: int) -> float | None:
    stage_key = _runtime_stage_key_for_display(stage_text)
    if stage_key is None:
        return None
    if stage_key == "export":
        span = max(1, EXPORT_PROGRESS_END_PCT - EXPORT_PROGRESS_START_PCT)
        return max(0.0, min(1.0, (max(0, min(100, int(pct))) - EXPORT_PROGRESS_START_PCT) / span))
    return _stage_progress_fraction(str(task.get("mode") or ""), stage_text, pct)


def _effective_runtime_stage_progress(task: dict[str, Any], stage: dict[str, Any], now: float) -> tuple[float, float, str]:
    predicted_seconds = max(0.1, float(stage.get("predicted_seconds") or 0.1))
    started_at = float(stage.get("started_at") or now)
    elapsed_seconds = max(0.0, now - started_at)
    live_fraction = stage.get("live_fraction")
    live_fraction_value = None
    if isinstance(live_fraction, (int, float)):
        live_fraction_value = max(0.0, min(1.0, float(live_fraction)))

    basis = str(stage.get("prediction_basis") or "fallback")
    sample_count = int(stage.get("prediction_samples") or 0)
    eta_state = "steady" if basis == "history" and sample_count >= 3 else "estimating"

    if bool(stage.get("supports_live_fraction")):
        if live_fraction_value is not None and live_fraction_value > 0:
            if live_fraction_value >= ETA_MIN_LIVE_FRACTION and elapsed_seconds >= ETA_MIN_LIVE_SECONDS:
                live_total = elapsed_seconds / max(live_fraction_value, 1e-3)
                live_weight = min(0.75, live_fraction_value * 0.75)
                predicted_seconds = max(
                    elapsed_seconds + 1.0,
                    (predicted_seconds * (1.0 - live_weight)) + (live_total * live_weight),
                )
                eta_state = "steady" if live_fraction_value >= 0.4 else "calibrating"
            else:
                eta_state = "calibrating"
            return predicted_seconds, live_fraction_value, eta_state
        fraction = min(0.97, elapsed_seconds / max(predicted_seconds, 1e-3))
        return predicted_seconds, fraction, eta_state

    fraction = min(0.97, elapsed_seconds / max(predicted_seconds, 1e-3))
    return predicted_seconds, fraction, eta_state


def _update_task_runtime_view(task: dict[str, Any], *, now: float | None = None) -> None:
    now = time.time() if now is None else now
    status = str(task.get("status") or "")
    runtime_plan = task.get("runtime_plan")
    if status != "running" or not isinstance(runtime_plan, dict):
        return

    stages = [stage for stage in list(runtime_plan.get("stages") or []) if isinstance(stage, dict)]
    if not stages:
        return

    completed_seconds = 0.0
    consumed_seconds = 0.0
    predicted_total_seconds = 0.0
    eta_state = "estimating"
    current_stage_key = _runtime_stage_key_for_display(str(task.get("stage") or ""))

    for stage in stages:
        stage_key = str(stage.get("stage_key") or "")
        if stage.get("completed_at") is not None:
            stage_seconds = max(0.1, float(stage.get("predicted_seconds") or 0.1))
            completed_seconds += stage_seconds
            predicted_total_seconds += stage_seconds
            consumed_seconds = completed_seconds
            continue
        if current_stage_key and stage_key == current_stage_key and stage.get("started_at") is not None:
            effective_seconds, fraction, stage_eta_state = _effective_runtime_stage_progress(task, stage, now)
            predicted_total_seconds += effective_seconds
            consumed_seconds = completed_seconds + (effective_seconds * max(0.0, min(1.0, fraction)))
            eta_state = stage_eta_state
            continue
        predicted_total_seconds += max(0.1, float(stage.get("predicted_seconds") or 0.1))

    if predicted_total_seconds <= 0:
        task["pct"] = max(0, min(99, int(task.get("pct") or 0)))
        task["eta_seconds"] = None
        task["eta_state"] = "estimating"
        return

    remaining_seconds = max(0.0, predicted_total_seconds - consumed_seconds)
    pct = int(math.floor(100.0 * (consumed_seconds / predicted_total_seconds)))
    pct = max(0, min(99, pct))
    if remaining_seconds > max(45.0, predicted_total_seconds * 0.15):
        pct = min(pct, 90)

    task["predicted_total_seconds"] = predicted_total_seconds
    task["pct"] = pct
    task["eta_seconds"] = _stabilize_eta(task, remaining_seconds, now=now, stage_text=current_stage_key or "")
    task["eta_state"] = "finishing" if remaining_seconds <= ETA_FINISHING_THRESHOLD_SECONDS else eta_state



app = FastAPI()
app.state.runtime_status_provider = None
tasks_lock = threading.RLock()
tasks: dict[str, dict[str, Any]] = {}
task_queue: queue.Queue[str] = queue.Queue()
task_runtime_lock = threading.RLock()
task_runtimes: dict[str, dict[str, Any]] = {}
queue_resume_event = threading.Event()
queue_resume_event.set()
previous_files_lock = threading.RLock()
previous_files_index: list[dict[str, Any]] = _load_previous_files_index()
model_download_lock = threading.RLock()
model_download_state: dict[str, Any] = {
    "status": "idle",
    "pct": 0,
    "step": "",
    "current_model": "",
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "download_rate_bytes_per_sec": 0.0,
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
        "preferred_port": DEFAULT_APP_PORT,
        "current_port": DEFAULT_APP_PORT,
        "client_url": f"http://127.0.0.1:{DEFAULT_APP_PORT}/",
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


def _previous_files_retention_seconds(settings: dict[str, Any] | None = None) -> int:
    source = settings or _compat_settings_payload()
    choice = str(source.get("previous_files_retention") or "1w")
    return PREVIOUS_FILES_RETENTION_CHOICES.get(choice, PREVIOUS_FILES_RETENTION_CHOICES["1w"])


def _previous_files_limit_gb(settings: dict[str, Any] | None = None) -> float:
    source = settings or _compat_settings_payload()
    return _coerce_storage_gb(
        source.get("previous_files_limit_gb"),
        PREVIOUS_FILES_LIMIT_GB_DEFAULT,
        minimum=PREVIOUS_FILES_LIMIT_GB_MIN,
        maximum=PREVIOUS_FILES_LIMIT_GB_MAX,
    )


def _previous_files_warn_gb(settings: dict[str, Any] | None = None) -> float:
    source = settings or _compat_settings_payload()
    limit_gb = _previous_files_limit_gb(source)
    warn_gb = _coerce_storage_gb(
        source.get("previous_files_warn_gb"),
        PREVIOUS_FILES_WARN_GB_DEFAULT,
        minimum=PREVIOUS_FILES_WARN_GB_MIN,
        maximum=PREVIOUS_FILES_WARN_GB_MAX,
    )
    return min(limit_gb, warn_gb)


def _gb_to_bytes(value_gb: float) -> int:
    return max(0, int(round(float(value_gb) * (1024**3))))


def _previous_files_limit_bytes(settings: dict[str, Any] | None = None) -> int:
    return _gb_to_bytes(_previous_files_limit_gb(settings))


def _previous_files_warn_bytes(settings: dict[str, Any] | None = None) -> int:
    return _gb_to_bytes(_previous_files_warn_gb(settings))


def _path_total_bytes(path: Path) -> int:
    try:
        if path.is_file():
            return max(0, int(path.stat().st_size))
        if not path.exists():
            return 0
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                with contextlib.suppress(OSError):
                    total += max(0, int(child.stat().st_size))
        return total
    except Exception:
        return 0


def _previous_file_entry_total_bytes(entry: dict[str, Any]) -> int:
    storage_dir = Path(str(entry.get("storage_dir") or "")).expanduser()
    return _path_total_bytes(storage_dir)


def _history_storage_payload(
    entries: list[dict[str, Any]] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_settings = settings or _compat_settings_payload()
    active_entries = entries if entries is not None else _prune_previous_files(save=False)
    usage_bytes = 0
    for entry in active_entries:
        size_bytes = max(0, int(entry.get("total_bytes") or 0))
        if size_bytes <= 0:
            size_bytes = _previous_file_entry_total_bytes(entry)
        usage_bytes += size_bytes
    limit_bytes = _previous_files_limit_bytes(active_settings)
    warn_bytes = _previous_files_warn_bytes(active_settings)
    return {
        "usage_bytes": usage_bytes,
        "limit_bytes": limit_bytes,
        "warn_bytes": warn_bytes,
        "near_limit": usage_bytes >= warn_bytes if warn_bytes > 0 else False,
        "at_limit": usage_bytes >= limit_bytes if limit_bytes > 0 else False,
    }


def _previous_file_output_paths(entry: dict[str, Any]) -> tuple[Path, list[Path]]:
    storage_dir = Path(str(entry.get("storage_dir") or "")).expanduser()
    if not storage_dir.exists() or not storage_dir.is_dir():
        raise AppError(ErrorCode.INVALID_REQUEST, "file doesn't exist")
    outputs_dir = storage_dir / "outputs"
    outputs = [outputs_dir / str(name) for name in (entry.get("outputs") or [])]
    existing_outputs = [path for path in outputs if path.exists() and path.is_file()]
    if outputs and not existing_outputs:
        raise AppError(ErrorCode.INVALID_REQUEST, "file doesn't exist")
    return outputs_dir, existing_outputs


def _public_previous_file(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(entry.get("id") or ""),
        "task_id": str(entry.get("task_id") or ""),
        "name": str(entry.get("original_name") or ""),
        "mode": str(entry.get("mode") or ""),
        "stems": list(entry.get("stems") or []),
        "outputs": list(entry.get("outputs") or []),
        "finished_at": float(entry.get("finished_at") or 0.0),
        "preset_settings": entry.get("preset_settings") if isinstance(entry.get("preset_settings"), dict) else None,
        "total_bytes": max(0, int(entry.get("total_bytes") or 0)),
        "artwork_url": f"/api/history/{entry['id']}/artwork",
    }


def _require_previous_file(entry_id: str) -> dict[str, Any]:
    _prune_previous_files(save=False)
    with previous_files_lock:
        for entry in previous_files_index:
            if str(entry.get("id") or "") == str(entry_id):
                return dict(entry)
    raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid previous file id")


def _prune_previous_files(*, save: bool = True) -> list[dict[str, Any]]:
    cutoff = time.time() - _previous_files_retention_seconds()
    limit_bytes = _previous_files_limit_bytes()
    kept: list[dict[str, Any]] = []
    removed_dirs: list[Path] = []
    with previous_files_lock:
        for entry in previous_files_index:
            storage_dir = Path(str(entry.get("storage_dir") or "")).expanduser()
            source_path = Path(str(entry.get("source_path") or "")).expanduser()
            if float(entry.get("finished_at") or 0.0) < cutoff:
                removed_dirs.append(storage_dir)
                continue
            if not storage_dir.exists() or not source_path.exists():
                removed_dirs.append(storage_dir)
                continue
            try:
                _previous_file_output_paths(entry)
            except AppError:
                removed_dirs.append(storage_dir)
                continue
            entry["total_bytes"] = _previous_file_entry_total_bytes(entry)
            kept.append(entry)
        kept.sort(key=lambda item: float(item.get("finished_at") or 0.0), reverse=True)
        if limit_bytes > 0:
            limited: list[dict[str, Any]] = []
            usage_bytes = 0
            for entry in kept:
                entry_size = max(0, int(entry.get("total_bytes") or 0))
                if usage_bytes + entry_size > limit_bytes:
                    removed_dirs.append(Path(str(entry.get("storage_dir") or "")).expanduser())
                    continue
                limited.append(entry)
                usage_bytes += entry_size
            kept = limited
        previous_files_index[:] = kept
        if save:
            _save_previous_files_index(previous_files_index)
    for path in removed_dirs:
        _cleanup_path(path)
    return [dict(entry) for entry in kept]


def _archive_previous_file(task_id: str, out_dir: Path, outputs: list[str]) -> dict[str, Any] | None:
    with tasks_lock:
        task = dict(tasks.get(task_id) or {})
    if not task:
        return None
    source_path = Path(str(task.get("source_path") or "")).expanduser()
    if not source_path.exists() or not outputs:
        return None
    entry_id = str(uuid.uuid4())
    storage_dir = _ensure_dir(PREVIOUS_FILES_DIR / entry_id)
    outputs_dir = _ensure_dir(storage_dir / "outputs")
    stored_outputs: list[str] = []
    for output_name in outputs:
        source_output = out_dir / str(output_name)
        if not source_output.exists() or not source_output.is_file():
            continue
        target_output = outputs_dir / str(output_name)
        target_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source_output, target_output)
        except Exception:
            logger.warning("failed to archive output %s for task %s", source_output, task_id, exc_info=True)
            continue
        stored_outputs.append(_relative_output_name(target_output, outputs_dir))
    if not stored_outputs:
        _cleanup_path(storage_dir)
        return None
    archived_source = storage_dir / source_path.name
    try:
        shutil.copy2(source_path, archived_source)
    except Exception:
        logger.warning("failed to archive source %s for task %s", source_path, task_id, exc_info=True)
        _cleanup_path(storage_dir)
        return None
    artwork_path = ""
    with contextlib.suppress(Exception):
        current_artwork = _extract_task_artwork(task_id)
        if current_artwork is not None and current_artwork.exists():
            archived_artwork = storage_dir / f"artwork{current_artwork.suffix.lower() or '.jpg'}"
            shutil.copy2(current_artwork, archived_artwork)
            artwork_path = str(archived_artwork)
    entry = {
        "id": entry_id,
        "task_id": task_id,
        "original_name": str(task.get("original_name") or archived_source.name),
        "mode": str(task.get("mode") or ""),
        "stems": _mode_to_stems(str(task.get("mode") or "")),
        "storage_dir": str(storage_dir),
        "source_path": str(archived_source),
        "source_name": archived_source.name,
        "outputs": stored_outputs,
        "finished_at": float(task.get("finished_at") or time.time()),
        "artwork_path": artwork_path,
        "output_format": str(task.get("output_format") or ""),
        "video_handling": str(task.get("video_handling") or "audio_only"),
        "preset_settings": task.get("preset_settings_snapshot") if isinstance(task.get("preset_settings_snapshot"), dict) else None,
        "total_bytes": _path_total_bytes(storage_dir),
    }
    with previous_files_lock:
        previous_files_index.insert(0, entry)
        _save_previous_files_index(previous_files_index)
    _prune_previous_files()
    return entry


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
    models_status = _models_status_payload()
    missing = list(models_status["missing"])
    prompt_state = str(_compat_settings_payload().get("model_prompt_state") or MODEL_PROMPT_PENDING)
    payload.update(
        {
            "missing": missing,
            "models_dir": str(MODEL_DIR),
            "models": models_status["models"],
            "downloaded_total_bytes": int(models_status["downloaded_total_bytes"]),
            "prompt_state": MODEL_PROMPT_COMPLETE if not missing else prompt_state,
        }
    )
    return payload


def _models_status_payload() -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    total_bytes = 0
    missing: list[str] = []
    for key in MODEL_SPECS:
        path = _locate_model_file(MODEL_SPECS[key].filename)
        size_bytes = 0
        if path is not None:
            with contextlib.suppress(OSError):
                size_bytes = max(0, int(path.stat().st_size))
        ready = path is not None and size_bytes >= 0
        if not ready:
            missing.append(key)
        total_bytes += size_bytes
        details.append(
            {
                "key": key,
                "label": MODEL_DISPLAY_NAMES.get(key, key),
                "ready": ready,
                "size_bytes": size_bytes if ready else None,
                "path": str(path) if path is not None else "",
            }
        )
    return {
        "missing": sorted(missing),
        "models_dir": str(MODEL_DIR),
        "models": details,
        "downloaded_total_bytes": total_bytes,
    }


def _model_retry_label(retry_count: int) -> str:
    return "too many to count" if retry_count > 9999 else str(max(0, retry_count))


MODEL_DOWNLOAD_MAX_RETRIES = 5


def _model_download_error_message(exc: Exception) -> str:
    if isinstance(exc, ModelDownloadError):
        return str(exc)
    return "download failed unexpectedly"


def _model_download_retry_delay_seconds(retry_count: int) -> int:
    return min(30, max(5, retry_count * 5))


def _should_retry_model_download(exc: Exception, retry_count: int) -> bool:
    if retry_count >= MODEL_DOWNLOAD_MAX_RETRIES:
        return False
    if isinstance(exc, ModelDownloadError):
        return exc.retryable
    return True


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
            download_rate_bytes_per_sec=0.0,
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
        download_rate_bytes_per_sec=0.0,
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
        download_rate_bytes_per_sec = 0.0
        if isinstance(started_at, (int, float)) and started_at and total_bytes > 0 and downloaded_bytes > 0:
            elapsed = max(1.0, time.time() - float(started_at))
            remaining = max(0, total_bytes - downloaded_bytes)
            download_rate_bytes_per_sec = max(downloaded_bytes / elapsed, 0.0)
            eta_seconds = int(round(remaining / max(download_rate_bytes_per_sec, 1)))
        _set_model_download_state(
            status="downloading",
            pct=int(update.get("pct") or 0),
            step=pretty,
            current_model=pretty,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            download_rate_bytes_per_sec=download_rate_bytes_per_sec,
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
            error_message = _model_download_error_message(exc)
            if not _should_retry_model_download(exc, retry_count):
                logger.error("model download attempt %s failed permanently", retry_label, exc_info=True)
                _set_model_download_state(
                    status="error",
                    step="download failed",
                    current_model="",
                    download_rate_bytes_per_sec=0.0,
                    eta_seconds=None,
                    retry_count=retry_count,
                    retry_label=retry_label,
                    error=error_message,
                )
                return
            retry_delay = _model_download_retry_delay_seconds(retry_count)
            logger.warning("model download attempt %s failed; retrying", retry_label, exc_info=True)
            _set_model_download_state(
                status="retrying",
                step="waiting to retry",
                current_model="",
                download_rate_bytes_per_sec=0.0,
                eta_seconds=retry_delay,
                retry_count=retry_count,
                retry_label=retry_label,
                error=f"{error_message}. retrying in {retry_delay}s... ({retry_label}/{MODEL_DOWNLOAD_MAX_RETRIES})",
            )
            time.sleep(retry_delay)

    _set_model_download_state(
        status="done",
        pct=100,
        step="models ready",
        current_model="",
        download_rate_bytes_per_sec=0.0,
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
        "multi_stem_export": _validate_multi_stem_export(
            str(task.get("multi_stem_export_snapshot") or _compat_settings_payload().get("multi_stem_export") or "zip")
        ),
        "video_handling": task["video_handling"],
        "delivery": str(task.get("delivery") or "folder"),
        "status": task["status"],
        "stage": task["stage"],
        "pct": task["pct"],
        "eta_seconds": task["eta_seconds"],
        "eta_state": task.get("eta_state"),
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
    _update_task_runtime_view(task, now=time.time())
    eta_seconds = task.get("eta_seconds")
    return int(eta_seconds) if isinstance(eta_seconds, (int, float)) else None


def _record_eta_sample(model_key: str, audio_seconds: float, elapsed_seconds: float) -> None:
    if model_key not in ETA_HISTORY_KEYS or elapsed_seconds <= 0:
        return
    _append_runtime_sample(
        "stage_samples",
        f"model:{model_key}",
        {
            "audio_seconds": float(audio_seconds or 0.0),
            "elapsed_seconds": float(elapsed_seconds),
            "recorded_at": time.time(),
        },
    )


def _stage_progress_fraction(mode: str, stage_text: str, pct: int) -> float | None:
    stage = str(stage_text or "").lower()
    clamped = max(0, min(100, int(pct)))
    stage_model = _stage_model_key(stage)
    if stage_model == "vocals":
        if mode in {"both_separate", "preset_voc_instrum"}:
            span = max(1, BOTH_SEPARATE_VOCALS_END_PCT - MODEL_PROGRESS_START_PCT)
            return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
        if mode == "mel_band_karaoke":
            span = max(1, BG_VOCAL_VOCALS_END_PCT - MODEL_PROGRESS_START_PCT)
            return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
        if mode == "preset_boost_harmonies":
            span = max(1, BOOST_HARMONIES_VOCALS_END_PCT - MODEL_PROGRESS_START_PCT)
            return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
        if mode == "preset_all_stems":
            span = max(1, ALL_STEMS_VOCALS_END_PCT - MODEL_PROGRESS_START_PCT)
            return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
        span = max(1, SINGLE_MODEL_PROGRESS_END_PCT - MODEL_PROGRESS_START_PCT)
        return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
    if stage_model == "instrumental":
        if mode in {"both_separate", "preset_voc_instrum"}:
            span = max(1, BOTH_SEPARATE_INSTRUMENTAL_END_PCT - BOTH_SEPARATE_INSTRUMENTAL_START_PCT)
            return max(0.0, min(1.0, (clamped - BOTH_SEPARATE_INSTRUMENTAL_START_PCT) / span))
        if mode == "preset_all_stems":
            span = max(1, ALL_STEMS_INSTRUMENTAL_END_PCT - ALL_STEMS_INSTRUMENTAL_START_PCT)
            return max(0.0, min(1.0, (clamped - ALL_STEMS_INSTRUMENTAL_START_PCT) / span))
        span = max(1, SINGLE_MODEL_PROGRESS_END_PCT - MODEL_PROGRESS_START_PCT)
        return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
    if stage_model in {
        "deux",
        "guitar",
        "mel_band_karaoke",
        "denoise",
        "bs_roformer_6s",
        "htdemucs_ft_drums",
        "htdemucs_ft_bass",
        "htdemucs_ft_other",
        "htdemucs_6s",
        "drumsep_6s",
        "drumsep_4s",
    }:
        if mode == "htdemucs_ft_other" and stage_model == "guitar":
            span = max(1, OTHER_FILTER_GUITAR_END_PCT - MODEL_PROGRESS_START_PCT)
            return max(0.0, min(1.0, (clamped - MODEL_PROGRESS_START_PCT) / span))
        if mode == "htdemucs_ft_other" and stage_model == "htdemucs_ft_other":
            span = max(1, OTHER_FILTER_OTHER_END_PCT - OTHER_FILTER_OTHER_START_PCT)
            return max(0.0, min(1.0, (clamped - OTHER_FILTER_OTHER_START_PCT) / span))
        if mode == "preset_boost_harmonies" and stage_model == "mel_band_karaoke":
            span = max(1, BOOST_HARMONIES_BACKGROUND_END_PCT - BOOST_HARMONIES_BACKGROUND_START_PCT)
            return max(0.0, min(1.0, (clamped - BOOST_HARMONIES_BACKGROUND_START_PCT) / span))
        if mode == "mel_band_karaoke" and stage_model == "mel_band_karaoke":
            span = max(1, BG_VOCAL_BACKGROUND_END_PCT - BG_VOCAL_BACKGROUND_START_PCT)
            return max(0.0, min(1.0, (clamped - BG_VOCAL_BACKGROUND_START_PCT) / span))
        if mode == "preset_all_stems" and stage_model == "mel_band_karaoke":
            span = max(1, ALL_STEMS_BACKGROUND_END_PCT - ALL_STEMS_BACKGROUND_START_PCT)
            return max(0.0, min(1.0, (clamped - ALL_STEMS_BACKGROUND_START_PCT) / span))
        if mode == "preset_all_stems" and stage_model == "bs_roformer_6s":
            span = max(1, ALL_STEMS_MULTI_END_PCT - ALL_STEMS_MULTI_START_PCT)
            return max(0.0, min(1.0, (clamped - ALL_STEMS_MULTI_START_PCT) / span))
        if mode == "preset_all_stems" and stage_model == "drumsep_6s":
            span = max(1, ALL_STEMS_DRUMS_END_PCT - ALL_STEMS_DRUMS_START_PCT)
            return max(0.0, min(1.0, (clamped - ALL_STEMS_DRUMS_START_PCT) / span))
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
        if task["stop_event"].is_set():
            raise TaskStopped()
        previous_pct = int(task.get("pct") or 0)
        previous_stage = str(task.get("stage") or "")
        now = time.time()
        if task["started_at"] is None:
            task["started_at"] = now
        if not isinstance(task.get("runtime_plan"), dict):
            _refresh_runtime_plan(task, audio_seconds=float(task.get("audio_seconds") or 0.0))
        if task["status"] not in TERMINAL_STATUSES:
            task["status"] = "running"
        next_stage_key = _runtime_stage_key_for_display(stage)
        previous_stage_key = _runtime_stage_key_for_display(previous_stage)
        if next_stage_key != previous_stage_key:
            _complete_runtime_stage(task, previous_stage_key, now=now, record_sample=True)
        if stage != previous_stage or task.get("stage_started_at") is None:
            task["stage_started_at"] = now
            task["eta_finish_at"] = None
            task["eta_stage"] = None
        if next_stage_key:
            current_stage = _runtime_plan_stage(task, next_stage_key)
            if current_stage is not None and current_stage.get("started_at") is None:
                current_stage["started_at"] = now
                current_stage["live_fraction"] = None
        task["stage"] = stage
        live_fraction = _stage_live_fraction_from_legacy_pct(task, stage, pct)
        if next_stage_key:
            current_stage = _runtime_plan_stage(task, next_stage_key)
            if current_stage is not None and live_fraction is not None:
                current_stage["live_fraction"] = live_fraction
        _update_task_runtime_view(task, now=now)
        if task["pct"] != previous_pct or stage != previous_stage:
            task["last_progress_at"] = now
            task["last_progress_pct"] = task["pct"]
            task["last_progress_stage"] = stage
        task["version"] += 1


def _mark_task_done(task_id: str, out_dir: Path, outputs: list[str]) -> None:
    cleanup_snapshot: dict[str, Any] | None = None
    with tasks_lock:
        task = tasks[task_id]
        now = time.time()
        _complete_runtime_stage(task, _runtime_stage_key_for_display(str(task.get("stage") or "")), now=now, record_sample=True)
        audio_seconds = float(task.get("audio_seconds") or 0.0)
        started_at = float(task.get("started_at") or now)
        if started_at > 0:
            _append_runtime_sample(
                "task_samples",
                f"task:{str(task.get('mode') or '')}",
                {
                    "audio_seconds": audio_seconds,
                    "elapsed_seconds": max(0.1, now - started_at),
                    "recorded_at": now,
                },
            )
        task["status"] = "done"
        task["stage"] = "Done"
        task["pct"] = 100
        task["eta_seconds"] = 0
        task["eta_state"] = None
        task["eta_finish_at"] = None
        task["eta_stage"] = None
        task["out_dir"] = str(out_dir)
        task["outputs"] = list(outputs)
        task["error"] = None
        task["guard_error"] = None
        task["finished_at"] = now
        task["version"] += 1
        if bool(task.get("cleared")):
            cleanup_snapshot = dict(task)
    if cleanup_snapshot is not None:
        _forget_task(task_id, cleanup_snapshot)
        return
    with contextlib.suppress(Exception):
        _archive_previous_file(task_id, out_dir, outputs)
    _prune_terminal_tasks()


def _mark_task_error(task_id: str, message: str) -> None:
    cleanup_snapshot: dict[str, Any] | None = None
    with tasks_lock:
        task = tasks[task_id]
        task["status"] = "error"
        task["stage"] = "Error"
        task["pct"] = max(0, int(task.get("pct", 0)))
        task["eta_seconds"] = None
        task["eta_state"] = None
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


def _output_subdir_for_label(mode: str, label: str) -> Path | None:
    if mode == "preset_all_stems" and label in ALL_STEMS_DRUM_OUTPUT_LABELS:
        return Path("drums")
    return None


def _relative_output_name(output_path: Path, root_dir: Path) -> str:
    with contextlib.suppress(Exception):
        return str(output_path.relative_to(root_dir))
    return output_path.name


def _unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem} ({index}){suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{stem}-{int(time.time())}{suffix}")


def _create_multi_stem_archive(task: dict[str, Any], output_paths: list[Path], out_dir: Path) -> Path:
    archive_name = f"{_download_label(task)}.zip"
    archive_path = _unique_output_path(out_dir / archive_name)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for output_path in output_paths:
            zf.write(output_path, arcname=_relative_output_name(output_path, out_dir))
    return archive_path


def _finalize_written_outputs(task_id: str, out_dir: Path, output_paths: list[Path]) -> list[Path]:
    if len(output_paths) <= 1:
        return list(output_paths)
    with tasks_lock:
        task = dict(tasks.get(task_id) or {})
    if not task:
        return list(output_paths)
    if str(task.get("delivery") or "folder") != "folder":
        return list(output_paths)
    if _validate_multi_stem_export(
        str(task.get("multi_stem_export_snapshot") or _compat_settings_payload().get("multi_stem_export") or "zip")
    ) != "zip":
        return list(output_paths)
    archive_path = _create_multi_stem_archive(task, output_paths, out_dir)
    for output_path in output_paths:
        _cleanup_path(output_path)
    return [archive_path]


def _copy_outputs_to_directory(output_paths: list[Path], destination_dir: Path) -> list[Path]:
    copied: list[Path] = []
    _ensure_dir(destination_dir)
    source_root = Path(os.path.commonpath([str(path.parent) for path in output_paths])) if output_paths else destination_dir
    for output_path in output_paths:
        relative_name = _relative_output_name(output_path, source_root)
        target = destination_dir / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target = target.with_name(f"{output_path.stem} ({int(time.time())}){output_path.suffix}")
        shutil.copy2(output_path, target)
        copied.append(target)
    return copied


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
            task["eta_state"] = None
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
            task["eta_state"] = None
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


def _pause_queue_processing() -> None:
    queue_resume_event.clear()


def _resume_queue_processing() -> None:
    queue_resume_event.set()


def _queue_processing_paused() -> bool:
    return not queue_resume_event.is_set()


def _request_task_stop(task_id: str) -> None:
    _pause_queue_processing()
    should_forget = False
    should_prune = False
    should_terminate_runtime = False
    with tasks_lock:
        task = tasks[task_id]
        task["stop_event"].set()
        if task["status"] == "queued":
            task["status"] = "stopped"
            task["stage"] = "Stopped"
            task["eta_seconds"] = None
            task["eta_state"] = None
            task["eta_finish_at"] = None
            task["eta_stage"] = None
            task["finished_at"] = time.time()
            should_forget = bool(task.get("cleared"))
            should_prune = not should_forget
        elif task["status"] == "running":
            task["stage"] = "Stopping"
            task["eta_seconds"] = None
            task["eta_state"] = None
            task["eta_finish_at"] = None
            task["eta_stage"] = None
            should_terminate_runtime = True
        task["version"] += 1
        logger.info(
            "stop requested for %s (status=%s, stage=%s, hard=%s)",
            task_id,
            task["status"],
            task["stage"],
            should_terminate_runtime,
        )
    if should_terminate_runtime:
        _terminate_task_runtime(task_id)
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
        task["eta_state"] = None
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
                task["eta_state"] = None
                task["eta_finish_at"] = None
                task["eta_stage"] = None
                task["finished_at"] = time.time()
            elif task["status"] == "running":
                task["stage"] = "Stopping"
                task["eta_seconds"] = None
                task["eta_state"] = None
                task["eta_finish_at"] = None
                task["eta_stage"] = None
            task["version"] += 1
    _resume_queue_processing()
    return task_ids


def _restart_task_payload(
    task_id: str,
    *,
    stems_raw: str | None = None,
    output_format: str | None = None,
    video_handling: str | None = None,
    output_root: str | None = None,
    output_same_as_input: bool | None = None,
    multi_stem_export: str | None = None,
    prioritize: bool = False,
) -> dict[str, Any]:
    old_task = _require_task(task_id)
    old_status = str(old_task.get("status") or "")
    if old_status in {"queued", "running"}:
        raise AppError(ErrorCode.INVALID_REQUEST, "Task is still processing.")
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
        multi_stem_export=multi_stem_export,
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
        status = str(task.get("status") or "")
        if status not in {"ready", "queued"}:
            raise AppError(ErrorCode.INVALID_REQUEST, "Task has already started.")
        if status == "queued" and float(task.get("pct") or 0) > 0:
            raise AppError(ErrorCode.INVALID_REQUEST, "Task is still processing.")
        task["mode"] = mode
        _refresh_runtime_plan(task, audio_seconds=float(task.get("audio_seconds") or 0.0))
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


def _task_runner_snapshot(task: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task.items() if key != "stop_event"}


def _apply_task_runner_snapshot(task_id: str, snapshot: dict[str, Any]) -> None:
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            return
        stop_event = task["stop_event"]
        cleared = bool(task.get("cleared"))
        for key, value in snapshot.items():
            if key == "stop_event":
                continue
            task[key] = value
        task["stop_event"] = stop_event
        task["cleared"] = cleared
        task["version"] = int(task.get("version") or 0) + 1


def _register_task_runtime(task_id: str, runtime: dict[str, Any]) -> None:
    with task_runtime_lock:
        task_runtimes[task_id] = runtime


def _pop_task_runtime(task_id: str) -> dict[str, Any] | None:
    with task_runtime_lock:
        return task_runtimes.pop(task_id, None)


def _current_task_runtime(task_id: str) -> dict[str, Any] | None:
    with task_runtime_lock:
        runtime = task_runtimes.get(task_id)
        return dict(runtime) if runtime is not None else None


def _terminate_task_runtime(task_id: str) -> bool:
    runtime = _current_task_runtime(task_id)
    if runtime is None:
        logger.info("stop terminate skipped for %s: no live runtime", task_id)
        return False
    process = runtime.get("process")
    if not isinstance(process, subprocess.Popen):
        logger.info("stop terminate skipped for %s: runtime missing process", task_id)
        return False
    if process.poll() is not None:
        logger.info("stop terminate skipped for %s: process already exited", task_id)
        return False
    try:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    except Exception:
        logger.warning("failed to terminate task runtime for %s", task_id, exc_info=True)
        return False
    logger.info("hard stop sent to task runtime %s (pid=%s)", task_id, process.pid)
    return True


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
        "multi_stem_export_snapshot": None,
        "preset_settings_snapshot": None,
        "delivery": delivery,
        "status": "queued" if auto_start else "ready",
        "stage": "Waiting in queue" if auto_start else "Ready",
        "pct": 0,
        "eta_seconds": None,
        "eta_state": None,
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
        "runtime_plan": {"stages": []},
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


def _validate_multi_stem_export(multi_stem_export: str) -> str:
    if multi_stem_export not in MULTI_STEM_EXPORT_CHOICES:
        return "zip"
    return multi_stem_export


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
    multi_stem_export: str | None = None,
) -> None:
    next_output_format = str(output_format or task.get("output_format") or _compat_settings_payload()["output_format"])
    next_video_handling = _validate_video_handling(
        str(video_handling or task.get("video_handling") or _compat_settings_payload()["video_handling"])
    )
    next_multi_stem_export = _validate_multi_stem_export(
        str(multi_stem_export or task.get("multi_stem_export_snapshot") or _compat_settings_payload().get("multi_stem_export") or "zip")
    )
    mode = str(task.get("mode") or "vocals")
    _validate_mode_and_output_format(mode, next_output_format)
    task["output_format"] = next_output_format
    task["video_handling"] = next_video_handling
    task["multi_stem_export_snapshot"] = next_multi_stem_export
    if output_root is not None:
        task["output_root_snapshot"] = str(_ensure_dir(Path(output_root).expanduser()))
    elif not str(task.get("output_root_snapshot") or "").strip():
        task["output_root_snapshot"] = str(_current_output_root())
    if output_same_as_input is not None:
        task["output_same_as_input_snapshot"] = bool(output_same_as_input)
    elif task.get("output_same_as_input_snapshot") is None:
        task["output_same_as_input_snapshot"] = bool(_compat_settings_payload().get("output_same_as_input"))
    task["preset_settings_snapshot"] = _preset_settings_payload_for_mode(mode, _compat_settings_payload())
    _refresh_runtime_plan(task, audio_seconds=float(task.get("audio_seconds") or 0.0))


def _register_task(
    *,
    original_name: str,
    source_path: Path,
    source_dir: str | None,
    mode: str,
    output_format: str,
    video_handling: str,
    multi_stem_export: str | None = None,
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
    _apply_task_start_settings(payload, multi_stem_export=multi_stem_export)
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
        task["eta_state"] = None
        task["eta_finish_at"] = None
        task["eta_stage"] = None
        task["started_at"] = None
        task["stage_started_at"] = None
        task["audio_seconds"] = None
        task["predicted_total_seconds"] = None
        _refresh_runtime_plan(task, audio_seconds=0.0)
        task["last_progress_at"] = time.time()
        task["last_progress_pct"] = 0
        task["last_progress_stage"] = task["stage"]
        task["guard_error"] = None
        task["version"] += 1
    _queue_task(task_id, front=front)
    return task


def _remove_task(task_id: str) -> dict[str, Any]:
    snapshot: dict[str, Any] | None = None
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
        status = str(task.get("status") or "")
        if status == "running":
            raise AppError(ErrorCode.INVALID_REQUEST, "Task is still processing.")
        task["stop_event"].set()
        snapshot = dict(task)
        tasks.pop(task_id, None)
    if snapshot is not None:
        _cleanup_task_runtime(task_id, snapshot)
    return snapshot or {}


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
        "eta_state": public.get("eta_state"),
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
        decoded_path = _decode_audio_to_wav(
            source_path,
            work_dir,
            source_info.channels,
            stop_check=lambda: _stop_check(task_id),
        )
        _stop_check(task_id)
        waveform = _load_waveform(decoded_path)
        mode = task["mode"]
        audio_seconds = waveform.shape[1] / 44100.0 if waveform.shape[1] > 0 else 0.0
        with tasks_lock:
            tasks[task_id]["audio_seconds"] = audio_seconds
            _refresh_runtime_plan(tasks[task_id], audio_seconds=audio_seconds)
            tasks[task_id]["predicted_total_seconds"] = _predict_task_runtime_seconds(mode, audio_seconds)
            _update_task_runtime_view(tasks[task_id], now=time.time())
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
        elif mode in {"both_separate", "preset_voc_instrum"}:
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
            guitar_pred = _run_model_for_spec(
                "guitar",
                guitar_model,
                waveform,
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
            vocals_model = manager.get("vocals")
            karaoke_model = manager.get("mel_band_karaoke")

            vocals_started_at = time.time()
            vocals_pred = _run_model_chunks(
                vocals_model,
                waveform,
                MODEL_SPECS["vocals"].segment,
                MODEL_SPECS["vocals"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running vocals model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, BG_VOCAL_VOCALS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("vocals", audio_seconds, time.time() - vocals_started_at)
            vocals_tensor = vocals_pred[0]

            karaoke_started_at = time.time()
            karaoke_pred = _run_model_for_spec(
                "mel_band_karaoke",
                karaoke_model,
                vocals_tensor,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running background vocal model",
                    _map_fraction(BG_VOCAL_BACKGROUND_START_PCT, BG_VOCAL_BACKGROUND_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("mel_band_karaoke", audio_seconds, time.time() - karaoke_started_at)
            bg_vocal_tensor = (
                karaoke_pred[1]
                if karaoke_pred.shape[0] > 1
                else _residual_output(vocals_tensor, karaoke_pred[0])
            )
            _append_named_output(temp_outputs, work_dir, "bg vocal", bg_vocal_tensor)
        elif mode == "preset_boost_harmonies":
            vocals_model = manager.get("vocals")
            harmony_model = manager.get("mel_band_karaoke")
            preset_settings = _task_boost_harmonies_settings(task)

            vocals_started_at = time.time()
            vocals_pred = _run_model_for_spec(
                "vocals",
                vocals_model,
                waveform,
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
            harmony_pred = _run_model_for_spec(
                "mel_band_karaoke",
                harmony_model,
                vocals_tensor,
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

            _stop_check(task_id)
            _set_task_progress(task_id, "Mixing boost harmonies", 95)
            boost_mix_tensor = _boost_overlay_mix(
                waveform,
                background_vocals_tensor,
                base_song_gain_db=preset_settings["base_song_gain_db"],
                overlay_gain_db=preset_settings["overlay_gain_db"],
            )
            _stop_check(task_id)
            _append_named_output(temp_outputs, work_dir, "boost harmonies", boost_mix_tensor)
        elif mode == "preset_boost_guitar":
            guitar_model = manager.get("guitar")
            preset_settings = _task_boost_guitar_settings(task)

            guitar_started_at = time.time()
            guitar_pred = _run_model_for_spec(
                "guitar",
                guitar_model,
                waveform,
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

            _stop_check(task_id)
            _set_task_progress(task_id, "Mixing boost guitar", 95)
            boost_mix_tensor = _boost_overlay_mix(
                waveform,
                guitar_tensor,
                base_song_gain_db=preset_settings["base_song_gain_db"],
                overlay_gain_db=preset_settings["overlay_gain_db"],
            )
            _stop_check(task_id)
            _append_named_output(temp_outputs, work_dir, "boost guitar", boost_mix_tensor)
        elif mode in {"denoise", "preset_denoise"}:
            denoise_model = manager.get("denoise")
            denoise_started_at = time.time()
            denoise_pred = _run_model_for_spec(
                "denoise",
                denoise_model,
                waveform,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running denoise model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("denoise", audio_seconds, time.time() - denoise_started_at)
            _append_named_output(temp_outputs, work_dir, "denoise", denoise_pred[0])
        elif mode == "bs_roformer_6s":
            bs_6s_model = manager.get("bs_roformer_6s")
            bs_6s_started_at = time.time()
            bs_6s_pred = _run_model_for_spec(
                "bs_roformer_6s",
                bs_6s_model,
                waveform,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running full mix model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("bs_roformer_6s", audio_seconds, time.time() - bs_6s_started_at)
            expected_labels = MODE_OUTPUT_LABELS["bs_roformer_6s"]
            if bs_6s_pred.shape[0] < len(expected_labels):
                raise AppError(ErrorCode.SEPARATION_FAILED, "full mix model returned incomplete output.")
            for label, tensor in zip(expected_labels, bs_6s_pred, strict=False):
                _append_named_output(temp_outputs, work_dir, label, tensor)
        elif mode in {"htdemucs_ft_drums", "htdemucs_ft_bass"}:
            fast_model = manager.get(mode)
            fast_started_at = time.time()
            fast_pred = _run_model_for_spec(
                mode,
                fast_model,
                waveform,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    f"Running {MODEL_DISPLAY_NAMES[mode]} model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample(mode, audio_seconds, time.time() - fast_started_at)
            _append_named_output(
                temp_outputs,
                work_dir,
                MODE_OUTPUT_LABELS[mode][0],
                _extract_target_tensor(mode, fast_model, fast_pred),
            )
        elif mode == "htdemucs_ft_other":
            guitar_model = manager.get("guitar")
            filtered_other_model = manager.get("htdemucs_ft_other")

            guitar_started_at = time.time()
            guitar_pred = _run_model_for_spec(
                "guitar",
                guitar_model,
                waveform,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running guitar model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, OTHER_FILTER_GUITAR_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("guitar", audio_seconds, time.time() - guitar_started_at)
            guitar_tensor = guitar_pred[0]
            non_guitar_waveform = _residual_output(waveform, guitar_tensor)

            filtered_other_started_at = time.time()
            filtered_other_pred = _run_model_for_spec(
                "htdemucs_ft_other",
                filtered_other_model,
                non_guitar_waveform,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    f"Running {MODEL_DISPLAY_NAMES['htdemucs_ft_other']} model",
                    _map_fraction(OTHER_FILTER_OTHER_START_PCT, OTHER_FILTER_OTHER_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("htdemucs_ft_other", audio_seconds, time.time() - filtered_other_started_at)
            _append_named_output(
                temp_outputs,
                work_dir,
                "other",
                _extract_target_tensor("htdemucs_ft_other", filtered_other_model, filtered_other_pred),
            )
        elif mode == "htdemucs_6s":
            htdemucs_6s_model = manager.get("htdemucs_6s")
            htdemucs_6s_started_at = time.time()
            htdemucs_6s_pred = _run_model_for_spec(
                "htdemucs_6s",
                htdemucs_6s_model,
                waveform,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running htdemucs4 6 stem model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("htdemucs_6s", audio_seconds, time.time() - htdemucs_6s_started_at)
            for label, tensor in zip(MODE_OUTPUT_LABELS["htdemucs_6s"], htdemucs_6s_pred, strict=False):
                _append_named_output(temp_outputs, work_dir, label, tensor)
        elif mode == "drumsep_6s":
            drumsep_6s_model = manager.get("drumsep_6s")
            drumsep_6s_started_at = time.time()
            drumsep_6s_pred = _run_model_for_spec(
                "drumsep_6s",
                drumsep_6s_model,
                waveform,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running drumsep 6 stem model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("drumsep_6s", audio_seconds, time.time() - drumsep_6s_started_at)
            for label, tensor in zip(MODE_OUTPUT_LABELS["drumsep_6s"], drumsep_6s_pred, strict=False):
                _append_named_output(temp_outputs, work_dir, label, tensor)
        elif mode == "drumsep_4s":
            drumsep_4s_model = manager.get("drumsep_4s")
            drumsep_4s_started_at = time.time()
            drumsep_4s_pred = _run_model_for_spec(
                "drumsep_4s",
                drumsep_4s_model,
                waveform,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running drum sep 4 stem model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, SINGLE_MODEL_PROGRESS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("drumsep_4s", audio_seconds, time.time() - drumsep_4s_started_at)
            for label, tensor in zip(MODE_OUTPUT_LABELS["drumsep_4s"], drumsep_4s_pred, strict=False):
                _append_named_output(temp_outputs, work_dir, label, tensor)
        elif mode == "preset_all_stems":
            vocals_model = manager.get("vocals")
            instrumental_model = manager.get("instrumental")
            karaoke_model = manager.get("mel_band_karaoke")
            bs_6s_model = manager.get("bs_roformer_6s")
            drumsep_6s_model = manager.get("drumsep_6s")

            vocals_started_at = time.time()
            vocals_pred = _run_model_chunks(
                vocals_model,
                waveform,
                MODEL_SPECS["vocals"].segment,
                MODEL_SPECS["vocals"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running vocals model",
                    _map_fraction(MODEL_PROGRESS_START_PCT, ALL_STEMS_VOCALS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("vocals", audio_seconds, time.time() - vocals_started_at)
            vocals_tensor = vocals_pred[0]

            instrumental_started_at = time.time()
            instrumental_pred = _run_model_chunks(
                instrumental_model,
                waveform,
                MODEL_SPECS["instrumental"].segment,
                MODEL_SPECS["instrumental"].overlap,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running instrumental model",
                    _map_fraction(ALL_STEMS_INSTRUMENTAL_START_PCT, ALL_STEMS_INSTRUMENTAL_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("instrumental", audio_seconds, time.time() - instrumental_started_at)
            instrumental_tensor = instrumental_pred[0]

            background_started_at = time.time()
            background_pred = _run_model_for_spec(
                "mel_band_karaoke",
                karaoke_model,
                vocals_tensor,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running harmony background model",
                    _map_fraction(ALL_STEMS_BACKGROUND_START_PCT, ALL_STEMS_BACKGROUND_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("mel_band_karaoke", audio_seconds, time.time() - background_started_at)
            lead_vocals_tensor = background_pred[0]
            background_vocals_tensor = (
                background_pred[1]
                if background_pred.shape[0] > 1
                else _residual_output(vocals_tensor, background_pred[0])
            )
            _append_named_output(temp_outputs, work_dir, "vocals", lead_vocals_tensor)
            _append_named_output(temp_outputs, work_dir, "background vocals", background_vocals_tensor)

            bs_6s_started_at = time.time()
            bs_6s_pred = _run_model_for_spec(
                "bs_roformer_6s",
                bs_6s_model,
                instrumental_tensor,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running full mix model",
                    _map_fraction(ALL_STEMS_MULTI_START_PCT, ALL_STEMS_MULTI_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("bs_roformer_6s", audio_seconds, time.time() - bs_6s_started_at)
            bs_outputs = {
                label: tensor
                for label, tensor in zip(MODE_OUTPUT_LABELS["bs_roformer_6s"], bs_6s_pred, strict=False)
            }
            for label in ("bass", "drums", "other", "guitar", "piano"):
                tensor = bs_outputs.get(label)
                if tensor is not None:
                    _append_named_output(temp_outputs, work_dir, label, tensor)

            drums_tensor = bs_outputs.get("drums")
            if drums_tensor is None:
                raise AppError(ErrorCode.SEPARATION_FAILED, "6-stem model did not return a drums stem.")

            drumsep_6s_started_at = time.time()
            drumsep_6s_pred = _run_model_for_spec(
                "drumsep_6s",
                drumsep_6s_model,
                drums_tensor,
                progress_cb=lambda frac: _set_task_progress(
                    task_id,
                    "Running drumsep 6 stem model",
                    _map_fraction(ALL_STEMS_DRUMS_START_PCT, ALL_STEMS_DRUMS_END_PCT, frac),
                ),
                stop_check=lambda: _stop_check(task_id),
            )
            _record_eta_sample("drumsep_6s", audio_seconds, time.time() - drumsep_6s_started_at)
            for label, tensor in zip(MODE_OUTPUT_LABELS["drumsep_6s"], drumsep_6s_pred, strict=False):
                _append_named_output(temp_outputs, work_dir, label, tensor)
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
            output_subdir = _output_subdir_for_label(mode, label)
            output_parent = output_dir / output_subdir if output_subdir is not None else output_dir
            if export_video:
                video_suffix, _audio_args = _resolve_video_output(source_info)
                final_path = output_parent / f"{_safe_stem(task['original_name'])} - {export_label}{video_suffix}"
                exported = _export_video_stem(
                    temp_path,
                    source_path,
                    final_path,
                    source_info,
                    stop_check=lambda: _stop_check(task_id),
                )
            else:
                assert export_plan is not None
                final_path = output_parent / f"{_safe_stem(task['original_name'])} - {export_label}{export_plan.suffix}"
                exported = _export_stem(
                    temp_path,
                    source_path,
                    final_path,
                    export_plan,
                    source_info.has_cover,
                    stop_check=lambda: _stop_check(task_id),
                )
            written_outputs.append(exported)
            exported_files.append(_relative_output_name(exported, output_dir))
            _set_task_progress(
                task_id,
                f"Exporting {label}",
                _map_fraction(EXPORT_PROGRESS_START_PCT, EXPORT_PROGRESS_END_PCT, index / max(1, total_exports)),
            )

        finalized_outputs = _finalize_written_outputs(task_id, output_dir, written_outputs)
        _mark_task_done(task_id, output_dir, [_relative_output_name(output_path, output_dir) for output_path in finalized_outputs])
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


def _emit_task_runner_event(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _task_runner_main(payload_path: Path) -> None:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("invalid task payload")
    task_id = str(payload.get("id") or "").strip()
    if not task_id:
        raise SystemExit("missing task id")
    payload["stop_event"] = threading.Event()
    with tasks_lock:
        tasks.clear()
        tasks[task_id] = payload

    original_set_task_progress = _set_task_progress
    original_mark_task_done = _mark_task_done
    original_mark_task_error = _mark_task_error
    original_mark_task_stopped = _mark_task_stopped

    def _patched_set_task_progress(task_id: str, stage: str, pct: int) -> None:
        original_set_task_progress(task_id, stage, pct)
        _emit_task_runner_event({"kind": "snapshot", "task": _task_runner_snapshot(tasks[task_id])})

    def _patched_mark_task_done(task_id: str, out_dir: Path, outputs: list[str]) -> None:
        _emit_task_runner_event({"kind": "done", "out_dir": str(out_dir), "outputs": list(outputs)})

    def _patched_mark_task_error(task_id: str, message: str) -> None:
        _emit_task_runner_event({"kind": "error", "message": message})

    def _patched_mark_task_stopped(task_id: str) -> None:
        _emit_task_runner_event({"kind": "stopped"})

    globals()["_set_task_progress"] = _patched_set_task_progress
    globals()["_mark_task_done"] = _patched_mark_task_done
    globals()["_mark_task_error"] = _patched_mark_task_error
    globals()["_mark_task_stopped"] = _patched_mark_task_stopped
    try:
        _process_task(task_id)
    finally:
        globals()["_set_task_progress"] = original_set_task_progress
        globals()["_mark_task_done"] = original_mark_task_done
        globals()["_mark_task_error"] = original_mark_task_error
        globals()["_mark_task_stopped"] = original_mark_task_stopped


def _write_task_runner_payload(task_id: str) -> Path:
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
        payload = _task_runner_snapshot(task)
    handle, raw_path = tempfile.mkstemp(prefix=f"taskrun_{task_id[:8]}_", suffix=".json", dir=str(WORK_DIR))
    path = Path(raw_path)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream)
    except Exception:
        _cleanup_path(path)
        raise
    return path


def _drain_task_runner_stderr(task_id: str, stream: Any) -> None:
    try:
        for line in iter(stream.readline, ""):
            text = str(line or "").rstrip()
            if text:
                logger.info("task runner %s: %s", task_id, text)
    except Exception:
        logger.debug("stderr drain failed for task %s", task_id, exc_info=True)


def _run_task_in_subprocess(task_id: str) -> None:
    with tasks_lock:
        task = tasks.get(task_id)
        if task is None:
            raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
        if task["stop_event"].is_set():
            _mark_task_stopped(task_id)
            return
    payload_path = _write_task_runner_payload(task_id)
    if getattr(sys, "frozen", False):
        command = [sys.executable, "--task-runner-input", str(payload_path)]
    else:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--task-runner-input", str(payload_path)]
    env = dict(os.environ)
    env["STEMSPLAT_DISABLE_BACKGROUND_THREADS"] = "1"
    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "bufsize": 1,
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kwargs)
    _register_task_runtime(task_id, {"process": process, "payload_path": str(payload_path)})
    with tasks_lock:
        live_task = tasks.get(task_id)
        if live_task is not None and live_task["stop_event"].is_set():
            _terminate_task_runtime(task_id)

    stderr_thread: threading.Thread | None = None
    if process.stderr is not None:
        stderr_thread = threading.Thread(
            target=_drain_task_runner_stderr,
            args=(task_id, process.stderr),
            daemon=True,
        )
        stderr_thread.start()

    terminal_event: tuple[str, dict[str, Any]] | None = None
    try:
        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                raw = str(line or "").strip()
                if not raw:
                    continue
                try:
                    message = json.loads(raw)
                except Exception:
                    logger.warning("task runner %s emitted invalid JSON: %s", task_id, raw)
                    continue
                kind = str(message.get("kind") or "")
                if kind == "snapshot":
                    snapshot = message.get("task")
                    if isinstance(snapshot, dict):
                        _apply_task_runner_snapshot(task_id, snapshot)
                    continue
                if kind in {"done", "error", "stopped"}:
                    terminal_event = (kind, message)
        return_code = process.wait()
    finally:
        _pop_task_runtime(task_id)
        if process.stdout is not None:
            with contextlib.suppress(Exception):
                process.stdout.close()
        if process.stderr is not None:
            with contextlib.suppress(Exception):
                process.stderr.close()
        if stderr_thread is not None:
            stderr_thread.join(timeout=1.0)
        _cleanup_path(payload_path)

    if terminal_event is not None:
        kind, message = terminal_event
        if kind == "done":
            out_dir = Path(str(message.get("out_dir") or ""))
            outputs = [str(item) for item in list(message.get("outputs") or []) if str(item).strip()]
            _mark_task_done(task_id, out_dir, outputs)
            return
        if kind == "error":
            _mark_task_error(task_id, str(message.get("message") or f"{ErrorCode.SEPARATION_FAILED}: task failed"))
            return
        _mark_task_stopped(task_id)
        return

    with tasks_lock:
        task = tasks.get(task_id)
        stop_requested = bool(task and task["stop_event"].is_set())
    if stop_requested:
        _mark_task_stopped(task_id)
        return
    _mark_task_error(
        task_id,
        f"{ErrorCode.SEPARATION_FAILED}: task worker exited unexpectedly ({return_code}).",
    )


def _next_task_id_for_worker() -> str:
    while True:
        queue_resume_event.wait()
        task_id = task_queue.get()
        if not _queue_processing_paused():
            return task_id
        _queue_task(task_id, front=True)
        task_queue.task_done()


def _task_worker() -> None:
    while True:
        task_id = _next_task_id_for_worker()
        try:
            _run_task_in_subprocess(task_id)
        except AppError as exc:
            _mark_task_error(task_id, f"{exc.code}: {exc.message}")
        except Exception as exc:  # pragma: no cover - safety net for worker orchestration
            logger.exception("task worker failed for %s", task_id)
            _mark_task_error(task_id, f"{ErrorCode.SEPARATION_FAILED}: {exc}")
        finally:
            task_queue.task_done()


def _background_threads_enabled() -> bool:
    return str(os.environ.get("STEMSPLAT_DISABLE_BACKGROUND_THREADS") or "").strip() not in {"1", "true", "yes"}


if _background_threads_enabled():
    threading.Thread(target=_task_worker, daemon=True).start()
    threading.Thread(target=_watchdog_loop, daemon=True).start()


@app.on_event("startup")
async def _startup_cleanup() -> None:
    _close_installer_ui()
    _cleanup_old_runtime_entries(WORK_DIR)
    _cleanup_old_runtime_entries(UPLOAD_DIR)
    _cleanup_old_runtime_entries(INTERMEDIATE_CACHE_DIR, INTERMEDIATE_CACHE_RETENTION_SECONDS)
    _prune_previous_files()
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


def _open_terminal_app() -> None:
    if sys.platform == "darwin":
        script_lines = [
            'tell application "Terminal"',
            "activate",
            'do script ""',
            "end tell",
        ]
        subprocess.run(
            ["osascript", *sum((["-e", line] for line in script_lines), [])],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return
    if os.name == "nt":
        subprocess.Popen(["cmd.exe"])
        return
    terminal = shutil.which("x-terminal-emulator") or shutil.which("gnome-terminal") or shutil.which("konsole")
    if terminal:
        subprocess.Popen([terminal])
        return
    raise AppError(ErrorCode.INVALID_REQUEST, "Could not open a terminal on this system.")


@app.get("/api/models_status")
async def models_status() -> dict[str, Any]:
    return _models_status_payload()


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


@app.post("/api/open_terminal")
async def open_terminal(request: Request) -> dict[str, str]:
    _require_local_request(request, "LAN clients cannot control the host machine.")
    try:
        _open_terminal_app()
    except AppError as exc:
        raise exc.to_http(500) from exc
    except Exception as exc:
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not open terminal: {exc}").to_http(500) from exc
    return {"status": "opened"}


@app.post("/api/import_paths")
async def import_paths(request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid import payload.").to_http()
    raw_paths = body.get("paths")
    raw_stems = body.get("stems")
    output_format = str(body.get("output_format") or _compat_settings_payload()["output_format"])
    multi_stem_export = _validate_multi_stem_export(
        str(body.get("multi_stem_export") or _compat_settings_payload().get("multi_stem_export") or "zip")
    )
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
            multi_stem_export=multi_stem_export,
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
    multi_stem_export: str = Form("zip"),
    video_handling: str = Form("audio_only"),
    source_dir: str | None = Form(None),
):
    try:
        _validate_mode_and_output_format(mode, output_format)
        multi_stem_export = _validate_multi_stem_export(multi_stem_export)
        video_handling = _validate_video_handling(video_handling)
        original_name, source_path = await _store_uploaded_file(file)
        payload = _register_task(
            original_name=original_name,
            source_path=source_path,
            source_dir=source_dir,
            mode=mode,
            output_format=output_format,
            video_handling=video_handling,
            multi_stem_export=multi_stem_export,
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
    multi_stem_export = body.get("multi_stem_export")
    video_handling = body.get("video_handling")
    output_root = body.get("output_root")
    prioritize = True if "prioritize" not in body else bool(body.get("prioritize"))
    output_same_as_input = body.get("output_same_as_input")
    try:
        payload = _restart_task_payload(
            task_id,
            stems_raw=stems_raw if isinstance(stems_raw, str) and stems_raw.strip() else None,
            output_format=str(output_format) if isinstance(output_format, str) and output_format.strip() else None,
            video_handling=str(video_handling) if isinstance(video_handling, str) and video_handling.strip() else None,
            output_root=str(output_root) if isinstance(output_root, str) and output_root.strip() else None,
            output_same_as_input=bool(output_same_as_input) if "output_same_as_input" in body else None,
            multi_stem_export=_validate_multi_stem_export(str(multi_stem_export))
            if isinstance(multi_stem_export, str) and multi_stem_export.strip()
            else None,
            prioritize=prioritize,
        )
    except AppError as exc:
        raise exc.to_http(404 if exc.message == "file doesn't exist" else 400)
    _resume_queue_processing()
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
            zf.write(output, arcname=_relative_output_name(output, _out_path))
    return FileResponse(
        archive_path,
        filename=archive_name,
        background=BackgroundTask(lambda: _cleanup_path(archive_path)),
    )


@app.get("/api/history")
async def list_previous_files() -> dict[str, Any]:
    entries = _prune_previous_files()
    entries.sort(key=lambda entry: float(entry.get("finished_at") or 0.0), reverse=True)
    return {
        "items": [_public_previous_file(entry) for entry in entries],
        "storage": _history_storage_payload(entries),
    }


@app.get("/api/history/{entry_id}/artwork")
async def previous_file_artwork(entry_id: str):
    entry = _require_previous_file(entry_id)
    artwork_path = Path(str(entry.get("artwork_path") or "")).expanduser()
    if not artwork_path.exists() or not artwork_path.is_file():
        raise HTTPException(status_code=404, detail="artwork not found")
    media_type = "image/png" if artwork_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(
        artwork_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/api/history/{entry_id}/reveal")
async def reveal_previous_file(entry_id: str, request: Request):
    _require_local_request(request, "LAN clients cannot control the host machine.")
    entry = _require_previous_file(entry_id)
    try:
        out_path, existing_outputs = _previous_file_output_paths(entry)
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


@app.get("/api/history/{entry_id}/download")
async def download_previous_file(entry_id: str):
    entry = _require_previous_file(entry_id)
    try:
        _out_path, existing_outputs = _previous_file_output_paths(entry)
    except AppError as exc:
        status = 404 if exc.message == "file doesn't exist" else 409
        raise exc.to_http(status) from exc
    if not existing_outputs:
        raise AppError(ErrorCode.INVALID_REQUEST, "file doesn't exist").to_http(404)
    if len(existing_outputs) == 1:
        output = existing_outputs[0]
        return FileResponse(output, filename=output.name)
    archive_dir = _ensure_dir(RUNTIME_DIR / "downloads")
    archive_name = f"{_safe_stem(str(entry.get('original_name') or 'stems'))}.zip"
    archive_path = archive_dir / f"history_{entry_id}_{archive_name}"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for output in existing_outputs:
            zf.write(output, arcname=_relative_output_name(output, _out_path))
    return FileResponse(
        archive_path,
        filename=archive_name,
        background=BackgroundTask(lambda: _cleanup_path(archive_path)),
    )


@app.post("/api/history/{entry_id}/copy")
async def copy_previous_file(entry_id: str, request: Request) -> dict[str, Any]:
    _require_local_request(request, "LAN clients cannot save files from the host machine.")
    body = await request.json()
    if not isinstance(body, dict):
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid copy payload.").to_http()
    destination = str(body.get("output_root") or "").strip()
    if not destination:
        raise AppError(ErrorCode.INVALID_REQUEST, "Missing destination folder.").to_http()
    destination_dir = _ensure_dir(Path(destination).expanduser())
    entry = _require_previous_file(entry_id)
    try:
        _out_path, existing_outputs = _previous_file_output_paths(entry)
    except AppError as exc:
        status = 404 if exc.message == "file doesn't exist" else 409
        raise exc.to_http(status) from exc
    try:
        copied = _copy_outputs_to_directory(existing_outputs, destination_dir)
    except Exception as exc:
        raise AppError(ErrorCode.INVALID_REQUEST, f"Could not save files: {exc}").to_http(500) from exc
    return {
        "status": "saved",
        "output_root": str(destination_dir),
        "outputs": [_relative_output_name(path, destination_dir) for path in copied],
    }


@app.post("/api/history/{entry_id}/reuse")
async def reuse_previous_file(entry_id: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    if not isinstance(body, dict):
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid reuse payload.").to_http()
    stems_raw = body.get("stems")
    if not isinstance(stems_raw, str) or not stems_raw.strip():
        raise AppError(ErrorCode.INVALID_REQUEST, "Invalid stem selection.").to_http()
    output_format = str(body.get("output_format") or _compat_settings_payload()["output_format"])
    multi_stem_export = _validate_multi_stem_export(
        str(body.get("multi_stem_export") or _compat_settings_payload().get("multi_stem_export") or "zip")
    )
    video_handling = _validate_video_handling(str(body.get("video_handling") or _compat_settings_payload()["video_handling"]))
    mode = _stems_to_mode(stems_raw)
    _validate_mode_and_output_format(mode, output_format)
    entry = _require_previous_file(entry_id)
    source_path = Path(str(entry.get("source_path") or "")).expanduser()
    if not source_path.exists() or not source_path.is_file():
        raise AppError(ErrorCode.INVALID_REQUEST, "file doesn't exist").to_http(404)
    payload = _register_task(
        original_name=str(entry.get("original_name") or source_path.name),
        source_path=source_path,
        source_dir=None,
        mode=mode,
        output_format=output_format,
        video_handling=video_handling,
        multi_stem_export=multi_stem_export,
        delivery="browser_download" if _is_remote_client(request) else "folder",
        auto_start=False,
    )
    return _compat_public_task(payload)


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
    if "multi_stem_export" in body:
        patch["multi_stem_export"] = _validate_multi_stem_export(str(body.get("multi_stem_export") or "zip"))
    output_root = body.get("output_root")
    if isinstance(output_root, str):
        try:
            resolved = _ensure_dir(Path(output_root).expanduser())
        except Exception as exc:
            raise AppError(ErrorCode.INVALID_REQUEST, f"Invalid output folder: {exc}").to_http() from exc
        patch["output_root"] = str(resolved)
    if "output_same_as_input" in body:
        patch["output_same_as_input"] = bool(body.get("output_same_as_input"))
    if "previous_files_retention" in body:
        retention_value = str(body.get("previous_files_retention") or "")
        if retention_value not in PREVIOUS_FILES_RETENTION_CHOICES:
            raise AppError(ErrorCode.INVALID_REQUEST, "Invalid previous files retention.").to_http()
        patch["previous_files_retention"] = retention_value
    if "previous_files_limit_gb" in body:
        patch["previous_files_limit_gb"] = _coerce_storage_gb(
            body.get("previous_files_limit_gb"),
            PREVIOUS_FILES_LIMIT_GB_DEFAULT,
            minimum=PREVIOUS_FILES_LIMIT_GB_MIN,
            maximum=PREVIOUS_FILES_LIMIT_GB_MAX,
        )
    if "previous_files_warn_gb" in body:
        patch["previous_files_warn_gb"] = _coerce_storage_gb(
            body.get("previous_files_warn_gb"),
            PREVIOUS_FILES_WARN_GB_DEFAULT,
            minimum=PREVIOUS_FILES_WARN_GB_MIN,
            maximum=PREVIOUS_FILES_WARN_GB_MAX,
        )
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
    if {"previous_files_retention", "previous_files_limit_gb", "previous_files_warn_gb"} & set(patch):
        _prune_previous_files()
    return _settings_response_payload(request)


@app.post("/upload")
async def compat_upload(
    request: Request,
    file: UploadFile = File(...),
    stems: str = Form("vocals"),
    output_format: str | None = Form(None),
    multi_stem_export: str | None = Form(None),
    video_handling: str | None = Form(None),
    source_dir: str | None = Form(None),
):
    try:
        mode = _stems_to_mode(stems)
        resolved_output_format = output_format or _compat_settings_payload()["output_format"]
        resolved_multi_stem_export = _validate_multi_stem_export(
            str(multi_stem_export or _compat_settings_payload().get("multi_stem_export") or "zip")
        )
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
            multi_stem_export=resolved_multi_stem_export,
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
    multi_stem_export = body.get("multi_stem_export")
    video_handling = body.get("video_handling")
    output_root = body.get("output_root")
    output_same_as_input = body.get("output_same_as_input")
    remote_client = _is_remote_client(request)
    request_host = _request_client_host(request)
    running_on_other_device = False
    try:
        with tasks_lock:
            task = tasks.get(task_id)
            if task is None:
                raise AppError(ErrorCode.TASK_NOT_FOUND, "Invalid task id")
            if remote_client:
                for other_id, other in tasks.items():
                    if other_id == task_id:
                        continue
                    if str(other.get("status") or "") != "running":
                        continue
                    other_host = str(other.get("queued_by_host") or "").strip()
                    if not other_host or other_host != request_host:
                        running_on_other_device = True
                        break
            if str(task.get("status") or "") == "ready":
                _apply_task_start_settings(
                    task,
                    output_format=str(output_format) if isinstance(output_format, str) and output_format.strip() else None,
                    video_handling=str(video_handling) if isinstance(video_handling, str) and video_handling.strip() else None,
                    output_root=str(output_root) if isinstance(output_root, str) and output_root.strip() else None,
                    output_same_as_input=bool(output_same_as_input) if "output_same_as_input" in body else None,
                    multi_stem_export=_validate_multi_stem_export(str(multi_stem_export))
                    if isinstance(multi_stem_export, str) and multi_stem_export.strip()
                    else None,
                )
            if request_host:
                task["queued_by_host"] = request_host
        task = _enqueue_task(task_id)
        _resume_queue_processing()
        payload = _compat_public_task(task)
        if running_on_other_device:
            payload["message"] = "another device is currently processing. this song was added to the queue."
        return payload
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
                    _update_task_runtime_view(task, now=now)
                    snapshot = _compat_public_task(task)
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
    logger.info("received /stop request for %s", task_id)
    _request_task_stop(task_id)
    return _compat_public_task(_require_task(task_id))


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    try:
        removed = _remove_task(task_id)
    except AppError as exc:
        status = 404 if exc.code == ErrorCode.TASK_NOT_FOUND else 400
        raise exc.to_http(status)
    return {"status": "removed", "id": task_id, "name": str(removed.get("original_name") or "")}


@app.post("/remove/{task_id}")
async def compat_remove(task_id: str):
    return await delete_task(task_id)


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
    prioritize = True if "prioritize" not in body else bool(body.get("prioritize"))
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
    _resume_queue_processing()
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
    parser.add_argument("--port", type=int, default=DEFAULT_APP_PORT)
    parser.add_argument("--task-runner-input", default="")
    args = parser.parse_args(argv)

    if args.task_runner_input:
        _task_runner_main(Path(args.task_runner_input).expanduser())
        return

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
                <strong>bg vocal</strong>
              </div>
            </label>
            <label class="mode-card" data-mode="bs_roformer_6s">
              <input type="radio" name="split-mode" value="bs_roformer_6s">
              <div class="checkbox"></div>
              <div>
                <strong>full mix</strong>
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
      mel_band_karaoke: 'bg vocal',
      bs_roformer_6s: 'full mix',
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
      if (mode === 'mel_band_karaoke') {
        return ['vocals', 'mel_band_karaoke'].filter((name) => missingModels.includes(name));
      }
      if (mode === 'bs_roformer_6s') return missingModels.includes('bs_roformer_6s') ? ['bs_roformer_6s'] : [];
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
        // Keep ETA calculation in place, but suppress it from the visible UI.
        const progressSide = `${pct}%`;
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
