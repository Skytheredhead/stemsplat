from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn


def config_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: config_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [config_namespace(item) for item in value]
    return value


def prefer_target_instrument(config: SimpleNamespace) -> list[str]:
    target = getattr(config.training, "target_instrument", None)
    if target:
        return [str(target)]
    return [str(item) for item in getattr(config.training, "instruments", [])]


def _windowing_array(window_size: int, fade_size: int) -> torch.Tensor:
    if fade_size <= 0:
        return torch.ones(window_size)
    fade_in = torch.linspace(0.0, 1.0, fade_size)
    fade_out = torch.linspace(1.0, 0.0, fade_size)
    window = torch.ones(window_size)
    window[:fade_size] = fade_in
    window[-fade_size:] = fade_out
    return window


class STFT:
    def __init__(self, config: SimpleNamespace) -> None:
        self.n_fft = int(config.n_fft)
        self.hop_length = int(config.hop_length)
        self.dim_f = int(config.dim_f)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        channels, frames = x.shape[-2:]
        reshaped = x.reshape([-1, frames])
        spec = torch.stft(
            reshaped,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = spec.permute([0, 3, 1, 2])
        spec = spec.reshape([*batch_dims, channels, 2, -1, spec.shape[-1]])
        spec = spec.reshape([*batch_dims, channels * 2, -1, spec.shape[-1]])
        return spec[..., : self.dim_f, :]

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        channels, freqs, frames = x.shape[-3:]
        full_freqs = self.n_fft // 2 + 1
        pad = torch.zeros([*batch_dims, channels, full_freqs - freqs, frames], device=x.device)
        padded = torch.cat([x, pad], dim=-2)
        padded = padded.reshape([*batch_dims, channels // 2, 2, full_freqs, frames]).reshape([-1, 2, full_freqs, frames])
        padded = padded.permute([0, 2, 3, 1])
        complex_spec = padded[..., 0] + (padded[..., 1] * 1j)
        signal = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
        )
        return signal.reshape([*batch_dims, 2, -1])


def get_norm(norm_type: str):
    def norm(channels: int):
        if norm_type == "BatchNorm":
            return nn.BatchNorm2d(channels)
        if norm_type == "InstanceNorm":
            return nn.InstanceNorm2d(channels, affine=True)
        if "GroupNorm" in norm_type:
            groups = int(norm_type.replace("GroupNorm", ""))
            return nn.GroupNorm(num_groups=groups, num_channels=channels)
        return nn.Identity()

    return norm


def get_act(act_type: str) -> nn.Module:
    if act_type == "gelu":
        return nn.GELU()
    if act_type == "relu":
        return nn.ReLU()
    if act_type.startswith("elu"):
        return nn.ELU(float(act_type.replace("elu", "")))
    raise ValueError(f"Unsupported activation: {act_type}")


class Upscale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: list[int], norm, act: nn.Module) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_channels),
            act,
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=scale,
                stride=scale,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downscale(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: list[int], norm, act: nn.Module) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_channels),
            act,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=scale,
                stride=scale,
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TFC_TDF(nn.Module):
    def __init__(self, in_channels: int, channels: int, blocks: int, freqs: int, bottleneck: int, norm, act: nn.Module) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        current_in = in_channels
        for _ in range(blocks):
            block = nn.Module()
            block.tfc1 = nn.Sequential(
                norm(current_in),
                act,
                nn.Conv2d(current_in, channels, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(channels),
                act,
                nn.Linear(freqs, freqs // bottleneck, bias=False),
                norm(channels),
                act,
                nn.Linear(freqs // bottleneck, freqs, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(channels),
                act,
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(current_in, channels, 1, 1, 0, bias=False)
            self.blocks.append(block)
            current_in = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            shortcut = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + shortcut
        return x


class TFC_TDF_net(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config

        norm = get_norm(str(config.model.norm))
        act = get_act(str(config.model.act))
        self.num_target_instruments = len(prefer_target_instrument(config))
        self.num_subbands = int(config.model.num_subbands)

        dim_c = self.num_subbands * int(config.audio.num_channels) * 2
        num_scales = int(config.model.num_scales)
        scale = list(config.model.scale)
        num_blocks = int(config.model.num_blocks_per_scale)
        channels = int(config.model.num_channels)
        growth = int(config.model.growth)
        bottleneck = int(config.model.bottleneck_factor)
        freqs = int(config.audio.dim_f) // self.num_subbands

        self.first_conv = nn.Conv2d(dim_c, channels, 1, 1, 0, bias=False)
        self.encoder_blocks = nn.ModuleList()
        current_channels = channels
        current_freqs = freqs
        for _ in range(num_scales):
            block = nn.Module()
            block.tfc_tdf = TFC_TDF(current_channels, current_channels, num_blocks, current_freqs, bottleneck, norm, act)
            block.downscale = Downscale(current_channels, current_channels + growth, scale, norm, act)
            current_freqs //= scale[1]
            current_channels += growth
            self.encoder_blocks.append(block)

        self.bottleneck_block = TFC_TDF(current_channels, current_channels, num_blocks, current_freqs, bottleneck, norm, act)

        self.decoder_blocks = nn.ModuleList()
        for _ in range(num_scales):
            block = nn.Module()
            block.upscale = Upscale(current_channels, current_channels - growth, scale, norm, act)
            current_freqs *= scale[1]
            current_channels -= growth
            block.tfc_tdf = TFC_TDF(2 * current_channels, current_channels, num_blocks, current_freqs, bottleneck, norm, act)
            self.decoder_blocks.append(block)

        self.final_conv = nn.Sequential(
            nn.Conv2d(current_channels + dim_c, current_channels, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(current_channels, self.num_target_instruments * dim_c, 1, 1, 0, bias=False),
        )

        self.stft = STFT(config.audio)

    def cac2cws(self, x: torch.Tensor) -> torch.Tensor:
        subbands = self.num_subbands
        batch, channels, freqs, frames = x.shape
        x = x.reshape(batch, channels, subbands, freqs // subbands, frames)
        return x.reshape(batch, channels * subbands, freqs // subbands, frames)

    def cws2cac(self, x: torch.Tensor) -> torch.Tensor:
        subbands = self.num_subbands
        batch, channels, freqs, frames = x.shape
        x = x.reshape(batch, channels // subbands, subbands, freqs, frames)
        return x.reshape(batch, channels // subbands, freqs * subbands, frames)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stft(x)
        mix = x = self.cac2cws(x)
        first_conv_out = x = self.first_conv(x)
        x = x.transpose(-1, -2)

        encoder_outputs: list[torch.Tensor] = []
        for block in self.encoder_blocks:
            x = block.tfc_tdf(x)
            encoder_outputs.append(x)
            x = block.downscale(x)

        x = self.bottleneck_block(x)

        for block in self.decoder_blocks:
            x = block.upscale(x)
            x = torch.cat([x, encoder_outputs.pop()], dim=1)
            x = block.tfc_tdf(x)

        x = x.transpose(-1, -2)
        x = x * first_conv_out
        x = self.final_conv(torch.cat([mix, x], dim=1))
        x = self.cws2cac(x)

        batch, channels, freqs, frames = x.shape
        x = x.reshape(batch, self.num_target_instruments, -1, freqs, frames)
        return self.stft.inverse(x)


def load_not_compatible_weights(model: nn.Module, state: dict[str, Any]) -> None:
    payload = state
    if "state" in payload:
        payload = payload["state"]
    if "state_dict" in payload:
        payload = payload["state_dict"]
    if "model_state_dict" in payload:
        payload = payload["model_state_dict"]

    new_state = model.state_dict()
    for key, target_tensor in list(new_state.items()):
        source_tensor = payload.get(key)
        if source_tensor is None:
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            new_state[key] = source_tensor
            continue
        if len(source_tensor.shape) != len(target_tensor.shape):
            continue
        patched = torch.zeros_like(target_tensor)
        slices = tuple(slice(0, min(int(source_tensor.shape[idx]), int(target_tensor.shape[idx]))) for idx in range(len(target_tensor.shape)))
        patched[slices] = source_tensor[slices]
        new_state[key] = patched
    model.load_state_dict(new_state, strict=False)


def demix_mdx23c(
    config: SimpleNamespace,
    model: nn.Module,
    mix: torch.Tensor,
    device: torch.device,
    progress_cb,
    stop_check,
) -> torch.Tensor:
    if not isinstance(mix, torch.Tensor):
        mix = torch.tensor(mix, dtype=torch.float32)

    chunk_size = int(getattr(getattr(config, "inference", SimpleNamespace()), "chunk_size", getattr(config.audio, "chunk_size")))
    num_overlap = max(1, int(getattr(config.inference, "num_overlap", 1)))
    fade_size = max(1, chunk_size // 10)
    step = max(1, chunk_size // num_overlap)
    border = max(0, chunk_size - step)
    mix_cpu = mix.detach().to(dtype=torch.float32, device="cpu")
    length_init = int(mix_cpu.shape[-1])
    if length_init > 2 * border and border > 0:
        mix_cpu = nn.functional.pad(mix_cpu, (border, border), mode="reflect")

    total_length = int(mix_cpu.shape[-1])
    instruments = prefer_target_instrument(config)
    result = torch.zeros((len(instruments), *mix_cpu.shape), dtype=torch.float32)
    counter = torch.zeros((len(instruments), *mix_cpu.shape), dtype=torch.float32)
    window = _windowing_array(chunk_size, fade_size).reshape(1, 1, -1)

    starts = list(range(0, total_length, step))
    if not starts:
        starts = [0]

    model = model.to(device).eval()
    progress_cb(0.0)
    with torch.no_grad():
        for index, start in enumerate(starts, start=1):
            stop_check()
            chunk = mix_cpu[:, start : start + chunk_size].to(device)
            chunk_len = int(chunk.shape[-1])
            pad_mode = "reflect" if chunk_len > chunk_size // 2 else "constant"
            if chunk_len < chunk_size:
                chunk = nn.functional.pad(chunk, (0, chunk_size - chunk_len), mode=pad_mode, value=0.0)
            predicted = model(chunk.unsqueeze(0))[0].detach().cpu()
            blend_window = window.clone()
            if start == 0:
                blend_window[..., :fade_size] = 1.0
            if start + chunk_size >= total_length:
                blend_window[..., -fade_size:] = 1.0
            result[..., start : start + chunk_len] += predicted[..., :chunk_len] * blend_window[..., :chunk_len]
            counter[..., start : start + chunk_len] += blend_window[..., :chunk_len]
            progress_cb(index / max(1, len(starts)))

    estimated = result / counter.clamp_min(1e-6)
    estimated = torch.nan_to_num(estimated, nan=0.0, posinf=0.0, neginf=0.0)
    if length_init > 2 * border and border > 0:
        estimated = estimated[..., border:-border]
    return estimated
