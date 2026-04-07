from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn.functional as F
from packaging import version
from torch import einsum, nn


if hasattr(torch.nn, "attention"):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _sdpa_ctx(config):
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


class Attend(nn.Module):
    def __init__(self, dropout: float = 0.0, flash: bool = False):
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
        is_cuda = q.is_cuda
        config = self.cuda_config if is_cuda else self.cpu_config
        with _sdpa_ctx(config):
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

    def forward(self, q, k, v):
        if self.flash:
            return self.flash_attn(q, k, v)

        scale = q.shape[-1] ** -0.5
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        return einsum("b h i j, b h j d -> b h i d", attn, v)
