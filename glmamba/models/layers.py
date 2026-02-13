from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class PatchEmbed2x2(nn.Module):
    """
    Non-overlapping 2x2 patches, stride=2.
    Returns a feature map at H/2, W/2 with channels=embed_dim.
    """

    def __init__(self, in_ch: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchUnembed2x2(nn.Module):
    """
    Inverse of PatchEmbed2x2 (upsample by 2).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _split_quadrants(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # x: (B,C,H,W)
    B, C, H, W = x.shape
    h2 = H // 2
    w2 = W // 2
    q1 = x[:, :, :h2, :w2]
    q2 = x[:, :, :h2, w2:]
    q3 = x[:, :, h2:, :w2]
    q4 = x[:, :, h2:, w2:]
    return q1, q2, q3, q4


def _merge_quadrants(q1: torch.Tensor, q2: torch.Tensor, q3: torch.Tensor, q4: torch.Tensor) -> torch.Tensor:
    top = torch.cat([q1, q2], dim=-1)
    bot = torch.cat([q3, q4], dim=-1)
    return torch.cat([top, bot], dim=-2)


class DeformBlock(nn.Module):
    """
    Deformable conv feature extractor.

    Paper uses offsets Δp and modulation masks Δm. We use torchvision's deform conv when
    available; otherwise we fall back to a standard conv (shape-compatible).
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Offset: 2*k*k per spatial location per group (we use group=1)
        self.offset_conv = nn.Conv2d(channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        # Mask: k*k per location
        self.mask_conv = nn.Conv2d(channels, kernel_size * kernel_size, kernel_size=3, padding=1)

        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.fallback = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=self.padding, bias=True)

        try:
            from torchvision.ops import deform_conv2d  # type: ignore

            self._deform_conv2d = deform_conv2d
        except Exception:
            self._deform_conv2d = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._deform_conv2d is None:
            return self.fallback(x)

        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))

        # torchvision.ops.deform_conv2d supports mask in newer versions; in older ones it doesn't.
        try:
            try:
                y = self._deform_conv2d(
                    input=x,
                    offset=offset,
                    weight=self.weight,
                    bias=self.bias,
                    padding=(self.padding, self.padding),
                    mask=mask,
                )
            except RuntimeError:
                # Some builds expose deform_conv2d but do not support the current device/dtype.
                return self.fallback(x)
        except TypeError:
            try:
                y = self._deform_conv2d(
                    input=x,
                    offset=offset,
                    weight=self.weight,
                    bias=self.bias,
                    padding=(self.padding, self.padding),
                )
            except RuntimeError:
                return self.fallback(x)
        return y


class Modulator(nn.Module):
    """
    Enhance Mamba features using deformable features as a gate (sigmoid).

    Implements the core described behavior:
      modulated = mamba_feat * sigmoid(deform_feat)
      fused = modulated + deform_feat
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.proj_def = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_mam = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, deform_feat: torch.Tensor, mamba_feat: torch.Tensor) -> torch.Tensor:
        d = self.proj_def(deform_feat)
        m = self.proj_mam(mamba_feat)
        gate = torch.sigmoid(d)
        m_mod = m * gate
        return m_mod + d


class MultiModalityFusion(nn.Module):
    """
    Paper fusion block: fuse similarity/difference/complementarity and then weighted sum.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        # Complementarity weights: conv(cat(F_lr, F_ref)) -> softmax -> split
        self.comp_conv = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, bias=True)

        # Weights for {di, sim, com} branches using GMP + FC + softmax
        self.w_fc = nn.Linear(3 * channels, 3, bias=True)

    def forward(self, f_lr: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        # difference and similarity
        f_di = f_lr - f_ref
        f_sim = f_lr * f_ref

        # complementarity
        w = self.comp_conv(torch.cat([f_lr, f_ref], dim=1))  # (B,2C,H,W)
        # softmax across channel dimension so it can be split into two maps
        w = torch.softmax(w, dim=1)
        w_lr, w_ref = torch.split(w, [f_lr.shape[1], f_ref.shape[1]], dim=1)
        f_com = f_lr * w_lr + f_ref * w_ref

        # dynamic weighting across fusion manners
        cat3 = torch.cat([f_di, f_sim, f_com], dim=1)  # (B,3C,H,W)
        pooled = F.adaptive_max_pool2d(cat3, 1).flatten(1)  # (B,3C)
        weights = torch.softmax(self.w_fc(pooled), dim=1)  # (B,3)

        w_di = weights[:, 0].view(-1, 1, 1, 1)
        w_sim = weights[:, 1].view(-1, 1, 1, 1)
        w_com = weights[:, 2].view(-1, 1, 1, 1)
        return f_di * w_di + f_sim * w_sim + f_com * w_com

