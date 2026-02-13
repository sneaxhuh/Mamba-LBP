from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ChannelAttention
from .ss2d import SS2D


class MambaBlock2D(nn.Module):
    """
    Matches the paper's high-level block structure (Fig.2A):
      LN -> split into two paths:
        path1: linear + activation
        path2: linear -> depthwise conv -> activation -> SS2D -> LN
      multiply paths, then channel attention
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.ln = nn.GroupNorm(num_groups=1, num_channels=channels)  # LN-like for 2D

        self.p1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )

        self.p2_in = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.p2_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.p2_act = nn.SiLU()
        self.ss2d = SS2D(channels)
        self.p2_norm = nn.GroupNorm(num_groups=1, num_channels=channels)

        self.ca = ChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        a = self.p1(h)
        b = self.p2_in(h)
        b = self.p2_dw(b)
        b = self.p2_act(b)
        b = self.ss2d(b)
        b = self.p2_norm(b)
        y = a * b
        y = self.ca(y)
        return x + y


class LocalMamba2D(nn.Module):
    """
    Local Mamba: split the feature map into 4 quadrants and apply the same MambaBlock2D
    independently (paper Fig.3B idea). Then merge back.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = MambaBlock2D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h2 = H // 2
        w2 = W // 2
        # If odd sizes, pad to even.
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            H = x.shape[2]
            W = x.shape[3]
            h2 = H // 2
            w2 = W // 2

        q1 = self.block(x[:, :, :h2, :w2])
        q2 = self.block(x[:, :, :h2, w2:])
        q3 = self.block(x[:, :, h2:, :w2])
        q4 = self.block(x[:, :, h2:, w2:])
        top = torch.cat([q1, q2], dim=-1)
        bot = torch.cat([q3, q4], dim=-1)
        y = torch.cat([top, bot], dim=-2)
        # unpad if needed
        return y[:, :, : (x.shape[2] - pad_h), : (x.shape[3] - pad_w)] if (pad_h or pad_w) else y

