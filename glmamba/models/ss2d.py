from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scan_expand_4dir(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """
    Expand (B,C,H,W) into 4 sequences (B, L, C) scanning:
      left->right, top->bottom, right->left, bottom->top.
    Returns sequences and (H,W) for merging.
    """
    B, C, H, W = x.shape
    # LR: row-major
    s_lr = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B,L,C)
    # TB: col-major
    s_tb = x.permute(0, 3, 2, 1).reshape(B, H * W, C)
    # RL and BT are reversals
    s_rl = torch.flip(s_lr, dims=[1])
    s_bt = torch.flip(s_tb, dims=[1])
    return s_lr, s_tb, s_rl, s_bt, (H, W)


def _scan_merge_4dir(
    y_lr: torch.Tensor,
    y_tb: torch.Tensor,
    y_rl: torch.Tensor,
    y_bt: torch.Tensor,
    hw: tuple[int, int],
) -> torch.Tensor:
    """
    Merge 4 sequences (B,L,C) back to (B,C,H,W) by reversing and averaging.
    """
    H, W = hw
    B, L, C = y_lr.shape
    assert L == H * W

    y_rl = torch.flip(y_rl, dims=[1])
    y_bt = torch.flip(y_bt, dims=[1])

    # restore spatial
    m_lr = y_lr.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B,C,H,W)
    m_rl = y_rl.reshape(B, H, W, C).permute(0, 3, 1, 2)

    m_tb = y_tb.reshape(B, W, H, C).permute(0, 3, 2, 1)  # (B,C,H,W)
    m_bt = y_bt.reshape(B, W, H, C).permute(0, 3, 2, 1)

    return (m_lr + m_tb + m_rl + m_bt) * 0.25


class LinearStateSpace1D(nn.Module):
    """
    A lightweight, pure-PyTorch linear-time sequence mixer used as a fallback when
    Mamba SSM kernels are not available.

    This is NOT the official Mamba selective scan, but preserves:
      - linear complexity in sequence length
      - directional scanning and merging (done in SS2D)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * dim, bias=True)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,C)
        u, v = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(u)

        v = v.transpose(1, 2)  # (B,C,L)
        v = self.dwconv(v)
        v = v.transpose(1, 2)  # (B,L,C)
        v = F.silu(v)
        y = u * v
        y = self.out_proj(self.norm(y))
        return y


class SS2D(nn.Module):
    """
    2D selective scan core used in Mamba block.

    If `mamba_ssm` is installed, this module will try to use its sequence mixer.
    Otherwise it uses a lightweight linear-time fallback (LinearStateSpace1D).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

        self.impl = None
        try:
            # Common API in mamba-ssm:
            # from mamba_ssm.modules.mamba_simple import Mamba
            from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore

            self.impl = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            self._uses_mamba = True
        except Exception:
            self.impl = LinearStateSpace1D(dim)
            self._uses_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> sequences -> process -> merge
        s_lr, s_tb, s_rl, s_bt, hw = _scan_expand_4dir(x)
        y_lr = self.impl(s_lr)
        y_tb = self.impl(s_tb)
        y_rl = self.impl(s_rl)
        y_bt = self.impl(s_bt)
        return _scan_merge_4dir(y_lr, y_tb, y_rl, y_bt, hw)

