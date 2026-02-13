from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Normalized Mean Squared Error.
    """
    pred = pred.detach()
    target = target.detach()
    num = torch.mean((pred - target) ** 2)
    den = torch.mean(target**2) + eps
    return float((num / den).cpu().item())


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1e-8) -> float:
    """
    PSNR in dB. Assumes images are scaled to [0, data_range].
    """
    pred = pred.detach()
    target = target.detach()
    mse = torch.mean((pred - target) ** 2).clamp_min(eps)
    return float((10.0 * torch.log10((data_range**2) / mse)).cpu().item())


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    return kernel2d


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-8,
) -> float:
    """
    SSIM (single-scale) for NCHW tensors. Returns mean SSIM over batch and channels.
    """
    pred = pred.detach()
    target = target.detach()
    assert pred.shape == target.shape and pred.ndim == 4
    B, C, H, W = pred.shape

    # ensure odd window and not larger than image
    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, H if H % 2 == 1 else H - 1, W if W % 2 == 1 else W - 1)
    if window_size < 3:
        # too small; fall back to a simple similarity proxy
        return float((1.0 - torch.mean((pred - target).abs())).clamp(0, 1).cpu().item())

    kernel = _gaussian_kernel(window_size, sigma, pred.device, pred.dtype)
    kernel = kernel.repeat(C, 1, 1, 1)  # (C,1,ws,ws)

    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu12

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + eps)
    return float(ssim_map.mean().cpu().item())

