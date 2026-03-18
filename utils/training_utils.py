"""
Reusable training utilities shared across trainer variants.
"""

from __future__ import annotations

import random
from typing import List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def knn(points: Tensor, k: int, max_pair_elements: int = 20_000_000) -> Tensor:
    """
    Return KNN distances [N, k] (includes self at index 0).

    Uses chunked pairwise distance computation to avoid OOM on large point sets.
    """
    n = int(points.shape[0])
    if n == 0:
        return torch.empty((0, k), dtype=points.dtype, device=points.device)

    k_eff = min(k, n)
    if n >= 50_000:
        try:
            from sklearn.neighbors import NearestNeighbors

            pts_np = points.detach().cpu().numpy()
            nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
            nn.fit(pts_np)
            dists, _ = nn.kneighbors(pts_np, return_distance=True)
            vals = torch.from_numpy(dists).to(points.device, dtype=points.dtype)
            if k_eff == k:
                return vals
            pad = torch.full((n, k - k_eff), float("inf"), dtype=vals.dtype, device=vals.device)
            return torch.cat([vals, pad], dim=-1)
        except Exception:
            pass

    if n * n <= max_pair_elements:
        d = torch.cdist(points, points)
        vals, _ = torch.topk(d, k=k_eff, largest=False, dim=-1)
    else:
        chunk = max(1, min(n, max_pair_elements // n))
        out_vals = []
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            d_chunk = torch.cdist(points[start:end], points)
            vals_chunk, _ = torch.topk(d_chunk, k=k_eff, largest=False, dim=-1)
            out_vals.append(vals_chunk)
        vals = torch.cat(out_vals, dim=0)

    if k_eff == k:
        return vals

    # Rare case: n < k. Pad with +inf to keep shape contract.
    pad = torch.full((n, k - k_eff), float("inf"), dtype=vals.dtype, device=vals.device)
    return torch.cat([vals, pad], dim=-1)


def compute_psnr(pred: Tensor, gt: Tensor, eps: float = 1e-8) -> Tensor:
    mse = F.mse_loss(pred, gt)
    return -10.0 * torch.log10(mse + eps)


def simple_ssim(img1: Tensor, img2: Tensor, padding: str = "valid") -> Tensor:
    """Tiny SSIM implementation; expects [B, 3, H, W] in [0, 1]."""
    c1, c2 = 0.01**2, 0.03**2
    ksize = 11
    sigma = 1.5
    coords = torch.arange(ksize, dtype=img1.dtype, device=img1.device) - ksize // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0).repeat(img1.shape[1], 1, 1, 1)
    pad = 0 if padding == "valid" else ksize // 2
    mu1 = F.conv2d(img1, kernel, padding=pad, groups=img1.shape[1])
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=img2.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=img1.shape[1]) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def parse_steps_csv(steps_csv: Optional[str]) -> Set[int]:
    if steps_csv is None:
        return set()
    text = steps_csv.strip()
    if len(text) == 0:
        return set()
    values: Set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if len(token) == 0:
            continue
        values.add(int(token))
    return values


def parse_float_csv(values_csv: Optional[str], default: List[float]) -> List[float]:
    if values_csv is None:
        return list(default)
    text = values_csv.strip()
    if len(text) == 0:
        return list(default)
    values: List[float] = []
    for token in text.split(","):
        token = token.strip()
        if len(token) == 0:
            continue
        values.append(float(token))
    return values if values else list(default)


def parse_name_csv(values_csv: Optional[str], default: List[str]) -> List[str]:
    if values_csv is None:
        return list(default)
    text = values_csv.strip()
    if len(text) == 0:
        return list(default)
    values = [token.strip() for token in text.split(",") if token.strip()]
    return values if values else list(default)


def soft_sigmoid(x: Tensor, softness: float) -> Tensor:
    return 1.0 / (1.0 + torch.exp(-softness * x))
