"""
hyperbolic_utils.py

Core math utilities for the Lorentz model (hyperboloid) and
hyperbolic rotations used by the HIM implementation.
"""

import torch
import math
from typing import Tuple

device = torch.device("cpu")

# ---------------- Lorentz model basics ---------------- #

def lorentz_inner(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Lorentzian scalar product ⟨x, y⟩_L with curvature γ."""
    time = -x[..., :1] * y[..., :1]
    space = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return time + space

def lorentz_distance2(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Squared Lorentzian distance d_L^2(x, y) as in the paper."""
    # lorentz_inner(x, y, gamma) は <x, y>_L を計算します。
    # 論文の定義: d_L^2(x,y) = -2*gamma - 2*<x,y>_L
    l_inner = lorentz_inner(x, y, gamma)
    return (-2.0 * gamma) - (2.0 * l_inner)

def project_to_lorentz(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Project Euclidean vector (n) to Lorentz model vector (n+1)."""
    spatial = x
    x0 = torch.sqrt(gamma + (spatial ** 2).sum(dim=-1, keepdim=True))
    return torch.cat([x0, spatial], dim=-1)

# ---------------- Rotations (block‑diag) -------------- #

def build_rotation_mat(thetas: torch.Tensor) -> torch.Tensor:
    """Create block‑diagonal 2×2 rotation matrices for each angle."""
    # thetas: (d/2,) tensor
    blocks = []
    for theta in thetas:
        c, s = torch.cos(theta), torch.sin(theta)
        blocks.append(torch.stack([torch.stack([c, -s]), torch.stack([s, c])]))
    return torch.block_diag(*blocks)

def rotate(x: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """Apply hyperbolic rotation on spatial coordinates only (leave x0)."""
    R = build_rotation_mat(thetas.to(x.device))
    spatial = torch.matmul(x[..., 1:], R.T)
    return torch.cat([x[..., :1], spatial], dim=-1)
