"""Perturbation types applied to a latent at a branch point, plus
deterministic per-ensemble-member seed derivation.
"""
import hashlib

import torch
import torch.nn.functional as F


def derive_seed(perturbation_seed: int, class_id: int, base_idx: int, branch_point: int,
                 type_name: str, k: int) -> int:
    """Deterministic seed for one ensemble member, stable across runs/processes
    (unlike Python's built-in hash(), which is salted per-process)."""
    key = f"{perturbation_seed}|{class_id}|{base_idx}|{branch_point}|{type_name}|{k}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _generator(seed: int, device) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def mock(latent: torch.Tensor, seed: int, params: dict) -> torch.Tensor:
    """Identity perturbation, used for dry-run verification."""
    return latent.clone()


def gaussian_noise(latent: torch.Tensor, seed: int, params: dict) -> torch.Tensor:
    std = params.get("std", 0.1)
    gen = _generator(seed, latent.device)
    noise = torch.randn(latent.shape, generator=gen, device=latent.device, dtype=latent.dtype)
    return latent + std * noise


def gaussian_blur(latent: torch.Tensor, seed: int, params: dict) -> torch.Tensor:
    kernel_size = params.get("kernel_size", 5)
    sigma = params.get("sigma", 1.0)
    padding = kernel_size // 2

    coords = torch.arange(kernel_size, dtype=torch.float32, device=latent.device) - padding
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]

    C = latent.shape[1]
    kernel = kernel_2d.expand(C, 1, kernel_size, kernel_size).to(latent.dtype)
    return F.conv2d(latent, kernel, padding=padding, groups=C)


def nxn_mask(latent: torch.Tensor, seed: int, params: dict) -> torch.Tensor:
    n = params.get("size", 8)
    fill_value = params.get("fill_value", 0.0)
    gen = _generator(seed, latent.device)

    _, _, H, W = latent.shape
    n = min(n, H, W)
    top = int(torch.randint(0, H - n + 1, (1,), generator=gen, device=latent.device).item())
    left = int(torch.randint(0, W - n + 1, (1,), generator=gen, device=latent.device).item())

    out = latent.clone()
    out[:, :, top:top + n, left:left + n] = fill_value
    return out


def scattered_mask(latent: torch.Tensor, seed: int, params: dict) -> torch.Tensor:
    fraction = params.get("fraction", 0.05)
    fill_value = params.get("fill_value", 0.0)
    gen = _generator(seed, latent.device)

    mask = torch.rand(latent.shape[2:], generator=gen, device=latent.device) < fraction
    out = latent.clone()
    out[:, :, mask] = fill_value
    return out


PERTURBATIONS = {
    "mock": mock,
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "nxn_mask": nxn_mask,
    "scattered_mask": scattered_mask,
}


def apply_perturbation(type_name: str, latent: torch.Tensor, seed: int, params: dict) -> torch.Tensor:
    fn = PERTURBATIONS[type_name]
    return fn(latent, seed, params)


def build_ensemble(type_name: str, latent: torch.Tensor, params: dict, perturbation_seed: int,
                    class_id: int, base_idx: int, branch_point: int, k_count: int) -> torch.Tensor:
    """Applies `type_name` independently to K copies of `latent` (shape (1, C, H, W)),
    each with its own deterministically-derived seed, and stacks them into a
    (K, C, H, W) batch."""
    members = []
    for k in range(k_count):
        seed = derive_seed(perturbation_seed, class_id, base_idx, branch_point, type_name, k)
        members.append(apply_perturbation(type_name, latent, seed, params))
    return torch.cat(members, dim=0)
