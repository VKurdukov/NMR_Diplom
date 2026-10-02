from __future__ import annotations

import numpy as np
import torch


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Quadrature weights for a non-uniform one-dimensional grid."""
    x = np.asarray(x, dtype=np.float64)
    weights = np.empty_like(x)
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    return weights


def nmr_kernel(
    field: np.ndarray,
    local_field: np.ndarray,
    larmor_field: float = 0.72525,
) -> np.ndarray:
    """Kernel used in the original diploma reconstruction scripts."""
    field = np.asarray(field, dtype=np.float64)[:, None]
    local_field = np.asarray(local_field, dtype=np.float64)[None, :]
    valid = local_field >= np.abs(field - larmor_field)
    safe_field = np.maximum(np.abs(field), 1e-12)
    safe_local = np.maximum(local_field, 1e-12)
    values = (field**2 - local_field**2 + larmor_field**2) / (
        safe_local * safe_field**2
    )
    return np.where(valid, values, 0.0)


def gaussian_blur_matrix(field: np.ndarray, sigma: float) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    delta = field[:, None] - field[None, :]
    blur = np.exp(-0.5 * (delta / sigma) ** 2)
    return blur / np.maximum(blur.sum(axis=1, keepdims=True), 1e-15)


def build_forward_matrix(
    field: np.ndarray,
    local_field: np.ndarray,
    larmor_field: float = 0.72525,
    gaussian_sigma: float | None = None,
) -> np.ndarray:
    """Build a quadrature-aware, Gaussian-broadened forward operator."""
    kernel = nmr_kernel(field, local_field, larmor_field)
    if gaussian_sigma is not None and gaussian_sigma > 0:
        kernel = gaussian_blur_matrix(field, gaussian_sigma) @ kernel
    return kernel * trapezoid_weights(local_field)[None, :]


def as_torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64, device=device)
