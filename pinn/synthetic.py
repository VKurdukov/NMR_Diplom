from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernel import build_forward_matrix, trapezoid_weights


@dataclass
class SyntheticCase:
    name: str
    local_field: np.ndarray
    field: np.ndarray
    distribution: np.ndarray
    clean_spectrum: np.ndarray
    noisy_spectrum: np.ndarray
    noise_std: float
    forward_matrix: np.ndarray


def normalize_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    # The diploma's first grid node is 1e-6 T. The continuous kernel is
    # singular there, so this artificial endpoint has zero physical weight.
    if grid[0] < 1e-4:
        values[0] = 0.0
    integral = float(np.sum(values * trapezoid_weights(grid)))
    if integral <= 0:
        raise ValueError("Distribution has zero integral")
    return values / integral


def distribution(kind: str, x: np.ndarray) -> np.ndarray:
    if kind == "gaussian":
        values = np.exp(-0.5 * ((x - 0.072) / 0.014) ** 2)
    elif kind == "lorentzian":
        gamma = 0.012
        values = gamma**2 / ((x - 0.082) ** 2 + gamma**2)
    elif kind == "mixture":
        gaussian = np.exp(-0.5 * ((x - 0.052) / 0.009) ** 2)
        gamma = 0.010
        lorentzian = gamma**2 / ((x - 0.112) ** 2 + gamma**2)
        values = 0.58 * gaussian + 0.42 * lorentzian
    elif kind == "double_gaussian":
        values = (0.72 * np.exp(-0.5 * ((x - 0.060) / 0.007) ** 2)
                  + 0.28 * np.exp(-0.5 * ((x - 0.105) / 0.017) ** 2))
    elif kind == "edge_peak":
        values = np.exp(-0.5 * ((x - 0.012) / 0.005) ** 2)
    elif kind == "step":
        values = ((x >= 0.052) & (x <= 0.105)).astype(np.float64)
    elif kind == "delta":
        # A delta distribution represented by one narrow, grid-resolved peak.
        dx = float(np.mean(np.diff(x)))
        values = np.exp(-0.5 * ((x - 0.078) / (0.55 * dx)) ** 2)
    else:
        raise ValueError(f"Unknown synthetic distribution: {kind}")
    return normalize_density(values, x)


def make_case(
    kind: str,
    seed: int = 1234,
    noise_fraction: float = 0.005,
    n_field: int = 111,
    n_local: int = 70,
) -> SyntheticCase:
    local_field = np.linspace(1e-6, 0.20, n_local, dtype=np.float64)
    field = np.linspace(0.50, 0.95, n_field, dtype=np.float64)
    forward = build_forward_matrix(field, local_field)
    truth = distribution(kind, local_field)
    clean = forward @ truth
    clean /= np.max(np.abs(clean))
    noise_std = noise_fraction
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, noise_std, size=clean.shape)
    return SyntheticCase(
        name=kind,
        local_field=local_field,
        field=field,
        distribution=truth,
        clean_spectrum=clean,
        noisy_spectrum=noisy,
        noise_std=noise_std,
        forward_matrix=forward,
    )


def all_cases(seed: int = 1234, noise_fraction: float = 0.005) -> list[SyntheticCase]:
    kinds = ["gaussian", "lorentzian", "mixture", "double_gaussian", "edge_peak", "step", "delta"]
    return [make_case(kind, seed + i, noise_fraction) for i, kind in enumerate(kinds)]
