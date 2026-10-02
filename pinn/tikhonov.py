from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

@dataclass
class TikhonovResult:
    distribution: np.ndarray
    spectrum: np.ndarray
    lambda_value: float
    background: float


def diploma_penalty(n: int, order: int = 2) -> np.ndarray:
    """Exact D matrix used by reconstruct_field_distribution.py.

    This intentionally has shape (n, n), including its boundary rows, to make
    the comparison with the diploma code exact rather than merely similar.
    """
    if order == 2:
        return (
            np.diag(np.ones(n - 1), -1)
            - 2.0 * np.diag(np.ones(n), 0)
            + np.diag(np.ones(n - 1), 1)
        )
    if order == 1:
        return np.diff(np.eye(n), axis=0)
    raise ValueError("order must be 1 or 2")


def _solve_diploma(
    kernel: np.ndarray,
    spectrum: np.ndarray,
    penalty: np.ndarray,
    lambda_value: float,
) -> tuple[np.ndarray, float]:
    n = kernel.shape[1]
    extended = np.hstack([kernel, np.ones((len(spectrum), 1))])
    penalty_extended = np.zeros((penalty.shape[0], n + 1))
    penalty_extended[:, :n] = penalty
    design = np.vstack([extended, np.sqrt(lambda_value) * penalty_extended])
    target = np.hstack([spectrum, np.zeros(penalty.shape[0])])
    lower = np.zeros(n + 1)
    lower[-1] = -1e20
    upper = np.full(n + 1, np.inf)
    result = lsq_linear(design, target, bounds=(lower, upper), lsmr_tol="auto")
    return result.x[:-1], float(result.x[-1])


def solve_tikhonov_diploma(
    kernel: np.ndarray,
    spectrum: np.ndarray,
    lambdas: np.ndarray | None = None,
    penalty_order: int = 2,
) -> TikhonovResult:
    """Reproduce the GCV and constrained solution in the diploma scripts.

    `kernel` is deliberately not quadrature-weighted: the diploma represents
    f(B_loc) as discrete values on its grid.
    """
    if lambdas is None:
        lambdas = np.logspace(0, 10, 100)
    penalty = diploma_penalty(kernel.shape[1], penalty_order)
    singular_values = np.linalg.svd(kernel, compute_uv=False)
    best_lambda = float(lambdas[0])
    best_gcv = np.inf
    for value in lambdas:
        estimate, background = _solve_diploma(kernel, spectrum, penalty, float(value))
        residual = spectrum - (kernel @ estimate + background)
        trace_h = np.sum(singular_values**2 / (singular_values**2 + value))
        score = np.linalg.norm(residual) ** 2 / max((len(spectrum) - trace_h) ** 2, 1e-24)
        if score < best_gcv:
            best_gcv = float(score)
            best_lambda = float(value)
    estimate, background = _solve_diploma(kernel, spectrum, penalty, best_lambda)
    return TikhonovResult(estimate, kernel @ estimate + background, best_lambda, background)
