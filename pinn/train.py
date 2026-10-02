from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .kernel import as_torch, trapezoid_weights
from .model import DistributionNet

@dataclass
class PinnResult:
    distribution: np.ndarray
    spectrum: np.ndarray
    background: float
    amplitude: float
    final_loss: float


def train_pinn(
    forward: np.ndarray,
    spectrum: np.ndarray,
    local_field: np.ndarray,
    seed: int = 16,
    adam_steps: int = 2500,
    lbfgs_steps: int = 250,
    smooth_weight: float = 2e-5,
    edge_weight: float = 2e-4,
    device: str = "cpu",
) -> PinnResult:
    torch.manual_seed(seed)
    model = DistributionNet().to(device=device, dtype=torch.float64)
    matrix = as_torch(forward, device)
    target = as_torch(spectrum, device)
    grid = as_torch(local_field, device)
    weights = as_torch(trapezoid_weights(local_field), device)
    grid_input = (2.0 * (grid - grid.min()) / (grid.max() - grid.min()) - 1.0)[:, None]
    # The diploma grid contains an artificial 1e-6 T endpoint. The analytical
    # kernel is proportional to 1/B_loc there; a softplus network can never
    # output exactly zero and would otherwise create a spurious delta spike.
    # This only removes that numerical endpoint, not a physical low-field peak.
    suppress_singular_endpoint = bool(local_field[0] < 1e-4)
    endpoint_taper = (grid - grid.min()) / (grid.max() - grid.min())
    # Match the scale before gradient descent. Experimental intensities are
    # normalized, whereas the discretized physical kernel may be O(10^2–10^3).
    # Starting amplitude at 1 can therefore trap Adam in an endpoint solution.
    with torch.no_grad():
        uniform_density = torch.ones_like(grid) / torch.sum(weights)
        if suppress_singular_endpoint:
            uniform_density = endpoint_taper
            uniform_density = uniform_density / torch.sum(uniform_density * weights)
        basis = matrix @ uniform_density
        amplitude0 = torch.clamp(torch.dot(target, basis) / torch.dot(basis, basis), min=1e-8)
        background0 = torch.mean(target - amplitude0 * basis)
        model.log_amplitude.copy_(torch.log(amplitude0))
        model.background.copy_(background0)

    def prediction_and_loss() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = model(grid_input)
        if suppress_singular_endpoint:
            raw = raw * endpoint_taper
        density = raw / torch.sum(raw * weights)
        amplitude = torch.exp(model.log_amplitude)
        prediction = amplitude * (matrix @ density) + model.background
        data_loss = torch.mean((prediction - target) ** 2)
        curvature = density[2:] - 2.0 * density[1:-1] + density[:-2]
        # Scale-free penalties: discourage oscillations and a non-zero far tail.
        smoothness = torch.mean(curvature**2) / (torch.mean(density**2) + 1e-12)
        edge = density[-1] ** 2 / (torch.mean(density**2) + 1e-12)
        loss = data_loss + smooth_weight * smoothness + edge_weight * edge
        return prediction, density, loss

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        _, _, loss = prediction_and_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=0.8,
        max_iter=lbfgs_steps,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer_lbfgs.zero_grad(set_to_none=True)
        _, _, value = prediction_and_loss()
        value.backward()
        return value

    optimizer_lbfgs.step(closure)
    with torch.no_grad():
        prediction, density, final_loss = prediction_and_loss()
    return PinnResult(
        distribution=density.cpu().numpy(),
        spectrum=prediction.cpu().numpy(),
        background=float(model.background.detach().cpu()),
        amplitude=float(torch.exp(model.log_amplitude.detach()).cpu()),
        final_loss=float(final_loss.cpu()),
    )
