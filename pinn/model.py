from __future__ import annotations

import torch
from torch import nn


class DistributionNet(nn.Module):
    """Positive neural representation of a local-field probability density."""

    def __init__(self, hidden: int = 96, depth: int = 4, fourier_features: int = 8) -> None:
        super().__init__()
        # Reduces the low-frequency bias of a plain tanh MLP for narrow peaks.
        self.register_buffer("frequencies", 2.0 ** torch.arange(fourier_features, dtype=torch.float64))
        layers: list[nn.Module] = []
        input_size = 1 + 2 * fourier_features
        for _ in range(depth):
            layers.extend([nn.Linear(input_size, hidden), nn.Tanh()])
            input_size = hidden
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)
        self.softplus = nn.Softplus(beta=2.0)
        self.log_amplitude = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.background = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, normalized_local_field: torch.Tensor) -> torch.Tensor:
        phase = torch.pi * normalized_local_field * self.frequencies
        features = torch.cat([normalized_local_field, torch.sin(phase), torch.cos(phase)], dim=1)
        return self.softplus(self.network(features).squeeze(-1)) + 1e-12
