from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ExperimentalSpectrum:
    temperature: float
    field: np.ndarray
    intensity: np.ndarray
    filename: str


def load_field_sweep(path: Path) -> ExperimentalSpectrum:
    """Read the first two numeric columns of a FieldSweep export."""
    points: list[tuple[float, float]] = []
    number = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
    for line in path.read_text(encoding="utf-8").splitlines():
        columns = line.strip().replace(",", ".").split()
        if len(columns) >= 2 and number.match(columns[0]) and number.match(columns[1]):
            points.append((float(columns[0]), float(columns[1])))
    if len(points) < 5:
        raise ValueError(f"No usable spectral points in {path}")
    data = np.asarray(points, dtype=np.float64)
    order = np.argsort(data[:, 0])
    match = re.search(r"(\d+(?:\.\d+)?)K", path.name)
    if not match:
        raise ValueError(f"Temperature not found in {path.name}")
    return ExperimentalSpectrum(
        temperature=float(match.group(1)),
        field=data[order, 0],
        intensity=data[order, 1],
        filename=path.name,
    )


def load_all_field_sweeps(directory: Path) -> list[ExperimentalSpectrum]:
    spectra = [load_field_sweep(path) for path in directory.glob("FieldSweep *.txt")]
    return sorted(spectra, key=lambda item: item.temperature)

