from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_all_field_sweeps
from .kernel import build_forward_matrix, nmr_kernel
from .tikhonov import solve_tikhonov_diploma
from .train import train_pinn


def normalized(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values)))
    return values / scale if scale > 0 else values


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PINN with diploma Tikhonov on FieldSweep spectra")
    parser.add_argument("--data", type=Path, default=Path("test_data"))
    parser.add_argument("--output", type=Path, default=Path("pinn/results_real"))
    parser.add_argument("--adam-steps", type=int, default=1200)
    parser.add_argument("--lbfgs-steps", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None, help="Use only first N temperatures (smoke test)")
    parser.add_argument("--temperatures", nargs="+", type=float, default=None, help="Exact temperatures to process")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spectra = load_all_field_sweeps(args.data)
    if args.temperatures:
        requested = np.asarray(args.temperatures)
        spectra = [item for item in spectra if np.any(np.isclose(item.temperature, requested))]
    if args.limit:
        spectra = spectra[:args.limit]

    local_field = np.linspace(1e-6, 0.2, 70, dtype=np.float64)
    rows: list[dict[str, float | str]] = []
    for index, item in enumerate(spectra, start=1):
        target = normalized(item.intensity)
        # This is the exact no-convolution variant of the diploma baseline.
        diploma_kernel = nmr_kernel(item.field, local_field)
        physical_forward = build_forward_matrix(item.field, local_field, gaussian_sigma=None)
        print(f"[{index}/{len(spectra)}] T={item.temperature:.2f} K: Tikhonov", flush=True)
        tikhonov = solve_tikhonov_diploma(diploma_kernel, target)
        print(f"[{index}/{len(spectra)}] T={item.temperature:.2f} K: PINN", flush=True)
        pinn = train_pinn(
            physical_forward, target, local_field, seed=2026 + index,
            adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps,
        )
        tik_rmse = float(np.sqrt(np.mean((tikhonov.spectrum - target) ** 2)))
        pinn_rmse = float(np.sqrt(np.mean((pinn.spectrum - target) ** 2)))
        rows.append({
            "temperature_K": item.temperature,
            "points": len(item.field),
            "tikhonov_rmse": tik_rmse,
            "pinn_rmse": pinn_rmse,
            "tikhonov_lambda": tikhonov.lambda_value,
            "pinn_amplitude": pinn.amplitude,
            "pinn_background": pinn.background,
        })
        tik_density = normalized(tikhonov.distribution)
        pinn_density = normalized(pinn.distribution)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(local_field, tik_density, "C1--", lw=2, label="Tikhonov (diploma)")
        axes[0].plot(local_field, pinn_density, "C0-", lw=2, label="PINN")
        axes[0].set(xlabel=r"$B_{loc}$, T", ylabel="Normalized f", title=f"T = {item.temperature:.2f} K")
        axes[0].grid(alpha=0.25)
        axes[0].legend()
        axes[1].plot(item.field, target, "k.", ms=4, label="Experiment")
        axes[1].plot(item.field, tikhonov.spectrum, "C1--", lw=1.8, label=f"Tikhonov, RMSE={tik_rmse:.4f}")
        axes[1].plot(item.field, pinn.spectrum, "C0-", lw=1.8, label=f"PINN, RMSE={pinn_rmse:.4f}")
        axes[1].set(xlabel=r"$B$, T", ylabel="Normalized intensity", title="Spectrum reconstruction")
        axes[1].grid(alpha=0.25)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.output / f"{item.temperature:.2f}K.png", dpi=160)
        plt.close(fig)

    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temps = np.asarray([row["temperature_K"] for row in rows], dtype=float)
    tik = np.asarray([row["tikhonov_rmse"] for row in rows], dtype=float)
    pinn = np.asarray([row["pinn_rmse"] for row in rows], dtype=float)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(temps, tik, "o--", label="Tikhonov (diploma)")
    axis.plot(temps, pinn, "o-", label="PINN")
    axis.set(xlabel="Temperature, K", ylabel="RMSE of normalized spectrum", title="Fit to experimental FieldSweep spectra")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "rmse_by_temperature.png", dpi=180)
    plt.close(fig)
    better = int(np.sum(pinn < tik))
    report = [
        "# FieldSweep: PINN vs diploma Tikhonov",
        "",
        "Both methods were fitted to each normalized experimental spectrum independently.",
        "The Tikhonov baseline exactly follows `reconstruct_field_distribution.py`: 70 nodes, unweighted kernel, its D matrix, non-negativity, free background and its GCV rule.",
        "PINN uses the same no-convolution kernel, but treats its output as a normalized density and includes quadrature in the integral.",
        "",
        f"PINN has lower in-sample spectral RMSE in {better} of {len(rows)} temperatures.",
        "",
        "| T, K | N | Tikhonov RMSE | PINN RMSE | lambda |",
        "|---:|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['temperature_K']:.2f} | {row['points']} | {row['tikhonov_rmse']:.5f} | {row['pinn_rmse']:.5f} | {row['tikhonov_lambda']:.3e} |"
        for row in rows
    )
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
