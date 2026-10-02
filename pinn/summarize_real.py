from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge grouped experimental PINN results")
    parser.add_argument("--input", type=Path, default=Path("pinn/results_real_v3"))
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for path in args.input.glob("group_*/metrics.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No group metrics in {args.input}")
    rows.sort(key=lambda row: float(row["temperature_K"]))
    fields = list(rows[0])
    with (args.input / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    temperature = np.asarray([float(row["temperature_K"]) for row in rows])
    tikhonov = np.asarray([float(row["tikhonov_rmse"]) for row in rows])
    pinn = np.asarray([float(row["pinn_rmse"]) for row in rows])
    relative_gain = 100 * (tikhonov - pinn) / tikhonov
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.plot(temperature, tikhonov, "o--", label="Tikhonov (diploma)")
    axis.plot(temperature, pinn, "o-", label="PINN")
    axis.set(xlabel="Temperature, K", ylabel="RMSE of normalized spectrum", title="Experimental FieldSweep reconstruction")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.input / "rmse_by_temperature.png", dpi=180)
    plt.close(figure)

    report = [
        "# FieldSweep: PINN vs diploma Tikhonov",
        "",
        "All 22 spectra in `test_data` were fitted independently after normalization by each spectrum's maximum intensity.",
        "The Tikhonov baseline exactly reproduces `reconstruct_field_distribution.py`: 70 nodes from 1e-6 to 0.2 T, unweighted kernel, its N×N second-difference penalty, non-negativity, free background, and its GCV rule over 1…1e10.",
        "PINN uses the same no-convolution NMR kernel, but represents a positive normalized density and evaluates the integral with quadrature. Its singular first numerical node is fixed to zero because the analytical kernel contains 1/B_loc there.",
        "",
        f"PINN has lower in-sample spectral RMSE in {np.sum(pinn < tikhonov)} of {len(rows)} temperatures.",
        f"Mean relative RMSE reduction: {np.mean(relative_gain):.2f}%; median: {np.median(relative_gain):.2f}%.",
        "This is an in-sample fit comparison, not evidence by itself that PINN recovers the physically true distribution; synthetic and held-out tests remain necessary.",
        "",
        "| T, K | points | Tikhonov RMSE | PINN RMSE | reduction | lambda |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {float(row['temperature_K']):.2f} | {row['points']} | {float(row['tikhonov_rmse']):.5f} | {float(row['pinn_rmse']):.5f} | {100 * (float(row['tikhonov_rmse']) - float(row['pinn_rmse'])) / float(row['tikhonov_rmse']):.2f}% | {float(row['tikhonov_lambda']):.3e} |"
        for row in rows
    )
    (args.input / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Merged {len(rows)} temperatures into {args.input}")


if __name__ == "__main__":
    main()

