from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge one-case synthetic benchmark reports")
    parser.add_argument("--input", type=Path, default=Path("pinn/results_v2"))
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    for path in args.input.glob("*/metrics.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No one-case metrics under {args.input}")
    order = ["gaussian", "lorentzian", "mixture", "double_gaussian", "edge_peak", "step", "delta"]
    rows.sort(key=lambda row: (order.index(row["case"]), row["method"]))
    with (args.input / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Expanded synthetic benchmark: PINN vs diploma Tikhonov",
        "",
        "Gaussian noise standard deviation: 0.005 of the normalized spectral amplitude.",
        "Tikhonov reproduces the exact discrete implementation in `reconstruct_field_distribution.py`.",
        "",
        "| Case | Method | distribution rel. L2 | spectrum RMSE, clean | W1, T |",
        "|---|---|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['case']} | {row['method']} | {float(row['distribution_rel_l2']):.4f} | {float(row['spectrum_rmse_clean']):.5f} | {float(row['wasserstein_1_tesla']):.6f} |"
        for row in rows
    )
    report.extend([
        "",
        "PINN has lower distribution error for the discontinuous step and a marginally lower distribution L2 error for the delta peak; on the delta spectrum Tikhonov has a slightly lower forward RMSE. Tikhonov is superior for the smooth Gaussian, Lorentzian, mixture, double-Gaussian and edge-peak cases in this single-spectrum formulation.",
        "This indicates that the present value of PINN is not a universally better single-spectrum regularizer. Its intended advantage must be tested in the next version: one shared f(B_loc, T) model across temperatures.",
    ])
    (args.input / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Merged {len(rows)} rows into {args.input}")


if __name__ == "__main__":
    main()
