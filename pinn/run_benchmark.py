from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .kernel import nmr_kernel, trapezoid_weights
from .synthetic import all_cases
from .tikhonov import solve_tikhonov_diploma
from .train import train_pinn


def relative_l2(truth: np.ndarray, estimate: np.ndarray, weights: np.ndarray) -> float:
    numerator = np.sum((truth - estimate) ** 2 * weights)
    denominator = np.sum(truth**2 * weights)
    return float(np.sqrt(numerator / denominator))


def wasserstein_1(truth: np.ndarray, estimate: np.ndarray, grid: np.ndarray) -> float:
    weights = trapezoid_weights(grid)
    cdf_truth = np.cumsum(truth * weights)
    cdf_estimate = np.cumsum(estimate * weights)
    return float(np.sum(np.abs(cdf_truth - cdf_estimate) * weights))


def main() -> None:
    parser = argparse.ArgumentParser(description="PINN vs Tikhonov synthetic NMR benchmark")
    parser.add_argument("--output", type=Path, default=Path("pinn/results"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--adam-steps", type=int, default=2500)
    parser.add_argument("--lbfgs-steps", type=int, default=250)
    parser.add_argument("--cases", nargs="+", default=None, help="Optional case names, for example: gaussian delta")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    cases = all_cases(args.seed, args.noise)
    if args.cases:
        requested = set(args.cases)
        cases = [case for case in cases if case.name in requested]
        unknown = requested - {case.name for case in cases}
        if unknown:
            raise ValueError(f"Unknown test case(s): {', '.join(sorted(unknown))}")
    for index, case in enumerate(cases):
        # Tikhonov receives the unweighted discrete kernel used in the diploma;
        # the PINN receives the quadrature-aware physical operator.
        diploma_kernel = nmr_kernel(case.field, case.local_field)
        print(f"[{index + 1}/{len(cases)}] {case.name}: Tikhonov", flush=True)
        tikhonov = solve_tikhonov_diploma(diploma_kernel, case.noisy_spectrum)
        print(f"[{index + 1}/{len(cases)}] {case.name}: PINN", flush=True)
        pinn = train_pinn(
            case.forward_matrix,
            case.noisy_spectrum,
            case.local_field,
            seed=args.seed + index,
            adam_steps=args.adam_steps,
            lbfgs_steps=args.lbfgs_steps,
        )
        weights = trapezoid_weights(case.local_field)
        for method, density, spectrum in (
            ("Tikhonov (diploma)", tikhonov.distribution / np.trapezoid(tikhonov.distribution, case.local_field), tikhonov.spectrum),
            ("PINN", pinn.distribution, pinn.spectrum),
        ):
            rows.append(
                {
                    "case": case.name,
                    "method": method,
                    "distribution_rel_l2": relative_l2(case.distribution, density, weights),
                    "spectrum_rmse_noisy": float(np.sqrt(np.mean((spectrum - case.noisy_spectrum) ** 2))),
                    "spectrum_rmse_clean": float(np.sqrt(np.mean((spectrum - case.clean_spectrum) ** 2))),
                    "wasserstein_1_tesla": wasserstein_1(case.distribution, density, case.local_field),
                    "lambda": tikhonov.lambda_value if method == "Tikhonov (diploma)" else np.nan,
                }
            )

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(case.local_field, case.distribution, "k-", lw=2.2, label="Truth")
        axes[0].plot(case.local_field, tikhonov.distribution / np.trapezoid(tikhonov.distribution, case.local_field), "C1--", lw=2, label="Tikhonov (diploma)")
        axes[0].plot(case.local_field, pinn.distribution, "C0-", lw=2, label="PINN")
        axes[0].set(xlabel=r"$B_{loc}$, T", ylabel=r"$f(B_{loc})$", title=case.name)
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(case.field, case.noisy_spectrum, "k.", ms=3, label="Noisy data")
        axes[1].plot(case.field, case.clean_spectrum, color="0.5", lw=2, label="Clean")
        axes[1].plot(case.field, tikhonov.spectrum, "C1--", lw=1.8, label="Tikhonov (diploma)")
        axes[1].plot(case.field, pinn.spectrum, "C0-", lw=1.8, label="PINN")
        axes[1].set(xlabel=r"$B$, T", ylabel="Normalized intensity", title="Forward reconstruction")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(args.output / f"{case.name}.png", dpi=180)
        plt.close(figure)

    csv_path = args.output / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# PINN and Tikhonov synthetic benchmark",
        "",
        f"Noise standard deviation: `{args.noise:.4f}` of normalized spectrum amplitude.",
        "",
        "| Case | Method | Distribution rel. L2 | Spectrum RMSE (clean) | W1, T |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['case']} | {row['method']} | "
            f"{row['distribution_rel_l2']:.4f} | {row['spectrum_rmse_clean']:.5f} | "
            f"{row['wasserstein_1_tesla']:.6f} |"
        )
    report.extend(
        [
            "",
            "The Tikhonov baseline is an exact reproduction of the diploma's discrete "
            "second-order penalty and GCV rule. PINN uses the normalized integral operator.",
        ]
    )
    (args.output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
