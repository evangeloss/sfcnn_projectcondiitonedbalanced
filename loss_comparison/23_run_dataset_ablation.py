"""Run a controlled 2x2 SFCNN dataset/normalization ablation.

The two factors are:
  1. channel diversity: 1000 x 31 pairs versus 4000 x 8 pairs;
  2. normalization: original target-assisted versus observation-only.

All cells use the same stratified SNR sampler, seeds, model initialization,
evaluation realizations, loss settings, and approximately 31k/32k samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOSS_RUNNER = Path(__file__).with_name("22_run_loss_comparison.py")
SNR_DB = np.arange(-10, 25, 5)


CELLS = (
    {
        "id": "standard_target_assisted",
        "label": "1,000 channels x 31 pairs | target-assisted",
        "train_channels": 1000,
        "val_channels": 200,
        "pairs": 31,
        "normalization": "target_assisted",
    },
    {
        "id": "diverse_target_assisted",
        "label": "4,000 channels x 8 pairs | target-assisted",
        "train_channels": 4000,
        "val_channels": 800,
        "pairs": 8,
        "normalization": "target_assisted",
    },
    {
        "id": "standard_observation_only",
        "label": "1,000 channels x 31 pairs | observation-only",
        "train_channels": 1000,
        "val_channels": 200,
        "pairs": 31,
        "normalization": "observation_only",
    },
    {
        "id": "diverse_observation_only",
        "label": "4,000 channels x 8 pairs | observation-only",
        "train_channels": 4000,
        "val_channels": 800,
        "pairs": 8,
        "normalization": "observation_only",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--losses",
        nargs="+",
        default=("residual_mse", "channel_nmse_cvar"),
        choices=("residual_mse", "channel_nmse", "hybrid", "channel_nmse_cvar"),
    )
    parser.add_argument("--eval-channels", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "dataset_ablation_results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small two-epoch functionality check.",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.eval_channels <= 0 or args.batch_size <= 0:
        parser.error("epochs, eval-channels, and batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    return args


def effective_cell(cell: dict[str, object], quick: bool) -> dict[str, object]:
    result = dict(cell)
    if quick:
        diverse = str(cell["id"]).startswith("diverse")
        result["train_channels"] = 12 if diverse else 6
        result["val_channels"] = 6 if diverse else 3
        result["pairs"] = 2 if diverse else 4
    return result


def run_cell(args: argparse.Namespace, cell: dict[str, object]) -> Path:
    cell_dir = args.output / str(cell["id"])
    command = [
        sys.executable,
        str(LOSS_RUNNER),
        "--epochs", str(2 if args.quick else args.epochs),
        "--losses", *args.losses,
        "--train-channels", str(cell["train_channels"]),
        "--val-channels", str(cell["val_channels"]),
        "--eval-channels", str(3 if args.quick else args.eval_channels),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--pairs-per-channel", str(cell["pairs"]),
        "--normalization", str(cell["normalization"]),
        "--seed", str(args.seed),
        "--evaluation-seed", str(args.evaluation_seed),
        "--output", str(cell_dir),
    ]
    print("\n" + "=" * 78, flush=True)
    print("ABLATION CELL:", cell["label"], flush=True)
    print("=" * 78, flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return cell_dir


def aggregate(
    args: argparse.Namespace,
    completed: list[tuple[dict[str, object], Path]],
) -> None:
    rows: list[dict[str, object]] = []
    arrays: dict[tuple[str, str], np.ndarray] = {}
    ls_reference: np.ndarray | None = None

    for cell, cell_dir in completed:
        report = json.loads(
            (cell_dir / "loss_comparison_report.json").read_text(encoding="utf-8")
        )
        saved = np.load(cell_dir / "loss_comparison_arrays.npz")
        if ls_reference is None:
            ls_reference = saved["nmse_ls"]
        for ranking in report["ranking"]:
            loss = str(ranking["loss"])
            values = saved[f"nmse_{loss}"]
            arrays[(str(cell["id"]), loss)] = values
            rows.append(
                {
                    "cell": cell["id"],
                    "dataset": cell["label"],
                    "loss": loss,
                    "train_channels": cell["train_channels"],
                    "pairs_per_channel": cell["pairs"],
                    "training_samples": int(cell["train_channels"]) * int(cell["pairs"]),
                    "normalization": cell["normalization"],
                    "mean_nmse_linear": ranking["mean_nmse_linear"],
                    "mean_nmse_db": ranking["mean_nmse_db"],
                    "l3_mean_nmse_linear": ranking["l3_mean_nmse_linear"],
                    "l3_mean_nmse_db": ranking["l3_mean_nmse_db"],
                    "final_validation_objective": ranking["final_validation_objective"],
                    "training_seconds": ranking["training_seconds"],
                }
            )

    rows.sort(key=lambda row: (str(row["loss"]), float(row["l3_mean_nmse_linear"])))
    with (args.output / "dataset_ablation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, len(args.losses), figsize=(7 * len(args.losses), 5), squeeze=False)
    for axis, loss in zip(axes.flat, args.losses):
        if ls_reference is not None:
            axis.semilogy(SNR_DB, ls_reference[2], "k--*", label="LS / TE")
        for cell, _ in completed:
            axis.semilogy(
                SNR_DB,
                arrays[(str(cell["id"]), loss)][2],
                "-o",
                markerfacecolor="white",
                label=str(cell["label"]),
            )
        axis.set_title(f"{loss}, test L=3")
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("NMSE")
        axis.set_xticks(SNR_DB)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Dataset diversity x normalization ablation")
    fig.tight_layout()
    fig.savefig(args.output / "dataset_ablation_l3_nmse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    compact_report = {
        "experiment": "2x2 channel-diversity and normalization ablation",
        "controlled_variables": {
            "stratified_snr_sampling": True,
            "seed": args.seed,
            "evaluation_seed": args.evaluation_seed,
            "epochs": 2 if args.quick else args.epochs,
            "evaluation_channels": 3 if args.quick else args.eval_channels,
            "losses": list(args.losses),
        },
        "cells": [cell for cell, _ in completed],
        "results": rows,
    }
    (args.output / "dataset_ablation_report.json").write_text(
        json.dumps(compact_report, indent=2), encoding="utf-8"
    )

    print("\n2x2 ablation complete. L=3 ranking by loss:")
    for loss in args.losses:
        print(f"\n{loss}")
        for row in (item for item in rows if item["loss"] == loss):
            print(
                f"  {row['cell']}: {float(row['l3_mean_nmse_linear']):.6e} "
                f"({float(row['l3_mean_nmse_db']):.2f} dB)"
            )
    print("\nResults:", args.output.resolve())


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    completed = []
    for template in CELLS:
        cell = effective_cell(template, args.quick)
        completed.append((cell, run_cell(args, cell)))
    aggregate(args, completed)


if __name__ == "__main__":
    main()
