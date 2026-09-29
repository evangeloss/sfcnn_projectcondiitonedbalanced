"""Plot the L=3 NMSE-versus-SNR curves from a loss-comparison run.

Example:

    python loss_comparison/plot_l3_nmse.py \
        --arrays loss_comparison_results/loss_comparison_arrays.npz \
        --output loss_comparison_results/l3_loss_comparison.png \
        --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "nmse_ls": "LS / TE",
    "nmse_residual_mse": "SFCNN — residual MSE (original)",
    "nmse_channel_nmse": "SFCNN — channel NMSE",
    "nmse_hybrid": "SFCNN — hybrid",
    "nmse_channel_nmse_cvar": "SFCNN — channel NMSE + CVaR",
}

STYLES = {
    "nmse_ls": dict(color="black", linestyle="--", marker="*", linewidth=2.0),
    "nmse_residual_mse": dict(color="#0072B2", linestyle="-", marker="o"),
    "nmse_channel_nmse": dict(color="#E69F00", linestyle="-", marker="s"),
    "nmse_hybrid": dict(color="#009E73", linestyle="-", marker="^"),
    "nmse_channel_nmse_cvar": dict(color="#D55E00", linestyle="-", marker="D"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = np.load(args.arrays)
    snr_db = np.arange(-10, 25, 5)
    l3_row = 2

    fig, axis = plt.subplots(figsize=(8.6, 6.2))
    for key in LABELS:
        if key not in arrays.files:
            continue
        values = arrays[key]
        if values.shape != (4, 7):
            raise ValueError(f"{key} must have shape (4, 7), got {values.shape}")
        style = STYLES[key].copy()
        if key != "nmse_ls":
            style.update(markersize=6, markerfacecolor="white", linewidth=1.8)
        else:
            style.update(markersize=9)
        axis.semilogy(snr_db, values[l3_row], label=LABELS[key], **style)

    epoch_text = f", {args.epochs} training epochs" if args.epochs else ""
    axis.set_title(f"Loss Comparison for L=3{epoch_text}")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("NMSE")
    axis.set_xticks(snr_db)
    axis.set_xlim(-10.5, 20.5)
    axis.set_ylim(1e-3, 2e1)
    axis.grid(True, which="major", linestyle="-", alpha=0.28)
    axis.grid(True, which="minor", linestyle=":", alpha=0.18)
    axis.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    svg_output = args.output.with_suffix(".svg")
    fig.savefig(svg_output, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", args.output)
    print("Saved:", svg_output)


if __name__ == "__main__":
    main()
