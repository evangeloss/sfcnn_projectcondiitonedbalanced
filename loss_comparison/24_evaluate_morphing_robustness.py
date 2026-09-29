"""Evaluate two trained M=8 SFCNN checkpoints across morphing amplitudes.

This script performs inference only. Each model receives the same generated
channels, pilot noise, and normalized observations for a paired comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_ROOT / "original_sfcnn"
CORE_FILES = (
    "01_path_parameter_generation.py",
    "03_deformed_array_geometry.py",
    "05_steering_vector.py",
    "06_fim_system_generation.py",
    "08_shared_path_channel_builder.py",
    "12_multi_pilot_generation.py",
    "13_pilot_transmission.py",
    "14_ridge_te_estimator.py",
    "17_sfcnn_model.py",
)


def load_core_definitions() -> None:
    namespace = globals()
    for filename in CORE_FILES:
        path = ORIGINAL_DIR / filename
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-checkpoint", type=Path, required=True)
    parser.add_argument("--mixed-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--morphing-ratios",
        type=float,
        nargs="+",
        default=(0.01, 0.02, 0.10, 0.30, 0.50),
    )
    parser.add_argument(
        "--snr-db", type=float, nargs="+", default=(0.0, 10.0, 20.0)
    )
    parser.add_argument("--path-counts", type=int, nargs="+", default=(3,))
    parser.add_argument("--eval-channels", type=int, default=10)
    parser.add_argument("--m-views", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "morphing_robustness_results",
    )
    args = parser.parse_args()
    for checkpoint in (args.fixed_checkpoint, args.mixed_checkpoint):
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {checkpoint}")
    if args.eval_channels <= 0 or args.m_views <= 0:
        parser.error("eval-channels and m-views must be positive")
    if any(value <= 0 for value in args.morphing_ratios):
        parser.error("all morphing ratios must be positive")
    if any(value <= 0 for value in args.path_counts):
        parser.error("all path counts must be positive")
    return args


def load_checkpoint_model(
    path: Path,
    *,
    m_views: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    saved_m = int(checkpoint.get("M", m_views))
    if saved_m != m_views:
        raise ValueError(f"{path} contains M={saved_m}, expected M={m_views}")
    model = SFCNN(m_views).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def build_codebook(m_views: int, seed: int) -> tuple[list[np.ndarray], ...]:
    c = 3e8
    fc = 28e9
    wavelength = c / fc
    spacing = wavelength / 8
    np.random.seed(seed)
    p_b_all, zeta_b_all, p_u_all, zeta_u_all = [], [], [], []
    for _ in range(m_views):
        p_b, zeta_b, p_u, zeta_u = generateFIMsystem(
            5, 5, spacing, spacing, 5, 5, spacing, spacing, fc, flag=False
        )
        p_b_all.append(p_b)
        zeta_b_all.append(zeta_b)
        p_u_all.append(p_u)
        zeta_u_all.append(zeta_u)
    return p_b_all, zeta_b_all, p_u_all, zeta_u_all


def build_pair_batch(
    true_channel: np.ndarray,
    estimates: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_subcarriers = true_channel.shape[2]
    m_views = len(estimates)
    n_b, n_u = true_channel.shape[:2]
    count = k_subcarriers - 1
    model_input = np.zeros((count, 4 * m_views, n_b, n_u), dtype=np.float32)
    reference0 = np.zeros((count, n_b, n_u), dtype=np.complex128)
    reference1 = np.zeros_like(reference0)
    scales = np.zeros(count, dtype=np.float64)

    for pair in range(count):
        available = []
        for view in range(m_views):
            available.append(estimates[view][:, :, pair].reshape(-1))
            available.append(estimates[view][:, :, pair + 1].reshape(-1))
        scale = np.max(np.abs(np.concatenate(available))) + 1e-8
        scales[pair] = scale
        for view in range(m_views):
            r0 = estimates[view][:, :, pair] / scale
            r1 = estimates[view][:, :, pair + 1] / scale
            offset = 4 * view
            model_input[pair, offset + 0] = r0.real
            model_input[pair, offset + 1] = r0.imag
            model_input[pair, offset + 2] = r1.real
            model_input[pair, offset + 3] = r1.imag
        reference0[pair] = estimates[0][:, :, pair] / scale
        reference1[pair] = estimates[0][:, :, pair + 1] / scale
    return model_input, reference0, reference1, scales


def evaluate(
    models: dict[str, torch.nn.Module],
    *,
    ratios: list[float],
    snr_values: list[float],
    path_counts: list[int],
    eval_channels: int,
    m_views: int,
    codebook: tuple[list[np.ndarray], ...],
    evaluation_seed: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    n_b = n_u = 25
    fc = 28e9
    fs = 100e3
    k_subcarriers = 32
    wavelength = 3e8 / fc
    base_ratio = 0.02
    p_b_all, zeta_b_all, p_u_all, zeta_u_all = codebook
    pilots = generate_multi_pilots(n_u, 1)
    zeta_b_zero = np.zeros((3, n_b), dtype=float)
    zeta_u_zero = np.zeros((3, n_u), dtype=float)
    shape = (len(ratios), len(path_counts), len(snr_values))
    errors = {name: np.zeros(shape, dtype=np.float64) for name in models}

    for ratio_index, ratio in enumerate(ratios):
        morphing_scale = ratio / base_ratio
        for model in models.values():
            setter = getattr(model, "set_morphing_ratio", None)
            if setter is not None:
                setter(ratio)
        for path_index, path_count in enumerate(path_counts):
            for snr_index, snr_db in enumerate(snr_values):
                # Identical propagation and noise draws for every morphing ratio.
                np.random.seed(evaluation_seed + 1009 * path_count + 37 * snr_index)
                error_sum = {name: 0.0 for name in models}
                power_sum = 0.0
                for _ in range(eval_channels):
                    params = generate_path_parameters(path_count, fc, fs)
                    true_channel = build_H_fim_from_paths(
                        params,
                        p_b_all[0],
                        zeta_b_zero,
                        p_u_all[0],
                        zeta_u_zero,
                        wavelength,
                        fs,
                        k_subcarriers,
                    )
                    estimates = []
                    for view in range(m_views):
                        deformed = build_H_fim_from_paths(
                            params,
                            p_b_all[view],
                            zeta_b_all[view] * morphing_scale,
                            p_u_all[view],
                            zeta_u_all[view] * morphing_scale,
                            wavelength,
                            fs,
                            k_subcarriers,
                        )
                        received, sigma2, _ = pilot_transmission_multi(
                            deformed, n_b, n_u, pilots, snr_db, k_subcarriers
                        )
                        estimate = np.zeros((n_b, n_u, k_subcarriers), dtype=complex)
                        for tone in range(k_subcarriers):
                            estimate[:, :, tone] = ridge_TE_multipilot(
                                received[:, :, tone, :], pilots, lambda_r=sigma2
                            )
                        estimates.append(estimate)

                    batch, reference0, reference1, scales = build_pair_batch(
                        true_channel, estimates
                    )
                    tensor = torch.from_numpy(batch).to(device)
                    for name, model in models.items():
                        with torch.no_grad():
                            prediction = model(tensor).cpu().numpy()
                        delta0 = prediction[:, 0] + 1j * prediction[:, 1]
                        delta1 = prediction[:, 2] + 1j * prediction[:, 3]
                        estimate0 = scales[:, None, None] * (reference0 + delta0)
                        estimate1 = scales[:, None, None] * (reference1 + delta1)
                        for pair in range(k_subcarriers - 1):
                            error_sum[name] += np.linalg.norm(
                                estimate0[pair] - true_channel[:, :, pair]
                            ) ** 2
                            error_sum[name] += np.linalg.norm(
                                estimate1[pair] - true_channel[:, :, pair + 1]
                            ) ** 2
                    for pair in range(k_subcarriers - 1):
                        power_sum += np.linalg.norm(true_channel[:, :, pair]) ** 2
                        power_sum += np.linalg.norm(true_channel[:, :, pair + 1]) ** 2

                for name in models:
                    errors[name][ratio_index, path_index, snr_index] = (
                        error_sum[name] / (power_sum + 1e-12)
                    )
                    print(
                        f"model={name}, b/lambda={ratio:.3f}, L={path_count}, "
                        f"SNR={snr_db:g} dB, "
                        f"NMSE={errors[name][ratio_index, path_index, snr_index]:.6e}",
                        flush=True,
                    )
    return errors


def save_results(
    output: Path,
    errors: dict[str, np.ndarray],
    ratios: list[float],
    snr_values: list[float],
    path_counts: list[int],
    configuration: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for ratio_index, ratio in enumerate(ratios):
        for path_index, path_count in enumerate(path_counts):
            for snr_index, snr_db in enumerate(snr_values):
                row = {
                    "morphing_b_over_lambda": ratio,
                    "path_count": path_count,
                    "snr_db": snr_db,
                }
                for name, values in errors.items():
                    value = float(values[ratio_index, path_index, snr_index])
                    row[f"nmse_{name}"] = value
                    row[f"nmse_db_{name}"] = float(10 * np.log10(value))
                rows.append(row)
    with (output / "morphing_robustness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        output / "morphing_robustness_arrays.npz",
        morphing_ratios=np.asarray(ratios),
        snr_db=np.asarray(snr_values),
        path_counts=np.asarray(path_counts),
        **{f"nmse_{name}": values for name, values in errors.items()},
    )
    report = {"configuration": configuration, "results": rows}
    (output / "morphing_robustness_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    for path_index, path_count in enumerate(path_counts):
        fig, axes = plt.subplots(
            1, len(snr_values), figsize=(5 * len(snr_values), 4.5), squeeze=False
        )
        for snr_index, snr_db in enumerate(snr_values):
            axis = axes[0, snr_index]
            for name, values in errors.items():
                axis.semilogy(
                    ratios,
                    values[:, path_index, snr_index],
                    "-o",
                    markerfacecolor="white",
                    label=name,
                )
            axis.set_xscale("log")
            axis.set_xlabel("Morphing ratio b/lambda")
            axis.set_ylabel("NMSE")
            axis.set_title(f"L={path_count}, SNR={snr_db:g} dB")
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()
        fig.suptitle("SFCNN morphing-range robustness (evaluation only)")
        fig.tight_layout()
        fig.savefig(
            output / f"morphing_robustness_L{path_count}.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    load_core_definitions()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    fixed_model, fixed_checkpoint = load_checkpoint_model(
        args.fixed_checkpoint, m_views=args.m_views, device=device
    )
    mixed_model, mixed_checkpoint = load_checkpoint_model(
        args.mixed_checkpoint, m_views=args.m_views, device=device
    )
    models = {"fixed_training": fixed_model, "mixed_training": mixed_model}
    codebook = build_codebook(args.m_views, args.seed)
    ratios = list(args.morphing_ratios)
    snr_values = list(args.snr_db)
    path_counts = list(args.path_counts)
    errors = evaluate(
        models,
        ratios=ratios,
        snr_values=snr_values,
        path_counts=path_counts,
        eval_channels=args.eval_channels,
        m_views=args.m_views,
        codebook=codebook,
        evaluation_seed=args.evaluation_seed,
        device=device,
    )
    save_results(
        args.output,
        errors,
        ratios,
        snr_values,
        path_counts,
        {
            "inference_only": True,
            "m_views": args.m_views,
            "morphing_ratios": ratios,
            "snr_db": snr_values,
            "path_counts": path_counts,
            "evaluation_channels": args.eval_channels,
            "seed": args.seed,
            "evaluation_seed": args.evaluation_seed,
            "normalization": "observation_only",
            "fixed_checkpoint": str(args.fixed_checkpoint),
            "mixed_checkpoint": str(args.mixed_checkpoint),
            "fixed_checkpoint_training_morphing": fixed_checkpoint.get(
                "training_morphing_b_over_lambda", "legacy fixed 0.02"
            ),
            "mixed_checkpoint_training_morphing": mixed_checkpoint.get(
                "training_morphing_b_over_lambda", "mixed range"
            ),
        },
    )
    print("Results saved to:", args.output.resolve())


if __name__ == "__main__":
    main()
