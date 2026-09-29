"""Train and compare selectable SFCNN losses on the original experiment.

Run from the repository root:

    python loss_comparison/22_run_loss_comparison.py

The default experiment uses the original system/data settings, trains every
selected loss for 20 epochs, evaluates all models on identical seeded test
realizations, and writes checkpoints, arrays, CSV/JSON summaries, and plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_ROOT / "original_sfcnn"
sys.path.insert(0, str(ORIGINAL_DIR))

from losses import SFCNNLoss, VALID_LOSSES
from optimized_dataset import generate_optimized_multideformation_dataset


# Load only the reusable definitions required by this experiment.  Notebook
# examples, sanity plots, and the original long-running cell 22 are excluded.
CORE_FILES = [
    "01_path_parameter_generation.py",
    "03_deformed_array_geometry.py",
    "05_steering_vector.py",
    "06_fim_system_generation.py",
    "08_shared_path_channel_builder.py",
    "12_multi_pilot_generation.py",
    "13_pilot_transmission.py",
    "14_ridge_te_estimator.py",
    "15_multideformation_dataset.py",
    "17_sfcnn_model.py",
    "18_ls_evaluation.py",
    "20_cnn_evaluation.py",
    "21_cuda_timing_helper.py",
]


def load_core_definitions() -> None:
    namespace = globals()
    for filename in CORE_FILES:
        path = ORIGINAL_DIR / filename
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), namespace)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--losses",
        nargs="+",
        choices=VALID_LOSSES,
        default=list(VALID_LOSSES),
        help="Losses to compare. Defaults to every immediately usable loss.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-channels", type=int, default=1000)
    parser.add_argument("--val-channels", type=int, default=200)
    parser.add_argument("--eval-channels", type=int, default=50)
    parser.add_argument(
        "--m-views",
        type=int,
        default=1,
        help="Number of deformation views M. The model receives 4*M channels.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--pairs-per-channel",
        type=int,
        default=None,
        help=(
            "Store this many random adjacent pairs per independent channel. "
            "Omit to preserve the original all-pairs dataset."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=("auto", "target_assisted", "observation_only"),
        default="auto",
        help=(
            "Dataset normalization. 'auto' preserves the prior behavior: "
            "target-assisted for the original all-pairs generator and "
            "observation-only when --pairs-per-channel is supplied."
        ),
    )
    parser.add_argument(
        "--morphing-min",
        type=float,
        default=None,
        help="Minimum training b/lambda. Must be used with --morphing-max.",
    )
    parser.add_argument(
        "--morphing-max",
        type=float,
        default=None,
        help="Maximum training b/lambda. Must be used with --morphing-min.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "loss_comparison_results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke-test with 2 epochs and small train/validation/evaluation sets.",
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 2
        args.train_channels = 20
        args.val_channels = 8
        args.eval_channels = 3

    for name in (
        "epochs",
        "train_channels",
        "val_channels",
        "eval_channels",
        "m_views",
        "batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.pairs_per_channel is not None and not 1 <= args.pairs_per_channel <= 31:
        parser.error("--pairs-per-channel must be between 1 and 31")
    if (args.morphing_min is None) != (args.morphing_max is None):
        parser.error("--morphing-min and --morphing-max must be supplied together")
    if args.morphing_min is not None:
        if args.morphing_min <= 0 or args.morphing_max < args.morphing_min:
            parser.error("morphing range must satisfy 0 < minimum <= maximum")
    return args


def build_dataloader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_loss(
    loss_name: str,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    *,
    m_views: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, list[float]], float]:
    """Train one loss with identical initialization and batch order."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = build_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, seed=seed
    )

    model = SFCNN(m_views).to(device)
    criterion = SFCNNLoss(
        name=loss_name,
        residual_weight=0.10,
        cvar_weight=0.10,
        worst_fraction=0.20,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    gradient_clip = 1.0
    history = {"train": [], "validation": []}

    sync_if_cuda(device)
    start = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        for model_input, target_delta in train_loader:
            model_input = model_input.to(device, non_blocking=True)
            target_delta = target_delta.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(model_input)
            loss = criterion(prediction, target_delta, model_input)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_sum += loss.item() * model_input.size(0)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for model_input, target_delta in val_loader:
                model_input = model_input.to(device, non_blocking=True)
                target_delta = target_delta.to(device, non_blocking=True)
                prediction = model(model_input)
                loss = criterion(prediction, target_delta, model_input)
                val_sum += loss.item() * model_input.size(0)

        train_value = train_sum / len(train_dataset)
        val_value = val_sum / len(val_dataset)
        history["train"].append(train_value)
        history["validation"].append(val_value)
        print(
            f"[{loss_name}] epoch {epoch + 1:02d}/{epochs}: "
            f"train={train_value:.6e}, validation={val_value:.6e}",
            flush=True,
        )

    sync_if_cuda(device)
    duration = time.perf_counter() - start
    return model, history, duration


def plot_training_histories(
    histories: dict[str, dict[str, list[float]]],
    output: Path,
) -> None:
    count = len(histories)
    cols = 2
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.2 * rows), squeeze=False)
    for axis, (name, history) in zip(axes.flat, histories.items()):
        epochs = np.arange(1, len(history["train"]) + 1)
        axis.semilogy(epochs, history["train"], label="Training")
        axis.semilogy(epochs, history["validation"], label="Validation")
        axis.set(title=name, xlabel="Epoch", ylabel="Selected loss")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    fig.suptitle("SFCNN loss comparison: training histories", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_nmse_results(
    results: dict[str, np.ndarray],
    nmse_ls: np.ndarray,
    output: Path,
) -> None:
    snr_db = np.arange(-10, 25, 5)
    path_counts = [1, 2, 3, 4]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for index, (axis, path_count) in enumerate(zip(axes.flat, path_counts)):
        axis.semilogy(
            snr_db,
            nmse_ls[index],
            "k--*",
            linewidth=1.8,
            markersize=7,
            label=f"LS / TE, L={path_count}",
        )
        for loss_name, values in results.items():
            axis.semilogy(
                snr_db,
                values[index],
                "-o",
                linewidth=1.6,
                markersize=5,
                markerfacecolor="white",
                label=loss_name,
            )
        axis.set_title(f"Test paths: L={path_count}")
        axis.set_xticks(snr_db)
        axis.set_ylim(1e-4, 2e1)
        axis.grid(True, which="both", alpha=0.3)
        axis.set_xlabel("SNR (dB)")
        axis.set_ylabel("NMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5))
    fig.suptitle("SFCNN evaluation by training loss", fontsize=15)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    load_core_definitions()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Losses:", ", ".join(args.losses))
    print("Epochs:", args.epochs)
    print("Results directory:", args.output.resolve())

    # Original system parameters.
    c = 3e8
    fc = 28e9
    wavelength = c / fc
    fs = 100e3
    k_subcarriers = 32
    l_train = 3
    snr_train_set = np.array([0, 5, 10, 15, 20])
    nh_b = nv_b = nh_u = nv_u = 5
    n_b = nh_b * nv_b
    n_u = nh_u * nv_u
    spacing = wavelength / 8
    m_views = args.m_views
    t_pilots = 1

    # Generate the codebook and both datasets once.  Every loss therefore sees
    # exactly the same channel realizations, noise draws, and validation data.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    p_b_all, zeta_b_all, p_u_all, zeta_u_all = [], [], [], []
    for _ in range(m_views):
        p_b, zeta_b, p_u, zeta_u = generateFIMsystem(
            nh_b,
            nv_b,
            spacing,
            spacing,
            nh_u,
            nv_u,
            spacing,
            spacing,
            fc,
            flag=False,
        )
        p_b_all.append(p_b)
        zeta_b_all.append(zeta_b)
        p_u_all.append(p_u)
        zeta_u_all.append(zeta_u)

    use_original_generator = all(
        (
            args.pairs_per_channel is None,
            args.normalization == "auto",
            args.morphing_min is None,
        )
    )
    dataset_generator = (
        generate_Dataset_multiDef_multipilot
        if use_original_generator
        else generate_optimized_multideformation_dataset
    )
    dataset_options = {"Tpilots": t_pilots}
    actual_pairs_per_channel = (
        k_subcarriers - 1
        if args.pairs_per_channel is None
        else args.pairs_per_channel
    )
    if args.normalization != "auto":
        actual_normalization = args.normalization
    elif args.pairs_per_channel is None:
        actual_normalization = "target_assisted"
    else:
        actual_normalization = "observation_only"
    if not use_original_generator:
        dataset_options["pairs_per_channel"] = actual_pairs_per_channel
        dataset_options["normalization_mode"] = actual_normalization
        if args.morphing_min is not None:
            dataset_options["morphing_ratio_range"] = (
                args.morphing_min,
                args.morphing_max,
            )

    print("Pairs per channel:", actual_pairs_per_channel)
    print("Normalization:", actual_normalization)
    print("Deformation views (M):", m_views)
    print("Model input channels (4*M):", 4 * m_views)
    if args.morphing_min is None:
        print("Training morphing b/lambda: fixed at 0.02")
    else:
        print(
            "Training morphing b/lambda: uniform in "
            f"[{args.morphing_min}, {args.morphing_max}]"
        )
    print("Evaluation morphing b/lambda: fixed at 0.02")

    print("Generating one shared training dataset...", flush=True)
    x_train, y_train = dataset_generator(
        n_b,
        n_u,
        fc,
        fs,
        l_train,
        k_subcarriers,
        args.train_channels,
        snr_train_set,
        p_b_all,
        zeta_b_all,
        p_u_all,
        zeta_u_all,
        m_views,
        **dataset_options,
    )
    print("Generating one shared validation dataset...", flush=True)
    x_val, y_val = dataset_generator(
        n_b,
        n_u,
        fc,
        fs,
        l_train,
        k_subcarriers,
        args.val_channels,
        snr_train_set,
        p_b_all,
        zeta_b_all,
        p_u_all,
        zeta_u_all,
        m_views,
        **dataset_options,
    )

    train_dataset = TensorDataset(
        torch.from_numpy(np.transpose(x_train, (3, 2, 0, 1))),
        torch.from_numpy(np.transpose(y_train, (3, 2, 0, 1))),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(np.transpose(x_val, (3, 2, 0, 1))),
        torch.from_numpy(np.transpose(y_val, (3, 2, 0, 1))),
    )
    del x_train, y_train, x_val, y_val

    models: dict[str, torch.nn.Module] = {}
    histories: dict[str, dict[str, list[float]]] = {}
    training_seconds: dict[str, float] = {}

    for loss_name in args.losses:
        print(f"\n{'=' * 72}\nTraining loss: {loss_name}\n{'=' * 72}")
        model, history, duration = train_one_loss(
            loss_name,
            train_dataset,
            val_dataset,
            m_views=m_views,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
        )
        models[loss_name] = model
        histories[loss_name] = history
        training_seconds[loss_name] = duration

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "loss_name": loss_name,
            "epochs": args.epochs,
            "L_train": l_train,
            "M": m_views,
            "N_B": n_b,
            "N_U": n_u,
            "K": k_subcarriers,
            "fc": fc,
            "fs": fs,
            "pairs_per_channel": actual_pairs_per_channel,
            "normalization": actual_normalization,
            "training_morphing_b_over_lambda": (
                0.02
                if args.morphing_min is None
                else [args.morphing_min, args.morphing_max]
            ),
            "training_history": history,
            "training_seconds": duration,
        }
        torch.save(
            checkpoint,
            args.output / f"sfcnn_{loss_name}_{args.epochs}epoch.pt",
        )

    plot_training_histories(
        histories, args.output / "loss_training_histories.png"
    )

    # One LS baseline and identical evaluation randomness for every model.
    print("\nEvaluating LS / TE baseline...", flush=True)
    np.random.seed(args.evaluation_seed)
    nmse_ls = NMSE_LSevaluation(
        n_b,
        n_u,
        fc,
        fs,
        k_subcarriers,
        args.eval_channels,
        p_b_all,
        zeta_b_all,
        p_u_all,
        zeta_u_all,
        pilotPow=1,
    )

    nmse_results: dict[str, np.ndarray] = {}
    inference_timing: dict[str, dict[str, float]] = {}
    for loss_name, model in models.items():
        print(f"\nEvaluating {loss_name} on the common test seed...", flush=True)
        np.random.seed(args.evaluation_seed)
        nmse, avg_time, std_time = NMSE_CNNevaluation2(
            n_b,
            n_u,
            fc,
            fs,
            k_subcarriers,
            args.eval_channels,
            p_b_all,
            zeta_b_all,
            p_u_all,
            zeta_u_all,
            m_views,
            model,
            device=device,
        )
        nmse_results[loss_name] = nmse
        inference_timing[loss_name] = {
            "average_seconds": float(avg_time),
            "std_seconds": float(std_time),
        }

    plot_nmse_results(
        nmse_results,
        nmse_ls,
        args.output / "loss_comparison_nmse.png",
    )

    rows = []
    for loss_name, values in nmse_results.items():
        rows.append(
            {
                "loss": loss_name,
                "mean_nmse_linear": float(values.mean()),
                "mean_nmse_db": float(10 * np.log10(values.mean())),
                "mean_pointwise_nmse_db": float(np.mean(10 * np.log10(values))),
                "l3_mean_nmse_linear": float(values[2].mean()),
                "l3_mean_nmse_db": float(10 * np.log10(values[2].mean())),
                "final_validation_objective": float(
                    histories[loss_name]["validation"][-1]
                ),
                "training_seconds": float(training_seconds[loss_name]),
                "inference_ms": 1000
                * inference_timing[loss_name]["average_seconds"],
            }
        )
    rows.sort(key=lambda row: row["mean_nmse_linear"])

    with (args.output / "loss_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.output / "loss_comparison_arrays.npz",
        nmse_ls=nmse_ls,
        **{f"nmse_{name}": value for name, value in nmse_results.items()},
    )

    report = {
        "configuration": {
            "losses": args.losses,
            "epochs": args.epochs,
            "train_channels": args.train_channels,
            "validation_channels": args.val_channels,
            "evaluation_channels": args.eval_channels,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "pairs_per_channel": actual_pairs_per_channel,
            "normalization": actual_normalization,
            "observation_only_normalization": (
                actual_normalization == "observation_only"
            ),
            "seed": args.seed,
            "evaluation_seed": args.evaluation_seed,
            "fc": fc,
            "fs": fs,
            "K": k_subcarriers,
            "M": m_views,
            "L_train": l_train,
            "SNR_train_set": snr_train_set.tolist(),
            "training_morphing_b_over_lambda": (
                0.02
                if args.morphing_min is None
                else [args.morphing_min, args.morphing_max]
            ),
            "training_morphing_distribution": (
                "fixed" if args.morphing_min is None else "uniform"
            ),
            "evaluation_morphing_b_over_lambda": 0.02,
        },
        "ranking_metric": "mean NMSE over all 4 path counts and 7 SNR points",
        "ranking": rows,
        "inference_timing": inference_timing,
    }
    (args.output / "loss_comparison_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\nLoss ranking (lower mean NMSE is better):")
    for index, row in enumerate(rows, start=1):
        print(
            f"{index}. {row['loss']}: "
            f"mean NMSE={row['mean_nmse_linear']:.6e} "
            f"({row['mean_nmse_db']:.2f} dB)"
        )
    print(f"\nBest loss: {rows[0]['loss']}")
    print("Results saved to:", args.output.resolve())


if __name__ == "__main__":
    main()
