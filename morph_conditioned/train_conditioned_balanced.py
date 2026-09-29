"""Range-balanced fine-tuning for the M=8 FiLM-conditioned SFCNN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_ROOT / "original_sfcnn"
LOSS_DIR = PROJECT_ROOT / "loss_comparison"
sys.path[:0] = [str(PROJECT_ROOT), str(ORIGINAL_DIR), str(LOSS_DIR)]

from losses import per_sample_channel_nmse
from optimized_dataset import generate_optimized_multideformation_dataset
from morph_conditioned.conditioned_model import (
    MorphConditionedSFCNN,
    load_original_sfcnn_weights,
)
from morph_conditioned.train_conditioned import (
    build_codebook,
    load_checkpoint,
    load_core_definitions,
    verify_weight_transfer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--conditioning-only-epochs", type=int, default=4)
    parser.add_argument("--train-channels", type=int, default=6000)
    parser.add_argument("--val-channels-per-cell", type=int, default=20)
    parser.add_argument("--pairs-per-channel", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--conditioning-learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--anchor-ratio", type=float, default=0.02)
    parser.add_argument("--anchor-probability", type=float, default=0.20)
    parser.add_argument(
        "--morphing-bin-edges",
        type=float,
        nargs="+",
        default=(0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50),
    )
    parser.add_argument(
        "--validation-ratios",
        type=float,
        nargs="+",
        default=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50),
    )
    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=(-10, -5, 0, 5, 10, 15, 20),
    )
    parser.add_argument("--range-cvar-weight", type=float, default=0.15)
    parser.add_argument("--residual-weight", type=float, default=0.02)
    parser.add_argument("--checkpoint-worst-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "morph_conditioned_balanced_results",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.quick:
        args.epochs = 2
        args.conditioning_only_epochs = 1
        args.train_channels = 24
        args.val_channels_per_cell = 1
        args.pairs_per_channel = 2
    edges = np.asarray(args.morphing_bin_edges)
    if edges.size < 2 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        parser.error("morphing-bin-edges must be positive and increasing")
    if not 0 <= args.conditioning_only_epochs <= args.epochs:
        parser.error("conditioning-only-epochs must be between 0 and epochs")
    if not 0 <= args.anchor_probability <= 1:
        parser.error("anchor-probability must be in [0, 1]")
    return args


def to_tensor_dataset(x: np.ndarray, y: np.ndarray, ratios: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.transpose(x, (3, 2, 0, 1))),
        torch.from_numpy(np.transpose(y, (3, 2, 0, 1))),
        torch.from_numpy(ratios.astype(np.float32, copy=False)),
    )


def generate_one_dataset(
    channel_count: int,
    *,
    args: argparse.Namespace,
    codebook,
    snr_values,
    morphing_range,
    anchor_probability: float,
    morphing_bin_edges=None,
) -> TensorDataset:
    p_b_all, zeta_b_all, p_u_all, zeta_u_all = codebook
    x, y, metadata = generate_optimized_multideformation_dataset(
        25,
        25,
        28e9,
        100e3,
        3,
        32,
        channel_count,
        np.asarray(snr_values, dtype=float),
        p_b_all,
        zeta_b_all,
        p_u_all,
        zeta_u_all,
        8,
        pairs_per_channel=args.pairs_per_channel,
        Tpilots=1,
        normalization_mode="observation_only",
        morphing_ratio_range=morphing_range,
        morphing_anchor_ratio=args.anchor_ratio,
        morphing_anchor_probability=anchor_probability,
        morphing_bin_edges=morphing_bin_edges,
        return_metadata=True,
    )
    return to_tensor_dataset(x, y, metadata["sample_morphing_ratio_b_over_lambda"])


def build_validation_dataset(args: argparse.Namespace, codebook) -> ConcatDataset:
    cells = []
    for ratio in args.validation_ratios:
        for snr in args.snr_db:
            cells.append(
                generate_one_dataset(
                    args.val_channels_per_cell,
                    args=args,
                    codebook=codebook,
                    snr_values=[snr],
                    morphing_range=(ratio, ratio),
                    anchor_probability=0.0,
                )
            )
    return ConcatDataset(cells)


def morph_bin_ids(ratios: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    # Anchor samples at 0.02 naturally belong to the first interval.
    return torch.bucketize(ratios.contiguous(), edges[1:-1], right=False)


def range_balanced_loss(
    prediction,
    target,
    observations,
    ratios,
    edges,
    *,
    range_weight: float,
    residual_weight: float,
):
    sample_nmse = per_sample_channel_nmse(prediction, target, observations)
    ids = morph_bin_ids(ratios, edges)
    bin_means = torch.stack(
        [sample_nmse[ids == bin_id].mean() for bin_id in torch.unique(ids)]
    )
    worst_count = max(1, int(np.ceil(0.34 * bin_means.numel())))
    range_cvar = torch.topk(bin_means, k=worst_count).values.mean()
    residual = F.mse_loss(prediction, target)
    return sample_nmse.mean() + range_weight * range_cvar + residual_weight * residual


def set_base_trainable(model: MorphConditionedSFCNN, trainable: bool) -> None:
    for parameter in model.base_parameters():
        parameter.requires_grad = trainable


def main() -> None:
    args = parse_args()
    load_core_definitions()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = load_checkpoint(args.checkpoint, device)
    if int(checkpoint.get("M", 8)) != 8:
        raise ValueError("the starting checkpoint must use M=8")
    model = MorphConditionedSFCNN(m_views=8).to(device)
    load_original_sfcnn_weights(model, checkpoint["model_state_dict"])
    transfer_error = verify_weight_transfer(model, checkpoint["model_state_dict"], device)
    print("Device:", device)
    print(f"Weight-transfer verification max error: {transfer_error:.3e}")

    codebook = build_codebook(args.seed)
    edges = np.asarray(args.morphing_bin_edges, dtype=float)
    print("Generating range-balanced training data...", flush=True)
    train_dataset = generate_one_dataset(
        args.train_channels,
        args=args,
        codebook=codebook,
        snr_values=args.snr_db,
        morphing_range=(float(edges[0]), float(edges[-1])),
        anchor_probability=args.anchor_probability,
        morphing_bin_edges=edges,
    )
    print("Generating fixed ratio/SNR validation grid...", flush=True)
    val_dataset = build_validation_dataset(args, codebook)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    set_base_trainable(model, args.conditioning_only_epochs == 0)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.conditioning_parameters()),
                "lr": args.warmup_learning_rate,
                "name": "conditioning",
            },
            {
                "params": list(model.base_parameters()),
                "lr": args.base_learning_rate,
                "name": "base",
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    edge_tensor = torch.tensor(edges, dtype=torch.float32, device=device)
    best_score = float("inf")
    best_state = None
    history = {"train_objective": [], "validation_nmse": [], "checkpoint_score": []}
    start = time.perf_counter()

    for epoch in range(args.epochs):
        if epoch == args.conditioning_only_epochs:
            set_base_trainable(model, True)
            optimizer.param_groups[0]["lr"] = args.conditioning_learning_rate
            optimizer.param_groups[1]["lr"] = args.base_learning_rate
            print("Unfroze CNN base with its smaller learning rate.", flush=True)

        model.train()
        if epoch < args.conditioning_only_epochs:
            for normalization in model.normalizations:
                normalization.eval()
        train_sum = 0.0
        for observations, target, ratios in train_loader:
            observations = observations.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            ratios = ratios.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observations, ratios)
            loss = range_balanced_loss(
                prediction,
                target,
                observations,
                ratios,
                edge_tensor,
                range_weight=args.range_cvar_weight,
                residual_weight=args.residual_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item() * observations.size(0)

        model.eval()
        total_nmse = 0.0
        total_samples = 0
        bin_sums = np.zeros(len(edges) - 1, dtype=float)
        bin_counts = np.zeros(len(edges) - 1, dtype=int)
        with torch.no_grad():
            for observations, target, ratios in val_loader:
                observations = observations.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                ratios = ratios.to(device, non_blocking=True)
                prediction = model(observations, ratios)
                sample_nmse = per_sample_channel_nmse(prediction, target, observations)
                ids = morph_bin_ids(ratios, edge_tensor)
                total_nmse += sample_nmse.sum().item()
                total_samples += sample_nmse.numel()
                for bin_id in range(len(bin_sums)):
                    selected = ids == bin_id
                    if selected.any():
                        bin_sums[bin_id] += sample_nmse[selected].sum().item()
                        bin_counts[bin_id] += int(selected.sum().item())

        mean_nmse = total_nmse / total_samples
        bin_means = bin_sums / np.maximum(bin_counts, 1)
        score = mean_nmse + args.checkpoint_worst_weight * float(bin_means.max())
        scheduler.step(score)
        train_value = train_sum / len(train_dataset)
        history["train_objective"].append(train_value)
        history["validation_nmse"].append(mean_nmse)
        history["checkpoint_score"].append(score)
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch + 1:02d}/{args.epochs}: train={train_value:.6e}, "
            f"val_nmse={mean_nmse:.6e}, robust_score={score:.6e}, "
            f"worst_bin={bin_means.max():.6e}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    training_seconds = time.perf_counter() - start
    checkpoint_path = args.output / "morph_conditioned_balanced_best.pt"
    payload = {
        "model_state_dict": best_state,
        "architecture": "MorphConditionedSFCNN-FiLM-balanced",
        "M": 8,
        "epochs": args.epochs,
        "conditioning_only_epochs": args.conditioning_only_epochs,
        "training_morphing_b_over_lambda": [float(edges[0]), float(edges[-1])],
        "morphing_bin_edges": edges.tolist(),
        "anchor_ratio": args.anchor_ratio,
        "anchor_probability": args.anchor_probability,
        "snr_db": list(args.snr_db),
        "normalization": "observation_only",
        "best_robust_validation_score": best_score,
        "history": history,
        "source_checkpoint": str(args.checkpoint),
    }
    torch.save(payload, checkpoint_path)
    report = {
        "configuration": vars(args) | {"checkpoint": str(args.checkpoint), "output": str(args.output)},
        "weight_transfer_max_error": transfer_error,
        "best_robust_validation_score": best_score,
        "training_seconds": training_seconds,
        "history": history,
    }
    (args.output / "balanced_training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    epochs = np.arange(1, len(history["train_objective"]) + 1)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(epochs, history["train_objective"], label="Training objective")
    axis.semilogy(epochs, history["validation_nmse"], label="Validation NMSE")
    axis.semilogy(epochs, history["checkpoint_score"], label="Robust checkpoint score")
    axis.set(xlabel="Epoch", ylabel="Loss", title="Range-balanced conditioned training")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "balanced_conditioned_training.png", dpi=180)
    plt.close(fig)
    print("Saved checkpoint:", checkpoint_path)
    print("Training seconds:", training_seconds)


if __name__ == "__main__":
    main()
