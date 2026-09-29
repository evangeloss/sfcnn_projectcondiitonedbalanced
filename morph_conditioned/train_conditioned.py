"""Fine-tune a FiLM-conditioned M=8 SFCNN from an existing checkpoint."""

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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = PROJECT_ROOT / "original_sfcnn"
LOSS_DIR = PROJECT_ROOT / "loss_comparison"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ORIGINAL_DIR))
sys.path.insert(0, str(LOSS_DIR))

from losses import SFCNNLoss
from optimized_dataset import generate_optimized_multideformation_dataset
from morph_conditioned.conditioned_model import (
    MorphConditionedSFCNN,
    load_original_sfcnn_weights,
)


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--conditioning-only-epochs", type=int, default=3)
    parser.add_argument("--train-channels", type=int, default=4000)
    parser.add_argument("--val-channels", type=int, default=800)
    parser.add_argument("--pairs-per-channel", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--morphing-min", type=float, default=0.01)
    parser.add_argument("--morphing-max", type=float, default=0.5)
    parser.add_argument("--anchor-ratio", type=float, default=0.02)
    parser.add_argument("--anchor-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "morph_conditioned_results",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if args.quick:
        args.epochs = 2
        args.conditioning_only_epochs = 1
        args.train_channels = 12
        args.val_channels = 6
        args.pairs_per_channel = 2
    if args.epochs <= 0 or args.train_channels <= 0 or args.val_channels <= 0:
        parser.error("epochs and channel counts must be positive")
    if not 0 <= args.conditioning_only_epochs <= args.epochs:
        parser.error("conditioning-only-epochs must be between 0 and epochs")
    if not 1 <= args.pairs_per_channel <= 31:
        parser.error("pairs-per-channel must be between 1 and 31")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if not 0 < args.morphing_min <= args.morphing_max:
        parser.error("morphing range must satisfy 0 < minimum <= maximum")
    if not 0 <= args.anchor_probability <= 1:
        parser.error("anchor-probability must be in [0, 1]")
    if args.anchor_ratio <= 0:
        parser.error("anchor-ratio must be positive")
    return args


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def verify_weight_transfer(
    model: MorphConditionedSFCNN,
    original_state: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    original = SFCNN(8).to(device)
    original.load_state_dict(original_state)
    original.eval()
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(123)
    sample = torch.randn(2, 32, 25, 25, generator=generator).to(device)
    ratio = torch.full((2,), 0.02, device=device)
    with torch.no_grad():
        difference = (original(sample) - model(sample, ratio)).abs().max().item()
    if difference > 1e-6:
        raise RuntimeError(f"weight-transfer verification failed: max error={difference}")
    return difference


def build_codebook(seed: int, m_views: int = 8) -> tuple[list[np.ndarray], ...]:
    fc = 28e9
    wavelength = 3e8 / fc
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


def generate_dataset(
    channel_count: int,
    *,
    args: argparse.Namespace,
    codebook: tuple[list[np.ndarray], ...],
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
        np.asarray([0, 5, 10, 15, 20]),
        p_b_all,
        zeta_b_all,
        p_u_all,
        zeta_u_all,
        8,
        pairs_per_channel=args.pairs_per_channel,
        Tpilots=1,
        normalization_mode="observation_only",
        morphing_ratio_range=(args.morphing_min, args.morphing_max),
        morphing_anchor_ratio=args.anchor_ratio,
        morphing_anchor_probability=args.anchor_probability,
        return_metadata=True,
    )
    ratios = metadata["sample_morphing_ratio_b_over_lambda"]
    return TensorDataset(
        torch.from_numpy(np.transpose(x, (3, 2, 0, 1))),
        torch.from_numpy(np.transpose(y, (3, 2, 0, 1))),
        torch.from_numpy(ratios),
    )


def build_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def set_base_trainable(model: MorphConditionedSFCNN, trainable: bool) -> None:
    for parameter in model.base_parameters():
        parameter.requires_grad = trainable


def main() -> None:
    args = parse_args()
    load_core_definitions()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("This is fine-tuning; the original mixed checkpoint is the starting point.")

    checkpoint = load_checkpoint(args.checkpoint, device)
    if int(checkpoint.get("M", 8)) != 8:
        raise ValueError("the starting checkpoint must use M=8")
    model = MorphConditionedSFCNN(m_views=8).to(device)
    load_original_sfcnn_weights(model, checkpoint["model_state_dict"])
    transfer_error = verify_weight_transfer(model, checkpoint["model_state_dict"], device)
    print(f"Weight-transfer verification max error: {transfer_error:.3e}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    codebook = build_codebook(args.seed)
    print("Generating anchored mixed-range training dataset...", flush=True)
    train_dataset = generate_dataset(args.train_channels, args=args, codebook=codebook)
    print("Generating anchored mixed-range validation dataset...", flush=True)
    val_dataset = generate_dataset(args.val_channels, args=args, codebook=codebook)
    train_loader = build_loader(
        train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = build_loader(
        val_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed
    )

    criterion = SFCNNLoss(
        name="channel_nmse_cvar",
        cvar_weight=0.10,
        worst_fraction=0.20,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {"train": [], "validation": []}
    best_validation = float("inf")
    best_state = None
    set_base_trainable(model, args.conditioning_only_epochs == 0)
    start = time.perf_counter()

    for epoch in range(args.epochs):
        if epoch == args.conditioning_only_epochs and epoch > 0:
            set_base_trainable(model, True)
            print("Unfroze the transferred SFCNN base network.", flush=True)

        model.train()
        if epoch < args.conditioning_only_epochs:
            for normalization in model.normalizations:
                normalization.eval()
        train_sum = 0.0
        for observations, target, ratio in train_loader:
            observations = observations.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            ratio = ratio.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observations, ratio)
            loss = criterion(prediction, target, observations)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item() * observations.size(0)

        model.eval()
        validation_sum = 0.0
        with torch.no_grad():
            for observations, target, ratio in val_loader:
                observations = observations.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                ratio = ratio.to(device, non_blocking=True)
                prediction = model(observations, ratio)
                loss = criterion(prediction, target, observations)
                validation_sum += loss.item() * observations.size(0)

        train_value = train_sum / len(train_dataset)
        validation_value = validation_sum / len(val_dataset)
        history["train"].append(train_value)
        history["validation"].append(validation_value)
        if validation_value < best_validation:
            best_validation = validation_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch {epoch + 1:02d}/{args.epochs}: "
            f"train={train_value:.6e}, validation={validation_value:.6e}",
            flush=True,
        )

    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    output_checkpoint = args.output / "morph_conditioned_sfcnn_best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": "MorphConditionedSFCNN-FiLM",
            "M": 8,
            "input_channels": 32,
            "epochs": args.epochs,
            "conditioning_only_epochs": args.conditioning_only_epochs,
            "learning_rate": args.learning_rate,
            "training_morphing_b_over_lambda": [
                args.morphing_min,
                args.morphing_max,
            ],
            "anchor_ratio": args.anchor_ratio,
            "anchor_probability": args.anchor_probability,
            "normalization": "observation_only",
            "loss": "channel_nmse_cvar",
            "history": history,
            "best_validation": best_validation,
            "source_checkpoint": str(args.checkpoint),
        },
        output_checkpoint,
    )

    epochs = np.arange(1, args.epochs + 1)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.semilogy(epochs, history["train"], label="Training")
    axis.semilogy(epochs, history["validation"], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Channel NMSE + CVaR", title="Morph-conditioned fine-tuning")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / "morph_conditioned_training.png", dpi=180)
    plt.close(fig)

    report = {
        "architecture": "FiLM-conditioned residual SFCNN",
        "configuration": {
            "M": 8,
            "epochs": args.epochs,
            "conditioning_only_epochs": args.conditioning_only_epochs,
            "train_channels": args.train_channels,
            "validation_channels": args.val_channels,
            "pairs_per_channel": args.pairs_per_channel,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "morphing_uniform_range": [args.morphing_min, args.morphing_max],
            "anchor_ratio": args.anchor_ratio,
            "anchor_probability": args.anchor_probability,
            "normalization": "observation_only",
            "loss": "channel_nmse_cvar",
            "seed": args.seed,
        },
        "weight_transfer_max_error": transfer_error,
        "best_validation": best_validation,
        "training_seconds": training_seconds,
        "history": history,
    }
    (args.output / "morph_conditioned_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Best validation objective:", best_validation)
    print("Saved checkpoint:", output_checkpoint)
    print("Results directory:", args.output.resolve())


if __name__ == "__main__":
    main()
