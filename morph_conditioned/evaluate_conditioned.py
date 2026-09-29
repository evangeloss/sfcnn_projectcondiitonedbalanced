"""Run an evaluation-only morphing sweep for a conditioned SFCNN checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from morph_conditioned.conditioned_model import MorphConditionedSFCNN


def load_evaluation_helpers():
    path = PROJECT_ROOT / "loss_comparison" / "24_evaluate_morphing_robustness.py"
    spec = importlib.util.spec_from_file_location("morphing_evaluation_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RatioBoundConditionedModel(nn.Module):
    """Expose a one-input interface to the shared evaluation implementation."""

    def __init__(self, model: MorphConditionedSFCNN) -> None:
        super().__init__()
        self.model = model
        self.ratio = 0.02

    def set_morphing_ratio(self, ratio: float) -> None:
        self.ratio = float(ratio)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ratios = torch.full(
            (observations.shape[0],),
            self.ratio,
            dtype=observations.dtype,
            device=observations.device,
        )
        return self.model(observations, ratios)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--morphing-ratios",
        type=float,
        nargs="+",
        default=(0.01, 0.02, 0.10, 0.30, 0.50),
    )
    parser.add_argument("--snr-db", type=float, nargs="+", default=(0, 10, 20))
    parser.add_argument("--path-counts", type=int, nargs="+", default=(3,))
    parser.add_argument("--eval-channels", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--evaluation-seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "morph_conditioned_evaluation",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    return args


def main() -> None:
    args = parse_args()
    helpers = load_evaluation_helpers()
    helpers.load_core_definitions()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MorphConditionedSFCNN(m_views=8).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    wrapped = RatioBoundConditionedModel(model).to(device)
    codebook = helpers.build_codebook(8, args.seed)
    ratios = list(args.morphing_ratios)
    snr_values = list(args.snr_db)
    path_counts = list(args.path_counts)
    errors = helpers.evaluate(
        {"morph_conditioned": wrapped},
        ratios=ratios,
        snr_values=snr_values,
        path_counts=path_counts,
        eval_channels=args.eval_channels,
        m_views=8,
        codebook=codebook,
        evaluation_seed=args.evaluation_seed,
        device=device,
    )
    helpers.save_results(
        args.output,
        errors,
        ratios,
        snr_values,
        path_counts,
        {
            "inference_only": True,
            "architecture": "MorphConditionedSFCNN-FiLM",
            "m_views": 8,
            "morphing_ratios": ratios,
            "snr_db": snr_values,
            "path_counts": path_counts,
            "evaluation_channels": args.eval_channels,
            "seed": args.seed,
            "evaluation_seed": args.evaluation_seed,
            "checkpoint": str(args.checkpoint),
        },
    )
    print("Conditioned evaluation saved to:", args.output.resolve())


if __name__ == "__main__":
    main()
