"""Clone, train, evaluate, and package the balanced conditioned SFCNN on Kaggle.

Kaggle setup:
1. Enable a GPU and Internet.
2. Add the old M=8 mixed-morphing result ZIP as a notebook input.
3. Paste this complete file into one Kaggle cell and run it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_URL = os.environ.get(
    "SFCNN_CONDITIONED_REPO_URL",
    "https://github.com/evangeloss/sfcnn_loss_comparison.git",
)
BRANCH = os.environ.get("SFCNN_BRANCH", "main")
CLONE_DIR = Path("/kaggle/working/sfcnn_balanced_conditioned_repository")
DISCOVERY_DIR = Path("/kaggle/working/mixed_checkpoint_input")
RESULTS_DIR = Path("/kaggle/working/morph_conditioned_balanced_output")
ARCHIVE_BASE = Path("/kaggle/working/morph_conditioned_balanced_results")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_or_update() -> None:
    if not CLONE_DIR.exists():
        run([
            "git", "clone", "--branch", BRANCH, "--single-branch",
            REPO_URL, str(CLONE_DIR),
        ])
        return
    if not (CLONE_DIR / ".git").is_dir():
        raise RuntimeError(f"{CLONE_DIR} exists but is not a Git repository")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=CLONE_DIR,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("The Kaggle checkout contains local changes.")
    run(["git", "fetch", "origin", BRANCH], cwd=CLONE_DIR)
    run(["git", "checkout", BRANCH], cwd=CLONE_DIR)
    run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=CLONE_DIR)


def locate_project() -> Path:
    candidates = (
        CLONE_DIR,
        CLONE_DIR / "sfcnn_morph_conditioned_project",
    )
    for candidate in candidates:
        trainer = candidate / "morph_conditioned" / "train_conditioned_balanced.py"
        if trainer.is_file():
            return candidate
    raise FileNotFoundError(
        "The repository does not contain "
        "morph_conditioned/train_conditioned_balanced.py. Commit and push the "
        "new project files to GitHub, then start a fresh Kaggle session."
    )


def extract_result_zips() -> None:
    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        for path in root.rglob("*.zip"):
            try:
                with zipfile.ZipFile(path) as archive:
                    if not any(
                        name.endswith("loss_comparison_report.json")
                        for name in archive.namelist()
                    ):
                        continue
                    destination = DISCOVERY_DIR / path.stem
                    destination.mkdir(parents=True, exist_ok=True)
                    archive.extractall(destination)
            except zipfile.BadZipFile:
                continue


def discover_mixed_checkpoint() -> Path:
    extract_result_zips()
    reports = []
    for root in (Path("/kaggle/input"), DISCOVERY_DIR):
        if root.exists():
            reports.extend(root.rglob("loss_comparison_report.json"))
    for report_path in reports:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            configuration = report["configuration"]
            training_range = configuration.get("training_morphing_b_over_lambda")
        except (OSError, KeyError, json.JSONDecodeError):
            continue
        if int(configuration.get("M", -1)) != 8:
            continue
        if not isinstance(training_range, list) or len(training_range) != 2:
            continue
        checkpoints = sorted(
            report_path.parent.glob("sfcnn_channel_nmse_cvar_*epoch.pt")
        )
        if checkpoints:
            print("Starting checkpoint:", checkpoints[-1])
            return checkpoints[-1]
    raise FileNotFoundError(
        "Could not find the old mixed-morphing M=8 checkpoint. Add "
        "sfcnn_m8_mixed_morphing_results.zip as a Kaggle input."
    )


def main() -> None:
    clone_or_update()
    project = locate_project()
    requirements = project / "requirements.txt"
    if requirements.is_file():
        run([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "-r",
            str(requirements),
        ])

    checkpoint = discover_mixed_checkpoint()
    trainer = project / "morph_conditioned" / "train_conditioned_balanced.py"
    run([
        sys.executable,
        str(trainer),
        "--checkpoint", str(checkpoint),
        "--epochs", "40",
        "--conditioning-only-epochs", "4",
        "--train-channels", "6000",
        "--val-channels-per-cell", "20",
        "--pairs-per-channel", "8",
        "--batch-size", "128",
        "--warmup-learning-rate", "0.0002",
        "--conditioning-learning-rate", "0.0001",
        "--base-learning-rate", "0.00002",
        "--weight-decay", "0.00001",
        "--anchor-ratio", "0.02",
        "--anchor-probability", "0.20",
        "--morphing-bin-edges",
        "0.01", "0.05", "0.10", "0.20", "0.30", "0.40", "0.50",
        "--validation-ratios",
        "0.01", "0.02", "0.05", "0.10", "0.15", "0.20", "0.30", "0.40", "0.50",
        "--snr-db", "-10", "-5", "0", "5", "10", "15", "20",
        "--range-cvar-weight", "0.15",
        "--residual-weight", "0.02",
        "--checkpoint-worst-weight", "0.25",
        "--seed", "1",
        "--output", str(RESULTS_DIR),
    ], cwd=project)

    trained_checkpoint = RESULTS_DIR / "morph_conditioned_balanced_best.pt"
    evaluator = project / "morph_conditioned" / "evaluate_conditioned.py"
    run([
        sys.executable,
        str(evaluator),
        "--checkpoint", str(trained_checkpoint),
        "--morphing-ratios",
        "0.01", "0.02", "0.05", "0.10", "0.15", "0.20", "0.30", "0.40", "0.50",
        "--snr-db", "-10", "-5", "0", "5", "10", "15", "20",
        "--path-counts", "3",
        "--eval-channels", "50",
        "--seed", "1",
        "--evaluation-seed", "2026",
        "--output", str(RESULTS_DIR / "evaluation"),
    ], cwd=project)

    archive = shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=RESULTS_DIR)
    print("\nBalanced conditioned training and evaluation finished.")
    print("Results folder:", RESULTS_DIR)
    print("Download this file:", archive)


if __name__ == "__main__":
    main()
