"""Clone the conditioned-SFCNN project and fine-tune it on Kaggle.

Before running, add the mixed-morphing M=8 result ZIP (or its unpacked files)
as a Kaggle input. Set REPO_URL to the GitHub repository containing this
project, enable Internet and a GPU, and run this file in one notebook cell.
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
    "https://github.com/YOUR_USERNAME/YOUR_CONDITIONED_REPOSITORY.git",
)
BRANCH = os.environ.get("SFCNN_BRANCH", "main")
CLONE_DIR = Path("/kaggle/working/sfcnn_morph_conditioned_repository")
DISCOVERY_DIR = Path("/kaggle/working/conditioned_checkpoint_input")
RESULTS_DIR = Path("/kaggle/working/morph_conditioned_output")
ARCHIVE_BASE = Path("/kaggle/working/morph_conditioned_results")


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_or_update() -> None:
    if "YOUR_USERNAME" in REPO_URL or "YOUR_CONDITIONED_REPOSITORY" in REPO_URL:
        raise ValueError("Set REPO_URL to the GitHub repository for this project.")
    if not CLONE_DIR.exists():
        run([
            "git", "clone", "--branch", BRANCH, "--single-branch",
            REPO_URL, str(CLONE_DIR),
        ])
        return
    if not (CLONE_DIR / ".git").is_dir():
        raise RuntimeError(f"{CLONE_DIR} is not a Git repository")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=CLONE_DIR,
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("The Kaggle checkout contains local changes.")
    run(["git", "fetch", "origin", BRANCH], cwd=CLONE_DIR)
    run(["git", "checkout", BRANCH], cwd=CLONE_DIR)
    run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=CLONE_DIR)


def locate_project() -> Path:
    candidates = (CLONE_DIR, CLONE_DIR / "sfcnn_morph_conditioned_project")
    for candidate in candidates:
        if (candidate / "morph_conditioned" / "train_conditioned.py").is_file():
            return candidate
    raise FileNotFoundError("Could not locate morph_conditioned/train_conditioned.py")


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
        checkpoints = sorted(report_path.parent.glob("sfcnn_channel_nmse_cvar_*epoch.pt"))
        if checkpoints:
            print("Starting mixed checkpoint:", checkpoints[-1])
            return checkpoints[-1]
    raise FileNotFoundError(
        "Could not find the mixed-morphing M=8 checkpoint. Add "
        "sfcnn_m8_mixed_morphing_results.zip as a Kaggle input."
    )


def main() -> None:
    clone_or_update()
    project = locate_project()
    requirements = project / "requirements.txt"
    if requirements.is_file():
        run([
            sys.executable, "-m", "pip", "install", "--quiet",
            "-r", str(requirements),
        ])
    checkpoint = discover_mixed_checkpoint()
    trainer = project / "morph_conditioned" / "train_conditioned.py"
    run([
        sys.executable,
        str(trainer),
        "--checkpoint", str(checkpoint),
        "--epochs", "15",
        "--conditioning-only-epochs", "3",
        "--train-channels", "4000",
        "--val-channels", "800",
        "--pairs-per-channel", "8",
        "--batch-size", "128",
        "--learning-rate", "0.0001",
        "--morphing-min", "0.01",
        "--morphing-max", "0.5",
        "--anchor-ratio", "0.02",
        "--anchor-probability", "0.5",
        "--seed", "1",
        "--output", str(RESULTS_DIR),
    ], cwd=project)
    conditioned_checkpoint = RESULTS_DIR / "morph_conditioned_sfcnn_best.pt"
    evaluator = project / "morph_conditioned" / "evaluate_conditioned.py"
    run([
        sys.executable,
        str(evaluator),
        "--checkpoint", str(conditioned_checkpoint),
        "--morphing-ratios", "0.01", "0.02", "0.10", "0.30", "0.50",
        "--snr-db", "0", "10", "20",
        "--path-counts", "3",
        "--eval-channels", "10",
        "--seed", "1",
        "--evaluation-seed", "2026",
        "--output", str(RESULTS_DIR / "evaluation"),
    ], cwd=project)
    archive = shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=RESULTS_DIR)
    print("\nFine-tuning complete.")
    print("Download:", archive)


if __name__ == "__main__":
    main()
