"""Run the preserved additional experiments in notebook order."""

from __future__ import annotations

import argparse
from pathlib import Path


def execute_cells(folder: Path, namespace: dict) -> None:
    for path in sorted(folder.glob("[0-9][0-9]_*.py")):
        print(f"\n--- Running {path.relative_to(folder.parent)} ---", flush=True)
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), namespace)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--without-original",
        action="store_true",
        help="Skip cells 00-22. Use only when their definitions/state already exist.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    namespace = {"__name__": "__main__"}
    if not args.without_original:
        execute_cells(root / "original_sfcnn", namespace)
    execute_cells(root / "additional_tests", namespace)


if __name__ == "__main__":
    main()
