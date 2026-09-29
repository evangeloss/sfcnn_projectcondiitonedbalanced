"""Run the original SFCNN notebook cells in order."""

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
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    namespace = {"__name__": "__main__"}
    execute_cells(root / "original_sfcnn", namespace)


if __name__ == "__main__":
    main()
