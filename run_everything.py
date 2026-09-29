"""Run every exported notebook cell in its original order."""

from pathlib import Path


def execute_cells(folder: Path, namespace: dict) -> None:
    for path in sorted(folder.glob("[0-9][0-9]_*.py")):
        print(f"\n--- Running {path.relative_to(folder.parent)} ---", flush=True)
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    shared = {"__name__": "__main__"}
    execute_cells(root / "original_sfcnn", shared)
    execute_cells(root / "additional_tests", shared)
