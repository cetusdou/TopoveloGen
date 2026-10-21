"""Repository-relative input and output paths for active experiments."""

from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def pancreas_reference(override=None, require=True) -> Path:
    path = Path(override).expanduser() if override is not None else (
        repository_root() / "data" / "Pancreas" / "endocrinogenesis_day15.h5ad"
    )
    if require and not path.is_file():
        raise FileNotFoundError(f"Pancreas reference not found: {path}")
    return path


def result_dir(name: str, root=None) -> Path:
    base = Path(root) if root is not None else repository_root() / "results"
    return base / name
