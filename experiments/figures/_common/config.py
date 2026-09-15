"""JSON config loading and path helpers for figure packages."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[3]


def ensure_repo_on_path() -> Path:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


def package_dir(file: Union[str, Path]) -> Path:
    """Return the figure package root for a file under panels/, compose/, pipeline/, or lib/."""
    path = Path(file).resolve()
    for parent in path.parents:
        if parent.parent.name == "figures" and (parent / "configs").is_dir():
            return parent
        if parent.name.startswith("fig") and (parent / "configs").is_dir():
            return parent
    raise FileNotFoundError(f"Cannot locate figure package root for {path}")


def load_json(path: Union[str, Path]) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a JSON object: {path}")
    return data


def resolve_path(value: Union[str, Path], *, base: Optional[Path] = None) -> Path:
    """Resolve path relative to repo root (absolute paths unchanged)."""
    p = Path(value)
    if p.is_absolute():
        return p
    return (base or REPO_ROOT) / p


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> dict:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(dict(out[key]), val)
        else:
            out[key] = val
    return out


def load_package_configs(
    pkg_dir: Path,
    panel_config: Optional[str] = None,
    compose_config: Optional[str] = None,
) -> dict:
    """Load shared + pipeline (+ optional panel/compose) configs."""
    configs = pkg_dir / "configs"
    shared = load_json(configs / "shared.json")
    pipeline = load_json(configs / "pipeline.json")
    merged: dict = deep_merge(shared, {"pipeline": pipeline})
    if panel_config:
        path = configs / panel_config
        if not path.exists() and not panel_config.endswith(".json"):
            path = configs / f"{panel_config}.json"
        merged["panel"] = load_json(path)
    if compose_config:
        path = configs / compose_config
        if not path.exists() and not compose_config.endswith(".json"):
            path = configs / f"{compose_config}.json"
        merged["compose"] = load_json(path)
    return merged


def result_dirs(cfg: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    """Return (out_dir, data_dir, fig_dir) from shared config."""
    out_dir = resolve_path(cfg["out_dir"])
    data_dir = out_dir / cfg.get("data_subdir", "data")
    fig_dir = out_dir / cfg.get("figures_subdir", "figures")
    return out_dir, data_dir, fig_dir
