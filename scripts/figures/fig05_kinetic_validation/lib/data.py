"""Load / apply Fig5 package configs and cached kinetic results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pickle

from scripts.figures._common.config import load_package_configs, result_dirs
from scripts.figures.fig05_kinetic_validation.lib import core

PKG_DIR = Path(__file__).resolve().parents[1]


def load_cfg(panel_config: str | None = None, compose_config: str | None = None) -> dict:
    cfg = load_package_configs(PKG_DIR, panel_config=panel_config, compose_config=compose_config)
    out_dir, data_dir, fig_dir = result_dirs(cfg)
    core.configure(
        out_dir=out_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        pipeline=cfg.get("pipeline"),
    )
    seed = int(cfg.get("seed", cfg.get("pipeline", {}).get("seed", core.SEED)))
    core.SEED = seed
    return cfg


def load_bundle(*, with_gamma: bool = False) -> Dict[str, Any]:
    """Load kinetic_tensors.pt; optionally recompute gamma sweeps (legacy plot-only)."""
    path = core.DATA_DIR / "kinetic_tensors.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached kinetic tensors: {path}\n"
            "Run: python scripts/figures/fig05_kinetic_validation/pipeline/run_pipeline.py"
        )
    data = pickle.load(open(str(path), "rb"))
    traj = data["traj"]
    result_default = {k: v for k, v in data["result_default"].items()}
    system = core.build_linear_system()
    bundle: Dict[str, Any] = {
        "system": system,
        "traj": traj,
        "result_default": result_default,
        "steps": int(traj.shape[0]),
    }
    if with_gamma:
        bundle["gamma_results"] = core.compute_gamma_results(system, traj)
    return bundle
