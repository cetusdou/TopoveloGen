"""Load / apply Fig2 package configs and cached trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from scripts.figures._common.config import load_package_configs, result_dirs
from scripts.figures.fig02_topology_primitives.lib import core

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
    pipe = cfg["pipeline"]
    if "panel_bounds" in pipe:
        core.PANEL_BOUNDS = {
            k: (tuple(v[0]), tuple(v[1])) for k, v in pipe["panel_bounds"].items()
        }
    if "panel_plot_config" in pipe:
        core.PANEL_PLOT_CONFIG = {k: dict(v) for k, v in pipe["panel_plot_config"].items()}
    if "figure_order" in pipe:
        core.FIGURE_2_ORDER = list(pipe["figure_order"])
    seed = int(cfg.get("seed", pipe.get("seed", core.SEED)))
    core.SEED = seed
    return cfg


def load_one_result(topology: str) -> Dict:
    builders = core.get_builders()
    npz_path = core.DATA_DIR / f"{topology}_traj.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing cached trajectory: {npz_path}\n"
            "Run: python scripts/figures/fig02_topology_primitives/pipeline/run_pipeline.py"
        )
    data = np.load(npz_path, allow_pickle=False)
    traj = np.asarray(data["traj"])
    metrics = {
        "mean_distance": float(data["mean_distance"]),
        "median_distance": float(data["median_distance"]),
        "p90_distance": float(data["p90_distance"]),
        "mean_flow_alignment": float(data["mean_flow_alignment"]),
        "mean_rot_flow_alignment": float(data["mean_rot_flow_alignment"]),
    }
    return {"system": builders[topology](), "traj": traj, "metrics": metrics}


def load_all_results() -> Dict[str, Dict]:
    return {name: load_one_result(name) for name in core.FIGURE_2_ORDER}
