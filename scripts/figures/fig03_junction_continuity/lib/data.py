"""Load / apply Fig3 package configs and cached junction results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import pickle

from scripts.figures._common.config import load_package_configs, result_dirs
from scripts.figures.fig03_junction_continuity.lib import core

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


def load_metric_rows() -> List[Dict[str, Any]]:
    path = core.DATA_DIR / "junction_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached metrics: {path}\n"
            "Run: python scripts/figures/fig03_junction_continuity/pipeline/run_pipeline.py"
        )
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_trajectories() -> Dict[str, Any]:
    path = core.DATA_DIR / "trajectories.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached trajectories: {path}\n"
            "Run: python scripts/figures/fig03_junction_continuity/pipeline/run_pipeline.py"
        )
    return pickle.load(open(str(path), "rb"))


def load_one_result(topology: str) -> Dict[str, Any]:
    trajs = load_trajectories()
    if topology not in trajs:
        raise KeyError(f"Topology {topology!r} not in {core.DATA_DIR / 'trajectories.pt'}")
    jcfg = core.junction_cfg(topology)
    return {
        "system": core.BUILDERS[topology](),
        "traj": trajs[topology],
        "cfg": jcfg,
    }


def load_all_results() -> Dict[str, Dict[str, Any]]:
    return {name: load_one_result(name) for name in core.FIGURE_ORDER}
