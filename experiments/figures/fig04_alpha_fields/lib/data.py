"""Load / apply Fig4 package configs and cached alpha-field results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pickle

from experiments.figures._common.config import load_package_configs, result_dirs
from experiments.figures.fig04_alpha_fields.lib import core

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


def load_cached_fields() -> Tuple[Any, Any, Any]:
    z_path = core.DATA_DIR / "latent_positions.pt"
    alpha_path = core.DATA_DIR / "alpha_matrix.pt"
    if not z_path.exists() or not alpha_path.exists():
        raise FileNotFoundError(
            f"Missing cached alpha-field data under {core.DATA_DIR}\n"
            "Run: python experiments/figures/fig04_alpha_fields/pipeline/run_pipeline.py"
        )
    system = core.build_cycle_bifurcation_system()
    z_cells = pickle.load(open(str(z_path), "rb"))
    alpha = pickle.load(open(str(alpha_path), "rb"))
    return system, z_cells, alpha


def load_bundle() -> Dict[str, Any]:
    system, z, alpha = load_cached_fields()
    return {"system": system, "z": z, "alpha": alpha}
