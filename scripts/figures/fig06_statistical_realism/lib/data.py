"""Load / apply Fig6 package configs and cached count matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from scripts.figures._common.config import load_package_configs, result_dirs
from scripts.figures.fig06_statistical_realism.lib import core

PKG_DIR = Path(__file__).resolve().parents[1]


def load_cfg(panel_config: str | None = None, compose_config: str | None = None) -> dict:
    cfg = load_package_configs(PKG_DIR, panel_config=panel_config, compose_config=compose_config)
    out_dir, data_dir, fig_dir = result_dirs(cfg)
    core.configure(
        out_dir=out_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        pipeline=cfg.get("pipeline"),
        shared=cfg,
    )
    return cfg


def load_bundle() -> Dict:
    return core.load_plot_bundle()
