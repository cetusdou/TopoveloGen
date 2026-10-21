"""Load / apply Fig7 package configs and cached velocity metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.figures._common.config import load_package_configs, result_dirs
from scripts.figures.fig07_embedding_masked_errors.lib import core

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


def load_main_and_sweep() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    return core.load_plot_data()
