"""Load / apply Fig4series configs; Stage2 result bundle for panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from experiments.figures._common.config import load_package_configs, result_dirs
from experiments.figures.fig4series_pancreas_stats.lib import core
from experiments.figures.fig4series_pancreas_stats.lib import metrics
from experiments.figure4_series_pipeline import Figure4Result

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


def load_result(*, force_rebuild: bool = False) -> Figure4Result:
    """Load Stage-1 cache and apply Stage-2 noise (always re-runs Stage 2)."""
    if not core.KINETICS_CACHE.exists() and not force_rebuild:
        raise FileNotFoundError(
            f"Kinetics cache not found: {core.KINETICS_CACHE}\n"
            "Run: python experiments/figures/fig4series_pancreas_stats/pipeline/run_pipeline.py"
        )
    kinetics = core.load_or_build_kinetics(force_rebuild=force_rebuild)
    return core.apply_stage2(kinetics)


def build_plot_bundle(result: Figure4Result, cfg: dict | None = None) -> Dict[str, Any]:
    stats_real = metrics.calc_6_metrics(result.real_spliced, result.real_unspliced)
    stats_sim = metrics.calc_6_metrics(result.spliced, result.unspliced)
    shared = cfg or {}
    return {
        "result": result,
        "stats_real": stats_real,
        "stats_sim": stats_sim,
        "cell_stats": metrics.compute_cell_stats(result.spliced, result.unspliced),
        "gene_s": metrics.compute_gene_stats(result.spliced),
        "label_real": shared.get("label_real", "Real"),
        "label_synthetic": shared.get("label_synthetic", "Synthetic"),
        "c_real": shared.get("colors", {}).get("real", "#4A90E2"),
        "c_sim": shared.get("colors", {}).get("synthetic", "#D0021B"),
        "c_u": shared.get("colors", {}).get("unspliced_accent", "#E87722"),
        "seed": int(shared.get("seed", core.SEED)),
    }


def load_bundle(cfg: dict | None = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = load_cfg()
    result = load_result()
    return build_plot_bundle(result, cfg)
