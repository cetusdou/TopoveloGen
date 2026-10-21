"""Shared draw helpers for Fig3 junction panels."""

from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures.fig03_junction_continuity.lib import core


def _bounds(result: Mapping, panel_cfg: Mapping):
    jcfg = result["cfg"]
    if panel_cfg.get("use_zoom"):
        xlim = tuple(jcfg["zoom_xlim"])
        ylim = tuple(jcfg["zoom_ylim"])
    else:
        xlim = tuple(jcfg["xlim"])
        ylim = tuple(jcfg["ylim"])
    return xlim, ylim


def draw_field_panel(ax: plt.Axes, result: Mapping, panel_cfg: Mapping) -> None:
    xlim, ylim = _bounds(result, panel_cfg)
    n_grid = core.N_GRID_ZOOM if panel_cfg.get("use_zoom") else core.N_GRID
    core.plot_field_panel(
        ax,
        result["system"],
        xlim,
        ylim,
        panel_cfg["title"],
        panel_cfg["topology"],
        n_grid=n_grid,
    )


def draw_trajectory_panel(
    ax: plt.Axes,
    result: Mapping,
    panel_cfg: Mapping,
    *,
    seed: int,
) -> None:
    xlim, ylim = _bounds(result, panel_cfg)
    rng = np.random.default_rng(seed)
    core.plot_trajectory_panel(
        ax,
        result["traj"],
        xlim,
        ylim,
        panel_cfg["title"],
        rng,
        n_show=core.N_TRAJ_SHOW,
        system=result["system"],
        topology_key=panel_cfg["topology"],
        show_potential_bg=bool(panel_cfg.get("show_potential_bg", False)),
    )


def draw_metrics_panel(ax: plt.Axes, metric_rows: list, panel_cfg: Mapping) -> None:
    core.plot_metrics_panel(ax, metric_rows, title=panel_cfg.get("title", "F  Junction continuity metrics"))
