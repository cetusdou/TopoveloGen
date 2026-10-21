"""Shared draw helpers for Fig2 topology panels."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures.fig02_topology_primitives.lib import core


def draw_topology_panel(ax: plt.Axes, result: Mapping, panel_cfg: Mapping, *, seed: int) -> None:
    topology = panel_cfg["topology"]
    xlim, ylim = core.PANEL_BOUNDS[topology]
    rng = np.random.default_rng(seed)
    core.plot_topology_panel(
        result["system"],
        result["traj"],
        ax,
        panel_cfg["title"],
        tuple(xlim),
        tuple(ylim),
        rng,
        topology=topology,
        n_traj_plot=int(panel_cfg.get("n_traj_plot", 20)),
        support_cut=float(core.SUPPORT_CUT),
    )


def draw_s1_distance(ax: plt.Axes, all_results: Mapping, panel_cfg: Mapping) -> None:
    names = []
    for n in core.FIGURE_2_ORDER:
        title = core.PANEL_TITLES[n]
        names.append(title.split(". ", 1)[1] if ". " in title else title)
    mean_d = [all_results[n]["metrics"]["mean_distance"] for n in core.FIGURE_2_ORDER]
    median_d = [all_results[n]["metrics"]["median_distance"] for n in core.FIGURE_2_ORDER]
    p90_d = [all_results[n]["metrics"]["p90_distance"] for n in core.FIGURE_2_ORDER]
    x = np.arange(len(names))
    w = float(panel_cfg.get("bar_width", 0.26))
    ax.bar(x - w, mean_d, width=w, label="Mean", color="C0", edgecolor="none")
    ax.bar(x, median_d, width=w, label="Median", color="C2", edgecolor="none", alpha=0.9)
    ax.bar(x + w, p90_d, width=w, label="P90", color="C3", edgecolor="none", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(panel_cfg.get("ylabel", "Distance to prescribed manifold"))
    ax.set_title(panel_cfg.get("title", "Supplementary Fig. S1"))
    ax.legend(loc="upper right", frameon=False)
