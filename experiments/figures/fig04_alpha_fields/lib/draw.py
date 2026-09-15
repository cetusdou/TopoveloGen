"""Shared draw helpers for Fig4 alpha-field panels."""

from __future__ import annotations

from typing import Mapping, Tuple

import matplotlib.pyplot as plt

from experiments.figures.fig04_alpha_fields.lib import core


def draw_topology(ax: plt.Axes, bundle: Mapping, panel_cfg: Mapping) -> None:
    core.plot_topology_panel(
        ax,
        bundle["system"],
        bundle["z"],
        title=panel_cfg.get("title"),
    )


def draw_alpha_scatter(
    ax: plt.Axes,
    bundle: Mapping,
    panel_cfg: Mapping,
    *,
    vmin_vmax: Tuple[float, float] | None = None,
) -> None:
    gene_key = panel_cfg["gene_key"]
    gidx = core.gene_index(gene_key)
    if vmin_vmax is None:
        vmin, vmax = core.shared_alpha_color_limits(bundle["alpha"])
    else:
        vmin, vmax = vmin_vmax
    sc = core.plot_alpha_scatter(
        ax,
        bundle["z"],
        bundle["alpha"][:, gidx],
        panel_cfg["title"],
        vmin=vmin,
        vmax=vmax,
        system=bundle["system"],
        show_topology=bool(panel_cfg.get("show_topology", True)),
    )
    ax.set_xlim(core.XLIM)
    ax.set_ylim(core.YLIM)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\alpha$")
    cbar.ax.tick_params(labelsize=7)


def draw_phase_heatmap(ax: plt.Axes, bundle: Mapping, panel_cfg: Mapping) -> None:
    core.plot_cycle_phase_heatmap(
        ax,
        bundle["system"],
        bundle["z"],
        bundle["alpha"],
        title=panel_cfg.get("title"),
    )
