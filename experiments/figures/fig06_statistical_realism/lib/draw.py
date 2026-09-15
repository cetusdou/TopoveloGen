"""Shared draw helpers for Fig6 statistical realism panels A–L."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from data_generation_module import CycleNode
from experiments.figures.fig06_statistical_realism.lib import core


def draw_topology(ax: plt.Axes, system, xlim, ylim) -> None:
    n = 70
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx, yy], axis=-1).astype(np.float32).reshape(-1, 2)
    u = system.potential_numpy(grid).reshape(n, n)
    ax.contourf(xx, yy, u, levels=20, cmap="Greys", alpha=0.08, zorder=0)
    for edge in system.edges:
        s, e = edge.start, edge.end
        ax.plot([s[0], e[0]], [s[1], e[1]], "k--", lw=0.8, alpha=0.45, zorder=1)
    cycle_nodes = [n for n in system.nodes if isinstance(n, CycleNode)]
    if cycle_nodes:
        cycle = cycle_nodes[0]
        ax.add_artist(
            plt.Circle(cycle.pos[:2], cycle.radius, fill=False, ls="--", color="0.4", lw=0.9, zorder=1)
        )
    else:
        for node in system.nodes:
            pos = node.pos[:2]
            ax.scatter(pos[0], pos[1], s=40, c="0.35", edgecolors="k", linewidths=0.5, zorder=2)


def plot_kde(ax: plt.Axes, values: np.ndarray, *, color: str, label: str, alpha: float = 0.3) -> None:
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return
    kde = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), 200)
    density = kde(grid)
    ax.plot(grid, density, color=color, lw=1.5, label=label)
    ax.fill_between(grid, density, color=color, alpha=alpha)


def _panel_limits(bundle: Mapping):
    system, z = bundle["system"], bundle["z"]
    xlim, ylim = core.XLIM, core.YLIM
    if not any(isinstance(n, CycleNode) for n in system.nodes):
        pad = 0.8
        xlim = (z[:, 0].min() - pad, z[:, 0].max() + pad)
        ylim = (z[:, 1].min() - pad, z[:, 1].max() + pad)
    return xlim, ylim


def draw_a(ax, bundle, panel_cfg: Mapping) -> None:
    xlim, ylim = _panel_limits(bundle)
    draw_topology(ax, bundle["system"], xlim, ylim)
    sc = ax.scatter(
        bundle["z"][:, 0],
        bundle["z"][:, 1],
        c=bundle["cell_stats"]["total_spliced"],
        s=5,
        cmap="plasma",
        alpha=0.55,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(panel_cfg.get("title", "A  Cells colored by spliced library size"))
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Total spliced")
    cbar.ax.tick_params(labelsize=7)


def draw_b(ax, bundle, panel_cfg: Mapping) -> None:
    cell_p = bundle["cell_poisson"]
    ts, tu = cell_p["total_spliced"], cell_p["total_unspliced"]
    bins = np.logspace(np.log10(max(ts.min(), tu.min(), 1)), np.log10(max(ts.max(), tu.max())), 40)
    ax.hist(ts, bins=bins, alpha=0.6, label="Spliced", color="C0", density=True)
    ax.hist(tu, bins=bins, alpha=0.6, label="Unspliced", color="C1", density=True)
    ax.set_xscale("log")
    ax.set_title(panel_cfg.get("title", "B  Library sizes (Poisson, no dropout)"))
    ax.set_xlabel("Total counts per cell")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)


def draw_c(ax, bundle, panel_cfg: Mapping) -> None:
    gene_s = bundle["gene_s"]
    mean, var = gene_s["mean"], gene_s["var"]
    mask = mean > 0.5
    ax.scatter(mean[mask], var[mask], s=3, alpha=0.35, c="C0", linewidths=0, rasterized=True)
    mx = np.logspace(np.log10(mean[mask].min()), np.log10(mean[mask].max()), 50)
    ax.plot(mx, mx, color="0.35", ls="--", lw=1.0, label="Poisson ($\\mathrm{Var=Mean}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(panel_cfg.get("title", "C  Gene mean–variance (spliced)"))
    ax.set_xlabel("Gene mean")
    ax.set_ylabel("Gene variance")
    ax.legend(fontsize=7)


def draw_d(ax, bundle, panel_cfg: Mapping) -> None:
    gene_s = bundle["gene_s"]
    mask = gene_s["mean"] > 0.5
    ax.scatter(
        gene_s["mean"][mask],
        gene_s["dropout"][mask],
        s=3,
        alpha=0.35,
        c="C2",
        linewidths=0,
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Gene mean (spliced)")
    ax.set_ylabel("Zero fraction")
    ax.set_title(panel_cfg.get("title", "D  Expression-dependent dropout"))
    ax.set_ylim(-0.02, 1.02)


def draw_e(ax, bundle, panel_cfg: Mapping) -> None:
    cell_p = bundle["cell_poisson"]
    ts, tu = cell_p["total_spliced"], cell_p["total_unspliced"]
    ax.scatter(ts, tu, s=2.5, alpha=0.25, c="0.35", linewidths=0, rasterized=True)
    lim = max(ts.max(), tu.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="C1", ls="--", lw=0.8, alpha=0.6, label="$U=S$")
    ax.set_xlabel("Total spliced counts")
    ax.set_ylabel("Total unspliced counts")
    ax.set_title(panel_cfg.get("title", "E  Spliced vs unspliced (Poisson)"))
    ax.legend(fontsize=7)
    med_ratio = float(np.median(cell_p["u_s_ratio"]))
    ax.text(0.05, 0.95, f"median $U/S$={med_ratio:.2f}", transform=ax.transAxes, va="top", fontsize=7)


def draw_f(ax, bundle, panel_cfg: Mapping) -> None:
    sweep = bundle["sweep_sparsity"]
    labels = list(sweep.keys())
    x = np.arange(len(labels))
    w = 0.35
    sp_s = [sweep[k][0] for k in labels]
    sp_u = [sweep[k][1] for k in labels]
    ax.bar(x - w / 2, sp_s, width=w, label="Spliced", color="C0", edgecolor="none")
    ax.bar(x + w / 2, sp_u, width=w, label="Unspliced", color="C1", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Sparsity (fraction zeros)")
    ax.set_ylim(0, min(1.05, max(sp_s + sp_u) * 1.2 + 0.05) if sp_s else 1.05)
    ax.set_title(panel_cfg.get("title", "F  Sparsity under noise settings"))
    ax.legend(fontsize=7)


def _colors(cfg: Mapping | None = None):
    colors = (cfg or {}).get("colors", {})
    return colors.get("reference", "#4A90E2"), colors.get("primary", "#D0021B")


def draw_g(ax, bundle, panel_cfg: Mapping, cfg: Mapping | None = None) -> None:
    c_ref, c_main = _colors(cfg)
    plot_kde(ax, np.log1p(bundle["stats_reference"]["lib_size"]), color=c_ref, label=bundle["label_reference"])
    plot_kde(ax, np.log1p(bundle["stats_primary"]["lib_size"]), color=c_main, label=bundle["label_primary"])
    ax.set_title(panel_cfg.get("title", "G  Library Size (Cell Depth)"))
    ax.set_xlabel("Log1p(Total Counts per Cell)")
    ax.legend(frameon=False, fontsize=7)


def draw_h(ax, bundle, panel_cfg: Mapping, cfg: Mapping | None = None) -> None:
    c_ref, c_main = _colors(cfg)
    plot_kde(ax, bundle["stats_reference"]["n_genes"], color=c_ref, label=bundle["label_reference"])
    plot_kde(ax, bundle["stats_primary"]["n_genes"], color=c_main, label=bundle["label_primary"])
    ax.set_title(panel_cfg.get("title", "H  Detected Genes (Cell Complexity)"))
    ax.set_xlabel("Number of Detected Genes")
    ax.legend(frameon=False, fontsize=7)


def draw_i(ax, bundle, panel_cfg: Mapping, cfg: Mapping | None = None) -> None:
    c_ref, c_main = _colors(cfg)
    mask_r = bundle["stats_reference"]["gene_mean"] > 0.01
    mask_m = bundle["stats_primary"]["gene_mean"] > 0.01
    plot_kde(ax, np.log1p(bundle["stats_reference"]["gene_mean"][mask_r]), color=c_ref, label=bundle["label_reference"])
    plot_kde(ax, np.log1p(bundle["stats_primary"]["gene_mean"][mask_m]), color=c_main, label=bundle["label_primary"])
    ax.set_title(panel_cfg.get("title", "I  Mean Expression Distribution"))
    ax.set_xlabel("Log1p(Gene Mean Expression)")
    ax.legend(frameon=False, fontsize=7)


def draw_j(ax, bundle, panel_cfg: Mapping, cfg: Mapping | None = None, seed: int | None = None) -> None:
    c_ref, c_main = _colors(cfg)
    rng = np.random.default_rng(seed if seed is not None else core.SEED)
    sr, sp = bundle["stats_reference"], bundle["stats_primary"]
    idx_r = rng.choice(len(sr["gene_mean"]), min(2000, len(sr["gene_mean"])), replace=False)
    idx_m = rng.choice(len(sp["gene_mean"]), min(2000, len(sp["gene_mean"])), replace=False)
    ax.scatter(
        np.log1p(sr["gene_mean"][idx_r]),
        np.log1p(sr["gene_var"][idx_r]),
        s=6,
        c=c_ref,
        alpha=0.3,
        label=bundle["label_reference"],
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        np.log1p(sp["gene_mean"][idx_m]),
        np.log1p(sp["gene_var"][idx_m]),
        s=6,
        c=c_main,
        alpha=0.3,
        label=bundle["label_primary"],
        linewidths=0,
        rasterized=True,
    )
    max_val = max(np.log1p(sr["gene_mean"]).max(), np.log1p(sp["gene_mean"]).max())
    line = np.linspace(0, max_val, 100)
    ax.plot(line, line, "k--", lw=1.0, alpha=0.5, label="Poisson")
    ax.set_title(panel_cfg.get("title", "J  Mean–Variance (Overdispersion)"))
    ax.set_xlabel("Log1p(Mean)")
    ax.set_ylabel("Log1p(Variance)")
    ax.legend(frameon=False, fontsize=7)


def draw_k(ax, bundle, panel_cfg: Mapping, cfg: Mapping | None = None, seed: int | None = None) -> None:
    c_ref, c_main = _colors(cfg)
    rng = np.random.default_rng(seed if seed is not None else core.SEED)
    sr, sp = bundle["stats_reference"], bundle["stats_primary"]
    idx_r = rng.choice(len(sr["drop_mean"]), min(2000, len(sr["drop_mean"])), replace=False)
    idx_m = rng.choice(len(sp["drop_mean"]), min(2000, len(sp["drop_mean"])), replace=False)
    ax.scatter(
        np.log1p(sr["drop_mean"][idx_r]),
        sr["drop_rate"][idx_r],
        s=6,
        c=c_ref,
        alpha=0.3,
        label=bundle["label_reference"],
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        np.log1p(sp["drop_mean"][idx_m]),
        sp["drop_rate"][idx_m],
        s=6,
        c=c_main,
        alpha=0.3,
        label=bundle["label_primary"],
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(panel_cfg.get("title", "K  Dropout Rate vs Mean"))
    ax.set_xlabel("Log1p(Mean)")
    ax.set_ylabel("Fraction of Zeros")
    ax.legend(frameon=False, fontsize=7)


def draw_l(ax, bundle, panel_cfg: Mapping, cfg: Mapping | None = None) -> None:
    c_ref, c_main = _colors(cfg)
    plot_kde(ax, bundle["stats_reference"]["u_frac"], color=c_ref, label=bundle["label_reference"])
    plot_kde(ax, bundle["stats_primary"]["u_frac"], color=c_main, label=bundle["label_primary"])
    ax.set_title(panel_cfg.get("title", "L  Unspliced Fraction (Global Kinetics)"))
    ax.set_xlabel("Unspliced Reads / Total Reads")
    ax.set_xlim(0, 0.5)
    ax.legend(frameon=False, fontsize=7)


DRAWERS = {
    "A": draw_a,
    "B": draw_b,
    "C": draw_c,
    "D": draw_d,
    "E": draw_e,
    "F": draw_f,
    "G": draw_g,
    "H": draw_h,
    "I": draw_i,
    "J": draw_j,
    "K": draw_k,
    "L": draw_l,
}
