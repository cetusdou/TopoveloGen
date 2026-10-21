"""Draw helpers for Fig4series main (A–I) and supplementary (S1–S4) panels."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures.fig4series_pancreas_stats.lib import core
from scripts.figures.fig4series_pancreas_stats.lib import metrics


def draw_a(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    sr, ss = bundle["stats_real"], bundle["stats_sim"]
    c_real, c_sim = bundle["c_real"], bundle["c_sim"]
    vals_r = np.log1p(sr["lib_size"])
    vals_s = np.log1p(ss["lib_size"])
    metrics.plot_kde(ax, vals_r, color=c_real, label=bundle["label_real"], seed=bundle["seed"])
    metrics.plot_kde(ax, vals_s, color=c_sim, label=bundle["label_synthetic"], seed=bundle["seed"])
    ax.set_title(panel_cfg.get("title", "A. Library Size (Cell Depth)"), fontsize=12)
    ax.set_xlabel("Log1p(Total Counts per Cell)", fontsize=10)
    lo = min(np.percentile(vals_r, 1), np.percentile(vals_s, 1)) - 0.1
    hi = max(np.percentile(vals_r, 99), np.percentile(vals_s, 99)) + 0.1
    ax.set_xlim(lo, hi)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)


def draw_b(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    result = bundle["result"]
    sr, ss = bundle["stats_real"], bundle["stats_sim"]
    c_real, c_sim = bundle["c_real"], bundle["c_sim"]
    n_real_genes = result.real_spliced.shape[1]
    n_sim_genes = result.spliced.shape[1]
    frac_r = sr["n_genes"] / n_real_genes * 100
    frac_s = ss["n_genes"] / n_sim_genes * 100
    metrics.plot_kde(ax, frac_r, color=c_real, label=bundle["label_real"], seed=bundle["seed"])
    metrics.plot_kde(ax, frac_s, color=c_sim, label=bundle["label_synthetic"], seed=bundle["seed"])
    ax.set_title(panel_cfg.get("title", "B. Fraction of Genes Detected"), fontsize=12)
    ax.set_xlabel("Genes Detected per Cell (%)", fontsize=10)
    lo_b = min(np.percentile(frac_r, 1), np.percentile(frac_s, 1)) - 1
    hi_b = max(np.percentile(frac_r, 99), np.percentile(frac_s, 99)) + 1
    ax.set_xlim(max(0, lo_b), hi_b)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)


def draw_c(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    sr, ss = bundle["stats_real"], bundle["stats_sim"]
    c_real, c_sim = bundle["c_real"], bundle["c_sim"]
    mask_r = sr["gene_mean"] > 0.01
    mask_s = ss["gene_mean"] > 0.01
    metrics.plot_kde(ax, np.log1p(sr["gene_mean"][mask_r]), color=c_real, label=bundle["label_real"], seed=bundle["seed"])
    metrics.plot_kde(ax, np.log1p(ss["gene_mean"][mask_s]), color=c_sim, label=bundle["label_synthetic"], seed=bundle["seed"])
    ax.set_title(panel_cfg.get("title", "C. Mean Expression Distribution"), fontsize=12)
    ax.set_xlabel("Log1p(Gene Mean Expression)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)


def draw_d(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    rng = np.random.default_rng(bundle["seed"])
    sr, ss = bundle["stats_real"], bundle["stats_sim"]
    c_real, c_sim = bundle["c_real"], bundle["c_sim"]
    idx_r = rng.choice(len(sr["gene_mean"]), min(2000, len(sr["gene_mean"])), replace=False)
    idx_s = rng.choice(len(ss["gene_mean"]), min(2000, len(ss["gene_mean"])), replace=False)
    ax.scatter(
        np.log1p(sr["gene_mean"][idx_r]),
        np.log1p(sr["gene_var"][idx_r]),
        s=10,
        c=c_real,
        alpha=0.3,
        label=bundle["label_real"],
    )
    ax.scatter(
        np.log1p(ss["gene_mean"][idx_s]),
        np.log1p(ss["gene_var"][idx_s]),
        s=10,
        c=c_sim,
        alpha=0.3,
        label=bundle["label_synthetic"],
    )
    mx = max(np.log1p(sr["gene_mean"]).max(), np.log1p(ss["gene_mean"]).max())
    ax.plot(np.linspace(0, mx, 100), np.linspace(0, mx, 100), "k--", lw=1.5, label="Poisson", alpha=0.5)
    ax.set_title(panel_cfg.get("title", "D. Mean-Variance (Overdispersion)"), fontsize=12)
    ax.set_xlabel("Log1p(Mean)", fontsize=10)
    ax.set_ylabel("Log1p(Variance)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)


def draw_e(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    rng = np.random.default_rng(bundle["seed"])
    sr, ss = bundle["stats_real"], bundle["stats_sim"]
    c_real, c_sim = bundle["c_real"], bundle["c_sim"]
    idx_rd = rng.choice(len(sr["drop_mean"]), min(2000, len(sr["drop_mean"])), replace=False)
    idx_sd = rng.choice(len(ss["drop_mean"]), min(2000, len(ss["drop_mean"])), replace=False)
    ax.scatter(
        np.log1p(sr["drop_mean"][idx_rd]),
        sr["drop_rate"][idx_rd],
        s=10,
        c=c_real,
        alpha=0.3,
        label=bundle["label_real"],
    )
    ax.scatter(
        np.log1p(ss["drop_mean"][idx_sd]),
        ss["drop_rate"][idx_sd],
        s=10,
        c=c_sim,
        alpha=0.3,
        label=bundle["label_synthetic"],
    )
    ax.set_title(panel_cfg.get("title", "E. Dropout Rate vs Mean"), fontsize=12)
    ax.set_xlabel("Log1p(Mean)", fontsize=10)
    ax.set_ylabel("Fraction of Zeros", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)


def draw_f(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    sr, ss = bundle["stats_real"], bundle["stats_sim"]
    c_real, c_sim = bundle["c_real"], bundle["c_sim"]
    metrics.plot_kde(ax, sr["u_frac"], color=c_real, label=bundle["label_real"], seed=bundle["seed"])
    metrics.plot_kde(ax, ss["u_frac"], color=c_sim, label=bundle["label_synthetic"], seed=bundle["seed"])
    ax.set_title(panel_cfg.get("title", "F. Unspliced Fraction (Global Kinetics)"), fontsize=12)
    ax.set_xlabel("Unspliced Reads / Total Reads", fontsize=10)
    ax.set_xlim(0, 0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)


def draw_g(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    result = bundle["result"]
    c_real, c_u = bundle["c_real"], bundle["c_u"]
    noise_sp = metrics.get_noise_stage_sparsities(result, seed=bundle["seed"])
    stage_g = ["clean", "nb_dropout", "figure4"]
    x_g = np.arange(len(stage_g))
    x_lbl_g = ["Clean", "NB+Dropout\n(simple)", "NB+Dropout\n(Figure4)"]
    sp_s_g = [noise_sp[k]["spliced"] for k in stage_g]
    sp_u_g = [noise_sp[k]["unspliced"] for k in stage_g]
    ax.plot(x_g, sp_s_g, "o-", color=c_real, lw=2, markersize=7, label="Spliced")
    ax.plot(x_g, sp_u_g, "s-", color=c_u, lw=2, markersize=7, label="Unspliced")
    ax.axhline(noise_sp["real"]["spliced"], color=c_real, ls="--", lw=1.5, alpha=0.65, label="Real S target")
    ax.axhline(noise_sp["real"]["unspliced"], color=c_u, ls="--", lw=1.5, alpha=0.65, label="Real U target")
    ax.set_xticks(x_g)
    ax.set_xticklabels(x_lbl_g, fontsize=9)
    ax.set_ylabel("Sparsity (fraction zeros)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(panel_cfg.get("title", "G. Sparsity"), fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.3)


def draw_h(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    sr, ss = bundle["stats_real"], bundle["stats_sim"]
    c_real, c_sim = bundle["c_real"], bundle["c_sim"]
    m_r, v_r = sr["gene_mean"], sr["gene_var"]
    m_s, v_s = ss["gene_mean"], ss["gene_var"]
    mask_fr, mask_fs = m_r > 0.5, m_s > 0.5
    fano_r = np.log1p(v_r[mask_fr] / (m_r[mask_fr] + 1e-8))
    fano_s = np.log1p(v_s[mask_fs] / (m_s[mask_fs] + 1e-8))
    metrics.plot_kde(ax, fano_r, color=c_real, label=bundle["label_real"], seed=bundle["seed"])
    metrics.plot_kde(ax, fano_s, color=c_sim, label=bundle["label_synthetic"], seed=bundle["seed"])
    ax.axvline(np.log1p(1.0), color="k", ls="--", lw=1.2, alpha=0.5, label="Poisson (FF=1)")
    ax.set_title(panel_cfg.get("title", "H. Gene Fano Factor"), fontsize=12)
    ax.set_xlabel("Log1p(Fano Factor = Var/Mean)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)


def draw_i(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    sr, ss = bundle["stats_real"], bundle["stats_sim"]
    c_real, c_sim = bundle["c_real"], bundle["c_sim"]
    gm_r = np.sort(sr["gene_mean"])[::-1]
    gm_s = np.sort(ss["gene_mean"])[::-1]
    gm_r = gm_r[gm_r > 0]
    gm_s = gm_s[gm_s > 0]
    rank_r = np.arange(1, len(gm_r) + 1) / len(gm_r)
    rank_s = np.arange(1, len(gm_s) + 1) / len(gm_s)
    ax.loglog(rank_r, gm_r, color=c_real, lw=1.8, alpha=0.85, label=bundle["label_real"])
    ax.loglog(rank_s, gm_s, color=c_sim, lw=1.8, alpha=0.85, label=bundle["label_synthetic"])
    ax.set_title(panel_cfg.get("title", "I. Gene Rank–Frequency"), fontsize=12)
    ax.set_xlabel("Normalized Gene Rank (high → low)", fontsize=10)
    ax.set_ylabel("Mean Expression", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3, which="both")


def draw_graph_topology(ax, system, z):
    pad = 0.8
    xlim = (z[:, 0].min() - pad, z[:, 0].max() + pad)
    ylim = (z[:, 1].min() - pad, z[:, 1].max() + pad)
    n = 60
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx, yy], axis=-1).astype(np.float32).reshape(-1, 2)
    u = system.potential_numpy(grid).reshape(n, n)
    ax.contourf(xx, yy, u, levels=20, cmap="Greys", alpha=0.08, zorder=0)
    for edge in system.edges:
        s, e = edge.start, edge.end
        ax.plot([s[0], e[0]], [s[1], e[1]], "k--", lw=0.8, alpha=0.45, zorder=1)
    for node in system.nodes:
        ax.scatter(node.pos[0], node.pos[1], s=35, c="0.35", edgecolors="k", linewidths=0.5, zorder=2)
    return xlim, ylim


def draw_s1(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    result = bundle["result"]
    cell_stats = bundle["cell_stats"]
    z = result.z
    xlim, ylim = draw_graph_topology(ax, result.system, z)
    sc = ax.scatter(
        z[:, 0],
        z[:, 1],
        c=cell_stats["total_spliced"],
        s=4,
        cmap="plasma",
        alpha=0.55,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(panel_cfg.get("title", "S1  Latent cells × spliced library size"))
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Total spliced")


def draw_s2(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    cell_stats = bundle["cell_stats"]
    ts, tu = cell_stats["total_spliced"], cell_stats["total_unspliced"]
    ax.scatter(ts, tu, s=2.5, alpha=0.25, c="0.35", linewidths=0, rasterized=True)
    lim = max(ts.max(), tu.max()) * 1.05
    ax.plot([0, lim], [0, lim], "C1", ls="--", lw=0.8, alpha=0.6, label="$U=S$")
    med = float(np.median(cell_stats["u_s_ratio"]))
    ax.set_title(panel_cfg.get("title", "S2  Spliced vs unspliced (synthetic)"))
    ax.set_xlabel("Total spliced")
    ax.set_ylabel("Total unspliced")
    ax.legend(fontsize=7)
    ax.text(0.05, 0.95, f"median $U/S$={med:.3f}", transform=ax.transAxes, va="top", fontsize=8)


def draw_s3(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    result = bundle["result"]
    labels = ["Real", "Synthetic"]
    sp_s = [metrics.sparsity(result.real_spliced), metrics.sparsity(result.spliced)]
    sp_u = [metrics.sparsity(result.real_unspliced), metrics.sparsity(result.unspliced)]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, sp_s, width=w, label="Spliced", color="C0")
    ax.bar(x + w / 2, sp_u, width=w, label="Unspliced", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sparsity (fraction zeros)")
    ax.set_ylim(0, 1.05)
    ax.set_title(panel_cfg.get("title", "S3  Sparsity comparison"))
    ax.legend(fontsize=7)


def draw_s4(ax, bundle: Mapping, panel_cfg: Mapping) -> None:
    gene_s = bundle["gene_s"]
    mask = gene_s["mean"] > 0.5
    ax.scatter(gene_s["mean"][mask], gene_s["var"][mask], s=3, alpha=0.35, c="C0", linewidths=0, rasterized=True)
    mx = np.logspace(np.log10(gene_s["mean"][mask].min()), np.log10(gene_s["mean"][mask].max()), 50)
    ax.plot(mx, mx, "0.35", ls="--", lw=1.0, label="Poisson")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(panel_cfg.get("title", "S4  Gene mean–variance (spliced, synthetic)"))
    ax.set_xlabel("Gene mean")
    ax.set_ylabel("Gene variance")
    ax.legend(fontsize=7)
