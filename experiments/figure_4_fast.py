"""
Deprecated entrypoint. Use the panelized package:
  experiments/figures/fig4series_pancreas_stats/
See experiments/figures/README.md for run instructions.

Figure4Series.ipynb pipeline + supplementary statistical_realism metrics.

TWO-LAYER ARCHITECTURE
======================
Layer 1 – continuous_kinetics  (slow, ~7 min, cached in kinetics_cache.pt)
  PAGA topology → GraphSimulator → GeneSynthesizer.generate_kinetics
  Cached by simulation hyper-parameters (SEED, N_STEPS, N_CELLS, …).
  Rebuild only if the topology or kinetics model changes.

Layer 2 – observation noise  (fast, <10 s, always re-runs)
  lib_size scaling → NB/Poisson → mean-dependent dropout → binomial downsampling
  Controlled entirely by PostProcessConfig.  Tune freely without touching Layer 1.

Usage (from repo root):
    python experiments/figure_4_fast.py                 # full run
    python experiments/figure_4_fast.py --plot-only     # reload kinetics, retune noise, replot
    python experiments/figure_4_fast.py --cache-only    # build kinetics cache only (no noise/plot)
    python experiments/figure_4_fast.py --force-kinetics  # force rebuild Stage-1 kinetics
"""


import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figure4_series_pipeline import (  # noqa: E402
    DEFAULT_PANCREAS_H5AD,
    DEFAULT_PP_CONFIG,
    KINETICS_CACHE_PATH,
    PostProcessConfig,
    SEED,
    Figure4Result,
    apply_observation_noise,
    load_or_build_kinetics_cache,
)
from experiments.figure_style import apply_publication_style, save_figure  # noqa: E402

OUT_DIR = REPO_ROOT / "results" / "statistical_realism_figure4"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
FIGURE_MAIN_SIZE = (10.5, 8.5)
FIGURE_MAIN_DPI  = 300
FIGURE_SUPP_SIZE = (8.0, 6.5)


# ---------------------------------------------------------------------------
# Statistics  (identical to notebook cell 21)
# ---------------------------------------------------------------------------
def calc_6_metrics(s: np.ndarray, u: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute all six validation metrics with fewer temporary arrays.

    The original version repeatedly materialized boolean matrices and copied
    arrays. Here x_total is kept as the only large temporary; reductions are
    then reused.
    """
    x_total = np.asarray(s) + np.asarray(u)
    n_cells = x_total.shape[0]

    lib_size = x_total.sum(axis=1)
    n_genes = np.count_nonzero(x_total, axis=1)
    gene_mean = x_total.mean(axis=0)
    gene_var = x_total.var(axis=0)

    mask = gene_mean > 0
    mean_for_drop = gene_mean[mask]
    # count_nonzero is usually faster than summing a boolean array.
    drop_rate = 1.0 - (np.count_nonzero(x_total[:, mask], axis=0) / n_cells)

    u_sum = np.asarray(u).sum(axis=1)
    total_sum = np.where(lib_size == 0, 1, lib_size)
    return {
        "lib_size": lib_size,
        "n_genes": n_genes,
        "gene_mean": gene_mean,
        "gene_var": gene_var,
        "drop_mean": mean_for_drop,
        "drop_rate": drop_rate,
        "u_frac": u_sum / total_sum,
    }


def compute_cell_stats(s: np.ndarray, u: np.ndarray) -> Dict[str, np.ndarray]:
    total_s = s.sum(axis=1)
    total_u = u.sum(axis=1)
    return {
        "total_spliced": total_s,
        "total_unspliced": total_u,
        "u_s_ratio": total_u / (total_s + 1e-8),
    }


def compute_gene_stats(x: np.ndarray) -> Dict[str, np.ndarray]:
    n_cells = x.shape[0]
    return {
        "mean": x.mean(axis=0),
        "var": x.var(axis=0),
        "dropout": 1.0 - (np.count_nonzero(x, axis=0) / n_cells),
    }


def sparsity(x: np.ndarray) -> float:
    return float(1.0 - np.count_nonzero(x) / x.size)


def _plot_kde(
    ax,
    values,
    *,
    color,
    label,
    alpha: float = 0.3,
    lw: float = 2.0,
    gridsize: int = 256,
    max_points: int = 12000,
    seed: int = SEED,
):
    """Fast KDE-like density curve using histogram smoothing.

    scipy.stats.gaussian_kde is accurate but repeatedly expensive for this
    script. For figure-level validation panels, smoothed histograms are much
    faster and visually nearly indistinguishable.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return
    if x.size > max_points:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(x.size, max_points, replace=False)]

    lo, hi = np.percentile(x, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    if hi <= lo:
        return

    hist, edges = np.histogram(x, bins=gridsize, range=(lo, hi), density=True)
    density = gaussian_filter1d(hist.astype(np.float64), sigma=4.0, mode="nearest")
    grid = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(grid, density, color=color, lw=lw, label=label)
    ax.fill_between(grid, density, color=color, alpha=alpha)


# ---------------------------------------------------------------------------
# Noise-stage sparsity panel helper
# ---------------------------------------------------------------------------
def _generate_noise_stage_sparsities(
    real_s: np.ndarray, real_u: np.ndarray,
    sim_s: np.ndarray, sim_u: np.ndarray,
    seed: int = SEED,
) -> Dict[str, Dict[str, float]]:
    """Generate sparsity data for 3 noise stages + Real using a fast local run.

    Uses the cycle-bifurcation system (same as results_statistical_realism.py)
    so no PAGA/velocity rerun is needed.  This illustrates how each noise stage
    progressively pushes synthetic sparsity toward the real-data target.
    """
    import contextlib, io

    from data_generation_module import (
        CycleNode, GeneSynthesizer, GraphEdge, GraphNode, UniversalGraphSystem,
    )
    from experiments.results_junction_continuity import init_cycle_rim_cells
    from experiments.results_topology_primitives import simulate_langevin

    rng = np.random.default_rng(seed)

    # Build cycle-bifurcation system (fast, deterministic)
    cycle = CycleNode("Cycle", pos=[0.0, 0.0], radius=1.0, potential=-2.5,
                      sigma=0.25, omega=2.0, clockwise=False, rotation_axes=(0, 1))
    fate1 = GraphNode("Fate1", pos=[3.0,  1.0], potential=-3.2, sigma=0.30,
                      basin_strength=0.75, basin_sigma=0.5)
    fate2 = GraphNode("Fate2", pos=[3.0, -1.0], potential=-3.2, sigma=0.30,
                      basin_strength=0.75, basin_sigma=0.5)
    system_cb = UniversalGraphSystem([cycle, fate1, fate2],
                                     [GraphEdge(cycle, fate1), GraphEdge(cycle, fate2)],
                                     fusion_temp=0.01)

    # Quick trajectory (fewer steps for speed)
    N_CB = 2000
    z0 = init_cycle_rim_cells(N_CB, radius=1.0, noise=0.06, seed=seed)
    _, z_cells = simulate_langevin(system_cb, z0, n_steps=2000, dt=0.01,
                                   diffusion=0.006, save_every=10, seed=seed)
    z_t = np.asarray(z_cells, dtype=np.float32)

    # Gene config (noise genes only, fast)
    N_G = 500
    noise_cfgs = [
        dict(name="clean",      sampling="poisson", dropout_prob=0.0,  dropout_slope=0.5, simulate_counts=False),
        dict(name="poisson",    sampling="poisson", dropout_prob=0.0,  dropout_slope=0.5, simulate_counts=True),
        dict(name="nb_dropout", sampling="nb",      dropout_prob=0.60, dropout_slope=0.25, simulate_counts=True),
    ]

    result_sp: Dict[str, Dict[str, float]] = {}
    for cfg in noise_cfgs:
        synth = GeneSynthesizer(
            system=system_cb,
            n_genes_total=N_G,
            edge_strength=5.0, cycle_strength=3.0,
            background_noise=0.02,
            lib_size_mean=35000.0, lib_size_sd=0.35,
            sampling=cfg["sampling"],
            dropout_prob=cfg["dropout_prob"],
            dropout_slope=cfg["dropout_slope"],
            splicing_rate=1.0, degradation_rate=0.5,
            seed=seed,
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            kinetics = synth.generate_kinetics(z_t, dt=0.01,
                                               simulate_counts=cfg["simulate_counts"],
                                               scale=1.5)
        xs = kinetics["spliced"]
        xu = kinetics["unspliced"]
        if isinstance(xs, np.ndarray):
            xs, xu = xs, xu
        # Apply binomial downsampling for the nb_dropout stage (matching our pipeline)
        if cfg["name"] == "nb_dropout":
            xs = rng.binomial(xs.astype(int), 0.20).astype(float)
            xu = rng.binomial(xu.astype(int), 0.15).astype(float)
        result_sp[cfg["name"]] = {
            "spliced":   float((xs == 0).mean()),
            "unspliced": float((xu == 0).mean()),
        }

    # Real pancreas reference
    result_sp["real"] = {
        "spliced":   float((real_s == 0).mean()),
        "unspliced": float((real_u == 0).mean()),
    }
    return result_sp




def _noise_sparsity_cache_path(seed: int) -> Path:
    return DATA_DIR / f"noise_stage_sparsities_seed{seed}.pt"


def _load_noise_stage_sparsities(seed: int) -> Dict[str, Dict[str, float]] | None:
    path = _noise_sparsity_cache_path(seed)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save_noise_stage_sparsities(data: Dict[str, Dict[str, float]], seed: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _noise_sparsity_cache_path(seed).open("wb") as handle:
        pickle.dump(data, handle)

# ---------------------------------------------------------------------------
# Notebook 6-panel validation + noise row
# ---------------------------------------------------------------------------
def plot_notebook_validation(
    stats_real: Dict, stats_sim: Dict,
    real_s: np.ndarray, real_u: np.ndarray,
    sim_s: np.ndarray, sim_u: np.ndarray,
    seed: int = SEED,
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    c_real, c_sim = "#4A90E2", "#D0021B"
    kwargs_kde = dict(alpha=0.3, lw=2.0)

    fig, axes = plt.subplots(3, 3, figsize=FIGURE_MAIN_SIZE, dpi=FIGURE_MAIN_DPI)
    plt.subplots_adjust(hspace=0.40, wspace=0.30)

    # ── Row 0: A / B / C ──────────────────────────────────────────────────────

    # Panel A – Library Size: clip to 1st–99th percentile for tight x-axis
    ax = axes[0, 0]
    vals_r = np.log1p(stats_real["lib_size"])
    vals_s = np.log1p(stats_sim["lib_size"])
    _plot_kde(ax, vals_r, color=c_real, label="Real", **kwargs_kde)
    _plot_kde(ax, vals_s, color=c_sim,  label="Synthetic",       **kwargs_kde)
    ax.set_title("A. Library Size (Cell Depth)", fontsize=12)
    ax.set_xlabel("Log1p(Total Counts per Cell)", fontsize=10)
    lo = min(np.percentile(vals_r, 1), np.percentile(vals_s, 1)) - 0.1
    hi = max(np.percentile(vals_r, 99), np.percentile(vals_s, 99)) + 0.1
    ax.set_xlim(lo, hi)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel B – Fraction of genes detected (normalised by total gene count)
    ax = axes[0, 1]
    n_real_genes = real_s.shape[1]    # 27998
    n_sim_genes  = sim_s.shape[1]     # 12000
    frac_r = stats_real["n_genes"] / n_real_genes * 100
    frac_s = stats_sim["n_genes"]  / n_sim_genes  * 100
    _plot_kde(ax, frac_r, color=c_real,
              label=f"Real", **kwargs_kde)
    _plot_kde(ax, frac_s, color=c_sim,
              label=f"Synthetic", **kwargs_kde)
    ax.set_title("B. Fraction of Genes Detected", fontsize=12)
    ax.set_xlabel("Genes Detected per Cell (%)", fontsize=10)
    lo_b = min(np.percentile(frac_r, 1), np.percentile(frac_s, 1)) - 1
    hi_b = max(np.percentile(frac_r, 99), np.percentile(frac_s, 99)) + 1
    ax.set_xlim(max(0, lo_b), hi_b)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel C – Mean Expression
    ax = axes[0, 2]
    mask_r = stats_real["gene_mean"] > 0.01
    mask_s = stats_sim["gene_mean"]  > 0.01
    _plot_kde(ax, np.log1p(stats_real["gene_mean"][mask_r]), color=c_real, label="Real",      **kwargs_kde)
    _plot_kde(ax, np.log1p(stats_sim["gene_mean"][mask_s]),  color=c_sim,  label="Synthetic", **kwargs_kde)
    ax.set_title("C. Mean Expression Distribution", fontsize=12)
    ax.set_xlabel("Log1p(Gene Mean Expression)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    # ── Row 1: D / E / F ──────────────────────────────────────────────────────

    ax = axes[1, 0]
    idx_r = rng.choice(len(stats_real["gene_mean"]), min(2000, len(stats_real["gene_mean"])), replace=False)
    idx_s = rng.choice(len(stats_sim["gene_mean"]),  min(2000, len(stats_sim["gene_mean"])),  replace=False)
    ax.scatter(np.log1p(stats_real["gene_mean"][idx_r]), np.log1p(stats_real["gene_var"][idx_r]),
               s=10, c=c_real, alpha=0.3, label="Real")
    ax.scatter(np.log1p(stats_sim["gene_mean"][idx_s]),  np.log1p(stats_sim["gene_var"][idx_s]),
               s=10, c=c_sim,  alpha=0.3, label="Synthetic")
    mx = max(np.log1p(stats_real["gene_mean"]).max(), np.log1p(stats_sim["gene_mean"]).max())
    ax.plot(np.linspace(0, mx, 100), np.linspace(0, mx, 100), "k--", lw=1.5, label="Poisson", alpha=0.5)
    ax.set_title("D. Mean-Variance (Overdispersion)", fontsize=12)
    ax.set_xlabel("Log1p(Mean)", fontsize=10)
    ax.set_ylabel("Log1p(Variance)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    ax = axes[1, 1]
    idx_rd = rng.choice(len(stats_real["drop_mean"]), min(2000, len(stats_real["drop_mean"])), replace=False)
    idx_sd = rng.choice(len(stats_sim["drop_mean"]),  min(2000, len(stats_sim["drop_mean"])),  replace=False)
    ax.scatter(np.log1p(stats_real["drop_mean"][idx_rd]), stats_real["drop_rate"][idx_rd],
               s=10, c=c_real, alpha=0.3, label="Real")
    ax.scatter(np.log1p(stats_sim["drop_mean"][idx_sd]),  stats_sim["drop_rate"][idx_sd],
               s=10, c=c_sim,  alpha=0.3, label="Synthetic")
    ax.set_title("E. Dropout Rate vs Mean", fontsize=12)
    ax.set_xlabel("Log1p(Mean)", fontsize=10)
    ax.set_ylabel("Fraction of Zeros", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    ax = axes[1, 2]
    _plot_kde(ax, stats_real["u_frac"], color=c_real, label="Real",      **kwargs_kde)
    _plot_kde(ax, stats_sim["u_frac"],  color=c_sim,  label="Synthetic", **kwargs_kde)
    ax.set_title("F. Unspliced Fraction (Global Kinetics)", fontsize=12)
    ax.set_xlabel("Unspliced Reads / Total Reads", fontsize=10)
    ax.set_xlim(0, 0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    # ── Row 2: G / H (I removed; was redundant with G) ────────────────────────
    noise_sp = _load_noise_stage_sparsities(seed)
    if noise_sp is None:
        print("  Computing noise-stage sparsities...")
        noise_sp = _generate_noise_stage_sparsities(real_s, real_u, sim_s, sim_u, seed=seed)
        _save_noise_stage_sparsities(noise_sp, seed)
    else:
        print("  Loaded cached noise-stage sparsities.")

    # Inject the actual Figure4 pipeline result as the final stage
    noise_sp["figure4"] = {
        "spliced":   float((sim_s == 0).mean()),
        "unspliced": float((sim_u == 0).mean()),
    }

    # Panel G – Sparsity progression toward Real target (line chart)
    ax = axes[2, 0]
    stage_g = ["clean", "nb_dropout", "figure4"]
    x_g = np.arange(len(stage_g))
    x_lbl_g = ["Clean", "NB+Dropout\n(simple)", "NB+Dropout\n(Figure4)"]
    sp_s_g = [noise_sp[k]["spliced"]   for k in stage_g]
    sp_u_g = [noise_sp[k]["unspliced"] for k in stage_g]
    ax.plot(x_g, sp_s_g, "o-", color=c_real,    lw=2, markersize=7, label="Spliced")
    ax.plot(x_g, sp_u_g, "s-", color="#E87722", lw=2, markersize=7, label="Unspliced")
    ax.axhline(noise_sp["real"]["spliced"],   color=c_real,    ls="--", lw=1.5, alpha=0.65, label="Real S target")
    ax.axhline(noise_sp["real"]["unspliced"], color="#E87722", ls="--", lw=1.5, alpha=0.65, label="Real U target")
    ax.set_xticks(x_g); ax.set_xticklabels(x_lbl_g, fontsize=9)
    ax.set_ylabel("Sparsity (fraction zeros)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("G. Sparsity", fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel H – Gene Fano Factor distribution (variance/mean, overdispersion)
    ax = axes[2, 1]
    m_r = stats_real["gene_mean"]; v_r = stats_real["gene_var"]
    m_s = stats_sim["gene_mean"];  v_s = stats_sim["gene_var"]
    mask_fr = m_r > 0.5;  mask_fs = m_s > 0.5
    fano_r = np.log1p(v_r[mask_fr] / (m_r[mask_fr] + 1e-8))
    fano_s = np.log1p(v_s[mask_fs] / (m_s[mask_fs] + 1e-8))
    _plot_kde(ax, fano_r, color=c_real, label="Real",      **kwargs_kde)
    _plot_kde(ax, fano_s, color=c_sim,  label="Synthetic", **kwargs_kde)
    ax.axvline(np.log1p(1.0), color="k", ls="--", lw=1.2, alpha=0.5, label="Poisson (FF=1)")
    ax.set_title("H. Gene Fano Factor", fontsize=12)
    ax.set_xlabel("Log1p(Fano Factor = Var/Mean)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel I – Gene rank-frequency (Zipf / power-law structure)
    #   Sorts genes by mean expression and plots rank vs mean on log-log axes.
    #   Real scRNA-seq follows a roughly power-law decay; matching curves indicate
    #   that synthetic data reproduces the high-to-low expression landscape shape,
    #   orthogonal to per-gene statistics already shown in C / D / H.
    ax = axes[2, 2]
    gm_r = np.sort(stats_real["gene_mean"])[::-1]
    gm_s = np.sort(stats_sim["gene_mean"])[::-1]
    gm_r = gm_r[gm_r > 0]; gm_s = gm_s[gm_s > 0]
    rank_r = np.arange(1, len(gm_r) + 1) / len(gm_r)
    rank_s = np.arange(1, len(gm_s) + 1) / len(gm_s)
    ax.loglog(rank_r, gm_r, color=c_real, lw=1.8, alpha=0.85, label="Real")
    ax.loglog(rank_s, gm_s, color=c_sim,  lw=1.8, alpha=0.85, label="Synthetic")
    ax.set_title("I. Gene Rank–Frequency", fontsize=12)
    ax.set_xlabel("Normalized Gene Rank (high → low)", fontsize=10)
    ax.set_ylabel("Mean Expression", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3, which="both")

    return fig


# ---------------------------------------------------------------------------
# Supplementary statistical_realism panels
# ---------------------------------------------------------------------------
def draw_graph_topology(ax, system, z):
    pad = 0.8
    xlim = (z[:, 0].min() - pad, z[:, 0].max() + pad)
    ylim = (z[:, 1].min() - pad, z[:, 1].max() + pad)
    n = 60
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.asarray(np.stack([xx, yy], axis=-1), dtype=np.float32).reshape(-1, 2)
    u = system.potential_numpy(grid).reshape(n, n)
    ax.contourf(xx, yy, u, levels=20, cmap="Greys", alpha=0.08, zorder=0)
    for edge in system.edges:
        s, e = edge.start, edge.end
        ax.plot([s[0], e[0]], [s[1], e[1]], "k--", lw=0.8, alpha=0.45, zorder=1)
    for node in system.nodes:
        ax.scatter(node.pos[0], node.pos[1], s=35, c="0.35", edgecolors="k", linewidths=0.5, zorder=2)
    return xlim, ylim


def plot_supplementary(result: Figure4Result) -> plt.Figure:
    apply_publication_style()
    s, u, z = result.spliced, result.unspliced, result.z
    xr_s, xr_u = result.real_spliced, result.real_unspliced
    cell_stats = compute_cell_stats(s, u)
    gene_s = compute_gene_stats(s)

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SUPP_SIZE)

    ax = axes[0, 0]
    xlim, ylim = draw_graph_topology(ax, result.system, z)
    sc = ax.scatter(z[:, 0], z[:, 1], c=cell_stats["total_spliced"], s=4, cmap="plasma",
                    alpha=0.55, linewidths=0, rasterized=True, zorder=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title("G  Latent cells × spliced library size")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Total spliced")

    ax = axes[0, 1]
    ts, tu = cell_stats["total_spliced"], cell_stats["total_unspliced"]
    ax.scatter(ts, tu, s=2.5, alpha=0.25, c="0.35", linewidths=0, rasterized=True)
    lim = max(ts.max(), tu.max()) * 1.05
    ax.plot([0, lim], [0, lim], "C1", ls="--", lw=0.8, alpha=0.6, label="$U=S$")
    med = float(np.median(cell_stats["u_s_ratio"]))
    ax.set_title("H  Spliced vs unspliced (synthetic)")
    ax.set_xlabel("Total spliced")
    ax.set_ylabel("Total unspliced")
    ax.legend(fontsize=7)
    ax.text(0.05, 0.95, f"median $U/S$={med:.3f}", transform=ax.transAxes, va="top", fontsize=8)

    ax = axes[1, 0]
    labels = ["Real", "Synthetic"]
    sp_s = [sparsity(xr_s), sparsity(s)]
    sp_u = [sparsity(xr_u), sparsity(u)]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, sp_s, width=w, label="Spliced", color="C0")
    ax.bar(x + w / 2, sp_u, width=w, label="Unspliced", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sparsity (fraction zeros)")
    ax.set_ylim(0, 1.05)
    ax.set_title("I  Sparsity comparison")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    mask = gene_s["mean"] > 0.5
    ax.scatter(gene_s["mean"][mask], gene_s["var"][mask], s=3, alpha=0.35, c="C0", linewidths=0, rasterized=True)
    mx = np.logspace(np.log10(gene_s["mean"][mask].min()), np.log10(gene_s["mean"][mask].max()), 50)
    ax.plot(mx, mx, "0.35", ls="--", lw=1.0, label="Poisson")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("J  Gene mean–variance (spliced, synthetic)")
    ax.set_xlabel("Gene mean")
    ax.set_ylabel("Gene variance")
    ax.legend(fontsize=7)

    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    return fig


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def save_csvs(result: Figure4Result) -> List[Dict]:
    """Save CSVs using vectorized NumPy output instead of Python row loops."""
    s, u, z = result.spliced, result.unspliced, result.z
    xr_s, xr_u = result.real_spliced, result.real_unspliced
    cell_stats = compute_cell_stats(s, u)
    gene_s, gene_u = compute_gene_stats(s), compute_gene_stats(u)

    cell_arr = np.column_stack([
        np.arange(z.shape[0]),
        z[:, 0],
        z[:, 1],
        cell_stats["total_spliced"],
        cell_stats["total_unspliced"],
        cell_stats["u_s_ratio"],
    ])
    np.savetxt(
        DATA_DIR / "cell_statistics.csv",
        cell_arr,
        delimiter=",",
        header="cell_idx,z1,z2,total_spliced,total_unspliced,u_s_ratio",
        comments="",
        fmt=["%d", "%.8g", "%.8g", "%.8g", "%.8g", "%.8g"],
    )

    gene_arr = np.column_stack([
        np.arange(s.shape[1]),
        gene_s["mean"], gene_s["var"], gene_s["dropout"],
        gene_u["mean"], gene_u["var"], gene_u["dropout"],
    ])
    np.savetxt(
        DATA_DIR / "gene_statistics.csv",
        gene_arr,
        delimiter=",",
        header="gene_idx,mean_spliced,var_spliced,dropout_spliced,mean_unspliced,var_unspliced,dropout_unspliced",
        comments="",
        fmt=["%d", "%.8g", "%.8g", "%.8g", "%.8g", "%.8g", "%.8g"],
    )

    xr_s_sum = xr_s.sum(1)
    xr_u_sum = xr_u.sum(1)
    total_s = cell_stats["total_spliced"]
    total_u = cell_stats["total_unspliced"]
    rows = [
        {
            "dataset": "real_pancreas",
            "n_cells": xr_s.shape[0], "n_genes": xr_s.shape[1],
            "sparsity_spliced": sparsity(xr_s), "sparsity_unspliced": sparsity(xr_u),
            "median_total_spliced": float(np.median(xr_s_sum)),
            "median_total_unspliced": float(np.median(xr_u_sum)),
            "median_u_s_ratio": float(np.median(xr_u_sum / (xr_s_sum + 1e-8))),
            "median_u_frac": float(np.median(xr_u_sum / (xr_s_sum + xr_u_sum + 1e-8))),
        },
        {
            "dataset": "synthetic_figure4",
            "n_cells": s.shape[0], "n_genes": s.shape[1],
            "sparsity_spliced": sparsity(s), "sparsity_unspliced": sparsity(u),
            "median_total_spliced": float(np.median(total_s)),
            "median_total_unspliced": float(np.median(total_u)),
            "median_u_s_ratio": float(np.median(cell_stats["u_s_ratio"])),
            "median_u_frac": float(np.median(total_u / (total_s + total_u + 1e-8))),
        },
    ]
    fields = list(rows[0].keys())
    with (DATA_DIR / "count_statistics.csv").open("w", newline="") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")
    return rows




# ---------------------------------------------------------------------------
# Layer 1 loader: continuous kinetics (cached, never retouched by noise tuning)
# ---------------------------------------------------------------------------
def load_or_build_kinetics(
    h5ad_path: Path | None = None,
    *,
    force_rebuild: bool = False,
) -> dict:
    """Load Stage-1 kinetics cache if available; otherwise run the full slow pipeline.

    Returns the raw kinetics cache dict (``xs_s_base``, ``xs_u_base``, ``z``,
    ``pseudotime``, real pancreas matrices, system, config_dict).

    The observation-noise layer (lib_size / NB / dropout / downsampling) is NOT
    applied here.  Call :func:`apply_observation_noise` separately so that noise
    parameters can be tuned without rerunning Stage 1.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return load_or_build_kinetics_cache(KINETICS_CACHE_PATH, h5ad_path, force_rebuild)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(
    h5ad_path: Path | None = None,
    *,
    force_rebuild: bool = False,
    cache_only: bool = False,
    pp_cfg: PostProcessConfig = DEFAULT_PP_CONFIG,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Layer 1: continuous kinetics (slow, cached) ────────────────────────
    print("Loading/building continuous kinetics (Layer 1 – cached)…")
    kinetics_cache = load_or_build_kinetics(h5ad_path, force_rebuild=force_rebuild)

    if cache_only:
        print("Cache-only mode: Layer-1 kinetics saved. Observation noise not applied.")
        return

    # ── Layer 2: observation noise (fast, always re-runs) ─────────────────
    print("Applying observation noise (Layer 2: lib_size / NB / dropout / downsampling)…")
    result = apply_observation_noise(kinetics_cache, pp_cfg)

    print("Saving CSV statistics...")
    rows = save_csvs(result)

    print("Computing validation metrics...")
    stats_real = calc_6_metrics(result.real_spliced, result.real_unspliced)
    stats_sim = calc_6_metrics(result.spliced, result.unspliced)

    print("Plotting notebook 6-panel validation...")
    fig_main = plot_notebook_validation(
        stats_real, stats_sim,
        real_s=result.real_spliced, real_u=result.real_unspliced,
        sim_s=result.spliced, sim_u=result.unspliced,
    )
    save_figure(fig_main, FIG_DIR / "figure4_statistical_validation.png")

    print("Plotting supplementary metrics...")
    fig_supp = plot_supplementary(result)
    save_figure(fig_supp, FIG_DIR / "figure4_supplementary_metrics.png")

    sim_row = next(r for r in rows if r["dataset"] == "synthetic_figure4")
    real_row = next(r for r in rows if r["dataset"] == "real_pancreas")
    print(f"\nSynthetic: median U/S={sim_row['median_u_s_ratio']:.3f}, "
          f"sparsity S={sim_row['sparsity_spliced']:.3f}")
    print(f"Real:      median U/S={real_row['median_u_s_ratio']:.3f}, "
          f"sparsity S={real_row['sparsity_spliced']:.3f}")
    print(f"Saved {FIG_DIR / 'figure4_statistical_validation.png'}")
    print(f"Saved {FIG_DIR / 'figure4_supplementary_metrics.png'}")


def plot_only(
    h5ad_path: Path | None = None,
    pp_cfg: PostProcessConfig = DEFAULT_PP_CONFIG,
) -> None:
    """Reload Layer-1 kinetics cache, re-apply observation noise, redraw figures.

    Layer 1 (continuous kinetics) is loaded from cache and never rerun.
    Layer 2 (lib_size / NB / dropout / downsampling) is always re-applied – edit
    PostProcessConfig at the top of figure4_series_pipeline.py to retune noise.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not KINETICS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Kinetics cache not found: {KINETICS_CACHE_PATH}\n"
            "Run once without --plot-only to build the Stage-1 kinetics cache."
        )

    print(f"Loading Layer-1 kinetics cache: {KINETICS_CACHE_PATH}")
    kinetics_cache = load_or_build_kinetics_cache(KINETICS_CACHE_PATH, h5ad_path, force_rebuild=False)

    print("Re-applying observation noise (Layer 2)…")
    result = apply_observation_noise(kinetics_cache, pp_cfg)

    stats_real = calc_6_metrics(result.real_spliced, result.real_unspliced)
    stats_sim = calc_6_metrics(result.spliced, result.unspliced)

    save_figure(
        plot_notebook_validation(stats_real, stats_sim,
                                 real_s=result.real_spliced, real_u=result.real_unspliced,
                                 sim_s=result.spliced, sim_u=result.unspliced),
        FIG_DIR / "figure4_statistical_validation.png",
    )
    save_figure(plot_supplementary(result), FIG_DIR / "figure4_supplementary_metrics.png")
    print(f"Replotted figures in {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Figure4 notebook + supplementary stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Two-layer architecture:\n"
            "  Layer 1 – continuous_kinetics : cached in kinetics_cache.pt (slow, ~7 min)\n"
            "  Layer 2 – observation noise   : lib_size / NB / dropout / downsampling (fast, <10 s)\n\n"
            "To retune the noise layer, edit PostProcessConfig in figure4_series_pipeline.py,\n"
            "then rerun with --plot-only (Layer 1 is not touched)."
        ),
    )
    parser.add_argument("--plot-only", action="store_true",
                        help="Load cached Layer-1 kinetics, re-apply noise layer, redraw figures.")
    parser.add_argument("--cache-only", action="store_true",
                        help="Build Layer-1 kinetics cache only (no noise, no stats, no plots).")
    parser.add_argument("--force-kinetics", action="store_true",
                        help="Force rebuild of Layer-1 kinetics cache (slow, ~7 min).")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_PANCREAS_H5AD,
                        help="Pancreas h5ad path")
    args = parser.parse_args()

    h5ad_arg = args.h5ad
    if args.plot_only:
        plot_only(h5ad_arg)
    else:
        run(h5ad_arg, force_rebuild=args.force_kinetics, cache_only=args.cache_only)
