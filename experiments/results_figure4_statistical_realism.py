"""
Figure4Series.ipynb pipeline + supplementary statistical_realism metrics.

TWO-STAGE PIPELINE
==================
Stage 1  (slow ~7 min, auto-cached in kinetics_cache.pt)
  PAGA topology → GraphSimulator → GeneSynthesizer baseline kinetics
  Only re-runs when simulation hyper-parameters change.

Stage 2  (fast <10 s, always re-applied)  ← observation noise layer
  Noise model: lib_size_mean, lib_size_sd, sampling, nb_theta,
               dropout_prob, dropout_slope, downsample_spliced, downsample_unspliced

Usage (from repo root):
    # Full run (Stage 1 + 2):
    NUMBA_CACHE_DIR=/tmp/numba_cache python experiments/results_figure4_statistical_realism.py

    # Fast retune (Stage 2 only, requires existing kinetics_cache.pt):
    python experiments/results_figure4_statistical_realism.py --tune

    # Re-plot from saved count matrices (no re-simulation):
    python experiments/results_figure4_statistical_realism.py --plot-only

    # Force-rebuild Stage-1 cache:
    python experiments/results_figure4_statistical_realism.py --rebuild-cache
"""


import argparse
import pickle
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figure4_series_pipeline import (  # noqa: E402
    DEFAULT_PANCREAS_H5AD,
    KINETICS_CACHE_PATH,
    PostProcessConfig,
    SEED,
    Figure4Result,
    load_or_build_kinetics_cache,
    run_figure4_pipeline,
    _apply_count_noise,
    _add_noise_genes,
    TARGET_N_GENES,
)
from experiments.figure_style import apply_publication_style, save_figure  # noqa: E402

# ---------------------------------------------------------------------------
# Edit noise-model parameters here – no need to re-run Stage 1
# ---------------------------------------------------------------------------
PP_CONFIG = PostProcessConfig(
    lib_size_mean        = 22000.0,
    lib_size_sd          = 0.40,
    sampling             = "nb",
    nb_theta             = 8.0,
    dropout_prob         = 0.68,
    dropout_slope        = 0.20,
    cell_size_sigma      = 0.18,
    downsample_spliced   = 0.12,
    downsample_unspliced = 0.10,
    seed                 = SEED,
)

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
    x_total = s + u
    lib_size = x_total.sum(axis=1)
    n_genes = (x_total > 0).sum(axis=1)
    gene_mean = x_total.mean(axis=0)
    gene_var = x_total.var(axis=0)
    mask = gene_mean > 0
    mean_for_drop = gene_mean[mask]
    drop_rate = (x_total[:, mask] == 0).sum(axis=0) / x_total.shape[0]
    u_sum = u.sum(axis=1)
    total_sum = lib_size.copy()
    total_sum[total_sum == 0] = 1
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
    return {"mean": x.mean(axis=0), "var": x.var(axis=0), "dropout": (x == 0).mean(axis=0)}


def sparsity(x: np.ndarray) -> float:
    return float((x == 0).mean())


def _plot_kde(ax, values, *, color, label, alpha: float = 0.3, lw: float = 2.0):
    """KDE curve matching seaborn kdeplot(fill=True, alpha=0.3, linewidth=2)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return
    kde = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), 400)
    density = kde(grid)
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
    import io, contextlib, tempfile, yaml

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


# ---------------------------------------------------------------------------
# Notebook 6-panel validation + noise row
# ---------------------------------------------------------------------------
def plot_notebook_validation(
    stats_real: Dict, stats_sim: Dict,
    real_s: np.ndarray, real_u: np.ndarray,
    sim_s: np.ndarray, sim_u: np.ndarray,
    seed: int = SEED,
) -> plt.Figure:
    np.random.seed(seed)
    c_real, c_sim = "#4A90E2", "#D0021B"
    kwargs_kde = dict(alpha=0.3, lw=2.0)

    fig, axes = plt.subplots(3, 3, figsize=FIGURE_MAIN_SIZE, dpi=FIGURE_MAIN_DPI)
    plt.subplots_adjust(hspace=0.40, wspace=0.30)

    # ── Row 0: A / B / C ──────────────────────────────────────────────────────

    # Panel A – Library Size: clip to 1st–99th percentile for tight x-axis
    ax = axes[0, 0]
    vals_r = np.log1p(stats_real["lib_size"])
    vals_s = np.log1p(stats_sim["lib_size"])
    _plot_kde(ax, vals_r, color=c_real, label="Real (Pancreas)", **kwargs_kde)
    _plot_kde(ax, vals_s, color=c_sim,  label="Synthetic",       **kwargs_kde)
    ax.set_title("A. Library Size (Cell Depth)", fontweight="bold", fontsize=12)
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
              label=f"Real ({n_real_genes:,} genes)", **kwargs_kde)
    _plot_kde(ax, frac_s, color=c_sim,
              label=f"Synthetic ({n_sim_genes:,} genes)", **kwargs_kde)
    ax.set_title("B. Fraction of Genes Detected", fontweight="bold", fontsize=12)
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
    ax.set_title("C. Mean Expression Distribution", fontweight="bold", fontsize=12)
    ax.set_xlabel("Log1p(Gene Mean Expression)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    # ── Row 1: D / E / F ──────────────────────────────────────────────────────

    ax = axes[1, 0]
    idx_r = np.random.choice(len(stats_real["gene_mean"]), min(2000, len(stats_real["gene_mean"])), replace=False)
    idx_s = np.random.choice(len(stats_sim["gene_mean"]),  min(2000, len(stats_sim["gene_mean"])),  replace=False)
    ax.scatter(np.log1p(stats_real["gene_mean"][idx_r]), np.log1p(stats_real["gene_var"][idx_r]),
               s=10, c=c_real, alpha=0.3, label="Real")
    ax.scatter(np.log1p(stats_sim["gene_mean"][idx_s]),  np.log1p(stats_sim["gene_var"][idx_s]),
               s=10, c=c_sim,  alpha=0.3, label="Synthetic")
    mx = max(np.log1p(stats_real["gene_mean"]).max(), np.log1p(stats_sim["gene_mean"]).max())
    ax.plot(np.linspace(0, mx, 100), np.linspace(0, mx, 100), "k--", lw=1.5, label="Poisson", alpha=0.5)
    ax.set_title("D. Mean-Variance (Overdispersion)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Log1p(Mean)", fontsize=10)
    ax.set_ylabel("Log1p(Variance)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    ax = axes[1, 1]
    idx_rd = np.random.choice(len(stats_real["drop_mean"]), min(2000, len(stats_real["drop_mean"])), replace=False)
    idx_sd = np.random.choice(len(stats_sim["drop_mean"]),  min(2000, len(stats_sim["drop_mean"])),  replace=False)
    ax.scatter(np.log1p(stats_real["drop_mean"][idx_rd]), stats_real["drop_rate"][idx_rd],
               s=10, c=c_real, alpha=0.3, label="Real")
    ax.scatter(np.log1p(stats_sim["drop_mean"][idx_sd]),  stats_sim["drop_rate"][idx_sd],
               s=10, c=c_sim,  alpha=0.3, label="Synthetic")
    ax.set_title("E. Dropout Rate vs Mean", fontweight="bold", fontsize=12)
    ax.set_xlabel("Log1p(Mean)", fontsize=10)
    ax.set_ylabel("Fraction of Zeros", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    ax = axes[1, 2]
    _plot_kde(ax, stats_real["u_frac"], color=c_real, label="Real",      **kwargs_kde)
    _plot_kde(ax, stats_sim["u_frac"],  color=c_sim,  label="Synthetic", **kwargs_kde)
    ax.set_title("F. Unspliced Fraction (Global Kinetics)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Unspliced Reads / Total Reads", fontsize=10)
    ax.set_xlim(0, 0.5)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

    # ── Row 2: G / H / I ──────────────────────────────────────────────────────
    print("  Computing noise-stage sparsities...")
    noise_sp = _generate_noise_stage_sparsities(real_s, real_u, sim_s, sim_u, seed=seed)

    # Inject the actual Figure4 pipeline result as the 4th stage
    noise_sp["figure4"] = {
        "spliced":   float((sim_s == 0).mean()),
        "unspliced": float((sim_u == 0).mean()),
    }

    # Panel G – Sparsity under noise settings (no Poisson stage)
    ax = axes[2, 0]
    stage_g = ["clean", "nb_dropout", "figure4", "real"]
    x_labels_g = ["Clean\n(no noise)", "NB+Dropout\n(simple)", "NB+Dropout\n(Figure4)", "Real\n(Pancreas)"]
    sp_s_g = [noise_sp[k]["spliced"]   for k in stage_g]
    sp_u_g = [noise_sp[k]["unspliced"] for k in stage_g]
    x = np.arange(len(stage_g))
    w = 0.35
    bars_s = ax.bar(x - w/2, sp_s_g, width=w, label="Spliced",   color=c_real, alpha=0.85)
    bars_u = ax.bar(x + w/2, sp_u_g, width=w, label="Unspliced", color="#E87722", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(x_labels_g, fontsize=8)
    ax.set_ylabel("Sparsity (fraction zeros)", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("G. Sparsity under Noise Settings", fontweight="bold", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3, axis="y")
    for bar in list(bars_s) + list(bars_u):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=7)

    # Panel H – Progression toward Real (line chart, 4 stages including Figure4)
    ax = axes[2, 1]
    stage_h = ["clean", "nb_dropout", "figure4"]
    x_h = np.arange(len(stage_h))
    x_lbl_h = ["Clean", "NB+Dropout\n(simple)", "NB+Dropout\n(Figure4)"]
    sp_s_h = [noise_sp[k]["spliced"]   for k in stage_h]
    sp_u_h = [noise_sp[k]["unspliced"] for k in stage_h]
    ax.plot(x_h, sp_s_h, "o-", color=c_real,    lw=2, markersize=7, label="Spliced")
    ax.plot(x_h, sp_u_h, "s-", color="#E87722", lw=2, markersize=7, label="Unspliced")
    ax.axhline(noise_sp["real"]["spliced"],   color=c_real,    ls="--", lw=1.5, alpha=0.65, label="Real S target")
    ax.axhline(noise_sp["real"]["unspliced"], color="#E87722", ls="--", lw=1.5, alpha=0.65, label="Real U target")
    ax.set_xticks(x_h); ax.set_xticklabels(x_lbl_h, fontsize=9)
    ax.set_ylabel("Sparsity (fraction zeros)", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("H. Sparsity Converging to Real Target", fontweight="bold", fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.3)

    # Panel I – Gene Fano Factor distribution (variance/mean, overdispersion)
    ax = axes[2, 2]
    # Filter genes with mean > 0.5 to avoid noise-dominated estimates
    m_r = stats_real["gene_mean"]; v_r = stats_real["gene_var"]
    m_s = stats_sim["gene_mean"];  v_s = stats_sim["gene_var"]
    mask_fr = m_r > 0.5;  mask_fs = m_s > 0.5
    fano_r = np.log1p(v_r[mask_fr] / (m_r[mask_fr] + 1e-8))
    fano_s = np.log1p(v_s[mask_fs] / (m_s[mask_fs] + 1e-8))
    _plot_kde(ax, fano_r, color=c_real, label="Real",      **kwargs_kde)
    _plot_kde(ax, fano_s, color=c_sim,  label="Synthetic", **kwargs_kde)
    ax.axvline(np.log1p(1.0), color="k", ls="--", lw=1.2, alpha=0.5, label="Poisson (FF=1)")
    ax.set_title("I. Gene Fano Factor (Overdispersion Index)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Log1p(Fano Factor = Var/Mean)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.3)

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
    s, u, z = result.spliced, result.unspliced, result.z
    xr_s, xr_u = result.real_spliced, result.real_unspliced
    cell_stats = compute_cell_stats(s, u)
    gene_s, gene_u = compute_gene_stats(s), compute_gene_stats(u)

    with (DATA_DIR / "cell_statistics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cell_idx", "z1", "z2", "total_spliced", "total_unspliced", "u_s_ratio"])
        w.writeheader()
        for i in range(z.shape[0]):
            w.writerow({
                "cell_idx": i, "z1": z[i, 0], "z2": z[i, 1],
                "total_spliced": cell_stats["total_spliced"][i],
                "total_unspliced": cell_stats["total_unspliced"][i],
                "u_s_ratio": cell_stats["u_s_ratio"][i],
            })

    with (DATA_DIR / "gene_statistics.csv").open("w", newline="") as f:
        fields = ["gene_idx", "mean_spliced", "var_spliced", "dropout_spliced",
                  "mean_unspliced", "var_unspliced", "dropout_unspliced"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g in range(s.shape[1]):
            w.writerow({
                "gene_idx": g,
                "mean_spliced": gene_s["mean"][g], "var_spliced": gene_s["var"][g],
                "dropout_spliced": gene_s["dropout"][g],
                "mean_unspliced": gene_u["mean"][g], "var_unspliced": gene_u["var"][g],
                "dropout_unspliced": gene_u["dropout"][g],
            })

    rows = [
        {
            "dataset": "real_pancreas",
            "n_cells": xr_s.shape[0], "n_genes": xr_s.shape[1],
            "sparsity_spliced": sparsity(xr_s), "sparsity_unspliced": sparsity(xr_u),
            "median_total_spliced": float(np.median(xr_s.sum(1))),
            "median_total_unspliced": float(np.median(xr_u.sum(1))),
            "median_u_s_ratio": float(np.median(xr_u.sum(1) / (xr_s.sum(1) + 1e-8))),
            "median_u_frac": float(np.median(xr_u.sum(1) / (xr_s.sum(1) + xr_u.sum(1) + 1e-8))),
        },
        {
            "dataset": "synthetic_figure4",
            "n_cells": s.shape[0], "n_genes": s.shape[1],
            "sparsity_spliced": sparsity(s), "sparsity_unspliced": sparsity(u),
            "median_total_spliced": float(np.median(cell_stats["total_spliced"])),
            "median_total_unspliced": float(np.median(cell_stats["total_unspliced"])),
            "median_u_s_ratio": float(np.median(cell_stats["u_s_ratio"])),
            "median_u_frac": float(np.median(calc_6_metrics(s, u)["u_frac"])),
        },
    ]
    with (DATA_DIR / "count_statistics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Main helpers
# ---------------------------------------------------------------------------
def _save_and_plot(result: Figure4Result) -> None:
    """Save count matrices, CSVs, and both figures."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving CSV statistics...")
    rows = save_csvs(result)
    pickle.dump(
        {
            "z":             np.asarray(result.z,             dtype=np.float32),
            "spliced":       np.asarray(result.spliced),
            "unspliced":     np.asarray(result.unspliced),
            "real_spliced":  np.asarray(result.real_spliced),
            "real_unspliced":np.asarray(result.real_unspliced),
        },
        open(str(DATA_DIR / "count_matrices.pt"), "wb"),
    )

    stats_real = calc_6_metrics(result.real_spliced, result.real_unspliced)
    stats_sim  = calc_6_metrics(result.spliced,      result.unspliced)

    print("Plotting notebook 6-panel validation...")
    fig_main = plot_notebook_validation(
        stats_real, stats_sim,
        real_s=result.real_spliced, real_u=result.real_unspliced,
        sim_s =result.spliced,      sim_u =result.unspliced,
    )
    save_figure(fig_main, FIG_DIR / "figure4_statistical_validation.png")

    print("Plotting supplementary metrics...")
    save_figure(plot_supplementary(result), FIG_DIR / "figure4_supplementary_metrics.png")

    sim_row  = next(r for r in rows if r["dataset"] == "synthetic_figure4")
    real_row = next(r for r in rows if r["dataset"] == "real_pancreas")
    print(f"\nSynthetic: median U/S={sim_row['median_u_s_ratio']:.3f}, "
          f"sparsity S={sim_row['sparsity_spliced']:.3f}")
    print(f"Real:      median U/S={real_row['median_u_s_ratio']:.3f}, "
          f"sparsity S={real_row['sparsity_spliced']:.3f}")
    print(f"Saved {FIG_DIR / 'figure4_statistical_validation.png'}")
    print(f"Saved {FIG_DIR / 'figure4_supplementary_metrics.png'}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run(h5ad_path: Path | None = None, force_rebuild: bool = False) -> None:
    """Full run: Stage 1 (cached) + Stage 2 + plots."""
    print("Figure4Series.ipynb pipeline...")
    result = run_figure4_pipeline(
        h5ad_path=h5ad_path,
        pp_cfg=PP_CONFIG,
        force_rebuild=force_rebuild,
    )
    _save_and_plot(result)


def tune() -> None:
    """Fast re-run: reload Stage-1 cache and re-apply Stage-2 noise model.

    Edit PP_CONFIG at the top of this file, then run:
        python experiments/results_figure4_statistical_realism.py --tune
    Completes in <10 seconds.
    """
    if not KINETICS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No kinetics cache found at {KINETICS_CACHE_PATH}.\n"
            "Run the full pipeline first (without --tune)."
        )
    print(f"[--tune] Loading kinetics cache from {KINETICS_CACHE_PATH.name} …")
    cache       = load_or_build_kinetics_cache(KINETICS_CACHE_PATH)
    xs_s_base   = cache["xs_s_base"].astype(np.float64)
    xs_u_base   = cache["xs_u_base"].astype(np.float64)
    z           = cache["z"]
    pseudotime  = cache["pseudotime"]
    xr_s        = cache["xr_s"].astype(np.float64)
    xr_u        = cache["xr_u"].astype(np.float64)
    system      = cache["system"]
    config_dict = cache["config_dict"]

    print(f"[--tune] Applying noise model (PP_CONFIG: "
          f"lib_size_mean={PP_CONFIG.lib_size_mean:.0f}, "
          f"dropout_prob={PP_CONFIG.dropout_prob:.2f}, "
          f"cell_size_σ={PP_CONFIG.cell_size_sigma:.2f}) …")
    xs_s, xs_u = _apply_count_noise(xs_s_base, xs_u_base, PP_CONFIG)
    xs_s, xs_u = _add_noise_genes(xs_s, xs_u, pseudotime, PP_CONFIG)

    # Quick stats printout for immediate feedback
    lib_s  = (xs_s + xs_u).sum(1)
    ng_s   = (xs_s + xs_u > 0).sum(1)
    lib_r  = (xr_s + xr_u).sum(1)
    ng_r   = (xr_s + xr_u > 0).sum(1)
    print(f"  Synthetic: lib_size={np.median(lib_s):.0f}  "
          f"log1p_std={np.log1p(lib_s).std():.3f}  "
          f"det_pct={ng_s.mean()/xs_s.shape[1]*100:.1f}%  "
          f"sparsity_S={(xs_s==0).mean():.3f}")
    print(f"  Real:      lib_size={np.median(lib_r):.0f}  "
          f"log1p_std={np.log1p(lib_r).std():.3f}  "
          f"det_pct={ng_r.mean()/xr_s.shape[1]*100:.1f}%  "
          f"sparsity_S={(xr_s==0).mean():.3f}")

    result = Figure4Result(
        system=system, z=z.astype(np.float64),
        spliced=xs_s, unspliced=xs_u,
        real_spliced=xr_s, real_unspliced=xr_u,
        pseudotime=pseudotime.astype(np.float64),
        config_dict=config_dict,
    )
    _save_and_plot(result)


def plot_only() -> None:
    """Re-plot from saved count_matrices.pt (no re-simulation)."""
    pt = DATA_DIR / "count_matrices.pt"
    if not pt.exists():
        raise FileNotFoundError("Run the full pipeline first.")
    data = pickle.load(open(str(pt), "rb"))
    s    = data["spliced"]
    u    = data["unspliced"]
    z    = data["z"]
    xr_s = data["real_spliced"]
    xr_u = data["real_unspliced"]

    # Load system from kinetics cache if available, else rebuild
    if KINETICS_CACHE_PATH.exists():
        cache  = pickle.load(open(str(KINETICS_CACHE_PATH), "rb"))
        system = cache["system"]
        config_dict = cache.get("config_dict", {})
    else:
        from experiments.figure4_series_pipeline import (
            build_system_from_paga_topology, load_pancreas_reference, prepare_adata_for_paga,
        )
        _, _, adata = load_pancreas_reference()
        system      = build_system_from_paga_topology(prepare_adata_for_paga(adata))
        config_dict = {}

    result = Figure4Result(
        system=system, z=z, spliced=s, unspliced=u,
        real_spliced=xr_s, real_unspliced=xr_u,
        pseudotime=np.zeros(len(z)), config_dict=config_dict,
    )
    stats_real = calc_6_metrics(xr_s, xr_u)
    stats_sim  = calc_6_metrics(s, u)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(
        plot_notebook_validation(stats_real, stats_sim,
                                 real_s=xr_s, real_u=xr_u, sim_s=s, sim_u=u),
        FIG_DIR / "figure4_statistical_validation.png",
    )
    save_figure(plot_supplementary(result), FIG_DIR / "figure4_supplementary_metrics.png")
    print(f"Replotted figures in {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Figure4 notebook + supplementary stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  (default)       Full run – Stage 1 (cached) + Stage 2 + plots
  --tune          Fast re-tune – reload Stage-1 cache, re-apply Stage-2 noise model (<10 s)
  --plot-only     Re-plot from saved count_matrices.pt
  --rebuild-cache Force-rebuild Stage-1 kinetics cache
        """,
    )
    parser.add_argument("--tune",          action="store_true", help="Fast Stage-2 retune only")
    parser.add_argument("--plot-only",     action="store_true", help="Replot from saved matrices")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild Stage-1 cache")
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_PANCREAS_H5AD)
    args = parser.parse_args()

    if args.tune:
        tune()
    elif args.plot_only:
        plot_only()
    else:
        run(args.h5ad if args.h5ad.exists() else None, force_rebuild=args.rebuild_cache)
