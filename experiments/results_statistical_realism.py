"""
Figure 6: TopoVeloGen reproduces statistical features of single-cell count data.

Generates count matrices from cycle-bifurcation latent cells via GeneSynthesizer
observation model and validates count statistics in a merged 4×3 figure:
original panels A–F (topology, observation model, sparsity sweep) plus
Figure4Series validation panels G–L (synthetic NB+dropout vs Poisson).

Usage (from repo root):
    python experiments/results_statistical_realism.py
    python experiments/results_statistical_realism.py --preset figure4
"""


import csv
import pickle
import io
import contextlib
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_module import (  # noqa: E402
    CycleNode,
    GeneSynthesizer,
    GraphEdge,
    GraphNode,
    UniversalGraphSystem,
)
from experiments.figure_style import apply_publication_style, save_figure  # noqa: E402
from experiments.results_junction_continuity import init_cycle_rim_cells  # noqa: E402
from experiments.results_topology_primitives import simulate_langevin  # noqa: E402

USE_FIGURE4_PIPELINE = False
VALIDATION_LABEL_PRIMARY = "NB+dropout"
VALIDATION_LABEL_REFERENCE = "Poisson"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 1
N_CELLS = 3000
N_GENES = 1000
N_STEPS = 4500
DT = 0.01
DIFFUSION = 0.006
ALPHA_SCALE = 1.5

OUT_DIR = REPO_ROOT / "results" / "statistical_realism"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

# Figure4Series.ipynb observation-model parameters (fast path: same topology/Langevin)
FIGURE4_OBS = dict(
    dropout_prob=0.25,
    dropout_slope=0.2,
    lib_size_mean=50000.0,
    lib_size_sd=0.2,
    background_noise=0.1,
)
FIGURE4_CELL_SIZE_SIGMA = 0.35
FIGURE4_CAPTURE_SPLICED = 0.20
FIGURE4_CAPTURE_UNSPLICED = 0.15
FIGURE4_NOISE_ALPHA_RANGE = (0.005, 0.05)

APPLY_POSTPROCESS = False
CELL_SIZE_SIGMA = 0.0
CAPTURE_RATE_SPLICED = 1.0
CAPTURE_RATE_UNSPLICED = 1.0
NOISE_ALPHA_RANGE = (0.01, 0.08)


@dataclass
class SynthConfig:
    name: str
    sampling: str = "nb"
    nb_dispersion: float = 1.5
    dropout_prob: float = 0.2
    dropout_slope: float = 0.5
    lib_size_mean: float = 10000.0
    lib_size_sd: float = 0.4
    background_noise: float = 0.02
    simulate_counts: bool = True


MAIN_CONFIG = SynthConfig(name="nb_dropout")
SWEEP_CONFIGS = [
    SynthConfig(name="clean", sampling="poisson", dropout_prob=0.0, simulate_counts=False),
    SynthConfig(name="poisson", sampling="poisson", dropout_prob=0.0, simulate_counts=True),
    SynthConfig(
        name="nb_dropout",
        sampling="nb",
        nb_dispersion=1.5,
        dropout_prob=0.2,
        dropout_slope=0.5,
        simulate_counts=True,
    ),
]


def configure_run(preset: str, out_dir: Optional[Path] = None) -> None:
    """Apply preset parameters (figure4 = full Figure4Series.ipynb pipeline)."""
    global OUT_DIR, DATA_DIR, FIG_DIR
    global MAIN_CONFIG, SWEEP_CONFIGS
    global APPLY_POSTPROCESS, CELL_SIZE_SIGMA
    global CAPTURE_RATE_SPLICED, CAPTURE_RATE_UNSPLICED, NOISE_ALPHA_RANGE
    global USE_FIGURE4_PIPELINE, VALIDATION_LABEL_PRIMARY, VALIDATION_LABEL_REFERENCE

    if preset == "figure4":
        OUT_DIR = out_dir or (REPO_ROOT / "results" / "statistical_realism_figure4")
        USE_FIGURE4_PIPELINE = True
        VALIDATION_LABEL_PRIMARY = "Synthetic"
        VALIDATION_LABEL_REFERENCE = "Real (Pancreas)"
        APPLY_POSTPROCESS = False  # post-processing handled inside figure4 pipeline
    elif out_dir is not None:
        OUT_DIR = out_dir
    else:
        return

    DATA_DIR = OUT_DIR / "data"
    FIG_DIR = OUT_DIR / "figures"


# ---------------------------------------------------------------------------
# Topology & cells
# ---------------------------------------------------------------------------
def build_cycle_bifurcation_system() -> UniversalGraphSystem:
    cycle = CycleNode(
        "Cycle",
        pos=[0.0, 0.0],
        radius=1.0,
        potential=-2.5,
        sigma=0.25,
        omega=2.0,
        clockwise=False,
        rotation_axes=(0, 1),
    )
    fate1 = GraphNode(
        "Fate1",
        pos=[3.0, 1.0],
        potential=-3.2,
        sigma=0.30,
        basin_strength=0.75,
        basin_sigma=0.5,
    )
    fate2 = GraphNode(
        "Fate2",
        pos=[3.0, -1.0],
        potential=-3.2,
        sigma=0.30,
        basin_strength=0.75,
        basin_sigma=0.5,
    )
    edges = [GraphEdge(cycle, fate1), GraphEdge(cycle, fate2)]
    return UniversalGraphSystem([cycle, fate1, fate2], edges, fusion_temp=0.01)


def sample_final_cells(system: UniversalGraphSystem, n_cells: int) -> np.ndarray:
    z0 = init_cycle_rim_cells(n_cells, radius=1.0, noise=0.06, seed=SEED)
    _, z_final = simulate_langevin(
        system,
        z0,
        n_steps=N_STEPS,
        dt=DT,
        diffusion=DIFFUSION,
        save_every=10,
        seed=SEED,
    )
    return z_final


# ---------------------------------------------------------------------------
# GeneSynthesizer setup
# ---------------------------------------------------------------------------
def setup_synthesizer(system: UniversalGraphSystem, cfg: SynthConfig) -> GeneSynthesizer:
    synth = GeneSynthesizer(
        system=system,
        n_genes_total=N_GENES,
        edge_strength=5.0,
        cycle_strength=3.0,
        background_noise=cfg.background_noise,
        lib_size_mean=cfg.lib_size_mean,
        lib_size_sd=cfg.lib_size_sd,
        sampling=cfg.sampling,
        nb_dispersion=cfg.nb_dispersion,
        dropout_prob=cfg.dropout_prob,
        dropout_slope=cfg.dropout_slope,
        splicing_rate=1.0,
        degradation_rate=0.5,
        seed=SEED,
    )

    # ~80 signal genes + noise fill
    synth.add_trend_gene(edge_idx=0, n_genes=15, pattern="sigmoid", include_cycle_rim=True)
    synth.add_trend_gene(edge_idx=1, n_genes=15, pattern="sigmoid", include_cycle_rim=True)
    synth.add_trend_gene(edge_idx=0, n_genes=10, pattern="impulse", include_cycle_rim=True)
    synth.add_trend_gene(edge_idx=1, n_genes=10, pattern="impulse", include_cycle_rim=True)
    phases = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    synth.add_cycle_gene(node_id="Cycle", n_genes=30, phase=phases.tolist(), angular_sigma=0.5)
    synth.add_noise_gene(n_genes=N_GENES - synth.current_gene_idx, alpha_range=NOISE_ALPHA_RANGE)

    return synth


def apply_observation_postprocess(s: np.ndarray, u: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Figure4Series steps 2.5–2.6: cell-size scaling + binomial capture downsampling."""
    if not APPLY_POSTPROCESS:
        return s, u

    rng = np.random.default_rng(seed)
    n_cells = s.shape[0]
    mu = -CELL_SIZE_SIGMA**2 / 2
    factors = np.clip(rng.lognormal(mu, CELL_SIZE_SIGMA, n_cells), 0.3, 2.5)
    s = s * factors[:, np.newaxis]
    u = u * factors[:, np.newaxis]

    s = rng.binomial(s.astype(int), CAPTURE_RATE_SPLICED).astype(np.float64)
    u = rng.binomial(u.astype(int), CAPTURE_RATE_UNSPLICED).astype(np.float64)
    return s, u


def generate_counts(
    synth: GeneSynthesizer,
    z: np.ndarray,
    cfg: SynthConfig,
) -> Dict[str, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = synth.generate_kinetics(
            z,
            dt=DT,
            scale=ALPHA_SCALE,
            simulate_counts=cfg.simulate_counts,
        )
    return result


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def to_numpy(x: np.ndarray) -> np.ndarray:
    return x if isinstance(x, np.ndarray) else np.asarray(x)


def compute_cell_stats(S: np.ndarray, U: np.ndarray) -> Dict[str, np.ndarray]:
    total_s = S.sum(axis=1)
    total_u = U.sum(axis=1)
    ratio = total_u / (total_s + 1e-8)
    return {"total_spliced": total_s, "total_unspliced": total_u, "u_s_ratio": ratio}


def compute_gene_stats(X: np.ndarray) -> Dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    dropout = (X == 0).mean(axis=0)
    return {"mean": mean, "var": var, "dropout": dropout}


def sparsity(X: np.ndarray) -> float:
    return float((X == 0).mean())


def calc_6_metrics(s: np.ndarray, u: np.ndarray) -> Dict[str, np.ndarray]:
    """Six validation metrics from Figure4Series.ipynb (S+U combined counts)."""
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


def save_cell_stats_csv(path: Path, z: np.ndarray, cell_stats: Dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cell_idx", "z1", "z2", "total_spliced", "total_unspliced", "u_s_ratio"],
        )
        writer.writeheader()
        for i in range(z.shape[0]):
            writer.writerow(
                {
                    "cell_idx": i,
                    "z1": float(z[i, 0]),
                    "z2": float(z[i, 1]),
                    "total_spliced": float(cell_stats["total_spliced"][i]),
                    "total_unspliced": float(cell_stats["total_unspliced"][i]),
                    "u_s_ratio": float(cell_stats["u_s_ratio"][i]),
                }
            )


def save_gene_stats_csv(path: Path, gene_stats_s: Dict[str, np.ndarray], gene_stats_u: Dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gene_idx",
                "mean_spliced",
                "var_spliced",
                "dropout_spliced",
                "mean_unspliced",
                "var_unspliced",
                "dropout_unspliced",
            ],
        )
        writer.writeheader()
        for g in range(len(gene_stats_s["mean"])):
            writer.writerow(
                {
                    "gene_idx": g,
                    "mean_spliced": float(gene_stats_s["mean"][g]),
                    "var_spliced": float(gene_stats_s["var"][g]),
                    "dropout_spliced": float(gene_stats_s["dropout"][g]),
                    "mean_unspliced": float(gene_stats_u["mean"][g]),
                    "var_unspliced": float(gene_stats_u["var"][g]),
                    "dropout_unspliced": float(gene_stats_u["dropout"][g]),
                }
            )


def save_count_summary_csv(path: Path, rows: List[Dict]) -> None:
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plotting — original panels + Figure4Series validation (merged 4×3)
# ---------------------------------------------------------------------------
FIGURE_MERGED_SIZE = (9.0, 12.0)


def draw_topology(ax: plt.Axes, system: UniversalGraphSystem, xlim, ylim) -> None:
    n = 70
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.asarray(np.stack([xx, yy], axis=-1), dtype=np.float32).reshape(-1, 2)
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


def _plot_kde(ax: plt.Axes, values: np.ndarray, *, color: str, label: str, alpha: float = 0.3) -> None:
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return
    kde = gaussian_kde(x)
    grid = np.linspace(x.min(), x.max(), 200)
    density = kde(grid)
    ax.plot(grid, density, color=color, lw=1.5, label=label)
    ax.fill_between(grid, density, color=color, alpha=alpha)


def plot_original_panels(
    axes: np.ndarray,
    system: UniversalGraphSystem,
    z: np.ndarray,
    s: np.ndarray,
    u: np.ndarray,
    s_poisson: np.ndarray,
    u_poisson: np.ndarray,
    sweep_sparsity: Dict[str, Tuple[float, float]],
) -> None:
    xlim = (-2.0, 4.0)
    ylim = (-2.5, 2.5)
    if not any(isinstance(n, CycleNode) for n in system.nodes):
        pad = 0.8
        xlim = (z[:, 0].min() - pad, z[:, 0].max() + pad)
        ylim = (z[:, 1].min() - pad, z[:, 1].max() + pad)
    cell_stats = compute_cell_stats(s, u)
    gene_s = compute_gene_stats(s)
    cell_p = compute_cell_stats(s_poisson, u_poisson)

    ax = axes[0, 0]
    draw_topology(ax, system, xlim, ylim)
    sc = ax.scatter(
        z[:, 0], z[:, 1], c=cell_stats["total_spliced"], s=5, cmap="plasma",
        alpha=0.55, linewidths=0, rasterized=True, zorder=3,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title("A  Cells colored by spliced library size")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Total spliced")
    cbar.ax.tick_params(labelsize=7)

    ax = axes[0, 1]
    ts, tu = cell_p["total_spliced"], cell_p["total_unspliced"]
    bins = np.logspace(np.log10(max(ts.min(), tu.min(), 1)), np.log10(max(ts.max(), tu.max())), 40)
    ax.hist(ts, bins=bins, alpha=0.6, label="Spliced", color="C0", density=True)
    ax.hist(tu, bins=bins, alpha=0.6, label="Unspliced", color="C1", density=True)
    ax.set_xscale("log")
    ax.set_title("B  Library sizes (Poisson, no dropout)")
    ax.set_xlabel("Total counts per cell")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    mean, var = gene_s["mean"], gene_s["var"]
    mask = mean > 0.5
    ax.scatter(mean[mask], var[mask], s=3, alpha=0.35, c="C0", linewidths=0, rasterized=True)
    mx = np.logspace(np.log10(mean[mask].min()), np.log10(mean[mask].max()), 50)
    ax.plot(mx, mx, color="0.35", ls="--", lw=1.0, label="Poisson ($\\mathrm{Var=Mean}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("C  Gene mean–variance (spliced)")
    ax.set_xlabel("Gene mean")
    ax.set_ylabel("Gene variance")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.scatter(gene_s["mean"][mask], gene_s["dropout"][mask], s=3, alpha=0.35, c="C2", linewidths=0, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Gene mean (spliced)")
    ax.set_ylabel("Zero fraction")
    ax.set_title("D  Expression-dependent dropout")
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1, 1]
    ts, tu = cell_p["total_spliced"], cell_p["total_unspliced"]
    ax.scatter(ts, tu, s=2.5, alpha=0.25, c="0.35", linewidths=0, rasterized=True)
    lim = max(ts.max(), tu.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="C1", ls="--", lw=0.8, alpha=0.6, label="$U=S$")
    ax.set_xlabel("Total spliced counts")
    ax.set_ylabel("Total unspliced counts")
    ax.set_title("E  Spliced vs unspliced (Poisson)")
    ax.legend(fontsize=7)
    med_ratio = float(np.median(cell_p["u_s_ratio"]))
    ax.text(0.05, 0.95, f"median $U/S$={med_ratio:.2f}", transform=ax.transAxes, va="top", fontsize=7)

    ax = axes[1, 2]
    labels = list(sweep_sparsity.keys())
    x = np.arange(len(labels))
    w = 0.35
    sp_s = [sweep_sparsity[k][0] for k in labels]
    sp_u = [sweep_sparsity[k][1] for k in labels]
    ax.bar(x - w / 2, sp_s, width=w, label="Spliced", color="C0", edgecolor="none")
    ax.bar(x + w / 2, sp_u, width=w, label="Unspliced", color="C1", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Sparsity (fraction zeros)")
    ax.set_ylim(0, min(1.05, max(sp_s + sp_u) * 1.2 + 0.05))
    ax.set_title("F  Sparsity under noise settings")
    ax.legend(fontsize=7)


def plot_validation_panels(
    axes: np.ndarray,
    stats_primary: Dict[str, np.ndarray],
    stats_reference: Dict[str, np.ndarray],
    *,
    label_primary: Optional[str] = None,
    label_reference: Optional[str] = None,
    seed: int = SEED,
) -> None:
    label_primary = label_primary or VALIDATION_LABEL_PRIMARY
    label_reference = label_reference or VALIDATION_LABEL_REFERENCE
    rng = np.random.default_rng(seed)
    c_ref, c_main = "#4A90E2", "#D0021B"

    ax = axes[0, 0]
    _plot_kde(ax, np.log1p(stats_reference["lib_size"]), color=c_ref, label=label_reference)
    _plot_kde(ax, np.log1p(stats_primary["lib_size"]), color=c_main, label=label_primary)
    ax.set_title("G  Library Size (Cell Depth)")
    ax.set_xlabel("Log1p(Total Counts per Cell)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    _plot_kde(ax, stats_reference["n_genes"], color=c_ref, label=label_reference)
    _plot_kde(ax, stats_primary["n_genes"], color=c_main, label=label_primary)
    ax.set_title("H  Detected Genes (Cell Complexity)")
    ax.set_xlabel("Number of Detected Genes")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 2]
    mask_r = stats_reference["gene_mean"] > 0.01
    mask_m = stats_primary["gene_mean"] > 0.01
    _plot_kde(ax, np.log1p(stats_reference["gene_mean"][mask_r]), color=c_ref, label=label_reference)
    _plot_kde(ax, np.log1p(stats_primary["gene_mean"][mask_m]), color=c_main, label=label_primary)
    ax.set_title("I  Mean Expression Distribution")
    ax.set_xlabel("Log1p(Gene Mean Expression)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    idx_r = rng.choice(len(stats_reference["gene_mean"]), min(2000, len(stats_reference["gene_mean"])), replace=False)
    idx_m = rng.choice(len(stats_primary["gene_mean"]), min(2000, len(stats_primary["gene_mean"])), replace=False)
    ax.scatter(
        np.log1p(stats_reference["gene_mean"][idx_r]), np.log1p(stats_reference["gene_var"][idx_r]),
        s=6, c=c_ref, alpha=0.3, label=label_reference, linewidths=0, rasterized=True,
    )
    ax.scatter(
        np.log1p(stats_primary["gene_mean"][idx_m]), np.log1p(stats_primary["gene_var"][idx_m]),
        s=6, c=c_main, alpha=0.3, label=label_primary, linewidths=0, rasterized=True,
    )
    max_val = max(np.log1p(stats_reference["gene_mean"]).max(), np.log1p(stats_primary["gene_mean"]).max())
    line = np.linspace(0, max_val, 100)
    ax.plot(line, line, "k--", lw=1.0, alpha=0.5, label="Poisson")
    ax.set_title("J  Mean–Variance (Overdispersion)")
    ax.set_xlabel("Log1p(Mean)")
    ax.set_ylabel("Log1p(Variance)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    idx_r_drop = rng.choice(len(stats_reference["drop_mean"]), min(2000, len(stats_reference["drop_mean"])), replace=False)
    idx_m_drop = rng.choice(len(stats_primary["drop_mean"]), min(2000, len(stats_primary["drop_mean"])), replace=False)
    ax.scatter(
        np.log1p(stats_reference["drop_mean"][idx_r_drop]), stats_reference["drop_rate"][idx_r_drop],
        s=6, c=c_ref, alpha=0.3, label=label_reference, linewidths=0, rasterized=True,
    )
    ax.scatter(
        np.log1p(stats_primary["drop_mean"][idx_m_drop]), stats_primary["drop_rate"][idx_m_drop],
        s=6, c=c_main, alpha=0.3, label=label_primary, linewidths=0, rasterized=True,
    )
    ax.set_title("K  Dropout Rate vs Mean")
    ax.set_xlabel("Log1p(Mean)")
    ax.set_ylabel("Fraction of Zeros")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 2]
    _plot_kde(ax, stats_reference["u_frac"], color=c_ref, label=label_reference)
    _plot_kde(ax, stats_primary["u_frac"], color=c_main, label=label_primary)
    ax.set_title("L  Unspliced Fraction (Global Kinetics)")
    ax.set_xlabel("Unspliced Reads / Total Reads")
    ax.set_xlim(0, 0.5)
    ax.legend(frameon=False, fontsize=7)


def plot_figure6_merged(
    system: UniversalGraphSystem,
    z: np.ndarray,
    s: np.ndarray,
    u: np.ndarray,
    s_poisson: np.ndarray,
    u_poisson: np.ndarray,
    sweep_sparsity: Dict[str, Tuple[float, float]],
    stats_primary: Dict[str, np.ndarray],
    stats_reference: Dict[str, np.ndarray],
) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(4, 3, figsize=FIGURE_MERGED_SIZE)
    plot_original_panels(axes[:2], system, z, s, u, s_poisson, u_poisson, sweep_sparsity)
    plot_validation_panels(axes[2:], stats_primary, stats_reference)
    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    return fig


def _load_sweep_sparsity() -> Dict[str, Tuple[float, float]]:
    summary_path = DATA_DIR / "count_statistics.csv"
    if not summary_path.exists():
        return {}
    sweep: Dict[str, Tuple[float, float]] = {}
    with summary_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["setting"] == "main_nb_dropout":
                continue
            sweep[row["setting"]] = (float(row["sparsity_spliced"]), float(row["sparsity_unspliced"]))
    return sweep


def _load_poisson_from_cache(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    if "spliced_poisson" in data:
        return (
            to_numpy(data["spliced_poisson"]).astype(np.float64),
            to_numpy(data["unspliced_poisson"]).astype(np.float64),
        )
    print("  Poisson baseline not cached; regenerating from latent positions...")
    z = to_numpy(data["z"])
    system = build_cycle_bifurcation_system()
    poisson_cfg = next(c for c in SWEEP_CONFIGS if c.name == "poisson")
    synth = setup_synthesizer(system, poisson_cfg)
    res = generate_counts(synth, np.asarray(z, dtype=np.float32), poisson_cfg)
    s_poisson = to_numpy(res["spliced"]).astype(np.float64)
    u_poisson = to_numpy(res["unspliced"]).astype(np.float64)
    return apply_observation_postprocess(s_poisson, u_poisson, SEED + 42)


def main_figure4() -> None:
    from experiments.figure4_series_pipeline import generate_poisson_baseline, run_figure4_pipeline

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Figure4Series.ipynb pipeline...")
    result = run_figure4_pipeline()
    S, U, z = result.spliced, result.unspliced, result.z
    n_genes = S.shape[1]

    print("  Generating Poisson baseline on final latent positions...")
    S_poisson, U_poisson = generate_poisson_baseline(result.system, result.config_dict, z, SEED + 7)

    cell_stats = compute_cell_stats(S, U)
    gene_s = compute_gene_stats(S)
    gene_u = compute_gene_stats(U)
    save_cell_stats_csv(DATA_DIR / "cell_statistics.csv", z, cell_stats)
    save_gene_stats_csv(DATA_DIR / "gene_statistics.csv", gene_s, gene_u)

    summary_rows = [
        {
            "setting": "synthetic_figure4",
            "n_cells": S.shape[0],
            "n_genes": n_genes,
            "sparsity_spliced": sparsity(S),
            "sparsity_unspliced": sparsity(U),
            "median_total_spliced": float(np.median(cell_stats["total_spliced"])),
            "median_total_unspliced": float(np.median(cell_stats["total_unspliced"])),
            "median_u_s_ratio": float(np.median(cell_stats["u_s_ratio"])),
        },
        {
            "setting": "real_pancreas",
            "n_cells": result.real_spliced.shape[0],
            "n_genes": result.real_spliced.shape[1],
            "sparsity_spliced": sparsity(result.real_spliced),
            "sparsity_unspliced": sparsity(result.real_unspliced),
            "median_total_spliced": float(np.median(result.real_spliced.sum(axis=1))),
            "median_total_unspliced": float(np.median(result.real_unspliced.sum(axis=1))),
            "median_u_s_ratio": float(
                np.median(result.real_unspliced.sum(axis=1) / (result.real_spliced.sum(axis=1) + 1e-8))
            ),
        },
        {
            "setting": "poisson_baseline",
            "n_cells": S_poisson.shape[0],
            "n_genes": n_genes,
            "sparsity_spliced": sparsity(S_poisson),
            "sparsity_unspliced": sparsity(U_poisson),
            "median_total_spliced": float(np.median(S_poisson.sum(axis=1))),
            "median_total_unspliced": float(np.median(U_poisson.sum(axis=1))),
            "median_u_s_ratio": float(np.median(U_poisson.sum(axis=1) / (S_poisson.sum(axis=1) + 1e-8))),
        },
    ]
    save_count_summary_csv(DATA_DIR / "count_statistics.csv", summary_rows)

    sweep_sparsity = {
        "real": (sparsity(result.real_spliced), sparsity(result.real_unspliced)),
        "synthetic": (sparsity(S), sparsity(U)),
        "poisson": (sparsity(S_poisson), sparsity(U_poisson)),
    }

    pickle.dump(
        {
            "z": np.asarray(z, dtype=np.float32),
            "spliced": np.asarray(S),
            "unspliced": np.asarray(U),
            "spliced_poisson": np.asarray(S_poisson),
            "unspliced_poisson": np.asarray(U_poisson),
            "real_spliced": np.asarray(result.real_spliced),
            "real_unspliced": np.asarray(result.real_unspliced),
            "figure4": True,
        },
        open(str(DATA_DIR / "count_matrices.pt"), "wb"),
    )

    print("Plotting merged figure (A–F + G–L real vs synthetic)...")
    stats_sim = calc_6_metrics(S, U)
    stats_real = calc_6_metrics(result.real_spliced, result.real_unspliced)
    fig = plot_figure6_merged(
        result.system, z, S, U, S_poisson, U_poisson, sweep_sparsity, stats_sim, stats_real
    )
    save_figure(fig, FIG_DIR / "figure6_statistical_realism.png")
    print(f"Saved {FIG_DIR / 'figure6_statistical_realism.png'}")


def plot_only_figure4() -> None:
    count_path = DATA_DIR / "count_matrices.pt"
    if not count_path.exists():
        raise FileNotFoundError("Missing cached count data. Run full pipeline first.")

    print("Loading cached Figure4 results...")
    data = pickle.load(open(str(count_path), "rb"))
    z = to_numpy(data["z"])
    S = to_numpy(data["spliced"]).astype(np.float64)
    U = to_numpy(data["unspliced"]).astype(np.float64)
    S_poisson, U_poisson = _load_poisson_from_cache(data)
    sweep_sparsity = _load_sweep_sparsity() or {
        "real": (sparsity(to_numpy(data["real_spliced"])), sparsity(to_numpy(data["real_unspliced"]))),
        "synthetic": (sparsity(S), sparsity(U)),
        "poisson": (sparsity(S_poisson), sparsity(U_poisson)),
    }

    from experiments.figure4_series_pipeline import build_system_from_paga_topology, load_pancreas_reference, prepare_adata_for_paga

    _, _, adata = load_pancreas_reference()
    system = build_system_from_paga_topology(prepare_adata_for_paga(adata))

    stats_sim = calc_6_metrics(S, U)
    stats_real = calc_6_metrics(to_numpy(data["real_spliced"]), to_numpy(data["real_unspliced"]))
    fig = plot_figure6_merged(
        system, z, S, U, S_poisson, U_poisson, sweep_sparsity, stats_sim, stats_real
    )
    save_figure(fig, FIG_DIR / "figure6_statistical_realism.png")
    print(f"Saved {FIG_DIR / 'figure6_statistical_realism.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Building topology & sampling cells...")
    system = build_cycle_bifurcation_system()
    z = sample_final_cells(system, N_CELLS)
    z_np = to_numpy(z)

    print(f"Generating main count matrix ({MAIN_CONFIG.name}, {N_GENES} genes)...")
    synth_main = setup_synthesizer(system, MAIN_CONFIG)
    result_main = generate_counts(synth_main, z, MAIN_CONFIG)
    S = to_numpy(result_main["spliced"]).astype(np.float64)
    U = to_numpy(result_main["unspliced"]).astype(np.float64)
    S, U = apply_observation_postprocess(S, U, SEED)

    cell_stats = compute_cell_stats(S, U)
    gene_s = compute_gene_stats(S)
    gene_u = compute_gene_stats(U)

    save_cell_stats_csv(DATA_DIR / "cell_statistics.csv", z_np, cell_stats)
    save_gene_stats_csv(DATA_DIR / "gene_statistics.csv", gene_s, gene_u)

    summary_rows = [
        {
            "setting": "main_nb_dropout",
            "n_cells": N_CELLS,
            "n_genes": N_GENES,
            "sparsity_spliced": sparsity(S),
            "sparsity_unspliced": sparsity(U),
            "median_total_spliced": float(np.median(cell_stats["total_spliced"])),
            "median_total_unspliced": float(np.median(cell_stats["total_unspliced"])),
            "median_u_s_ratio": float(np.median(cell_stats["u_s_ratio"])),
        }
    ]

    print("Noise-setting sweep...")
    sweep_sparsity: Dict[str, Tuple[float, float]] = {}
    S_poisson = U_poisson = None
    for cfg in SWEEP_CONFIGS:
        synth = setup_synthesizer(system, cfg)
        res = generate_counts(synth, z, cfg)
        s_mat = to_numpy(res["spliced"]).astype(np.float64)
        u_mat = to_numpy(res["unspliced"]).astype(np.float64)
        s_mat, u_mat = apply_observation_postprocess(s_mat, u_mat, SEED + hash(cfg.name) % 10000)
        if cfg.name == "poisson":
            S_poisson, U_poisson = s_mat, u_mat
        sp_s, sp_u = sparsity(s_mat), sparsity(u_mat)
        sweep_sparsity[cfg.name] = (sp_s, sp_u)
        summary_rows.append(
            {
                "setting": cfg.name,
                "n_cells": N_CELLS,
                "n_genes": N_GENES,
                "sparsity_spliced": sp_s,
                "sparsity_unspliced": sp_u,
                "median_total_spliced": float(np.median(s_mat.sum(axis=1))),
                "median_total_unspliced": float(np.median(u_mat.sum(axis=1))),
                "median_u_s_ratio": float(
                    np.median(u_mat.sum(axis=1) / (s_mat.sum(axis=1) + 1e-8))
                ),
            }
        )
        print(f"  {cfg.name}: sparsity S={sp_s:.3f}, U={sp_u:.3f}")

    assert S_poisson is not None and U_poisson is not None

    save_count_summary_csv(DATA_DIR / "count_statistics.csv", summary_rows)

    pickle.dump(
        {
            "z": z,
            "spliced": np.asarray(S),
            "unspliced": np.asarray(U),
            "spliced_poisson": np.asarray(S_poisson),
            "unspliced_poisson": np.asarray(U_poisson),
        },
        open(str(DATA_DIR / "count_matrices.pt"), "wb"),
    )

    print("Plotting merged figure (A–F topology/observation + G–L validation)...")
    stats_main = calc_6_metrics(S, U)
    stats_poisson = calc_6_metrics(S_poisson, U_poisson)
    fig = plot_figure6_merged(
        system, z_np, S, U, S_poisson, U_poisson, sweep_sparsity, stats_main, stats_poisson
    )
    save_figure(fig, FIG_DIR / "figure6_statistical_realism.png")

    print(f"\nMain (NB+dropout): sparsity S={summary_rows[0]['sparsity_spliced']:.3f}")
    poisson_row = next(r for r in summary_rows if r["setting"] == "poisson")
    print(f"Poisson: median U/S={poisson_row['median_u_s_ratio']:.3f}")
    print(f"Saved {FIG_DIR / 'figure6_statistical_realism.png'}")


def plot_only() -> None:
    count_path = DATA_DIR / "count_matrices.pt"
    if not count_path.exists():
        raise FileNotFoundError("Missing cached count data. Run full pipeline first.")

    print("Loading cached results for merged figure...")
    data = pickle.load(open(str(count_path), "rb"))
    z = to_numpy(data["z"])
    S = to_numpy(data["spliced"]).astype(np.float64)
    U = to_numpy(data["unspliced"]).astype(np.float64)
    S_poisson, U_poisson = _load_poisson_from_cache(data)
    sweep_sparsity = _load_sweep_sparsity()
    system = build_cycle_bifurcation_system()

    stats_main = calc_6_metrics(S, U)
    stats_poisson = calc_6_metrics(S_poisson, U_poisson)
    fig = plot_figure6_merged(
        system, z, S, U, S_poisson, U_poisson, sweep_sparsity, stats_main, stats_poisson
    )
    save_figure(fig, FIG_DIR / "figure6_statistical_realism.png")
    print(f"Saved {FIG_DIR / 'figure6_statistical_realism.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 6: statistical realism")
    parser.add_argument("--plot-only", action="store_true", help="Replot from cached count data")
    parser.add_argument(
        "--preset",
        choices=["default", "figure4"],
        default="default",
        help="figure4: full Figure4Series.ipynb pipeline (PAGA + notebook params)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output directory")
    args = parser.parse_args()
    if args.preset == "figure4":
        from experiments import results_figure4_statistical_realism as f4

        if args.out_dir is not None:
            f4.OUT_DIR = args.out_dir
            f4.DATA_DIR = f4.OUT_DIR / "data"
            f4.FIG_DIR = f4.OUT_DIR / "figures"
        if args.plot_only:
            f4.plot_only()
        else:
            f4.run()
        raise SystemExit(0)

    configure_run(args.preset, args.out_dir)
    if args.plot_only:
        plot_only_figure4() if USE_FIGURE4_PIPELINE else plot_only()
    else:
        main_figure4() if USE_FIGURE4_PIPELINE else main()
