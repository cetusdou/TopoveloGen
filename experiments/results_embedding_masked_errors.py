"""
Deprecated entrypoint. Use the panelized package:
  experiments/figures/fig07_embedding_masked_errors/
See experiments/figures/README.md for run instructions.

Figure 7: Low-dimensional velocity visualizations can mask high-dimensional kinetic errors.

Generates cycle-bifurcation (and other) synthetic count data with ground-truth gene
velocity, runs scVelo, and compares low-D embedding agreement vs high-D accuracy.

Usage (from repo root):
    NUMBA_CACHE_DIR=/tmp/numba_cache python experiments/results_embedding_masked_errors.py
    python experiments/results_embedding_masked_errors.py --plot-only
"""


import argparse
import csv
import io
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figure_style import FIGURE_2X3_SIZE, apply_publication_style, save_figure  # noqa: E402

# Numba cache (required for scanpy/scvelo in some environments)
os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

from data_generation_module import CycleNode, GeneSynthesizer, UniversalGraphSystem  # noqa: E402
from experiments.results_junction_continuity import init_cycle_rim_cells  # noqa: E402
from experiments.results_statistical_realism import build_cycle_bifurcation_system  # noqa: E402
from experiments.results_topology_primitives import (  # noqa: E402
    build_bifurcation_system,
    build_cycle_linear_system,
    build_cycle_system,
    build_linear_system,
    initialize_cells,
    simulate_langevin,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 1
DT = 0.01
ALPHA_SCALE = 1.5

# Main figure (cycle-bifurcation)
MAIN_N_CELLS = 250
MAIN_N_STEPS = 1800
MAIN_SAVE_EVERY = 10
MAIN_N_GENES = 500
MAIN_SUBSAMPLE = 4000

# Topology sweep (panel F)
SWEEP_N_CELLS = 180
SWEEP_N_STEPS = 1200
SWEEP_SAVE_EVERY = 10
SWEEP_N_GENES = 350
SWEEP_SUBSAMPLE = 2500

TAU_LOW = 0.7
TAU_HIGH = 0.2

OUT_DIR = REPO_ROOT / "results" / "embedding_masked_errors"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

TOPOLOGY_BUILDERS: Dict[str, Callable[[], UniversalGraphSystem]] = {
    "linear": build_linear_system,
    "bifurcation": build_bifurcation_system,
    "cycle": build_cycle_system,
    "cycle_linear": build_cycle_linear_system,
    "cycle_bifurcation": build_cycle_bifurcation_system,
}


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def setup_synthesizer(system: UniversalGraphSystem, n_genes: int) -> GeneSynthesizer:
    synth = GeneSynthesizer(
        system=system,
        n_genes_total=n_genes,
        edge_strength=5.0,
        cycle_strength=3.0,
        background_noise=0.015,
        lib_size_mean=10000.0,
        lib_size_sd=0.35,
        sampling="nb",
        nb_dispersion=1.5,
        dropout_prob=0.15,
        dropout_slope=0.5,
        splicing_rate=1.0,
        degradation_rate=0.5,
        seed=SEED,
    )
    n_branch = max(1, n_genes // 25)
    n_imp = max(1, n_genes // 50)
    n_cycle = max(8, n_genes // 12)
    n_edges = len(system.edges)
    if n_edges >= 1:
        synth.add_trend_gene(edge_idx=0, n_genes=n_branch, pattern="sigmoid", include_cycle_rim=True)
        if n_edges >= 2:
            synth.add_trend_gene(edge_idx=1, n_genes=n_branch, pattern="sigmoid", include_cycle_rim=True)
            synth.add_trend_gene(edge_idx=0, n_genes=n_imp, pattern="impulse", include_cycle_rim=True)
            synth.add_trend_gene(edge_idx=1, n_genes=n_imp, pattern="impulse", include_cycle_rim=True)
        else:
            synth.add_trend_gene(edge_idx=0, n_genes=n_imp, pattern="impulse", include_cycle_rim=True)
    if any(isinstance(n, CycleNode) for n in system.nodes):
        if n_edges == 0:
            n_cycle = max(n_cycle, n_genes // 3)
        phases = np.linspace(0, 2 * np.pi, n_cycle, endpoint=False)
        synth.add_cycle_gene(node_id="Cycle", n_genes=n_cycle, phase=phases.tolist(), angular_sigma=0.5)
    synth.add_noise_gene(n_genes=n_genes - synth.current_gene_idx, alpha_range=(0.01, 0.06))
    synth.gene_names = [f"g{i}" for i in range(n_genes)]
    return synth


def simulate_trajectory(
    system: UniversalGraphSystem,
    topology: str,
    n_cells: int,
    n_steps: int,
    save_every: int,
) -> np.ndarray:
    z0 = initialize_cells(topology if topology != "cycle_bifurcation" else "cycle", n_cells, system, SEED)
    if topology == "cycle_bifurcation":
        z0 = init_cycle_rim_cells(n_cells, radius=1.0, noise=0.05, seed=SEED)
    traj, _ = simulate_langevin(
        system,
        z0,
        n_steps=n_steps,
        dt=DT,
        diffusion=0.006,
        save_every=save_every,
        seed=SEED,
    )
    return traj


def finite_time_displacement_from_traj(traj: np.ndarray, dt: float) -> np.ndarray:
    """Per-observation finite-time latent displacement [T*N, D] from differences."""
    t_steps, n_cells, d = traj.shape
    v = np.zeros(t_steps, n_cells, d)
    v[:-1] = (traj[1:] - traj[:-1]) / dt
    v[-1] = v[-2]
    return v.reshape(-1, d)


def latent_velocity_from_traj(traj: np.ndarray, dt: float) -> np.ndarray:
    """Deprecated alias for finite_time_displacement_from_traj."""
    return finite_time_displacement_from_traj(traj, dt)


def latent_drift_from_system(system: UniversalGraphSystem, z: np.ndarray) -> np.ndarray:
    """Evaluate the numerical latent drift field b_num(z) at observed cells."""
    z_t = np.asarray(z, dtype=np.float32)
    drift = system.get_drift(z_t)
    return drift


def flatten_traj(traj: np.ndarray) -> np.ndarray:
    t_steps, n_cells, d = traj.shape
    return traj.reshape(t_steps * n_cells, d)


def subsample_observations(n_obs: int, n_keep: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_keep = min(n_keep, n_obs)
    return np.sort(rng.choice(n_obs, size=n_keep, replace=False))


def generate_kinetics_bundle(
    system: UniversalGraphSystem,
    traj: np.ndarray,
    n_genes: int,
    save_every: int,
) -> Dict[str, np.ndarray]:
    synth = setup_synthesizer(system, n_genes)
    with contextlib.redirect_stdout(io.StringIO()):
        clean = synth.generate_kinetics(
            traj, dt=DT * save_every, scale=ALPHA_SCALE, simulate_counts=False
        )
        capture = synth._draw_capture_factor(clean["spliced"])
        spliced_counts = synth._sample_counts(clean["spliced"], capture_factor=capture)
        unspliced_counts = synth._sample_counts(
            clean["unspliced"],
            capture_factor=capture,
            capture_scale=synth.unspliced_capture_ratio,
        )
    t_steps, n_cells, _ = traj.shape
    n_obs = t_steps * n_cells

    def flat(x: np.ndarray) -> np.ndarray:
        return x.reshape(n_obs, -1)

    gene_names = list(synth.gene_names)
    return {
        "spliced_counts": flat(spliced_counts),
        "unspliced_counts": flat(unspliced_counts),
        "spliced_clean": flat(clean["spliced"]),
        "unspliced_clean": flat(clean["unspliced"]),
        "velocity_gt": flat(clean["velocity"]),
        "alpha_gt": flat(clean["alpha"]),
        "gene_names": np.array(gene_names),
    }


# ---------------------------------------------------------------------------
# scVelo
# ---------------------------------------------------------------------------
def run_scvelo(
    s_counts: np.ndarray,
    u_counts: np.ndarray,
    z_latent: np.ndarray,
    v_latent_gt: np.ndarray,
    gene_names: np.ndarray,
) -> Dict[str, Any]:
    import anndata as ad
    import scanpy as sc
    import scvelo as scv

    scv.settings.verbosity = 0
    sc.settings.verbosity = 0

    adata = ad.AnnData(X=s_counts.copy())
    adata.var_names = gene_names.astype(str)
    adata.layers["spliced"] = s_counts.copy()
    adata.layers["unspliced"] = u_counts.copy()
    adata.obsm["X_latent"] = z_latent.copy()
    adata.obsm["X_umap"] = z_latent.copy()
    adata.obsm["velocity_latent_gt"] = v_latent_gt.copy()

    scv.pp.filter_and_normalize(adata, min_shared_counts=15)
    n_top = min(200, adata.n_vars)
    if adata.n_vars > n_top:
        s_tot = np.asarray(adata.layers["spliced"].sum(axis=0)).ravel()
        u_tot = np.asarray(adata.layers["unspliced"].sum(axis=0)).ravel()
        top_idx = np.argsort(s_tot + u_tot)[::-1][:n_top]
        adata = adata[:, top_idx].copy()
    scv.pp.moments(adata, n_pcs=min(30, adata.n_obs - 1, adata.n_vars - 1), n_neighbors=30)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata)
    scv.tl.velocity_embedding(adata, basis="umap")

    v_pred = adata.layers["velocity"]
    if hasattr(v_pred, "toarray"):
        v_pred = v_pred.toarray()
    v_pred = np.asarray(v_pred)

    v_umap = np.asarray(adata.obsm["velocity_umap"])

    return {
        "adata": adata,
        "velocity_pred": v_pred,
        "velocity_umap": v_umap,
        "gene_names_used": adata.var_names.astype(str),
    }


def embed_velocity_on_umap(adata, velocity: np.ndarray, vkey: str = "velocity_in") -> np.ndarray:
    import scvelo as scv

    a = adata.copy()
    a.layers[vkey] = velocity
    scv.tl.velocity_graph(a, vkey=vkey)
    scv.tl.velocity_embedding(a, basis="umap", vkey=vkey)
    return np.asarray(a.obsm[f"{vkey}_umap"])


def align_velocity_gt(
    velocity_gt_full: np.ndarray,
    gene_names_full: np.ndarray,
    gene_names_used: np.ndarray,
) -> np.ndarray:
    name_to_idx = {g: i for i, g in enumerate(gene_names_full.astype(str))}
    cols = [name_to_idx[g] for g in gene_names_used]
    return velocity_gt_full[:, cols]


def cosine_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return num / den


# ---------------------------------------------------------------------------
# Regions (cycle-bifurcation)
# ---------------------------------------------------------------------------
def assign_regions(system: UniversalGraphSystem, z: np.ndarray) -> Dict[str, np.ndarray]:
    z_t = np.asarray(z, dtype=np.float32)
    regions: Dict[str, np.ndarray] = {}

    cycle = next((n for n in system.nodes if isinstance(n, CycleNode)), None)
    if cycle is not None:
        center = np.asarray(cycle.pos[:2], dtype=np.float32)
        dist_rim = np.abs(np.linalg.norm(z_t - center, axis=1) - cycle.radius)
        regions["cycle_rim"] = dist_rim < 0.25

    if len(system.edges) >= 2 and cycle is not None:
        docks = np.stack([e.start for e in system.edges[:2]], axis=0)
        d0 = np.linalg.norm(z - docks[0], axis=1)
        d1 = np.linalg.norm(z - docks[1], axis=1)
        regions["junction"] = np.minimum(d0, d1) < 0.4
        regions["branch1"] = np.sqrt(system.edges[0].distance_sq(z_t)) < 0.35
        regions["branch2"] = np.sqrt(system.edges[1].distance_sq(z_t)) < 0.35
    elif len(system.edges) == 1:
        regions["edge"] = np.sqrt(system.edges[0].distance_sq(z_t)) < 0.35

    return regions


def mismatch_mask(cos_low: np.ndarray, cos_high: np.ndarray) -> np.ndarray:
    return (cos_low > TAU_LOW) & (cos_high < TAU_HIGH)


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------
def analyze_topology(
    topology: str,
    n_cells: int,
    n_steps: int,
    save_every: int,
    n_genes: int,
    subsample: int,
) -> Dict[str, Any]:
    system = TOPOLOGY_BUILDERS[topology]()
    traj = simulate_trajectory(system, topology, n_cells, n_steps, save_every)
    bundle = generate_kinetics_bundle(system, traj, n_genes, save_every)

    z_all = flatten_traj(traj)
    v_lat_all = latent_drift_from_system(system, z_all)
    idx = subsample_observations(len(z_all), subsample, SEED)

    s = bundle["spliced_counts"][idx]
    u = bundle["unspliced_counts"][idx]
    z = z_all[idx]
    v_lat = v_lat_all[idx]
    v_gt_full = bundle["velocity_gt"][idx]
    genes = bundle["gene_names"]

    sc = run_scvelo(s, u, z, v_lat, genes)
    v_gt = align_velocity_gt(v_gt_full, genes, sc["gene_names_used"])
    v_pred = sc["velocity_pred"]

    cos_high = cosine_rows(v_pred, v_gt)
    v_umap_gt = embed_velocity_on_umap(sc["adata"], v_gt, vkey="velocity_gt")
    cos_low = cosine_rows(sc["velocity_umap"], v_umap_gt)
    mm = mismatch_mask(cos_low, cos_high)

    regions = assign_regions(system, z)
    region_fracs = {k: float(mm[v].mean()) if v.any() else float("nan") for k, v in regions.items()}

    return {
        "topology": topology,
        "system": system,
        "z": z,
        "v_latent_gt": v_lat,
        "velocity_umap_gt": v_umap_gt,
        "cos_high": cos_high,
        "cos_low": cos_low,
        "mismatch": mm,
        "mismatch_frac": float(mm.mean()),
        "regions": regions,
        "region_mismatch_frac": region_fracs,
        "adata": sc["adata"],
        "velocity_umap": sc["velocity_umap"],
        "n_obs": len(z),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _quiver_subsample(
    ax: plt.Axes,
    xy: np.ndarray,
    v: np.ndarray,
    stride: int = 25,
    scale: float = 1.0,
    color: str = "C0",
    alpha: float = 0.85,
) -> None:
    n = len(xy)
    idx = np.arange(0, n, max(1, stride))
    v_s = v[idx]
    ax.quiver(
        xy[idx, 0],
        xy[idx, 1],
        v_s[:, 0],
        v_s[:, 1],
        angles="xy",
        scale_units="xy",
        scale=scale,
        color=color,
        alpha=alpha,
        width=0.0025,
        zorder=4,
    )


def plot_figure7(main: Dict[str, Any], sweep: List[Dict[str, Any]]) -> plt.Figure:
    apply_publication_style()
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_2X3_SIZE)
    z = main["z"]
    system = main["system"]

    # A: GT latent drift
    ax = axes[0, 0]
    ax.scatter(z[:, 0], z[:, 1], s=1.5, c="#CCCCCC", linewidths=0, rasterized=True)
    _quiver_subsample(ax, z, main["v_latent_gt"], stride=35, scale=8.0, color="C0")
    ax.set_title("A  Ground-truth latent drift")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")

    # B: inferred embedding velocity
    ax = axes[0, 1]
    ax.scatter(z[:, 0], z[:, 1], s=1.5, c="#DDDDDD", linewidths=0, rasterized=True)
    _quiver_subsample(ax, z, main["velocity_umap"], stride=35, scale=8.0, color="C1")
    ax.set_title("B  scVelo embedding velocity")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")

    # C: high-D cosine
    ax = axes[0, 2]
    sc_c = ax.scatter(
        z[:, 0],
        z[:, 1],
        c=main["cos_high"],
        s=2.5,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title("C  High-D velocity cosine")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    cbar = fig.colorbar(sc_c, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(r"$C^{G}$", labelpad=2)

    # D: low vs high scatter
    ax = axes[1, 0]
    ax.scatter(main["cos_low"], main["cos_high"], s=2.5, alpha=0.2, c="#666666", linewidths=0, rasterized=True)
    ax.axvline(TAU_LOW, color="C1", ls="--", lw=1.0)
    ax.axhline(TAU_HIGH, color="C1", ls="--", lw=1.0)
    ax.set_xlabel(r"Low-D cosine ($C^{2D}$)")
    ax.set_ylabel(r"High-D cosine ($C^{G}$)")
    ax.set_title("D  Embedding vs gene-space agreement")
    n_mm = int(main["mismatch"].sum())
    ax.text(
        0.98,
        0.02,
        f"mismatch = {n_mm}/{len(z)} ({main['mismatch_frac']:.1%})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    # E: mismatch on topology
    ax = axes[1, 1]
    mm = main["mismatch"]
    ax.scatter(z[~mm, 0], z[~mm, 1], s=1.5, c="#BBBBBB", linewidths=0, label="agreement", rasterized=True)
    ax.scatter(z[mm, 0], z[mm, 1], s=5, c="C1", linewidths=0, label="low-D ok, high-D wrong", rasterized=True)
    for edge in system.edges:
        s, e = edge.start, edge.end
        ax.plot([s[0], e[0]], [s[1], e[1]], color="#888888", ls="--", lw=0.8, alpha=0.6)
    ax.set_title("E  Mismatch cells on topology")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=1.5)

    # F: mismatch vs complexity
    ax = axes[1, 2]
    order = ["linear", "bifurcation", "cycle", "cycle_linear", "cycle_bifurcation"]
    fracs = [next(r["mismatch_frac"] for r in sweep if r["topology"] == t) for t in order]
    labels = [t.replace("_", "-") for t in order]
    ax.bar(range(len(order)), fracs, color="C0", edgecolor="none", width=0.72)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mismatch fraction")
    ax.set_ylim(0, max(fracs) * 1.25 + 0.01)
    ax.set_title("F  Mismatch vs topology complexity")

    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    return fig


def load_plot_data() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import anndata as ad

    metrics_path = DATA_DIR / "velocity_accuracy_metrics.csv"
    topo_path = DATA_DIR / "mismatch_by_topology.csv"
    adata_path = DATA_DIR / "adata_velocity_benchmark.h5ad"
    for path in (metrics_path, topo_path, adata_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing cached result: {path}. Run full pipeline first.")

    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    z = np.array([[float(r["z1"]), float(r["z2"])] for r in rows])
    cos_high = np.array([float(r["cos_high"]) for r in rows])
    cos_low = np.array([float(r["cos_low"]) for r in rows])
    mismatch = np.array([bool(int(r["mismatch"])) for r in rows])

    adata = ad.read_h5ad(adata_path)
    v_latent_gt = np.asarray(adata.obsm["velocity_latent_gt"])
    velocity_umap = np.asarray(adata.obsm["velocity_umap"])

    with topo_path.open(newline="") as f:
        sweep_rows = list(csv.DictReader(f))

    sweep = [
        {
            "topology": r["topology"],
            "mismatch_frac": float(r["mismatch_frac"]),
        }
        for r in sweep_rows
    ]

    system = build_cycle_bifurcation_system()
    main = {
        "topology": "cycle_bifurcation",
        "system": system,
        "z": z,
        "v_latent_gt": v_latent_gt,
        "cos_high": cos_high,
        "cos_low": cos_low,
        "mismatch": mismatch,
        "mismatch_frac": float(mismatch.mean()),
        "velocity_umap": velocity_umap,
        "n_obs": len(z),
    }
    return main, sweep


def save_figure7(fig: plt.Figure) -> None:
    save_figure(fig, FIG_DIR / "figure7_lowD_highD_mismatch.png")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_csv_rows(path: Path, rows: List[Dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    fields = fieldnames or sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Main analysis: cycle_bifurcation ===")
    main = analyze_topology(
        "cycle_bifurcation",
        MAIN_N_CELLS,
        MAIN_N_STEPS,
        MAIN_SAVE_EVERY,
        MAIN_N_GENES,
        MAIN_SUBSAMPLE,
    )
    print(f"  mismatch fraction: {main['mismatch_frac']:.3f}")
    print(f"  mean cos_high: {main['cos_high'].mean():.3f}, cos_low: {main['cos_low'].mean():.3f}")
    for reg, frac in main["region_mismatch_frac"].items():
        print(f"  mismatch @ {reg}: {frac:.3f}")

    # Save per-cell metrics
    cell_rows = [
        {
            "cell_idx": i,
            "z1": float(main["z"][i, 0]),
            "z2": float(main["z"][i, 1]),
            "cos_high": float(main["cos_high"][i]),
            "cos_low": float(main["cos_low"][i]),
            "mismatch": int(main["mismatch"][i]),
        }
        for i in range(main["n_obs"])
    ]
    save_csv_rows(DATA_DIR / "velocity_accuracy_metrics.csv", cell_rows)

    region_rows = [
        {"region": k, "mismatch_frac": v, "topology": "cycle_bifurcation"}
        for k, v in main["region_mismatch_frac"].items()
    ]
    save_csv_rows(DATA_DIR / "mismatch_by_region.csv", region_rows)

    main["adata"].write_h5ad(DATA_DIR / "adata_velocity_benchmark.h5ad")

    print("\n=== Topology sweep (panel F) ===")
    sweep_results: List[Dict[str, Any]] = []
    sweep_rows: List[Dict] = []
    for topo in ["linear", "bifurcation", "cycle", "cycle_linear", "cycle_bifurcation"]:
        print(f"  {topo}...")
        res = analyze_topology(
            topo,
            SWEEP_N_CELLS,
            SWEEP_N_STEPS,
            SWEEP_SAVE_EVERY,
            SWEEP_N_GENES,
            SWEEP_SUBSAMPLE,
        )
        sweep_results.append(res)
        sweep_rows.append(
            {
                "topology": topo,
                "mismatch_frac": res["mismatch_frac"],
                "mean_cos_high": float(res["cos_high"].mean()),
                "mean_cos_low": float(res["cos_low"].mean()),
                "n_cells": res["n_obs"],
            }
        )
        print(f"    mismatch={res['mismatch_frac']:.3f}")

    save_csv_rows(DATA_DIR / "mismatch_by_topology.csv", sweep_rows)

    print("\nPlotting Figure 7...")
    fig = plot_figure7(main, sweep_results)
    save_figure7(fig)

    print(f"\nSaved {FIG_DIR / 'figure7_lowD_highD_mismatch.png'}")


def plot_only() -> None:
    print("Loading cached results for Figure 7...")
    main, sweep = load_plot_data()
    print(f"  mismatch fraction: {main['mismatch_frac']:.3f}")
    fig = plot_figure7(main, sweep)
    save_figure7(fig)
    print(f"Saved {FIG_DIR / 'figure7_lowD_highD_mismatch.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 7: embedding masked velocity errors")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Replot from cached CSV/h5ad without rerunning scVelo",
    )
    args = parser.parse_args()
    if args.plot_only:
        plot_only()
    else:
        main()
