"""Metrics + CSV helpers for Fig4series pancreas stats (from figure_4_fast)."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Dict, List

import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d

from experiments.figures.fig4series_pancreas_stats.lib import core
from experiments.figure4_series_pipeline import Figure4Result, SEED as PIPE_SEED


def calc_6_metrics(s: np.ndarray, u: np.ndarray) -> Dict[str, np.ndarray]:
    x_total = np.asarray(s) + np.asarray(u)
    n_cells = x_total.shape[0]
    lib_size = x_total.sum(axis=1)
    n_genes = np.count_nonzero(x_total, axis=1)
    gene_mean = x_total.mean(axis=0)
    gene_var = x_total.var(axis=0)
    mask = gene_mean > 0
    mean_for_drop = gene_mean[mask]
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


def plot_kde(
    ax,
    values,
    *,
    color,
    label,
    alpha: float = 0.3,
    lw: float = 2.0,
    gridsize: int = 256,
    max_points: int = 12000,
    seed: int | None = None,
):
    """Fast KDE-like density curve using histogram smoothing (figure_4_fast)."""
    seed = core.SEED if seed is None else seed
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


def noise_sparsity_cache_path(seed: int) -> Path:
    return core.DATA_DIR / f"noise_stage_sparsities_seed{seed}.pt"


def load_noise_stage_sparsities(seed: int) -> Dict[str, Dict[str, float]] | None:
    path = noise_sparsity_cache_path(seed)
    if not path.exists():
        return None
    return pickle.load(open(str(path), "rb"))


def save_noise_stage_sparsities(data: Dict[str, Dict[str, float]], seed: int) -> None:
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, open(str(noise_sparsity_cache_path(seed)), "wb"))


def generate_noise_stage_sparsities(
    real_s: np.ndarray,
    real_u: np.ndarray,
    sim_s: np.ndarray,
    sim_u: np.ndarray,
    seed: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Illustrative sparsity stages (cycle-bifurcation demo; from figure_4_fast)."""
    from data_generation_module import (
        CycleNode,
        GeneSynthesizer,
        GraphEdge,
        GraphNode,
        UniversalGraphSystem,
    )
    from experiments.results_junction_continuity import init_cycle_rim_cells
    from experiments.results_topology_primitives import simulate_langevin

    seed = core.SEED if seed is None else seed
    demo = core.NOISE_STAGE_DEMO or {}
    rng = np.random.default_rng(seed)

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
    system_cb = UniversalGraphSystem(
        [cycle, fate1, fate2],
        [GraphEdge(cycle, fate1), GraphEdge(cycle, fate2)],
        fusion_temp=0.01,
    )

    n_cb = int(demo.get("n_cells", 2000))
    n_steps = int(demo.get("n_steps", 2000))
    z0 = init_cycle_rim_cells(n_cb, radius=1.0, noise=0.06, seed=seed)
    _, z_cells = simulate_langevin(
        system_cb, z0, n_steps=n_steps, dt=0.01, diffusion=0.006, save_every=10, seed=seed
    )
    z_t = np.asarray(z_cells, dtype=np.float32)

    n_g = int(demo.get("n_genes", 500))
    ds_s = float(demo.get("downsample_spliced", 0.20))
    ds_u = float(demo.get("downsample_unspliced", 0.15))
    noise_cfgs = [
        dict(name="clean", sampling="poisson", dropout_prob=0.0, dropout_slope=0.5, simulate_counts=False),
        dict(name="poisson", sampling="poisson", dropout_prob=0.0, dropout_slope=0.5, simulate_counts=True),
        dict(name="nb_dropout", sampling="nb", dropout_prob=0.60, dropout_slope=0.25, simulate_counts=True),
    ]

    result_sp: Dict[str, Dict[str, float]] = {}
    for cfg in noise_cfgs:
        synth = GeneSynthesizer(
            system=system_cb,
            n_genes_total=n_g,
            edge_strength=5.0,
            cycle_strength=3.0,
            background_noise=0.02,
            lib_size_mean=float(demo.get("lib_size_mean", 35000.0)),
            lib_size_sd=float(demo.get("lib_size_sd", 0.35)),
            sampling=cfg["sampling"],
            dropout_prob=cfg["dropout_prob"],
            dropout_slope=cfg["dropout_slope"],
            splicing_rate=1.0,
            degradation_rate=0.5,
            seed=seed,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            kinetics = synth.generate_kinetics(
                z_t, dt=0.01, simulate_counts=cfg["simulate_counts"], scale=1.5
            )
        xs = np.asarray(kinetics["spliced"])
        xu = np.asarray(kinetics["unspliced"])
        if cfg["name"] == "nb_dropout":
            xs = rng.binomial(xs.astype(int), ds_s).astype(float)
            xu = rng.binomial(xu.astype(int), ds_u).astype(float)
        result_sp[cfg["name"]] = {
            "spliced": float((xs == 0).mean()),
            "unspliced": float((xu == 0).mean()),
        }

    result_sp["real"] = {
        "spliced": float((real_s == 0).mean()),
        "unspliced": float((real_u == 0).mean()),
    }
    # silence unused sim args used by callers for figure4 injection
    _ = (sim_s, sim_u, PIPE_SEED)
    return result_sp


def get_noise_stage_sparsities(result: Figure4Result, seed: int | None = None) -> Dict[str, Dict[str, float]]:
    seed = core.SEED if seed is None else seed
    noise_sp = load_noise_stage_sparsities(seed)
    if noise_sp is None:
        print("  Computing noise-stage sparsities...")
        noise_sp = generate_noise_stage_sparsities(
            result.real_spliced, result.real_unspliced, result.spliced, result.unspliced, seed=seed
        )
        save_noise_stage_sparsities(noise_sp, seed)
    else:
        print("  Loaded cached noise-stage sparsities.")
    noise_sp = dict(noise_sp)
    noise_sp["figure4"] = {
        "spliced": float((result.spliced == 0).mean()),
        "unspliced": float((result.unspliced == 0).mean()),
    }
    return noise_sp


def save_csvs(result: Figure4Result) -> List[Dict]:
    s, u, z = result.spliced, result.unspliced, result.z
    xr_s, xr_u = result.real_spliced, result.real_unspliced
    cell_stats = compute_cell_stats(s, u)
    gene_s, gene_u = compute_gene_stats(s), compute_gene_stats(u)
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)

    cell_arr = np.column_stack(
        [
            np.arange(z.shape[0]),
            z[:, 0],
            z[:, 1],
            cell_stats["total_spliced"],
            cell_stats["total_unspliced"],
            cell_stats["u_s_ratio"],
        ]
    )
    np.savetxt(
        core.DATA_DIR / "cell_statistics.csv",
        cell_arr,
        delimiter=",",
        header="cell_idx,z1,z2,total_spliced,total_unspliced,u_s_ratio",
        comments="",
        fmt=["%d", "%.8g", "%.8g", "%.8g", "%.8g", "%.8g"],
    )

    gene_arr = np.column_stack(
        [
            np.arange(s.shape[1]),
            gene_s["mean"],
            gene_s["var"],
            gene_s["dropout"],
            gene_u["mean"],
            gene_u["var"],
            gene_u["dropout"],
        ]
    )
    np.savetxt(
        core.DATA_DIR / "gene_statistics.csv",
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
            "n_cells": xr_s.shape[0],
            "n_genes": xr_s.shape[1],
            "sparsity_spliced": sparsity(xr_s),
            "sparsity_unspliced": sparsity(xr_u),
            "median_total_spliced": float(np.median(xr_s_sum)),
            "median_total_unspliced": float(np.median(xr_u_sum)),
            "median_u_s_ratio": float(np.median(xr_u_sum / (xr_s_sum + 1e-8))),
            "median_u_frac": float(np.median(xr_u_sum / (xr_s_sum + xr_u_sum + 1e-8))),
        },
        {
            "dataset": "synthetic_figure4",
            "n_cells": s.shape[0],
            "n_genes": s.shape[1],
            "sparsity_spliced": sparsity(s),
            "sparsity_unspliced": sparsity(u),
            "median_total_spliced": float(np.median(total_s)),
            "median_total_unspliced": float(np.median(total_u)),
            "median_u_s_ratio": float(np.median(cell_stats["u_s_ratio"])),
            "median_u_frac": float(np.median(total_u / (total_s + total_u + 1e-8))),
        },
    ]
    fields = list(rows[0].keys())
    with (core.DATA_DIR / "count_statistics.csv").open("w", newline="") as f:
        f.write(",".join(fields) + "\n")
        for row in rows:
            f.write(",".join(str(row[k]) for k in fields) + "\n")
    return rows
