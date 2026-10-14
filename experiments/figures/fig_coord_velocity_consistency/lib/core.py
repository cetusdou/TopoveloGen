"""Coordinate-consistency diagnostics for ground-truth RNA velocity."""

from __future__ import annotations

import csv
import io
import contextlib
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

SEED = 1
DT = 0.01
ALPHA_SCALE = 1.5
DIFFUSION = 0.006
TAU_HIGH = 0.7
LIBRARY_TARGET = 10000.0

MAIN = dict(
    topology="cycle_bifurcation",
    n_cells=250,
    n_steps=1800,
    save_every=10,
    n_genes=500,
    subsample=4000,
)
SYNTH_KW = dict(
    background_noise=0.015,
    lib_size_mean=10000.0,
    lib_size_sd=0.35,
    sampling="nb",
    nb_dispersion=1.5,
    dropout_prob=0.15,
    dropout_slope=0.5,
    splicing_rate=1.0,
    degradation_rate=0.5,
    noise_alpha_range=(0.01, 0.06),
)
SCVELO_KW = dict(
    min_shared_counts=15,
    n_top_genes=200,
    n_pcs=30,
    n_neighbors=30,
    mode="stochastic",
)
METHOD_TABLE: List[Dict[str, str]] = []

OUT_DIR = REPO_ROOT / "results" / "coord_velocity_consistency"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

TOPOLOGY_BUILDERS = {
    "linear": build_linear_system,
    "bifurcation": build_bifurcation_system,
    "cycle": build_cycle_system,
    "cycle_linear": build_cycle_linear_system,
    "cycle_bifurcation": build_cycle_bifurcation_system,
}


def ensure_numba_cache(path: Optional[str] = None) -> None:
    cache = Path(path) if path else REPO_ROOT / ".numba_cache"
    if not cache.is_absolute():
        cache = REPO_ROOT / cache
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache)


def configure(
    *,
    out_dir=None,
    data_dir=None,
    fig_dir=None,
    pipeline: dict | None = None,
    shared: dict | None = None,
) -> None:
    global OUT_DIR, DATA_DIR, FIG_DIR
    global SEED, DT, DIFFUSION, ALPHA_SCALE, TAU_HIGH, LIBRARY_TARGET
    global MAIN, SYNTH_KW, SCVELO_KW, METHOD_TABLE

    if shared:
        ensure_numba_cache(shared.get("numba_cache_dir"))
        SEED = int(shared.get("seed", SEED))
    else:
        ensure_numba_cache()

    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    elif out_dir is not None:
        DATA_DIR = OUT_DIR / "data"
    if fig_dir is not None:
        FIG_DIR = Path(fig_dir)
    elif out_dir is not None:
        FIG_DIR = OUT_DIR / "figures"

    if not pipeline:
        return
    SEED = int(pipeline.get("seed", SEED))
    DT = float(pipeline.get("dt", DT))
    DIFFUSION = float(pipeline.get("diffusion", DIFFUSION))
    ALPHA_SCALE = float(pipeline.get("alpha_scale", ALPHA_SCALE))
    TAU_HIGH = float(pipeline.get("tau_high", TAU_HIGH))
    LIBRARY_TARGET = float(pipeline.get("library_target", LIBRARY_TARGET))
    if "main" in pipeline:
        MAIN = {**MAIN, **pipeline["main"]}
    if "synthesizer" in pipeline:
        sk = dict(pipeline["synthesizer"])
        if "noise_alpha_range" in sk:
            sk["noise_alpha_range"] = tuple(sk["noise_alpha_range"])
        SYNTH_KW = {**SYNTH_KW, **sk}
    if "scvelo" in pipeline:
        SCVELO_KW = {**SCVELO_KW, **pipeline["scvelo"]}
    if "method_table" in pipeline:
        METHOD_TABLE = list(pipeline["method_table"])


def setup_synthesizer(system: UniversalGraphSystem, n_genes: int,
                      signal_reference_genes: int | None = None) -> GeneSynthesizer:
    synth = GeneSynthesizer(
        system=system,
        n_genes_total=n_genes,
        edge_strength=5.0,
        cycle_strength=3.0,
        background_noise=float(SYNTH_KW["background_noise"]),
        lib_size_mean=float(SYNTH_KW["lib_size_mean"]),
        lib_size_sd=float(SYNTH_KW["lib_size_sd"]),
        sampling=str(SYNTH_KW["sampling"]),
        nb_dispersion=float(SYNTH_KW["nb_dispersion"]),
        dropout_prob=float(SYNTH_KW["dropout_prob"]),
        dropout_slope=float(SYNTH_KW["dropout_slope"]),
        splicing_rate=float(SYNTH_KW["splicing_rate"]),
        degradation_rate=float(SYNTH_KW["degradation_rate"]),
        seed=SEED,
    )
    signal_reference_genes = n_genes if signal_reference_genes is None else signal_reference_genes
    n_branch = max(1, signal_reference_genes // 25)
    n_imp = max(1, signal_reference_genes // 50)
    n_cycle = max(8, signal_reference_genes // 12)
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
            n_cycle = max(n_cycle, signal_reference_genes // 3)
        phases = np.linspace(0, 2 * np.pi, n_cycle, endpoint=False)
        synth.add_cycle_gene(node_id="Cycle", n_genes=n_cycle, phase=phases.tolist(), angular_sigma=0.5)
    synth.add_noise_gene(n_genes=n_genes - synth.current_gene_idx, alpha_range=SYNTH_KW["noise_alpha_range"])
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
        diffusion=DIFFUSION,
        save_every=save_every,
        seed=SEED,
    )
    return traj


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
) -> Dict[str, np.ndarray]:
    synth = setup_synthesizer(system, n_genes)
    with contextlib.redirect_stdout(io.StringIO()):
        sample_dt = DT * int(MAIN["save_every"])
        clean = synth.generate_kinetics(traj, dt=sample_dt, scale=ALPHA_SCALE, simulate_counts=False)
        capture = synth._draw_capture_factor(clean["spliced"])
        counts = {
            "spliced": synth._sample_counts(clean["spliced"], capture_factor=capture),
            "unspliced": synth._sample_counts(clean["unspliced"], capture_factor=capture,
                                              capture_scale=synth.unspliced_capture_ratio),
        }
    t_steps, n_cells, _ = traj.shape
    n_obs = t_steps * n_cells

    def flat(x: np.ndarray) -> np.ndarray:
        return np.asarray(x).reshape(n_obs, -1)

    return {
        "spliced_counts": flat(counts["spliced"]),
        "unspliced_counts": flat(counts["unspliced"]),
        "spliced_clean": flat(clean["spliced"]),
        "unspliced_clean": flat(clean["unspliced"]),
        "velocity_gt": flat(clean["velocity"]),
        "gene_names": np.array(list(synth.gene_names)),
    }


def run_scvelo(
    s_counts: np.ndarray,
    u_counts: np.ndarray,
    z_latent: np.ndarray,
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

    scv.pp.filter_and_normalize(adata, min_shared_counts=int(SCVELO_KW["min_shared_counts"]))
    n_top = min(int(SCVELO_KW["n_top_genes"]), adata.n_vars)
    if adata.n_vars > n_top:
        s_tot = np.asarray(adata.layers["spliced"].sum(axis=0)).ravel()
        u_tot = np.asarray(adata.layers["unspliced"].sum(axis=0)).ravel()
        top_idx = np.argsort(s_tot + u_tot)[::-1][:n_top]
        adata = adata[:, top_idx].copy()
    scv.pp.moments(
        adata,
        n_pcs=min(int(SCVELO_KW["n_pcs"]), adata.n_obs - 1, adata.n_vars - 1),
        n_neighbors=int(SCVELO_KW["n_neighbors"]),
    )
    scv.tl.velocity(adata, mode=str(SCVELO_KW["mode"]))

    v_pred = adata.layers["velocity"]
    if hasattr(v_pred, "toarray"):
        v_pred = v_pred.toarray()
    return {
        "velocity_pred": np.asarray(v_pred, dtype=np.float64),
        "gene_names_used": np.asarray(adata.var_names.astype(str)),
    }


def align_genes(
    mat_full: np.ndarray,
    gene_names_full: np.ndarray,
    gene_names_used: np.ndarray,
) -> np.ndarray:
    name_to_idx = {g: i for i, g in enumerate(gene_names_full.astype(str))}
    cols = [name_to_idx[g] for g in gene_names_used]
    return np.asarray(mat_full[:, cols], dtype=np.float64)


def cosine_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return num / den


def library_scales(s: np.ndarray, library_target: float | None = None) -> np.ndarray:
    """c_i = L / S_i with S_i = sum_g s_ig."""
    L = float(LIBRARY_TARGET if library_target is None else library_target)
    S = np.asarray(s, dtype=np.float64).sum(axis=1)
    return L / np.maximum(S, 1e-8)


def transform_velocity_library(v_abs: np.ndarray, c: np.ndarray) -> np.ndarray:
    return c[:, None] * v_abs


def transform_velocity_proportion(s: np.ndarray, v_abs: np.ndarray) -> np.ndarray:
    """v_prop = (S v_abs - s * sum(v_abs)) / S^2."""
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v_abs, dtype=np.float64)
    S = s.sum(axis=1, keepdims=True)
    sum_v = v.sum(axis=1, keepdims=True)
    return (S * v - s * sum_v) / np.maximum(S * S, 1e-12)


def transform_velocity_log(s: np.ndarray, v_abs: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """v_log ≈ rho / (1 + rho s) * v_abs  (ignore d rho/dt)."""
    s = np.asarray(s, dtype=np.float64)
    v = np.asarray(v_abs, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64)[:, None]
    return (r / (1.0 + r * s)) * v


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    r = spearmanr(x, y, nan_policy="omit")
    return float(r.correlation) if np.isfinite(r.correlation) else float("nan")


def flip_fraction(a: np.ndarray, b: np.ndarray, tau: float) -> float:
    """Fraction of cells whose pass/fail relative to tau differs between scores."""
    pa = np.asarray(a) >= tau
    pb = np.asarray(b) >= tau
    return float(np.mean(pa != pb))


def build_coordinate_metrics(
    s: np.ndarray,
    v_abs: np.ndarray,
    v_hat: np.ndarray,
    *,
    library_target: float | None = None,
    tau_high: float | None = None,
) -> Dict[str, Any]:
    tau = float(TAU_HIGH if tau_high is None else tau_high)
    c = library_scales(s, library_target)
    v_lib = transform_velocity_library(v_abs, c)
    v_prop = transform_velocity_proportion(s, v_abs)
    v_log = transform_velocity_log(s, v_abs, c)

    cos_abs = cosine_rows(v_hat, v_abs)
    cos_lib = cosine_rows(v_hat, v_lib)
    cos_prop = cosine_rows(v_hat, v_prop)
    cos_log = cosine_rows(v_hat, v_log)

    per_cell = {
        "cos_abs": cos_abs,
        "cos_lib": cos_lib,
        "cos_prop": cos_prop,
        "cos_log": cos_log,
        "c_i": c,
        "S_i": s.sum(axis=1),
    }

    def stats(x: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.nanmean(x)),
            "median": float(np.nanmedian(x)),
        }

    summary = {
        "n_cells": int(s.shape[0]),
        "n_genes": int(s.shape[1]),
        "tau_high": tau,
        "library_target": float(LIBRARY_TARGET if library_target is None else library_target),
        "abs_mean": stats(cos_abs)["mean"],
        "abs_median": stats(cos_abs)["median"],
        "lib_mean": stats(cos_lib)["mean"],
        "lib_median": stats(cos_lib)["median"],
        "prop_mean": stats(cos_prop)["mean"],
        "prop_median": stats(cos_prop)["median"],
        "log_mean": stats(cos_log)["mean"],
        "log_median": stats(cos_log)["median"],
        "spearman_abs_lib": spearman_corr(cos_abs, cos_lib),
        "spearman_abs_prop": spearman_corr(cos_abs, cos_prop),
        "spearman_abs_log": spearman_corr(cos_abs, cos_log),
        "flip_abs_prop": flip_fraction(cos_abs, cos_prop, tau),
        "flip_abs_log": flip_fraction(cos_abs, cos_log, tau),
        "max_abs_diff_abs_lib": float(np.nanmax(np.abs(cos_abs - cos_lib))),
    }
    return {
        "per_cell": per_cell,
        "summary": summary,
        "v_lib": v_lib,
        "v_prop": v_prop,
        "v_log": v_log,
        "c": c,
    }


def run_experiment() -> Dict[str, Any]:
    topology = str(MAIN["topology"])
    system = TOPOLOGY_BUILDERS[topology]()
    print(f"Simulating topology={topology} …")
    traj = simulate_trajectory(
        system,
        topology,
        int(MAIN["n_cells"]),
        int(MAIN["n_steps"]),
        int(MAIN["save_every"]),
    )
    z_flat = flatten_traj(traj)
    bundle = generate_kinetics_bundle(system, traj, int(MAIN["n_genes"]))
    idx = subsample_observations(z_flat.shape[0], int(MAIN["subsample"]), SEED)

    s_counts = bundle["spliced_counts"][idx]
    u_counts = bundle["unspliced_counts"][idx]
    s_clean = bundle["spliced_clean"][idx]
    v_gt = bundle["velocity_gt"][idx]
    z = z_flat[idx]
    gene_names = bundle["gene_names"]

    print("Running scVelo (stochastic) …")
    scv_out = run_scvelo(s_counts, u_counts, z, gene_names)
    genes_used = scv_out["gene_names_used"]
    v_hat = scv_out["velocity_pred"]

    s_aln = align_genes(s_clean, gene_names, genes_used)
    v_abs = align_genes(v_gt, gene_names, genes_used)

    print("Computing coordinate-transformed GT cosines …")
    metrics = build_coordinate_metrics(s_aln, v_abs, v_hat)

    return {
        "s": s_aln,
        "v_abs": v_abs,
        "v_hat": v_hat,
        "gene_names": genes_used,
        "z": z,
        "per_cell": metrics["per_cell"],
        "summary": metrics["summary"],
        "method_table": list(METHOD_TABLE),
        "c": metrics["c"],
    }


def save_results(result: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    per = result["per_cell"]
    n = len(per["cos_abs"])
    per_path = DATA_DIR / "coord_cosine_per_cell.csv"
    with per_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["cell_idx", "cos_abs", "cos_lib", "cos_prop", "cos_log", "c_i", "S_i"],
        )
        w.writeheader()
        for i in range(n):
            w.writerow(
                {
                    "cell_idx": i,
                    "cos_abs": float(per["cos_abs"][i]),
                    "cos_lib": float(per["cos_lib"][i]),
                    "cos_prop": float(per["cos_prop"][i]),
                    "cos_log": float(per["cos_log"][i]),
                    "c_i": float(per["c_i"][i]),
                    "S_i": float(per["S_i"][i]),
                }
            )

    summary = result["summary"]
    sum_path = DATA_DIR / "coord_summary.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    table = result.get("method_table") or METHOD_TABLE
    tab_path = DATA_DIR / "method_coordinate_table.csv"
    if table:
        fields = list(table[0].keys())
        with tab_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(table)

    bundle_path = DATA_DIR / "bundle.pkl"
    with bundle_path.open("wb") as f:
        pickle.dump(
            {
                "s": result["s"],
                "v_abs": result["v_abs"],
                "v_hat": result["v_hat"],
                "c": result["c"],
                "gene_names": result["gene_names"],
                "z": result["z"],
                "per_cell": result["per_cell"],
                "summary": result["summary"],
                "method_table": table,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {per_path}")
    print(f"Saved {sum_path}")
    print(f"Saved {tab_path}")
    print(f"Saved {bundle_path}")


def load_bundle() -> Dict[str, Any]:
    path = DATA_DIR / "bundle.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cache: {path}\n"
            "Run: python experiments/figures/fig_coord_velocity_consistency/pipeline/run_pipeline.py"
        )
    with path.open("rb") as f:
        return pickle.load(f)


def load_summary_rows() -> Dict[str, float]:
    path = DATA_DIR / "coord_summary.csv"
    if not path.exists():
        return dict(load_bundle()["summary"])
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError(f"Empty summary: {path}")
    out: Dict[str, float] = {}
    for k, v in rows[0].items():
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v  # type: ignore[assignment]
    return out


def load_method_table() -> List[Dict[str, str]]:
    path = DATA_DIR / "method_coordinate_table.csv"
    if path.exists():
        with path.open() as f:
            return list(csv.DictReader(f))
    return list(METHOD_TABLE)
