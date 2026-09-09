"""
Projection-masking diagnosis for synthetic scVelo velocity inference.

Tests whether projection-based evaluation, gene-space accuracy, and topological
organization of errors are decoupled on cycle-bifurcation synthetic data.

Key metrics (per cell):
  C^G_pred       — gene-space cosine (predicted vs GT velocity)
  C^2D_pred      — embedding cosine (predicted vs GT latent velocity)
  C^2D_GT embed  — embedding cosine (GT velocity embedded vs GT latent)
  Δ_proj         — C^2D_pred − C^G_pred
  masking cell   — C^2D_pred > τ_low AND C^G_pred < τ_high  (τ_low=0.7, τ_high=0.2)

Headline (main text):
  Projection-based velocity evaluation has a ceiling, and gene-space errors are
  not random but concentrate in dynamic modules and specific topological regions.

Usage (from repo root):
    NUMBA_CACHE_DIR=/tmp/numba_cache python experiments/results_projection_masking_diagnosis.py
    python experiments/results_projection_masking_diagnosis.py --plot-only
    python experiments/results_projection_masking_diagnosis.py --quick
"""

from __future__ import annotations

import argparse
import csv
import io
import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figure_style import apply_publication_style, save_figure  # noqa: E402

os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

from data_generation_module import CycleNode, GeneSynthesizer, UniversalGraphSystem  # noqa: E402
from experiments.results_embedding_masked_errors import (  # noqa: E402
    DT,
    SEED,
    TAU_HIGH,
    TAU_LOW,
    align_velocity_gt,
    assign_regions,
    cosine_rows,
    flatten_traj,
    latent_velocity_from_traj,
    mismatch_mask,
    simulate_trajectory,
    subsample_observations,
)
from experiments.results_statistical_realism import build_cycle_bifurcation_system  # noqa: E402
from experiments.projection_masking_metrics import (  # noqa: E402
    EVALUATION_CEILING_CAPTION,
    delta_projection_masking,
    summarize_delta_proj,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ALPHA_SCALE = 1.5
TOPOLOGY = "cycle_bifurcation"

N_CELLS = 220
N_STEPS = 1400
SAVE_EVERY = 10
SUBSAMPLE = 2500

GENE_SWEEP_FULL = [
    200, 300, 400, 500, 650, 800, 1000, 1200, 1500,
    1800, 2100, 2500, 3000, 3500, 4000, 4500,
]
GENE_SWEEP_QUICK = [300, 500, 800, 1200, 1800, 2500, 3500]

OUT_DIR = REPO_ROOT / "results" / "projection_masking_diagnosis"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

REGION_COLORS = {
    "cycle_rim": "#4477AA",
    "junction": "#EE6677",
    "branch1": "#228833",
    "branch2": "#CCBB44",
    "edge": "#AA3377",
    "other": "#CCCCCC",
}
REGION_PRIORITY = ("junction", "cycle_rim", "branch1", "branch2", "edge")


# ---------------------------------------------------------------------------
# Data generation (with gene-type labels)
# ---------------------------------------------------------------------------
def setup_synthesizer_typed(system: UniversalGraphSystem, n_genes: int) -> Tuple[GeneSynthesizer, np.ndarray]:
    """Return synthesizer and per-gene type codes: 0=dynamic, 1=noise."""
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
    gene_types = np.zeros(n_genes, dtype=np.int8)

    n_branch = max(1, n_genes // 25)
    n_imp = max(1, n_genes // 50)
    n_cycle = max(8, n_genes // 12)
    n_edges = len(system.edges)
    idx = 0

    def mark_dynamic(count: int) -> None:
        nonlocal idx
        gene_types[idx : idx + count] = 0
        idx += count

    def mark_noise(count: int) -> None:
        nonlocal idx
        gene_types[idx : idx + count] = 1
        idx += count

    if n_edges >= 1:
        synth.add_trend_gene(edge_idx=0, n_genes=n_branch, pattern="sigmoid", include_cycle_rim=True)
        mark_dynamic(n_branch)
        if n_edges >= 2:
            synth.add_trend_gene(edge_idx=1, n_genes=n_branch, pattern="sigmoid", include_cycle_rim=True)
            mark_dynamic(n_branch)
            synth.add_trend_gene(edge_idx=0, n_genes=n_imp, pattern="impulse", include_cycle_rim=True)
            mark_dynamic(n_imp)
            synth.add_trend_gene(edge_idx=1, n_genes=n_imp, pattern="impulse", include_cycle_rim=True)
            mark_dynamic(n_imp)
        else:
            synth.add_trend_gene(edge_idx=0, n_genes=n_imp, pattern="impulse", include_cycle_rim=True)
            mark_dynamic(n_imp)
    if any(isinstance(n, CycleNode) for n in system.nodes):
        if n_edges == 0:
            n_cycle = max(n_cycle, n_genes // 3)
        phases = np.linspace(0, 2 * np.pi, n_cycle, endpoint=False)
        synth.add_cycle_gene(node_id="Cycle", n_genes=n_cycle, phase=phases.tolist(), angular_sigma=0.5)
        mark_dynamic(n_cycle)

    n_noise = n_genes - synth.current_gene_idx
    synth.add_noise_gene(n_genes=n_noise, alpha_range=(0.01, 0.06))
    mark_noise(n_noise)
    synth.gene_names = [f"g{i}" for i in range(n_genes)]
    return synth, gene_types


def generate_kinetics_bundle_typed(
    system: UniversalGraphSystem,
    traj: torch.Tensor,
    n_genes: int,
) -> Dict[str, np.ndarray]:
    synth, gene_types = setup_synthesizer_typed(system, n_genes)
    with contextlib.redirect_stdout(io.StringIO()):
        clean = synth.generate_kinetics(traj, dt=DT, scale=ALPHA_SCALE, simulate_counts=False)
        counts = synth.generate_kinetics(traj, dt=DT, scale=ALPHA_SCALE, simulate_counts=True)
    t_steps, n_cells, _ = traj.shape
    n_obs = t_steps * n_cells

    def flat(x: torch.Tensor) -> np.ndarray:
        return x.reshape(n_obs, -1).detach().cpu().numpy()

    return {
        "spliced_counts": flat(counts["spliced"]),
        "unspliced_counts": flat(counts["unspliced"]),
        "velocity_gt": flat(clean["velocity"]),
        "gene_names": np.array(synth.gene_names),
        "gene_types": gene_types,
    }


# ---------------------------------------------------------------------------
# scVelo + diagnostics
# ---------------------------------------------------------------------------
def run_scvelo(
    s_counts: np.ndarray,
    u_counts: np.ndarray,
    z_latent: np.ndarray,
    v_latent_gt: np.ndarray,
    gene_names: np.ndarray,
    *,
    n_top_genes: Optional[int] = 200,
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
    if n_top_genes is not None and adata.n_vars > n_top_genes:
        s_tot = np.asarray(adata.layers["spliced"].sum(axis=0)).ravel()
        u_tot = np.asarray(adata.layers["unspliced"].sum(axis=0)).ravel()
        top_idx = np.argsort(s_tot + u_tot)[::-1][:n_top_genes]
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
        "hvg_mask_full": np.isin(gene_names.astype(str), adata.var_names.astype(str)),
    }


def embed_velocity_on_umap(adata, velocity: np.ndarray, vkey: str = "velocity_in") -> np.ndarray:
    import scvelo as scv

    a = adata.copy()
    a.layers[vkey] = velocity
    scv.tl.velocity_graph(a, vkey=vkey)
    scv.tl.velocity_embedding(a, basis="umap", vkey=vkey)
    return np.asarray(a.obsm[f"{vkey}_umap"])


def pca_projectable_fraction(
    adata,
    velocity_gt: np.ndarray,
    gene_names_full: np.ndarray,
    gene_names_used: np.ndarray,
    n_pcs: int = 30,
) -> np.ndarray:
    """Fraction of ||v_gt|| captured by top expression PCs (M1 proxy)."""
    name_to_idx = {g: i for i, g in enumerate(gene_names_full.astype(str))}
    cols = [name_to_idx[g] for g in gene_names_used]
    v = velocity_gt[:, cols]

    x = np.asarray(adata.layers["Ms"] if "Ms" in adata.layers else adata.X)
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    k = min(n_pcs, vt.shape[0])
    basis = vt[:k].T  # (n_genes_used, k)
    coeff = v @ basis
    v_sub = coeff @ basis.T
    num = np.linalg.norm(v_sub, axis=1)
    den = np.linalg.norm(v, axis=1) + 1e-8
    return num / den


def module_cosines(
    v_pred: np.ndarray,
    v_gt: np.ndarray,
    gene_names_used: np.ndarray,
    gene_names_full: np.ndarray,
    gene_types_full: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Per-cell cosine restricted to dynamic (0) or noise (1) genes among used set."""
    name_to_idx = {g: i for i, g in enumerate(gene_names_full.astype(str))}
    used_types = np.array([gene_types_full[name_to_idx[g]] for g in gene_names_used])
    out: Dict[str, np.ndarray] = {}
    for label, code in (("dynamic", 0), ("noise", 1)):
        mask = used_types == code
        if mask.sum() < 3:
            out[f"cos_{label}"] = np.full(v_pred.shape[0], np.nan)
            continue
        out[f"cos_{label}"] = cosine_rows(v_pred[:, mask], v_gt[:, mask])
    out["frac_noise_used"] = np.full(v_pred.shape[0], float((used_types == 1).mean()))
    return out


def analyze_case(
    n_genes: int,
    *,
    n_top_genes: Optional[int] = 200,
    reuse_bundle: Optional[Dict[str, np.ndarray]] = None,
    reuse_traj: Optional[torch.Tensor] = None,
    reuse_system: Optional[UniversalGraphSystem] = None,
) -> Dict[str, Any]:
    system = reuse_system or build_cycle_bifurcation_system()
    traj = reuse_traj
    if traj is None:
        traj = simulate_trajectory(system, TOPOLOGY, N_CELLS, N_STEPS, SAVE_EVERY)

    bundle = reuse_bundle
    if bundle is None or bundle["velocity_gt"].shape[1] != n_genes:
        bundle = generate_kinetics_bundle_typed(system, traj, n_genes)

    z_all = flatten_traj(traj)
    v_lat_all = latent_velocity_from_traj(traj, DT)
    idx = subsample_observations(len(z_all), SUBSAMPLE, SEED)

    s = bundle["spliced_counts"][idx]
    u = bundle["unspliced_counts"][idx]
    z = z_all[idx]
    v_lat = v_lat_all[idx]
    v_gt_full = bundle["velocity_gt"][idx]
    genes = bundle["gene_names"]
    gene_types = bundle["gene_types"]

    sc = run_scvelo(s, u, z, v_lat, genes, n_top_genes=n_top_genes)
    v_gt = align_velocity_gt(v_gt_full, genes, sc["gene_names_used"])
    v_pred = sc["velocity_pred"]

    cos_high = cosine_rows(v_pred, v_gt)
    cos_low = cosine_rows(sc["velocity_umap"], v_lat)
    delta_proj = delta_projection_masking(cos_low, cos_high)
    mm = mismatch_mask(cos_low, cos_high)

    # M4: GT embedding ceiling
    v_umap_gt = embed_velocity_on_umap(sc["adata"], v_gt)
    cos_low_gt = cosine_rows(v_umap_gt, v_lat)

    # M1: PCA subspace capture
    pca_frac = pca_projectable_fraction(sc["adata"], v_gt_full, genes, sc["gene_names_used"])

    # M3: module-stratified cosines
    mod = module_cosines(v_pred, v_gt, sc["gene_names_used"], genes, gene_types)

    regions = assign_regions(system, z)
    region_mm = {k: float(mm[v].mean()) if v.any() else float("nan") for k, v in regions.items()}

    return {
        "n_genes": n_genes,
        "n_top_genes": n_top_genes,
        "system": system,
        "z": z,
        "v_latent_gt": v_lat,
        "cos_high": cos_high,
        "cos_low": cos_low,
        "cos_gene": cos_high,
        "cos_umap": cos_low,
        "delta_proj": delta_proj,
        "cos_low_gt": cos_low_gt,
        "pca_frac": pca_frac,
        "mismatch": mm,
        "mismatch_frac": float(mm.mean()),
        "regions": regions,
        "region_mismatch_frac": region_mm,
        "velocity_umap": sc["velocity_umap"],
        "velocity_umap_gt": v_umap_gt,
        "n_obs": len(z),
        "frac_noise_genes": float((gene_types == 1).mean()),
        **mod,
    }


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def run_gene_sweep(gene_list: List[int]) -> Tuple[List[Dict[str, Any]], List[Dict]]:
    system = build_cycle_bifurcation_system()
    traj = simulate_trajectory(system, TOPOLOGY, N_CELLS, N_STEPS, SAVE_EVERY)
    max_genes = max(gene_list)
    bundle_max = generate_kinetics_bundle_typed(system, traj, max_genes)

    results: List[Dict[str, Any]] = []
    rows: List[Dict] = []
    for ng in gene_list:
        print(f"  n_genes={ng}...")
        bundle = bundle_max if ng == max_genes else generate_kinetics_bundle_typed(system, traj, ng)
        res = analyze_case(
            ng,
            reuse_system=system,
            reuse_traj=traj,
            reuse_bundle=bundle,
        )
        results.append(res)
        rows.append(
            {
                "n_genes": ng,
                "mismatch_frac": res["mismatch_frac"],
                "mean_cos_high": float(np.nanmean(res["cos_high"])),
                "mean_cos_low": float(np.nanmean(res["cos_low"])),
                "mean_cos_gene": float(np.nanmean(res["cos_gene"])),
                "mean_cos_umap": float(np.nanmean(res["cos_umap"])),
                "mean_delta_proj": float(np.nanmean(res["delta_proj"])),
                "frac_delta_proj_positive": float(np.mean(res["delta_proj"] > 0)),
                "mean_cos_low_gt": float(np.nanmean(res["cos_low_gt"])),
                "mean_cos_dynamic": float(np.nanmean(res["cos_dynamic"])),
                "mean_cos_noise": float(np.nanmean(res["cos_noise"])),
                "mean_pca_frac": float(np.nanmean(res["pca_frac"])),
                "frac_noise_genes": res["frac_noise_genes"],
            }
        )
        print(
            f"    mismatch={res['mismatch_frac']:.3f}  "
            f"cos_high={rows[-1]['mean_cos_high']:.3f}  cos_low={rows[-1]['mean_cos_low']:.3f}"
        )
    return results, rows


def run_hvg_ablation(main_n_genes: int = 2000) -> List[Dict]:
    rows = []
    for n_top in (200, 500, None):
        label = "all" if n_top is None else str(n_top)
        print(f"  HVG ablation n_top={label}...")
        res = analyze_case(main_n_genes, n_top_genes=n_top)
        rows.append(
            {
                "n_top_genes": label,
                "n_genes_total": main_n_genes,
                "mismatch_frac": res["mismatch_frac"],
                "mean_cos_high": float(np.nanmean(res["cos_high"])),
                "mean_cos_low": float(np.nanmean(res["cos_low"])),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _cell_region_labels(regions: Dict[str, np.ndarray], n: int) -> np.ndarray:
    labels = np.full(n, "other", dtype=object)
    for name in reversed(REGION_PRIORITY):
        if name in regions:
            labels[regions[name]] = name
    return labels


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho)


def _binned_mean_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < n_bins:
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers, means, lo, hi = [], [], [], []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1] if i < n_bins - 1 else x <= edges[i + 1])
        if m.sum() < 5:
            continue
        vals = y[m]
        mu = float(np.mean(vals))
        se = float(np.std(vals) / np.sqrt(len(vals)))
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(mu)
        lo.append(mu - 1.96 * se)
        hi.append(mu + 1.96 * se)
    return np.array(centers), np.array(means), np.array(lo), np.array(hi)


def _attach_regions(main: Dict[str, Any]) -> Dict[str, Any]:
    if "regions" not in main:
        main["regions"] = assign_regions(main["system"], main["z"])
    main["region_labels"] = _cell_region_labels(main["regions"], len(main["z"]))
    return main


def region_masking_stats(main: Dict[str, Any]) -> List[Dict[str, float]]:
    """Masking-cell fraction within each topological region."""
    main = _attach_regions(main)
    mm = main["mismatch"]
    rows: List[Dict[str, float]] = []
    for reg in REGION_PRIORITY + ("other",):
        sel = main["region_labels"] == reg
        n = int(sel.sum())
        if n == 0:
            continue
        rows.append(
            {
                "region": reg,
                "n_cells": n,
                "masking_frac": float(mm[sel].mean()),
                "n_masking": int(mm[sel].sum()),
            }
        )
    rows.sort(key=lambda r: r["masking_frac"], reverse=True)
    return rows


def _subsample_idx(n: int, n_keep: int, seed: int = 0) -> np.ndarray:
    n_keep = min(n_keep, n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=n_keep, replace=False))


def _draw_topology_map(ax: plt.Axes, main: Dict[str, Any], *, panel_letter: str = "D") -> None:
    """Latent topology map — pale background, masking cells as hollow red rings."""
    z = main["z"]
    mm = main["mismatch"]
    labels = main["region_labels"]

    for reg in REGION_PRIORITY + ("other",):
        sel = labels == reg
        if not sel.any():
            continue
        ax.scatter(
            z[sel, 0],
            z[sel, 1],
            s=1.8,
            c=REGION_COLORS.get(reg, "#DDDDDD"),
            alpha=0.12 if reg != "other" else 0.06,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    if mm.any():
        ax.scatter(
            z[mm, 0],
            z[mm, 1],
            s=36,
            facecolors="none",
            edgecolors="#CC0000",
            linewidths=1.15,
            rasterized=True,
            zorder=6,
            label="masking cell",
        )

    for edge in main["system"].edges:
        s, e = edge.start.cpu().numpy(), edge.end.cpu().numpy()
        ax.plot([s[0], e[0]], [s[1], e[1]], color="#999999", ls="--", lw=0.65, alpha=0.55, zorder=2)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{panel_letter}  Masking cells on cycle–bifurcation topology")
    ax.legend(fontsize=6, loc="lower left", frameon=False, markerscale=0.9)


def _draw_region_fraction_bar(ax: plt.Axes, main: Dict[str, Any]) -> None:
    """Horizontal bar chart: masking-cell fraction per topological region."""
    stats = region_masking_stats(main)
    if not stats:
        ax.set_visible(False)
        return

    # Consistent region order (not sorted by value) for readability
    order = [r for r in REGION_PRIORITY if any(s["region"] == r for s in stats)]
    order += [s["region"] for s in stats if s["region"] not in order]
    fracs = [next(s["masking_frac"] for s in stats if s["region"] == r) for r in order]
    colors = [REGION_COLORS.get(r, "#AAAAAA") for r in order]
    labels = [r.replace("_", " ") for r in order]
    y = np.arange(len(order))

    ax.barh(y, fracs, color=colors, edgecolor="none", height=0.62, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Masking-cell fraction")
    ax.set_xlim(0, max(fracs) * 1.22 + 0.02)
    ax.invert_yaxis()
    for i, (f, n) in enumerate(zip(fracs, order)):
        n_mask = next(s["n_masking"] for s in stats if s["region"] == n)
        ax.text(f + 0.008, i, f"{f:.0%} ({n_mask})", va="center", fontsize=6)
    overall = float(main["mismatch"].mean())
    ax.axvline(overall, color="#CC0000", ls="--", lw=0.9, alpha=0.7)
    ax.text(
        overall + 0.005,
        len(order) - 0.35,
        f"overall {overall:.0%}",
        color="#CC0000",
        fontsize=6,
        va="top",
    )


def _draw_embedding_ceiling_panel(ax: plt.Axes, main: Dict[str, Any], *, panel_letter: str = "C") -> None:
    """C^2D_GT embed vs C^2D_pred — faint background, masking cells highlighted."""
    cgt = main["cos_low_gt"]
    cpred = main["cos_low"]
    mm = main["mismatch"]
    labels = main["region_labels"]

    bg = ~mm
    if bg.any():
        bg_idx = _subsample_idx(int(bg.sum()), min(900, int(bg.sum())), seed=SEED)
        bg_cells = np.where(bg)[0][bg_idx]
        for reg in REGION_PRIORITY + ("other",):
            sel = np.isin(bg_cells, np.where(labels == reg)[0])
            if not sel.any():
                continue
            idx = bg_cells[sel]
            ax.scatter(
                cgt[idx],
                cpred[idx],
                s=2.0,
                c=REGION_COLORS.get(reg, "#DDDDDD"),
                alpha=0.10,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

    if mm.any():
        ax.scatter(
            cgt[mm],
            cpred[mm],
            s=28,
            facecolors="none",
            edgecolors="#CC0000",
            linewidths=1.1,
            rasterized=True,
            zorder=5,
            label="masking cell",
        )

    gt_ceiling = float(np.nanmean(cgt))
    ax.axhline(
        TAU_LOW,
        color="#CC0000",
        ls="--",
        lw=1.0,
        alpha=0.85,
        label=rf"masking threshold ($C^{{2D}}_{{\mathrm{{pred}}}}={TAU_LOW}$)",
    )
    ax.axhline(
        gt_ceiling,
        color="#228833",
        ls=":",
        lw=1.0,
        alpha=0.9,
        label=rf"mean $C^{{2D}}_{{\mathrm{{GT\ embed}}}}$={gt_ceiling:.2f}",
    )
    ax.plot([-1, 1], [-1, 1], "k--", lw=0.6, alpha=0.25)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$C^{2D}_{\mathrm{GT\ embed}}$")
    ax.set_ylabel(r"$C^{2D}_{\mathrm{pred}}$")
    ax.set_title(f"{panel_letter}  Low-dimensional projection imposes an embedding ceiling")

    top_regs = region_masking_stats(main)[:2]
    reg_txt = ", ".join(f"{r['region']}: {r['masking_frac']:.0%}" for r in top_regs)
    ax.text(
        0.97,
        0.03,
        f"masking: {mm.mean():.1%} ({int(mm.sum())}/{len(mm)})\n"
        f"top regions — {reg_txt}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.92),
    )
    ax.legend(fontsize=5.5, loc="upper left", frameon=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_diagnosis(
    sweep: List[Dict[str, Any]],
    sweep_rows: List[Dict],
    main: Dict[str, Any],
) -> plt.Figure:
    """Main figure: A–C (gene sweeps) + D (topology map + region bar)."""
    main = _attach_regions(main)
    apply_publication_style()
    fig = plt.figure(figsize=(9.0, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.40, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_topo = fig.add_subplot(gs[1, 0:2])
    ax_bar = fig.add_subplot(gs[1, 2])

    ng = [int(r["n_genes"]) for r in sweep_rows]

    # A — mismatch persists across gene numbers
    mm_frac = [float(r["mismatch_frac"]) for r in sweep_rows]
    ax_a.plot(ng, mm_frac, "o-", color="C1", lw=1.5, ms=4)
    ax_a.set_xlabel("Total genes")
    ax_a.set_ylabel("Masking-cell fraction")
    ax_a.set_title("A  Mismatch persists across gene numbers")
    ax_a.set_ylim(0, max(mm_frac) * 1.2 + 0.02)
    ax_a.text(
        0.03,
        0.97,
        rf"masking: $C^{{2D}}_{{\mathrm{{pred}}}}>{TAU_LOW:.1f}$, "
        rf"$C^G_{{\mathrm{{pred}}}}<{TAU_HIGH:.1f}$",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=6,
    )

    # B — three decoupled agreement measures
    ax_b.plot(ng, [float(r["mean_cos_high"]) for r in sweep_rows], "o-", label=r"$C^G_{\mathrm{pred}}$", color="C0", lw=1.5)
    ax_b.plot(ng, [float(r["mean_cos_low"]) for r in sweep_rows], "s-", label=r"$C^{2D}_{\mathrm{pred}}$", color="C1", lw=1.5)
    ax_b.plot(ng, [float(r["mean_cos_low_gt"]) for r in sweep_rows], "^--", label=r"$C^{2D}_{\mathrm{GT\ embed}}$", color="C2", lw=1.2)
    ax_b.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.45)
    ax_b.set_xlabel("Total genes")
    ax_b.set_ylabel("Mean cosine vs GT")
    ax_b.set_title("B  Embedding- and gene-space agreements are decoupled")
    ax_b.legend(fontsize=5.5, loc="lower left", frameon=False)

    # C — dynamic vs noise gene-space error
    ax_c.plot(ng, [float(r["mean_cos_dynamic"]) for r in sweep_rows], "o-", label="dynamic genes", color="C0", lw=1.5)
    ax_c.plot(ng, [float(r["mean_cos_noise"]) for r in sweep_rows], "s-", label="noise genes", color="C3", lw=1.5)
    ax_c.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.45)
    ax_c.set_xlabel("Total genes")
    ax_c.set_ylabel(r"Mean $C^G_{\mathrm{pred}}$")
    ax_c.set_title("C  Gene-space errors concentrate in dynamic genes")
    ax_c.legend(fontsize=6, frameon=False)

    _draw_topology_map(ax_topo, main, panel_letter="E")
    _draw_region_fraction_bar(ax_bar, main)
    ax_bar.set_title("E  Masking fraction by region", fontsize=8, pad=6)

    fig.suptitle(EVALUATION_CEILING_CAPTION, fontsize=8.5, y=0.99)
    return fig


def plot_supplementary(main: Dict[str, Any], sweep_rows: List[Dict]) -> plt.Figure:
    """Supplementary: embedding ceiling scatter, Δ_proj histogram, PCA capture."""
    main = _attach_regions(main)
    apply_publication_style()
    fig = plt.figure(figsize=(8.0, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.32)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_pca = fig.add_subplot(gs[0, 1])
    ax_ceil = fig.add_subplot(gs[1, :])
    mm = main["mismatch"]

    vals = main["delta_proj"][np.isfinite(main["delta_proj"])]
    ax_hist.hist(vals, bins=60, color="#BBBBBB", alpha=0.75, edgecolor="none", label="all cells")
    if mm.any():
        ax_hist.hist(
            main["delta_proj"][mm],
            bins=60,
            color="#CC0000",
            alpha=0.45,
            edgecolor="none",
            label="masking cells",
        )
    ax_hist.axvline(0.0, color="k", ls="--", lw=0.9, alpha=0.55)
    mean_d = float(np.mean(vals))
    ax_hist.axvline(mean_d, color="k", ls="-", lw=1.0)
    ax_hist.set_xlabel(r"$\Delta_{\mathrm{proj}} = C^{2D}_{\mathrm{pred}} - C^G_{\mathrm{pred}}$")
    ax_hist.set_ylabel("# cells")
    ax_hist.set_title("A  Projection masking index")
    ax_hist.legend(fontsize=6, frameon=False)
    ax_hist.text(0.97, 0.95, rf"mean={mean_d:.2f}", transform=ax_hist.transAxes, ha="right", va="top", fontsize=7)

    xc, ym, ylo, yhi = _binned_mean_ci(main["pca_frac"], main["delta_proj"], n_bins=10)
    if xc.size:
        yerr = np.vstack([ym - ylo, yhi - ym])
        ax_pca.errorbar(xc, ym, yerr=yerr, fmt="o-", color="C0", lw=1.2, ms=4, capsize=2.5)
    rho = _spearman_rho(main["pca_frac"], main["delta_proj"])
    ax_pca.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.45)
    ax_pca.set_xlabel("PCA capture of GT velocity")
    ax_pca.set_ylabel(r"$\Delta_{\mathrm{proj}}$")
    ax_pca.set_title("B  Gene-space capture vs $\\Delta_{\\mathrm{proj}}$ (weak)")
    ax_pca.text(
        0.97,
        0.03,
        rf"Spearman $\rho={rho:.2f}$" if np.isfinite(rho) else "",
        transform=ax_pca.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
    )

    _draw_embedding_ceiling_panel(ax_ceil, main, panel_letter="C")

    fig.suptitle("Supplementary: projection masking diagnostics", fontsize=9, y=0.99)
    return fig


def plot_simulation_projection_masking(main: Dict[str, Any], sweep_rows: List[Dict]) -> plt.Figure:
    """Alias kept for backward compatibility — routes to supplementary figure."""
    return plot_supplementary(main, sweep_rows)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_cell_diagnostics(path: Path, res: Dict[str, Any]) -> None:
    rows = []
    for i in range(res["n_obs"]):
        rows.append(
            {
                "cell_idx": i,
                "n_genes": res["n_genes"],
                "z1": float(res["z"][i, 0]),
                "z2": float(res["z"][i, 1]),
                "cos_high": float(res["cos_high"][i]),
                "cos_low": float(res["cos_low"][i]),
                "cos_gene": float(res["cos_gene"][i]),
                "cos_umap": float(res["cos_umap"][i]),
                "delta_proj": float(res["delta_proj"][i]),
                "cos_low_gt": float(res["cos_low_gt"][i]),
                "cos_dynamic": float(res["cos_dynamic"][i]) if not np.isnan(res["cos_dynamic"][i]) else "",
                "cos_noise": float(res["cos_noise"][i]) if not np.isnan(res["cos_noise"][i]) else "",
                "pca_frac": float(res["pca_frac"][i]),
                "mismatch": int(res["mismatch"][i]),
            }
        )
    save_csv(path, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(quick: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    gene_list = GENE_SWEEP_QUICK if quick else GENE_SWEEP_FULL
    main_n_genes = gene_list[-1]

    print("=== Gene-count sweep (M2) ===")
    sweep, sweep_rows = run_gene_sweep(gene_list)
    save_csv(DATA_DIR / "gene_sweep_summary.csv", sweep_rows)

    print(f"\n=== Main diagnosis @ n_genes={main_n_genes} ===")
    main = sweep[-1]
    save_cell_diagnostics(DATA_DIR / "cell_diagnostics.csv", main)
    reg_rows = region_masking_stats(main)
    save_csv(DATA_DIR / "masking_by_region.csv", reg_rows)
    print("  Masking by region (top):", reg_rows[:3])

    print("\n=== HVG ablation (M3) ===")
    hvg_rows = run_hvg_ablation(main_n_genes)
    save_csv(DATA_DIR / "hvg_ablation.csv", hvg_rows)

    # Mechanism summary stats
    mm = main["mismatch"]
    cos_low_gt_mm = main["cos_low_gt"][mm]
    summary = {
        "mismatch_frac": main["mismatch_frac"],
        "mismatch_cos_low_mean": float(np.nanmean(main["cos_low"][mm])) if mm.any() else float("nan"),
        "mismatch_cos_high_mean": float(np.nanmean(main["cos_high"][mm])) if mm.any() else float("nan"),
        "agree_cos_low_mean": float(np.nanmean(main["cos_low"][~mm])),
        "agree_cos_high_mean": float(np.nanmean(main["cos_high"][~mm])),
        "gt_embed_cos_low_mean_mismatch": float(np.nanmean(cos_low_gt_mm)) if mm.any() else float("nan"),
        "frac_mismatch_gt_embed_ok": float(np.mean(cos_low_gt_mm > TAU_LOW)) if mm.any() else float("nan"),
        "pca_frac_mismatch_mean": float(np.nanmean(main["pca_frac"][mm])) if mm.any() else float("nan"),
        "pca_frac_agree_mean": float(np.nanmean(main["pca_frac"][~mm])),
    }
    save_csv(DATA_DIR / "mechanism_summary.csv", [summary])
    print("  Mechanism summary:", summary)

    pm_stats = summarize_delta_proj(main["delta_proj"])
    save_csv(
        DATA_DIR / "projection_masking_index_summary.csv",
        [{"method": "scvelo", "topology": TOPOLOGY, "n_genes": main_n_genes, **pm_stats}],
    )

    # Validate: mismatch fraction stable across gene counts (~20%)
    mm_vals = [float(r["mismatch_frac"]) for r in sweep_rows]
    stable_masking = mm_vals[-1] >= 0.15 and abs(mm_vals[-1] - mm_vals[0]) < 0.05
    print(f"  Masking fraction stable across gene counts (~20%): {stable_masking}")

    print("\nPlotting...")
    fig = plot_diagnosis(sweep, sweep_rows, main)
    save_figure(fig, FIG_DIR / "projection_masking_diagnosis")

    fig_supp = plot_supplementary(main, sweep_rows)
    save_figure(fig_supp, FIG_DIR / "projection_masking_supplementary")

    print(f"\nSaved {FIG_DIR / 'projection_masking_diagnosis.png'}")
    print(f"Saved {FIG_DIR / 'projection_masking_supplementary.png'}")


def plot_only() -> None:
    import csv as csv_mod

    sweep_path = DATA_DIR / "gene_sweep_summary.csv"
    cell_path = DATA_DIR / "cell_diagnostics.csv"
    if not sweep_path.exists() or not cell_path.exists():
        raise FileNotFoundError("Run full pipeline first (missing cached CSV).")

    with sweep_path.open(newline="") as f:
        sweep_rows = list(csv_mod.DictReader(f))
    with cell_path.open(newline="") as f:
        cell_rows = list(csv_mod.DictReader(f))

    main_n = int(sweep_rows[-1]["n_genes"])
    z = np.array([[float(r["z1"]), float(r["z2"])] for r in cell_rows])
    main = {
        "n_genes": main_n,
        "mismatch_frac": float(sweep_rows[-1]["mismatch_frac"]),
        "z": z,
        "cos_high": np.array([float(r["cos_high"]) for r in cell_rows]),
        "cos_low": np.array([float(r["cos_low"]) for r in cell_rows]),
        "cos_gene": np.array([float(r.get("cos_gene", r["cos_high"])) for r in cell_rows]),
        "cos_umap": np.array([float(r.get("cos_umap", r["cos_low"])) for r in cell_rows]),
        "delta_proj": np.array([
            float(r["delta_proj"]) if r.get("delta_proj", "") != "" else float(r["cos_low"]) - float(r["cos_high"])
            for r in cell_rows
        ]),
        "cos_low_gt": np.array([float(r["cos_low_gt"]) for r in cell_rows]),
        "pca_frac": np.array([float(r["pca_frac"]) for r in cell_rows]),
        "mismatch": np.array([bool(int(r["mismatch"])) for r in cell_rows]),
        "system": build_cycle_bifurcation_system(),
    }
    sweep = [{"n_genes": int(r["n_genes"])} for r in sweep_rows]
    fig = plot_diagnosis(sweep, sweep_rows, main)
    save_figure(fig, FIG_DIR / "projection_masking_diagnosis")
    fig_supp = plot_supplementary(main, sweep_rows)
    save_figure(fig_supp, FIG_DIR / "projection_masking_supplementary")
    print(f"Saved {FIG_DIR / 'projection_masking_diagnosis.png'}")
    print(f"Saved {FIG_DIR / 'projection_masking_supplementary.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projection-masking mechanism diagnosis")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Smaller gene sweep for fast iteration")
    args = parser.parse_args()
    if args.plot_only:
        plot_only()
    else:
        main(quick=args.quick)
