"""
Figure4Series.ipynb data-generation pipeline.

TWO-STAGE DESIGN
================
Stage 1  (slow, ~7 min, cached in kinetics_cache.pt)
  PAGA topology → GraphSimulator → GeneSynthesizer.generate_kinetics
  Cache key: simulation hyper-parameters (SEED, N_STEPS, N_CELLS, …)
  Result: baseline Poisson counts (no dropout, canonical lib-size 50 000)
          + latent trajectory z + real pancreas matrices + system + config

Stage 2  (fast, <10 s, always re-runs)  ← observation noise layer
  PostProcessConfig → _apply_count_noise → _add_noise_genes
  Tunable without touching the cache:
    lib_size_mean, lib_size_sd, sampling, nb_theta, dropout_prob, dropout_slope,
    cell_size_sigma, downsample_spliced, downsample_unspliced, chunk_size

To force a fresh Stage-1 run, delete  kinetics_cache.pt.
"""

from __future__ import annotations

import io
import contextlib
import hashlib
import json
import pickle
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scanpy as sc
import torch
import yaml
from scipy import sparse

from data_generation_module import GeneSynthesizer, GraphEdge, GraphNode, UniversalGraphSystem
from data_generation_module import GraphSimulator
from data_generation_module.generate_gene_config import auto_generate_config_from_system

# ---------------------------------------------------------------------------
# Simulation hyper-parameters  (changing any of these invalidates the cache)
# ---------------------------------------------------------------------------
SEED              = 42
N_SAMPLES         = 1000
N_CELLS           = 5000
D_GENES_BASE      = 3000
N_STEPS           = 150
DT                = 0.05
NOISE_LEVEL       = 0.15
ALPHA_SCALE       = 1.5
N_SAMPLING_STEPS  = 20
PAGA_MIN_CONNECTIVITY = 0.04
FUSION_TEMP       = 0.05
TARGET_N_GENES    = 27000

# ---------------------------------------------------------------------------
# Post-processing / noise-model parameters  (fast to retune, Stage 2)
# ---------------------------------------------------------------------------
@dataclass
class PostProcessConfig:
    """Observation-noise layer parameters.  Edit freely – Stage-1 kinetics never rerun.

    Two-layer separation
    --------------------
    continuous_kinetics  ← Stage 1, cached, not touched here
    observation noise    ← this dataclass: lib_size / NB / dropout / downsampling

    Tuning guide
    ------------
    Panel A (library size)       → lib_size_mean, lib_size_sd
    Panel B (genes detected %)   → downsample_spliced/unspliced, dropout_prob/slope
    Panel D/I (overdispersion)   → sampling="nb", nb_theta (smaller = more overdispersed)
    Panel H (Fano factor width)  → nb_theta, nb_theta_gene_sd (gene-level dispersion heterogeneity)
    Panel F centre (unspliced fraction) → downsample_unspliced (shifts centre)
    Panel F width  (peak spread)        → u_cell_size_sigma  (widens peak)
    """
    # ── Library size ─────────────────────────────────────────────────────
    lib_size_mean:       float = 65000.0  # target median lib-size (pre-downsampling)
    lib_size_sd:         float = 0.30     # log-scale std of per-cell lib-size
    cell_size_sigma:     float = 0.20     # lognormal σ for additional cell-size variation

    # ── Count sampling (Poisson or NB) ───────────────────────────────────
    sampling:            str   = "nb"     # "poisson" or "nb"
    nb_theta:            float = 20.0      # NB dispersion; smaller → more overdispersed
    nb_theta_gene_sd:    float = 0.55     # log-σ of per-gene θ; widens Fano distribution
    chunk_size:          int   = 512      # rows per chunk for NB Gamma-Poisson sampling

    # ── Dropout ──────────────────────────────────────────────────────────
    dropout_prob:        float = 0.66     # peak P(dropout) for zero-mean genes
    dropout_slope:       float = 0.20     # steepness of mean-dependent dropout decay

    # ── Binomial downsampling (capture rate) ─────────────────────────────
    downsample_spliced:   float = 0.12    # binomial capture for spliced counts
    downsample_unspliced: float = 0.04    # binomial capture for unspliced counts

    # ── Unspliced-specific variation (widens u_frac peak) ────────────────
    # Independent per-cell lognormal noise applied ONLY to unspliced counts,
    # after the shared cell-size scaling.  Physically: cells differ in burst
    # frequency / premRNA stability independent of total mRNA content.
    # Increase to widen the u_frac distribution without shifting its centre.
    u_cell_size_sigma:    float = 0.40    # σ of independent per-cell noise for unspliced

    seed:                int   = SEED

DEFAULT_PP_CONFIG = PostProcessConfig()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_PANCREAS_H5AD = Path(
    "/data/doujy/Documents/Code/Single_Cell_neuron/data/Pancreas/endocrinogenesis_day15.h5ad"
)
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "results" / "statistical_realism_figure4" / "data"
KINETICS_CACHE_PATH = _DEFAULT_CACHE_DIR / "kinetics_cache.pt"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class Figure4Result:
    system:        UniversalGraphSystem
    z:             np.ndarray
    spliced:       np.ndarray
    unspliced:     np.ndarray
    real_spliced:  np.ndarray
    real_unspliced: np.ndarray
    pseudotime:    np.ndarray
    config_dict:   dict


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------
def _sim_cache_key() -> str:
    """Stable string key for the Stage-1 cache.

    'continuous_kinetics' marks that Stage-1 stores raw continuous rates
    (simulate_counts=False), not Poisson integers.  Changing this string
    invalidates any older cache built with simulate_counts=True.
    """
    params = dict(
        continuous_kinetics=True,          # cache stores continuous rates, not counts
        seed=SEED, n_samples=N_SAMPLES, n_steps=N_STEPS, dt=DT,
        noise_level=NOISE_LEVEL, alpha_scale=ALPHA_SCALE,
        n_sampling_steps=N_SAMPLING_STEPS, n_cells=N_CELLS,
        d_genes_base=D_GENES_BASE, target_n_genes=TARGET_N_GENES,
        paga_min_conn=PAGA_MIN_CONNECTIVITY, fusion_temp=FUSION_TEMP,
    )
    return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _to_dense(x) -> np.ndarray:
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def load_pancreas_reference(h5ad_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, object]:
    import scvelo as scv
    path = Path(h5ad_path) if h5ad_path else DEFAULT_PANCREAS_H5AD
    if path.exists() and path.stat().st_size > 1_000_000:
        adata = sc.read_h5ad(path)
    else:
        adata = scv.datasets.pancreas()
    if "spliced" not in adata.layers or "unspliced" not in adata.layers:
        raise ValueError("Pancreas reference missing spliced/unspliced layers")
    return _to_dense(adata.layers["spliced"]), _to_dense(adata.layers["unspliced"]), adata


def _cluster_connectivities_from_neighbors(
    adata, cluster_key: str = "clusters"
) -> sparse.csr_matrix:
    clusters = adata.obs[cluster_key].astype("category")
    codes    = clusters.cat.codes.to_numpy()
    n_clust  = len(clusters.cat.categories)
    conn     = adata.obsp["connectivities"].tocsr()

    out = np.zeros((n_clust, n_clust), dtype=np.float64)
    for i in range(conn.shape[0]):
        s_, e_ = conn.indptr[i], conn.indptr[i + 1]
        ci = codes[i]
        if ci < 0:
            continue
        for j_idx in range(s_, e_):
            cj = codes[conn.indices[j_idx]]
            if cj < 0:
                continue
            out[ci, cj] += conn.data[j_idx]
            out[cj, ci] += conn.data[j_idx]

    out = np.maximum(out, out.T)
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    out = out / row_sums

    if "velocity_pseudotime" in adata.obs:
        pt = adata.obs["velocity_pseudotime"].to_numpy()
        pt_cluster = np.array([
            pt[codes == c].mean() if (codes == c).any() else 0.5
            for c in range(n_clust)
        ])
        directed = np.zeros_like(out)
        for i in range(n_clust):
            for j in range(n_clust):
                if i != j and out[i, j] > 0 and pt_cluster[i] <= pt_cluster[j]:
                    directed[i, j] = out[i, j]
        out = directed

    return sparse.csr_matrix(out)


def prepare_adata_for_paga(adata) -> object:
    import scvelo as scv
    adata_a = adata.copy()
    if "clusters" in adata.obs:
        adata_a.obs["clusters"] = adata.obs["clusters"].astype("category")
    sc.pp.normalize_per_cell(adata_a)
    sc.pp.log1p(adata_a)
    sc.pp.pca(adata_a, n_comps=30)
    sc.pp.neighbors(adata_a, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata_a)
    scv.pp.moments(adata_a, n_neighbors=30, n_pcs=30)
    scv.tl.velocity(adata_a, mode="stochastic")
    scv.tl.velocity_graph(adata_a)
    if "clusters" not in adata_a.obs:
        sc.tl.leiden(adata_a, resolution=0.5, key_added="clusters")
    adata_a.obs["clusters"] = adata_a.obs["clusters"].astype("category")
    try:
        scv.tl.paga(adata_a, groups="clusters")
    except Exception as exc_scv:
        try:
            sc.tl.paga(adata_a, groups="clusters", model="v1.0")
        except Exception:
            print(f"  Warning: PAGA failed ({exc_scv}); using neighbor-cluster fallback.")
            conn = _cluster_connectivities_from_neighbors(adata_a, "clusters")
            adata_a.uns["paga"] = {
                "groups": "clusters",
                "connectivities": conn,
                "connectivities_tree": conn.copy(),
            }
    return adata_a


def build_system_from_paga_topology(
    adata,
    *,
    use_umap_layout: bool = True,
    min_connectivity: float = PAGA_MIN_CONNECTIVITY,
) -> UniversalGraphSystem:
    clusters = adata.obs["clusters"].astype("category")
    cluster_names = clusters.cat.categories.tolist()
    paga_conn = adata.uns["paga"]["connectivities"]
    coo = paga_conn.tocoo() if hasattr(paga_conn, "tocoo") else sparse.coo_matrix(paga_conn)

    if use_umap_layout and "X_umap" in adata.obsm:
        umap = adata.obsm["X_umap"]
        centers = np.array([umap[clusters == c].mean(axis=0) for c in cluster_names])
        rng_ = centers.max(0) - centers.min(0); rng_[rng_ == 0] = 1.0
        normalized = (centers - centers.min(0)) / rng_ * 8.0 - 4.0
        pos = {c: normalized[i].tolist() for i, c in enumerate(cluster_names)}
    else:
        pos = {c: [i * 2.0, 0.0] for i, c in enumerate(cluster_names)}

    cc = clusters.value_counts(); mc = cc.max()
    nodes = [
        GraphNode(id=c, pos=pos[c],
                  potential=-1.0 - (cc[c] / mc) * 2.0,
                  sigma=0.3 + (cc[c] / mc) * 0.3,
                  basin_strength=0.1 * (cc[c] / mc),
                  basin_sigma=(0.3 + (cc[c] / mc) * 0.3) * 1.5)
        for c in cluster_names
    ]
    edges = [
        GraphEdge(nodes[i], nodes[j])
        for i, j, w in zip(coo.row, coo.col, coo.data)
        if w >= min_connectivity and i != j
    ]
    return UniversalGraphSystem(nodes, edges, fusion_temp=FUSION_TEMP)


def _find_start_node(system: UniversalGraphSystem) -> str:
    in_deg = {n.id: 0 for n in system.nodes}
    for e in system.edges:
        in_deg[e.node_end.id] += 1
    min_in = min(in_deg.values())
    cands = [nid for nid, d in in_deg.items() if d == min_in]
    if len(cands) > 1:
        return max(cands, key=lambda nid: next(n.potential for n in system.nodes if n.id == nid))
    return cands[0]


def _build_gene_config(system: UniversalGraphSystem, seed: int) -> dict:
    n_edges = len(system.edges)
    n_cycle = sum(1 for n in system.nodes if n.__class__.__name__ == "CycleNode")
    edge_f, cycle_f, bg_f = 0.3, (0.2 if n_cycle else 0.0), 0.1
    n_per_edge = int(D_GENES_BASE * edge_f)
    n_cyc      = int(D_GENES_BASE * cycle_f) if n_cycle else 0
    n_bg       = int(D_GENES_BASE * bg_f)
    diff = D_GENES_BASE - (n_per_edge * n_edges + n_cyc + n_bg)
    n_bg += diff
    egd = {
        i: {"sigmoid": int(n_per_edge * 0.4), "sigmoid_reverse": int(n_per_edge * 0.2),
            "sticky": int(n_per_edge * 0.2), "impulse": int(n_per_edge * 0.2)}
        for i in range(n_edges)
    }
    return auto_generate_config_from_system(
        system, n_genes_per_cycle=n_cyc, seed=seed,
        cycle_beta_range=(0.4, 4.0), cycle_gamma_range=(0.4, 2.5),
        edge_beta_range=(0.4, 4.5), edge_gamma_range=(0.3, 2.5),
        cycle_angular_sigma=1.0, sigmoid_center_range=(0.1, 0.9),
        sigmoid_slope_range=(3.0, 20.0), impulse_center_range=(0.05, 0.95),
        impulse_width_range=(0.03, 0.25), edge_gene_distribution=egd,
        n_noise_genes=n_bg, noise_alpha_range=(0.005, 0.05),
        noise_beta_range=(0.3, 1.5), noise_gamma_range=(0.2, 1.2),
        default_include_cycle_rim=True,
    )


def _sample_cells(
    xs_s: np.ndarray, xs_u: np.ndarray, traj: np.ndarray, seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total_steps = xs_s.shape[0] // N_SAMPLES
    start_step  = max(0, total_steps - N_SAMPLING_STEPS)
    start_idx, end_idx = start_step * N_SAMPLES, total_steps * N_SAMPLES
    available   = end_idx - start_idx
    n_take      = min(N_CELLS, available)
    rng = np.random.default_rng(seed)
    local_idx   = rng.choice(available, size=n_take, replace=False)
    indices     = local_idx + start_idx
    time_idx    = indices // N_SAMPLES
    pseudotime  = time_idx.astype(float) / (total_steps - 1) if total_steps > 1 else np.zeros(n_take)
    traj_flat   = traj.reshape(-1, 2)
    z           = traj_flat[indices]
    return xs_s[indices], xs_u[indices], z, np.clip(pseudotime, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Stage-2: fast noise-model post-processing
# ---------------------------------------------------------------------------
def _sample_counts(
    rng: np.random.Generator,
    lam: np.ndarray,
    sampling: str,
    nb_theta: float,
    theta_per_gene: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Discrete count sampling: Poisson or NB via Gamma-Poisson mixture.

    NB parameterisation: mean = lam, variance = lam + lam²/θ.
    Smaller θ → stronger overdispersion.  Works for non-integer θ.

    theta_per_gene: optional (n_genes,) array of per-gene dispersion parameters.
    When provided, each gene gets its own θ, producing heterogeneous Fano factors.
    """
    lam = np.maximum(lam, 0.0)
    if sampling == "poisson":
        return rng.poisson(lam).astype(np.float64)
    if theta_per_gene is not None:
        g = rng.gamma(shape=theta_per_gene[None, :], scale=lam / theta_per_gene[None, :])
    else:
        g = rng.gamma(shape=nb_theta, scale=lam / nb_theta)
    return rng.poisson(g).astype(np.float64)


def _sample_gene_thetas(
    rng: np.random.Generator, n_genes: int, cfg: PostProcessConfig,
) -> Optional[np.ndarray]:
    """Per-gene NB dispersion; None when gene-level heterogeneity is disabled."""
    if cfg.nb_theta_gene_sd <= 0.0:
        return None
    mu = np.log(cfg.nb_theta) - cfg.nb_theta_gene_sd ** 2 / 2.0
    return np.clip(
        rng.lognormal(mu, cfg.nb_theta_gene_sd, n_genes), 0.5, 200.0,
    )


def _apply_count_noise(
    xs_s_base: np.ndarray,
    xs_u_base: np.ndarray,
    cfg: PostProcessConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply lib-size scaling, dropout, count sampling, and binomial downsampling.

    Works on pre-cached baseline Poisson counts (no dropout, canonical lib-size)
    and produces the final discrete count matrices for validation.

    Step  1  per-cell lib-size (lognormal)
    Step  2  cell-size variation (lognormal)
    Step  3  rescale from cached to target lib-size
    Step  4  mean-dependent dropout
    Step  5  count sampling (Poisson or NB via cfg.sampling / cfg.nb_theta)
    Step  6  binomial downsampling (cfg.downsample_spliced / cfg.downsample_unspliced)
    """
    rng = np.random.default_rng(cfg.seed)
    n_cells, n_genes = xs_s_base.shape

    # 1. Per-cell target library size (lognormal)
    mu_lib     = np.log(cfg.lib_size_mean) - cfg.lib_size_sd**2 / 2
    target_lib = rng.lognormal(mu_lib, cfg.lib_size_sd, n_cells)

    # 2. Cell-size variation (lognormal)
    cs_mu     = -cfg.cell_size_sigma**2 / 2
    cell_size = np.clip(rng.lognormal(cs_mu, cfg.cell_size_sigma, n_cells), 0.3, 2.5)

    # 3. Scale continuous rates to target library size.
    #    xs_s_base / xs_u_base are now dimensionless expression rates (simulate_counts=False).
    #    We normalise each cell to sum to target_lib * cell_size so that the subsequent
    #    Poisson/NB draw produces the correct expected total count per cell.
    base_sum = (xs_s_base + xs_u_base).sum(1)       # per-cell sum of continuous rates
    scale    = target_lib * cell_size / (base_sum + 1e-8)
    xs_s     = xs_s_base * scale[:, None]
    xs_u     = xs_u_base * scale[:, None]

    # 4. Mean-dependent dropout  P(drop|g) = dropout_prob * exp(-dropout_slope * μ_g)
    gene_mean = xs_s.mean(0)
    p_drop    = cfg.dropout_prob * np.exp(-cfg.dropout_slope * (gene_mean + 1e-6))
    p_drop    = np.clip(p_drop, 0.0, 0.999)
    drop_mask = rng.random((n_cells, n_genes)) < p_drop[None, :]
    xs_s[drop_mask] = 0.0
    xs_u[drop_mask] = 0.0

    # Per-gene NB dispersion → heterogeneous Fano factors across genes
    theta_g = _sample_gene_thetas(rng, n_genes, cfg)

    # 4.5. Independent per-cell noise on unspliced only.
    #   This breaks the perfect u/s correlation introduced by the shared cell_size
    #   scaling, producing a wider u_frac distribution without shifting its centre.
    if cfg.u_cell_size_sigma > 0.0:
        u_sigma = cfg.u_cell_size_sigma
        u_noise = rng.lognormal(-u_sigma ** 2 / 2.0, u_sigma, n_cells)   # mean=1
        u_noise = np.clip(u_noise, 0.1, 10.0)
        xs_u = xs_u * u_noise[:, None]

    # 5+6. Count sampling with capture rate folded in – no separate binomial step.
    #   NB thinning:   Binomial(NB(θ, μ), p)     = NB(θ, μ·p)
    #   Poisson thin:  Binomial(Poisson(μ), p)   = Poisson(μ·p)
    #   → multiply means by capture rates before sampling; binomial becomes exact & free.
    #   Stack s and u along genes → one gamma+Poisson draw for both.
    xs_su = _sample_counts(
        rng,
        np.hstack([xs_s * cfg.downsample_spliced, xs_u * cfg.downsample_unspliced]),
        cfg.sampling, cfg.nb_theta,
        theta_per_gene=np.hstack([theta_g, theta_g]) if theta_g is not None else None,
    )                                                    # (n_cells, 2*n_genes)
    xs_s = xs_su[:, :n_genes]
    xs_u = xs_su[:, n_genes:]
    return xs_s, xs_u


def _add_noise_genes(
    xs_s: np.ndarray, xs_u: np.ndarray, pseudotime: np.ndarray,
    cfg: PostProcessConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add background noise genes to reach TARGET_N_GENES.

    Vectorised implementation – no Python loops over cells or genes.
    Gamma sampling is chunked over genes (GENE_CHUNK columns at a time) to cap
    peak memory at ≈ 2 × n_cells × GENE_CHUNK × 4 bytes instead of allocating
    the full (n_cells × n_noise) matrix at once.
    """
    n_noise = TARGET_N_GENES - xs_s.shape[1]
    if n_noise <= 0:
        return xs_s, xs_u

    GENE_CHUNK = cfg.chunk_size * 4   # gene-dimension chunk (default 2048)

    rng     = np.random.default_rng(cfg.seed + 10)
    n_cells = xs_s.shape[0]

    base_alpha    = np.clip(rng.exponential(scale=0.1, size=n_noise) + 0.005, 0.005, 0.5)
    beta_per_cell = 0.6 + (1.5 - 0.6) * pseudotime          # (n_cells,)
    sat_idx       = rng.choice(n_cells, size=int(n_cells * 0.25), replace=False)
    beta_factor   = np.ones(n_cells); beta_factor[sat_idx] = 0.4
    beta_per_cell *= beta_factor

    sig_a, sig_g = 0.5, 0.3
    # Vectorised: lognormal accepts array mean → single numpy call
    alpha_noise = rng.lognormal(np.log(base_alpha) - sig_a**2 / 2, sig_a)   # (n_noise,)
    gamma_noise = rng.lognormal(np.log(0.5)        - sig_g**2 / 2, sig_g, n_noise)

    # s_steady is constant across cells; u_steady varies per cell.
    s_scale_1d = alpha_noise / gamma_noise   # (n_noise,) – same for all cells
    # ── Analytic lib-size scaling (no sampling needed) ────────────────────
    # raw_nl_u per cell = Σ_g α_g / β_cell = α_noise.sum() / β_cell   (vectorised)
    # raw_nl_s = Σ_g α_g/γ_g = s_scale_1d.sum()                       (scalar, same all cells)
    raw_nl   = alpha_noise.sum() / beta_per_cell + s_scale_1d.sum()  # (n_cells,)
    core_lib = xs_s.sum(1) + xs_u.sum(1)                             # (n_cells,)
    comp     = 1.0 / (cfg.downsample_spliced + 1e-8)
    scale    = np.where(raw_nl > 0, core_lib * 0.40 * comp / (raw_nl + 1e-8), 1.0)

    # ── Gene-chunked NB/Poisson with scale + capture rate folded in ───────
    # NB thinning: Binomial(NB(θ,μ), p) = NB(θ, μ·p) → no separate binomial step.
    # Effective rate already incorporates lib-size scale AND capture rate.
    nu_parts: list = []
    ns_parts: list = []
    ds = cfg.downsample_spliced
    du = cfg.downsample_unspliced
    eff_u = (scale * du)[:, None]   # per-cell unspliced factor  (n_cells, 1)
    eff_s = (scale * ds)[:, None]   # per-cell spliced factor    (n_cells, 1)
    theta_noise = _sample_gene_thetas(rng, n_noise, cfg)

    for g0 in range(0, n_noise, GENE_CHUNK):
        g1  = min(g0 + GENE_CHUNK, n_noise)
        a_g = alpha_noise[g0:g1]                                      # (gc,)
        u_eff = (a_g[None, :] / beta_per_cell[:, None]) * eff_u      # (n_cells, gc)
        s_eff = s_scale_1d[g0:g1][None, :] * eff_s                   # (n_cells, gc)
        # Stack u+s → single NB draw per chunk; binomial already absorbed
        su = _sample_counts(
            rng, np.hstack([u_eff, s_eff]), cfg.sampling, cfg.nb_theta,
            theta_per_gene=(
                np.hstack([theta_noise[g0:g1], theta_noise[g0:g1]])
                if theta_noise is not None else None
            ),
        )                                                              # (n_cells, 2*gc)
        gc = g1 - g0
        nu_parts.append(su[:, :gc].astype(np.float32))
        ns_parts.append(su[:, gc:].astype(np.float32))

    noise_u = np.hstack(nu_parts).astype(np.float64)
    noise_s = np.hstack(ns_parts).astype(np.float64)
    return np.hstack([xs_s, noise_s]), np.hstack([xs_u, noise_u])


# ---------------------------------------------------------------------------
# Stage-1: build and save / load the kinetics cache
# ---------------------------------------------------------------------------
def _build_kinetics_cache(h5ad_path: Optional[Path] = None) -> dict:
    """Run the full slow pipeline and return a cache dict."""
    np.random.seed(SEED); torch.manual_seed(SEED)

    print("  Loading pancreas reference & building PAGA topology...")
    xr_s, xr_u, adata = load_pancreas_reference(h5ad_path)
    adata_a = prepare_adata_for_paga(adata)
    system  = build_system_from_paga_topology(adata_a)
    start   = _find_start_node(system)
    print(f"    nodes={len(system.nodes)}, edges={len(system.edges)}, start={start}")

    print("  Simulating trajectory (GraphSimulator)...")
    with contextlib.redirect_stdout(io.StringIO()):
        traj = GraphSimulator.run(system, start, n_cells=N_SAMPLES, steps=N_STEPS,
                                  dt=DT, noise_level=NOISE_LEVEL)

    print("  Generating gene config & continuous kinetics (simulate_counts=False)...")
    config_dict = _build_gene_config(system, SEED)
    # Stage 1 stores raw continuous rates – no lib-size scaling, no sampling, no dropout.
    # All observation noise is applied in Stage 2 (PostProcessConfig).
    synth = GeneSynthesizer(
        system=system, n_genes_total=len(config_dict["genes"]),
        edge_strength=5.0, cycle_strength=3.0, background_noise=0.1,
        splicing_rate=1.0, degradation_rate=0.5,
        seed=SEED, enable_global_decay=False,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_dict, tmp, default_flow_style=False, sort_keys=False)
        cfg_path = tmp.name
    synth.load_from_config(cfg_path, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        kinetics = synth.generate_kinetics(traj, dt=DT, simulate_counts=False, scale=ALPHA_SCALE)

    xs_s = kinetics["spliced"];   xs_u = kinetics["unspliced"]
    if isinstance(xs_s, torch.Tensor): xs_s = xs_s.numpy()
    if isinstance(xs_u, torch.Tensor): xs_u = xs_u.numpy()

    print(f"  Sampling {N_CELLS} cells from last {N_SAMPLING_STEPS} time steps...")
    xs_s, xs_u, z, pseudotime = _sample_cells(xs_s, xs_u, traj, SEED)

    return dict(
        cache_key   = _sim_cache_key(),
        xs_s_base   = xs_s.astype(np.float32),
        xs_u_base   = xs_u.astype(np.float32),
        z           = z.astype(np.float32),
        pseudotime  = pseudotime.astype(np.float32),
        xr_s        = xr_s.astype(np.float32),
        xr_u        = xr_u.astype(np.float32),
        config_dict = config_dict,
        system      = system,           # picklable via torch.save
    )


def load_or_build_kinetics_cache(
    cache_path: Path = KINETICS_CACHE_PATH,
    h5ad_path:  Optional[Path] = None,
    force_rebuild: bool = False,
) -> dict:
    """Load Stage-1 cache from disk, rebuilding if missing or stale."""
    cache_path = Path(cache_path)
    key = _sim_cache_key()

    if not force_rebuild and cache_path.exists():
        try:
            saved = torch.load(cache_path, weights_only=False)
            if saved.get("cache_key") == key:
                print(f"  [Stage 1] Loaded kinetics cache from {cache_path.name}")
                return saved
            else:
                print("  [Stage 1] Cache key mismatch – rebuilding…")
        except Exception as e:
            print(f"  [Stage 1] Cache load failed ({e}) – rebuilding…")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result = _build_kinetics_cache(h5ad_path)
    torch.save(result, cache_path)
    print(f"  [Stage 1] Kinetics cache saved → {cache_path}")
    return result


# ---------------------------------------------------------------------------
# Stage-2 public entry point: observation-noise layer
# ---------------------------------------------------------------------------
def apply_observation_noise(
    cache: dict,
    pp_cfg: PostProcessConfig = DEFAULT_PP_CONFIG,
) -> Figure4Result:
    """Stage 2: apply the fast observation-noise model to a Stage-1 kinetics cache.

    This is the *tunable* layer (lib_size scaling, NB/Poisson sampling, dropout,
    binomial downsampling).  It runs in <10 s and can be called repeatedly with
    different ``pp_cfg`` values without touching the slow Stage-1 kinetics.

    Parameters
    ----------
    cache:
        Dict returned by :func:`load_or_build_kinetics_cache`.  Must contain
        ``xs_s_base``, ``xs_u_base``, ``z``, ``pseudotime``, ``xr_s``,
        ``xr_u``, ``system``, ``config_dict``.
    pp_cfg:
        Noise-model parameters.  Edit freely – Stage 1 is never re-run.
    """
    xs_s_base   = cache["xs_s_base"].astype(np.float64)
    xs_u_base   = cache["xs_u_base"].astype(np.float64)
    z           = cache["z"]
    pseudotime  = cache["pseudotime"]
    xr_s        = cache["xr_s"].astype(np.float64)
    xr_u        = cache["xr_u"].astype(np.float64)
    system      = cache["system"]
    config_dict = cache["config_dict"]

    print(f"  [Stage 2] Applying noise model (lib_size_mean={pp_cfg.lib_size_mean:.0f}, "
          f"dropout_prob={pp_cfg.dropout_prob:.2f}, cell_size_σ={pp_cfg.cell_size_sigma:.2f})…")
    xs_s, xs_u = _apply_count_noise(xs_s_base, xs_u_base, pp_cfg)

    print(f"  [Stage 2] Adding noise genes → {TARGET_N_GENES} total genes…")
    xs_s, xs_u = _add_noise_genes(xs_s, xs_u, pseudotime, pp_cfg)

    print(f"  Done: {xs_s.shape[0]} cells × {xs_s.shape[1]} genes")
    return Figure4Result(
        system=system, z=z.astype(np.float64),
        spliced=xs_s, unspliced=xs_u,
        real_spliced=xr_s, real_unspliced=xr_u,
        pseudotime=pseudotime.astype(np.float64),
        config_dict=config_dict,
    )


# ---------------------------------------------------------------------------
# Convenience entry point (Stage 1 + Stage 2 combined)
# ---------------------------------------------------------------------------
def run_figure4_pipeline(
    h5ad_path:     Optional[Path] = None,
    pp_cfg:        PostProcessConfig = DEFAULT_PP_CONFIG,
    cache_path:    Path = KINETICS_CACHE_PATH,
    force_rebuild: bool = False,
) -> Figure4Result:
    """Load/build Stage-1 kinetics cache, then apply Stage-2 observation noise."""
    cache = load_or_build_kinetics_cache(cache_path, h5ad_path, force_rebuild)
    return apply_observation_noise(cache, pp_cfg)


# ---------------------------------------------------------------------------
# Helpers still used by other scripts
# ---------------------------------------------------------------------------
def generate_poisson_baseline(
    system: UniversalGraphSystem, config_dict: dict,
    z: np.ndarray, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Poisson baseline on final latent positions (for supplementary panels).

    Uses the same continuous-kinetics approach as Stage 1 (simulate_counts=False),
    then applies a clean Poisson noise layer at lib_size=50 000.
    """
    synth = GeneSynthesizer(
        system=system, n_genes_total=len(config_dict["genes"]),
        edge_strength=5.0, cycle_strength=3.0, background_noise=0.1,
        splicing_rate=1.0, degradation_rate=0.5, seed=seed,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(config_dict, tmp, default_flow_style=False, sort_keys=False)
        synth.load_from_config(tmp.name, verbose=False)
    with contextlib.redirect_stdout(io.StringIO()):
        res = synth.generate_kinetics(z, dt=DT, simulate_counts=False, scale=ALPHA_SCALE)
    s, u = res["spliced"], res["unspliced"]
    if isinstance(s, torch.Tensor): s, u = s.numpy(), u.numpy()
    pp = PostProcessConfig(lib_size_mean=50000.0, dropout_prob=0.0,
                           sampling="poisson", seed=seed)
    s, u = _apply_count_noise(s.astype(np.float64), u.astype(np.float64), pp)
    return s, u
