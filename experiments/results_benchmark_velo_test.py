"""
Deprecated entrypoint. Use the panelized package:
  experiments/figures/fig_bench_velo_test/
See experiments/figures/README.md for run instructions.

Benchmark velocity methods on cycle-linear synthetic data (benchmark_velo_test).

Loads precomputed AnnData outputs for DeepVelo, UniTVelo, and scVelo, compares
gene-space and low-dimensional velocity against ground truth, and saves individual
panels plus a combined figure under results/benchmark_velo_test/figures/.

Vector exports (PDF/SVG) keep text, axes, histograms, and streamlines editable;
scatter/point-cloud layers are rasterized. PNG is also written at 300 dpi.

Usage (from repo root):
    NUMBA_CACHE_DIR=/tmp/numba_cache python experiments/results_benchmark_velo_test.py
    python experiments/results_benchmark_velo_test.py --plot-only
"""


import argparse
import csv
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figure_style import (  # noqa: E402
    FIGURE_SINGLE_SIZE,
    apply_publication_style,
)
from experiments.projection_masking_metrics import (  # noqa: E402
    PROJECTION_MASKING_CAPTION,
    delta_projection_masking,
    draw_delta_proj_facets,
    save_projection_masking_per_cell_csv,
    save_projection_masking_summary_csv,
    summarize_delta_proj,
)

os.environ.setdefault("NUMBA_CACHE_DIR", str(REPO_ROOT / ".numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

BENCH_ROOT = REPO_ROOT / "benchmark_velo_test"
BENCH_RESULTS = BENCH_ROOT / "results"
H5AD_FULL = BENCH_ROOT / "cycle_linear_2000genes_n5000.h5ad"

OUT_DIR = REPO_ROOT / "results" / "benchmark_velo_test"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

METHOD_DIRS = {
    "deepvelo": BENCH_RESULTS / "deepvelo",
    "unitvelo": BENCH_RESULTS / "unitvelo",
    "scvelo": BENCH_RESULTS / "scvelo",
}

METHOD_VKEYS = {
    "deepvelo": ("deepvelo_velo", "velocity"),
    "unitvelo": ("velocity",),
    "scvelo": ("velocity",),
}

COLOR_KEY = "true_time"
METHOD_COLORS = {
    "deepvelo": "#0072B2",
    "unitvelo": "#D55E00",
    "scvelo": "#009E73",
}
METHOD_ORDER = ("deepvelo", "unitvelo", "scvelo")
STREAM_KW = dict(
    legend_loc=None,
    density=1.2,
    linewidth=0.9,
    arrowsize=0.9,
    size=6,
    alpha=0.5,
    colorbar=False,
    show=False,
)

_EDITABLE_TEXT_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


# ---------------------------------------------------------------------------
# I/O helpers (from benchmark_velo_test/compare_results.ipynb)
# ---------------------------------------------------------------------------
def pick_h5ad(folder: Path, prefer_vis: bool = True) -> Path:
    h5ads = sorted(folder.glob("*.h5ad"))
    if not h5ads:
        raise FileNotFoundError(f"No .h5ad found in {folder}")
    if prefer_vis:
        vis = [p for p in h5ads if "_vis" in p.stem]
        if vis:
            return vis[0]
    return h5ads[0]


def safe_read_h5ad(path: Path | str) -> ad.AnnData:
    path = Path(path)
    try:
        return ad.read_h5ad(path)
    except Exception as e:
        msg = str(e)
        if "encoding_type" not in msg and "encoding-type" not in msg:
            raise
        tmp_dir = Path(tempfile.mkdtemp(prefix="h5ad_sanitized_"))
        tmp_path = tmp_dir / path.name
        shutil.copy2(path, tmp_path)
        with h5py.File(tmp_path, "r+") as f:
            if "uns" in f and "log1p" in f["uns"] and "base" in f["uns/log1p"]:
                del f["uns/log1p/base"]
                if len(f["uns/log1p"].keys()) == 0:
                    del f["uns/log1p"]
        return ad.read_h5ad(tmp_path)


def infer_velocity_layer(adata: ad.AnnData, preferred: Iterable[str]) -> str:
    for k in preferred:
        if k in adata.layers:
            return k
    for k in ("velocity", "velocity_s", "velocity_u", "velo", "V"):
        if k in adata.layers:
            return k
    raise KeyError(f"No velocity layer in {list(adata.layers.keys())}")


def infer_gt_velocity_layer(adata: ad.AnnData) -> str:
    for k in ("gt_velo", "gt_velocity", "velocity_gt", "V_gt"):
        if k in adata.layers:
            return k
    raise KeyError(f"No ground-truth velocity in {list(adata.layers.keys())}")


def align_adatas(*adatas: ad.AnnData) -> List[ad.AnnData]:
    common_cells = adatas[0].obs_names
    common_genes = adatas[0].var_names
    for a in adatas[1:]:
        common_cells = common_cells.intersection(a.obs_names)
        common_genes = common_genes.intersection(a.var_names)
    if len(common_cells) == 0 or len(common_genes) == 0:
        raise ValueError("No overlapping cells/genes across AnnData objects")
    out = []
    for a in adatas:
        a2 = a[common_cells, common_genes].copy()
        out.append(a2)
    ref = out[0]
    return [a[ref.obs_names, ref.var_names].copy() for a in out]


def _rowwise_cosine_dense(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dot = np.einsum("ij,ij->i", x, y)
    nx = np.sqrt(np.einsum("ij,ij->i", x, x))
    ny = np.sqrt(np.einsum("ij,ij->i", y, y))
    return dot / (nx * ny + eps)


def _rowwise_cosine_sparse(x: sparse.spmatrix, y: sparse.spmatrix, eps: float = 1e-12) -> np.ndarray:
    dot = np.asarray(x.multiply(y).sum(axis=1)).ravel()
    nx = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    ny = np.sqrt(np.asarray(y.multiply(y).sum(axis=1)).ravel())
    return dot / (nx * ny + eps)


def rowwise_cosine(x, y) -> np.ndarray:
    if sparse.issparse(x) or sparse.issparse(y):
        x = x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(x)
        y = y.tocsr() if sparse.issparse(y) else sparse.csr_matrix(y)
        return _rowwise_cosine_sparse(x, y)
    return _rowwise_cosine_dense(np.asarray(x), np.asarray(y))


def rowwise_mse(x, y) -> np.ndarray:
    if sparse.issparse(x) or sparse.issparse(y):
        x = x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(x)
        y = y.tocsr() if sparse.issparse(y) else sparse.csr_matrix(y)
        diff = x - y
        return np.asarray(diff.multiply(diff).mean(axis=1)).ravel()
    diff = np.asarray(x) - np.asarray(y)
    return np.mean(diff * diff, axis=1)


def pick_basis(adata: ad.AnnData) -> str:
    if "X_umap" in adata.obsm:
        return "umap"
    if "X_pca" in adata.obsm:
        return "pca"
    raise KeyError(f"No UMAP/PCA in obsm: {list(adata.obsm.keys())}")


def ensure_velocity_embedding(adata: ad.AnnData, basis: str, vkey: str) -> str:
    obsm_key = f"{vkey}_{basis}"
    if obsm_key in adata.obsm:
        return obsm_key
    import scvelo as scv

    scv.settings.verbosity = 0
    scv.tl.velocity_embedding(adata, basis=basis, vkey=vkey)
    if obsm_key not in adata.obsm:
        raise KeyError(f"Missing {obsm_key} after scvelo embedding")
    return obsm_key


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_method_adatas() -> Dict[str, ad.AnnData]:
    out: Dict[str, ad.AnnData] = {}
    for name, folder in METHOD_DIRS.items():
        if not folder.exists():
            print(f"[skip] missing folder for {name}: {folder}")
            continue
        path = pick_h5ad(folder)
        out[name] = safe_read_h5ad(path)
        print(f"Loaded {name}: {path.name} {out[name].shape}")
    if not out:
        raise FileNotFoundError(f"No method results under {BENCH_RESULTS}")
    return out


def method_vkey(name: str, adata: ad.AnnData) -> str:
    return infer_velocity_layer(adata, METHOD_VKEYS.get(name, ("velocity",)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_summary(
    adatas: Dict[str, ad.AnnData],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], str]:
    names = list(adatas.keys())
    aligned = align_adatas(*[adatas[n] for n in names])
    aligned_map = {n: a for n, a in zip(names, aligned)}

    ref = aligned[0]
    vkey_gt = infer_gt_velocity_layer(ref)
    V_gt = ref.layers[vkey_gt]
    basis = pick_basis(ref)

    cos_gene: Dict[str, np.ndarray] = {}
    mse_gene: Dict[str, np.ndarray] = {}
    cos_lowdim: Dict[str, np.ndarray] = {}
    rows = []

    for name in names:
        adata = aligned_map[name]
        vkey = method_vkey(name, adata)
        V = adata.layers[vkey]
        cos_gene[name] = rowwise_cosine(V, V_gt)
        mse_gene[name] = rowwise_mse(V, V_gt)
        key_ld = ensure_velocity_embedding(adata, basis=basis, vkey=vkey)
        Vld = np.asarray(adata.obsm[key_ld])
        key_gt_ld = ensure_velocity_embedding(ref, basis=basis, vkey=vkey_gt)
        Vld_gt = np.asarray(ref.obsm[key_gt_ld])
        cos_lowdim[name] = rowwise_cosine(Vld, Vld_gt)
        rows.append(
            {
                "method": name,
                "velocity_layer": vkey,
                "cosine_gene_velocity_vs_gt": float(np.nanmean(cos_gene[name])),
                "mse_gene_velocity_vs_gt": float(np.nanmean(mse_gene[name])),
                "cosine_lowdim_velocity_vs_gt": float(np.nanmean(cos_lowdim[name])),
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
                "basis": basis,
            }
        )

    return pd.DataFrame(rows), cos_gene, mse_gene, cos_lowdim, basis


def compute_delta_proj(
    cos_gene: Dict[str, np.ndarray],
    cos_lowdim: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Per-cell projection masking index Δ_proj = Cos_UMAP − Cos_gene."""
    return {
        name: delta_projection_masking(cos_lowdim[name], cos_gene[name])
        for name in cos_gene
    }


def augment_summary_with_projection_masking(
    summary: pd.DataFrame,
    cos_gene: Dict[str, np.ndarray],
    cos_lowdim: Dict[str, np.ndarray],
) -> pd.DataFrame:
    out = summary.copy()
    delta = compute_delta_proj(cos_gene, cos_lowdim)
    extra_rows = []
    for _, row in out.iterrows():
        name = str(row["method"])
        stats = summarize_delta_proj(delta[name])
        extra_rows.append(stats)
    extra_df = pd.DataFrame(extra_rows)
    return pd.concat([out.reset_index(drop=True), extra_df], axis=1)


def compute_projection_distortion() -> pd.DataFrame:
    if not H5AD_FULL.exists():
        raise FileNotFoundError(f"Missing full count matrix: {H5AD_FULL}")
    adata_full = safe_read_h5ad(H5AD_FULL)
    X = np.asarray(adata_full.X, dtype=np.float32)
    n_cells, d0 = X.shape
    target_dims = list(range(10, 500, 10))

    rng = np.random.default_rng(0)
    n_pairs = 30000
    idx_i = rng.integers(0, n_cells, size=n_pairs)
    idx_j = rng.integers(0, n_cells, size=n_pairs)
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    D_orig = np.linalg.norm(X[idx_i] - X[idx_j], axis=1)
    D_orig = np.maximum(D_orig, 1e-8)

    n_repeat = 5
    rows = []
    for k in target_dims:
        eps_all = []
        for _ in range(n_repeat):
            G = rng.normal(0.0, 1.0, size=(d0, k)).astype(np.float32) / np.sqrt(k)
            Y = X @ G
            D_proj = np.linalg.norm(Y[idx_i] - Y[idx_j], axis=1)
            eps = np.abs(D_proj - D_orig) / D_orig
            eps_all.append(float(np.mean(eps)))
        rows.append(
            {
                "target_dim": k,
                "epsilon_mean": float(np.mean(eps_all)),
                "epsilon_std": float(np.std(eps_all)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _color_key(adata: ad.AnnData) -> Optional[str]:
    if COLOR_KEY in adata.obs.columns:
        return COLOR_KEY
    return None


def _strip_colorbars(fig: plt.Figure) -> None:
    for ax in list(fig.axes):
        if ax.get_label() == "<colorbar>":
            ax.remove()


def _finalize_umap_ax(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def draw_umap_scatter(adata: ad.AnnData, ax: plt.Axes, title: str = "Raw UMAP") -> None:
    import scvelo as scv

    scv.settings.verbosity = 0
    ckey = _color_key(adata)
    if ckey is not None:
        try:
            scv.pl.scatter(
                adata,
                basis="umap",
                color=ckey,
                color_map="viridis",
                colorbar=False,
                ax=ax,
                show=False,
            )
        except Exception:
            X = np.asarray(adata.obsm["X_umap"])
            ax.scatter(
                X[:, 0],
                X[:, 1],
                s=4,
                c=np.asarray(adata.obs[ckey]),
                cmap="viridis",
                alpha=0.85,
                linewidths=0,
                rasterized=True,
            )
    else:
        X = np.asarray(adata.obsm["X_umap"])
        ax.scatter(X[:, 0], X[:, 1], s=4, c="#BFBFBF", alpha=0.85, linewidths=0, rasterized=True)
    _finalize_umap_ax(ax, title)


def draw_velocity_stream(adata: ad.AnnData, vkey: str, ax: plt.Axes, title: str) -> None:
    import scvelo as scv

    scv.settings.verbosity = 0
    ckey = _color_key(adata)
    scv.pl.velocity_embedding_stream(
        adata,
        basis="umap",
        vkey=vkey,
        color=ckey,
        ax=ax,
        **STREAM_KW,
    )
    _finalize_umap_ax(ax, title)


def draw_velocity_grid(adata: ad.AnnData, vkey: str, ax: plt.Axes, title: str) -> bool:
    import scvelo as scv

    scv.settings.verbosity = 0
    graph_key = f"{vkey}_graph"
    umap_key = f"{vkey}_umap"
    if graph_key not in adata.uns or umap_key not in adata.obsm:
        ax.set_visible(False)
        return False
    ckey = _color_key(adata)
    scv.pl.velocity_embedding_grid(
        adata,
        basis="umap",
        vkey=vkey,
        color=ckey,
        colorbar=False,
        ax=ax,
        show=False,
    )
    _finalize_umap_ax(ax, title)
    return True


def draw_gene_cosine_histogram(cos_gene: Dict[str, np.ndarray], ax: plt.Axes) -> None:
    for name, vals in cos_gene.items():
        ax.hist(
            vals[np.isfinite(vals)],
            bins=60,
            color=METHOD_COLORS.get(name, "#888888"),
            alpha=0.7,
            label=f"{name} vs GT",
        )
    ax.set_title("Gene-space cosine vs GT")
    ax.set_xlabel("cosine")
    ax.set_ylabel("# cells")
    ax.legend(frameon=False)


def draw_gene_mse_histogram(mse_gene: Dict[str, np.ndarray], ax: plt.Axes) -> None:
    for name, vals in mse_gene.items():
        plot_vals = vals[np.isfinite(vals)]
        if name == "unitvelo":
            plot_vals = plot_vals / 10.0
            label = f"{name} vs GT (/10)"
        else:
            label = f"{name} vs GT"
        ax.hist(plot_vals, bins=60, color=METHOD_COLORS.get(name, "#888888"), alpha=0.7, label=label)
    ax.set_title("Gene-space MSE vs GT")
    ax.set_xlabel("MSE")
    ax.legend(frameon=False)


def draw_lowdim_cosine_histogram(cos_lowdim: Dict[str, np.ndarray], basis: str, ax: plt.Axes) -> None:
    for name, vals in cos_lowdim.items():
        ax.hist(
            vals[np.isfinite(vals)],
            bins=60,
            color=METHOD_COLORS.get(name, "#888888"),
            alpha=0.7,
            label=f"{name} vs GT",
        )
    ax.set_title(f"Low-dim cosine vs GT ({basis})")
    ax.set_xlabel("cosine")
    ax.set_ylabel("# cells")
    ax.legend(frameon=False)


def _rasterize_point_clouds(ax: plt.Axes) -> None:
    """Rasterize scatter/point-cloud layers only; keep text and vectors editable."""
    from matplotlib.collections import PathCollection

    for artist in ax.get_children():
        if isinstance(artist, PathCollection):
            artist.set_rasterized(True)


def _rasterize_stream_decorations(ax: plt.Axes) -> None:
    """Rasterize scVelo stream decorations when PDF export rejects them."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch

    for artist in ax.get_children():
        if isinstance(artist, (LineCollection, FancyArrowPatch)):
            artist.set_rasterized(True)


def _prepare_figure_for_export(fig: plt.Figure) -> None:
    _strip_colorbars(fig)
    for ax in fig.axes:
        if ax.get_label() != "<colorbar>":
            _rasterize_point_clouds(ax)


def save_export_figure(fig: plt.Figure, out_stem: Path | str, dpi: int = 300) -> None:
    """Save PNG/SVG/PDF with editable text; rasterize point clouds only."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    _prepare_figure_for_export(fig)
    with plt.rc_context(_EDITABLE_TEXT_RC):
        fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
        fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
        try:
            fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
        except ValueError:
            for ax in fig.axes:
                if ax.get_label() != "<colorbar>":
                    _rasterize_stream_decorations(ax)
            fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _single_panel_figure() -> plt.Figure:
    panel_w, panel_h = FIGURE_SINGLE_SIZE
    fig, ax = plt.subplots(figsize=(panel_w, panel_h), constrained_layout=True)
    return fig, ax


def plot_umap_panel(adata: ad.AnnData, out_stem: str) -> None:
    apply_publication_style()
    fig, ax = _single_panel_figure()
    draw_umap_scatter(adata, ax, "Raw UMAP")
    save_export_figure(fig, FIG_DIR / out_stem)


def plot_velocity_stream_panel(adata: ad.AnnData, vkey: str, title: str, out_stem: str) -> None:
    apply_publication_style()
    fig, ax = _single_panel_figure()
    draw_velocity_stream(adata, vkey, ax, title)
    save_export_figure(fig, FIG_DIR / out_stem)


def plot_velocity_grid_panel(adata: ad.AnnData, vkey: str, title: str, out_stem: str) -> None:
    apply_publication_style()
    fig, ax = _single_panel_figure()
    if not draw_velocity_grid(adata, vkey, ax, title):
        plt.close(fig)
        return
    save_export_figure(fig, FIG_DIR / out_stem)


def plot_gene_cosine_histogram_panel(cos_gene: Dict[str, np.ndarray]) -> None:
    apply_publication_style()
    fig, ax = _single_panel_figure()
    draw_gene_cosine_histogram(cos_gene, ax)
    save_export_figure(fig, FIG_DIR / "panel_gene_cosine_histogram")


def plot_gene_mse_histogram_panel(mse_gene: Dict[str, np.ndarray]) -> None:
    apply_publication_style()
    fig, ax = _single_panel_figure()
    draw_gene_mse_histogram(mse_gene, ax)
    save_export_figure(fig, FIG_DIR / "panel_gene_mse_histogram")


def plot_lowdim_cosine_histogram_panel(cos_lowdim: Dict[str, np.ndarray], basis: str) -> None:
    apply_publication_style()
    fig, ax = _single_panel_figure()
    draw_lowdim_cosine_histogram(cos_lowdim, basis, ax)
    save_export_figure(fig, FIG_DIR / "panel_lowdim_cosine_histogram")


def plot_projection_masking_index_panel(
    cos_gene: Dict[str, np.ndarray],
    cos_lowdim: Dict[str, np.ndarray],
) -> None:
    """Δ_proj distributions for DeepVelo, UniTVelo, and scVelo (separate facets)."""
    apply_publication_style()
    delta = compute_delta_proj(cos_gene, cos_lowdim)
    methods = [m for m in METHOD_ORDER if m in delta]
    if not methods:
        return

    panel_w, panel_h = FIGURE_SINGLE_SIZE
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(panel_w * len(methods), panel_h * 0.95),
        constrained_layout=True,
        sharey=True,
    )
    if len(methods) == 1:
        axes = [axes]

    draw_delta_proj_facets(
        delta,
        axes,
        method_order=methods,
        method_colors=METHOD_COLORS,
    )
    for ax in axes:
        if ax is axes[0]:
            ax.set_ylabel(r"# cells")
        else:
            ax.set_ylabel("")

    fig.suptitle(
        r"Projection masking index  $\Delta_{\mathrm{proj}} = "
        r"\mathrm{Cos}_{\mathrm{UMAP}} - \mathrm{Cos}_{\mathrm{gene}}$",
        fontsize=9,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        PROJECTION_MASKING_CAPTION,
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        transform=fig.transFigure,
    )
    save_export_figure(fig, FIG_DIR / "figure_projection_masking_index")


def plot_projection_distortion_panel(df_eps: pd.DataFrame) -> None:
    apply_publication_style()
    fig, ax = _single_panel_figure()
    ax.errorbar(
        df_eps["target_dim"],
        df_eps["epsilon_mean"],
        yerr=df_eps["epsilon_std"],
        fmt="-o",
        color="#0072B2",
        ecolor="#999999",
        elinewidth=0.8,
        capsize=2.5,
        markersize=3.5,
    )
    ax.set_title("Gaussian random projection distortion")
    ax.set_xlabel("Projected dimension")
    ax.set_ylabel("Relative distance error")
    save_export_figure(fig, FIG_DIR / "panel_gaussian_projection_distortion")


def plot_individual_panels(
    adatas: Dict[str, ad.AnnData],
    cos_gene: Dict[str, np.ndarray],
    mse_gene: Dict[str, np.ndarray],
    cos_lowdim: Dict[str, np.ndarray],
    basis: str,
    df_eps: Optional[pd.DataFrame] = None,
) -> None:
    ref = adatas["deepvelo"] if "deepvelo" in adatas else adatas[next(iter(adatas))]
    vkey_gt = infer_gt_velocity_layer(ref)

    plot_umap_panel(ref, "panel_umap_true_time")
    plot_velocity_stream_panel(ref, vkey_gt, "GT velocity streamlines", "panel_gt_velocity_streamlines")

    for name in ("deepvelo", "unitvelo", "scvelo"):
        if name not in adatas:
            continue
        adata = adatas[name]
        vkey = method_vkey(name, adata)
        plot_velocity_stream_panel(
            adata,
            vkey,
            f"{name} velocity streamlines",
            f"panel_{name}_velocity_streamlines",
        )
        plot_velocity_grid_panel(
            adata,
            vkey,
            f"{name} velocity grid",
            f"panel_{name}_velocity_grid",
        )

    plot_gene_cosine_histogram_panel(cos_gene)
    plot_gene_mse_histogram_panel(mse_gene)
    plot_lowdim_cosine_histogram_panel(cos_lowdim, basis)
    if df_eps is not None and not df_eps.empty:
        plot_projection_distortion_panel(df_eps)


def plot_composite_figure(
    adatas: Dict[str, ad.AnnData],
    cos_gene: Dict[str, np.ndarray],
    mse_gene: Dict[str, np.ndarray],
    cos_lowdim: Dict[str, np.ndarray],
    basis: str,
) -> None:
    apply_publication_style()
    panel_w, panel_h = FIGURE_SINGLE_SIZE
    fig = plt.figure(figsize=(panel_w * 4, panel_h * 3), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.82])

    ref = adatas["deepvelo"] if "deepvelo" in adatas else adatas[next(iter(adatas))]
    vkey_gt = infer_gt_velocity_layer(ref)

    # Row 1: Raw UMAP + velocity grid plots
    row0 = gs[0, :].subgridspec(1, 5, wspace=0.06)
    draw_umap_scatter(ref, fig.add_subplot(row0[0, 0]), "Raw UMAP")
    draw_velocity_grid(ref, vkey_gt, fig.add_subplot(row0[0, 1]), "GT grid")
    draw_velocity_grid(
        adatas["deepvelo"],
        method_vkey("deepvelo", adatas["deepvelo"]),
        fig.add_subplot(row0[0, 2]),
        "DeepVelo grid",
    )
    draw_velocity_grid(
        adatas["unitvelo"],
        method_vkey("unitvelo", adatas["unitvelo"]),
        fig.add_subplot(row0[0, 3]),
        "UniTVelo grid",
    )
    draw_velocity_grid(
        adatas["scvelo"],
        method_vkey("scvelo", adatas["scvelo"]),
        fig.add_subplot(row0[0, 4]),
        "scVelo grid",
    )

    # Row 2: velocity streamplots
    row1 = gs[1, :].subgridspec(1, 4, wspace=0.06)
    draw_velocity_stream(ref, vkey_gt, fig.add_subplot(row1[0, 0]), "GT velocity")
    draw_velocity_stream(
        adatas["deepvelo"],
        method_vkey("deepvelo", adatas["deepvelo"]),
        fig.add_subplot(row1[0, 1]),
        "DeepVelo",
    )
    draw_velocity_stream(
        adatas["unitvelo"],
        method_vkey("unitvelo", adatas["unitvelo"]),
        fig.add_subplot(row1[0, 2]),
        "UniTVelo",
    )
    draw_velocity_stream(
        adatas["scvelo"],
        method_vkey("scvelo", adatas["scvelo"]),
        fig.add_subplot(row1[0, 3]),
        "scVelo",
    )

    # Row 3: metric histograms (3 panels spanning full width)
    row2 = gs[2, :].subgridspec(1, 3, wspace=0.18)
    draw_gene_cosine_histogram(cos_gene, fig.add_subplot(row2[0, 0]))
    draw_gene_mse_histogram(mse_gene, fig.add_subplot(row2[0, 1]))
    draw_lowdim_cosine_histogram(cos_lowdim, basis, fig.add_subplot(row2[0, 2]))

    out_stem = FIG_DIR / "figure_benchmark_velo_test"
    save_export_figure(fig, out_stem)
    print(
        "Saved composite figure:",
        out_stem.with_suffix(".png"),
        out_stem.with_suffix(".pdf"),
        out_stem.with_suffix(".svg"),
    )


def save_summary_csv(summary: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "method_comparison_summary.csv"
    summary.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print("Saved", path)


def save_projection_csv(df_eps: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "projection_distortion.csv"
    df_eps.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    print("Saved", path)


def save_projection_masking_csv(
    cos_gene: Dict[str, np.ndarray],
    cos_lowdim: Dict[str, np.ndarray],
) -> None:
    per_cell = save_projection_masking_per_cell_csv(
        DATA_DIR / "projection_masking_per_cell.csv",
        cos_gene,
        cos_lowdim,
    )
    summary_pm = save_projection_masking_summary_csv(
        DATA_DIR / "projection_masking_summary.csv",
        cos_gene,
        cos_lowdim,
    )
    print("Saved", DATA_DIR / "projection_masking_per_cell.csv")
    print("Saved", DATA_DIR / "projection_masking_summary.csv")
    print(summary_pm.to_string(index=False))


def generate_figures(
    adatas: Dict[str, ad.AnnData],
    summary: pd.DataFrame,
    cos_gene: Dict[str, np.ndarray],
    mse_gene: Dict[str, np.ndarray],
    cos_lowdim: Dict[str, np.ndarray],
    basis: str,
    df_eps: Optional[pd.DataFrame] = None,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_individual_panels(adatas, cos_gene, mse_gene, cos_lowdim, basis, df_eps=df_eps)
    plot_composite_figure(adatas, cos_gene, mse_gene, cos_lowdim, basis)
    plot_projection_masking_index_panel(cos_gene, cos_lowdim)
    print("Figures written to", FIG_DIR)
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Replot from cached CSV (still loads h5ad for velocity panels)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DATA_DIR / "method_comparison_summary.csv"
    proj_path = DATA_DIR / "projection_distortion.csv"

    adatas = load_method_adatas()

    per_cell_pm_path = DATA_DIR / "projection_masking_per_cell.csv"

    if args.plot_only and summary_path.exists() and per_cell_pm_path.exists():
        summary = pd.read_csv(summary_path)
        basis = str(summary["basis"].iloc[0])
        df_eps = pd.read_csv(proj_path) if proj_path.exists() else None
        pm_df = pd.read_csv(per_cell_pm_path)
        cos_gene, mse_gene, cos_lowdim = {}, {}, {}
        for name in pm_df["method"].unique():
            sub = pm_df[pm_df["method"] == name]
            cos_gene[name] = sub["cos_gene"].to_numpy()
            cos_lowdim[name] = sub["cos_umap"].to_numpy()
        names = list(adatas.keys())
        aligned = align_adatas(*[adatas[n] for n in names])
        aligned_map = {n: a for n, a in zip(names, aligned)}
        ref = aligned[0]
        vkey_gt = infer_gt_velocity_layer(ref)
        V_gt = ref.layers[vkey_gt]
        for name in names:
            if name in cos_gene:
                adata = aligned_map[name]
                vkey = method_vkey(name, adata)
                mse_gene[name] = rowwise_mse(adata.layers[vkey], V_gt)
            else:
                adata = aligned_map[name]
                vkey = method_vkey(name, adata)
                cos_gene[name] = rowwise_cosine(adata.layers[vkey], V_gt)
                mse_gene[name] = rowwise_mse(adata.layers[vkey], V_gt)
                key_ld = ensure_velocity_embedding(adata, basis=basis, vkey=vkey)
                key_gt_ld = ensure_velocity_embedding(ref, basis=basis, vkey=vkey_gt)
                cos_lowdim[name] = rowwise_cosine(
                    np.asarray(adata.obsm[key_ld]), np.asarray(ref.obsm[key_gt_ld])
                )
    elif args.plot_only and summary_path.exists():
        summary = pd.read_csv(summary_path)
        basis = str(summary["basis"].iloc[0])
        df_eps = pd.read_csv(proj_path) if proj_path.exists() else None
        names = list(adatas.keys())
        aligned = align_adatas(*[adatas[n] for n in names])
        aligned_map = {n: a for n, a in zip(names, aligned)}
        ref = aligned[0]
        vkey_gt = infer_gt_velocity_layer(ref)
        V_gt = ref.layers[vkey_gt]
        cos_gene, mse_gene, cos_lowdim = {}, {}, {}
        for name in names:
            adata = aligned_map[name]
            vkey = method_vkey(name, adata)
            cos_gene[name] = rowwise_cosine(adata.layers[vkey], V_gt)
            mse_gene[name] = rowwise_mse(adata.layers[vkey], V_gt)
            key_ld = ensure_velocity_embedding(adata, basis=basis, vkey=vkey)
            key_gt_ld = ensure_velocity_embedding(ref, basis=basis, vkey=vkey_gt)
            cos_lowdim[name] = rowwise_cosine(
                np.asarray(adata.obsm[key_ld]), np.asarray(ref.obsm[key_gt_ld])
            )
        save_projection_masking_csv(cos_gene, cos_lowdim)
    else:
        summary, cos_gene, mse_gene, cos_lowdim, basis = compute_summary(adatas)
        summary = augment_summary_with_projection_masking(summary, cos_gene, cos_lowdim)
        save_summary_csv(summary)
        save_projection_masking_csv(cos_gene, cos_lowdim)
        df_eps = compute_projection_distortion()
        save_projection_csv(df_eps)

    generate_figures(adatas, summary, cos_gene, mse_gene, cos_lowdim, basis, df_eps=df_eps)


if __name__ == "__main__":
    main()
