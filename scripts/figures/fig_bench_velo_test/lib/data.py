"""Cached-data loader for fig_bench_velo_test."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from scripts.figures.fig_bench_velo_test.lib import core
from scripts.figures._common.config import load_package_configs, result_dirs

PKG_DIR = Path(__file__).resolve().parents[1]


def load_cfg(panel_config: str | None = None, compose_config: str | None = None) -> dict:
    cfg = load_package_configs(PKG_DIR, panel_config=panel_config, compose_config=compose_config)
    out_dir, data_dir, fig_dir = result_dirs(cfg)
    core.configure(out_dir=out_dir, data_dir=data_dir, fig_dir=fig_dir, shared=cfg, pipeline=cfg.get("pipeline"))
    return cfg


def _pick_h5ad(folder: Path, prefer_vis: bool = True) -> Path:
    h5ads = sorted(folder.glob("*.h5ad"))
    if not h5ads:
        raise FileNotFoundError(f"No .h5ad found in {folder}")
    if prefer_vis:
        vis = [p for p in h5ads if "_vis" in p.stem]
        if vis:
            return vis[0]
    return h5ads[0]


def _safe_read_h5ad(path: Path) -> Any:
    import anndata as ad

    try:
        return ad.read_h5ad(path)
    except Exception:
        # Some h5ads can have encoding_type issues; try the legacy sanitizer.
        import h5py
        import shutil
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp(prefix="h5ad_sanitized_"))
        tmp_path = tmp_dir / path.name
        shutil.copy2(path, tmp_path)
        with h5py.File(tmp_path, "r+") as f:
            if "uns" in f and "log1p" in f["uns"] and "base" in f["uns/log1p"]:
                del f["uns/log1p/base"]
            if len(f["uns/log1p"].keys()) == 0:
                del f["uns/log1p"]
        return ad.read_h5ad(tmp_path)


def _infer_velocity_layer(adata: Any, preferred: Tuple[str, ...]) -> str:
    for k in preferred:
        if k in adata.layers:
            return k
    for k in ("velocity", "velocity_s", "velocity_u", "velo", "V"):
        if k in adata.layers:
            return k
    raise KeyError(f"No velocity layer in {list(adata.layers.keys())}")


def _infer_gt_velocity_layer(adata: Any) -> str:
    for k in ("gt_velo", "gt_velocity", "velocity_gt", "V_gt"):
        if k in adata.layers:
            return k
    raise KeyError(f"No ground-truth velocity layer in {list(adata.layers.keys())}")


def _align_adatas(*adatas: Any) -> List[Any]:
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


def _rowwise_mse(x: Any, y: Any) -> np.ndarray:
    from scipy import sparse

    if sparse.issparse(x) or sparse.issparse(y):
        x = x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(x)
        y = y.tocsr() if sparse.issparse(y) else sparse.csr_matrix(y)
        diff = x - y
        return np.asarray(diff.multiply(diff).mean(axis=1)).ravel()

    x = np.asarray(x)
    y = np.asarray(y)
    diff = x - y
    return np.mean(diff * diff, axis=1)


def _load_basis_from_summary(path: Path) -> str:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return str(row.get("basis", "umap"))
    raise FileNotFoundError(f"Empty summary CSV: {path}")


def _load_projection_masking_per_cell(
    path: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """
    Returns (cos_gene_by_method, cos_umap_by_method, cell_idx_sorted).
    """
    cos_gene: Dict[str, List[Tuple[int, float]]] = {}
    cos_umap: Dict[str, List[Tuple[int, float]]] = {}
    methods_seen: set[str] = set()

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            method = str(r["method"])
            i = int(r["cell_idx"])
            cg = float(r["cos_gene"])
            cu = float(r["cos_umap"])
            methods_seen.add(method)
            cos_gene.setdefault(method, []).append((i, cg))
            cos_umap.setdefault(method, []).append((i, cu))

    cos_gene_out: Dict[str, np.ndarray] = {}
    cos_umap_out: Dict[str, np.ndarray] = {}

    # Use the first method to define the canonical ordering.
    some_method = next(iter(methods_seen))
    idx_sorted = np.array(sorted([i for i, _ in cos_gene[some_method]]), dtype=int)

    for m in methods_seen:
        rows_g = sorted(cos_gene[m], key=lambda t: t[0])
        rows_u = sorted(cos_umap[m], key=lambda t: t[0])
        # Defensive checks: keep consistent ordering by cell_idx.
        if np.array([i for i, _ in rows_g]).shape[0] != idx_sorted.shape[0]:
            # Fallback to the exact ordering we have.
            idx_sorted = np.array([i for i, _ in rows_g], dtype=int)
        cos_gene_out[m] = np.array([v for _, v in rows_g], dtype=float)
        cos_umap_out[m] = np.array([v for _, v in rows_u], dtype=float)

    return cos_gene_out, cos_umap_out, idx_sorted


_BUNDLE_CACHE: Optional[Dict[str, Any]] = None


def load_bundle() -> Dict[str, Any]:
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is not None:
        return _BUNDLE_CACHE

    # Lazy imports for heavier deps (anndata).
    import anndata as ad  # noqa: F401

    data_dir = core.DATA_DIR
    summary_path = data_dir / core.SUMMARY_CSV
    per_cell_path = data_dir / core.PROJECTION_MASKING_PER_CELL_CSV
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing cached summary CSV: {summary_path}")
    if not per_cell_path.exists():
        raise FileNotFoundError(f"Missing cached projection masking CSV: {per_cell_path}")

    basis = _load_basis_from_summary(summary_path)
    cos_gene, cos_umap, _cell_idx = _load_projection_masking_per_cell(per_cell_path)

    # Load method adatas for MSE + velocity plots.
    adatas: Dict[str, Any] = {}
    for method in core.METHOD_ORDER:
        folder = core.METHOD_DIRS.get(method)
        if not folder:
            continue
        if not folder.exists():
            continue
        h5ad_path = _pick_h5ad(folder, prefer_vis=core.PREFER_VIS_H5AD)
        adatas[method] = _safe_read_h5ad(h5ad_path)

    if not adatas:
        raise FileNotFoundError(f"No method results found under expected dirs: {core.METHOD_DIRS}")

    # Reference method: prefer deepvelo if present (matches legacy).
    ref_method = "deepvelo" if "deepvelo" in adatas else next(iter(adatas.keys()))
    method_names = list(core.METHOD_ORDER) if all(m in adatas for m in core.METHOD_ORDER) else list(adatas.keys())
    aligned = _align_adatas(*[adatas[m] for m in method_names])
    aligned_map = {m: a for m, a in zip(method_names, aligned)}

    ref_adata = aligned_map[ref_method]
    vkey_gt = _infer_gt_velocity_layer(ref_adata)
    V_gt = ref_adata.layers[vkey_gt]

    mse_gene: Dict[str, np.ndarray] = {}
    for m in method_names:
        vkey = core.METHOD_VKEYS[m][0] if m in core.METHOD_VKEYS else "velocity"
        vkey = _infer_velocity_layer(aligned_map[m], tuple(core.METHOD_VKEYS.get(m, (vkey,))))
        V = aligned_map[m].layers[vkey]
        mse_gene[m] = _rowwise_mse(V, V_gt)

    # Projection distortion (optional).
    df_eps: Optional[Dict[str, np.ndarray]] = None
    dist_path = data_dir / core.PROJECTION_DISTORTION_CSV
    if dist_path.exists():
        target_dim: List[float] = []
        eps_mean: List[float] = []
        eps_std: List[float] = []
        with dist_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                target_dim.append(float(r["target_dim"]))
                eps_mean.append(float(r["epsilon_mean"]))
                eps_std.append(float(r["epsilon_std"]))
        df_eps = {
            "target_dim": np.array(target_dim, dtype=float),
            "epsilon_mean": np.array(eps_mean, dtype=float),
            "epsilon_std": np.array(eps_std, dtype=float),
        }

    _BUNDLE_CACHE = {
        "basis": basis,
        "adatas": aligned_map,
        "ref_method": ref_method,
        "ref_adata": ref_adata,
        "vkey_gt": vkey_gt,
        "cos_gene": cos_gene,
        "cos_lowdim": cos_umap,
        "mse_gene": mse_gene,
        "df_eps": df_eps,
        "method_order": list(core.METHOD_ORDER),
        "method_colors": core.METHOD_COLORS,
    }
    return _BUNDLE_CACHE

