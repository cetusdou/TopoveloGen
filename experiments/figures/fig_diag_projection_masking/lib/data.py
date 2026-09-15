"""Cached-data loader for fig_diag_projection_masking."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from experiments.figures._common.config import load_package_configs, result_dirs
from experiments.figures.fig_diag_projection_masking.lib import core

PKG_DIR = Path(__file__).resolve().parents[1]


def load_cfg(panel_config: str | None = None, compose_config: str | None = None) -> dict:
    cfg = load_package_configs(PKG_DIR, panel_config=panel_config, compose_config=compose_config)
    out_dir, data_dir, fig_dir = result_dirs(cfg)
    core.configure(
        out_dir=out_dir,
        data_dir=data_dir,
        fig_dir=fig_dir,
        shared=cfg,
        pipeline=cfg.get("pipeline"),
    )
    return cfg


def _to_float(x: str) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    return float(s)


def _cell_region_labels(regions: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    labels = np.full(n, "other", dtype=object)
    for name in reversed(core.REGION_PRIORITY):
        if name in regions:
            labels[np.asarray(regions[name], dtype=bool)] = name
    return labels


def load_main_and_sweep() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sweep_path = core.DATA_DIR / core.GENE_SWEEP_CSV
    cell_path = core.DATA_DIR / core.CELL_DIAGNOSTICS_CSV
    for p in (sweep_path, cell_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing cached result: {p}\n"
                "Run: python experiments/figures/fig_diag_projection_masking/pipeline/run_pipeline.py"
            )

    sweep_rows: List[Dict[str, Any]] = []
    with sweep_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sweep_rows.append(
                {
                    "n_genes": int(r["n_genes"]),
                    "mismatch_frac": float(r["mismatch_frac"]),
                    "mean_cos_high": _to_float(r.get("mean_cos_high", "nan")),
                    "mean_cos_low": _to_float(r.get("mean_cos_low", "nan")),
                    "mean_cos_low_gt": _to_float(r.get("mean_cos_low_gt", "nan")),
                    "mean_cos_dynamic": _to_float(r.get("mean_cos_dynamic", "nan")),
                    "mean_cos_noise": _to_float(r.get("mean_cos_noise", "nan")),
                    "mean_pca_frac": _to_float(r.get("mean_pca_frac", "nan")),
                    "frac_noise_genes": _to_float(r.get("frac_noise_genes", "nan")),
                }
            )

    rows: List[Dict[str, Any]] = []
    with cell_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    n = len(rows)
    if n == 0:
        raise ValueError(f"Empty cached CSV: {cell_path}")

    n_genes = int(rows[0]["n_genes"])
    z = np.array([[float(r["z1"]), float(r["z2"])] for r in rows], dtype=float)

    main: Dict[str, Any] = {
        "n_genes": n_genes,
        "z": z,
        "cos_high": np.array([_to_float(r["cos_high"]) for r in rows], dtype=float),
        "cos_low": np.array([_to_float(r["cos_low"]) for r in rows], dtype=float),
        "cos_gene": np.array([_to_float(r.get("cos_gene", r["cos_high"])) for r in rows], dtype=float),
        "cos_umap": np.array([_to_float(r.get("cos_umap", r["cos_low"])) for r in rows], dtype=float),
        "delta_proj": np.array([_to_float(r.get("delta_proj", "")) for r in rows], dtype=float),
        "cos_low_gt": np.array([_to_float(r["cos_low_gt"]) for r in rows], dtype=float),
        "pca_frac": np.array([_to_float(r["pca_frac"]) for r in rows], dtype=float),
        "cos_dynamic": np.array([_to_float(r.get("cos_dynamic", "")) for r in rows], dtype=float),
        "cos_noise": np.array([_to_float(r.get("cos_noise", "")) for r in rows], dtype=float),
        "mismatch": np.array([bool(int(r["mismatch"])) for r in rows], dtype=bool),
        "n_obs": n,
    }

    # Region labels for panels D/E/C.
    from experiments.results_embedding_masked_errors import assign_regions  # noqa: E402
    from experiments.results_statistical_realism import build_cycle_bifurcation_system  # noqa: E402

    system = build_cycle_bifurcation_system()
    regions = assign_regions(system, z)
    main["system"] = system
    main["regions"] = regions
    main["region_labels"] = _cell_region_labels(regions, n)

    return main, sweep_rows

