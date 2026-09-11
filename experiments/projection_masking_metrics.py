"""Projection masking metrics shared across benchmark and simulation diagnostics.

Delta_proj = Cos_proj - Cos_gene, using the same RNA-state velocity reference
before and after projection.

Positive Delta_proj indicates that projection-based evaluation is more
optimistic than direct gene-space evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Benchmark methods (UMAP cosine often > gene-space cosine)
PROJECTION_MASKING_CAPTION = (
    "Projection-based evaluation can be more optimistic than gene-space evaluation "
    "(Delta_proj = Cos_proj - Cos_gene)."
)

# Synthetic scVelo diagnosis (embedding ceiling + structured gene-space error)
EVALUATION_CEILING_CAPTION = (
    "Projection-based velocity evaluation has a ceiling, and gene-space errors "
    "concentrate in dynamic modules and specific topological regions."
)


def delta_projection_masking(cos_umap: np.ndarray, cos_gene: np.ndarray) -> np.ndarray:
    """Per-cell projection masking index: Cos_proj - Cos_gene."""
    return np.asarray(cos_umap, dtype=np.float64) - np.asarray(cos_gene, dtype=np.float64)


def summarize_delta_proj(delta: np.ndarray) -> Dict[str, float]:
    valid = delta[np.isfinite(delta)]
    if valid.size == 0:
        return {
            "mean_delta_proj": float("nan"),
            "median_delta_proj": float("nan"),
            "std_delta_proj": float("nan"),
            "frac_delta_proj_positive": float("nan"),
        }
    return {
        "mean_delta_proj": float(np.mean(valid)),
        "median_delta_proj": float(np.median(valid)),
        "std_delta_proj": float(np.std(valid)),
        "frac_delta_proj_positive": float(np.mean(valid > 0)),
    }


def per_cell_projection_masking_rows(
    method: str,
    cos_gene: np.ndarray,
    cos_umap: np.ndarray,
    *,
    extra: Optional[Mapping[str, np.ndarray]] = None,
) -> List[Dict]:
    delta = delta_projection_masking(cos_umap, cos_gene)
    rows: List[Dict] = []
    for i in range(len(delta)):
        row = {
            "cell_idx": i,
            "method": method,
            "cos_gene": float(cos_gene[i]),
            "cos_umap": float(cos_umap[i]),
            "delta_proj": float(delta[i]),
        }
        if extra:
            for k, arr in extra.items():
                row[k] = float(arr[i]) if np.isfinite(arr[i]) else ""
        rows.append(row)
    return rows


def save_projection_masking_per_cell_csv(
    path: Path,
    cos_gene: Mapping[str, np.ndarray],
    cos_umap: Mapping[str, np.ndarray],
    *,
    extra_by_method: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for method in cos_gene:
        extra = extra_by_method.get(method) if extra_by_method else None
        rows.extend(
            per_cell_projection_masking_rows(
                method, cos_gene[method], cos_umap[method], extra=extra
            )
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def save_projection_masking_summary_csv(
    path: Path,
    cos_gene: Mapping[str, np.ndarray],
    cos_umap: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for method in cos_gene:
        cg = cos_gene[method]
        cu = cos_umap[method]
        stats = summarize_delta_proj(delta_projection_masking(cu, cg))
        rows.append(
            {
                "method": method,
                "mean_cos_gene": float(np.nanmean(cg)),
                "mean_cos_umap": float(np.nanmean(cu)),
                **stats,
            }
        )
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def draw_delta_proj_facets(
    delta_by_method: Mapping[str, np.ndarray],
    ax_row: Iterable[plt.Axes],
    *,
    method_order: List[str],
    method_colors: Mapping[str, str],
    bins: int = 60,
    xlim: tuple = (-1.05, 1.05),
) -> None:
    for ax, name in zip(ax_row, method_order):
        vals = delta_by_method[name]
        vals = vals[np.isfinite(vals)]
        color = method_colors.get(name, "#888888")
        ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(0.0, color="k", ls="--", lw=0.9, alpha=0.55)
        mean_d = float(np.mean(vals)) if vals.size else float("nan")
        ax.axvline(mean_d, color=color, ls="-", lw=1.2, alpha=0.9)
        ax.set_xlim(*xlim)
        ax.set_title(name)
        ax.set_xlabel(r"$\Delta_{\mathrm{proj}}$")
        ax.text(
            0.98,
            0.95,
            rf"mean={mean_d:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )


def draw_delta_proj_scatter_gene_vs_umap(
    cos_gene: np.ndarray,
    cos_umap: np.ndarray,
    ax: plt.Axes,
    *,
    color: str = "#666666",
    alpha: float = 0.12,
    s: float = 2.5,
) -> None:
    ax.scatter(cos_gene, cos_umap, s=s, alpha=alpha, c=color, linewidths=0, rasterized=True)
    ax.plot([-1, 1], [-1, 1], "k--", lw=0.8, alpha=0.45)
    ax.set_xlabel(r"$\mathrm{Cos}_{\mathrm{gene}}$")
    ax.set_ylabel(r"$\mathrm{Cos}_{\mathrm{UMAP}}$")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
