"""Core constants / config for fig_diag_projection_masking.

This package is panelized plotting on top of legacy cached CSVs:
- `data/gene_sweep_summary.csv`
- `data/cell_diagnostics.csv`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Defaults (kept aligned with the legacy script's defaults)
# ---------------------------------------------------------------------------

OUT_DIR = Path("results/projection_masking_diagnosis")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

SEED = 1
TAU_LOW = 0.7
TAU_HIGH = 0.2

REGION_PRIORITY = ["junction", "cycle_rim", "branch1", "branch2", "edge"]
REGION_COLORS = {
    "cycle_rim": "#4477AA",
    "junction": "#EE6677",
    "branch1": "#228833",
    "branch2": "#CCBB44",
    "edge": "#AA3377",
    "other": "#CCCCCC",
}

PCA_BINNED_N_BINS = 10
EMBEDDING_BG_SUBSAMPLE_CAP = 900

GENE_SWEEP_CSV = "gene_sweep_summary.csv"
CELL_DIAGNOSTICS_CSV = "cell_diagnostics.csv"


def configure(
    *,
    out_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    fig_dir: str | Path | None = None,
    shared: Optional[Mapping[str, Any]] = None,
    pipeline: Optional[Mapping[str, Any]] = None,
) -> None:
    """Configure globals from JSON configs."""

    global OUT_DIR, DATA_DIR, FIG_DIR
    global SEED, TAU_LOW, TAU_HIGH
    global REGION_PRIORITY, REGION_COLORS
    global PCA_BINNED_N_BINS, EMBEDDING_BG_SUBSAMPLE_CAP

    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    elif out_dir is not None:
        DATA_DIR = Path(out_dir) / "data"

    if fig_dir is not None:
        FIG_DIR = Path(fig_dir)
    elif out_dir is not None:
        FIG_DIR = Path(out_dir) / "figures"

    if not shared:
        return

    SEED = int(shared.get("seed", SEED))
    TAU_LOW = float(shared.get("tau_low", TAU_LOW))
    TAU_HIGH = float(shared.get("tau_high", TAU_HIGH))

    REGION_PRIORITY = list(shared.get("region_priority", REGION_PRIORITY))
    REGION_COLORS = dict(shared.get("region_colors", REGION_COLORS))

    PCA_BINNED_N_BINS = int(shared.get("pca_binned_n_bins", PCA_BINNED_N_BINS))
    EMBEDDING_BG_SUBSAMPLE_CAP = int(
        shared.get("embedding_bg_subsample_cap", EMBEDDING_BG_SUBSAMPLE_CAP)
    )

