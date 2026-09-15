"""Core config / constants for fig_bench_velo_test."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

OUT_DIR = Path("results/benchmark_velo_test")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

BENCH_ROOT = Path("benchmark_velo_test")

METHOD_DIRS: dict[str, Path] = {
    "deepvelo": BENCH_ROOT / "results" / "deepvelo",
    "unitvelo": BENCH_ROOT / "results" / "unitvelo",
    "scvelo": BENCH_ROOT / "results" / "scvelo",
}

METHOD_VKEYS: dict[str, tuple[str, ...]] = {
    "deepvelo": ("deepvelo_velo", "velocity"),
    "unitvelo": ("velocity",),
    "scvelo": ("velocity",),
}

METHOD_COLORS: dict[str, str] = {
    "deepvelo": "#0072B2",
    "unitvelo": "#D55E00",
    "scvelo": "#009E73",
}

METHOD_ORDER: tuple[str, ...] = ("deepvelo", "unitvelo", "scvelo")

COLOR_KEY = "true_time"
PREFER_VIS_H5AD = True

SUMMARY_CSV = "method_comparison_summary.csv"
PROJECTION_MASKING_PER_CELL_CSV = "projection_masking_per_cell.csv"
PROJECTION_MASKING_SUMMARY_CSV = "projection_masking_summary.csv"
PROJECTION_DISTORTION_CSV = "projection_distortion.csv"

HIST_BINS = 60
HIST_ALPHA = 0.7
UNITVELO_MSE_SCALE = 10.0

STREAM_KW: dict[str, Any] = {
    "legend_loc": None,
    "density": 1.2,
    "linewidth": 0.9,
    "arrowsize": 0.9,
    "size": 6,
    "alpha": 0.5,
    "colorbar": False,
    "show": False,
}

PROJECTION_BINS = 60
PROJECTION_XLIM = (-1.05, 1.05)


def configure(
    *,
    out_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    fig_dir: str | Path | None = None,
    shared: Optional[Mapping[str, Any]] = None,
    pipeline: Optional[Mapping[str, Any]] = None,
) -> None:
    global OUT_DIR, DATA_DIR, FIG_DIR
    global BENCH_ROOT, METHOD_DIRS, METHOD_VKEYS, METHOD_COLORS, METHOD_ORDER
    global COLOR_KEY, PREFER_VIS_H5AD
    global HIST_BINS, HIST_ALPHA, UNITVELO_MSE_SCALE
    global STREAM_KW
    global PROJECTION_BINS, PROJECTION_XLIM

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

    if not shared:
        return

    BENCH_ROOT = Path(shared.get("bench_root", BENCH_ROOT))
    method_dirs_raw = shared.get("method_dirs", {})
    if method_dirs_raw:
        METHOD_DIRS = {k: Path(v) for k, v in method_dirs_raw.items()}

    method_vkeys_raw = shared.get("method_vkeys", {})
    if method_vkeys_raw:
        METHOD_VKEYS = {k: tuple(v) for k, v in method_vkeys_raw.items()}

    method_colors_raw = shared.get("colors", {})
    if method_colors_raw:
        METHOD_COLORS = dict(method_colors_raw)

    METHOD_ORDER = tuple(shared.get("method_order", METHOD_ORDER))

    COLOR_KEY = str(shared.get("color_key", COLOR_KEY))
    PREFER_VIS_H5AD = bool(shared.get("prefer_vis_h5ad", PREFER_VIS_H5AD))

    HIST_BINS = int(shared.get("hist", {}).get("bins", HIST_BINS))
    HIST_ALPHA = float(shared.get("hist", {}).get("alpha", HIST_ALPHA))
    UNITVELO_MSE_SCALE = float(shared.get("unitvelo_mse_scale", UNITVELO_MSE_SCALE))

    STREAM_KW = {**STREAM_KW, **dict(shared.get("stream", {}))}

    PROJECTION_BINS = int(shared.get("projection_masking", {}).get("bins", PROJECTION_BINS))
    xl = shared.get("projection_masking", {}).get("xlim", list(PROJECTION_XLIM))
    PROJECTION_XLIM = (float(xl[0]), float(xl[1]))

