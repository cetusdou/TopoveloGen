"""Fig4series core: thin wrap of experiments.figure4_series_pipeline (Stage 1/2)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# scanpy/numba need a writable cache before importing figure4_series_pipeline
_numba = Path(os.environ.get("NUMBA_CACHE_DIR", REPO_ROOT / ".numba_cache"))
_numba.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba))

import experiments.figure4_series_pipeline as f4pipe  # noqa: E402
from experiments.figure4_series_pipeline import (  # noqa: E402
    DEFAULT_PANCREAS_H5AD,
    DEFAULT_PP_CONFIG,
    PostProcessConfig,
    Figure4Result,
    apply_observation_noise,
    load_or_build_kinetics_cache,
)

SEED = 42
OUT_DIR = REPO_ROOT / "results" / "statistical_realism_figure4"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
KINETICS_CACHE = DATA_DIR / "kinetics_cache.pt"
PANCREAS_H5AD: Optional[Path] = DEFAULT_PANCREAS_H5AD
PP_CONFIG = DEFAULT_PP_CONFIG
STAGE1 = {}
NOISE_STAGE_DEMO = {}
FIGURE_MAIN_SIZE = (10.5, 8.5)
FIGURE_MAIN_DPI = 300
FIGURE_SUPP_SIZE = (8.0, 6.5)


def configure(*, out_dir=None, data_dir=None, fig_dir=None, pipeline: dict | None = None, shared: dict | None = None) -> None:
    """Apply JSON paths; push Stage1 into figure4_series_pipeline globals; build Stage2 PostProcessConfig."""
    global OUT_DIR, DATA_DIR, FIG_DIR, KINETICS_CACHE, PANCREAS_H5AD, PP_CONFIG, SEED
    global STAGE1, NOISE_STAGE_DEMO, FIGURE_MAIN_SIZE, FIGURE_MAIN_DPI, FIGURE_SUPP_SIZE

    if shared:
        SEED = int(shared.get("seed", SEED))

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
        KINETICS_CACHE = DATA_DIR / "kinetics_cache.pt"
        return

    STAGE1 = dict(pipeline.get("stage1", {}))
    NOISE_STAGE_DEMO = dict(pipeline.get("noise_stage_demo", {}))
    if "figure_main_size" in pipeline:
        FIGURE_MAIN_SIZE = tuple(pipeline["figure_main_size"])
    FIGURE_MAIN_DPI = int(pipeline.get("figure_main_dpi", FIGURE_MAIN_DPI))
    if "figure_supp_size" in pipeline:
        FIGURE_SUPP_SIZE = tuple(pipeline["figure_supp_size"])

    # Push Stage-1 hyper-parameters into figure4_series_pipeline module globals
    # (cache key / simulation use these; JSON is the editable source of truth).
    _stage1_map = {
        "seed": "SEED",
        "n_samples": "N_SAMPLES",
        "n_cells": "N_CELLS",
        "d_genes_base": "D_GENES_BASE",
        "n_steps": "N_STEPS",
        "dt": "DT",
        "noise_level": "NOISE_LEVEL",
        "alpha_scale": "ALPHA_SCALE",
        "n_sampling_steps": "N_SAMPLING_STEPS",
        "paga_min_connectivity": "PAGA_MIN_CONNECTIVITY",
        "fusion_temp": "FUSION_TEMP",
        "target_n_genes": "TARGET_N_GENES",
    }
    for json_key, attr in _stage1_map.items():
        if json_key in STAGE1:
            setattr(f4pipe, attr, type(getattr(f4pipe, attr))(STAGE1[json_key]))

    cache_name = STAGE1.get("kinetics_cache_name", "kinetics_cache.pt")
    KINETICS_CACHE = DATA_DIR / cache_name

    h5ad = STAGE1.get("pancreas_h5ad")
    if h5ad:
        p = Path(h5ad)
        PANCREAS_H5AD = p if p.exists() else None
    else:
        PANCREAS_H5AD = DEFAULT_PANCREAS_H5AD if DEFAULT_PANCREAS_H5AD.exists() else None

    pp = dict(pipeline.get("stage2_postprocess", {}))
    known = {f.name for f in PostProcessConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in pp.items() if k in known}
    if "seed" not in filtered:
        filtered["seed"] = int(STAGE1.get("seed", SEED))
    PP_CONFIG = PostProcessConfig(**filtered)
    SEED = int(filtered.get("seed", SEED))


def load_or_build_kinetics(*, force_rebuild: bool = False) -> dict:
    """Stage 1: load or build continuous kinetics cache via figure4_series_pipeline."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return load_or_build_kinetics_cache(KINETICS_CACHE, PANCREAS_H5AD, force_rebuild)


def apply_stage2(kinetics_cache: dict, pp_cfg: PostProcessConfig | None = None) -> Figure4Result:
    """Stage 2: observation noise (lib_size / NB / dropout / downsampling)."""
    return apply_observation_noise(kinetics_cache, pp_cfg or PP_CONFIG)


def run_stages(*, force_rebuild: bool = False, cache_only: bool = False) -> Figure4Result | None:
    """Run Stage1 (+ optional Stage2). Returns None if cache_only."""
    print("Loading/building continuous kinetics (Stage 1 – cached)…")
    kinetics_cache = load_or_build_kinetics(force_rebuild=force_rebuild)
    if cache_only:
        print("Cache-only mode: Stage-1 kinetics saved. Observation noise not applied.")
        return None
    print("Applying observation noise (Stage 2)…")
    return apply_stage2(kinetics_cache)
