"""One-factor-at-a-time, seed-replicated same-reference PCA velocity audit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Preserve the original import surface for external experiment scripts.
from experiments.revision_factor_core import (
    BASE, DEFAULT_SYNTH, SOURCE_DEPENDENCIES, TOPOLOGIES, core, cosine, infer_and_smooth,
    region_labels, run_one, simulate,
)
from experiments.revision_cache import prepare_cache

OUT = ROOT / "results" / "revision_factor_regions"
SEEDS = [11, 22, 33, 44, 55]
CONDITIONS = {
    "baseline": {},
    "diffusion_low": {"diffusion": .002},
    "diffusion_high": {"diffusion": .012},
    "kinetics_slow": {"splicing_rate": .5, "degradation_rate": .25},
    "kinetics_fast": {"splicing_rate": 2., "degradation_rate": 1.},
    "dropout_none": {"dropout_prob": 0.},
    "dropout_high": {"dropout_prob": .3},
    "cells_low": {"observations": 300},
    "cells_high": {"observations": 900},
    "genes_fixed_low": {"genes": 80, "signal_reference_genes": 160},
    "genes_fixed_high": {"genes": 320, "signal_reference_genes": 160},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT,
                        help="Fresh result directory for a new code/config revision")
    args = parser.parse_args()
    out = args.out.resolve()
    config = {"base": BASE, "seeds": SEEDS,
        "topologies": TOPOLOGIES, "conditions": CONDITIONS, "synth_defaults": DEFAULT_SYNTH,
        "estimand": "Oracle capture-corrected deterministic scVelo; within each dataset prediction and RNA-state reference use the same fitted PCA map",
        "regions": "Prespecified geometric bins; low_velocity is lowest within-dataset decile of reference norm and overrides geometry",
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scvelo", "scikit-learn")},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), *SOURCE_DEPENDENCIES)}}
    prepare_cache(out, config)
    dataset_dir = out / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    legacy_partial = out / "per_cell.csv"
    if legacy_partial.exists():
        prior = pd.read_csv(legacy_partial)
        for (condition, topology, seed), group in prior.groupby(["condition", "topology", "seed"]):
            if condition not in CONDITIONS:
                continue
            target = dataset_dir / f"{condition}_{topology}_{seed}.csv"
            if not target.exists():
                group.to_csv(target, index=False)
    frames = []
    for condition, override in CONDITIONS.items():
        for topology in TOPOLOGIES:
            for seed in SEEDS:
                target = dataset_dir / f"{condition}_{topology}_{seed}.csv"
                if target.exists():
                    frames.append(pd.read_csv(target))
                    continue
                print(condition, topology, seed, flush=True)
                frame = run_one(condition, override, topology, seed)
                frame.to_csv(target, index=False)
                frames.append(frame)
    cells = pd.concat(frames, ignore_index=True)
    cells.to_csv(out / "per_cell_combined.csv", index=False)
    valid = cells[np.isfinite(cells.gene_cosine) & np.isfinite(cells.projected_cosine)]
    grouped = valid.groupby(["condition", "topology", "seed", "region"], observed=True)
    rows = grouped.agg(n=("delta", "size"), mean_gene=("gene_cosine", "mean"),
                       mean_projected=("projected_cosine", "mean"), mean_delta=("delta", "mean"),
                       masking_fraction=("masking_label", "mean")).reset_index()
    rows.to_csv(out / "seed_region_metrics.csv", index=False)
    overall = valid.groupby(["condition", "topology", "seed"]).agg(
        n=("delta", "size"), mean_gene=("gene_cosine", "mean"),
        mean_projected=("projected_cosine", "mean"), mean_delta=("delta", "mean"),
        masking_fraction=("masking_label", "mean")).reset_index()
    overall.to_csv(out / "seed_overall_metrics.csv", index=False)
    summary = overall.groupby(["condition", "topology"]).agg(
        mean_delta=("mean_delta", "mean"), sd_delta=("mean_delta", "std"),
        mean_masking=("masking_fraction", "mean"), sd_masking=("masking_fraction", "std")).reset_index()
    summary["ci95_half_delta"] = stats.t.ppf(.975, len(SEEDS)-1) * summary.sd_delta / np.sqrt(len(SEEDS))
    summary["ci95_half_masking"] = stats.t.ppf(.975, len(SEEDS)-1) * summary.sd_masking / np.sqrt(len(SEEDS))
    summary.to_csv(out / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for offset, (topology, group) in enumerate(summary.groupby("topology")):
        q = group.set_index("condition").loc[list(CONDITIONS)]
        x = np.arange(len(q)) + (offset - 1) * .16
        axes[0].errorbar(x, q.mean_delta, yerr=q.ci95_half_delta, fmt="o", capsize=2, label=topology)
        axes[1].errorbar(x, q.mean_masking, yerr=q.ci95_half_masking, fmt="o", capsize=2, label=topology)
    for ax, ylabel in zip(axes, ["Projected - gene cosine", "Masking-label fraction"]):
        ax.set_xticks(range(len(CONDITIONS)), list(CONDITIONS), rotation=65, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "factor_scan.pdf")
    fig.savefig(out / "factor_scan.png", dpi=180)
    baseline_regions = rows[rows.condition == "baseline"]
    regions = ["cycle_rim", "cycle_exit", "junction", "terminal_branch", "low_velocity"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for offset, (topology, group) in enumerate(baseline_regions.groupby("topology")):
        metric = group.groupby("region").agg(mean_delta=("mean_delta", "mean"),
            sd_delta=("mean_delta", "std"), mean_masking=("masking_fraction", "mean"),
            sd_masking=("masking_fraction", "std"), n_seeds=("seed", "nunique"))
        x = [i + (offset - 1) * .16 for i, name in enumerate(regions) if name in metric.index]
        for ax, value, sd in zip(axes, ["mean_delta", "mean_masking"], ["sd_delta", "sd_masking"]):
            q = metric.loc[[name for name in regions if name in metric.index]]
            half = stats.t.ppf(.975, q.n_seeds - 1) * q[sd].fillna(0) / np.sqrt(q.n_seeds)
            ax.errorbar(x, q[value], yerr=half, fmt="o", capsize=2, label=topology)
    for ax, ylabel in zip(axes, ["Projected - gene cosine", "Masking-label fraction"]):
        ax.set_xticks(range(len(regions)), regions, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(out / "region_scan.pdf")
    fig.savefig(out / "region_scan.png", dpi=180)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
