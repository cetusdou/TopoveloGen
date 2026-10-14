"""One-factor-at-a-time, seed-replicated same-reference PCA velocity audit."""
from __future__ import annotations

import json
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
from sklearn.decomposition import PCA

from experiments.figures.fig_coord_velocity_consistency.lib import core
from experiments.revision_coordinate_projection import cosine, infer_and_smooth, simulate

OUT = ROOT / "results" / "revision_factor_regions"
SEEDS = [11, 22, 33, 44, 55]
TOPOLOGIES = ["cycle", "bifurcation", "cycle_bifurcation"]
BASE = dict(dt=.01, steps=1200, burnin_steps=200, diffusion=.006,
            trajectories=48, genes=160, observations=600, save_every=20,
            neighbors=20)
DEFAULT_SYNTH = dict(core.SYNTH_KW)
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


def region_labels(topology, z, gt):
    x, y = z[:, 0], z[:, 1]
    radius = np.linalg.norm(z, axis=1)
    labels = np.full(len(z), "other", dtype="U18")
    if topology == "cycle":
        labels[np.abs(radius - 1) < .35] = "cycle_rim"
    elif topology == "bifurcation":
        labels[np.abs(x) < .35] = "junction"
        labels[x > 1.5] = "terminal_branch"
    else:
        labels[np.abs(radius - 1) < .35] = "cycle_rim"
        labels[(x > .7) & (x < 1.4) & (np.abs(y) < .7)] = "cycle_exit"
        labels[(x >= 1.4) & (x < 2.0)] = "junction"
        labels[x >= 2.0] = "terminal_branch"
    norm = np.linalg.norm(gt, axis=1)
    labels[norm <= np.quantile(norm, .1)] = "low_velocity"
    return labels


def run_one(condition, override, topology, seed):
    cfg = {**BASE, **{k: v for k, v in override.items() if k in BASE}}
    if "signal_reference_genes" in override:
        cfg["signal_reference_genes"] = override["signal_reference_genes"]
    core.SYNTH_KW = {**DEFAULT_SYNTH, **{k: v for k, v in override.items() if k in DEFAULT_SYNTH}}
    data = simulate(cfg, topology, seed)
    pred, gt, _, observed, _, _ = infer_and_smooth(data, cfg)
    pca = PCA(n_components=2, svd_solver="full").fit(observed)
    gene = cosine(pred, gt)
    projected = cosine(pred @ pca.components_.T, gt @ pca.components_.T)
    labels = region_labels(topology, data["z"], gt)
    frame = pd.DataFrame(dict(condition=condition, topology=topology, seed=seed,
                              region=labels, gene_cosine=gene, projected_cosine=projected,
                              delta=projected-gene, pred_norm=np.linalg.norm(pred, axis=1),
                              gt_norm=np.linalg.norm(gt, axis=1)))
    frame["masking_label"] = (frame.projected_cosine > .7) & (frame.gene_cosine < .2)
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_dir = OUT / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    legacy_partial = OUT / "per_cell.csv"
    if legacy_partial.exists():
        prior = pd.read_csv(legacy_partial)
        for (condition, topology, seed), group in prior.groupby(["condition", "topology", "seed"]):
            if condition not in CONDITIONS:
                continue
            target = dataset_dir / f"{condition}_{topology}_{seed}.csv"
            if not target.exists():
                group.to_csv(target, index=False)
    (OUT / "config.json").write_text(json.dumps({"base": BASE, "seeds": SEEDS,
        "topologies": TOPOLOGIES, "conditions": CONDITIONS, "synth_defaults": DEFAULT_SYNTH,
        "estimand": "Oracle capture-corrected deterministic scVelo; within each dataset prediction and RNA-state reference use the same fitted PCA map",
        "regions": "Prespecified geometric bins; low_velocity is lowest within-dataset decile of reference norm and overrides geometry",
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "scvelo", "scikit-learn")},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), ROOT / "experiments/revision_coordinate_projection.py",
                                       ROOT / "experiments/figures/fig_coord_velocity_consistency/lib/core.py")}}, indent=2), encoding="utf-8")
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
    cells.to_csv(OUT / "per_cell_combined.csv", index=False)
    valid = cells[np.isfinite(cells.gene_cosine) & np.isfinite(cells.projected_cosine)]
    grouped = valid.groupby(["condition", "topology", "seed", "region"], observed=True)
    rows = grouped.agg(n=("delta", "size"), mean_gene=("gene_cosine", "mean"),
                       mean_projected=("projected_cosine", "mean"), mean_delta=("delta", "mean"),
                       masking_fraction=("masking_label", "mean")).reset_index()
    rows.to_csv(OUT / "seed_region_metrics.csv", index=False)
    overall = valid.groupby(["condition", "topology", "seed"]).agg(
        n=("delta", "size"), mean_gene=("gene_cosine", "mean"),
        mean_projected=("projected_cosine", "mean"), mean_delta=("delta", "mean"),
        masking_fraction=("masking_label", "mean")).reset_index()
    overall.to_csv(OUT / "seed_overall_metrics.csv", index=False)
    summary = overall.groupby(["condition", "topology"]).agg(
        mean_delta=("mean_delta", "mean"), sd_delta=("mean_delta", "std"),
        mean_masking=("masking_fraction", "mean"), sd_masking=("masking_fraction", "std")).reset_index()
    summary["ci95_half_delta"] = stats.t.ppf(.975, len(SEEDS)-1) * summary.sd_delta / np.sqrt(len(SEEDS))
    summary["ci95_half_masking"] = stats.t.ppf(.975, len(SEEDS)-1) * summary.sd_masking / np.sqrt(len(SEEDS))
    summary.to_csv(OUT / "summary.csv", index=False)
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
    fig.savefig(OUT / "factor_scan.pdf")
    fig.savefig(OUT / "factor_scan.png", dpi=180)
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
    fig.savefig(OUT / "region_scan.pdf")
    fig.savefig(OUT / "region_scan.png", dpi=180)
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
