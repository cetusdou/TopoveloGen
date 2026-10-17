"""Matched-seed three-factor sensitivity of coordinate-matched PCA agreement."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_cache import prepare_cache
from experiments.revision_factor_core import BASE, DEFAULT_SYNTH, SOURCE_DEPENDENCIES, run_one

OUT = ROOT / "results" / "revision_interaction_scan"
SEEDS = (11, 22, 33, 44, 55)
TOPOLOGIES = ("cycle", "bifurcation", "cycle_bifurcation")
LEVELS = {
    "diffusion": {"low": .002, "high": .012},
    "kinetics": {"slow": (.5, .25), "fast": (2., 1.)},
    "dropout": {"none": 0., "high": .3},
}


def conditions():
    for diffusion, kinetics, dropout in itertools.product(*[LEVELS[key] for key in LEVELS]):
        beta, gamma = LEVELS["kinetics"][kinetics]
        key = f"d-{diffusion}_k-{kinetics}_o-{dropout}"
        override = dict(diffusion=LEVELS["diffusion"][diffusion],
                        splicing_rate=beta, degradation_rate=gamma,
                        dropout_prob=LEVELS["dropout"][dropout])
        yield key, diffusion, kinetics, dropout, override


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT,
                        help="Fresh result directory for a new code/config revision")
    args = parser.parse_args()
    out = args.out.resolve()
    source = [Path(__file__), *SOURCE_DEPENDENCIES]
    config = dict(base=BASE, synth_defaults=DEFAULT_SYNTH, seeds=SEEDS,
                  topologies=TOPOLOGIES, levels=LEVELS,
                  estimand="Mean paired PCA minus gene-space cosine on valid cells; oracle capture-corrected deterministic scVelo",
                  design="Full 2x2x2 crossing of diffusion, kinetics, dropout within each topology; cell and gene counts fixed",
                  versions={name: importlib.metadata.version(name) for name in
                            ("numpy", "scipy", "pandas", "scvelo", "scikit-learn")},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    prepare_cache(out, config)
    dataset_dir = out / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    rows = []
    for key, diffusion, kinetics, dropout, override in conditions():
        for topology in TOPOLOGIES:
            for seed in SEEDS:
                target = dataset_dir / f"{key}_{topology}_{seed}.csv"
                if target.exists():
                    frame = pd.read_csv(target)
                else:
                    print(key, topology, seed, flush=True)
                    frame = run_one(key, override, topology, seed)
                    frame.to_csv(target, index=False)
                valid = frame[np.isfinite(frame.gene_cosine) & np.isfinite(frame.projected_cosine)]
                if valid.empty:
                    raise ValueError(f"No valid cells: {target}")
                rows.append(dict(condition=key, diffusion=diffusion, kinetics=kinetics,
                                 dropout=dropout, topology=topology, seed=seed,
                                 n_valid=len(valid), valid_fraction=len(valid)/len(frame),
                                 mean_gene=valid.gene_cosine.mean(),
                                 mean_projected=valid.projected_cosine.mean(),
                                 mean_delta=valid.delta.mean(),
                                 masking_fraction=valid.masking_label.mean()))
    metrics = pd.DataFrame(rows)
    if len(metrics) != 8 * len(TOPOLOGIES) * len(SEEDS):
        raise AssertionError("Incomplete factorial design")
    metrics.to_csv(out / "seed_metrics.csv", index=False)
    summary = metrics.groupby(["topology", "diffusion", "kinetics", "dropout"], observed=True).agg(
        mean_delta=("mean_delta", "mean"), sd_delta=("mean_delta", "std"),
        mean_masking=("masking_fraction", "mean"),
        n_seeds=("seed", "nunique")).reset_index()
    summary["ci95_half_delta"] = stats.t.ppf(.975, summary.n_seeds-1) * summary.sd_delta / np.sqrt(summary.n_seeds)
    summary.to_csv(out / "summary.csv", index=False)
    contrasts = []
    for (topology, dropout, seed), group in metrics.groupby(["topology", "dropout", "seed"]):
        values = group.set_index(["diffusion", "kinetics"]).mean_delta
        contrast = (values.loc["high", "fast"] - values.loc["high", "slow"]
                    - values.loc["low", "fast"] + values.loc["low", "slow"])
        contrasts.append(dict(topology=topology, dropout=dropout, seed=seed,
                              diffusion_x_kinetics=float(contrast)))
    pd.DataFrame(contrasts).to_csv(out / "paired_interaction_contrasts.csv", index=False)
    vmax = max(.25, np.ceil(np.max(np.abs(summary.mean_delta)) * 10) / 10)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey=True, layout="constrained")
    for row, dropout in enumerate(LEVELS["dropout"]):
        for col, topology in enumerate(TOPOLOGIES):
            subset = summary[(summary.topology == topology) & (summary.dropout == dropout)]
            grid = subset.pivot(index="kinetics", columns="diffusion", values="mean_delta").loc[
                ["slow", "fast"], ["low", "high"]]
            ax = axes[row, col]
            im = ax.imshow(grid, vmin=-vmax, vmax=vmax, cmap="RdBu_r", aspect="auto")
            for y, x in itertools.product(range(2), range(2)):
                val = grid.iloc[y, x]
                ax.text(x, y, f"{val:+.2f}", ha="center", va="center",
                        color="white" if abs(val) > .65*vmax else "black", fontsize=9)
            ax.set_xticks([0, 1], ["D=.002", "D=.012"])
            ax.set_yticks([0, 1], ["slow", "fast"])
            if row == 0:
                ax.set_title(topology.replace("_", " "))
            if col == 0:
                ax.set_ylabel(f"dropout {dropout}\nkinetics")
    fig.colorbar(im, ax=axes, label="PCA - gene cosine", shrink=.82)
    fig.savefig(out / "interaction_scan.pdf")
    plt.close(fig)
    print(summary[["topology", "diffusion", "kinetics", "dropout", "mean_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
