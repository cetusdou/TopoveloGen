"""Paired-seed cell/gene factorial and selected diffusion/dropout closure audit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

from scripts.sensitivity import revision_factor_core as regions
from scripts.common.revision_cache import prepare_cache
from scripts.common.velocity_metrics import cosine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results" / "revision_cell_gene_factor_closure"
SEEDS = (11, 22, 33)
TOPOLOGIES = tuple(regions.TOPOLOGIES)
CELLS = (300, 900)
GENES = (80, 320)
DIFFUSION = (.002, .012)
DROPOUT = (0., .3)


def design():
    for topology, cells, genes in itertools.product(TOPOLOGIES, CELLS, GENES):
        yield dict(family="cell_gene", topology=topology, cells=cells, genes=genes,
                   diffusion=regions.BASE["diffusion"], dropout=regions.DEFAULT_SYNTH["dropout_prob"])
    for topology, diffusion, dropout in itertools.product(TOPOLOGIES, DIFFUSION, DROPOUT):
        yield dict(family="diffusion_dropout", topology=topology, cells=600, genes=160,
                   diffusion=diffusion, dropout=dropout)


def dataset_id(row, seed):
    return (f"{row['family']}_{row['topology']}_c{row['cells']}_g{row['genes']}"
            f"_d{row['diffusion']:.3f}_p{row['dropout']:.2f}_s{seed}")


def run_dataset(row, seed):
    cfg = {**regions.BASE, "observations": row["cells"], "genes": row["genes"],
           "diffusion": row["diffusion"], "signal_reference_genes": 160}
    regions.core.SYNTH_KW = {**regions.DEFAULT_SYNTH, "dropout_prob": row["dropout"]}
    data = regions.simulate(cfg, row["topology"], seed)
    pred, truth, _, observed, _, _ = regions.infer_and_smooth(data, cfg)
    assert pred.shape == truth.shape == observed.shape == (row["cells"], row["genes"])
    assert np.isfinite(pred).all() and np.isfinite(truth).all() and np.isfinite(observed).all()
    pca = PCA(n_components=2, svd_solver="full").fit(observed)
    before = cosine(pred, truth)
    after = cosine(pred @ pca.components_.T, truth @ pca.components_.T)
    valid = np.isfinite(before) & np.isfinite(after)
    assert valid.sum() > 0
    frame = pd.DataFrame({**row, "seed": seed, "cell_index": np.arange(len(before)),
                          "region": regions.region_labels(row["topology"], data["z"], truth),
                          "gene_cosine": before, "projected_cosine": after,
                          "delta": after - before, "valid_pair": valid,
                          "masking_label": valid & (after > .7) & (before < .2)})
    return frame


def metrics(frame, keys):
    valid = frame.loc[frame.valid_pair]
    result = valid.groupby(keys, observed=True).agg(
        n_valid=("delta", "size"), mean_gene=("gene_cosine", "mean"),
        mean_projected=("projected_cosine", "mean"), mean_delta=("delta", "mean"),
        masking_fraction=("masking_label", "mean")).reset_index()
    total = frame.groupby(keys, observed=True).size().rename("n_total").reset_index()
    return total.merge(result, on=keys, validate="one_to_one")


def summarize(seeds, keys):
    measures = ("mean_gene", "mean_projected", "mean_delta", "masking_fraction")
    rows = []
    for key, group in seeds.groupby(keys, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(keys, key))
        record["n_seeds"] = len(group)
        record["min_valid_cells"] = int(group.n_valid.min())
        for measure in measures:
            values = group[measure].to_numpy(dtype=float)
            center = float(values.mean())
            half = float(stats.t.ppf(.975, len(values)-1) * stats.sem(values)) if len(values) > 1 else np.nan
            record[measure] = center
            record[measure + "_ci95_low"] = center - half
            record[measure + "_ci95_high"] = center + half
        rows.append(record)
    return pd.DataFrame(rows)


def paired_contrasts(seed_metrics):
    contrasts = []
    definitions = (
        ("cell_gene", "genes", 80, 320, ["topology", "seed", "cells"]),
        ("cell_gene", "cells", 300, 900, ["topology", "seed", "genes"]),
        ("diffusion_dropout", "diffusion", .002, .012, ["topology", "seed", "dropout"]),
        ("diffusion_dropout", "dropout", 0., .3, ["topology", "seed", "diffusion"]),
    )
    for family, factor, low, high, keys in definitions:
        part = seed_metrics[seed_metrics.family == family]
        left = part[part[factor] == low]
        right = part[part[factor] == high]
        merged = left.merge(right, on=keys, suffixes=("_low", "_high"), validate="one_to_one")
        assert len(merged) == len(left) == len(right)
        for _, item in merged.iterrows():
            row = {key: item[key] for key in keys}
            row.update(family=family, factor=factor, low=low, high=high)
            for metric in ("mean_gene", "mean_projected", "mean_delta", "masking_fraction"):
                row[metric + "_difference"] = item[metric + "_high"] - item[metric + "_low"]
            contrasts.append(row)
    return pd.DataFrame(contrasts)


def figure(overall, region_summary, contrast_summary, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    colors = {"cycle": "#087e76", "bifurcation": "#bd4f59", "cycle_bifurcation": "#62529a"}
    main = overall[overall.family == "cell_gene"]
    for topology, group in main.groupby("topology"):
        group = group.sort_values(["cells", "genes"])
        x = np.arange(len(group))
        label = [f"{int(r.cells)}/{int(r.genes)}" for r in group.itertuples()]
        axes[0, 0].errorbar(x, group.mean_delta, yerr=[group.mean_delta-group.mean_delta_ci95_low,
            group.mean_delta_ci95_high-group.mean_delta], marker="o", capsize=2,
            color=colors[topology], label=topology)
        axes[0, 0].set_xticks(x, label)
    axes[0, 0].set(title="Cell/gene factorial", xlabel="Cells / genes", ylabel="Projected - gene cosine")
    axes[0, 0].legend(frameon=False, fontsize=8)
    contrast = overall[overall.family == "diffusion_dropout"]
    for topology, group in contrast.groupby("topology"):
        group = group.sort_values(["diffusion", "dropout"])
        x = np.arange(len(group))
        label = [f"{r.diffusion:.3f}/{r.dropout:.1f}" for r in group.itertuples()]
        axes[0, 1].plot(x, group.mean_delta, marker="o", color=colors[topology], label=topology)
        axes[0, 1].set_xticks(x, label)
    axes[0, 1].set(title="Selected diffusion/dropout grid", xlabel="Diffusion / dropout", ylabel="Projected - gene cosine")
    baseline = region_summary[(region_summary.family == "cell_gene") &
                              (region_summary.cells == 300) & (region_summary.genes == 80)]
    for topology, group in baseline.groupby("topology"):
        axes[1, 0].scatter(group.region, group.mean_delta, color=colors[topology], label=topology)
    axes[1, 0].set(title="Regions at 300 cells / 80 genes", ylabel="Projected - gene cosine")
    axes[1, 0].tick_params(axis="x", rotation=40)
    pc = contrast_summary[contrast_summary.factor.isin(["cells", "genes"])]
    for topology, group in pc.groupby("topology"):
        axes[1, 1].scatter(group.factor + ":" + group.level.astype(str),
                           group.mean_delta_difference, color=colors[topology], label=topology)
    axes[1, 1].axhline(0, color="gray", linewidth=.7)
    axes[1, 1].set(title="Paired high - low effects", ylabel="Change in projection gap")
    axes[1, 1].tick_params(axis="x", rotation=40)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(out / "factor_closure.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--out", type=Path, default=OUT,
                        help="Fresh result directory for a new code/config revision")
    args = parser.parse_args()
    out = args.out.resolve()
    rows = list(design())
    expected = len(rows) * len(args.seeds)
    assert len(rows) == 24 and len(set(args.seeds)) == len(args.seeds)
    source_paths = [Path(__file__), *regions.SOURCE_DEPENDENCIES]
    provenance = dict(
        seeds=args.seeds, dataset_count=expected, design_rows=rows,
        crossed={"cell_gene": ["topology", "cells", "genes"],
                 "diffusion_dropout": ["topology", "diffusion", "dropout"]},
        not_crossed={"cell_gene": {"diffusion": regions.BASE["diffusion"],
                                     "dropout": regions.DEFAULT_SYNTH["dropout_prob"]},
                     "diffusion_dropout": {"cells": 600, "genes": 160}},
        fixed={"signal_reference_genes": 160, "base": regions.BASE,
               "synth_defaults": regions.DEFAULT_SYNTH,
               "reference": "Same smoothed RNA-state truth before/after one PCA fit to smoothed observed RNA state",
               "estimator": "Oracle capture-corrected deterministic scVelo; no cross-method comparison",
               "replication_unit": "Independent seeds, paired across all design rows"},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        versions={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "scvelo", "scikit-learn")})
    config_bytes = json.dumps(provenance, sort_keys=True, indent=2).encode("utf-8")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    provenance["config_sha256"] = config_hash
    prepare_cache(out, provenance)
    dataset_dir = out / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    frames = []
    for row in rows:
        for seed in args.seeds:
            path = dataset_dir / (dataset_id(row, seed) + ".csv")
            if path.exists():
                frame = pd.read_csv(path)
            else:
                print(path.stem, flush=True)
                frame = run_dataset(row, seed)
                frame.to_csv(path, index=False)
            assert len(frame) == row["cells"]
            assert frame.valid_pair.any() and np.isfinite(frame.loc[frame.valid_pair, ["gene_cosine", "projected_cosine", "delta"]].to_numpy()).all()
            frames.append(frame)
    assert len(frames) == expected
    assert len(list(dataset_dir.glob("*.csv"))) == expected
    cells = pd.concat(frames, ignore_index=True)
    keys = ["family", "topology", "cells", "genes", "diffusion", "dropout", "seed"]
    assert cells.groupby(keys).ngroups == expected
    seeds = metrics(cells, keys)
    assert len(seeds) == expected and (seeds.n_valid > 0).all()
    region_seeds = metrics(cells, keys + ["region"])
    summary = summarize(seeds, keys[:-1])
    region_summary = summarize(region_seeds, keys[:-1] + ["region"])
    contrasts = paired_contrasts(seeds)
    contrast_keys = ["family", "factor", "topology"]
    contrasts["level"] = np.where(contrasts.factor.eq("genes"), contrasts.get("cells", np.nan),
                         np.where(contrasts.factor.eq("cells"), contrasts.get("genes", np.nan),
                         np.where(contrasts.factor.eq("diffusion"), contrasts.get("dropout", np.nan), contrasts.get("diffusion", np.nan))))
    contrast_summary = summarize_contrasts(contrasts, contrast_keys + ["level"])
    seeds.to_csv(out / "seed_overall_metrics.csv", index=False)
    region_seeds.to_csv(out / "seed_region_metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    region_summary.to_csv(out / "region_summary.csv", index=False)
    contrasts.to_csv(out / "paired_seed_contrasts.csv", index=False)
    contrast_summary.to_csv(out / "paired_contrast_summary.csv", index=False)
    figure(summary, region_summary, contrast_summary, out)
    print(f"Verified {expected} datasets, {len(cells)} cells, {len(region_seeds)} seed-region rows; {out}", flush=True)


def summarize_contrasts(contrasts, keys):
    rows = []
    for key, group in contrasts.groupby(keys, observed=True):
        record = dict(zip(keys, key))
        record["n_seeds"] = len(group)
        for metric in ("mean_gene", "mean_projected", "mean_delta", "masking_fraction"):
            values = group[metric + "_difference"].to_numpy(dtype=float)
            half = float(stats.t.ppf(.975, len(values)-1) * stats.sem(values)) if len(values) > 1 else np.nan
            record[metric + "_difference"] = float(values.mean())
            record[metric + "_difference_ci95_low"] = float(values.mean() - half)
            record[metric + "_difference_ci95_high"] = float(values.mean() + half)
        rows.append(record)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
