"""Focused Figure 4 higher-order count-statistics replay (R1 Major 14).

The metrics are held out from explicit marginal tuning, not from the reference
dataset used to construct/calibrate the simulator. Run from the repository root.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import platform
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_pancreas_higher_order"
os.environ.setdefault("NUMBA_CACHE_DIR", str(OUT / "numba_cache"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import anndata as ad  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

from experiments.figure4_series_pipeline import (  # noqa: E402
    PostProcessConfig,
    _add_noise_genes,
    _apply_count_noise,
)
from experiments.data_paths import pancreas_reference, result_dir

H5AD = pancreas_reference(require=False)
CACHE = result_dir("statistical_realism_figure4") / "data" / "kinetics_cache.pt"
CONFIG = ROOT / "experiments" / "figures" / "fig4series_pancreas_stats" / "configs" / "pipeline.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(s: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = s + u
    lib = total.sum(axis=1)
    factor = 1e4 / np.maximum(lib, 1)
    return np.log1p(s * factor[:, None]), np.log1p(u * factor[:, None]), np.log1p(total * factor[:, None])


def gene_us(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    s = s - s.mean(axis=0)
    u = u - u.mean(axis=0)
    den = np.sqrt((s * s).sum(axis=0) * (u * u).sum(axis=0))
    return np.divide((s * u).sum(axis=0), den, out=np.full(s.shape[1], np.nan), where=den > 0)


def conditional_fano(counts: np.ndarray, library: np.ndarray) -> np.ndarray:
    """Per-gene mean Fano across within-sample library-size quartiles."""
    cuts = np.quantile(library, [0.25, 0.5, 0.75])
    bins = np.searchsorted(cuts, library, side="right")
    values = []
    for b in range(4):
        x = counts[bins == b]
        mean = x.mean(axis=0)
        var = x.var(axis=0, ddof=1)
        values.append(np.divide(var, mean, out=np.full_like(mean, np.nan), where=mean >= 0.1))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(np.asarray(values), axis=0)


def pair_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cov = np.cov(x, rowvar=False)
    sd = np.sqrt(np.diag(cov))
    den = np.outer(sd, sd)
    cor = np.divide(cov, den, out=np.full_like(cov, np.nan), where=den > 0)
    tri = np.triu_indices(x.shape[1], 1)
    return cov[tri], cor[tri]


def cell_distances(x: np.ndarray) -> np.ndarray:
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0)
    return np.sqrt(d2[np.triu_indices(x.shape[0], 1)] / x.shape[1])


def finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def summarize(real: np.ndarray, sim: np.ndarray) -> dict:
    real, sim = finite(real), finite(sim)
    if not len(real) or not len(sim):
        return {"real_n": len(real), "synthetic_n": len(sim), "failure": "no finite values"}
    rm, sm = float(np.median(real)), float(np.median(sim))
    return {
        "real_n": len(real), "synthetic_n": len(sim),
        "real_median": rm, "synthetic_median": sm,
        "median_difference_synthetic_minus_real": sm - rm,
        "median_ratio_synthetic_over_real": sm / rm if rm != 0 else None,
        "wasserstein_1": float(wasserstein_distance(real, sim)),
        "real_q10_q90": np.quantile(real, [0.1, 0.9]).tolist(),
        "synthetic_q10_q90": np.quantile(sim, [0.1, 0.9]).tolist(),
    }


def rank_matched_indices(real_mean: np.ndarray, sim_mean: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Uniform quantile positions cover low and high abundance without cherry-picking.
    q = (np.arange(n) + rng.random(n)) / n
    ri = np.argsort(real_mean, kind="stable")[np.minimum((q * len(real_mean)).astype(int), len(real_mean) - 1)]
    si = np.argsort(sim_mean, kind="stable")[np.minimum((q * len(sim_mean)).astype(int), len(sim_mean) - 1)]
    return ri, si


def calculate(s: np.ndarray, u: np.ndarray) -> dict[str, np.ndarray]:
    sl, ul, total = normalized(s, u)
    lib = (s + u).sum(axis=1)
    cov, corr = pair_stats(total)
    z = np.divide(total - total.mean(axis=0), total.std(axis=0),
                  out=np.zeros_like(total), where=total.std(axis=0) > 0)
    return {
        "u_s_raw": gene_us(s, u),
        "u_s_normalized": gene_us(sl, ul),
        "conditional_fano_spliced": conditional_fano(s, lib),
        "gene_covariance_log_cpm": cov,
        "gene_correlation_log_cpm": corr,
        "cell_distance_log_cpm": cell_distances(total),
        "cell_distance_gene_z": cell_distances(z),
    }


def joint_us_histogram(s: np.ndarray, u: np.ndarray) -> dict:
    """Fixed coarse grid, with zero in its own bin and tails pooled at 6."""
    x = np.minimum(np.log1p(u), 6.0 - 1e-9).ravel()
    y = np.minimum(np.log1p(s), 6.0 - 1e-9).ravel()
    counts, _, _ = np.histogram2d(x, y, bins=12, range=((0, 6), (0, 6)))
    return {
        "probability": counts / counts.sum(),
        "joint_zero_mass": float(np.mean((s == 0) & (u == 0))),
        "u_zero_mass": float(np.mean(u == 0)),
        "s_zero_mass": float(np.mean(s == 0)),
        "tail_fraction": float(np.mean((np.log1p(u) >= 6) | (np.log1p(s) >= 6))),
        "pairs": int(s.size),
    }


def draw_panels(first: dict, summary: dict, joint_first: dict, output: Path) -> None:
    names = [
        ("u_s_raw", "Gene-wise spliced/unspliced r", None),
        ("u_s_normalized", "Gene-wise r, log-normalized", None),
        ("conditional_fano_spliced", "Conditional spliced Fano", "log"),
        ("gene_covariance_log_cpm", "Gene-pair covariance, log counts per 10k", None),
        ("gene_correlation_log_cpm", "Gene-pair correlation, log counts per 10k", None),
        ("cell_distance_log_cpm", "Cell-pair RMS distance, log counts per 10k", None),
        ("cell_distance_gene_z", "Cell-pair RMS distance, gene z", None),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(5, 2, figsize=(10.5, 13.7), constrained_layout=True)
    colors = {"real": "#16706b", "synthetic": "#c75b35"}
    for ax, (key, title, scale) in zip(axes.flat, names):
        vals = [finite(first["real"][key]), finite(first["synthetic"][key])]
        if scale == "log":
            vals = [v[v > 0] for v in vals]
            vals = [np.log10(v) for v in vals]
        lo = min(np.quantile(v, 0.005) for v in vals if len(v))
        hi = max(np.quantile(v, 0.995) for v in vals if len(v))
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 45)
        for label, v in zip(colors, vals):
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8,
                    color=colors[label], label=label.capitalize())
            ax.axvline(np.median(v), color=colors[label], lw=0.8, ls="--")
        ax.set_title(title, loc="left", weight="semibold")
        ax.set_ylabel("Density")
        ax.set_xlabel("log10 Fano" if scale == "log" else "Value")
        w1_label = "W1 (raw)" if scale == "log" else "W1"
        ax.text(0.98, 0.96, f"{w1_label} = {summary[key]['wasserstein_1']:.3g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    axes.flat[0].legend(frameon=False)
    axes[3, 1].axis("off")
    axes[3, 1].text(0, 0.9, "Equal-size samples; abundance-quantile-matched genes\n"
                       "No gene identity correspondence. First deterministic draw shown.\n"
                       "Intervals in report are subsample sensitivity, not CIs.\n"
                       "Same pancreas reference informed simulator construction.",
                       va="top", linespacing=1.5)
    for ax, label in zip(axes[4], ("real", "synthetic")):
        probability = joint_first[label]["probability"]
        image = ax.imshow(np.log10(probability.T + 1e-6), origin="lower",
                          extent=(0, 6, 0, 6), vmin=-6, vmax=0,
                          cmap="viridis", interpolation="nearest", aspect="equal")
        ax.set_title(f"Joint counts: {label.capitalize()}", loc="left", weight="semibold")
        ax.set_xlabel("log1p(unspliced count)")
        ax.set_ylabel("log1p(spliced count)")
        ax.set_xticks([0, 2, 4, 6])
        ax.set_yticks([0, 2, 4, 6])
        ax.text(0.98, 0.96, f"P(0, 0) = {joint_first[label]['joint_zero_mass']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="white", bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2})
    fig.colorbar(image, ax=axes[4].tolist(), label="log10 bin probability (+1e-6)",
                 shrink=0.85, pad=0.02)
    axes[3, 1].text(0, 0.43,
                    f"Joint all-pair TV = {summary['joint_us_log1p']['first_draw_tv']:.3f}; "
                    f"nonzero TV = {summary['joint_us_log1p']['first_draw_nonzero_tv']:.3f}\n"
                    "Fixed 0.5 bins; values above 6 pooled in last bin.\n"
                    "Raw-count scale and shared zero mass affect TV.",
                    va="top", linespacing=1.5)
    fig.suptitle("Pancreas higher-order count statistics | Figure 4 current-code replay",
                 fontsize=13, weight="semibold")
    fig.savefig(output, bbox_inches="tight", metadata={"Title": "Pancreas higher-order count statistics"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=14014)
    parser.add_argument("--cells", type=int, default=512)
    parser.add_argument("--genes", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.cells < 20 or args.genes < 8 or args.repeats < 1:
        parser.error("Use >=20 cells, >=8 genes, and >=1 repeats")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numba_cache").mkdir(exist_ok=True)
    for path in (H5AD, CACHE, CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pp = PostProcessConfig(**config["stage2_postprocess"])
    print("Loading Figure 4 kinetics cache", flush=True)
    with CACHE.open("rb") as handle:
        cache = pickle.load(handle)
    if cache.get("cache_key") != "59e79e596f5d":
        raise ValueError("Unexpected Stage 1 cache key; current-code replay not verified")
    # Read the public processed h5ad independently and verify cache reference arrays.
    adata = ad.read_h5ad(H5AD, backed="r")
    try:
        if adata.shape != cache["xr_s"].shape:
            raise ValueError("Cache and h5ad reference shapes differ")
        probe_rows = np.array([0, 73, adata.n_obs - 1])
        probe_cols = np.array([0, 113, adata.n_vars - 1])
        for layer, key in (("spliced", "xr_s"), ("unspliced", "xr_u")):
            selected = adata.layers[layer][probe_rows, :]
            actual = (selected.toarray() if sparse.issparse(selected) else np.asarray(selected))[:, probe_cols]
            if not np.array_equal(actual, cache[key][np.ix_(probe_rows, probe_cols)]):
                raise ValueError(f"{layer} cache does not match processed h5ad")
    finally:
        adata.file.close()
    print("Replaying Stage 2 observation noise", flush=True)
    sim_s, sim_u = _apply_count_noise(cache["xs_s_base"], cache["xs_u_base"], pp)
    sim_s, sim_u = _add_noise_genes(sim_s, sim_u, cache["pseudotime"], pp)
    real_s, real_u = cache["xr_s"], cache["xr_u"]
    if min(real_s.shape[0], sim_s.shape[0]) < args.cells or min(real_s.shape[1], sim_s.shape[1]) < args.genes:
        parser.error("Sample size exceeds a matrix dimension")
    real_mean = np.asarray((real_s.mean(axis=0) + real_u.mean(axis=0)), dtype=float)
    sim_mean = np.asarray((sim_s.mean(axis=0) + sim_u.mean(axis=0)), dtype=float)
    real_library = real_s.sum(axis=1) + real_u.sum(axis=1)
    sim_library = sim_s.sum(axis=1) + sim_u.sum(axis=1)
    rng = np.random.default_rng(args.seed)
    runs, draws, calibration, joint_runs = [], [], [], []
    for rep in range(args.repeats):
        ri = rng.choice(real_s.shape[0], args.cells, replace=False)
        si = rng.choice(sim_s.shape[0], args.cells, replace=False)
        rg, sg = rank_matched_indices(real_mean, sim_mean, args.genes, rng)
        rs, ru = real_s[np.ix_(ri, rg)].astype(float), real_u[np.ix_(ri, rg)].astype(float)
        ss, su = sim_s[np.ix_(si, sg)].astype(float), sim_u[np.ix_(si, sg)].astype(float)
        real = calculate(rs, ru)
        synth = calculate(ss, su)
        joint_runs.append({"real": joint_us_histogram(rs, ru),
                           "synthetic": joint_us_histogram(ss, su)})
        calibration.append({label: {
            "median_full_library_size": float(np.median(library[indices])),
            "sampled_nonzero_fraction": float(np.mean(s + u > 0)),
            "sampled_mean_count_per_gene_cell": float(np.mean(s + u)),
        } for label, library, indices, s, u in (
            ("real", real_library, ri, rs, ru),
            ("synthetic", sim_library, si, ss, su),
        )})
        runs.append({"real": real, "synthetic": synth})
        draws.append({"real_cell_indices": ri.tolist(), "synthetic_cell_indices": si.tolist(),
                      "real_gene_indices": rg.tolist(), "synthetic_gene_indices": sg.tolist()})
        print(f"Sample {rep + 1}/{args.repeats}", flush=True)
    summary = {}
    for key in runs[0]["real"]:
        per = [summarize(run["real"][key], run["synthetic"][key]) for run in runs]
        if any("failure" in x for x in per):
            summary[key] = {"failure": "at least one subsample had no finite values", "per_repeat": per}
            continue
        delta = np.array([x["median_difference_synthetic_minus_real"] for x in per])
        w1 = np.array([x["wasserstein_1"] for x in per])
        summary[key] = {**per[0], "subsample_median_difference_range": [float(delta.min()), float(delta.max())],
                        "subsample_wasserstein_range": [float(w1.min()), float(w1.max())],
                        "per_repeat": per}
    joint_per_repeat = []
    for joint in joint_runs:
        real_joint, sim_joint = joint["real"], joint["synthetic"]
        tv = 0.5 * np.abs(real_joint["probability"] - sim_joint["probability"]).sum()
        real_nonzero = real_joint["probability"].copy()
        sim_nonzero = sim_joint["probability"].copy()
        real_nonzero[0, 0] = 0
        sim_nonzero[0, 0] = 0
        real_nonzero /= real_nonzero.sum()
        sim_nonzero /= sim_nonzero.sum()
        nonzero_tv = 0.5 * np.abs(real_nonzero - sim_nonzero).sum()
        joint_per_repeat.append({
            "tv": float(tv),
            "nonzero_tv": float(nonzero_tv),
            "real_joint_zero_mass": real_joint["joint_zero_mass"],
            "synthetic_joint_zero_mass": sim_joint["joint_zero_mass"],
            "real_u_zero_mass": real_joint["u_zero_mass"],
            "synthetic_u_zero_mass": sim_joint["u_zero_mass"],
            "real_s_zero_mass": real_joint["s_zero_mass"],
            "synthetic_s_zero_mass": sim_joint["s_zero_mass"],
            "real_tail_fraction": real_joint["tail_fraction"],
            "synthetic_tail_fraction": sim_joint["tail_fraction"],
            "pairs_per_group": real_joint["pairs"],
        })
    summary["joint_us_log1p"] = {
        "first_draw_tv": joint_per_repeat[0]["tv"],
        "first_draw_nonzero_tv": joint_per_repeat[0]["nonzero_tv"],
        "subsample_tv_range": [min(x["tv"] for x in joint_per_repeat),
                                max(x["tv"] for x in joint_per_repeat)],
        "subsample_nonzero_tv_range": [min(x["nonzero_tv"] for x in joint_per_repeat),
                                        max(x["nonzero_tv"] for x in joint_per_repeat)],
        "bin_edges_log1p": np.linspace(0, 6, 13).tolist(),
        "first_draw_bin_probability": {
            label: joint_runs[0][label]["probability"].tolist()
            for label in ("real", "synthetic")
        },
        "per_repeat": joint_per_repeat,
    }
    old_summary_path = OUT / "summary.json"
    old_manifest_path = OUT / "manifest.json"
    if old_summary_path.exists() and old_manifest_path.exists():
        old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        same_design = (old_manifest.get("seed") == args.seed
                       and old_manifest.get("cells_per_group") == args.cells
                       and old_manifest.get("genes_per_group") == args.genes
                       and old_manifest.get("repeats") == args.repeats
                       and old_manifest.get("stage1_cache_key") == cache["cache_key"])
        if same_design:
            changed = [key for key, value in old_summary.items()
                       if key != "joint_us_log1p" and summary.get(key) != value]
            if changed or old_manifest.get("draws") != draws:
                raise RuntimeError(f"Existing metrics or draws changed: {changed}; refusing to overwrite")
    manifest = {
        "scope": "Held-out metrics on same reference; not a held-out dataset or independent validation",
        "seed": args.seed, "cells_per_group": args.cells, "genes_per_group": args.genes,
        "repeats": args.repeats, "sampling": "Independent cells; uniform abundance-quantile matched gene ranks, no identity pairing",
        "count_scale": "Raw counts for raw u-s r and conditional Fano; log1p(total or component counts / cell total * 1e4) for other metrics",
        "conditional_dispersion": "Mean of per-gene sample variance/mean across library-size quartiles with bin mean >=0.1",
        "distance": "Euclidean distance divided by sqrt(number of genes)",
        "joint_us_log1p": "Within each group, flatten the same sampled cell x gene spliced/unspliced count entries into (log1p(u), log1p(s)) pairs. Common fixed 12 x 12 bins over [0,6) on both axes; values >=6 pooled in final bin. TV = 0.5 * sum absolute bin-probability differences. Nonzero TV excludes the (0,0) bin and renormalizes each group separately (positive counts have log1p >=0.693, so the first bin contains only zero-zero pairs). Both are raw-count-, bin- and sample-sensitive descriptive distances; not CIs or independent-cell/gene tests.",
        "uncertainty": "Min/max across deterministic subsamples; sensitivity only, not confidence intervals. Gene pairs and cell pairs are dependent.",
        "stage1_cache_key": cache["cache_key"], "stage2_postprocess": dataclasses.asdict(pp),
        "files_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (H5AD, CACHE, CONFIG, Path(__file__))},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__, "anndata": ad.__version__},
        "shapes": {"real": list(real_s.shape), "synthetic": list(sim_s.shape)},
        "calibration_by_draw": calibration,
        "draws": draws,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    draw_panels(runs[0], summary, joint_runs[0], OUT / "higher_order_comparison.pdf")
    lines = ["# R1 Major 14: pancreas higher-order count statistics", "",
             "Held-out metrics, **not a held-out dataset**. The same pancreas reference guided simulator topology and marginal calibration. No gene identity correspondence exists between synthetic and real matrices; genes were abundance-quantile matched. These are descriptive comparisons, not independent generalization estimates.", "",
             f"Current-code Figure 4 Stage 2 replay from cache; {args.repeats} deterministic draws of {args.cells} cells and {args.genes} genes per group. Full provenance and exact indices: `manifest.json`.", "",
             "| Metric | Real median | Synthetic median | Synthetic - real | W1 | Across-draw difference range |", "|---|---:|---:|---:|---:|---:|"]
    for key, value in summary.items():
        if key == "joint_us_log1p":
            continue
        if "failure" in value:
            lines.append(f"| {key} | failure: {value['failure']} | | | | |")
        else:
            lo, hi = value["subsample_median_difference_range"]
            lines.append(f"| {key} | {value['real_median']:.3g} | {value['synthetic_median']:.3g} | {value['median_difference_synthetic_minus_real']:.3g} | {value['wasserstein_1']:.3g} | [{lo:.3g}, {hi:.3g}] |")
    first_joint = joint_per_repeat[0]
    tv_lo, tv_hi = summary["joint_us_log1p"]["subsample_tv_range"]
    nz_lo, nz_hi = summary["joint_us_log1p"]["subsample_nonzero_tv_range"]
    lines += ["", "W1 is the one-dimensional Wasserstein distance. Ranges reflect sample sensitivity only; correlated gene/cell pairs preclude naive pair-bootstrap confidence intervals. Raw-count conditional Fano conditions on library-size quartiles but may retain mean-expression and cell-state confounding. Log-normalization addresses cell-depth scale; gene-z distances probe shape after marginal scale removal. Neither is a fitted calibration correction.", "",
              "## Joint spliced/unspliced counts", "",
              f"The two PDF heatmaps use the same first prespecified draw: {first_joint['pairs_per_group']:,} same-cell/same-gene (log1p(u), log1p(s)) pairs per group. Their coarse common-bin total variation is **{first_joint['tv']:.3f}**; across the same eight draws it ranges [{tv_lo:.3f}, {tv_hi:.3f}]. Excluding (0,0) and renormalizing the remaining pairs gives **{first_joint['nonzero_tv']:.3f}** (range [{nz_lo:.3f}, {nz_hi:.3f}]), exposing shape differences otherwise diluted by zeros. The fixed 12 x 12 grid spans 0-6 on each axis in 0.5 increments, with values >=6 pooled into the last bin. Joint zero mass is {first_joint['real_joint_zero_mass']:.3f} real versus {first_joint['synthetic_joint_zero_mass']:.3f} synthetic; marginal u-zero mass is {first_joint['real_u_zero_mass']:.3f} versus {first_joint['synthetic_u_zero_mass']:.3f}, and s-zero mass {first_joint['real_s_zero_mass']:.3f} versus {first_joint['synthetic_s_zero_mass']:.3f}. Tail pooling affects {first_joint['real_tail_fraction']:.4f} and {first_joint['synthetic_tail_fraction']:.4f} of pairs, respectively.", "",
              "TV is descriptive and depends on the sampled cells/genes, raw-count scale, shared zero mass and chosen bins. The pairs are not independent observations and the across-draw range is not a confidence interval. No gene identities or cells are paired between real and synthetic groups.", "",
              "## Scale check", "",
              f"First draw full-library median: real {calibration[0]['real']['median_full_library_size']:.0f}, synthetic {calibration[0]['synthetic']['median_full_library_size']:.0f} counts/cell. Sampled nonzero fraction: real {calibration[0]['real']['sampled_nonzero_fraction']:.3f}, synthetic {calibration[0]['synthetic']['sampled_nonzero_fraction']:.3f}. Sampled mean count per gene-cell: real {calibration[0]['real']['sampled_mean_count_per_gene_cell']:.3g}, synthetic {calibration[0]['synthetic']['sampled_mean_count_per_gene_cell']:.3g}. These differences can shift raw covariance, Fano and distance distributions; normalized and gene-z views are sensitivity analyses, not proof of biological calibration.", "",
              "Failures/limitations: no final metric failures; no independent held-out biological sample, no matched gene identities, no paired-cell comparison, no inferential p-values. See `summary.json` for finite sample counts and full per-draw effects."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "report.md", flush=True)


if __name__ == "__main__":
    main()
