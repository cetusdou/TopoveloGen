"""R1 Major 8/9/12: paired, cell-level projection audit of archived simulations.

Run from the repository root with .venv-revision/Scripts/python.exe
experiments/revision_projection_closure.py. No simulation or velocity fitting is rerun.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from experiments.common.projection import positive_cosine_neighbor_projection as kernel_project

ARCHIVE = ROOT / "results" / "revision_coordinate_projection"
OUT = ROOT / "results" / "revision_projection_closure"
TOPOLOGIES = ("cycle", "bifurcation", "cycle_bifurcation")
SEEDS = (11, 22, 33, 44, 55)
FLOORS = (1e-8, 1e-6, 1e-4, 1e-2)
HIGHS = (.5, .7, .9)
LOWS = (0., .2, .4)
NATIVE_SETTINGS = dict(xkey="Ms", n_neighbors=20, n_recurse_neighbors=1,
                       sqrt_transform=False, approx=False, mode_neighbors="distances",
                       n_jobs=1, backend="threading", show_progress_bar=False)
EMBEDDING_SETTINGS = dict(basis="umap", scale=10, self_transitions=True,
                          use_negative_cosines=True, direct_pca_projection=False,
                          retain_scale=False, autoscale=False, all_comps=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a, b, floor):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    pa, pb = na[na > 0], nb[nb > 0]
    fa = floor * np.median(pa) if len(pa) else np.inf
    fb = floor * np.median(pb) if len(pb) else np.inf
    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (na > fa) & (nb > fb)
    result = np.full(len(a), np.nan)
    result[valid] = np.einsum("ij,ij->i", a[valid], b[valid]) / (na[valid] * nb[valid])
    return np.clip(result, -1, 1)


def region_labels(topology, z):
    x, y = z[:, 0], z[:, 1]
    if topology == "cycle":
        angle = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
        return np.array(["quadrant_1", "quadrant_2", "quadrant_3", "quadrant_4"])[
            np.floor(angle / (np.pi / 2)).astype(int)]
    if topology == "bifurcation":
        return np.where(x < 0, "trunk_x_lt_0", np.where(y >= 0, "upper_x_ge_0", "lower_x_ge_0"))
    return np.where(x < 1, "cycle_x_lt_1", np.where(y >= 0, "upper_x_ge_1", "lower_x_ge_1"))


def fixed_distances(d):
    indices = d["neighbor_indices"]
    state = np.log1p(d["sc"] / d["capture"])
    distance = np.linalg.norm(state[indices] - state[:, None, :], axis=2)
    distance = np.maximum(distance, np.finfo(np.float32).eps)
    n, k = indices.shape
    graph = sparse.csr_matrix((distance.ravel(),
                               (np.repeat(np.arange(n), k), indices.ravel())), shape=(n, n))
    if np.any(np.diff(graph.indptr) != k):
        raise ValueError("Archived neighbor indices are not unique per cell")
    return graph


def native_project(d, velocity, distances):
    import anndata as ad
    import scvelo as scv

    adata = ad.AnnData(X=d["observed_smoothed"].astype(np.float32))
    adata.layers["Ms"] = d["observed_smoothed"].astype(np.float32)
    adata.layers["velocity"] = velocity.astype(np.float32)
    adata.var["velocity_genes"] = np.ones(adata.n_vars, dtype=bool)
    adata.obsm["X_umap"] = d["umap"].copy()
    adata.obsp["distances"] = distances.copy()
    adata.obsp["connectivities"] = distances.copy().astype(bool).astype(float)
    adata.uns["neighbors"] = {"distances_key": "distances",
                               "connectivities_key": "connectivities",
                               "params": {"n_neighbors": 20, "method": "archived_fixed_knn"}}
    scv.tl.velocity_graph(adata, vkey="velocity", **NATIVE_SETTINGS)
    scv.tl.velocity_embedding(adata, vkey="velocity", **EMBEDDING_SETTINGS)
    result = np.asarray(adata.obsm["velocity_umap"], dtype=float)
    if result.shape != (adata.n_obs, 2) or not np.isfinite(result).all():
        raise ValueError("Native embedding is nonfinite or has unexpected shape")
    return result


def metric_row(topology, seed, region, operator, score, baseline, mask):
    valid = mask & np.isfinite(score) & np.isfinite(baseline)
    n = int(valid.sum())
    return dict(topology=topology, seed=seed, region=region, operator=operator,
                n_cells=int(mask.sum()), n_valid=n, excluded_fraction=1 - n / int(mask.sum()),
                mean_cosine=float(np.mean(score[valid])) if n else np.nan,
                median_cosine=float(np.median(score[valid])) if n else np.nan,
                mean_delta_vs_gene=float(np.mean(score[valid] - baseline[valid])) if n else np.nan,
                median_delta_vs_gene=float(np.median(score[valid] - baseline[valid])) if n else np.nan)


def common_score_cells(scores):
    return np.logical_and.reduce([np.isfinite(score) for score in scores.values()])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ARCHIVE / "provenance.json").read_text())
    provenance = dict(archive_configuration=metadata, archive_provenance_sha256=sha256(ARCHIVE / "provenance.json"),
                      script_sha256=sha256(Path(__file__)), native_graph_settings=NATIVE_SETTINGS,
                      native_embedding_settings=EMBEDDING_SETTINGS,
                      fixed_graph="Archived neighbor_indices; distances in log1p(sc/capture) space, no refitting",
                      gene_mask="all 160 archived genes, identical for prediction and reference",
                      main_comparison_cells="intersection of gene, PCA, native UMAP, and positive-cosine UMAP valid cells within each dataset",
                      pca="pred/gt_smoothed @ archived pca_loadings.T (no recentering)",
                      umap_kernel="positive gene-space cosine to archived neighbor state displacements; row normalize; weighted raw UMAP displacement; zero if no positive weight",
                      regions={"cycle": "latent z angle quadrants [0,pi/2), ...",
                               "bifurcation": "z_x < 0 trunk; else sign(z_y) upper/lower",
                               "cycle_bifurcation": "z_x < 1 cycle; else sign(z_y) upper/lower"},
                      norm_floors=FLOORS, threshold_high=HIGHS, threshold_low=LOWS,
                      threshold_event="projected cosine > high AND gene cosine < low, among paired valid cells",
                      replication_unit="dataset seed; cell scores are descriptive, not independent replicates",
                      versions={p: importlib.metadata.version(p) for p in
                                ("numpy", "scipy", "pandas", "matplotlib", "scvelo", "anndata")},
                      archives={})
    all_cells, all_rows, sensitivity, failures = [], [], [], []
    for topology in TOPOLOGIES:
        for seed in SEEDS:
            name = f"{topology}_seed{seed}"
            folder = ARCHIVE / name
            paths = {f: folder / f for f in ("data.npz", "smoothing_weights.npz")}
            provenance["archives"][name] = {f: sha256(p) for f, p in paths.items()}
            with np.load(paths["data.npz"]) as archive:
                d = {key: archive[key] for key in archive.files}
            weights = sparse.load_npz(paths["smoothing_weights.npz"])
            if weights.shape != (600, 600) or d["pred"].shape != d["gt_smoothed"].shape or d["pred"].shape != (600, 160):
                raise ValueError(f"Archive shape mismatch: {name}")
            for stored, source in (("gt_smoothed", d["v"]), ("state_smoothed", d["s"]),
                                   ("observed_smoothed", d["sc"] / d["capture"])):
                if not np.allclose(weights @ source, d[stored], rtol=1e-7, atol=1e-8):
                    raise ValueError(f"Archived smoothing weights do not reproduce {stored}: {name}")
            pred, gt = d["pred"], d["gt_smoothed"]
            pairs = {"pca_linear": (pred @ d["pca_loadings"].T, gt @ d["pca_loadings"].T),
                     "umap_positive_cosine_kernel":
                         (kernel_project(pred, d["observed_smoothed"], d["umap"], d["neighbor_indices"]),
                          kernel_project(gt, d["observed_smoothed"], d["umap"], d["neighbor_indices"]))}
            try:
                distances = fixed_distances(d)
                pairs["scvelo_native_umap"] = (native_project(d, pred, distances),
                                                 native_project(d, gt, distances))
                print(f"{name}: native scVelo succeeded", flush=True)
            except Exception as exc:
                failures.append(dict(dataset=name, exception_type=type(exc).__name__,
                                     message=str(exc), traceback=traceback.format_exc()))
                print(f"{name}: native scVelo FAILED: {type(exc).__name__}: {exc}", flush=True)
            regions = region_labels(topology, d["z"])
            scores = {"gene": cosine(pred, gt, 1e-6)}
            scores.update({key: cosine(*pair, 1e-6) for key, pair in pairs.items()})
            if len(scores) != 4:
                raise ValueError(f"Missing projection operator for common-cell comparison: {name}")
            common_valid = common_score_cells(scores)
            common_scores = {key: np.where(common_valid, score, np.nan)
                             for key, score in scores.items()}
            cell = pd.DataFrame(dict(topology=topology, seed=seed, cell=np.arange(len(pred)),
                                     region=regions, trajectory=d["trajectory"],
                                     simulation_time=d["time"], common_valid=common_valid,
                                     **{f"cosine_{k}": v for k, v in scores.items()}))
            for key in pairs:
                cell[f"delta_{key}_vs_gene"] = common_scores[key] - common_scores["gene"]
            all_cells.append(cell)
            for region in ("all", *np.unique(regions)):
                mask = np.ones(len(pred), bool) if region == "all" else regions == region
                for key, score in common_scores.items():
                    all_rows.append(metric_row(topology, seed, region, key, score,
                                               common_scores["gene"], mask))
            for floor in FLOORS:
                base = cosine(pred, gt, floor)
                for key, pair in pairs.items():
                    projected = cosine(*pair, floor)
                    valid = np.isfinite(base) & np.isfinite(projected)
                    sensitivity.append(dict(topology=topology, seed=seed, operator=key, kind="norm_floor",
                                            floor=floor, high=np.nan, low=np.nan, n_valid=int(valid.sum()),
                                            eligible_fraction=float(valid.mean()),
                                            mean_cosine=float(np.mean(projected[valid])) if valid.any() else np.nan,
                                            mean_delta_vs_gene=float(np.mean(projected[valid] - base[valid])) if valid.any() else np.nan,
                                            event_fraction=np.nan))
            base = scores["gene"]
            for key in pairs:
                projected = scores[key]
                valid = np.isfinite(base) & np.isfinite(projected)
                for high in HIGHS:
                    for low in LOWS:
                        sensitivity.append(dict(topology=topology, seed=seed, operator=key, kind="threshold",
                                                floor=1e-6, high=high, low=low, n_valid=int(valid.sum()),
                                                eligible_fraction=float(valid.mean()), mean_cosine=np.nan,
                                                mean_delta_vs_gene=np.nan,
                                                event_fraction=float(np.mean((projected[valid] > high) & (base[valid] < low))) if valid.any() else np.nan))
    cells = pd.concat(all_cells, ignore_index=True)
    rows = pd.DataFrame(all_rows)
    sense = pd.DataFrame(sensitivity)
    cells.to_csv(OUT / "per_cell.csv", index=False)
    rows.to_csv(OUT / "seed_region_metrics.csv", index=False)
    rows[rows.region == "all"].to_csv(OUT / "seed_metrics.csv", index=False)
    sense.to_csv(OUT / "sensitivity.csv", index=False)
    summary = []
    for (topology, region, operator), part in rows.groupby(["topology", "region", "operator"]):
        values = part.mean_delta_vs_gene.to_numpy()
        half = stats.t.ppf(.975, len(values)-1) * stats.sem(values) if len(values) > 1 else np.nan
        summary.append(dict(topology=topology, region=region, operator=operator, seeds=len(values),
                            mean_cosine=float(part.mean_cosine.mean()), mean_delta_vs_gene=float(np.mean(values)),
                            delta_ci95_low=float(np.mean(values)-half), delta_ci95_high=float(np.mean(values)+half),
                            mean_excluded_fraction=float(part.excluded_fraction.mean())))
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    (OUT / "native_failures.json").write_text(json.dumps(failures, indent=2))
    provenance["native_failures"] = len(failures)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, layout="constrained")
    colors = {"pca_linear": "#178078", "umap_positive_cosine_kernel": "#B64F5A",
              "scvelo_native_umap": "#5263AA"}
    for ax, topology in zip(axes, TOPOLOGIES):
        part = rows[(rows.topology == topology) & (rows.region == "all")]
        for i, (operator, group) in enumerate(part.groupby("operator")):
            if operator == "gene":
                continue
            values = group.mean_delta_vs_gene.to_numpy()
            half = stats.t.ppf(.975, len(values)-1) * stats.sem(values) if len(values) > 1 else 0
            ax.errorbar(i, values.mean(), yerr=half, fmt="o", capsize=3,
                        color=colors[operator], label=operator.replace("_", " "))
        ax.axhline(0, color="0.5", lw=.8)
        ax.set_title(topology.replace("_", " "))
        ax.set_xticks([])
        ax.set_ylim(-1, 1)
    axes[0].set_ylabel("Paired mean cosine change vs gene space\n(95% seed-level t interval)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.06), frameon=False)
    fig.savefig(OUT / "projection_closure.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Finished {len(all_cells)} archives; native failures={len(failures)}; {OUT}", flush=True)


if __name__ == "__main__":
    main()
