"""Paired integration-step audit of the active RNA-velocity metrics.

The finest simulated path is a numerical comparator, not analytical truth.
Run with .venv-revision/Scripts/python.exe from the code repository root.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from experiments.figures.fig_coord_velocity_consistency.lib import core
from experiments.common.velocity_metrics import cosine
from experiments.common.coordinate_simulation import infer_and_smooth
from experiments.common.native_projection import fixed_distances, native_project, region_labels

OUT = ROOT / "results" / "revision_main_metric_timestep"
TOPOLOGIES = ("cycle", "bifurcation", "cycle_bifurcation")
SEEDS = (11, 22, 33)
STEPS = (0.01, 0.005, 0.0025)
FINE_STEP = STEPS[-1]
CFG = dict(duration=12.0, burnin=2.0, sampling_interval=0.2,
           diffusion=0.006, trajectories=48, genes=160, observations=600,
           neighbors=20)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_data(topology, seed, step):
    core.configure(pipeline={"seed": seed})
    streams = np.random.SeedSequence(seed).generate_state(5)
    np.random.seed(int(streams[0]))
    system = core.TOPOLOGY_BUILDERS[topology]()
    synth = core.setup_synthesizer(system, CFG["genes"])
    if topology == "cycle_bifurcation":
        z = core.init_cycle_rim_cells(CFG["trajectories"], radius=1., noise=.05,
                                      seed=int(streams[1]))
    else:
        z = core.initialize_cells(topology, CFG["trajectories"], system,
                                  int(streams[1]))
    z = np.asarray(z, dtype=float)
    beta, gamma = synth.beta[None, :], synth.gamma[None, :]
    alpha = synth._compute_transcription_rate(z, scale=core.ALPHA_SCALE)
    u, s = alpha / beta, alpha / gamma

    fine_n = int(round(CFG["duration"] / FINE_STEP))
    ratio = int(round(step / FINE_STEP))
    assert np.isclose(ratio * FINE_STEP, step) and fine_n % ratio == 0
    increments = np.random.default_rng(int(streams[2])).standard_normal(
        (fine_n, CFG["trajectories"], z.shape[1])) * np.sqrt(FINE_STEP)
    increments = increments.reshape(-1, ratio, CFG["trajectories"], z.shape[1]).sum(axis=1)
    save_stride = int(round(CFG["sampling_interval"] / step))
    burnin_step = int(round(CFG["burnin"] / step))
    assert np.isclose(save_stride * step, CFG["sampling_interval"])
    snapshots = []
    negatives = 0
    for i, dw in enumerate(increments, start=1):
        alpha = synth._compute_transcription_rate(z, current_s=s, scale=core.ALPHA_SCALE)
        un = u + step * (alpha - beta * u)
        sn = s + step * (beta * u - gamma * s)
        negatives += int((un < 0).sum() + (sn < 0).sum())
        u, s = np.maximum(un, 0), np.maximum(sn, 0)
        z = z + step * system.get_drift(z) + np.sqrt(2 * CFG["diffusion"]) * dw
        if i >= burnin_step and i % save_stride == 0:
            snapshots.append((z.copy(), s.copy(), u.copy(),
                              (beta * u - gamma * s).copy(),
                              np.arange(len(z)), np.full(len(z), i * step)))
    arrays = [np.concatenate([item[j] for item in snapshots]) for j in range(6)]
    selection = np.random.default_rng(int(streams[3])).choice(
        len(arrays[0]), size=CFG["observations"], replace=False)
    z, s, u, v, trajectory, time = [a[selection] for a in arrays]
    np.random.seed(int(streams[4]))
    capture = synth._draw_capture_factor(s)
    sc = synth._sample_counts(s, capture_factor=capture)
    uc = synth._sample_counts(u, capture_factor=capture,
                              capture_scale=synth.unspliced_capture_ratio)
    assert np.max(np.abs(v - (beta * u - gamma * s))) < 1e-8
    assert np.isfinite(z).all() and np.isfinite(s).all() and np.isfinite(u).all()
    return dict(z=z, s=s, u=u, v=v, trajectory=trajectory, time=time,
                sc=sc, uc=uc, capture=capture,
                capture_ratio=synth.unspliced_capture_ratio,
                beta=synth.beta, gamma=synth.gamma, streams=streams,
                negative_fraction=negatives / (2 * len(increments) *
                                               CFG["trajectories"] * CFG["genes"]))


def evaluate(data, topology, seed, step):
    from umap import UMAP

    pred, truth, state, observed, indices, weights = infer_and_smooth(data, CFG)
    loadings = PCA(n_components=2, svd_solver="full").fit(observed).components_
    cp = cosine(pred @ loadings.T, truth @ loadings.T)
    cg = cosine(pred, truth)
    layout = UMAP(n_neighbors=CFG["neighbors"], n_components=2, min_dist=.3,
                  random_state=seed, n_jobs=1).fit_transform(np.log1p(observed))
    projection_input = dict(neighbor_indices=indices, sc=data["sc"],
                            capture=data["capture"], observed_smoothed=observed,
                            umap=layout)
    graph = fixed_distances(projection_input)
    cn = cosine(native_project(projection_input, pred, graph),
                native_project(projection_input, truth, graph))
    valid = np.isfinite(cp) & np.isfinite(cg)
    native_valid = np.isfinite(cn) & np.isfinite(cg)
    if valid.sum() < 0.8 * len(valid):
        raise ValueError(f"Too few valid vectors: {topology}, {seed}, {step}")
    if native_valid.sum() < 0.8 * len(valid):
        raise ValueError(f"Too few native projected vectors: {topology}, {seed}, {step}")
    delta = cp - cg
    native_delta = cn - cg
    mask = valid & (cp > .7) & (cg < .2)
    native_mask = native_valid & (cn > .7) & (cg < .2)
    cells = pd.DataFrame(dict(topology=topology, seed=seed, dt=step,
                              cell=np.arange(len(cg)), trajectory=data["trajectory"],
                              simulation_time=data["time"],
                              region=region_labels(topology, data["z"]),
                              gene_cosine=cg, pca_cosine=cp, native_cosine=cn,
                              delta_cosine=delta, native_delta_cosine=native_delta,
                              masking_label=mask, native_masking_label=native_mask,
                              valid=valid, native_valid=native_valid,
                              pred_norm=np.linalg.norm(pred, axis=1),
                              truth_norm=np.linalg.norm(truth, axis=1)))
    return cells


def summarize(cells):
    rows = []
    for (topology, seed, dt, region), item in cells.groupby(
            ["topology", "seed", "dt", "region"]):
            good = item[item.valid]
            native = item[item.native_valid]
            rows.append(dict(topology=topology, seed=seed, dt=dt, region=region,
                             n_cells=len(item), n_valid=len(good),
                             n_native_valid=len(native),
                             gene_cosine=good.gene_cosine.mean(),
                             pca_cosine=good.pca_cosine.mean(),
                             delta_cosine=good.delta_cosine.mean(),
                             masking_fraction=good.masking_label.mean(),
                             native_cosine=native.native_cosine.mean(),
                             native_delta_cosine=native.native_delta_cosine.mean(),
                             native_masking_fraction=native.native_masking_label.mean()))
    for (topology, seed, dt), item in cells.groupby(["topology", "seed", "dt"]):
        good = item[item.valid]
        native = item[item.native_valid]
        rows.append(dict(topology=topology, seed=seed, dt=dt, region="all",
                         n_cells=len(item), n_valid=len(good),
                         n_native_valid=len(native),
                         gene_cosine=good.gene_cosine.mean(),
                         pca_cosine=good.pca_cosine.mean(),
                         delta_cosine=good.delta_cosine.mean(),
                         masking_fraction=good.masking_label.mean(),
                         native_cosine=native.native_cosine.mean(),
                         native_delta_cosine=native.native_delta_cosine.mean(),
                         native_masking_fraction=native.native_masking_label.mean()))
    return pd.DataFrame(rows)


def plot(summary):
    frame = summary[summary.region == "all"]
    metrics = ("gene_cosine", "pca_cosine", "native_cosine",
               "delta_cosine", "native_delta_cosine", "native_masking_fraction")
    titles = ("Gene-space cosine", "PCA cosine", "Native UMAP cosine",
              "PCA minus gene", "Native UMAP minus gene", "Native masking-label fraction")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
    colors = ("#247d8f", "#bc6a35", "#55713d")
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for topology, color in zip(TOPOLOGIES, colors):
            part = frame[frame.topology == topology]
            mean = part.groupby("dt")[metric].mean().reindex(STEPS)
            ax.plot(STEPS, mean, marker="o", color=color, label=topology.replace("_", "--"))
            for seed in SEEDS:
                one = part[part.seed == seed].set_index("dt").reindex(STEPS)
                ax.plot(STEPS, one[metric], color=color, alpha=.25, linewidth=.7)
        ax.set_title(title)
        ax.set_xlabel("Integration step")
        ax.set_xticks(STEPS)
        ax.invert_xaxis()
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUT / "main_metric_timestep.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_cells, run_rows = [], []
    for topology in TOPOLOGIES:
        for seed in SEEDS:
            times = None
            trajectories = None
            for step in STEPS:
                data = paired_data(topology, seed, step)
                if times is None:
                    times, trajectories = data["time"], data["trajectory"]
                else:
                    assert np.allclose(data["time"], times)
                    assert np.array_equal(data["trajectory"], trajectories)
                cells = evaluate(data, topology, seed, step)
                all_cells.append(cells)
                run_rows.append(dict(topology=topology, seed=seed, dt=step,
                                     negative_fraction=data["negative_fraction"],
                                     valid_cells=int(cells.valid.sum()),
                                     native_valid_cells=int(cells.native_valid.sum())))
                print(f"{topology} seed={seed} dt={step}: "
                      f"valid={cells.valid.sum()} native={cells.native_valid.sum()} "
                      f"pca_delta={cells.delta_cosine.mean():.4f} "
                      f"native_delta={cells.native_delta_cosine.mean():.4f}",
                      flush=True)
    frame = pd.concat(all_cells, ignore_index=True)
    frame.to_csv(OUT / "per_cell.csv", index=False)
    summary = summarize(frame)
    summary.to_csv(OUT / "per_seed_region.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUT / "runs.csv", index=False)
    plot(summary)
    (OUT / "provenance.json").write_text(json.dumps(dict(
        config=CFG, topologies=TOPOLOGIES, seeds=SEEDS, steps=STEPS,
        source_sha256=sha256(Path(__file__)), core_sha256=sha256(Path(core.__file__)),
        coupling="coarse Brownian increments sum shared finest increments",
        coordinate="oracle-corrected abundance with fixed 20-neighbor-plus-self moments",
        reference="stored-state beta*u-gamma*s, smoothed with the same matrix",
        projections="dataset-fitted 2D PCA; native scVelo UMAP with matched key settings",
        masking="projected cosine > 0.7 and gene cosine < 0.2 among paired valid cells",
        versions={name: importlib.metadata.version(name) for name in
                  ("numpy", "pandas", "scvelo", "scikit-learn", "matplotlib")}),
        indent=2), encoding="utf-8")
    assert len(frame) == len(TOPOLOGIES) * len(SEEDS) * len(STEPS) * CFG["observations"]


if __name__ == "__main__":
    main()
