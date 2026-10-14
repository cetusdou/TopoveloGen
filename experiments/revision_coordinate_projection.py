"""Supplementary simulated-data audit; not a reproduction of the old figures.

Known capture factors are removed before inference (oracle measurement control).
Both vectors are pushed forward for coordinate comparisons. A separate GT-only
diagnostic deliberately measures the effect of a coordinate mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from experiments.figures.fig_coord_velocity_consistency.lib import core


def cosine(a, b, relative_floor=1e-6):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    fa = relative_floor * np.median(na[na > 0]) if np.any(na > 0) else np.inf
    fb = relative_floor * np.median(nb[nb > 0]) if np.any(nb > 0) else np.inf
    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (na > fa) & (nb > fb)
    out = np.full(len(a), np.nan)
    out[valid] = np.einsum("ij,ij->i", a[valid] / na[valid, None], b[valid] / nb[valid, None])
    return np.clip(out, -1, 1)


def pushforward(s, v, target=10000.):
    total = s.sum(1, keepdims=True)
    vp = target * (v - s * v.sum(1, keepdims=True) / total) / total
    return vp, vp / (1 + target * s / total)


def simulate(cfg, topology, seed):
    core.configure(pipeline={"seed": seed})
    streams = np.random.SeedSequence(seed).generate_state(5)
    np.random.seed(int(streams[0]))
    system = core.TOPOLOGY_BUILDERS[topology]()
    synth = core.setup_synthesizer(system, cfg["genes"], cfg.get("signal_reference_genes"))
    rng = np.random.default_rng(int(streams[2]))
    if topology == "cycle_bifurcation":
        z = core.init_cycle_rim_cells(cfg["trajectories"], radius=1., noise=.05, seed=int(streams[1]))
    else:
        z = core.initialize_cells(topology, cfg["trajectories"], system, int(streams[1]))
    z = np.asarray(z, dtype=float)
    beta, gamma = synth.beta[None, :], synth.gamma[None, :]
    alpha = synth._compute_transcription_rate(z, scale=core.ALPHA_SCALE)
    u, s = alpha / beta, alpha / gamma
    snapshots = []
    negative = 0
    for step in range(1, cfg["steps"] + 1):
        alpha = synth._compute_transcription_rate(z, current_s=s, scale=core.ALPHA_SCALE)
        un = u + cfg["dt"] * (alpha - beta * u)
        sn = s + cfg["dt"] * (beta * u - gamma * s)
        negative += int((un < 0).sum() + (sn < 0).sum())
        u, s = np.maximum(un, 0), np.maximum(sn, 0)
        z = z + cfg["dt"] * system.get_drift(z) + np.sqrt(2 * cfg["diffusion"] * cfg["dt"]) * rng.standard_normal(z.shape)
        if step >= cfg["burnin_steps"] and step % cfg["save_every"] == 0:
            snapshots.append((z.copy(), s.copy(), u.copy(), (beta*u-gamma*s).copy(),
                              np.arange(len(z)), np.full(len(z), step * cfg["dt"])))
    arrays = [np.concatenate([row[j] for row in snapshots]) for j in range(6)]
    idx = np.random.default_rng(int(streams[3])).choice(len(arrays[0]), size=min(cfg["observations"], len(arrays[0])), replace=False)
    z, s, u, v, trajectory, time = [x[idx] for x in arrays]
    np.random.seed(int(streams[4]))
    capture = synth._draw_capture_factor(s)
    sc = synth._sample_counts(s, capture_factor=capture)
    uc = synth._sample_counts(u, capture_factor=capture, capture_scale=synth.unspliced_capture_ratio)
    modules = synth.modules
    return dict(z=z, s=s, u=u, v=v, trajectory=trajectory, time=time, sc=sc, uc=uc,
                capture=capture, capture_ratio=synth.unspliced_capture_ratio,
                beta=synth.beta, gamma=synth.gamma, streams=streams,
                negative_fraction=negative/(2*cfg["steps"]*cfg["trajectories"]*cfg["genes"]), modules=modules)


def infer_and_smooth(data, cfg):
    import anndata as ad
    import scvelo as scv
    scv.settings.verbosity = 0
    # Known capture correction is an oracle control, not an inferred library size.
    s = data["sc"] / data["capture"]
    u = data["uc"] / data["capture"] / data["capture_ratio"]
    indices = NearestNeighbors(n_neighbors=cfg["neighbors"]).fit(np.log1p(s)).kneighbors(return_distance=False)
    rows = np.repeat(np.arange(len(s)), cfg["neighbors"] + 1)
    cols = np.column_stack([np.arange(len(s)), indices]).ravel()
    w = sparse.csr_matrix((np.full(len(rows), 1/(cfg["neighbors"]+1)), (rows, cols)), shape=(len(s), len(s)))
    ms, mu = np.asarray(w@s), np.asarray(w@u)
    adata = ad.AnnData(X=np.log1p(ms).astype(np.float32))
    adata.layers["spliced"], adata.layers["unspliced"] = s.astype(np.float32), u.astype(np.float32)
    adata.layers["Ms"], adata.layers["Mu"] = ms.astype(np.float32), mu.astype(np.float32)
    scv.tl.velocity(adata, mode="deterministic", use_raw=False, min_r2=-np.inf, min_likelihood=-np.inf)
    pred = np.asarray(adata.layers["velocity"], dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite scVelo output")
    return pred, np.asarray(w@data["v"]), np.asarray(w@data["s"]), ms, indices, w


def kernel_project(v, x, y, indices):
    dx = x[indices] - x[:, None, :]
    den = np.linalg.norm(v, axis=1)[:, None] * np.linalg.norm(dx, axis=2)
    similarity = np.divide(np.einsum("ig,ikg->ik", v, dx), den, out=np.zeros_like(den), where=den>0)
    weights = np.maximum(similarity, 0.)
    totals = weights.sum(1, keepdims=True)
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals>0)
    return np.einsum("ik,ikd->id", weights, y[indices]-y[:, None, :])


def evaluate(data, cfg, topology, seed, dest):
    from umap import UMAP
    pred, gt, state, observed, indices, w = infer_and_smooth(data, cfg)
    pp, pl = pushforward(state, pred)
    gp, gl = pushforward(state, gt)
    scores = {"absolute": cosine(pred, gt), "proportion_both": cosine(pp, gp),
              "log_proportion_both": cosine(pl, gl), "proportion_gt_only": cosine(pred, gp)}
    pca = PCA(n_components=2, svd_solver="full").fit(observed)
    y_pca = pca.transform(observed)
    y_umap = UMAP(n_neighbors=cfg["neighbors"], n_components=2, min_dist=.3,
                  random_state=seed, n_jobs=1).fit_transform(np.log1p(observed))
    projected_pairs = {"pca_linear": (pred@pca.components_.T, gt@pca.components_.T),
                       "umap_kernel": (kernel_project(pred, observed, y_umap, indices), kernel_project(gt, observed, y_umap, indices))}
    for name, pair in projected_pairs.items():
        scores[name] = cosine(*pair)
    frame = pd.DataFrame(scores)
    frame["norm_pred"], frame["norm_gt"] = np.linalg.norm(pred, axis=1), np.linalg.norm(gt, axis=1)
    frame["trajectory"], frame["simulation_time"] = data["trajectory"], data["time"]
    frame.to_csv(dest/"per_cell.csv", index=False)
    np.savez_compressed(dest/"data.npz", **{k:v for k,v in data.items() if isinstance(v,np.ndarray)},
                        pred=pred, gt_smoothed=gt, state_smoothed=state, observed_smoothed=observed,
                        pca=y_pca, pca_loadings=pca.components_, umap=y_umap, neighbor_indices=indices)
    sparse.save_npz(dest/"smoothing_weights.npz", w)
    (dest/"modules.json").write_text(json.dumps(data["modules"], indent=2, default=lambda x: np.asarray(x).tolist()))
    (dest/"measurement.json").write_text(json.dumps({"capture_ratio": data["capture_ratio"], "sampling":core.SYNTH_KW}, indent=2))
    rows, thresholds = [], []
    for name, score in scores.items():
        valid = np.isfinite(score) & np.isfinite(scores["absolute"])
        delta = score[valid]-scores["absolute"][valid]
        rows.append(dict(topology=topology, seed=seed, metric=name, n_valid=int(valid.sum()),
                         excluded_fraction=1-valid.mean(), mean_cosine=float(np.mean(score[valid])),
                         mean_delta=float(np.mean(delta)), median_cosine=float(np.median(score[valid])),
                         negative_fraction=data["negative_fraction"]))
        if name in ("pca_linear", "umap_kernel"):
            for high in (.5,.7,.9):
                for low in (0.,.2,.4):
                    thresholds.append(dict(topology=topology, seed=seed, projection=name, high=high, low=low,
                                           fraction=float(np.mean((score[valid]>high)&(scores["absolute"][valid]<low))), n_valid=int(valid.sum())))
    for floor in (1e-8,1e-6,1e-4,1e-2):
        value = cosine(pred, gt, floor)
        for name, pair in {"gene": (pred,gt), **projected_pairs}.items():
            proj = cosine(*pair, relative_floor=floor)
            valid = np.isfinite(proj) & np.isfinite(value)
            thresholds.append(dict(topology=topology, seed=seed, projection="norm_floor_"+name, high=floor, low=-1.,
                fraction=float(valid.mean()), n_valid=int(valid.sum()), mean_cosine=float(np.mean(proj[valid])),
                mean_delta=float(np.mean(proj[valid]-value[valid])),
                masking_fraction=float(np.mean((proj[valid]>.7)&(value[valid]<.2)))))
    c = data["capture"]
    # Algebraic control on unsmoothed truth and a nonzero deterministic perturbation.
    a, b = data["v"], data["v"] + .01
    discrepancy = float(np.nanmax(np.abs(cosine(a,b,0)-cosine(c*a,c*b,0))))
    (dest/"checks.json").write_text(json.dumps({"positive_scale_max_cosine_difference":discrepancy,
        "all_finite_predictions":bool(np.isfinite(pred).all()), "snapshot_velocity":"beta*u-gamma*s at stored state"}, indent=2))
    return rows, thresholds


def summarize(out, rows, thresholds):
    df = pd.DataFrame(rows)
    df.to_csv(out/"seed_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out/"sensitivity.csv", index=False)
    sensitivity_summary = []
    for key, part in pd.DataFrame(thresholds).groupby(["topology","projection","high","low"]):
        values = part.fraction.to_numpy()
        half = stats.t.ppf(.975,len(values)-1)*stats.sem(values) if len(values)>1 else np.nan
        sensitivity_summary.append(dict(zip(["topology","projection","high","low"],key),
            mean_fraction=values.mean(),ci95_low=values.mean()-half,ci95_high=values.mean()+half))
    pd.DataFrame(sensitivity_summary).to_csv(out/"sensitivity_summary.csv",index=False)
    summary = []
    for (topology, metric), part in df.groupby(["topology", "metric"]):
        vals = part.mean_cosine.to_numpy()
        half = stats.t.ppf(.975,len(vals)-1)*stats.sem(vals) if len(vals)>1 else np.nan
        delta_half = stats.t.ppf(.975,len(vals)-1)*stats.sem(part.mean_delta) if len(vals)>1 else np.nan
        summary.append(dict(topology=topology, metric=metric, seeds=len(vals), mean=vals.mean(),
                            ci95_low=vals.mean()-half, ci95_high=vals.mean()+half,
                            mean_delta=part.mean_delta.mean(), delta_ci95_low=part.mean_delta.mean()-delta_half,
                            delta_ci95_high=part.mean_delta.mean()+delta_half, excluded_fraction=part.excluded_fraction.mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(out/"summary.csv", index=False)
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    colors = {"cycle":"#007F73", "bifurcation":"#B64558", "cycle_bifurcation":"#71579C"}
    for ax, names in zip(axes, [["absolute","proportion_both","log_proportion_both"],["absolute","pca_linear","umap_kernel"]]):
        for j, (topology, group) in enumerate(summary.groupby("topology")):
            q = group.set_index("metric").loc[names]
            value = q["mean"] if ax is axes[0] else q.mean_delta
            error = (q.ci95_high-q.ci95_low)/2 if ax is axes[0] else (q.delta_ci95_high-q.delta_ci95_low)/2
            ax.errorbar(np.arange(3)+(j-1)*.12,value,yerr=error,
                        marker="o",capsize=3,label=topology,color=colors.get(topology))
        ax.set_xticks(range(3),[n.replace("_","\n") for n in names])
        ax.set_ylabel("Mean cosine (95% seed-level t interval)")
        ax.axhline(0,color="gray",lw=.6)
        ax.set_ylim(-1.05,1.05)
    axes[0].set_title("Coordinate pushforward of both vectors")
    axes[1].set_title("Same velocity pair before / after projection")
    axes[1].set_ylabel("Paired mean cosine change (95% seed interval)")
    axes[1].legend(fontsize=8)
    fig.savefig(out/"coordinate_projection.png",dpi=180)
    fig.savefig(out/"coordinate_projection.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11,22,33,44,55])
    parser.add_argument("--topologies", nargs="+", default=["cycle","bifurcation","cycle_bifurcation"])
    parser.add_argument("--out", default="results/revision_coordinate_projection")
    args = parser.parse_args()
    cfg = dict(dt=.01, steps=1200, burnin_steps=200, diffusion=.006, trajectories=48,
               genes=160, observations=600, save_every=20, neighbors=20, seeds=args.seeds, topologies=args.topologies)
    out = ROOT/args.out
    out.mkdir(parents=True,exist_ok=True)
    provenance = dict(config=cfg, interpretation="Supplementary oracle capture-corrected deterministic scVelo control; not default preprocessing or cross-method validation",
        smoothing="Fixed uniform kNN operator including self, built from log1p capture-corrected counts; both GT and observations smoothed",
        projection="PCA linear pushforward; UMAP positive cosine kernel over fixed neighbors, row normalized, zero if no positive weights; no grid scaling",
        uncertainty="Independent dataset seeds are replication units; snapshots within a trajectory are correlated",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/"data_generation_module/graph_core.py",ROOT/"data_generation_module/synthesizer_gene.py",
             Path(core.__file__),ROOT/"experiments/results_topology_primitives.py",ROOT/"experiments/results_statistical_realism.py"]},
        synthesizer_settings=core.SYNTH_KW,
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        versions={p:importlib.metadata.version(p) for p in ["numpy","scipy","pandas","scvelo","scanpy","anndata","umap-learn","scikit-learn"]})
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2))
    rows, thresholds = [], []
    for topology in args.topologies:
        for seed in args.seeds:
            dest = out/f"{topology}_seed{seed}"
            dest.mkdir(exist_ok=True)
            print(f"RUN {topology} seed={seed}",flush=True)
            data = simulate(cfg,topology,seed)
            r,t = evaluate(data,cfg,topology,seed,dest)
            rows.extend(r); thresholds.extend(t)
            pd.DataFrame(rows).to_csv(out/"seed_metrics.csv",index=False)
            pd.DataFrame(thresholds).to_csv(out/"sensitivity.csv",index=False)
    summarize(out,rows,thresholds)
    print(f"COMPLETE {out}",flush=True)


if __name__ == "__main__":
    main()
