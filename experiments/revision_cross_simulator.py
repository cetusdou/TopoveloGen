"""Within-simulator scVelo reference checks for two distinct RNA generators."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvelo as scv
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_scmultisim_comparison"
TOPO = ROOT / "results" / "revision_three_method_benchmark"


def moments(s, u, v):
    n = len(s)
    indices = NearestNeighbors(n_neighbors=20).fit(np.log1p(s)).kneighbors(return_distance=False)
    cols = np.column_stack((np.arange(n), indices)).ravel()
    rows = np.repeat(np.arange(n), 21)
    weight = sparse.csr_matrix((np.full(len(cols), 1 / 21), (rows, cols)), shape=(n, n))
    return tuple(np.asarray(weight @ x) for x in (s, u, v))


def score(pred, true):
    pn = np.linalg.norm(pred, axis=1)
    tn = np.linalg.norm(true, axis=1)
    valid = np.isfinite(pred).all(axis=1) & (pn > 1e-8) & (tn > 1e-8)
    cosine = np.full(len(pred), np.nan)
    cosine[valid] = np.einsum("ij,ij->i", pred[valid], true[valid]) / (pn[valid] * tn[valid])
    return valid, cosine


def main():
    scv.settings.verbosity = 0
    rows = []
    provenance = []
    for seed in (11, 22, 33):
        sources = []
        for simulator in ("TopoVeloGen", "scMultiSim"):
            if simulator == "TopoVeloGen":
                input_path = TOPO / f"bifurcation_seed{seed}.npz"
                data = np.load(input_path)
                true = data["gt"]
                a = ad.AnnData(X=np.log1p(data["Ms"]).astype("float32"))
                a.layers["Ms"] = data["Ms"].copy()
                a.layers["Mu"] = data["Mu"].copy()
                scv.tl.velocity(a, mode="deterministic", use_raw=False,
                                min_r2=-np.inf, min_likelihood=-np.inf)
                pred = np.asarray(a.layers["velocity"], dtype=float)
                np.savez_compressed(OUT / f"topovelogen_seed{seed}_scvelo033.npz",
                                    pred=pred.astype("float32"), gt=true.astype("float32"))
                input_files = [input_path]
            else:
                input_files = [OUT / f"{layer}_seed{seed}.csv" for layer in ("spliced", "unspliced", "velocity")]
                s, u, v = (pd.read_csv(path).to_numpy(dtype=float) for path in input_files)
                ms, mu, true = moments(s, u, v)
                a = ad.AnnData(X=np.log1p(ms).astype("float32"))
                a.layers["Ms"] = ms.astype("float32")
                a.layers["Mu"] = mu.astype("float32")
                scv.tl.velocity(a, mode="deterministic", use_raw=False,
                                min_r2=-np.inf, min_likelihood=-np.inf)
                pred = np.asarray(a.layers["velocity"], dtype=float)
                np.savez_compressed(OUT / f"scmultisim_seed{seed}_scvelo.npz",
                                    pred=pred.astype("float32"), gt=true.astype("float32"))
            valid, cosine = score(pred, true)
            sources.append(dict(simulator=simulator, seed=seed, valid_fraction=float(np.mean(valid)),
                                median_cosine=float(np.nanmedian(cosine))))
            provenance.append(dict(simulator=simulator, seed=seed,
                input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in input_files},
                cells=int(len(true)), genes=int(true.shape[1])))
            for i, value in enumerate(cosine):
                rows.append(dict(simulator=simulator, seed=seed, cell=i,
                                 cosine=value, valid=bool(valid[i])))
        print("completed", seed, sources, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "per_cell.csv", index=False)
    summary = frame.groupby(["simulator", "seed"]).agg(median_cosine=("cosine", "median"),
        mean_cosine=("cosine", "mean"), valid_fraction=("valid", "mean")).reset_index()
    summary.to_csv(OUT / "seed_summary.csv", index=False)
    (OUT / "provenance.json").write_text(json.dumps(dict(scvelo=scv.__version__,
        comparison="within-simulator directional score only; distinct generative models and noise scales are not matched",
        neighbors=20, moment_weight="uniform including self", inputs=provenance), indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = {"TopoVeloGen": "#197e72", "scMultiSim": "#b56736"}
    for j, simulator in enumerate(colors):
        values = summary.loc[summary.simulator == simulator, "median_cosine"].to_numpy()
        x = j + np.linspace(-.09, .09, len(values))
        ax.scatter(x, values, s=42, color=colors[simulator], zorder=3)
        ax.plot([j-.17, j+.17], [np.mean(values)] * 2, color=colors[simulator], lw=2)
    ax.set_xticks([0, 1], list(colors))
    ax.set_ylabel("Median cell cosine to own simulator reference")
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="0.7", lw=.8)
    fig.tight_layout()
    fig.savefig(OUT / "cross_simulator_context.pdf")


if __name__ == "__main__":
    main()
