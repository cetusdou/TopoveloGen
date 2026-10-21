"""Controlled three-method RNA-velocity comparison on archived simulator states.

The fit commands may run in different Python environments. The shared NPZ input
fixes oracle capture correction, moments, cell order, genes and RNA reference.
This is not a reconstruction of the historical Figure 5 runs.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
os.environ.setdefault("DGLBACKEND", "pytorch")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

OUT = ROOT / "data" / "results" / "revision_three_method_benchmark"
SEEDS = (11, 22, 33, 44, 55, 66)
TOPOLOGIES = ("cycle", "bifurcation", "cycle_bifurcation")
CFG = dict(dt=.01, steps=1000, burnin_steps=200, diffusion=.006,
           trajectories=32, genes=160, observations=360, save_every=20,
           neighbors=20)


def versions():
    names = ("numpy", "scipy", "scikit-learn", "anndata", "scanpy", "scvelo",
             "unitvelo", "deepvelo", "torch", "tensorflow-cpu", "dgl")
    result = {"python": sys.version.split()[0]}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def dataset_name(topology, seed):
    return f"{topology}_seed{seed}"


def prepare(topologies, seeds, observations):
    from scripts.common.coordinate_simulation import simulate
    from scripts.figures.fig_coord_velocity_consistency.lib import core

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {**CFG, "observations": observations}
    rows = []
    for topology in topologies:
        for seed in seeds:
            name = dataset_name(topology, seed)
            dest = OUT / f"{name}.npz"
            data = simulate(cfg, topology, seed)
            capture = data["capture"]
            s = data["sc"] / capture
            u = data["uc"] / capture / data["capture_ratio"]
            n = len(s)
            k = min(cfg["neighbors"], n - 1)
            indices = NearestNeighbors(n_neighbors=k).fit(np.log1p(s)).kneighbors(return_distance=False)
            cols = np.column_stack((np.arange(n), indices)).ravel()
            rows_idx = np.repeat(np.arange(n), k + 1)
            w = sparse.csr_matrix((np.full(len(cols), 1 / (k + 1)), (rows_idx, cols)), shape=(n, n))
            ms, mu = np.asarray(w @ s), np.asarray(w @ u)
            gt, state = np.asarray(w @ data["v"]), np.asarray(w @ data["s"])
            assert ms.shape == mu.shape == gt.shape == state.shape == (n, cfg["genes"])
            assert all(np.isfinite(x).all() for x in (ms, mu, gt, state))
            assert np.max(np.abs(data["v"] - (data["beta"] * data["u"] - data["gamma"] * data["s"]))) < 1e-8
            np.savez_compressed(dest, Ms=ms.astype("float32"), Mu=mu.astype("float32"),
                gt=gt.astype("float32"), state=state.astype("float32"),
                spliced=s.astype("float32"), unspliced=u.astype("float32"),
                z=data["z"].astype("float32"), trajectory=data["trajectory"],
                time=data["time"], capture=capture.astype("float32"),
                neighbor_indices=indices, genes=np.array([f"g{j}" for j in range(cfg["genes"])], dtype="U12"))
            rows.append(dict(dataset=name, topology=topology, seed=seed, cells=n,
                genes=cfg["genes"], input_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                negative_candidate_fraction=data["negative_fraction"]))
            print("prepared", name, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "inputs.csv", index=False)
    (OUT / "input_config.json").write_text(json.dumps({
        "config": cfg, "seeds": list(seeds), "topologies": list(topologies),
        "coordinate": "oracle capture-corrected continuous-abundance units; fixed uniform neighbor moments applied to prediction target and RNA-state reference",
        "capture_correction": "spliced counts / cell capture; unspliced counts / cell capture / fixed unspliced efficiency ratio",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "simulator_source_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        "synthesizer_settings": core.SYNTH_KW,
        "versions": versions()}, indent=2), encoding="utf-8")


def make_adata(data):
    import anndata as ad
    a = ad.AnnData(X=np.log1p(data["Ms"]).astype("float32"))
    a.var_names = list(data["genes"].astype(str))
    for layer in ("Ms", "Mu", "spliced", "unspliced"):
        a.layers[layer] = data[layer].copy()
    a.obsm["X_pca"] = PCA(n_components=min(20, a.n_obs - 1, a.n_vars - 1),
                           svd_solver="full").fit_transform(a.X).astype("float32")
    return a


def fit(method, name):
    import scvelo as scv
    scv.settings.verbosity = 0
    path = OUT / f"{name}.npz"
    data = np.load(path)
    a = make_adata(data)
    seed = int(name.rsplit("seed", 1)[1])
    np.random.seed(seed)
    settings = dict(method=method, dataset=name, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        versions=versions(), input_coordinate="oracle capture-corrected abundance; 20-neighbor uniform Ms/Mu supplied; X=log1p(Ms)")
    try:
        if method == "scvelo":
            settings["fit"] = dict(mode="deterministic", use_raw=False, min_r2="-inf", min_likelihood="-inf")
            scv.tl.velocity(a, mode="deterministic", use_raw=False,
                            min_r2=-np.inf, min_likelihood=-np.inf)
        elif method == "unitvelo":
            import unitvelo as utv
            from unitvelo.config import Configuration
            scv.tl.velocity(a, mode="deterministic", use_raw=False,
                            min_r2=-np.inf, min_likelihood=-np.inf)
            a.obs["all_cells"] = "all"
            conf = Configuration()
            conf.GPU = -1
            conf.BASIS = "pca"
            conf.N_PCS = 20
            conf.N_NEIGHBORS = 20
            conf.N_TOP_GENES = a.n_vars
            conf.VGENES = "raws"
            conf.RESCALE_DATA = False
            conf.MAX_ITER = 600
            settings["fit"] = dict(normalize=False, GPU=-1, BASIS="pca", VGENES="raws",
                RESCALE_DATA=False, MAX_ITER=600, NUM_REP=conf.NUM_REP,
                FIT_OPTION=conf.FIT_OPTION, initializer="scvelo deterministic velocity_gamma on identical Ms/Mu",
                label="all_cells, one constant noninformative label; no root specified")
            work = OUT / "method_work" / method / name
            work.mkdir(parents=True, exist_ok=True)
            prior = Path.cwd()
            try:
                os.chdir(work)
                a = utv.run_model(a, label="all_cells", config_file=conf, normalize=False)
            finally:
                os.chdir(prior)
        elif method == "deepvelo":
            import deepvelo as dv
            import torch
            torch.manual_seed(seed)
            conf = dv.Constants.default_configs
            conf["n_gpu"] = 0
            conf["data_loader"]["args"].update(num_workers=0, topC=20, topG=20)
            work = OUT / "method_work" / method / name
            work.mkdir(parents=True, exist_ok=True)
            conf["trainer"].update(epochs=100, tensorboard=False, save_period=1000,
                                   save_dir=str(work))
            settings["fit"] = conf
            scv.pp.neighbors(a, n_neighbors=20, n_pcs=20)
            dv.train(a, conf)
        else:
            raise ValueError(method)
        pred = np.asarray(a.layers["velocity"], dtype="float64")
        if pred.shape != data["gt"].shape or not np.isfinite(pred).all():
            raise ValueError(f"bad output shape or nonfinite values: {pred.shape}")
        genes = np.asarray(a.var_names.astype(str))
        if not np.array_equal(genes, data["genes"]):
            raise ValueError("method changed the common gene set")
        mask = np.asarray(a.var.get("velocity_genes", np.ones(a.n_vars)), dtype=bool)
        if mask.shape != (a.n_vars,):
            mask = np.ones(a.n_vars, dtype=bool)
        dest = OUT / f"{name}_{method}.npz"
        np.savez_compressed(dest, pred=pred.astype("float32"), genes=genes,
                            method_gene_mask=mask)
        settings.update(status="success", n_genes_used=int(mask.sum()),
            output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
            output_coordinate="method velocity layer fitted from supplied oracle-corrected Ms/Mu; temporal scale is method-specific and not physically calibrated")
        print("fitted", method, name, flush=True)
    except Exception as exc:
        settings.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        print("failed", method, name, repr(exc), flush=True)
    (OUT / f"{name}_{method}.json").write_text(json.dumps(settings, indent=2, default=str), encoding="utf-8")
    return settings["status"] == "success"


def common_valid_cells(predictions, reference):
    reference_norm = np.linalg.norm(reference, axis=1)
    positive_reference = reference_norm[np.isfinite(reference_norm) & (reference_norm > 0)]
    reference_floor = 1e-6 * np.median(positive_reference) if len(positive_reference) else np.inf
    reference_valid = np.isfinite(reference).all(axis=1) & (reference_norm > reference_floor)
    method_masks = {}
    for method, prediction in predictions.items():
        prediction_norm = np.linalg.norm(prediction, axis=1)
        positive_prediction = prediction_norm[np.isfinite(prediction_norm) & (prediction_norm > 0)]
        prediction_floor = 1e-6 * np.median(positive_prediction) if len(positive_prediction) else np.inf
        method_masks[method] = (reference_valid & np.isfinite(prediction).all(axis=1)
                                & (prediction_norm > prediction_floor))
    common = np.logical_and.reduce(list(method_masks.values()))
    return common, method_masks


def summarize():
    rows = []
    for path in sorted(OUT.glob("*_seed*.npz")):
        if any(path.stem.endswith("_" + m) for m in ("scvelo", "unitvelo", "deepvelo")):
            continue
        name = path.stem
        files = {m: OUT / f"{name}_{m}.npz" for m in ("scvelo", "unitvelo", "deepvelo")}
        if not all(p.exists() for p in files.values()):
            continue
        data = np.load(path)
        gt = np.asarray(data["gt"], dtype=float)
        preds = {m: np.load(p) for m, p in files.items()}
        mask = np.logical_and.reduce([p["method_gene_mask"] for p in preds.values()])
        if mask.sum() < 10:
            continue
        gt = gt[:, mask]
        predictions = {method: np.asarray(item["pred"], dtype=float)[:, mask]
                       for method, item in preds.items()}
        common_valid, method_masks = common_valid_cells(predictions, gt)
        if not common_valid.any():
            raise ValueError(f"No cells valid for all three methods: {name}")
        for method, item in preds.items():
            pred = predictions[method]
            pn, gn = np.linalg.norm(pred, axis=1), np.linalg.norm(gt, axis=1)
            valid = common_valid
            cos = np.full(len(pred), np.nan)
            cos[valid] = np.einsum("ij,ij->i", pred[valid], gt[valid]) / (pn[valid] * gn[valid])
            dot = float(np.einsum("ij,ij->", pred[valid], gt[valid]))
            squared = float(np.einsum("ij,ij->", pred[valid], pred[valid]))
            scale = max(0., dot / squared) if squared > 0 else 0.
            calibrated_nrmse = float(np.linalg.norm(scale * pred[valid] - gt[valid]) /
                                     np.linalg.norm(gt[valid]))
            for i in range(len(pred)):
                rows.append(dict(dataset=name, topology=name.rsplit("_seed", 1)[0],
                    seed=int(name.rsplit("seed", 1)[1]), method=method, cell=i,
                    n_common_genes=int(mask.sum()), valid=bool(valid[i]),
                    method_valid=bool(method_masks[method][i]),
                    cosine=float(cos[i]), pred_norm=float(pn[i]), gt_norm=float(gn[i]),
                    fitted_positive_scale=scale, calibrated_nrmse=calibrated_nrmse))
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No complete three-method datasets; no ranking produced", flush=True)
        return
    frame.to_csv(OUT / "per_cell.csv", index=False)
    per_seed = frame.groupby(["topology", "seed", "method"], as_index=False).agg(
        n_valid=("valid", "sum"), n_method_valid=("method_valid", "sum"),
        n_common_genes=("n_common_genes", "first"),
        mean_cosine=("cosine", "mean"), median_cosine=("cosine", "median"),
        median_pred_norm=("pred_norm", "median"), median_gt_norm=("gt_norm", "median"),
        fitted_positive_scale=("fitted_positive_scale", "first"),
        calibrated_nrmse=("calibrated_nrmse", "first"))
    per_seed["raw_norm_ratio"] = per_seed.median_pred_norm / per_seed.median_gt_norm
    per_seed.to_csv(OUT / "per_seed.csv", index=False)
    summary = per_seed.groupby(["topology", "method"], as_index=False).agg(
        seeds=("seed", "nunique"), mean_cosine=("mean_cosine", "mean"),
        sd_cosine=("mean_cosine", "std"), common_genes=("n_common_genes", "min"),
        mean_calibrated_nrmse=("calibrated_nrmse", "mean"),
        sd_calibrated_nrmse=("calibrated_nrmse", "std"),
        sd_raw_norm_ratio=("raw_norm_ratio", "std"),
        mean_raw_norm_ratio=("raw_norm_ratio", "mean"))
    summary.to_csv(OUT / "summary.csv", index=False)
    assert (per_seed.n_valid > 0).all()
    assert np.isfinite(per_seed.mean_cosine).all()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    fields = (("mean_cosine", "sd_cosine", "Mean cell-wise cosine"),
              ("mean_calibrated_nrmse", "sd_calibrated_nrmse", "Calibrated NRMSE"),
              ("mean_raw_norm_ratio", "sd_raw_norm_ratio", "Raw median norm ratio"))
    methods = (("scvelo", "scVelo", "#366b85"),
               ("unitvelo", "UniTVelo", "#d28a45"),
               ("deepvelo", "DeepVelo", "#678e70"))
    topologies = TOPOLOGIES
    x = np.arange(len(topologies))
    width = .25
    for ax, (field, sd_field, title) in zip(axes, fields):
        for j, (method, label, color) in enumerate(methods):
            values = [summary.loc[(summary.topology == t) & (summary.method == method), field].iloc[0]
                      for t in topologies]
            errors = [summary.loc[(summary.topology == t) & (summary.method == method), sd_field].iloc[0]
                      for t in topologies]
            ax.bar(x + (j - 1) * width, values, width, yerr=errors, label=label,
                   color=color, edgecolor="white", linewidth=.5, capsize=2,
                   error_kw={"elinewidth": .9, "capthick": .9}, zorder=3)
        ax.set_xticks(x, ("Cycle", "Bifurcation", "Cycle +\nbifurcation"))
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(-.5, 1)
    axes[0].axhline(0, color="0.35", linewidth=.7)
    axes[-1].axhline(1, color="0.35", linewidth=.7, linestyle="--")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "three_method_controlled.pdf")
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "fit", "fit-all", "summarize"))
    ap.add_argument("--method", choices=("scvelo", "unitvelo", "deepvelo"))
    ap.add_argument("--dataset")
    ap.add_argument("--topologies", nargs="+", default=TOPOLOGIES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--observations", type=int, default=CFG["observations"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args.topologies, args.seeds, args.observations)
    elif args.stage == "fit":
        if not args.method or not args.dataset:
            ap.error("fit requires --method and --dataset")
        if not fit(args.method, args.dataset):
            raise SystemExit(1)
    elif args.stage == "fit-all":
        names = [dataset_name(topology, seed) for topology in args.topologies for seed in args.seeds]
        methods = (args.method,) if args.method else ("scvelo", "unitvelo", "deepvelo")
        failures = []
        for name in names:
            for method in methods:
                path = OUT / f"{name}_{method}.npz"
                meta_path = OUT / f"{name}_{method}.json"
                if not args.force and path.exists() and meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    input_path = OUT / f"{name}.npz"
                    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest() if input_path.exists() else None
                    if (meta.get("status") == "success" and input_hash == meta.get("input_sha256")
                            and meta["versions"]["scvelo"] == versions()["scvelo"]):
                        print("retained", method, name, flush=True)
                        continue
                log = OUT / "method_work" / method / name / "run.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w", encoding="utf-8", errors="replace") as stream:
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        ok = fit(method, name)
                print("fitted" if ok else "failed", method, name, flush=True)
                if not ok:
                    failures.append((method, name))
        if failures:
            raise SystemExit(f"{len(failures)} method fits failed: {failures}")
    else:
        summarize()


if __name__ == "__main__":
    main()
