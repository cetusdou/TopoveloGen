"""Shared controlled simulation and scVelo inference setup."""

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from experiments.figures.fig_coord_velocity_consistency.lib import core


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
