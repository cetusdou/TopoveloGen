"""Shared simulation and scoring for the revision factor experiments.

The factor, interaction, and cell/gene runners use this module without
importing one another's plotting or file-writing code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.figures.fig_coord_velocity_consistency.lib import core
from scripts.common.velocity_metrics import cosine
from scripts.common.coordinate_simulation import infer_and_smooth, simulate

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DEPENDENCIES = (
    Path(__file__),
    ROOT / "scripts/projection/revision_coordinate_projection.py",
    ROOT / "scripts/figures/fig_coord_velocity_consistency/lib/core.py",
    ROOT / "scripts/topology/results_topology_primitives.py",
    ROOT / "scripts/counts/results_statistical_realism.py",
    ROOT / "data_generation_module/graph_core.py",
    ROOT / "data_generation_module/synthesizer_gene.py",
    ROOT / "data_generation_module/presets.py",
)

BASE = dict(dt=.01, steps=1200, burnin_steps=200, diffusion=.006,
            trajectories=48, genes=160, observations=600, save_every=20,
            neighbors=20)
DEFAULT_SYNTH = dict(core.SYNTH_KW)
TOPOLOGIES = ("cycle", "bifurcation", "cycle_bifurcation")


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
