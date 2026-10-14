"""Coupled-step convergence probe for cycle-bifurcation branch outcomes."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_generation_module import build_topology
from experiments.revision_numerical_kinetics import CONFIG, run_coupled

OUT = ROOT / "results" / "revision_branch_convergence"
SEEDS = (11, 22, 33, 44, 55)
DTS = (.02, .01, .005, .0025)
HORIZON = 8.
CELLS = 24
NOISE = .15


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CONFIG["dts"] = list(DTS)
    CONFIG["noise"] = NOISE
    system, start = build_topology("cycle_bifurcation")
    cell_rows, seed_rows = [], []
    for seed in SEEDS:
        print("seed", seed, flush=True)
        rng = np.random.default_rng(seed)
        initial = rng.standard_normal((CELLS, 2))
        fine_dt = min(DTS)
        dw = rng.standard_normal((round(HORIZON / fine_dt), CELLS, 2)) * np.sqrt(fine_dt)
        paths = {}
        for dt in DTS:
            paths[dt], roundoff = run_coupled(system, start, initial, dw, dt)
            if not np.isfinite(paths[dt]).all() or roundoff > 1e-5:
                raise ValueError(f"Nonfinite path or excessive Brownian rounding: {seed}, {dt}")
        reference = paths[fine_dt]
        reference_label = np.sign(reference[-1, :, 1])
        for dt in DTS:
            path = paths[dt]
            ratio = round(dt / fine_dt)
            endpoint = path[-1]
            labels = np.sign(endpoint[:, 1])
            differences = path - reference[::ratio]
            path_rms = float(np.sqrt(np.mean(np.sum(differences ** 2, axis=-1))))
            label_disagreement = float(np.mean(labels != reference_label))
            seed_rows.append(dict(seed=seed, dt=dt, upper_fraction=float(np.mean(labels > 0)),
                                  label_disagreement=label_disagreement, path_rms=path_rms,
                                  n_terminal=int(np.count_nonzero(endpoint[:, 0] > 1.2)),
                                  minimum_terminal_abs_y=float(np.min(np.abs(endpoint[:, 1])))))
            for cell in range(CELLS):
                cell_rows.append(dict(seed=seed, cell=cell, dt=dt,
                                      final_x=float(endpoint[cell, 0]), final_y=float(endpoint[cell, 1]),
                                      branch="upper" if labels[cell] > 0 else "lower",
                                      agrees_with_finest=bool(labels[cell] == reference_label[cell])))
    summary = pd.DataFrame(seed_rows)
    summary.to_csv(OUT / "seed_metrics.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(OUT / "cell_labels.csv", index=False)
    if (summary.n_terminal != CELLS).any() or (summary.minimum_terminal_abs_y < .5).any():
        raise ValueError("Branch labels are not operationally separated at the final time")
    config = dict(seeds=SEEDS, dts=DTS, horizon=HORIZON, cells=CELLS, noise_amplitude=NOISE,
                  diffusion_coefficient=NOISE**2/2, topology="cycle_bifurcation",
                  initial_distribution="independent standard normal in two latent dimensions",
                  coupling="shared fine Brownian increments aggregated to each coarser step",
                  branch_rule="sign of final y among trajectories with final x>1.2; all abs(final y)>.5",
                  reference="dt=.0025 is a numerical comparator, not exact continuous-time truth",
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), ROOT / "experiments/revision_numerical_kinetics.py")})
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    aggregate = summary.groupby("dt").agg(mean_path_rms=("path_rms", "mean"),
        mean_disagreement=("label_disagreement", "mean"),
        total_disagreement=("label_disagreement", lambda x: int(round(CELLS * x.sum()))),
        mean_upper_fraction=("upper_fraction", "mean")).reset_index()
    aggregate.to_csv(OUT / "summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), layout="constrained")
    for seed, group in summary.groupby("seed"):
        axes[0].plot(group.dt, group.path_rms, "o-", alpha=.35, linewidth=.9, color="#56758a")
        axes[1].plot(group.dt, group.label_disagreement, "o-", alpha=.35, linewidth=.9, color="#a55440")
    axes[0].plot(aggregate.dt, aggregate.mean_path_rms, "o-", color="#174d67", linewidth=2, label="seed mean")
    axes[1].plot(aggregate.dt, aggregate.mean_disagreement, "o-", color="#8d3727", linewidth=2, label="seed mean")
    for ax, ylabel in zip(axes, ("Path RMS vs step .0025", "Branch-label disagreement")):
        ax.set_xticks(DTS, [str(x) for x in DTS])
        ax.set_xlabel("Integration step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.savefig(OUT / "branch_convergence.pdf")
    plt.close(fig)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
