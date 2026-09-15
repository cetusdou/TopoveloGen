"""Run all topology simulations and write trajectory caches + metrics CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figures._common.style import apply_publication_style  # noqa: E402
from experiments.figures.fig02_topology_primitives.lib import core  # noqa: E402
from experiments.figures.fig02_topology_primitives.lib.data import load_cfg  # noqa: E402


def main(*, plot_only: bool = False) -> None:
    cfg = load_cfg()
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)

    if plot_only:
        print("Pipeline plot-only: caches already expected under", core.DATA_DIR)
        return

    pipe = cfg["pipeline"]
    builders = core.get_builders()
    metrics_rows = []

    for name in pipe["figure_order"]:
        print(f"Running topology: {name}")
        system = builders[name]()
        z0 = core.initialize_cells(name, core.N_CELLS, system, seed=core.SEED)
        sim_kw = {"n_steps": core.N_STEPS, "diffusion": core.DIFFUSION, **core.SIM_OVERRIDES.get(name, {})}
        traj, z_final = core.simulate_langevin(
            system,
            z0,
            dt=core.DT,
            save_every=core.SAVE_EVERY,
            seed=core.SEED,
            **sim_kw,
        )
        metrics = core.compute_topology_metrics(system, z_final)
        core.save_trajectory_npz(name, traj, metrics)
        metrics_rows.append(
            {
                "topology": name,
                **metrics,
                "n_cells": core.N_CELLS,
                "n_steps": sim_kw["n_steps"],
                "diffusion": sim_kw["diffusion"],
            }
        )
        print(
            f"  mean_dist={metrics['mean_distance']:.4f}, "
            f"flow_align={metrics['mean_flow_alignment']:.4f}"
        )

    core.save_metrics_csv(metrics_rows, core.DATA_DIR / "topology_metrics.csv")
    apply_publication_style()
    print(f"Saved metrics: {core.DATA_DIR / 'topology_metrics.csv'}")
    print("Next: run panels or compose_main.py / compose_supp.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig2 pipeline: simulate topologies")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation (caches must exist)")
    args = parser.parse_args()
    main(plot_only=args.plot_only)
