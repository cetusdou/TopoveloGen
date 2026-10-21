"""Run latent sampling + alpha computation; write caches and metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pickle

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.fig04_alpha_fields.lib import core  # noqa: E402
from scripts.figures.fig04_alpha_fields.lib.data import load_cfg  # noqa: E402


def main(*, plot_only: bool = False) -> None:
    cfg = load_cfg()
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)

    if plot_only:
        print("Pipeline plot-only: caches already expected under", core.DATA_DIR)
        return

    print("Building cycle-bifurcation topology...")
    system = core.build_cycle_bifurcation_system()
    docks = core.cycle_exit_docks(system)
    for i, d in enumerate(docks):
        print(f"  exit dock {i}: ({d[0]:.3f}, {d[1]:.3f})")

    print("Simulating latent cells (Langevin)...")
    z_cells = core.sample_latent_cells(system)
    print(f"  n_cells = {z_cells.shape[0]}")

    print("Setting up GeneSynthesizer...")
    synth = core.setup_gene_synthesizer(system)

    print("Computing alpha(z)...")
    alpha = core.build_full_alpha_matrix(synth, system, z_cells)

    pickle.dump(z_cells, open(str(core.DATA_DIR / "latent_positions.pt"), "wb"))
    pickle.dump(alpha, open(str(core.DATA_DIR / "alpha_matrix.pt"), "wb"))
    core.save_gene_metadata(synth, core.DATA_DIR / "gene_metadata.csv")

    metrics = core.compute_alpha_metrics(system, z_cells, alpha, synth)
    core.save_metrics_csv(metrics, core.DATA_DIR / "alpha_field_metrics.csv")

    print(f"Branch1 specificity: {metrics[0]['branch_specificity']:.3f}")
    print(f"Branch2 specificity: {metrics[1]['branch_specificity']:.3f}")
    print(f"Junction exit localization: {metrics[2]['localization_score']:.3f}")
    print("Next: run panels or compose_main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig4 pipeline: alpha fields")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation (caches must exist)")
    args = parser.parse_args()
    main(plot_only=args.plot_only)
