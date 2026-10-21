"""Run junction simulations and write trajectory caches + metrics CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pickle

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.fig03_junction_continuity.lib import core  # noqa: E402
from scripts.figures.fig03_junction_continuity.lib.data import load_cfg  # noqa: E402


def main(*, plot_only: bool = False) -> None:
    cfg = load_cfg()
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)

    if plot_only:
        print("Pipeline plot-only: caches already expected under", core.DATA_DIR)
        return

    pipe = cfg["pipeline"]
    results = {}
    metric_rows = []

    for name in pipe["figure_order"]:
        print(f"=== {name} ===")
        out = core.run_one_junction(name)
        results[name] = out
        metrics = out["metrics"]
        metric_rows.append(metrics)
        rim = metrics.get("rim_docking_error")
        if rim is not None:
            print(f"  rim docking error: {rim:.4f} (0 = on rim)")
        print(
            f"  junction smoothness: {metrics['drift_smoothness_junction']:.3f}, "
            f"passage: {metrics['passage_success']:.3f}"
        )

    core.save_metrics_csv(metric_rows, core.DATA_DIR / "junction_metrics.csv")
    pickle.dump({k: results[k]["traj"] for k in results}, open(str(core.DATA_DIR / "trajectories.pt"), "wb"))
    print(f"Saved metrics: {core.DATA_DIR / 'junction_metrics.csv'}")
    print(f"Saved trajectories: {core.DATA_DIR / 'trajectories.pt'}")
    print("Next: run panels or compose_main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig3 pipeline: junction continuity")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation (caches must exist)")
    args = parser.parse_args()
    main(plot_only=args.plot_only)
