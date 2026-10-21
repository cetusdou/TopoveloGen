"""Simulate continuous kinetics, run scVelo, write coordinate-consistency caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.fig_coord_velocity_consistency.lib import core  # noqa: E402
from scripts.figures.fig_coord_velocity_consistency.lib.data import load_cfg  # noqa: E402


def main(*, plot_only: bool = False) -> None:
    cfg = load_cfg()
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)

    if plot_only:
        bundle = core.load_bundle()
        print(
            f"Loaded cache: n_cells={bundle['summary']['n_cells']}, "
            f"abs_mean={bundle['summary']['abs_mean']:.3f}, "
            f"prop_mean={bundle['summary']['prop_mean']:.3f}"
        )
        print("Next: run panels or compose_main.py")
        return

    result = core.run_experiment()
    core.save_results(result)
    s = result["summary"]
    print(
        f"Summary: abs={s['abs_mean']:.3f}, lib={s['lib_mean']:.3f}, "
        f"prop={s['prop_mean']:.3f}, log={s['log_mean']:.3f}; "
        f"flip_abs_prop={s['flip_abs_prop']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coordinate consistency pipeline")
    parser.add_argument("--plot-only", action="store_true", help="Only verify caches exist")
    args = parser.parse_args()
    main(plot_only=args.plot_only)
