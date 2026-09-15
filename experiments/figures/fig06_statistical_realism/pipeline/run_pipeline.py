"""Run Fig6 pipeline: sample cells, generate counts, write caches + CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figures.fig06_statistical_realism.lib import core  # noqa: E402
from experiments.figures.fig06_statistical_realism.lib.data import load_cfg  # noqa: E402


def main(*, plot_only: bool = False) -> None:
    load_cfg()
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    if plot_only:
        print("Pipeline plot-only: caches expected under", core.DATA_DIR)
        return
    core.run_pipeline()
    print(f"Saved counts: {core.DATA_DIR / 'count_matrices.pt'}")
    print("Next: run panels or compose_main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig6 pipeline: statistical realism counts")
    parser.add_argument("--plot-only", action="store_true", help="Skip generation (caches must exist)")
    args = parser.parse_args()
    main(plot_only=args.plot_only)
