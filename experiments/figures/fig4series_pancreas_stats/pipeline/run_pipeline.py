"""Run Fig4series two-layer pipeline: Stage1 kinetics cache + Stage2 observation noise."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figures.fig4series_pancreas_stats.lib import core  # noqa: E402
from experiments.figures.fig4series_pancreas_stats.lib import metrics  # noqa: E402
from experiments.figures.fig4series_pancreas_stats.lib.data import load_cfg  # noqa: E402


def main(*, plot_only: bool = False, cache_only: bool = False, force_kinetics: bool = False) -> None:
    load_cfg()
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)

    if plot_only:
        if not core.KINETICS_CACHE.exists():
            raise FileNotFoundError(
                f"Kinetics cache not found: {core.KINETICS_CACHE}\n"
                "Run once without --plot-only / --cache-only to build Stage-1."
            )
        print("Pipeline plot-only: Stage-1 cache present; panels re-apply Stage-2 on load.")
        return

    result = core.run_stages(force_rebuild=force_kinetics, cache_only=cache_only)
    if result is None:
        return

    print("Saving CSV statistics...")
    rows = metrics.save_csvs(result)
    sim_row = next(r for r in rows if r["dataset"] == "synthetic_figure4")
    real_row = next(r for r in rows if r["dataset"] == "real_pancreas")
    print(
        f"Synthetic: median U/S={sim_row['median_u_s_ratio']:.3f}, "
        f"sparsity S={sim_row['sparsity_spliced']:.3f}"
    )
    print(
        f"Real:      median U/S={real_row['median_u_s_ratio']:.3f}, "
        f"sparsity S={real_row['sparsity_spliced']:.3f}"
    )
    print("Next: run panels or compose_main.py / compose_supp.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fig4series pipeline: Stage1 kinetics + Stage2 PostProcessConfig",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Two-layer architecture:\n"
            "  Stage 1 – continuous kinetics : cached in kinetics_cache.pt (slow)\n"
            "  Stage 2 – observation noise   : PostProcessConfig in configs/pipeline.json (fast)\n"
        ),
    )
    parser.add_argument("--plot-only", action="store_true", help="Skip generation; require Stage-1 cache")
    parser.add_argument("--cache-only", action="store_true", help="Build Stage-1 cache only")
    parser.add_argument("--force-kinetics", action="store_true", help="Force rebuild Stage-1 cache")
    args = parser.parse_args()
    main(plot_only=args.plot_only, cache_only=args.cache_only, force_kinetics=args.force_kinetics)
