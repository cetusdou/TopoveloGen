"""Pipeline wrapper for fig_bench_velo_test.

The panelized package draws from legacy cached CSVs produced by:
`experiments/results_benchmark_velo_test.py`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figures.fig_bench_velo_test.lib import core
from experiments.figures.fig_bench_velo_test.lib.data import load_cfg


def main(*, plot_only: bool = False, refresh: bool = False) -> None:
    cfg = load_cfg()
    req = cfg["pipeline"].get("required_csvs", [])
    opt = cfg["pipeline"].get("optional_csvs", [])
    recompute_if_missing = bool(cfg["pipeline"].get("recompute_if_missing", True))

    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)

    missing = [name for name in req if not (core.DATA_DIR / name).exists()]
    if plot_only:
        if missing:
            raise FileNotFoundError(
                "Missing cached CSV(s) required for plot-only:\n- " + "\n- ".join(str(core.DATA_DIR / m) for m in missing)
            )
        print("Pipeline plot-only: cached CSVs already exist under", core.DATA_DIR)
        return

    if missing and not recompute_if_missing and not refresh:
        raise FileNotFoundError(
            "Cached CSV(s) missing and recompute_if_missing is false:\n- "
            + "\n- ".join(str(core.DATA_DIR / m) for m in missing)
        )

    if missing or refresh:
        print("Running legacy benchmark computation to refresh caches...")
        legacy_script = REPO_ROOT / "experiments" / "results_benchmark_velo_test.py"
        cmd = [sys.executable, str(legacy_script)]
        if plot_only:
            cmd.append("--plot-only")
        subprocess.check_call(cmd)
        print("Legacy computation finished; panelized outputs can be generated.")
    else:
        print("Caches already present under", core.DATA_DIR)
        print("Next: run panels or compose_main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig_bench_velo_test pipeline")
    parser.add_argument("--plot-only", action="store_true", help="Skip computation (requires cached CSVs)")
    parser.add_argument("--refresh", action="store_true", help="Force recomputation via legacy script")
    args = parser.parse_args()
    cfg = load_cfg()
    plot_only_default = bool(cfg["pipeline"].get("plot_only", False))
    main(plot_only=(plot_only_default or bool(args.plot_only)), refresh=bool(args.refresh))

