"""Pipeline for fig_diag_projection_masking.

This panelized package draws from legacy cached CSVs.
If caches are missing, we re-run the legacy computation entrypoint:
`scripts/projection/results_projection_masking_diagnosis.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.fig_diag_projection_masking.lib import core
from scripts.figures.fig_diag_projection_masking.lib.data import load_cfg


def main(*, plot_only: bool = False, quick: bool = False, refresh: bool = False) -> None:
    cfg = load_cfg()
    req = cfg["pipeline"].get("required_csvs", [core.GENE_SWEEP_CSV, core.CELL_DIAGNOSTICS_CSV])
    recompute_if_missing = bool(cfg["pipeline"].get("recompute_if_missing", True))
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)

    missing = [name for name in req if not (core.DATA_DIR / name).exists()]
    if plot_only:
        if missing:
            raise FileNotFoundError(
                "Missing cached CSV(s); cannot plot-only:\n- "
                + "\n- ".join(str(core.DATA_DIR / m) for m in missing)
            )
        print("Pipeline plot-only: cached CSVs already exist under", core.DATA_DIR)
        return

    if missing or refresh:
        if missing and not recompute_if_missing and not refresh:
            raise FileNotFoundError(
                "Cached CSVs missing and recompute_if_missing is false:\n- "
                + "\n- ".join(str(core.DATA_DIR / m) for m in missing)
            )

        print("Recomputing cached CSVs via legacy entrypoint...")
        import scripts.projection.results_projection_masking_diagnosis as legacy  # noqa: E402

        legacy.main(quick=quick)
        print("Legacy computation finished; panelized plots can now be generated.")
    else:
        print("Caches already present under", core.DATA_DIR)
        print("Next: run panels or compose_main.py / compose_supp.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig diag_projection_masking pipeline")
    parser.add_argument("--plot-only", action="store_true", help="Skip computation (requires cached CSVs)")
    parser.add_argument("--quick", action="store_true", help="Use legacy quick gene sweep")
    parser.add_argument("--refresh", action="store_true", help="Force legacy recomputation")
    args = parser.parse_args()
    cfg = load_cfg()
    quick_default = bool(cfg["pipeline"].get("quick", False))
    plot_only_default = bool(cfg["pipeline"].get("plot_only", False))
    main(
        plot_only=(plot_only_default or bool(args.plot_only)),
        quick=(quick_default or bool(args.quick)),
        refresh=bool(args.refresh),
    )

