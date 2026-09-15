"""Compose Figure 3 main 2×3 gallery from panel draw() functions."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from experiments.figures._common.style import apply_publication_style, save_figure
from experiments.figures.fig03_junction_continuity.lib import core
from experiments.figures.fig03_junction_continuity.lib.data import (
    load_all_results,
    load_cfg,
    load_metric_rows,
)


PANEL_MODULES = {
    "panel_a_cycle_linear_field": "experiments.figures.fig03_junction_continuity.panels.panel_a_cycle_linear_field",
    "panel_b_cycle_linear_zoom": "experiments.figures.fig03_junction_continuity.panels.panel_b_cycle_linear_zoom",
    "panel_c_cycle_linear_traj": "experiments.figures.fig03_junction_continuity.panels.panel_c_cycle_linear_traj",
    "panel_d_bifurcation_field": "experiments.figures.fig03_junction_continuity.panels.panel_d_bifurcation_field",
    "panel_e_cycle_bifurcation_field": "experiments.figures.fig03_junction_continuity.panels.panel_e_cycle_bifurcation_field",
    "panel_f_metrics": "experiments.figures.fig03_junction_continuity.panels.panel_f_metrics",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_main.json")
    compose = cfg["compose"]
    all_results = load_all_results()
    metric_rows = load_metric_rows()
    apply_publication_style()

    fig, axes = plt.subplots(
        int(compose["nrows"]),
        int(compose["ncols"]),
        figsize=tuple(compose["figsize"]),
    )
    axes_flat = axes.flatten()

    i = 0
    for row in compose["panels"]:
        for panel_key in row:
            mod = importlib.import_module(PANEL_MODULES[panel_key])
            panel_cfg = load_cfg(panel_config=f"{panel_key}.json")
            if panel_key == "panel_f_metrics":
                mod.draw(axes_flat[i], metric_rows, panel_cfg)
            else:
                topology = panel_cfg["panel"]["topology"]
                mod.draw(axes_flat[i], all_results[topology], panel_cfg)
            i += 1

    tl = compose.get("tight_layout") or {}
    fig.tight_layout(**{k: float(v) for k, v in tl.items()})

    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
