"""Compose Figure 2 main 2×3 gallery from panel draw() functions."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from scripts.figures._common.style import apply_publication_style, save_figure
from scripts.figures.fig02_topology_primitives.lib import core
from scripts.figures.fig02_topology_primitives.lib.data import load_all_results, load_cfg


PANEL_MODULES = {
    "panel_a_linear": "scripts.figures.fig02_topology_primitives.panels.panel_a_linear",
    "panel_b_bifurcation": "scripts.figures.fig02_topology_primitives.panels.panel_b_bifurcation",
    "panel_c_cycle": "scripts.figures.fig02_topology_primitives.panels.panel_c_cycle",
    "panel_d_cycle_linear": "scripts.figures.fig02_topology_primitives.panels.panel_d_cycle_linear",
    "panel_e_skeleton": "scripts.figures.fig02_topology_primitives.panels.panel_e_skeleton",
    "panel_f_cycle_bifurcation": "scripts.figures.fig02_topology_primitives.panels.panel_f_cycle_bifurcation",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_main.json")
    compose = cfg["compose"]
    all_results = load_all_results()
    apply_publication_style()

    fig, axes = plt.subplots(
        int(compose["nrows"]),
        int(compose["ncols"]),
        figsize=tuple(compose["figsize"]),
        constrained_layout=bool(compose.get("constrained_layout", True)),
    )
    axes_flat = axes.flatten()

    i = 0
    for row in compose["panels"]:
        for panel_key in row:
            mod = importlib.import_module(PANEL_MODULES[panel_key])
            panel_cfg = load_cfg(panel_config=f"{panel_key}.json")
            topology = panel_cfg["panel"]["topology"]
            mod.draw(axes_flat[i], all_results[topology], panel_cfg)
            i += 1

    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
