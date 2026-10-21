"""Compose Figure (main) for fig_diag_projection_masking."""

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
from scripts.figures.fig_diag_projection_masking.lib import core
from scripts.figures.fig_diag_projection_masking.lib.data import load_cfg, load_main_and_sweep
from scripts.projection.projection_masking_metrics import EVALUATION_CEILING_CAPTION

PANEL_MODULES = {
    "panel_a_mismatch_vs_gene_count": "scripts.figures.fig_diag_projection_masking.panels.panel_a_mismatch_vs_gene_count",
    "panel_b_decoupled_agreements": "scripts.figures.fig_diag_projection_masking.panels.panel_b_decoupled_agreements",
    "panel_c_gene_module_errors": "scripts.figures.fig_diag_projection_masking.panels.panel_c_gene_module_errors",
    "panel_d_topology_masking_map": "scripts.figures.fig_diag_projection_masking.panels.panel_d_topology_masking_map",
    "panel_e_region_masking_fraction": "scripts.figures.fig_diag_projection_masking.panels.panel_e_region_masking_fraction",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_main.json")
    compose = cfg["compose"]

    main_data, sweep_rows = load_main_and_sweep()
    apply_publication_style()

    grid = compose["grid"]
    fig = plt.figure(figsize=tuple(compose["figsize"]))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=grid["height_ratios"],
        hspace=float(grid["hspace"]),
        wspace=float(grid["wspace"]),
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])

    panels = compose["panels"]
    modules = [
        panels["a"],
        panels["b"],
        panels["c"],
        panels["d"],
        panels["e"],
    ]
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e]

    for ax, panel_key in zip(axes, modules):
        mod = importlib.import_module(PANEL_MODULES[panel_key])
        panel_cfg = load_cfg(panel_config=f"{panel_key}.json")["panel"]
        mod.draw(ax, main_data, sweep_rows, panel_cfg)

    fig.suptitle(EVALUATION_CEILING_CAPTION, fontsize=8.5, y=0.99)

    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)

