"""Compose Figure 4 main 2×3 gallery from panel draw() functions."""

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
from experiments.figures.fig04_alpha_fields.lib import core
from experiments.figures.fig04_alpha_fields.lib.data import load_bundle, load_cfg


PANEL_MODULES = {
    "panel_a_topology": "experiments.figures.fig04_alpha_fields.panels.panel_a_topology",
    "panel_b_branch1_sigmoid": "experiments.figures.fig04_alpha_fields.panels.panel_b_branch1_sigmoid",
    "panel_c_branch2_sigmoid": "experiments.figures.fig04_alpha_fields.panels.panel_c_branch2_sigmoid",
    "panel_d_junction_exit": "experiments.figures.fig04_alpha_fields.panels.panel_d_junction_exit",
    "panel_e_cycle_phase": "experiments.figures.fig04_alpha_fields.panels.panel_e_cycle_phase",
    "panel_f_phase_heatmap": "experiments.figures.fig04_alpha_fields.panels.panel_f_phase_heatmap",
}

SCATTER_PANELS = {
    "panel_b_branch1_sigmoid",
    "panel_c_branch2_sigmoid",
    "panel_d_junction_exit",
    "panel_e_cycle_phase",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_main.json")
    compose = cfg["compose"]
    bundle = load_bundle()
    vmin_vmax = core.shared_alpha_color_limits(bundle["alpha"])
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
            if panel_key in SCATTER_PANELS:
                mod.draw(axes_flat[i], bundle, panel_cfg, vmin_vmax=vmin_vmax)
            else:
                mod.draw(axes_flat[i], bundle, panel_cfg)
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
