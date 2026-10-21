"""Compose 2×2 supplementary figure for coordinate consistency."""

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
from scripts.figures.fig_coord_velocity_consistency.lib import core
from scripts.figures.fig_coord_velocity_consistency.lib.data import load_bundle, load_cfg

PANEL_MODULES = {
    "panel_a_diagram": "scripts.figures.fig_coord_velocity_consistency.panels.panel_a_diagram",
    "panel_b_cosine_dist": "scripts.figures.fig_coord_velocity_consistency.panels.panel_b_cosine_dist",
    "panel_c_scatter": "scripts.figures.fig_coord_velocity_consistency.panels.panel_c_scatter",
    "panel_d_summary": "scripts.figures.fig_coord_velocity_consistency.panels.panel_d_summary",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_main.json")
    compose = cfg["compose"]
    bundle = load_bundle()
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
            mod.draw(axes_flat[i], bundle, panel_cfg)
            i += 1

    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
