"""Compose Figure 7 main 2×3 gallery from panel draw() functions."""

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
from experiments.figures.fig07_embedding_masked_errors.lib import core
from experiments.figures.fig07_embedding_masked_errors.lib.data import load_cfg, load_main_and_sweep

PANEL_MODULES = {
    "panel_a_gt_velocity": "experiments.figures.fig07_embedding_masked_errors.panels.panel_a_gt_velocity",
    "panel_b_scvelo_velocity": "experiments.figures.fig07_embedding_masked_errors.panels.panel_b_scvelo_velocity",
    "panel_c_highd_cosine": "experiments.figures.fig07_embedding_masked_errors.panels.panel_c_highd_cosine",
    "panel_d_embedding_vs_gene": "experiments.figures.fig07_embedding_masked_errors.panels.panel_d_embedding_vs_gene",
    "panel_e_mismatch_topology": "experiments.figures.fig07_embedding_masked_errors.panels.panel_e_mismatch_topology",
    "panel_f_mismatch_complexity": "experiments.figures.fig07_embedding_masked_errors.panels.panel_f_mismatch_complexity",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_main.json")
    compose = cfg["compose"]
    main_data, sweep = load_main_and_sweep()
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
            mod.draw(axes_flat[i], main_data, panel_cfg, sweep=sweep)
            i += 1

    if compose.get("tight_layout", True):
        fig.tight_layout(
            pad=float(compose.get("pad", 0.4)),
            w_pad=float(compose.get("w_pad", 0.6)),
            h_pad=float(compose.get("h_pad", 0.8)),
        )

    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
