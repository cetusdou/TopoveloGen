"""Compose Figure (supplementary) for fig_diag_projection_masking."""

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
from experiments.figures.fig_diag_projection_masking.lib import core
from experiments.figures.fig_diag_projection_masking.lib.data import load_cfg, load_main_and_sweep

PANEL_MODULES = {
    "panel_a_projection_masking_index": "experiments.figures.fig_diag_projection_masking.panels.panel_a_projection_masking_index",
    "panel_b_pca_capture_vs_delta_proj": "experiments.figures.fig_diag_projection_masking.panels.panel_b_pca_capture_vs_delta_proj",
    "panel_c_embedding_ceiling": "experiments.figures.fig_diag_projection_masking.panels.panel_c_embedding_ceiling",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_supp.json")
    compose = cfg["compose"]
    main_data, sweep_rows = load_main_and_sweep()

    apply_publication_style()

    grid = compose["grid"]
    fig = plt.figure(figsize=tuple(compose["figsize"]))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=grid["height_ratios"],
        hspace=float(grid["hspace"]),
        wspace=float(grid["wspace"]),
    )

    ax_hist = fig.add_subplot(gs[0, 0])
    ax_pca = fig.add_subplot(gs[0, 1])
    ax_ceil = fig.add_subplot(gs[1, :])

    panels = compose["panels"]
    for ax, panel_key in zip([ax_hist, ax_pca, ax_ceil], [panels["a"], panels["b"], panels["c"]]):
        mod = importlib.import_module(PANEL_MODULES[panel_key])
        panel_cfg = load_cfg(panel_config=f"{panel_key}.json")["panel"]
        mod.draw(ax, main_data, sweep_rows, panel_cfg)

    fig.suptitle("Supplementary: projection masking diagnostics", fontsize=9, y=0.99)

    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)

