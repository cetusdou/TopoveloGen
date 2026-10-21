"""Compose Figure for fig_bench_velo_test (legacy-compatible composite)."""

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
from scripts.figures.fig_bench_velo_test.lib import core
from scripts.figures.fig_bench_velo_test.lib.data import load_cfg, load_bundle

PANEL_MODULES = {
    "panel_umap_true_time": "scripts.figures.fig_bench_velo_test.panels.panel_umap_true_time",
    "panel_gt_velocity_grid": "scripts.figures.fig_bench_velo_test.panels.panel_gt_velocity_grid",
    "panel_deepvelo_velocity_grid": "scripts.figures.fig_bench_velo_test.panels.panel_deepvelo_velocity_grid",
    "panel_unitvelo_velocity_grid": "scripts.figures.fig_bench_velo_test.panels.panel_unitvelo_velocity_grid",
    "panel_scvelo_velocity_grid": "scripts.figures.fig_bench_velo_test.panels.panel_scvelo_velocity_grid",
    "panel_gt_velocity_streamlines": "scripts.figures.fig_bench_velo_test.panels.panel_gt_velocity_streamlines",
    "panel_deepvelo_velocity_streamlines": "scripts.figures.fig_bench_velo_test.panels.panel_deepvelo_velocity_streamlines",
    "panel_unitvelo_velocity_streamlines": "scripts.figures.fig_bench_velo_test.panels.panel_unitvelo_velocity_streamlines",
    "panel_scvelo_velocity_streamlines": "scripts.figures.fig_bench_velo_test.panels.panel_scvelo_velocity_streamlines",
    "panel_gene_cosine_histogram": "scripts.figures.fig_bench_velo_test.panels.panel_gene_cosine_histogram",
    "panel_gene_mse_histogram": "scripts.figures.fig_bench_velo_test.panels.panel_gene_mse_histogram",
    "panel_lowdim_cosine_histogram": "scripts.figures.fig_bench_velo_test.panels.panel_lowdim_cosine_histogram",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_main.json")
    compose = cfg["compose"]
    bundle = load_bundle()

    apply_publication_style()

    fig = plt.figure(figsize=tuple(compose["figsize"]), constrained_layout=True)
    gs = fig.add_gridspec(3, 4, height_ratios=compose["height_ratios"])

    # Legacy composite uses irregular row subgrids:
    # - row0: 5 panels
    # - row1: 4 panels
    # - row2: 3 panels
    axes_for_panel_key: dict[str, list[plt.Axes]] = {}
    for r_idx, row in enumerate(compose["rows"]):
        sub = gs[r_idx, :].subgridspec(1, len(row), wspace=float(compose["row_wspaces"][r_idx]))
        axes_for_panel_key[compose["rows"][r_idx][0] if row else f"row{r_idx}"] = []
        # We'll just construct axes in draw loop below.

    for r_idx, row in enumerate(compose["rows"]):
        sub = gs[r_idx, :].subgridspec(1, len(row), wspace=float(compose["row_wspaces"][r_idx]))
        for c_idx, panel_key in enumerate(row):
            ax = fig.add_subplot(sub[0, c_idx])
            mod = importlib.import_module(PANEL_MODULES[panel_key])
            panel_cfg = load_cfg(panel_config=f"{panel_key}.json")["panel"]
            mod.draw(ax, bundle, panel_cfg)

    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)

