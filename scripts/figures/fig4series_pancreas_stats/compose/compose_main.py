"""Compose Figure4Series main 3×3 validation (panels A–I)."""

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
from scripts.figures.fig4series_pancreas_stats.lib import core
from scripts.figures.fig4series_pancreas_stats.lib.data import load_bundle, load_cfg

PANEL_MODULES = {
    "panel_a_library_size": "scripts.figures.fig4series_pancreas_stats.panels.panel_a_library_size",
    "panel_b_genes_detected": "scripts.figures.fig4series_pancreas_stats.panels.panel_b_genes_detected",
    "panel_c_mean_expression": "scripts.figures.fig4series_pancreas_stats.panels.panel_c_mean_expression",
    "panel_d_overdispersion": "scripts.figures.fig4series_pancreas_stats.panels.panel_d_overdispersion",
    "panel_e_dropout_vs_mean": "scripts.figures.fig4series_pancreas_stats.panels.panel_e_dropout_vs_mean",
    "panel_f_unspliced_frac": "scripts.figures.fig4series_pancreas_stats.panels.panel_f_unspliced_frac",
    "panel_g_sparsity": "scripts.figures.fig4series_pancreas_stats.panels.panel_g_sparsity",
    "panel_h_fano_factor": "scripts.figures.fig4series_pancreas_stats.panels.panel_h_fano_factor",
    "panel_i_rank_frequency": "scripts.figures.fig4series_pancreas_stats.panels.panel_i_rank_frequency",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_main.json")
    compose = cfg["compose"]
    bundle = load_bundle(cfg)
    apply_publication_style()

    fig, axes = plt.subplots(
        int(compose["nrows"]),
        int(compose["ncols"]),
        figsize=tuple(compose["figsize"]),
        dpi=int(compose.get("dpi", core.FIGURE_MAIN_DPI)),
    )
    plt.subplots_adjust(hspace=float(compose.get("hspace", 0.40)), wspace=float(compose.get("wspace", 0.30)))
    axes_flat = axes.flatten()

    i = 0
    for row in compose["panels"]:
        for panel_key in row:
            mod = importlib.import_module(PANEL_MODULES[panel_key])
            panel_cfg = load_cfg(panel_config=f"{panel_key}.json")
            mod.draw(axes_flat[i], bundle, panel_cfg)
            i += 1

    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
