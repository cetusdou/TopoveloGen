"""Compose Figure 6 main 4×3 gallery from panel draw() functions."""

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
from scripts.figures.fig06_statistical_realism.lib import core
from scripts.figures.fig06_statistical_realism.lib.data import load_bundle, load_cfg

PANEL_MODULES = {
    "panel_a_cells_library": "scripts.figures.fig06_statistical_realism.panels.panel_a_cells_library",
    "panel_b_library_poisson": "scripts.figures.fig06_statistical_realism.panels.panel_b_library_poisson",
    "panel_c_mean_variance": "scripts.figures.fig06_statistical_realism.panels.panel_c_mean_variance",
    "panel_d_dropout": "scripts.figures.fig06_statistical_realism.panels.panel_d_dropout",
    "panel_e_spliced_unspliced": "scripts.figures.fig06_statistical_realism.panels.panel_e_spliced_unspliced",
    "panel_f_sparsity_sweep": "scripts.figures.fig06_statistical_realism.panels.panel_f_sparsity_sweep",
    "panel_g_library_kde": "scripts.figures.fig06_statistical_realism.panels.panel_g_library_kde",
    "panel_h_detected_genes": "scripts.figures.fig06_statistical_realism.panels.panel_h_detected_genes",
    "panel_i_mean_expression": "scripts.figures.fig06_statistical_realism.panels.panel_i_mean_expression",
    "panel_j_overdispersion": "scripts.figures.fig06_statistical_realism.panels.panel_j_overdispersion",
    "panel_k_dropout_vs_mean": "scripts.figures.fig06_statistical_realism.panels.panel_k_dropout_vs_mean",
    "panel_l_unspliced_frac": "scripts.figures.fig06_statistical_realism.panels.panel_l_unspliced_frac",
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
    )
    axes_flat = axes.flatten()

    i = 0
    for row in compose["panels"]:
        for panel_key in row:
            mod = importlib.import_module(PANEL_MODULES[panel_key])
            panel_cfg = load_cfg(panel_config=f"{panel_key}.json")
            mod.draw(axes_flat[i], bundle, panel_cfg)
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
