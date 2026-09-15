"""Compose Figure4Series supplementary 2×2 (panels S1–S4; formerly G–J)."""

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
from experiments.figures.fig4series_pancreas_stats.lib import core
from experiments.figures.fig4series_pancreas_stats.lib.data import load_bundle, load_cfg

PANEL_MODULES = {
    "panel_s1_latent_library": "experiments.figures.fig4series_pancreas_stats.panels.panel_s1_latent_library",
    "panel_s2_spliced_unspliced": "experiments.figures.fig4series_pancreas_stats.panels.panel_s2_spliced_unspliced",
    "panel_s3_sparsity_compare": "experiments.figures.fig4series_pancreas_stats.panels.panel_s3_sparsity_compare",
    "panel_s4_mean_variance": "experiments.figures.fig4series_pancreas_stats.panels.panel_s4_mean_variance",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_supp.json")
    compose = cfg["compose"]
    bundle = load_bundle(cfg)
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
