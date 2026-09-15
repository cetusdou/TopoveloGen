"""Panel: Projection masking index Δ_proj (Δ_proj facets)."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.figures._common.style import apply_publication_style, save_figure
from experiments.figures.fig_bench_velo_test.lib import core
from experiments.figures.fig_bench_velo_test.lib.data import load_cfg, load_bundle
from experiments.figures.fig_bench_velo_test.lib import draw as drawers

PANEL_CONFIG = "panel_projection_masking_index.json"


def main(out: Path | None = None) -> None:
    cfg = load_cfg(panel_config=PANEL_CONFIG)
    panel = cfg["panel"]
    bundle = load_bundle()

    apply_publication_style()

    methods = [m for m in core.METHOD_ORDER if m in bundle["cos_gene"] and m in bundle["cos_lowdim"]]
    if not methods:
        print("No matching methods found for Δ_proj facets; skipping.")
        return

    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=tuple(panel["figsize"]),
        constrained_layout=True,
        sharey=True,
    )
    if len(methods) == 1:
        axes = [axes]

    drawers.draw_projection_masking_index_facets(
        axes,
        bundle["cos_gene"],
        bundle["cos_lowdim"],
        method_order=methods,
        method_colors=core.METHOD_COLORS,
        bins=core.PROJECTION_BINS,
        xlim=core.PROJECTION_XLIM,
    )
    if axes:
        axes[0].set_ylabel(r"# cells")

    fig.suptitle(
        r"Projection masking index  $\Delta_{\mathrm{proj}} = "
        r"\mathrm{Cos}_{\mathrm{UMAP}} - \mathrm{Cos}_{\mathrm{gene}}$",
        fontsize=9,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        drawers.PROJECTION_MASKING_CAPTION if hasattr(drawers, "PROJECTION_MASKING_CAPTION") else "",
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        transform=fig.transFigure,
    )

    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out or (core.FIG_DIR / panel["out_name"])
    save_figure(fig, out_path, formats=tuple(panel.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)

