"""Panel: Gaussian random projection distortion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.figures._common.style import apply_publication_style, save_figure
from scripts.figures.fig_bench_velo_test.lib import core
from scripts.figures.fig_bench_velo_test.lib.data import load_cfg, load_bundle
from scripts.figures.fig_bench_velo_test.lib import draw as drawers

PANEL_CONFIG = "panel_gaussian_projection_distortion.json"


def main(out: Path | None = None) -> None:
    cfg = load_cfg(panel_config=PANEL_CONFIG)
    panel = cfg["panel"]
    bundle = load_bundle()

    df_eps = bundle.get("df_eps")
    if df_eps is None:
        print("projection_distortion.csv missing; skipping panel.")
        return

    apply_publication_style()
    fig, ax = plt.subplots(figsize=tuple(panel["figsize"]), constrained_layout=True)
    drawers.draw_projection_distortion_panel(ax, df_eps)

    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out or (core.FIG_DIR / panel["out_name"])
    save_figure(fig, out_path, formats=tuple(panel.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)

