"""Figure 6 panel L: Unspliced Fraction (Global Kinetics)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from experiments.figures._common.style import apply_publication_style, save_figure
from experiments.figures.fig06_statistical_realism.lib import core
from experiments.figures.fig06_statistical_realism.lib.data import load_bundle, load_cfg
from experiments.figures.fig06_statistical_realism.lib.draw import draw_l

PANEL_CONFIG = "panel_l_unspliced_frac.json"


def draw(ax, bundle, cfg):
    """Draw this panel onto *ax*. Used by compose_main."""
    panel = cfg["panel"]
    draw_l(ax, bundle, panel, cfg=cfg)


def main(out: Path | None = None) -> None:
    cfg = load_cfg(panel_config=PANEL_CONFIG)
    panel = cfg["panel"]
    bundle = load_bundle()
    apply_publication_style()
    fig, ax = plt.subplots(figsize=tuple(panel["figsize"]), constrained_layout=True)
    draw(ax, bundle, cfg)
    out_path = out or (core.FIG_DIR / panel["out_name"])
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=tuple(panel.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
