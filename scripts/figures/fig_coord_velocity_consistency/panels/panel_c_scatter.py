"""Coordinate consistency panel C."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from scripts.figures._common.style import apply_publication_style, save_figure
from scripts.figures.fig_coord_velocity_consistency.lib import core
from scripts.figures.fig_coord_velocity_consistency.lib.data import load_bundle, load_cfg
from scripts.figures.fig_coord_velocity_consistency.lib.draw import draw_scatter

PANEL_CONFIG = "panel_c_scatter.json"


def draw(ax, bundle, cfg):
    draw_scatter(ax, bundle, panel_cfg=cfg['panel'], shared_cfg=cfg)


def main(out: Path | None = None) -> None:
    cfg = load_cfg(panel_config=PANEL_CONFIG)
    panel = cfg["panel"]
    bundle = load_bundle() if "1" == "1" else {}
    apply_publication_style()
    fig, ax = plt.subplots(figsize=tuple(panel["figsize"]))
    draw(ax, bundle, cfg)
    fig.tight_layout()
    out_path = out or (core.FIG_DIR / panel["out_name"])
    save_figure(fig, out_path, formats=tuple(panel.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
