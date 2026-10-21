"""Figure 2 panel F: Cycle–bifurcation hybrid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from scripts.figures._common.style import apply_publication_style, save_figure
from scripts.figures.fig02_topology_primitives.lib.data import load_cfg, load_one_result
from scripts.figures.fig02_topology_primitives.lib.draw import draw_topology_panel
from scripts.figures.fig02_topology_primitives.lib import core

PANEL_CONFIG = "panel_f_cycle_bifurcation.json"


def draw(ax, result, cfg):
    """Draw this panel onto *ax*. Used by compose_main."""
    panel = cfg["panel"]
    draw_topology_panel(ax, result, panel, seed=int(cfg.get("seed", core.SEED)))


def main(out: Path | None = None) -> None:
    cfg = load_cfg(panel_config=PANEL_CONFIG)
    panel = cfg["panel"]
    result = load_one_result(panel["topology"])
    apply_publication_style()
    fig, ax = plt.subplots(figsize=tuple(panel["figsize"]), constrained_layout=True)
    draw(ax, result, cfg)
    out_path = out or (core.FIG_DIR / panel["out_name"])
    save_figure(fig, out_path, formats=tuple(panel.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
