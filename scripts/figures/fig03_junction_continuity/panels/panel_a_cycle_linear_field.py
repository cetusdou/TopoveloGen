"""Figure 3 panel A: Cycle–linear potential & drift."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from scripts.figures._common.style import apply_publication_style, save_figure
from scripts.figures.fig03_junction_continuity.lib import core
from scripts.figures.fig03_junction_continuity.lib.data import load_cfg, load_one_result
from scripts.figures.fig03_junction_continuity.lib.draw import draw_field_panel

PANEL_CONFIG = "panel_a_cycle_linear_field.json"


def draw(ax, result, cfg):
    draw_field_panel(ax, result, cfg["panel"])


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
