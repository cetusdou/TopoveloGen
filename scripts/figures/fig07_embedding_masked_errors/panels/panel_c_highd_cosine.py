"""Figure 7 panel C: High-D velocity cosine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from scripts.figures._common.style import apply_publication_style, save_figure
from scripts.figures.fig07_embedding_masked_errors.lib import core
from scripts.figures.fig07_embedding_masked_errors.lib.data import load_cfg, load_main_and_sweep
from scripts.figures.fig07_embedding_masked_errors.lib.draw import draw_c

PANEL_CONFIG = "panel_c_highd_cosine.json"


def draw(ax, main, cfg, sweep=None):
    """Draw this panel onto *ax*. Used by compose_main."""
    panel = cfg["panel"]
    draw_c(ax, main, panel, fig=ax.figure)


def main(out: Path | None = None) -> None:
    cfg = load_cfg(panel_config=PANEL_CONFIG)
    panel = cfg["panel"]
    main_data, _sweep = load_main_and_sweep()
    apply_publication_style()
    fig, ax = plt.subplots(figsize=tuple(panel["figsize"]), constrained_layout=True)
    draw(ax, main_data, cfg)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out or (core.FIG_DIR / panel["out_name"])
    save_figure(fig, out_path, formats=tuple(panel.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
