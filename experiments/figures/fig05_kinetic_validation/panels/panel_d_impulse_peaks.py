"""Figure 5 panel D: Impulse peak ordering t_α < t_u < t_s."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from experiments.figures._common.style import apply_publication_style, save_figure
from experiments.figures.fig05_kinetic_validation.lib import core
from experiments.figures.fig05_kinetic_validation.lib.data import load_bundle, load_cfg
from experiments.figures.fig05_kinetic_validation.lib.draw import draw_d

PANEL_CONFIG = "panel_d_impulse_peaks.json"


def draw(ax, bundle, cfg):
    draw_d(ax, bundle, cfg["panel"])


def main(out: Path | None = None) -> None:
    cfg = load_cfg(panel_config=PANEL_CONFIG)
    panel = cfg["panel"]
    bundle = load_bundle(with_gamma=False)
    apply_publication_style()
    fig, ax = plt.subplots(figsize=tuple(panel["figsize"]), constrained_layout=True)
    draw(ax, bundle, cfg)
    out_path = out or (core.FIG_DIR / panel["out_name"])
    save_figure(fig, out_path, formats=tuple(panel.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
