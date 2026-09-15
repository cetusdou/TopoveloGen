"""Compose Figure 2 supplementary S1 from panel_s1_distance.draw()."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from experiments.figures._common.style import apply_publication_style, save_figure
from experiments.figures.fig02_topology_primitives.lib import core
from experiments.figures.fig02_topology_primitives.lib.data import load_all_results, load_cfg
from experiments.figures.fig02_topology_primitives.panels import panel_s1_distance


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_supp.json", panel_config="panel_s1_distance.json")
    compose = cfg["compose"]
    all_results = load_all_results()
    apply_publication_style()
    fig, ax = plt.subplots(figsize=tuple(cfg["panel"]["figsize"]))
    panel_s1_distance.draw(ax, all_results, cfg)
    fig.tight_layout()
    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
