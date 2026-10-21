"""Compose Figure 5 main 2×3 gallery from panel draw() functions."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from scripts.figures._common.style import apply_publication_style, save_figure
from scripts.figures.fig05_kinetic_validation.lib import core
from scripts.figures.fig05_kinetic_validation.lib.data import load_bundle, load_cfg


PANEL_MODULES = {
    "panel_a_trajectory_alpha": "scripts.figures.fig05_kinetic_validation.panels.panel_a_trajectory_alpha",
    "panel_b_sigmoid_traces": "scripts.figures.fig05_kinetic_validation.panels.panel_b_sigmoid_traces",
    "panel_c_impulse_phase": "scripts.figures.fig05_kinetic_validation.panels.panel_c_impulse_phase",
    "panel_d_impulse_peaks": "scripts.figures.fig05_kinetic_validation.panels.panel_d_impulse_peaks",
    "panel_e_gamma_hysteresis": "scripts.figures.fig05_kinetic_validation.panels.panel_e_gamma_hysteresis",
    "panel_f_velocity_consistency": "scripts.figures.fig05_kinetic_validation.panels.panel_f_velocity_consistency",
}


def main(out: Path | None = None) -> None:
    cfg = load_cfg(compose_config="compose_main.json")
    compose = cfg["compose"]
    bundle = load_bundle(with_gamma=True)
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

    tl = compose.get("tight_layout") or {}
    fig.tight_layout(**{k: float(v) for k, v in tl.items()})

    out_path = out or (core.FIG_DIR / compose["out_name"])
    save_figure(fig, out_path, formats=tuple(compose.get("formats", ("png", "pdf"))))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    main(args.out)
