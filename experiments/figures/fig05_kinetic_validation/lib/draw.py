"""Shared draw helpers for Fig5 kinetic panels."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt

from experiments.figures.fig05_kinetic_validation.lib import core


def draw_a(ax: plt.Axes, bundle: Mapping, panel_cfg: Mapping) -> None:
    core.plot_panel_a_trajectory_alpha(
        ax, bundle["system"], bundle["traj"], bundle["result_default"], panel_cfg["title"]
    )


def draw_b(ax: plt.Axes, bundle: Mapping, panel_cfg: Mapping) -> None:
    core.plot_panel_b_sigmoid_traces(
        ax, bundle["result_default"], bundle["steps"], panel_cfg["title"]
    )


def draw_c(ax: plt.Axes, bundle: Mapping, panel_cfg: Mapping) -> None:
    core.plot_panel_c_impulse_phase(
        ax, bundle["result_default"], bundle["steps"], panel_cfg["title"]
    )


def draw_d(ax: plt.Axes, bundle: Mapping, panel_cfg: Mapping) -> None:
    core.plot_panel_d_impulse_peaks(
        ax, bundle["result_default"], bundle["steps"], panel_cfg["title"]
    )


def draw_e(ax: plt.Axes, bundle: Mapping, panel_cfg: Mapping) -> None:
    if "gamma_results" not in bundle:
        raise KeyError("bundle missing gamma_results; load_bundle(with_gamma=True)")
    core.plot_panel_e_gamma_hysteresis(
        ax, bundle["gamma_results"], bundle["steps"], panel_cfg["title"]
    )


def draw_f(ax: plt.Axes, bundle: Mapping, panel_cfg: Mapping) -> None:
    core.plot_panel_f_velocity_consistency(
        ax, bundle["result_default"], bundle["steps"], panel_cfg["title"]
    )
