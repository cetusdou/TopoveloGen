"""Shared draw helpers for Fig7 embedding-masked error panels A–F."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from experiments.figures.fig07_embedding_masked_errors.lib import core


def quiver_subsample(ax, xy, v, stride=None, scale=None, color="C0", alpha=0.85) -> None:
    stride = int(stride if stride is not None else core.PLOT_KW["quiver_stride"])
    scale = float(scale if scale is not None else core.PLOT_KW["quiver_scale"])
    n = len(xy)
    idx = np.arange(0, n, max(1, stride))
    v_s = v[idx]
    ax.quiver(
        xy[idx, 0],
        xy[idx, 1],
        v_s[:, 0],
        v_s[:, 1],
        angles="xy",
        scale_units="xy",
        scale=scale,
        color=color,
        alpha=alpha,
        width=0.0025,
        zorder=4,
    )


def draw_a(ax, main, panel_cfg: Mapping) -> None:
    z = main["z"]
    ax.scatter(z[:, 0], z[:, 1], s=1.5, c="#CCCCCC", linewidths=0, rasterized=True)
    quiver_subsample(ax, z, main["v_latent_gt"], color="C0")
    ax.set_title(panel_cfg.get("title", "A  Ground-truth latent velocity"))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")


def draw_b(ax, main, panel_cfg: Mapping) -> None:
    z = main["z"]
    ax.scatter(z[:, 0], z[:, 1], s=1.5, c="#DDDDDD", linewidths=0, rasterized=True)
    quiver_subsample(ax, z, main["velocity_umap"], color="C1")
    ax.set_title(panel_cfg.get("title", "B  scVelo embedding velocity"))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")


def draw_c(ax, main, panel_cfg: Mapping, fig=None) -> None:
    z = main["z"]
    sc_c = ax.scatter(
        z[:, 0],
        z[:, 1],
        c=main["cos_high"],
        s=2.5,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(panel_cfg.get("title", "C  High-D velocity cosine"))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    cbar = (fig or ax.figure).colorbar(sc_c, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(r"$C^{G}$", labelpad=2)


def draw_d(ax, main, panel_cfg: Mapping) -> None:
    z = main["z"]
    ax.scatter(main["cos_low"], main["cos_high"], s=2.5, alpha=0.2, c="#666666", linewidths=0, rasterized=True)
    ax.axvline(core.TAU_LOW, color="C1", ls="--", lw=1.0)
    ax.axhline(core.TAU_HIGH, color="C1", ls="--", lw=1.0)
    ax.set_xlabel(r"Low-D cosine ($C^{2D}$)")
    ax.set_ylabel(r"High-D cosine ($C^{G}$)")
    ax.set_title(panel_cfg.get("title", "D  Embedding vs gene-space agreement"))
    n_mm = int(main["mismatch"].sum())
    ax.text(
        0.98,
        0.02,
        f"mismatch = {n_mm}/{len(z)} ({main['mismatch_frac']:.1%})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )


def draw_e(ax, main, panel_cfg: Mapping) -> None:
    z = main["z"]
    mm = main["mismatch"]
    system = main["system"]
    ax.scatter(z[~mm, 0], z[~mm, 1], s=1.5, c="#BBBBBB", linewidths=0, label="agreement", rasterized=True)
    ax.scatter(
        z[mm, 0],
        z[mm, 1],
        s=5,
        c="C1",
        linewidths=0,
        label="low-D ok, high-D wrong",
        rasterized=True,
    )
    for edge in system.edges:
        s, e = np.asarray(edge.start), np.asarray(edge.end)
        ax.plot([s[0], e[0]], [s[1], e[1]], color="#888888", ls="--", lw=0.8, alpha=0.6)
    ax.set_title(panel_cfg.get("title", "E  Mismatch cells on topology"))
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=1.5)


def draw_f(ax, sweep, panel_cfg: Mapping) -> None:
    order = list(core.SWEEP.get("topologies", ["linear", "bifurcation", "cycle", "cycle_linear", "cycle_bifurcation"]))
    fracs = [next(r["mismatch_frac"] for r in sweep if r["topology"] == t) for t in order]
    labels = [t.replace("_", "-") for t in order]
    ax.bar(range(len(order)), fracs, color="C0", edgecolor="none", width=0.72)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mismatch fraction")
    ax.set_ylim(0, max(fracs) * 1.25 + 0.01 if fracs else 1.0)
    ax.set_title(panel_cfg.get("title", "F  Mismatch vs topology complexity"))


DRAWERS = {
    "A": draw_a,
    "B": draw_b,
    "C": draw_c,
    "D": draw_d,
    "E": draw_e,
    "F": draw_f,
}
