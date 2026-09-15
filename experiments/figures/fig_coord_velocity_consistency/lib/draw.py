"""Drawing helpers for coordinate-consistency panels."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _colors(cfg: Mapping[str, Any]) -> Dict[str, str]:
    defaults = {
        "abs": "#4A90E2",
        "lib": "#2ECC71",
        "prop": "#D0021B",
        "log": "#E87722",
    }
    return {**defaults, **dict(cfg.get("colors") or {})}


def draw_diagram(ax: plt.Axes, panel_cfg: Mapping[str, Any], shared_cfg: Mapping[str, Any]) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(panel_cfg.get("title", "A. Ground-truth velocity coordinates"), fontsize=11)

    boxes = [
        (0.4, 3.4, 2.4, 1.6, "absolute\n$v^{abs}=ds/dt$", "#4A90E2"),
        (3.8, 3.4, 2.4, 1.6, "library-scaled\n$\\tilde v=c\\,v^{abs}$", "#2ECC71"),
        (7.2, 4.2, 2.4, 1.4, "proportion\n$v^{prop}=dp/dt$", "#D0021B"),
        (7.2, 2.0, 2.4, 1.4, "log-approx\n$v^{log}\\propto\\frac{\\rho}{1+\\rho s}v^{abs}$", "#E87722"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    ax.add_patch(FancyArrowPatch((2.9, 4.2), (3.7, 4.2), arrowstyle="->", mutation_scale=12, color="0.3"))
    ax.add_patch(FancyArrowPatch((6.3, 4.5), (7.1, 4.8), arrowstyle="->", mutation_scale=12, color="0.3"))
    ax.add_patch(FancyArrowPatch((6.3, 3.9), (7.1, 2.9), arrowstyle="->", mutation_scale=12, color="0.3"))
    ax.text(5.0, 1.0, "cosine invariant under cell-wise $c_i>0$\nproportion / log change direction", ha="center", fontsize=8, color="0.25")


def draw_cosine_distributions(
    ax: plt.Axes,
    bundle: Mapping[str, Any],
    panel_cfg: Mapping[str, Any],
    shared_cfg: Mapping[str, Any],
) -> None:
    colors = _colors(shared_cfg)
    keys = list(panel_cfg.get("keys", ["cos_abs", "cos_lib", "cos_prop", "cos_log"]))
    labels = list(panel_cfg.get("labels", ["absolute", "library-scaled", "proportion", "log-approx"]))
    color_keys = ["abs", "lib", "prop", "log"]
    per = bundle["per_cell"]

    data = [np.asarray(per[k], dtype=np.float64) for k in keys]
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[color_keys[i]])
        body.set_alpha(0.55)
        body.set_edgecolor("0.2")
    if "cmedians" in parts:
        parts["cmedians"].set_color("0.15")
        parts["cmedians"].set_linewidth(1.2)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("cosine(scVelo, GT)", fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8)
    ax.set_title(panel_cfg.get("title", "B. Cosine by GT coordinate"), fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)


def draw_scatter(
    ax: plt.Axes,
    bundle: Mapping[str, Any],
    panel_cfg: Mapping[str, Any],
    shared_cfg: Mapping[str, Any],
) -> None:
    colors = _colors(shared_cfg)
    per = bundle["per_cell"]
    cos_abs = np.asarray(per["cos_abs"], dtype=np.float64)
    cos_prop = np.asarray(per["cos_prop"], dtype=np.float64)
    cos_log = np.asarray(per["cos_log"], dtype=np.float64)

    max_points = int(panel_cfg.get("max_points", 2500))
    rng = np.random.default_rng(int(shared_cfg.get("seed", 1)))
    if cos_abs.size > max_points:
        idx = rng.choice(cos_abs.size, max_points, replace=False)
        cos_abs, cos_prop, cos_log = cos_abs[idx], cos_prop[idx], cos_log[idx]

    ax.scatter(
        cos_abs, cos_prop,
        s=float(panel_cfg.get("point_size", 6)),
        c=colors["prop"],
        alpha=float(panel_cfg.get("point_alpha", 0.35)),
        linewidths=0,
        rasterized=True,
        label="proportion",
    )
    ax.scatter(
        cos_abs, cos_log,
        s=float(panel_cfg.get("point_size", 6)) * 0.85,
        c=colors["log"],
        alpha=float(panel_cfg.get("point_alpha", 0.35)) * 0.85,
        linewidths=0,
        rasterized=True,
        label="log-approx",
    )
    lims = [-1.05, 1.05]
    ax.plot(lims, lims, "k--", lw=1.0, alpha=0.6, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\cos_{abs}$", fontsize=9)
    ax.set_ylabel(r"$\cos_{prop}$ / $\cos_{log}$", fontsize=9)
    ax.set_title(panel_cfg.get("title", "C. cos_abs vs cos_prop / cos_log"), fontsize=11)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.35)


def draw_summary(
    ax: plt.Axes,
    bundle: Mapping[str, Any],
    panel_cfg: Mapping[str, Any],
    shared_cfg: Mapping[str, Any],
) -> None:
    ax.axis("off")
    ax.set_title(panel_cfg.get("title", "D. Summary and method coordinates"), fontsize=11)
    summary = bundle["summary"]
    fs = float(panel_cfg.get("fontsize", 7))

    rows = [
        ["coordinate", "mean cos", "median cos", "Spearman vs abs", "flip vs abs"],
        [
            "absolute",
            f"{summary['abs_mean']:.3f}",
            f"{summary['abs_median']:.3f}",
            "1.000",
            "—",
        ],
        [
            "library-scaled",
            f"{summary['lib_mean']:.3f}",
            f"{summary['lib_median']:.3f}",
            f"{summary['spearman_abs_lib']:.3f}",
            f"Δmax={summary['max_abs_diff_abs_lib']:.2e}",
        ],
        [
            "proportion",
            f"{summary['prop_mean']:.3f}",
            f"{summary['prop_median']:.3f}",
            f"{summary['spearman_abs_prop']:.3f}",
            f"{summary['flip_abs_prop']:.3f}",
        ],
        [
            "log-approx",
            f"{summary['log_mean']:.3f}",
            f"{summary['log_median']:.3f}",
            f"{summary['spearman_abs_log']:.3f}",
            f"{summary['flip_abs_log']:.3f}",
        ],
    ]
    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="upper center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fs)
    table.scale(1.05, 1.25)

    methods = bundle.get("method_table") or []
    if methods:
        mrows = [[m.get("method", ""), m.get("velocity_output_coordinate", ""), m.get("compared_gt", "")] for m in methods]
        mtable = ax.table(
            cellText=mrows,
            colLabels=["method", "velocity output coordinate", "compared GT"],
            loc="lower center",
            cellLoc="left",
            bbox=[0.0, 0.02, 1.0, 0.38],
        )
        mtable.auto_set_font_size(False)
        mtable.set_fontsize(max(fs - 0.5, 6))
        mtable.scale(1.0, 1.15)

    ax.text(
        0.5,
        0.46,
        f"tau_high={summary.get('tau_high', 0.7):.2f}; "
        f"n_cells={int(summary.get('n_cells', 0))}, n_genes={int(summary.get('n_genes', 0))}",
        transform=ax.transAxes,
        ha="center",
        fontsize=fs,
        color="0.3",
    )
