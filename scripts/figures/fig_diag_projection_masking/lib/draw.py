"""Per-panel drawing helpers for fig_diag_projection_masking."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures.fig_diag_projection_masking.lib import core


def _subsample_idx(n: int, n_keep: int, seed: int = 0) -> np.ndarray:
    n_keep = min(n_keep, n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=n_keep, replace=False)) if n_keep > 0 else np.array([], dtype=int)


def region_masking_stats(main: Mapping[str, Any]) -> List[Dict[str, float]]:
    """Return region stats sorted by masking fraction (desc)."""
    mm: np.ndarray = np.asarray(main["mismatch"], dtype=bool)
    labels: np.ndarray = np.asarray(main["region_labels"], dtype=object)
    out: List[Dict[str, float]] = []

    for reg in list(core.REGION_PRIORITY) + ["other"]:
        sel = labels == reg
        n = int(sel.sum())
        if n == 0:
            continue
        out.append(
            {
                "region": reg,
                "n_cells": n,
                "masking_frac": float(mm[sel].mean()),
                "n_masking": int(mm[sel].sum()),
            }
        )

    out.sort(key=lambda r: r["masking_frac"], reverse=True)
    return out


def _binned_mean_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < n_bins:
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.linspace(x.min(), x.max(), n_bins + 1)

    centers: List[float] = []
    means: List[float] = []
    lo: List[float] = []
    hi: List[float] = []

    for i in range(n_bins):
        in_bin = (x >= edges[i]) & (x < edges[i + 1] if i < n_bins - 1 else x <= edges[i + 1])
        if int(in_bin.sum()) < 5:
            continue
        vals = y[in_bin]
        mu = float(np.mean(vals))
        se = float(np.std(vals) / math.sqrt(len(vals)))
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        means.append(mu)
        lo.append(mu - 1.96 * se)
        hi.append(mu + 1.96 * se)

    return np.asarray(centers), np.asarray(means), np.asarray(lo), np.asarray(hi)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 10:
        return float("nan")
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho)


def draw_panel_a_mismatch_vs_gene_count(
    ax: plt.Axes,
    main: Mapping[str, Any],
    sweep: List[Mapping[str, Any]],
    panel_cfg: Mapping[str, Any],
) -> None:
    ng = [int(r["n_genes"]) for r in sweep]
    mm_frac = [float(r["mismatch_frac"]) for r in sweep]

    ax.plot(ng, mm_frac, "o-", color="C1", lw=1.5, ms=4)
    ax.set_xlabel("Total genes")
    ax.set_ylabel("Masking-cell fraction")
    ax.set_title(panel_cfg.get("title", "A  Mismatch persists across gene numbers"))
    ax.set_ylim(0, max(mm_frac) * 1.2 + 0.02)
    ax.text(
        0.03,
        0.97,
        rf"masking: $C^{{2D}}_{{\mathrm{{pred}}}}>{core.TAU_LOW:.1f}$, "
        rf"$C^G_{{\mathrm{{pred}}}}<{core.TAU_HIGH:.1f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6,
    )


def draw_panel_b_decoupled_agreements(
    ax: plt.Axes,
    main: Mapping[str, Any],
    sweep: List[Mapping[str, Any]],
    panel_cfg: Mapping[str, Any],
) -> None:
    ng = [int(r["n_genes"]) for r in sweep]

    ax.plot(
        ng,
        [float(r["mean_cos_high"]) for r in sweep],
        "o-",
        label=r"$C^G_{\mathrm{pred}}$",
        color="C0",
        lw=1.5,
    )
    ax.plot(
        ng,
        [float(r["mean_cos_low"]) for r in sweep],
        "s-",
        label=r"$C^{2D}_{\mathrm{pred}}$",
        color="C1",
        lw=1.5,
    )
    ax.plot(
        ng,
        [float(r["mean_cos_low_gt"]) for r in sweep],
        "^--",
        label=r"$C^{2D}_{\mathrm{GT\ embed}}$",
        color="C2",
        lw=1.2,
    )
    ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.45)
    ax.set_xlabel("Total genes")
    ax.set_ylabel("Mean cosine vs GT")
    ax.set_title(panel_cfg.get("title", "B  Embedding- and gene-space agreements are decoupled"))
    ax.legend(fontsize=5.5, loc="lower left", frameon=False)


def draw_panel_c_gene_module_errors(
    ax: plt.Axes,
    main: Mapping[str, Any],
    sweep: List[Mapping[str, Any]],
    panel_cfg: Mapping[str, Any],
) -> None:
    ng = [int(r["n_genes"]) for r in sweep]

    ax.plot(
        ng,
        [float(r["mean_cos_dynamic"]) for r in sweep],
        "o-",
        label="dynamic genes",
        color="C0",
        lw=1.5,
    )
    ax.plot(
        ng,
        [float(r["mean_cos_noise"]) for r in sweep],
        "s-",
        label="noise genes",
        color="C3",
        lw=1.5,
    )
    ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.45)
    ax.set_xlabel("Total genes")
    ax.set_ylabel(r"Mean $C^G_{\mathrm{pred}}$")
    ax.set_title(panel_cfg.get("title", "C  Gene-space errors concentrate in dynamic genes"))
    ax.legend(fontsize=6, frameon=False)


def draw_panel_d_topology_masking_map(
    ax: plt.Axes,
    main: Mapping[str, Any],
    sweep: List[Mapping[str, Any]],
    panel_cfg: Mapping[str, Any],
) -> None:
    z: np.ndarray = np.asarray(main["z"], dtype=float)
    mm: np.ndarray = np.asarray(main["mismatch"], dtype=bool)
    labels: np.ndarray = np.asarray(main["region_labels"], dtype=object)
    system = main["system"]

    # Pale background: each region's scatter points.
    for reg in list(core.REGION_PRIORITY) + ["other"]:
        sel = labels == reg
        if not bool(sel.any()):
            continue
        ax.scatter(
            z[sel, 0],
            z[sel, 1],
            s=1.8,
            c=core.REGION_COLORS.get(reg, "#DDDDDD"),
            alpha=0.12 if reg != "other" else 0.06,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    # Masking cells as hollow rings.
    if bool(mm.any()):
        ax.scatter(
            z[mm, 0],
            z[mm, 1],
            s=36,
            facecolors="none",
            edgecolors="#CC0000",
            linewidths=1.15,
            rasterized=True,
            zorder=6,
            label="masking cell",
        )

    for edge in system.edges:
        s = np.asarray(edge.start)
        e = np.asarray(edge.end)
        ax.plot([s[0], e[0]], [s[1], e[1]], color="#999999", ls="--", lw=0.65, alpha=0.55, zorder=2)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(panel_cfg.get("title", "D  Masking cells on cycle-bifurcation topology"))
    ax.legend(fontsize=6, loc="lower left", frameon=False, markerscale=0.9)


def draw_panel_e_region_masking_fraction(
    ax: plt.Axes,
    main: Mapping[str, Any],
    sweep: List[Mapping[str, Any]],
    panel_cfg: Mapping[str, Any],
) -> None:
    stats = region_masking_stats(main)
    if not stats:
        ax.set_visible(False)
        return

    # Keep the region order stable for readability (not sorted by masking fraction).
    order = [r for r in core.REGION_PRIORITY if any(s["region"] == r for s in stats)]
    order += [s["region"] for s in stats if s["region"] not in order]

    fracs = [next(s["masking_frac"] for s in stats if s["region"] == r) for r in order]
    colors = [core.REGION_COLORS.get(r, "#AAAAAA") for r in order]
    labels = [r.replace("_", " ") for r in order]
    y = np.arange(len(order))

    ax.barh(y, fracs, color=colors, edgecolor="none", height=0.62, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Masking-cell fraction")
    ax.set_xlim(0, max(fracs) * 1.22 + 0.02)
    ax.invert_yaxis()

    mm: np.ndarray = np.asarray(main["mismatch"], dtype=bool)
    overall = float(mm.mean()) if mm.size else float("nan")

    for i, (f, r) in enumerate(zip(fracs, order)):
        n_mask = next(s["n_masking"] for s in stats if s["region"] == r)
        ax.text(f + 0.008, i, f"{f:.0%} ({n_mask})", va="center", fontsize=6)

    ax.axvline(overall, color="#CC0000", ls="--", lw=0.9, alpha=0.7)
    ax.text(
        overall + 0.005,
        len(order) - 0.35,
        f"overall {overall:.0%}",
        color="#CC0000",
        fontsize=6,
        va="top",
    )

    ax.set_title(panel_cfg.get("title", "E  Masking-cell fraction by region"), fontsize=8, pad=6)


def draw_supp_panel_a_projection_masking_index(
    ax: plt.Axes,
    main: Mapping[str, Any],
    sweep: List[Mapping[str, Any]],
    panel_cfg: Mapping[str, Any],
) -> None:
    mm = np.asarray(main["mismatch"], dtype=bool)
    delta = np.asarray(main["delta_proj"], dtype=float)

    vals = delta[np.isfinite(delta)]
    ax.hist(vals, bins=60, color="#BBBBBB", alpha=0.75, edgecolor="none", label="all cells")
    if bool(mm.any()):
        ax.hist(
            delta[mm],
            bins=60,
            color="#CC0000",
            alpha=0.45,
            edgecolor="none",
            label="masking cells",
        )
    ax.axvline(0.0, color="k", ls="--", lw=0.9, alpha=0.55)
    mean_d = float(np.mean(vals)) if vals.size else float("nan")
    ax.axvline(mean_d, color="k", ls="-", lw=1.0)
    ax.set_xlabel(r"$\Delta_{\mathrm{proj}} = C^{2D}_{\mathrm{pred}} - C^G_{\mathrm{pred}}$")
    ax.set_ylabel("# cells")
    ax.set_title(panel_cfg.get("title", "A  Projection masking index"))
    ax.legend(fontsize=6, frameon=False)
    if np.isfinite(mean_d):
        ax.text(
            0.97,
            0.95,
            rf"mean={mean_d:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )


def draw_supp_panel_b_pca_capture_vs_delta_proj(
    ax: plt.Axes,
    main: Mapping[str, Any],
    sweep: List[Mapping[str, Any]],
    panel_cfg: Mapping[str, Any],
) -> None:
    x = np.asarray(main["pca_frac"], dtype=float)
    y = np.asarray(main["delta_proj"], dtype=float)

    xc, ym, ylo, yhi = _binned_mean_ci(x, y, n_bins=core.PCA_BINNED_N_BINS)
    if xc.size:
        yerr = np.vstack([ym - ylo, yhi - ym])
        ax.errorbar(
            xc,
            ym,
            yerr=yerr,
            fmt="o-",
            color="C0",
            lw=1.2,
            ms=4,
            capsize=2.5,
        )
    rho = _spearman_rho(x, y)
    ax.axhline(0.0, color="k", ls=":", lw=0.6, alpha=0.45)
    ax.set_xlabel("PCA capture of GT velocity")
    ax.set_ylabel(r"$\Delta_{\mathrm{proj}}$")
    ax.set_title(panel_cfg.get("title", r"B  Gene-space capture vs $\Delta_{\mathrm{proj}}$ (weak)"))
    if np.isfinite(rho):
        ax.text(
            0.97,
            0.03,
            rf"Spearman $\rho={rho:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
        )


def draw_supp_panel_c_embedding_ceiling(
    ax: plt.Axes,
    main: Mapping[str, Any],
    sweep: List[Mapping[str, Any]],
    panel_cfg: Mapping[str, Any],
) -> None:
    cgt = np.asarray(main["cos_low_gt"], dtype=float)
    cpred = np.asarray(main["cos_low"], dtype=float)
    mm = np.asarray(main["mismatch"], dtype=bool)
    labels: np.ndarray = np.asarray(main["region_labels"], dtype=object)

    bg = ~mm
    if bool(bg.any()):
        bg_idx = _subsample_idx(int(bg.sum()), min(core.EMBEDDING_BG_SUBSAMPLE_CAP, int(bg.sum())), seed=core.SEED)
        bg_cells = np.where(bg)[0][bg_idx]
        for reg in list(core.REGION_PRIORITY) + ["other"]:
            sel = np.isin(bg_cells, np.where(labels == reg)[0])
            if not bool(sel.any()):
                continue
            idx = bg_cells[sel]
            ax.scatter(
                cgt[idx],
                cpred[idx],
                s=2.0,
                c=core.REGION_COLORS.get(reg, "#DDDDDD"),
                alpha=0.10,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

    if bool(mm.any()):
        ax.scatter(
            cgt[mm],
            cpred[mm],
            s=28,
            facecolors="none",
            edgecolors="#CC0000",
            linewidths=1.1,
            rasterized=True,
            zorder=5,
            label="masking cell",
        )

    gt_ceiling = float(np.nanmean(cgt)) if np.isfinite(np.nanmean(cgt)) else float("nan")

    ax.axhline(
        core.TAU_LOW,
        color="#CC0000",
        ls="--",
        lw=1.0,
        alpha=0.85,
        label=rf"masking threshold ($C^{{2D}}_{{\mathrm{{pred}}}}={core.TAU_LOW}$)",
    )
    if np.isfinite(gt_ceiling):
        ax.axhline(
            gt_ceiling,
            color="#228833",
            ls=":",
            lw=1.0,
            alpha=0.9,
            label=rf"mean $C^{{2D}}_{{\mathrm{{GT\ embed}}}}$={gt_ceiling:.2f}",
        )

    ax.plot([-1, 1], [-1, 1], "k--", lw=0.6, alpha=0.25)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$C^{2D}_{\mathrm{GT\ embed}}$")
    ax.set_ylabel(r"$C^{2D}_{\mathrm{pred}}$")
    ax.set_title(panel_cfg.get("title", "C  Low-dimensional projection imposes an embedding ceiling"))

    top_regs = region_masking_stats(main)[:2]
    reg_txt = ", ".join(f"{r['region']}: {r['masking_frac']:.0%}" for r in top_regs)
    ax.text(
        0.97,
        0.03,
        f"masking: {float(mm.mean()):.1%} ({int(mm.sum())}/{len(mm)})\n"
        f"top regions — {reg_txt}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.92),
    )
    ax.legend(fontsize=5.5, loc="upper left", frameon=False)


DRAWERS_MAIN = {
    "panel_a_mismatch_vs_gene_count": draw_panel_a_mismatch_vs_gene_count,
    "panel_b_decoupled_agreements": draw_panel_b_decoupled_agreements,
    "panel_c_gene_module_errors": draw_panel_c_gene_module_errors,
    "panel_d_topology_masking_map": draw_panel_d_topology_masking_map,
    "panel_e_region_masking_fraction": draw_panel_e_region_masking_fraction,
}

DRAWERS_SUPP = {
    "panel_a_projection_masking_index": draw_supp_panel_a_projection_masking_index,
    "panel_b_pca_capture_vs_delta_proj": draw_supp_panel_b_pca_capture_vs_delta_proj,
    "panel_c_embedding_ceiling": draw_supp_panel_c_embedding_ceiling,
}

