"""Drawing helpers for fig_bench_velo_test."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from experiments.figures.fig_bench_velo_test.lib import core
from experiments.projection_masking_metrics import PROJECTION_MASKING_CAPTION


def _color_key(adata: Any) -> Optional[str]:
    if core.COLOR_KEY and hasattr(adata, "obs") and core.COLOR_KEY in adata.obs.columns:
        return core.COLOR_KEY
    return None


def _finalize_umap_ax(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def draw_umap_true_time_panel(adata: Any, ax: plt.Axes, title: str = "Raw UMAP") -> None:
    # scvelo is only required at draw-time.
    import scvelo as scv

    scv.settings.verbosity = 0
    ckey = _color_key(adata)
    if ckey is not None:
        try:
            scv.pl.scatter(
                adata,
                basis="umap",
                color=ckey,
                color_map="viridis",
                colorbar=False,
                ax=ax,
                show=False,
            )
            _finalize_umap_ax(ax, title)
            return
        except Exception:
            pass

    X = np.asarray(adata.obsm["X_umap"])
    if ckey is not None:
        c = np.asarray(adata.obs[ckey])
        ax.scatter(
            X[:, 0],
            X[:, 1],
            s=4,
            c=c,
            cmap="viridis",
            alpha=0.85,
            linewidths=0,
            rasterized=True,
        )
    else:
        ax.scatter(X[:, 0], X[:, 1], s=4, c="#BFBFBF", alpha=0.85, linewidths=0, rasterized=True)

    _finalize_umap_ax(ax, title)


def _infer_velocity_layer(adata: Any, preferred: tuple[str, ...]) -> str:
    for k in preferred:
        if k in adata.layers:
            return k
    for k in ("velocity", "velocity_s", "velocity_u", "velo", "V"):
        if k in adata.layers:
            return k
    raise KeyError(f"No velocity layer in {list(adata.layers.keys())}")


def draw_velocity_stream_panel(adata: Any, vkey: str, ax: plt.Axes, title: str) -> None:
    import scvelo as scv

    scv.settings.verbosity = 0
    ckey = _color_key(adata)
    scv.pl.velocity_embedding_stream(
        adata,
        basis="umap",
        vkey=vkey,
        color=ckey,
        ax=ax,
        **core.STREAM_KW,
    )
    _finalize_umap_ax(ax, title)


def draw_velocity_grid_panel(adata: Any, vkey: str, ax: plt.Axes, title: str) -> bool:
    import scvelo as scv

    scv.settings.verbosity = 0
    graph_key = f"{vkey}_graph"
    umap_key = f"{vkey}_umap"
    if graph_key not in getattr(adata, "uns", {}) or umap_key not in getattr(adata, "obsm", {}):
        ax.set_visible(False)
        return False

    ckey = _color_key(adata)
    scv.pl.velocity_embedding_grid(
        adata,
        basis="umap",
        vkey=vkey,
        color=ckey,
        colorbar=False,
        ax=ax,
        show=False,
    )
    _finalize_umap_ax(ax, title)
    return True


def draw_gene_cosine_histogram_panel(cos_gene: Mapping[str, np.ndarray], ax: plt.Axes) -> None:
    for name, vals in cos_gene.items():
        plot_vals = np.asarray(vals)[np.isfinite(vals)]
        ax.hist(
            plot_vals,
            bins=core.HIST_BINS,
            color=core.METHOD_COLORS.get(name, "#888888"),
            alpha=core.HIST_ALPHA,
            label=f"{name} vs GT",
        )
    ax.set_title("Gene-space cosine vs GT")
    ax.set_xlabel("cosine")
    ax.set_ylabel("# cells")
    ax.legend(frameon=False)


def draw_gene_mse_histogram_panel(mse_gene: Mapping[str, np.ndarray], ax: plt.Axes) -> None:
    for name, vals in mse_gene.items():
        plot_vals = np.asarray(vals)[np.isfinite(vals)]
        if name == "unitvelo":
            plot_vals = plot_vals / core.UNITVELO_MSE_SCALE
            label = f"{name} vs GT (/10)"
        else:
            label = f"{name} vs GT"
        ax.hist(
            plot_vals,
            bins=core.HIST_BINS,
            color=core.METHOD_COLORS.get(name, "#888888"),
            alpha=core.HIST_ALPHA,
            label=label,
        )
    ax.set_title("Gene-space MSE vs GT")
    ax.set_xlabel("MSE")
    ax.legend(frameon=False)


def draw_lowdim_cosine_histogram_panel(cos_lowdim: Mapping[str, np.ndarray], basis: str, ax: plt.Axes) -> None:
    for name, vals in cos_lowdim.items():
        plot_vals = np.asarray(vals)[np.isfinite(vals)]
        ax.hist(
            plot_vals,
            bins=core.HIST_BINS,
            color=core.METHOD_COLORS.get(name, "#888888"),
            alpha=core.HIST_ALPHA,
            label=f"{name} vs GT",
        )
    ax.set_title(f"Low-dim cosine vs GT ({basis})")
    ax.set_xlabel("cosine")
    ax.set_ylabel("# cells")
    ax.legend(frameon=False)


def draw_projection_masking_index_facets(
    ax_row: list[plt.Axes],
    cos_gene: Mapping[str, np.ndarray],
    cos_lowdim: Mapping[str, np.ndarray],
    *,
    method_order: list[str],
    method_colors: Mapping[str, str],
    bins: int = 60,
    xlim: tuple[float, float] = (-1.05, 1.05),
) -> None:
    for ax, name in zip(ax_row, method_order):
        if name not in cos_gene or name not in cos_lowdim:
            ax.set_visible(False)
            continue
        delta = np.asarray(cos_lowdim[name], dtype=float) - np.asarray(cos_gene[name], dtype=float)
        vals = delta[np.isfinite(delta)]
        color = method_colors.get(name, "#888888")
        ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(0.0, color="k", ls="--", lw=0.9, alpha=0.55)
        mean_d = float(np.mean(vals)) if vals.size else float("nan")
        ax.axvline(mean_d, color=color, ls="-", lw=1.2, alpha=0.9)
        ax.set_xlim(*xlim)
        ax.set_title(name)
        ax.set_xlabel(r"$\Delta_{\mathrm{proj}}$")
        ax.text(
            0.98,
            0.95,
            rf"mean={mean_d:.2f}" if vals.size else "",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
        )


def draw_projection_distortion_panel(ax: plt.Axes, df_eps: Mapping[str, np.ndarray]) -> None:
    target_dim = np.asarray(df_eps["target_dim"])
    eps_mean = np.asarray(df_eps["epsilon_mean"])
    eps_std = np.asarray(df_eps["epsilon_std"])

    ax.errorbar(
        target_dim,
        eps_mean,
        yerr=eps_std,
        fmt="-o",
        color="#0072B2",
        ecolor="#999999",
        elinewidth=0.8,
        capsize=2.5,
        markersize=3.5,
    )
    ax.set_title("Gaussian random projection distortion")
    ax.set_xlabel("Projected dimension")
    ax.set_ylabel("Relative distance error")

