"""
Legacy notebook-export script. Prefer the panelized packages under scripts/figures/ (see scripts/figures/README.md).
"""

# ==============================================================================
# Beautified plotting cell for FigSeriesMain1_TopologyCollection
# Drop this cell after `topology_data` has been generated.
# It assumes these variables/functions already exist:
#   topology_types, topology_labels, topology_data, save_current_figure
# and these classes exist from data_generation_module:
#   CycleNode, SkeletonNode
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.linewidth": 0.7,
})
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

COLOR_DICT = {
"deep_blue_gray": "#396185",
"blue_gray": "#6981a3",
"light_blue": "#93b1d1",
"lighter_blue": "#a8c9e8",
"very_light_blue": "#bde0fe",
"light_purple": "#ded4ee",
"light_pink": "#ffc8dd",
"pink": "#f1b5c8",
"deep_pink": "#e2a2b3",
"deeper_pink": "#db8c9f",
"deepest_pink": "#cb7892",
}

COLOR_LIST = [
COLOR_DICT["deep_blue_gray"],
COLOR_DICT["blue_gray"],
COLOR_DICT["light_blue"],
COLOR_DICT["light_pink"],
COLOR_DICT["deep_pink"],
COLOR_DICT["deepest_pink"],
]

if "TopoVelo_grad" not in plt.colormaps():
    plt.colormaps.register(
    LinearSegmentedColormap.from_list(
    "TopoVelo_grad",
    [
    COLOR_DICT["deep_blue_gray"],
    COLOR_DICT["light_blue"],
    COLOR_DICT["light_pink"],
    COLOR_DICT["deepest_pink"],
    ],
    )
    )

if "TopoVelo_cat" not in plt.colormaps():
    plt.colormaps.register(
    ListedColormap(COLOR_LIST, name="TopoVelo_cat")
    )

TIME_CMAP = plt.get_cmap("TopoVelo_grad")
CAT_CMAP = plt.get_cmap("TopoVelo_cat")

# ---------- Visual constants ----------
TIME_CMAP = plt.get_cmap("TopoVelo_grad")
CAT_CMAP = plt.get_cmap("TopoVelo_cat")
STREAM_COLOR = "#5f6368"
EDGE_COLOR = "#8a8f94"
TEXT_COLOR = "#222222"
LIGHT_SPINE = "#b8b8b8"


def to_numpy(data):
    """Convert Tensor/List/array-like object to numpy array."""
    if hasattr(data, "detach"):
        data = data
    if hasattr(data, "cpu"):
        data = data
    if hasattr(data, "numpy"):
        data = data
    return np.asarray(data)


def pad_limits(x, y, pad=0.10):
    """Return padded equal-aspect limits."""
    x = np.asarray(x)
    y = np.asarray(y)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xr = max(xmax - xmin, 1e-6)
    yr = max(ymax - ymin, 1e-6)
    # keep equal visual scale by expanding the shorter axis
    span = max(xr, yr)
    xc = 0.5 * (xmin + xmax)
    yc = 0.5 * (ymin + ymax)
    half = 0.5 * span * (1 + 2 * pad)
    return (xc - half, xc + half), (yc - half, yc + half)


def add_arrow_to_path(ax, x, y, n_arrows=2, color=EDGE_COLOR, size=8, lw=0.0, zorder=4):
    """Place small arrows along a path."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 3 or n_arrows <= 0:
        return
    idxs = np.linspace(1, len(x) - 2, n_arrows + 2, dtype=int)[1:-1]
    for i in idxs:
        p0 = (x[i], y[i])
        p1 = (x[i + 1], y[i + 1])
        ax.add_patch(FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>",
            mutation_scale=size,
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        ))


def draw_topology_schematic_beautified(ax, system, font_size=7.5):
    """Draw a compact schematic of the user-defined topological skeleton."""
    ax.set_aspect("equal")
    ax.axis("off")

    nodes = system.nodes
    edges = system.edges
    all_x, all_y = [], []

    # Edges first
    for edge in edges:
        start = to_numpy(edge.start)
        end = to_numpy(edge.end)
        x = np.linspace(start[0], end[0], 80)
        y = np.linspace(start[1], end[1], 80)
        ax.plot(x, y, color=EDGE_COLOR, lw=1.15, ls=(0, (2.2, 2.2)), zorder=1)
        add_arrow_to_path(ax, x, y, n_arrows=1, color=EDGE_COLOR, size=7, zorder=2)
        all_x.extend([start[0], end[0]])
        all_y.extend([start[1], end[1]])

    # Nodes
    n_nodes = max(len(nodes), 1)
    for idx, node in enumerate(nodes):
        color = CAT_CMAP(idx / max(n_nodes - 1, 1))
        pos = to_numpy(node.pos)

        if isinstance(node, CycleNode):
            theta = np.linspace(0, 2 * np.pi, 180)
            if getattr(node, "clockwise", False):
                theta = theta[::-1]
            x = pos[0] + node.radius * np.cos(theta)
            y = pos[1] + node.radius * np.sin(theta)
            ax.plot(x, y, color="white", lw=6.0, zorder=3)
            ax.plot(x, y, color=color, lw=3.0, zorder=4)
            add_arrow_to_path(ax, x, y, n_arrows=3, color=color, size=7, zorder=5)
            all_x.extend(x); all_y.extend(y)

        elif isinstance(node, SkeletonNode):
            skel = to_numpy(node.skeleton_points)
            x, y = skel[:, 0], skel[:, 1]
            ax.plot(x, y, color="white", lw=6.0, zorder=3)
            ax.plot(x, y, color=color, lw=3.0, zorder=4)
            add_arrow_to_path(ax, x, y, n_arrows=4, color=color, size=7, zorder=5)
            all_x.extend(x); all_y.extend(y)

        else:
            ax.scatter(
                pos[0], pos[1],
                s=82,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=5,
            )
            ax.scatter(
                pos[0], pos[1],
                s=88,
                facecolor="none",
                edgecolor="#333333",
                linewidth=0.45,
                zorder=6,
            )
            all_x.append(pos[0]); all_y.append(pos[1])

    # Labels: small and understated; avoid labels for very crowded multifurcation if needed
    if len(all_x) == 0:
        return
    y_span = max(all_y) - min(all_y)
    offset = 0.07 * y_span if y_span > 0 else 0.25

    for node in nodes:
        pos = to_numpy(node.pos)
        label_y = pos[1]
        if isinstance(node, CycleNode):
            label_y = pos[1] - 0.75 * node.radius
        elif isinstance(node, SkeletonNode):
            label_y = np.min(to_numpy(node.skeleton_points)[:, 1])
        ax.text(
            pos[0], label_y - offset, node.id,
            ha="center", va="top",
            fontsize=font_size,
            color=TEXT_COLOR,
            zorder=7,
        )

    (xlim, ylim) = pad_limits(np.array(all_x), np.array(all_y), pad=0.22)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def style_latent_axis(ax):
    """Minimal style for the latent dynamics panel."""
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines():
        sp.set_visible(False)


def style_trace_axis(ax):
    """Small trace axis style."""
    ax.grid(True, color="#eeeeee", lw=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT_SPINE)
    ax.spines["bottom"].set_color(LIGHT_SPINE)
    ax.tick_params(length=2.0, width=0.6, colors="#555555", pad=1.5)


def plot_coordinate_trace(ax, time_points, values, color, ylabel=None, show_xlabel=False):
    """Plot median and 10--90% envelope of one latent coordinate over time."""
    q10 = np.percentile(values, 10, axis=1)
    q50 = np.percentile(values, 50, axis=1)
    q90 = np.percentile(values, 90, axis=1)
    ax.fill_between(time_points, q10, q90, color=color, alpha=0.18, lw=0, zorder=1)
    ax.plot(time_points, q50, color=color, lw=1.2, zorder=2)
    ax.set_xlim(time_points[0], time_points[-1])
    ax.set_ylabel(ylabel or "", fontsize=7.4, labelpad=2)
    if show_xlabel:
        ax.set_xlabel("Time", fontsize=7.4, labelpad=1.5)
    else:
        ax.set_xticklabels([])
    style_trace_axis(ax)


def make_beautified_topology_collection(save_stem="FigSeriesMain1_TopologyCollection_beautified"):
    """Create the refined 6 x 3 topology collection figure."""
    n_rows = len(topology_types)
    fig = plt.figure(figsize=(7.2, 10.4), dpi=300, constrained_layout=False)
    gs = fig.add_gridspec(
        n_rows, 3,
        width_ratios=[0.88, 1.26, 1.22],
        hspace=0.34,
        wspace=0.18,
        left=0.055,
        right=0.955,
        top=0.935,
        bottom=0.055,
    )

    # Column headers
    header_y = 0.972
    fig.text(0.145, header_y, "Topology", ha="center", va="center", fontsize=9.5, fontweight="bold")
    fig.text(0.440, header_y, "Latent dynamics", ha="center", va="center", fontsize=9.5, fontweight="bold")
    fig.text(0.765, header_y, "Latent coordinate traces", ha="center", va="center", fontsize=9.5, fontweight="bold")

    # One shared time normalization for all latent dynamics panels
    t_min = min(topology_data[t]["time_points"][0] for t in topology_types)
    t_max = max(topology_data[t]["time_points"][-1] for t in topology_types)
    norm = Normalize(vmin=t_min, vmax=t_max)

    for idx, topo_type in enumerate(topology_types):
        data = topology_data[topo_type]
        title_text = topology_labels[idx].split(". ", 1)[1]
        z_trajectory = data["z_trajectory"]
        time_points = data["time_points"]

        # Sampled snapshot cells for scatter display
        sampled_z, sampled_time_labels = sample_snapshot_data(
            z_trajectory, time_points,
            n_samples_per_time=60,
            random_seed=42 + idx,
        )

        # A-F label and topology schematic
        ax_topo = fig.add_subplot(gs[idx, 0])
        draw_topology_schematic_beautified(ax_topo, data["system"], font_size=6.8)
        ax_topo.text(
            -0.05, 1.02, chr(ord("A") + idx),
            transform=ax_topo.transAxes,
            ha="right", va="bottom",
            fontsize=13,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        # Latent dynamics panel
        ax_dyn = fig.add_subplot(gs[idx, 1])
        grid_X, grid_Y, velocity_x, velocity_y = compute_velocity_field(
            z_trajectory, time_points, grid_resolution=34
        )
        ax_dyn.streamplot(
            grid_X, grid_Y, velocity_x, velocity_y,
            density=1.05,
            color=STREAM_COLOR,
            linewidth=0.42,
            arrowsize=0.58,
            arrowstyle="->",
            minlength=0.12,
            zorder=1,
        )
        sc = ax_dyn.scatter(
            sampled_z[:, 0], sampled_z[:, 1],
            c=sampled_time_labels,
            cmap=TIME_CMAP,
            norm=norm,
            s=4.8,
            alpha=0.78,
            linewidth=0,
            rasterized=True,
            zorder=3,
        )
        xlim, ylim = pad_limits(z_trajectory[:, :, 0].ravel(), z_trajectory[:, :, 1].ravel(), pad=0.07)
        ax_dyn.set_xlim(*xlim)
        ax_dyn.set_ylim(*ylim)
        ax_dyn.set_title(title_text, pad=2.5, fontsize=9.2, color=TEXT_COLOR)
        style_latent_axis(ax_dyn)

        # Coordinate traces
        gs_trace = gs[idx, 2].subgridspec(2, 1, hspace=0.08)
        ax_z1 = fig.add_subplot(gs_trace[0])
        ax_z2 = fig.add_subplot(gs_trace[1], sharex=ax_z1)
        plot_coordinate_trace(
            ax_z1,
            time_points,
            z_trajectory[:, :, 0],
            color=CAT_CMAP(0.15),
            ylabel=r"$z_1$",
            show_xlabel=False,
        )
        plot_coordinate_trace(
            ax_z2,
            time_points,
            z_trajectory[:, :, 1],
            color=CAT_CMAP(0.82),
            ylabel=r"$z_2$",
            show_xlabel=(idx == n_rows - 1),
        )

    # Shared colorbar, avoids six repeated colorbars
    cax = fig.add_axes([0.965, 0.18, 0.012, 0.62])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=TIME_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Simulation time", fontsize=8, labelpad=4)
    cbar.ax.tick_params(labelsize=7, length=2)

    # Trace legend
    legend_handles = [
        Line2D([0], [0], color=CAT_CMAP(0.15), lw=1.4, label=r"median $z_1$"),
        Line2D([0], [0], color=CAT_CMAP(0.82), lw=1.4, label=r"median $z_2$"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.95, 0.028),
        frameon=False,
        fontsize=7.2,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.2,
    )

    save_current_figure(save_stem)
    return fig


fig = make_beautified_topology_collection()
print("✓ Beautified topology collection figure saved")
