"""
Figure 2: Graph-based energy-flow primitives generate controllable latent manifolds.

Runs Langevin dynamics on six representative topologies, saves metrics CSV,
individual panels, and a combined 2x3 landscape gallery (A–F).

Distance-to-manifold quantification is saved to CSV and plotted separately as
Supplementary Figure S1 (not in the main Figure 2 composite).

Usage (from repo root):
    python experiments/results_topology_primitives.py
"""

from __future__ import annotations

import csv
import sys
import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figure_style import (  # noqa: E402
    FIGURE_2X3_SIZE,
    FIGURE_SINGLE_SIZE,
    apply_publication_style,
    save_figure,
)
from data_generation_module import (  # noqa: E402
    CycleNode,
    GraphEdge,
    GraphNode,
    SkeletonNode,
    UniversalGraphSystem,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
N_CELLS = 500
N_STEPS = 1000
DT = 0.01
DIFFUSION = 0.01
SAVE_EVERY = 5
SEED = 1

OUT_DIR = REPO_ROOT / "results" / "topology_primitives"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

FIGURE_2_ORDER = [
    "linear",
    "bifurcation",
    "cycle",
    "cycle_linear",
    "skeleton",
    "cycle_bifurcation",
]

PANEL_BOUNDS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "linear": ((-3.0, 3.0), (-2.0, 2.0)),
    "bifurcation": ((-3.0, 3.0), (-2.5, 2.5)),
    "cycle": ((-2.5, 2.5), (-2.5, 2.5)),
    "cycle_linear": ((-1.5, 4.0), (-2.5, 2.5)),
    "skeleton": ((-3.0, 3.0), (-2.5, 2.5)),
    "cycle_bifurcation": ((-1.5, 4.0), (-2.5, 2.5)),
}

PANEL_TITLES = {
    "linear": "A. Linear trajectory",
    "bifurcation": "B. Bifurcation",
    "cycle": "C. Cycle",
    "cycle_linear": "D. Cycle–linear hybrid",
    "skeleton": "E. Skeleton (S-curve)",
    "cycle_bifurcation": "F. Cycle–bifurcation hybrid",
}

# Landscape-supported visualization (Dynamo-style: full domain field, masked display).
SUPPORT_CUT = 0.06
N_GRID_PLOT = 120
LOW_SUPPORT_OVERLAY_ALPHA = 0.40
LANDSCAPE_CMAP = "RdBu"  # red valleys (low U), blue background (high U)
STREAM_COLOR = "#1a1a2e"
SKELETON_COLOR = "0.12"

# Per-panel streamplot density.
PANEL_PLOT_CONFIG: Dict[str, Dict] = {
    "linear": {"stream_density": 1.0, "stream_alpha": 0.72},
    "bifurcation": {"stream_density": 1.0, "stream_alpha": 0.72},
    "cycle": {"stream_density": 1.2, "stream_alpha": 0.75},
    "cycle_linear": {"stream_density": 1.1, "stream_alpha": 0.75, "mark_exit": True},
    "skeleton": {"stream_density": 1.2, "stream_alpha": 0.75},
    "cycle_bifurcation": {"stream_density": 1.1, "stream_alpha": 0.75},
}

# Per-topology simulation overrides (optional)
# Edge-only topologies need longer integration + lower noise: mid-edge drift is weak
# and shallow terminal wells trap cells before x≈2.
SIM_OVERRIDES: Dict[str, Dict] = {
    "linear": {"n_steps": 3500, "diffusion": 0.006},
    "bifurcation": {"n_steps": 4000, "diffusion": 0.006},
    "skeleton": {"n_steps": 2000, "diffusion": 0.007},
    "cycle_bifurcation": {"n_steps": 4500, "diffusion": 0.006},
}

# Streamplot: mask shallow / background regions (dynamo-style: arrows only on landscape)
STREAM_THRESHOLD = -0.25  # stricter cutoff → narrower stream band
STREAM_PERCENTILE = 50  # keep ~deepest half of U<0 grid (smaller mask than p72)
STREAM_STRIDE = 3  # subsample fine grid for streamplot
STREAM_DENSITY = 1.0
STREAM_MIN_SPEED = 0.12  # hide weak drift at channel rims


# ---------------------------------------------------------------------------
# Langevin simulator
# ---------------------------------------------------------------------------
def simulate_langevin(
    system: UniversalGraphSystem,
    z0: torch.Tensor,
    n_steps: int = N_STEPS,
    dt: float = DT,
    diffusion: float = DIFFUSION,
    save_every: int = SAVE_EVERY,
    seed: int = SEED,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Overdamped Langevin: z_{t+dt} = z_t + b(z_t) dt + sqrt(2 D dt) * noise.

    Returns:
        traj_history: [n_saved, n_cells, dims]
        final_positions: [n_cells, dims]
    """
    torch.manual_seed(seed)
    z = z0.clone().float()
    noise_scale = float(np.sqrt(2.0 * diffusion * dt))
    saved: List[torch.Tensor] = [z.clone()]

    for step in range(1, n_steps + 1):
        with torch.enable_grad():
            drift = system.get_drift(z).detach()
        z = z + drift * dt + noise_scale * torch.randn_like(z)
        if step % save_every == 0:
            saved.append(z.clone())

    traj = torch.stack(saved, dim=0)
    return traj, z.detach()


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------
def build_linear_system() -> UniversalGraphSystem:
    # Shallow source + deep sink: drives cells along the edge (see example_generate Fate=-4.5).
    start = GraphNode("Start", [-2.0, 0.0], potential=-1.5, sigma=0.35)
    end = GraphNode(
        "End",
        [2.0, 0.0],
        potential=-4.5,
        sigma=0.22,
        basin_strength=0.6,
        basin_sigma=0.45,
    )
    edge = GraphEdge(start, end)
    return UniversalGraphSystem([start, end], [edge], fusion_temp=0.01)


def build_bifurcation_system() -> UniversalGraphSystem:
    stem = GraphNode("Stem", [-2.0, 0.0], potential=-1.5, sigma=0.35)
    branch = GraphNode("Branch", [0.0, 0.0], potential=-2.0, sigma=0.30)
    fate1 = GraphNode(
        "Fate1",
        [2.0, 1.0],
        potential=-4.5,
        sigma=0.22,
        basin_strength=0.8,
        basin_sigma=0.4,
    )
    fate2 = GraphNode(
        "Fate2",
        [2.0, -1.0],
        potential=-4.5,
        sigma=0.22,
        basin_strength=0.8,
        basin_sigma=0.4,
    )
    e0 = GraphEdge(stem, branch)
    e1 = GraphEdge(branch, fate1)
    e2 = GraphEdge(branch, fate2)
    return UniversalGraphSystem([stem, branch, fate1, fate2], [e0, e1, e2], fusion_temp=0.01)


def build_cycle_system() -> UniversalGraphSystem:
    cycle = CycleNode(
        "Cycle",
        pos=[0.0, 0.0],
        radius=1.2,
        potential=-2.5,
        sigma=0.25,
        omega=3.0,
        clockwise=False,
        rotation_axes=(0, 1),
    )
    return UniversalGraphSystem([cycle], [], fusion_temp=0.01)


def build_cycle_linear_system() -> UniversalGraphSystem:
    cycle = CycleNode(
        "Cycle",
        pos=[0.0, 0.0],
        radius=1.0,
        potential=-2.5,
        sigma=0.25,
        omega=2.0,
        clockwise=False,
        rotation_axes=(0, 1),
    )
    terminal = GraphNode(
        "Terminal",
        pos=[3.0, 0.0],
        potential=-3.2,
        sigma=0.35,
        basin_strength=1.0,
        basin_sigma=0.5,
    )
    exit_edge = GraphEdge(cycle, terminal)
    return UniversalGraphSystem([cycle, terminal], [exit_edge], fusion_temp=0.01)


def build_cycle_bifurcation_system() -> UniversalGraphSystem:
    cycle = CycleNode(
        "Cycle",
        pos=[0.0, 0.0],
        radius=1.0,
        potential=-2.5,
        sigma=0.25,
        omega=2.0,
        clockwise=False,
        rotation_axes=(0, 1),
    )
    fate1 = GraphNode(
        "Fate1",
        pos=[3.0, 1.0],
        potential=-3.2,
        sigma=0.30,
        basin_strength=0.75,
        basin_sigma=0.5,
    )
    fate2 = GraphNode(
        "Fate2",
        pos=[3.0, -1.0],
        potential=-3.2,
        sigma=0.30,
        basin_strength=0.75,
        basin_sigma=0.5,
    )
    edges = [GraphEdge(cycle, fate1), GraphEdge(cycle, fate2)]
    return UniversalGraphSystem([cycle, fate1, fate2], edges, fusion_temp=0.01)


def build_skeleton_system(kind: str = "s_curve") -> UniversalGraphSystem:
    if kind == "s_curve":
        t = np.linspace(-2, 2, 200)
        x = t
        y = np.sin(2 * t)
        skeleton = np.stack([x, y], axis=1)
        is_closed = False
    elif kind == "figure8":
        t = np.linspace(0, 2 * np.pi, 300)
        x = np.sin(t)
        y = np.sin(2 * t) / 2
        skeleton = np.stack([x, y], axis=1)
        is_closed = True
    else:
        raise ValueError(f"Unknown skeleton kind: {kind}")

    node = SkeletonNode(
        "Skeleton",
        skeleton_points=skeleton,
        potential=-2.5,
        sigma=0.25,
        flow_strength=3.5,
        is_closed=is_closed,
    )
    return UniversalGraphSystem([node], [], fusion_temp=0.01)


# ---------------------------------------------------------------------------
# Cell initialization
# ---------------------------------------------------------------------------
def initialize_cells(
    topology: str,
    n_cells: int,
    system: UniversalGraphSystem,
    seed: int = SEED,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    small = 0.08

    if topology in ("linear", "bifurcation"):
        z0 = rng.normal([-2.0, 0.0], small, size=(n_cells, 2)).astype(np.float32)
        return torch.tensor(z0)

    if topology == "cycle":
        center = np.array([0.0, 0.0])
        radius = 1.2
        angles = rng.uniform(0, 2 * np.pi, size=n_cells)
        z0 = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        z0 += rng.normal(0, small, size=z0.shape)
        return torch.tensor(z0.astype(np.float32))

    if topology == "cycle_linear":
        cycle = next(n for n in system.nodes if isinstance(n, CycleNode))
        center = np.array(cycle.pos[:2], dtype=np.float64)
        radius = cycle.radius
        angles = rng.uniform(0, 2 * np.pi, size=n_cells)
        z0 = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        z0 += rng.normal(0, small, size=z0.shape)
        return torch.tensor(z0.astype(np.float32))

    if topology == "skeleton":
        skel = next(n for n in system.nodes if isinstance(n, SkeletonNode))
        pts = skel.skeleton_points.numpy()
        idx = rng.integers(0, len(pts), size=n_cells)
        z0 = pts[idx] + rng.normal(0, small, size=(n_cells, 2))
        return torch.tensor(z0.astype(np.float32))

    if topology == "cycle_bifurcation":
        cycle = next(n for n in system.nodes if isinstance(n, CycleNode))
        center = np.array(cycle.pos[:2], dtype=np.float64)
        radius = cycle.radius
        angles = rng.uniform(0, 2 * np.pi, size=n_cells)
        z0 = center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        z0 += rng.normal(0, small, size=z0.shape)
        return torch.tensor(z0.astype(np.float32))

    raise ValueError(f"Unknown topology: {topology}")


# ---------------------------------------------------------------------------
# Grid + plotting
# ---------------------------------------------------------------------------
def make_grid(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    n_grid: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs = torch.linspace(xlim[0], xlim[1], n_grid)
    ys = torch.linspace(ylim[0], ylim[1], n_grid)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return xx, yy, grid


def compute_manifold_support(system: UniversalGraphSystem, grid: torch.Tensor) -> torch.Tensor:
    """
    ρ_max(z) = max_c exp(-d_c² / 2σ_c²): confidence that z lies on a graph component.
    """
    support_list: List[torch.Tensor] = []
    with torch.no_grad():
        for edge in system.edges:
            d2, _, _, sig = edge.get_component_data(grid)
            rho = torch.exp(-d2 / (2.0 * sig**2 + 1e-8))
            support_list.append(rho)
        for node in system.nodes:
            d2, _, _, sig = node.get_potential_and_flow(grid)
            rho = torch.exp(-d2 / (2.0 * sig**2 + 1e-8))
            support_list.append(rho)
    if not support_list:
        return torch.zeros(grid.shape[0], device=grid.device, dtype=grid.dtype)
    return torch.stack(support_list, dim=1).max(dim=1).values


def compute_grid(
    system: UniversalGraphSystem,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    n_grid: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx_t, yy_t, grid = make_grid(xlim, ylim, n_grid)
    with torch.no_grad():
        u = system.potential_numpy(grid).reshape(n_grid, n_grid)
    with torch.enable_grad():
        drift = system.get_drift(grid).detach().cpu().numpy()
    drift = drift.reshape(n_grid, n_grid, 2)
    return xx_t.numpy(), yy_t.numpy(), u, drift


def square_panel_limits(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Expand the shorter axis so x and y spans match (square data window)."""
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    span = max(xspan, yspan)
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    half = 0.5 * span
    return (xmid - half, xmid + half), (ymid - half, ymid + half)


def compute_landscape_fields(
    system: UniversalGraphSystem,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    n_grid: int = N_GRID_PLOT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Potential, drift, and manifold-support on a regular grid."""
    xx_t, yy_t, grid = make_grid(xlim, ylim, n_grid)
    with torch.no_grad():
        u = system.potential_numpy(grid).reshape(n_grid, n_grid)
        rho = compute_manifold_support(system, grid).reshape(n_grid, n_grid).cpu().numpy()
    with torch.enable_grad():
        drift = system.get_drift(grid).detach().cpu().numpy().reshape(n_grid, n_grid, 2)
    return xx_t.numpy(), yy_t.numpy(), u, drift, rho


def _landscape_stream_threshold(u: np.ndarray) -> float:
    """
    Adaptive cutoff: keep streamlines only where potential is in the channel.

    TopoVeloGen channels have U < 0; flat background is near 0 (same idea as
    GraphVisualizer.stream_threshold and dynamo landscape plots).
    """
    in_channel = u.ravel()[u.ravel() < 0]
    if in_channel.size == 0:
        return STREAM_THRESHOLD
    adaptive = float(np.percentile(in_channel, STREAM_PERCENTILE))
    return min(STREAM_THRESHOLD, adaptive)


def plot_landscape_streamlines(
    ax: plt.Axes,
    system: UniversalGraphSystem,
    xx: np.ndarray,
    yy: np.ndarray,
    u: np.ndarray,
    *,
    stream_threshold: Optional[float] = None,
    stream_stride: int = STREAM_STRIDE,
    stream_density: float = STREAM_DENSITY,
    min_speed: float = STREAM_MIN_SPEED,
    stream_alpha: float = 0.45,
) -> None:
    """Draw streamlines only on the latent landscape (masked shallow potential)."""
    sl = slice(None, None, stream_stride)
    xx_s = xx[sl, sl]
    yy_s = yy[sl, sl]
    u_s = u[sl, sl]

    grid = torch.tensor(
        np.stack([xx_s.ravel(), yy_s.ravel()], axis=-1), dtype=torch.float32
    )
    with torch.enable_grad():
        drift = system.get_drift(grid).detach().cpu().numpy()
    vx = drift[:, 0].reshape(xx_s.shape)
    vy = drift[:, 1].reshape(xx_s.shape)

    thr = stream_threshold if stream_threshold is not None else _landscape_stream_threshold(u)
    mask = u_s > thr
    vx[mask] = np.nan
    vy[mask] = np.nan

    speed = np.sqrt(vx * vx + vy * vy)
    speed[mask] = np.nan
    vx[speed < min_speed] = np.nan
    vy[speed < min_speed] = np.nan

    ax.streamplot(
        xx_s,
        yy_s,
        vx,
        vy,
        color=(1.0, 1.0, 1.0, stream_alpha),
        density=stream_density,
        linewidth=0.4,
        arrowsize=0.5,
        zorder=2,
    )


def _draw_graph_skeleton(ax: plt.Axes, system: UniversalGraphSystem) -> None:
    """Prescribed graph/skeleton overlay (user-specified target manifold)."""
    skel_kw = {"color": SKELETON_COLOR, "ls": "--", "lw": 1.3, "alpha": 0.92, "zorder": 5}
    for edge in system.edges:
        s = edge.start.cpu().numpy()
        e = edge.end.cpu().numpy()
        ax.plot([s[0], e[0]], [s[1], e[1]], **skel_kw)
    for node in system.nodes:
        if isinstance(node, CycleNode):
            circ = plt.Circle(
                node.pos[:2],
                node.radius,
                fill=False,
                ls="--",
                color=SKELETON_COLOR,
                lw=1.3,
                alpha=0.92,
                zorder=5,
            )
            ax.add_artist(circ)
        elif isinstance(node, SkeletonNode):
            sk = node.skeleton_points.cpu().numpy()
            ax.plot(sk[:, 0], sk[:, 1], **skel_kw)


def _plot_low_support_overlay(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    rho_grid: np.ndarray,
    *,
    support_cut: float = SUPPORT_CUT,
    alpha: float = LOW_SUPPORT_OVERLAY_ALPHA,
) -> None:
    """Wash out regions with ρ_max below the support cutoff."""
    low_support = rho_grid < support_cut
    if not np.any(low_support):
        return
    ax.contourf(
        xx,
        yy,
        low_support.astype(float),
        levels=[0.5, 1.5],
        colors=["white"],
        alpha=alpha,
        zorder=1,
    )


def _plot_support_boundary(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    rho_grid: np.ndarray,
    *,
    support_cut: float = SUPPORT_CUT,
) -> None:
    ax.contour(
        xx,
        yy,
        rho_grid,
        levels=[support_cut],
        colors=[SKELETON_COLOR],
        linewidths=0.7,
        linestyles=":",
        alpha=0.55,
        zorder=2,
    )


def _plot_support_masked_streamplot(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    drift: np.ndarray,
    rho_grid: np.ndarray,
    *,
    support_cut: float = SUPPORT_CUT,
    stream_stride: int = 2,
    stream_density: float = STREAM_DENSITY,
    stream_alpha: float = 0.75,
    min_speed: float = 0.02,
) -> None:
    """Streamplot on regular grid; drift masked outside ρ_max support region."""
    sl = np.s_[::stream_stride, ::stream_stride]
    xx_s = xx[sl]
    yy_s = yy[sl]
    vx = drift[:, :, 0][sl].astype(np.float64, copy=True)
    vy = drift[:, :, 1][sl].astype(np.float64, copy=True)
    rho_s = rho_grid[sl]

    unsupported = rho_s <= support_cut
    speed = np.sqrt(vx * vx + vy * vy)
    weak = speed < min_speed
    hide = unsupported | weak
    vx[hide] = np.nan
    vy[hide] = np.nan

    rgb = mcolors.to_rgb(STREAM_COLOR)
    ax.streamplot(
        xx_s,
        yy_s,
        vx,
        vy,
        color=(*rgb, stream_alpha),
        density=stream_density,
        linewidth=0.65,
        arrowsize=0.85,
        broken_streamlines=False,
        zorder=7,
    )


def _support_at_points(system: UniversalGraphSystem, points: np.ndarray) -> np.ndarray:
    grid = torch.tensor(points, dtype=torch.float32)
    with torch.no_grad():
        return compute_manifold_support(system, grid).cpu().numpy()


def plot_topology_panel(
    system: UniversalGraphSystem,
    traj: torch.Tensor,
    ax: plt.Axes,
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    rng: np.random.Generator,
    topology: str,
    n_traj_plot: int = 20,
    support_cut: float = SUPPORT_CUT,
) -> None:
    cfg = PANEL_PLOT_CONFIG[topology]
    xlim, ylim = square_panel_limits(xlim, ylim)
    xx, yy, u, drift, rho = compute_landscape_fields(system, xlim, ylim, n_grid=N_GRID_PLOT)

    # 1. Full-domain landscape (red valleys, blue background)
    ax.contourf(xx, yy, u, levels=60, cmap=LANDSCAPE_CMAP, alpha=0.92, zorder=0)
    # 2. De-emphasize low-support regions
    _plot_low_support_overlay(ax, xx, yy, rho, support_cut=support_cut)
    # 3. Manifold support boundary
    _plot_support_boundary(ax, xx, yy, rho, support_cut=support_cut)
    # 4. Prescribed skeleton
    _draw_graph_skeleton(ax, system)

    if cfg.get("mark_exit") and system.edges:
        exit_pt = system.edges[0].start.cpu().numpy()
        ax.scatter(
            exit_pt[0],
            exit_pt[1],
            marker="*",
            s=60,
            c="#D55E00",
            edgecolors=SKELETON_COLOR,
            linewidths=0.5,
            zorder=6,
        )

    # 5. Trajectories and cells (only high-support final positions)
    n_cells = traj.shape[1]
    idx = rng.choice(n_cells, size=min(n_traj_plot, n_cells), replace=False)
    for j in idx:
        path = traj[:, j, :].cpu().numpy()
        ax.plot(path[:, 0], path[:, 1], color="0.35", lw=0.4, alpha=0.45, zorder=4)

    final_z = traj[-1].cpu().numpy()
    rho_cells = _support_at_points(system, final_z)
    supported = rho_cells > support_cut
    ax.scatter(
        final_z[supported, 0],
        final_z[supported, 1],
        s=3.0,
        c="#D55E00",
        alpha=0.75,
        zorder=6,
        edgecolors="none",
    )

    # 6. Support-masked streamplot — drawn last so streamlines stay visible
    _plot_support_masked_streamplot(
        ax,
        xx,
        yy,
        drift,
        rho,
        support_cut=support_cut,
        stream_density=cfg.get("stream_density", STREAM_DENSITY),
        stream_alpha=cfg.get("stream_alpha", 0.75),
    )

    ax.set_title(title.replace(". ", "  ", 1))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _edge_tangent(edge: GraphEdge, z: torch.Tensor) -> torch.Tensor:
    vec = edge.vec.to(z.device)
    return vec / (torch.norm(vec) + 1e-8)


def _graphnode_fallback_tangent(system: UniversalGraphSystem, node: GraphNode, z: torch.Tensor) -> torch.Tensor:
    """Direction along the closest incident edge (by projection), for point nodes."""
    incident = [
        e
        for e in system.edges
        if e.node_start.id == node.id or e.node_end.id == node.id
    ]
    if not incident:
        return torch.zeros_like(z)
    best_d2 = torch.full((z.shape[0],), float("inf"), device=z.device)
    best_t = torch.zeros_like(z)
    for edge in incident:
        d2 = edge.distance_sq(z)
        t = _edge_tangent(edge, z).unsqueeze(0).expand(z.shape[0], -1)
        mask = d2 < best_d2
        best_d2 = torch.where(mask, d2, best_d2)
        best_t = torch.where(mask.unsqueeze(1), t, best_t)
    return best_t


def _component_distances_and_tangents(
    system: UniversalGraphSystem, z: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-cell min distance and tangent from nearest graph component."""
    n = z.shape[0]
    d2_parts: List[torch.Tensor] = []
    tangent_parts: List[torch.Tensor] = []

    for edge in system.edges:
        d2_parts.append(edge.distance_sq(z))
        tangent_parts.append(_edge_tangent(edge, z).unsqueeze(0).expand(n, -1))

    for node in system.nodes:
        d2, _, flow, _ = node.get_potential_and_flow(z)
        d2_parts.append(d2)
        flow_norm = torch.norm(flow, dim=1, keepdim=True)
        t_flow = flow / (flow_norm + 1e-8)
        if float(flow_norm.max()) > 1e-6:
            tangent_parts.append(t_flow)
        elif isinstance(node, GraphNode):
            tangent_parts.append(_graphnode_fallback_tangent(system, node, z))
        else:
            tangent_parts.append(t_flow)

    d2_stack = torch.stack(d2_parts, dim=1)
    tan_stack = torch.stack(tangent_parts, dim=1)
    nearest = torch.argmin(d2_stack, dim=1)
    d_min = torch.sqrt(torch.gather(d2_stack, 1, nearest.unsqueeze(1)).squeeze(1))
    t_target = tan_stack[torch.arange(n, device=z.device), nearest]
    t_norm = torch.norm(t_target, dim=1, keepdim=True)
    t_target = torch.where(t_norm > 1e-6, t_target / (t_norm + 1e-8), t_target)
    return d_min, t_target


def compute_topology_metrics(
    system: UniversalGraphSystem,
    z_final: torch.Tensor,
) -> Dict[str, float]:
    d_min, t_target = _component_distances_and_tangents(system, z_final)
    with torch.enable_grad():
        b = system.get_drift(z_final).detach()
    b_norm = torch.norm(b, dim=1) + 1e-8
    t_norm = torch.norm(t_target, dim=1) + 1e-8
    cos = torch.clamp((b * t_target).sum(dim=1) / (b_norm * t_norm), -1.0, 1.0)

    with torch.no_grad():
        _, f_rot = system.compute_fused_field(z_final)
    f_norm = torch.norm(f_rot, dim=1) + 1e-8
    cos_rot = torch.clamp((f_rot * t_target).sum(dim=1) / (f_norm * t_norm), -1.0, 1.0)

    return {
        "mean_distance": float(d_min.mean()),
        "median_distance": float(d_min.median()),
        "p90_distance": float(torch.quantile(d_min, 0.9)),
        "mean_flow_alignment": float(cos.mean()),
        "mean_rot_flow_alignment": float(cos_rot.mean()),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_metrics_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "topology",
        "mean_distance",
        "median_distance",
        "p90_distance",
        "mean_flow_alignment",
        "mean_rot_flow_alignment",
        "n_cells",
        "n_steps",
        "diffusion",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_trajectory_npz(name: str, traj: torch.Tensor, metrics: Dict) -> None:
    path = DATA_DIR / f"{name}_traj.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, traj=traj.numpy(), **{k: np.array(v) for k, v in metrics.items()})


def plot_figure_2(all_results: Dict[str, Dict], out_path: Path) -> None:
    """2×3 landscape gallery: same visual language in every panel."""
    apply_publication_style()
    rng = np.random.default_rng(SEED)

    fig, axes = plt.subplots(2, 3, figsize=FIGURE_2X3_SIZE, constrained_layout=True)
    axes_flat = axes.flatten()

    for i, name in enumerate(FIGURE_2_ORDER):
        res = all_results[name]
        xlim, ylim = PANEL_BOUNDS[name]
        plot_topology_panel(
            res["system"],
            res["traj"],
            axes_flat[i],
            PANEL_TITLES[name],
            xlim,
            ylim,
            rng,
            topology=name,
        )

    save_figure(fig, out_path)


def plot_supplementary_s1(all_results: Dict[str, Dict], out_path: Path) -> None:
    """Supplementary S1: distance-to-manifold quantification (not in main Figure 2)."""
    apply_publication_style()
    names = [PANEL_TITLES[n].split(". ", 1)[1] for n in FIGURE_2_ORDER]
    mean_d = [all_results[n]["metrics"]["mean_distance"] for n in FIGURE_2_ORDER]
    median_d = [all_results[n]["metrics"]["median_distance"] for n in FIGURE_2_ORDER]
    p90_d = [all_results[n]["metrics"]["p90_distance"] for n in FIGURE_2_ORDER]

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(x - w, mean_d, width=w, label="Mean", color="C0", edgecolor="none")
    ax.bar(x, median_d, width=w, label="Median", color="C2", edgecolor="none", alpha=0.9)
    ax.bar(x + w, p90_d, width=w, label="P90", color="C3", edgecolor="none", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Distance to prescribed manifold")
    ax.set_title("Supplementary Fig. S1  Distance-to-manifold")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    builders: Dict[str, Callable[[], UniversalGraphSystem]] = {
        "linear": build_linear_system,
        "bifurcation": build_bifurcation_system,
        "cycle": build_cycle_system,
        "cycle_linear": build_cycle_linear_system,
        "skeleton": lambda: build_skeleton_system("s_curve"),
        "cycle_bifurcation": build_cycle_bifurcation_system,
    }

    all_results: Dict[str, Dict] = {}
    metrics_rows: List[Dict] = []
    rng = np.random.default_rng(SEED)

    for name, builder in builders.items():
        print(f"Running topology: {name}")
        system = builder()
        z0 = initialize_cells(name, N_CELLS, system, seed=SEED)
        sim_kw = {"n_steps": N_STEPS, "diffusion": DIFFUSION, **SIM_OVERRIDES.get(name, {})}
        traj, z_final = simulate_langevin(
            system,
            z0,
            dt=DT,
            save_every=SAVE_EVERY,
            seed=SEED,
            **sim_kw,
        )
        metrics = compute_topology_metrics(system, z_final)
        all_results[name] = {"system": system, "traj": traj, "metrics": metrics}
        save_trajectory_npz(name, traj, metrics)

        xlim, ylim = PANEL_BOUNDS[name]
        apply_publication_style()
        fig_single, ax_single = plt.subplots(figsize=FIGURE_SINGLE_SIZE, constrained_layout=True)
        plot_topology_panel(
            system, traj, ax_single, PANEL_TITLES[name], xlim, ylim, rng, topology=name
        )
        save_figure(fig_single, FIG_DIR / f"panel_{name}.png", formats=("png",))

        metrics_rows.append(
            {
                "topology": name,
                **metrics,
                "n_cells": N_CELLS,
                "n_steps": sim_kw["n_steps"],
                "diffusion": sim_kw["diffusion"],
            }
        )
        print(
            f"  mean_dist={metrics['mean_distance']:.4f}, "
            f"flow_align={metrics['mean_flow_alignment']:.4f}, "
            f"rot_align={metrics['mean_rot_flow_alignment']:.4f}"
        )

    save_metrics_csv(metrics_rows, DATA_DIR / "topology_metrics.csv")
    plot_figure_2(all_results, FIG_DIR / "figure2_topology_primitives.png")
    plot_supplementary_s1(all_results, FIG_DIR / "supplementary_s1_topology_quantification.png")
    print(f"\nSaved metrics: {DATA_DIR / 'topology_metrics.csv'}")
    print(f"Saved figure:  {FIG_DIR / 'figure2_topology_primitives.png'}")
    print(f"Saved supp.:   {FIG_DIR / 'supplementary_s1_topology_quantification.png'}")


def load_plot_results() -> Dict[str, Dict]:
    builders: Dict[str, Callable[[], UniversalGraphSystem]] = {
        "linear": build_linear_system,
        "bifurcation": build_bifurcation_system,
        "cycle": build_cycle_system,
        "cycle_linear": build_cycle_linear_system,
        "skeleton": lambda: build_skeleton_system("s_curve"),
        "cycle_bifurcation": build_cycle_bifurcation_system,
    }
    all_results: Dict[str, Dict] = {}
    for name, builder in builders.items():
        npz_path = DATA_DIR / f"{name}_traj.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing cached trajectory: {npz_path}")
        data = np.load(npz_path, allow_pickle=False)
        traj = torch.tensor(data["traj"])
        metrics = {
            "mean_distance": float(data["mean_distance"]),
            "median_distance": float(data["median_distance"]),
            "p90_distance": float(data["p90_distance"]),
            "mean_flow_alignment": float(data["mean_flow_alignment"]),
            "mean_rot_flow_alignment": float(data["mean_rot_flow_alignment"]),
        }
        all_results[name] = {"system": builder(), "traj": traj, "metrics": metrics}
    return all_results


def plot_only() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading cached trajectories for Figure 2...")
    all_results = load_plot_results()
    plot_figure_2(all_results, FIG_DIR / "figure2_topology_primitives.png")
    plot_supplementary_s1(all_results, FIG_DIR / "supplementary_s1_topology_quantification.png")
    print(f"Saved {FIG_DIR / 'figure2_topology_primitives.png'}")
    print(f"Saved {FIG_DIR / 'supplementary_s1_topology_quantification.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 2: topology primitives")
    parser.add_argument("--plot-only", action="store_true", help="Replot from cached npz")
    args = parser.parse_args()
    if args.plot_only:
        plot_only()
    else:
        main()
