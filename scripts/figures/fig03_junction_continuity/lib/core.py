"""Figure 3 core: junction systems, fields, metrics, and panel plotting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from data_generation_module import CycleNode, GraphEdge, GraphNode, UniversalGraphSystem
from scripts.figures.fig02_topology_primitives.lib.core import (
    plot_landscape_streamlines,
    simulate_langevin,
)

# ---------------------------------------------------------------------------
# Defaults (overridden by configure / pipeline JSON)
# ---------------------------------------------------------------------------
SEED = 1
N_CELLS = 500
N_STEPS = 4000
DT = 0.01
DIFFUSION = 0.006
SAVE_EVERY = 5
JUNCTION_RADIUS = 0.55
N_GRID = 110
N_GRID_ZOOM = 100
N_GRID_TRAJ_BG = 80
N_TRAJ_SHOW = 50
INIT_RIM_NOISE = 0.05
INIT_RIM_RADIUS = 1.0
INIT_STEM_MEAN = (-2.0, 0.0)
INIT_STEM_STD = 0.08
FIELD_LEVELS = 50
TRAJ_BG_LEVELS = 40
TRAJ_BG_ALPHA = 0.55
METRICS_BAR_WIDTH = 0.25

OUT_DIR = Path("data/results/junction_continuity")
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

JUNCTIONS: Dict[str, Dict[str, Any]] = {}
FIGURE_ORDER: List[str] = ["cycle_linear", "bifurcation", "cycle_bifurcation"]


def configure(*, out_dir=None, data_dir=None, fig_dir=None, pipeline: dict | None = None) -> None:
    global OUT_DIR, DATA_DIR, FIG_DIR
    global N_CELLS, N_STEPS, DT, DIFFUSION, SAVE_EVERY, SEED
    global JUNCTION_RADIUS, N_GRID, N_GRID_ZOOM, N_GRID_TRAJ_BG, N_TRAJ_SHOW
    global INIT_RIM_NOISE, INIT_RIM_RADIUS, INIT_STEM_MEAN, INIT_STEM_STD
    global FIELD_LEVELS, TRAJ_BG_LEVELS, TRAJ_BG_ALPHA, METRICS_BAR_WIDTH
    global JUNCTIONS, FIGURE_ORDER

    if out_dir is not None:
        OUT_DIR = Path(out_dir)
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    elif out_dir is not None:
        DATA_DIR = OUT_DIR / "data"
    if fig_dir is not None:
        FIG_DIR = Path(fig_dir)
    elif out_dir is not None:
        FIG_DIR = OUT_DIR / "figures"
    if not pipeline:
        return

    N_CELLS = int(pipeline.get("n_cells", N_CELLS))
    N_STEPS = int(pipeline.get("n_steps", N_STEPS))
    DT = float(pipeline.get("dt", DT))
    DIFFUSION = float(pipeline.get("diffusion", DIFFUSION))
    SAVE_EVERY = int(pipeline.get("save_every", SAVE_EVERY))
    SEED = int(pipeline.get("seed", SEED))
    JUNCTION_RADIUS = float(pipeline.get("junction_radius", JUNCTION_RADIUS))
    N_GRID = int(pipeline.get("n_grid", N_GRID))
    N_GRID_ZOOM = int(pipeline.get("n_grid_zoom", N_GRID_ZOOM))
    N_GRID_TRAJ_BG = int(pipeline.get("n_grid_traj_bg", N_GRID_TRAJ_BG))
    N_TRAJ_SHOW = int(pipeline.get("n_traj_show", N_TRAJ_SHOW))
    INIT_RIM_NOISE = float(pipeline.get("init_rim_noise", INIT_RIM_NOISE))
    INIT_RIM_RADIUS = float(pipeline.get("init_rim_radius", INIT_RIM_RADIUS))
    if "init_stem_mean" in pipeline:
        INIT_STEM_MEAN = tuple(pipeline["init_stem_mean"])
    INIT_STEM_STD = float(pipeline.get("init_stem_std", INIT_STEM_STD))
    FIELD_LEVELS = int(pipeline.get("field_levels", FIELD_LEVELS))
    TRAJ_BG_LEVELS = int(pipeline.get("traj_bg_levels", TRAJ_BG_LEVELS))
    TRAJ_BG_ALPHA = float(pipeline.get("traj_bg_alpha", TRAJ_BG_ALPHA))
    METRICS_BAR_WIDTH = float(pipeline.get("metrics_bar_width", METRICS_BAR_WIDTH))
    if "figure_order" in pipeline:
        FIGURE_ORDER = list(pipeline["figure_order"])
    if "junctions" in pipeline:
        JUNCTIONS = {k: dict(v) for k, v in pipeline["junctions"].items()}


# ---------------------------------------------------------------------------
# Junction systems
# ---------------------------------------------------------------------------
def build_cycle_linear_junction() -> UniversalGraphSystem:
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
        sigma=0.32,
        basin_strength=0.9,
        basin_sigma=0.55,
    )
    edge = GraphEdge(cycle, terminal, barrier_height=0.0)
    return UniversalGraphSystem([cycle, terminal], [edge], fusion_temp=0.01)


def build_bifurcation_junction() -> UniversalGraphSystem:
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
    edges = [
        GraphEdge(stem, branch),
        GraphEdge(branch, fate1),
        GraphEdge(branch, fate2),
    ]
    return UniversalGraphSystem([stem, branch, fate1, fate2], edges, fusion_temp=0.01)


def build_cycle_bifurcation_junction() -> UniversalGraphSystem:
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
        [3.0, 1.0],
        potential=-3.2,
        sigma=0.30,
        basin_strength=0.75,
        basin_sigma=0.5,
    )
    fate2 = GraphNode(
        "Fate2",
        [3.0, -1.0],
        potential=-3.2,
        sigma=0.30,
        basin_strength=0.75,
        basin_sigma=0.5,
    )
    edges = [GraphEdge(cycle, fate1), GraphEdge(cycle, fate2)]
    return UniversalGraphSystem([cycle, fate1, fate2], edges, fusion_temp=0.01)


BUILDERS = {
    "cycle_linear": build_cycle_linear_junction,
    "bifurcation": build_bifurcation_junction,
    "cycle_bifurcation": build_cycle_bifurcation_junction,
}


def init_cycle_rim_cells(
    n_cells: int,
    radius: float | None = None,
    noise: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    radius = INIT_RIM_RADIUS if radius is None else radius
    noise = INIT_RIM_NOISE if noise is None else noise
    seed = SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n_cells)
    z0 = radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    z0 += rng.normal(0, noise, size=z0.shape)
    return z0.astype(np.float32)


def init_stem_cells(n_cells: int, seed: int | None = None) -> np.ndarray:
    seed = SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    z0 = rng.normal(list(INIT_STEM_MEAN), INIT_STEM_STD, size=(n_cells, 2)).astype(np.float32)
    return z0


def initialize_cells(topology_key: str, n_cells: int | None = None) -> np.ndarray:
    n_cells = N_CELLS if n_cells is None else n_cells
    if topology_key == "bifurcation":
        return init_stem_cells(n_cells)
    return init_cycle_rim_cells(n_cells)


def junction_cfg(topology_key: str) -> Dict[str, Any]:
    if not JUNCTIONS:
        raise RuntimeError("JUNCTIONS empty; call load_cfg() / configure() first")
    return JUNCTIONS[topology_key]


# ---------------------------------------------------------------------------
# Grid / fields
# ---------------------------------------------------------------------------
def make_grid(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    n_grid: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], n_grid)
    ys = np.linspace(ylim[0], ylim[1], n_grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    return xx, yy, grid


def compute_potential_and_drift(
    system: UniversalGraphSystem,
    grid: np.ndarray,
    n_grid: int,
) -> Tuple[np.ndarray, np.ndarray]:
    u = system.potential_numpy(grid).reshape(n_grid, n_grid)
    drift = system.get_drift(grid).reshape(n_grid, n_grid, 2)
    return u, drift


def junction_center_for(system: UniversalGraphSystem, topology_key: str) -> np.ndarray:
    if topology_key == "cycle_linear":
        return np.asarray(system.edges[0].start)
    if topology_key == "bifurcation":
        branch = next(n for n in system.nodes if n.id == "Branch")
        return np.array(branch.pos[:2], dtype=np.float64)
    if topology_key == "cycle_bifurcation":
        starts = np.stack([e.start for e in system.edges], axis=0)
        return starts.mean(axis=0)
    raise ValueError(topology_key)


def rim_docking_error(system: UniversalGraphSystem) -> Optional[float]:
    cycle = next((n for n in system.nodes if isinstance(n, CycleNode)), None)
    if cycle is None or not system.edges:
        return None
    edge = system.edges[0]
    center = np.asarray(cycle.pos[:2], dtype=np.float32)
    start = edge.start.astype(np.float32)
    dist = float(np.linalg.norm(start - center))
    return abs(dist - cycle.radius)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _normalize_vectors(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def drift_smoothness_score(drift: np.ndarray, region_mask: Optional[np.ndarray] = None) -> float:
    d = _normalize_vectors(drift.astype(np.float64))
    cos_x = (d[:, :-1, :] * d[:, 1:, :]).sum(axis=-1)
    cos_y = (d[:-1, :, :] * d[1:, :, :]).sum(axis=-1)
    if region_mask is not None:
        mx = region_mask[:, :-1] & region_mask[:, 1:]
        my = region_mask[:-1, :] & region_mask[1:, :]
        vals = []
        if mx.any():
            vals.append(cos_x[mx])
        if my.any():
            vals.append(cos_y[my])
        if not vals:
            return float("nan")
        return float(np.concatenate(vals).mean())
    return float(np.concatenate([cos_x.ravel(), cos_y.ravel()]).mean())


def potential_roughness(u: np.ndarray, region_mask: Optional[np.ndarray] = None) -> float:
    u_t = u.astype(np.float64)
    dx = u_t[:, 1:] - u_t[:, :-1]
    dy = u_t[1:, :] - u_t[:-1, :]
    if region_mask is not None:
        mx = region_mask[:, :-1] & region_mask[:, 1:]
        my = region_mask[:-1, :] & region_mask[1:, :]
        vals = []
        if mx.any():
            vals.append(dx[mx])
        if my.any():
            vals.append(dy[my])
        if not vals:
            return float("nan")
        return float(np.concatenate(vals).std())
    return float(np.concatenate([dx.ravel(), dy.ravel()]).std())


def region_mask_from_center(
    xx: np.ndarray, yy: np.ndarray, center: np.ndarray, radius: float
) -> np.ndarray:
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2


def passage_success_cycle_linear(traj: np.ndarray, passage_x: float) -> float:
    final = traj[-1]
    return float((final[:, 0] > passage_x).astype(np.float32).mean())


def passage_success_bifurcation(traj: np.ndarray, passage_x: float) -> float:
    final = traj[-1]
    return float((final[:, 0] > passage_x).astype(np.float32).mean())


def branch_balance_bifurcation(
    traj: np.ndarray, passage_x: float, y_thresh: float
) -> Tuple[float, float]:
    final = traj[-1]
    downstream = final[:, 0] > passage_x
    if downstream.sum() == 0:
        return float("nan"), float("nan")
    b1 = ((final[:, 1] > y_thresh) & downstream).astype(np.float32).sum() / downstream.astype(np.float32).sum()
    b2 = ((final[:, 1] < -y_thresh) & downstream).astype(np.float32).sum() / downstream.astype(np.float32).sum()
    return float(b1), float(b2)


def passage_success_cycle_bifurcation(
    traj: np.ndarray, passage_x: float, y_thresh: float
) -> float:
    final = traj[-1]
    ok = (final[:, 0] > passage_x) & (np.abs(final[:, 1]) > y_thresh)
    return float(ok.astype(np.float32).mean())


def compute_junction_metrics(
    topology_key: str,
    system: UniversalGraphSystem,
    traj: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    u: np.ndarray,
    drift: np.ndarray,
) -> Dict[str, Any]:
    jcfg = junction_cfg(topology_key)
    center = junction_center_for(system, topology_key)
    jmask = region_mask_from_center(xx, yy, center, JUNCTION_RADIUS)

    row: Dict[str, Any] = {
        "topology": topology_key,
        "drift_smoothness_global": drift_smoothness_score(drift),
        "drift_smoothness_junction": drift_smoothness_score(drift, jmask),
        "potential_roughness_global": potential_roughness(u),
        "potential_roughness_junction": potential_roughness(u, jmask),
        "junction_center_x": float(center[0]),
        "junction_center_y": float(center[1]),
        "rim_docking_error": rim_docking_error(system),
    }

    if topology_key == "cycle_linear":
        row["passage_success"] = passage_success_cycle_linear(traj, float(jcfg["passage_x"]))
        edge = system.edges[0]
        cycle = next(n for n in system.nodes if isinstance(n, CycleNode))
        row["edge_start_x"] = float(edge.start[0])
        row["edge_start_y"] = float(edge.start[1])
        row["cycle_center_x"] = float(cycle.pos[0])
        row["cycle_center_y"] = float(cycle.pos[1])
        row["cycle_radius"] = float(cycle.radius)
    elif topology_key == "bifurcation":
        row["passage_success"] = passage_success_bifurcation(traj, float(jcfg["passage_x"]))
        b1, b2 = branch_balance_bifurcation(
            traj, float(jcfg["passage_x"]), float(jcfg["branch_y_thresh"])
        )
        row["branch_upper_frac"] = b1
        row["branch_lower_frac"] = b2
    elif topology_key == "cycle_bifurcation":
        row["passage_success"] = passage_success_cycle_bifurcation(
            traj, float(jcfg["passage_x"]), float(jcfg["branch_y_thresh"])
        )

    return row


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def annotate_junction(
    ax: plt.Axes,
    system: UniversalGraphSystem,
    topology_key: str,
    center: np.ndarray,
) -> None:
    for edge in system.edges:
        s, e = np.asarray(edge.start), np.asarray(edge.end)
        ax.plot([s[0], e[0]], [s[1], e[1]], "w:", lw=1.2, alpha=0.65, zorder=4)
    for node in system.nodes:
        if isinstance(node, CycleNode):
            ax.add_artist(
                plt.Circle(
                    node.pos[:2],
                    node.radius,
                    fill=False,
                    ls="--",
                    color="cyan",
                    lw=1.3,
                    alpha=0.85,
                    zorder=4,
                )
            )
            ax.plot(
                node.pos[0],
                node.pos[1],
                "x",
                color="red",
                ms=7,
                mew=1.5,
                zorder=6,
                label="Cycle center",
            )
    ax.plot(center[0], center[1], "*", color="lime", ms=12, mew=0.8, zorder=7, label="Junction dock")
    circ = plt.Circle(
        (center[0], center[1]),
        JUNCTION_RADIUS,
        fill=False,
        ls="-",
        color="yellow",
        lw=1.0,
        alpha=0.7,
        zorder=5,
    )
    ax.add_artist(circ)


def plot_field_panel(
    ax: plt.Axes,
    system: UniversalGraphSystem,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    topology_key: str,
    n_grid: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_grid = N_GRID if n_grid is None else n_grid
    xx, yy, grid = make_grid(xlim, ylim, n_grid)
    u, drift = compute_potential_and_drift(system, grid, n_grid)
    ax.contourf(xx, yy, u, levels=FIELD_LEVELS, cmap="viridis_r", alpha=0.88)
    plot_landscape_streamlines(ax, system, xx, yy, u)
    center = junction_center_for(system, topology_key)
    annotate_junction(ax, system, topology_key, center)
    ax.set_title(title.replace(". ", "  ", 1))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    return xx, yy, u, drift


def plot_trajectory_panel(
    ax: plt.Axes,
    traj: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    rng: np.random.Generator,
    n_show: int | None = None,
    system: Optional[UniversalGraphSystem] = None,
    topology_key: Optional[str] = None,
    *,
    show_potential_bg: bool = False,
) -> None:
    n_show = N_TRAJ_SHOW if n_show is None else n_show
    if show_potential_bg and system is not None:
        xx_t, yy_t, grid_t = make_grid(xlim, ylim, N_GRID_TRAJ_BG)
        u_t, _ = compute_potential_and_drift(system, grid_t, N_GRID_TRAJ_BG)
        ax.contourf(xx_t, yy_t, u_t, levels=TRAJ_BG_LEVELS, cmap="viridis_r", alpha=TRAJ_BG_ALPHA, zorder=0)

    if system is not None and topology_key is not None:
        center = junction_center_for(system, topology_key)
        annotate_junction(ax, system, topology_key, center)

    t, n, _ = traj.shape
    idx = rng.choice(n, size=min(n_show, n), replace=False)
    for i in idx:
        path = np.asarray(traj[:, i, :])
        ax.plot(path[:, 0], path[:, 1], color="white", lw=0.45, alpha=0.45, zorder=3)
    final = np.asarray(traj[-1])
    ax.scatter(final[:, 0], final[:, 1], s=3, c="C1", alpha=0.65, zorder=5, edgecolors="none")
    ax.set_title(title.replace(". ", "  ", 1))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


def plot_metrics_panel(ax: plt.Axes, rows: List[Dict[str, Any]], title: str = "F  Junction continuity metrics") -> None:
    labels = [str(r["topology"]).replace("_", "-") for r in rows]
    x = np.arange(len(labels))
    w = METRICS_BAR_WIDTH
    smooth = [float(r["drift_smoothness_junction"]) for r in rows]
    passage = [float(r["passage_success"]) for r in rows]
    rough = [float(r["potential_roughness_junction"]) for r in rows]
    rough_n = np.array(rough)
    rough_n = rough_n / (rough_n.max() + 1e-8)

    ax.bar(x - w, smooth, width=w, label="Drift smoothness (junction)")
    ax.bar(x, passage, width=w, label="Passage success")
    ax.bar(x + w, 1.0 - rough_n, width=w, label="1 − norm. pot. roughness")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")


def save_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_junction(topology_key: str) -> Dict[str, Any]:
    jcfg = junction_cfg(topology_key)
    system = BUILDERS[topology_key]()
    z0 = initialize_cells(topology_key)
    traj, _ = simulate_langevin(
        system,
        z0,
        n_steps=int(jcfg["n_steps"]),
        dt=DT,
        diffusion=DIFFUSION,
        save_every=SAVE_EVERY,
        seed=SEED,
    )
    xlim = tuple(jcfg["xlim"])
    ylim = tuple(jcfg["ylim"])
    xx, yy, grid = make_grid(xlim, ylim, N_GRID)
    u, drift = compute_potential_and_drift(system, grid, N_GRID)
    metrics = compute_junction_metrics(topology_key, system, traj, xx, yy, u, drift)
    return {
        "system": system,
        "traj": traj,
        "metrics": metrics,
        "cfg": jcfg,
        "xx": xx,
        "yy": yy,
        "u": u,
        "drift": drift,
    }
