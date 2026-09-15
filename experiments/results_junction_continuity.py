"""
Figure 3: Soft energy-flow fusion preserves continuity at topological junctions.

Evaluates cycle-linear, bifurcation, and cycle-bifurcation junctions: smart rim docking,
local potential/drift fields, trajectories crossing junctions, and continuity metrics.

Usage (from repo root):
    python experiments/results_junction_continuity.py
"""


import csv
import pickle
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_module import CycleNode, GraphEdge, GraphNode, UniversalGraphSystem  # noqa: E402
from experiments.figure_style import FIGURE_2X3_SIZE, apply_publication_style, save_figure  # noqa: E402
from experiments.results_topology_primitives import (  # noqa: E402
    plot_landscape_streamlines,
    simulate_langevin,
)

# ---------------------------------------------------------------------------
# Config
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

OUT_DIR = REPO_ROOT / "results" / "junction_continuity"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"


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


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def init_cycle_rim_cells(
    n_cells: int,
    radius: float = 1.0,
    noise: float = 0.05,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=n_cells)
    z0 = radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    z0 += rng.normal(0, noise, size=z0.shape)
    return np.asarray(z0.astype(np.float32))


def init_stem_cells(n_cells: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z0 = rng.normal([-2.0, 0.0], 0.08, size=(n_cells, 2)).astype(np.float32)
    return np.asarray(z0)


# ---------------------------------------------------------------------------
# Grid / fields
# ---------------------------------------------------------------------------
def make_grid(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    n_grid: int = N_GRID,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(xlim[0], xlim[1], n_grid)
    ys = np.linspace(ylim[0], ylim[1], n_grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([xx.flatten(), yy.flatten()], axis=1)
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
        return system.edges[0].start
    if topology_key == "bifurcation":
        branch = next(n for n in system.nodes if n.id == "Branch")
        return np.array(branch.pos[:2], dtype=np.float64)
    if topology_key == "cycle_bifurcation":
        # Use mean of cycle exit docks on upper/lower branches
        starts = np.stack([e.start for e in system.edges], axis=0)
        return starts.mean(axis=0)
    raise ValueError(topology_key)


def rim_docking_error(system: UniversalGraphSystem) -> Optional[float]:
    """| ||edge.start - center|| - R |; 0 means perfect rim docking."""
    cycle = next((n for n in system.nodes if isinstance(n, CycleNode)), None)
    if cycle is None or not system.edges:
        return None
    edge = system.edges[0]
    center = np.asarray(cycle.pos[:2], dtype=np.float32)
    start = edge.start.astype(np.float32)
    dist = np.linalg.norm(start - center).item()
    return abs(dist - cycle.radius)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _normalize_vectors(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def drift_smoothness_score(drift: np.ndarray, region_mask: Optional[np.ndarray] = None) -> float:
    """Mean cosine similarity of drift between grid neighbors (higher = smoother)."""
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


def passage_success_cycle_linear(traj: np.ndarray) -> float:
    final = traj[-1]
    return float((final[:, 0] > 2.2).astype(np.float32).mean())


def passage_success_bifurcation(traj: np.ndarray) -> float:
    final = traj[-1]
    return float((final[:, 0] > 1.5).astype(np.float32).mean())


def branch_balance_bifurcation(traj: np.ndarray) -> Tuple[float, float]:
    final = traj[-1]
    downstream = final[:, 0] > 1.5
    if downstream.sum() == 0:
        return float("nan"), float("nan")
    b1 = ((final[:, 1] > 0.25) & downstream).astype(np.float32).sum() / downstream.astype(np.float32).sum()
    b2 = ((final[:, 1] < -0.25) & downstream).astype(np.float32).sum() / downstream.astype(np.float32).sum()
    return float(b1), float(b2)


def passage_success_cycle_bifurcation(traj: np.ndarray) -> float:
    final = traj[-1]
    ok = (final[:, 0] > 2.3) & (np.abs(final[:, 1]) > 0.25)
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
        row["passage_success"] = passage_success_cycle_linear(traj)
        edge = system.edges[0]
        cycle = next(n for n in system.nodes if isinstance(n, CycleNode))
        row["edge_start_x"] = float(edge.start[0])
        row["edge_start_y"] = float(edge.start[1])
        row["cycle_center_x"] = float(cycle.pos[0])
        row["cycle_center_y"] = float(cycle.pos[1])
        row["cycle_radius"] = float(cycle.radius)
    elif topology_key == "bifurcation":
        row["passage_success"] = passage_success_bifurcation(traj)
        b1, b2 = branch_balance_bifurcation(traj)
        row["branch_upper_frac"] = b1
        row["branch_lower_frac"] = b2
    elif topology_key == "cycle_bifurcation":
        row["passage_success"] = passage_success_cycle_bifurcation(traj)

    return row


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _annotate_junction(
    ax: plt.Axes,
    system: UniversalGraphSystem,
    topology_key: str,
    center: np.ndarray,
) -> None:
    for edge in system.edges:
        s, e = edge.start, edge.end
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
            ax.plot(node.pos[0], node.pos[1], "x", color="red", ms=7, mew=1.5, zorder=6, label="Cycle center")
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
    n_grid: int = N_GRID,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xx, yy, grid = make_grid(xlim, ylim, n_grid)
    u, drift = compute_potential_and_drift(system, grid, n_grid)
    ax.contourf(xx, yy, u, levels=50, cmap="viridis_r", alpha=0.88)
    plot_landscape_streamlines(ax, system, xx, yy, u)
    center = junction_center_for(system, topology_key)
    _annotate_junction(ax, system, topology_key, center)
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
    n_show: int = 50,
    system: Optional[UniversalGraphSystem] = None,
    topology_key: Optional[str] = None,
) -> None:
    if system is not None and topology_key is not None:
        center = junction_center_for(system, topology_key)
        _annotate_junction(ax, system, topology_key, center)

    t, n, _ = traj.shape
    idx = rng.choice(n, size=min(n_show, n), replace=False)
    for i in idx:
        path = traj[:, i, :]
        ax.plot(path[:, 0], path[:, 1], color="white", lw=0.45, alpha=0.45, zorder=3)
    final = traj[-1]
    ax.scatter(final[:, 0], final[:, 1], s=3, c="C1", alpha=0.65, zorder=5, edgecolors="none")
    ax.set_title(title.replace(". ", "  ", 1))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")


def plot_metrics_panel(ax: plt.Axes, rows: List[Dict[str, Any]]) -> None:
    labels = [r["topology"].replace("_", "-") for r in rows]
    x = np.arange(len(labels))
    w = 0.25
    smooth = [float(r["drift_smoothness_junction"]) for r in rows]
    passage = [float(r["passage_success"]) for r in rows]
    rough = [float(r["potential_roughness_junction"]) for r in rows]
    # normalize roughness for display (lower is better)
    rough_n = np.array(rough)
    rough_n = rough_n / (rough_n.max() + 1e-8)

    ax.bar(x - w, smooth, width=w, label="Drift smoothness (junction)")
    ax.bar(x, passage, width=w, label="Passage success")
    ax.bar(x + w, 1.0 - rough_n, width=w, label="1 − norm. pot. roughness")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("F  Junction continuity metrics")
    ax.legend(loc="lower right")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plot_figure3(results: Dict[str, Dict[str, Any]], metric_rows: List[Dict[str, Any]]) -> None:
    apply_publication_style()
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_2X3_SIZE)
    cl = results["cycle_linear"]
    bi = results["bifurcation"]
    cb = results["cycle_bifurcation"]

    plot_field_panel(
        axes[0, 0],
        cl["system"],
        cl["cfg"]["xlim"],
        cl["cfg"]["ylim"],
        "A. Cycle–linear: potential & drift",
        "cycle_linear",
    )
    plot_field_panel(
        axes[0, 1],
        cl["system"],
        cl["cfg"]["zoom_xlim"],
        cl["cfg"]["zoom_ylim"],
        "B. Cycle–linear: junction zoom",
        "cycle_linear",
        n_grid=N_GRID_ZOOM,
    )
    xx_t, yy_t, grid_t = make_grid(cl["cfg"]["xlim"], cl["cfg"]["ylim"], 80)
    u_t, _ = compute_potential_and_drift(cl["system"], grid_t, 80)
    axes[0, 2].contourf(xx_t, yy_t, u_t, levels=40, cmap="viridis_r", alpha=0.55, zorder=0)
    plot_trajectory_panel(
        axes[0, 2],
        cl["traj"],
        cl["cfg"]["xlim"],
        cl["cfg"]["ylim"],
        "C. Cycle–linear: trajectories",
        rng,
        system=cl["system"],
        topology_key="cycle_linear",
    )

    plot_field_panel(
        axes[1, 0],
        bi["system"],
        bi["cfg"]["xlim"],
        bi["cfg"]["ylim"],
        "D. Bifurcation: potential & drift",
        "bifurcation",
    )
    plot_field_panel(
        axes[1, 1],
        cb["system"],
        cb["cfg"]["xlim"],
        cb["cfg"]["ylim"],
        "E. Cycle–bifurcation: mixed junction",
        "cycle_bifurcation",
    )
    plot_metrics_panel(axes[1, 2], metric_rows)

    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    save_figure(fig, FIG_DIR / "figure3_junction_continuity.png")


def get_junction_configs() -> List[Dict[str, Any]]:
    return [
        {
            "key": "cycle_linear",
            "builder": build_cycle_linear_junction,
            "init": lambda: init_cycle_rim_cells(N_CELLS, radius=1.0),
            "xlim": (-1.8, 4.0),
            "ylim": (-2.2, 2.2),
            "zoom_xlim": (0.0, 2.6),
            "zoom_ylim": (-1.4, 1.4),
            "n_steps": 4500,
        },
        {
            "key": "bifurcation",
            "builder": build_bifurcation_junction,
            "init": lambda: init_stem_cells(N_CELLS),
            "xlim": (-2.8, 2.8),
            "ylim": (-2.0, 2.0),
            "zoom_xlim": (-0.9, 1.4),
            "zoom_ylim": (-1.3, 1.3),
            "n_steps": 4000,
        },
        {
            "key": "cycle_bifurcation",
            "builder": build_cycle_bifurcation_junction,
            "init": lambda: init_cycle_rim_cells(N_CELLS, radius=1.0),
            "xlim": (-2.0, 4.0),
            "ylim": (-2.5, 2.5),
            "zoom_xlim": (0.0, 3.2),
            "zoom_ylim": (-2.0, 2.0),
            "n_steps": 5000,
        },
    ]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    configs = get_junction_configs()
    results: Dict[str, Dict[str, Any]] = {}
    metric_rows: List[Dict[str, Any]] = []

    for cfg in configs:
        key = cfg["key"]
        print(f"=== {key} ===")
        system = cfg["builder"]()
        z0 = cfg["init"]()
        traj, z_final = simulate_langevin(
            system,
            z0,
            n_steps=cfg["n_steps"],
            dt=DT,
            diffusion=DIFFUSION,
            save_every=SAVE_EVERY,
            seed=SEED,
        )

        xx, yy, grid = make_grid(cfg["xlim"], cfg["ylim"], N_GRID)
        u, drift = compute_potential_and_drift(system, grid, N_GRID)
        metrics = compute_junction_metrics(key, system, traj, xx, yy, u, drift)
        metric_rows.append(metrics)
        results[key] = {
            "system": system,
            "traj": traj,
            "metrics": metrics,
            "cfg": cfg,
            "xx": xx,
            "yy": yy,
            "u": u,
            "drift": drift,
        }

        rim = metrics.get("rim_docking_error")
        if rim is not None:
            print(f"  rim docking error: {rim:.4f} (0 = on rim)")
            print(f"  edge.start: ({metrics.get('edge_start_x', 'n/a')}, {metrics.get('edge_start_y', 'n/a')})")
        print(
            f"  junction smoothness: {metrics['drift_smoothness_junction']:.3f}, "
            f"passage: {metrics['passage_success']:.3f}"
        )

    fieldnames = sorted({k for row in metric_rows for k in row.keys()})
    with (DATA_DIR / "junction_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    pickle.dump(
        {k: results[k]["traj"] for k in results},
        open(str(DATA_DIR / "trajectories.pt"), "wb"),
    )

    plot_figure3(results, metric_rows)

    print(f"\nSaved {DATA_DIR / 'junction_metrics.csv'}")
    print(f"Saved {FIG_DIR / 'figure3_junction_continuity.png'}")


def plot_only() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = DATA_DIR / "junction_metrics.csv"
    traj_path = DATA_DIR / "trajectories.pt"
    if not metrics_path.exists() or not traj_path.exists():
        raise FileNotFoundError("Missing cached junction results. Run full pipeline first.")

    with metrics_path.open(newline="") as f:
        metric_rows = list(csv.DictReader(f))
    trajs = pickle.load(open(str(traj_path), "rb"))

    results: Dict[str, Dict[str, Any]] = {}
    for cfg in get_junction_configs():
        key = cfg["key"]
        results[key] = {"system": cfg["builder"](), "traj": trajs[key], "cfg": cfg}

    print("Loading cached results for Figure 3...")
    plot_figure3(results, metric_rows)
    print(f"Saved {FIG_DIR / 'figure3_junction_continuity.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 3: junction continuity")
    parser.add_argument("--plot-only", action="store_true", help="Replot from cached data")
    args = parser.parse_args()
    if args.plot_only:
        plot_only()
    else:
        main()
