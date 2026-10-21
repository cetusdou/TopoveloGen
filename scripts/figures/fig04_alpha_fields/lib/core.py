"""Figure 4 core: alpha fields on cycle–bifurcation (numpy API)."""


import csv
import pickle
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_module import (  # noqa: E402
    CycleNode,
    GeneSynthesizer,
    GraphEdge,
    GraphNode,
    UniversalGraphSystem,
)
from scripts.figures._common.style import FIGURE_2X3_SIZE, apply_publication_style, save_figure  # noqa: E402
from scripts.topology.results_junction_continuity import init_cycle_rim_cells  # noqa: E402
from scripts.topology.results_topology_primitives import simulate_langevin  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 1
N_CELLS = 800
N_SUBSAMPLE = 3000
N_STEPS = 5000
DT = 0.01
DIFFUSION = 0.006
SAVE_EVERY = 10
ALPHA_SCALE = 1.5

# Junction exit impulse: Gaussian bumps at cycle-rim exit docks (shared across branches)
JUNCTION_IMPULSE_AMPLITUDE = 4.5
JUNCTION_IMPULSE_SIGMA = 0.11
JUNCTION_RIM_GATE_SIGMA = 0.14

# Heatmap: rotate phase origin so circular boundary does not split P0/P7 peaks
PHASE_HEATMAP_OFFSET = np.pi

SIGMOID_CENTER = 0.35
SIGMOID_SLOPE = 8.0

OUT_DIR = REPO_ROOT / "data" / "results" / "alpha_fields"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

GENE_LAYOUT = {
    "Branch1_sigmoid": 0,
    "Branch2_sigmoid": 1,
    "Junction_exit": 2,
    "cycle_start": 3,
    "cycle_count": 8,
    "noise_start": 11,
    "noise_count": 10,
}
N_GENES = GENE_LAYOUT["noise_start"] + GENE_LAYOUT["noise_count"]  # 21

INIT_RIM_RADIUS = 1.0
INIT_RIM_NOISE = 0.05
SYNTH_CFG = {
    "edge_strength": 5.0,
    "cycle_strength": 3.0,
    "background_noise": 0.01,
    "sampling": "poisson",
    "dropout_prob": 0.0,
    "angular_sigma": 0.45,
    "noise_alpha_range": (0.01, 0.05),
}
XLIM = (-2.0, 4.0)
YLIM = (-2.5, 2.5)
TOPOLOGY_GRID_N = 80
TOPOLOGY_CONTOUR_LEVELS = 30
TOPOLOGY_LINE_LEVELS = 12
BRANCH_NEAR_EDGE = 0.35
BRANCH_FATE_X = 2.0
BRANCH_FATE_Y = 0.25
BRANCH_NEAR_Y = 0.15
LOCALIZATION_TOP_FRAC = 0.2
CYCLE_RIM_MASK = 0.25
CYCLE_HEATMAP_RIM = 0.22
HEATMAP_MIN_CELLS = 20
ALPHA_SCATTER_S = 5.0
COLOR_QUANTILE = 0.98
COLOR_VMAX_CAP = 5.0
COLOR_VMIN_FLOOR = 0.5
METRICS_MIN_CYCLE_CELLS = 10



# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def configure(*, out_dir=None, data_dir=None, fig_dir=None, pipeline: dict | None = None) -> None:
    global OUT_DIR, DATA_DIR, FIG_DIR
    global SEED, N_CELLS, N_SUBSAMPLE, N_STEPS, DT, DIFFUSION, SAVE_EVERY, ALPHA_SCALE
    global INIT_RIM_RADIUS, INIT_RIM_NOISE
    global JUNCTION_IMPULSE_AMPLITUDE, JUNCTION_IMPULSE_SIGMA, JUNCTION_RIM_GATE_SIGMA
    global PHASE_HEATMAP_OFFSET, SIGMOID_CENTER, SIGMOID_SLOPE
    global GENE_LAYOUT, N_GENES, SYNTH_CFG
    global XLIM, YLIM, TOPOLOGY_GRID_N, TOPOLOGY_CONTOUR_LEVELS, TOPOLOGY_LINE_LEVELS
    global BRANCH_NEAR_EDGE, BRANCH_FATE_X, BRANCH_FATE_Y, BRANCH_NEAR_Y
    global LOCALIZATION_TOP_FRAC, CYCLE_RIM_MASK, CYCLE_HEATMAP_RIM, HEATMAP_MIN_CELLS
    global ALPHA_SCATTER_S, COLOR_QUANTILE, COLOR_VMAX_CAP, COLOR_VMIN_FLOOR
    global METRICS_MIN_CYCLE_CELLS

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

    SEED = int(pipeline.get("seed", SEED))
    N_CELLS = int(pipeline.get("n_cells", N_CELLS))
    N_SUBSAMPLE = int(pipeline.get("n_subsample", N_SUBSAMPLE))
    N_STEPS = int(pipeline.get("n_steps", N_STEPS))
    DT = float(pipeline.get("dt", DT))
    DIFFUSION = float(pipeline.get("diffusion", DIFFUSION))
    SAVE_EVERY = int(pipeline.get("save_every", SAVE_EVERY))
    ALPHA_SCALE = float(pipeline.get("alpha_scale", ALPHA_SCALE))
    INIT_RIM_RADIUS = float(pipeline.get("init_rim_radius", INIT_RIM_RADIUS))
    INIT_RIM_NOISE = float(pipeline.get("init_rim_noise", INIT_RIM_NOISE))
    JUNCTION_IMPULSE_AMPLITUDE = float(
        pipeline.get("junction_impulse_amplitude", JUNCTION_IMPULSE_AMPLITUDE)
    )
    JUNCTION_IMPULSE_SIGMA = float(pipeline.get("junction_impulse_sigma", JUNCTION_IMPULSE_SIGMA))
    JUNCTION_RIM_GATE_SIGMA = float(
        pipeline.get("junction_rim_gate_sigma", JUNCTION_RIM_GATE_SIGMA)
    )
    PHASE_HEATMAP_OFFSET = float(pipeline.get("phase_heatmap_offset", PHASE_HEATMAP_OFFSET))
    SIGMOID_CENTER = float(pipeline.get("sigmoid_center", SIGMOID_CENTER))
    SIGMOID_SLOPE = float(pipeline.get("sigmoid_slope", SIGMOID_SLOPE))
    if "gene_layout" in pipeline:
        GENE_LAYOUT = {k: int(v) for k, v in pipeline["gene_layout"].items()}
        N_GENES = GENE_LAYOUT["noise_start"] + GENE_LAYOUT["noise_count"]
    if "synth" in pipeline:
        SYNTH_CFG = dict(pipeline["synth"])
        if "noise_alpha_range" in SYNTH_CFG:
            SYNTH_CFG["noise_alpha_range"] = tuple(SYNTH_CFG["noise_alpha_range"])
    if "xlim" in pipeline:
        XLIM = tuple(pipeline["xlim"])
    if "ylim" in pipeline:
        YLIM = tuple(pipeline["ylim"])
    TOPOLOGY_GRID_N = int(pipeline.get("topology_grid_n", TOPOLOGY_GRID_N))
    TOPOLOGY_CONTOUR_LEVELS = int(pipeline.get("topology_contour_levels", TOPOLOGY_CONTOUR_LEVELS))
    TOPOLOGY_LINE_LEVELS = int(pipeline.get("topology_line_levels", TOPOLOGY_LINE_LEVELS))
    BRANCH_NEAR_EDGE = float(pipeline.get("branch_near_edge", BRANCH_NEAR_EDGE))
    BRANCH_FATE_X = float(pipeline.get("branch_fate_x", BRANCH_FATE_X))
    BRANCH_FATE_Y = float(pipeline.get("branch_fate_y", BRANCH_FATE_Y))
    BRANCH_NEAR_Y = float(pipeline.get("branch_near_y", BRANCH_NEAR_Y))
    LOCALIZATION_TOP_FRAC = float(pipeline.get("localization_top_frac", LOCALIZATION_TOP_FRAC))
    CYCLE_RIM_MASK = float(pipeline.get("cycle_rim_mask", CYCLE_RIM_MASK))
    CYCLE_HEATMAP_RIM = float(pipeline.get("cycle_heatmap_rim", CYCLE_HEATMAP_RIM))
    HEATMAP_MIN_CELLS = int(pipeline.get("heatmap_min_cells", HEATMAP_MIN_CELLS))
    ALPHA_SCATTER_S = float(pipeline.get("alpha_scatter_s", ALPHA_SCATTER_S))
    COLOR_QUANTILE = float(pipeline.get("color_quantile", COLOR_QUANTILE))
    COLOR_VMAX_CAP = float(pipeline.get("color_vmax_cap", COLOR_VMAX_CAP))
    COLOR_VMIN_FLOOR = float(pipeline.get("color_vmin_floor", COLOR_VMIN_FLOOR))
    METRICS_MIN_CYCLE_CELLS = int(pipeline.get("metrics_min_cycle_cells", METRICS_MIN_CYCLE_CELLS))

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

def cycle_exit_docks(system: UniversalGraphSystem) -> List[np.ndarray]:
    docks = []
    for edge in system.edges:
        if isinstance(edge.node_start, CycleNode):
            docks.append(edge.start.astype(np.float32))
    return docks

def compute_junction_exit_alpha(
    system: UniversalGraphSystem,
    z: np.ndarray,
    amplitude: float = JUNCTION_IMPULSE_AMPLITUDE,
    sigma: float = JUNCTION_IMPULSE_SIGMA,
) -> np.ndarray:
    """
    Shared exit-transient field: localized Gaussians at smart-docked cycle rim exits.

    alpha_exit(z) = rim_gate(z) * max_e A exp(-||z - dock_e||^2 / 2 sigma^2)
    """
    docks = cycle_exit_docks(system)
    if not docks:
        return np.zeros(z.shape[0], )

    cycle = next(n for n in system.nodes if isinstance(n, CycleNode))
    center = np.asarray(cycle.pos[:2], dtype=np.float32)
    dist_rim = np.abs(np.linalg.norm(z - center, axis=1) - cycle.radius)
    rim_gate = np.exp(-(dist_rim**2) / (2 * JUNCTION_RIM_GATE_SIGMA**2))

    bumps = []
    for dock in docks:
        dock = dock
        d2 = ((z - dock) ** 2).sum(axis=1)
        bumps.append(amplitude * np.exp(-d2 / (2 * sigma**2)))
    exit_field = np.stack(bumps, axis=1).max(axis=1)
    return exit_field * rim_gate

def setup_gene_synthesizer(system: UniversalGraphSystem) -> GeneSynthesizer:
    synth = GeneSynthesizer(
        system=system,
        n_genes_total=N_GENES,
        edge_strength=5.0,
        cycle_strength=3.0,
        background_noise=0.01,
        sampling="poisson",
        dropout_prob=0.0,
        seed=SEED,
    )

    synth.add_trend_gene(
        edge_idx=0,
        n_genes=1,
        pattern="sigmoid",
        params={
            "center": np.asarray([SIGMOID_CENTER]),
            "slope": np.asarray([SIGMOID_SLOPE]),
        },
        include_cycle_rim=True,
    )
    synth.add_trend_gene(
        edge_idx=1,
        n_genes=1,
        pattern="sigmoid",
        params={
            "center": np.asarray([SIGMOID_CENTER]),
            "slope": np.asarray([SIGMOID_SLOPE]),
        },
        include_cycle_rim=True,
    )

    phases = np.linspace(0, 2 * np.pi, GENE_LAYOUT["cycle_count"], endpoint=False)
    synth.add_cycle_gene(
        node_id="Cycle",
        n_genes=GENE_LAYOUT["cycle_count"],
        phase=phases.tolist(),
        angular_sigma=0.45,
    )
    synth.add_noise_gene(n_genes=GENE_LAYOUT["noise_count"], alpha_range=(0.01, 0.05))

    names = [None] * synth.n_genes
    names[0] = "Branch1_sigmoid"
    names[1] = "Branch2_sigmoid"
    names[2] = "Junction_exit_impulse"
    for i in range(GENE_LAYOUT["cycle_count"]):
        names[GENE_LAYOUT["cycle_start"] + i] = f"Cycle_phase_{i}"
    for i in range(GENE_LAYOUT["noise_count"]):
        names[GENE_LAYOUT["noise_start"] + i] = f"Noise_{i}"
    synth.gene_names = names

    return synth

def gene_types() -> List[str]:
    types = ["branch_sigmoid", "branch_sigmoid", "junction_exit_impulse"]
    types += ["cycle_phase"] * GENE_LAYOUT["cycle_count"]
    types += ["noise"] * GENE_LAYOUT["noise_count"]
    return types

def build_full_alpha_matrix(
    synth: GeneSynthesizer,
    system: UniversalGraphSystem,
    z: np.ndarray,
) -> np.ndarray:
    """Merge GeneSynthesizer output with junction-exit column at index 2."""
    cc = GENE_LAYOUT["cycle_count"]
    nc = GENE_LAYOUT["noise_count"]
    alpha_raw = synth._compute_transcription_rate(z, current_s=None, scale=ALPHA_SCALE)

    alpha_full = np.zeros(z.shape[0], N_GENES, )
    alpha_full[:, 0] = alpha_raw[:, 0]
    alpha_full[:, 1] = alpha_raw[:, 1]
    alpha_full[:, 2] = compute_junction_exit_alpha(system, z)
    cs = GENE_LAYOUT["cycle_start"]
    alpha_full[:, cs : cs + cc] = alpha_raw[:, 2 : 2 + cc]
    ns = GENE_LAYOUT["noise_start"]
    alpha_full[:, ns : ns + nc] = alpha_raw[:, 2 + cc : 2 + cc + nc]
    return alpha_full

def sample_latent_cells(system: UniversalGraphSystem) -> np.ndarray:
    z0 = init_cycle_rim_cells(N_CELLS, radius=1.0, noise=0.05, seed=SEED)
    traj, _ = simulate_langevin(
        system,
        z0,
        n_steps=N_STEPS,
        dt=DT,
        diffusion=DIFFUSION,
        save_every=SAVE_EVERY,
        seed=SEED,
    )
    z_all = traj.reshape(-1, 2)
    n = z_all.shape[0]
    if n > N_SUBSAMPLE:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(n, size=N_SUBSAMPLE, replace=False)
        z_all = z_all[idx]
    return z_all

def branch_region_masks(
    system: UniversalGraphSystem, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    e0, e1 = system.edges[0], system.edges[1]
    near0 = np.sqrt(e0.distance_sq(z)) < 0.35
    near1 = np.sqrt(e1.distance_sq(z)) < 0.35
    fate1 = (z[:, 0] > 2.0) & (z[:, 1] > 0.25)
    fate2 = (z[:, 0] > 2.0) & (z[:, 1] < -0.25)
    mask_b1 = (near0 & (z[:, 1] > -0.15)) | fate1
    mask_b2 = (near1 & (z[:, 1] < 0.15)) | fate2
    return mask_b1, mask_b2

def localization_score_edge(
    system: UniversalGraphSystem,
    z: np.ndarray,
    alpha_g: np.ndarray,
    edge_idx: int,
    top_frac: float = 0.2,
) -> float:
    edge = system.edges[edge_idx]
    k = max(1, int(len(alpha_g) * top_frac))
    top_idx = np.argsort(alpha_g)[-k:]
    d_top = np.sqrt(edge.distance_sq(z[top_idx])).mean()
    d_all = np.sqrt(edge.distance_sq(z)).mean()
    return float(1.0 - d_top / (d_all + 1e-8))

def localization_score_exit_docks(
    system: UniversalGraphSystem, z: np.ndarray, alpha_g: np.ndarray, top_frac: float = 0.2
) -> float:
    docks = cycle_exit_docks(system)
    if not docks:
        return float("nan")
    d_min = np.full((z.shape[0],), float("inf"))
    for dock in docks:
        d = np.linalg.norm(z - dock, axis=1)
        d_min = np.minimum(d_min, d)
    k = max(1, int(len(alpha_g) * top_frac))
    top_idx = np.argsort(alpha_g)[-k:]
    d_top = d_min[top_idx].mean()
    d_all = d_min.mean()
    return float(1.0 - d_top / (d_all + 1e-8))

def branch_specificity(
    alpha_g: np.ndarray, mask_b1: np.ndarray, mask_b2: np.ndarray
) -> float:
    if mask_b1.sum() == 0 or mask_b2.sum() == 0:
        return float("nan")
    m1 = alpha_g[mask_b1].mean()
    m2 = alpha_g[mask_b2].mean()
    return float((m1 - m2) / (m1 + m2 + 1e-8))

def compute_alpha_metrics(
    system: UniversalGraphSystem,
    z: np.ndarray,
    alpha: np.ndarray,
    synth: GeneSynthesizer,
) -> List[Dict[str, Any]]:
    mask_b1, mask_b2 = branch_region_masks(system, z)
    rows: List[Dict[str, Any]] = []

    rows.append(
        {
            "gene": "Branch1_sigmoid",
            "gene_idx": 0,
            "type": "branch_sigmoid",
            "localization_score": localization_score_edge(system, z, alpha[:, 0], 0),
            "branch_specificity": branch_specificity(alpha[:, 0], mask_b1, mask_b2),
            "mean_alpha": float(alpha[:, 0].mean()),
        }
    )
    rows.append(
        {
            "gene": "Branch2_sigmoid",
            "gene_idx": 1,
            "type": "branch_sigmoid",
            "localization_score": localization_score_edge(system, z, alpha[:, 1], 1),
            "branch_specificity": branch_specificity(alpha[:, 1], mask_b2, mask_b1),
            "mean_alpha": float(alpha[:, 1].mean()),
        }
    )
    rows.append(
        {
            "gene": "Junction_exit_impulse",
            "gene_idx": 2,
            "type": "junction_exit_impulse",
            "localization_score": localization_score_exit_docks(system, z, alpha[:, 2]),
            "branch_specificity": float("nan"),
            "mean_alpha": float(alpha[:, 2].mean()),
        }
    )

    cycle_node = next(n for n in system.nodes if isinstance(n, CycleNode))
    phase = cycle_node.get_projection_phase(z)
    center = np.asarray(cycle_node.pos[:2], dtype=np.float32)
    dist_rim = np.abs(np.linalg.norm(z - center, axis=1) - cycle_node.radius)
    mask_cycle = dist_rim < 0.25

    for i in range(GENE_LAYOUT["cycle_count"]):
        gidx = GENE_LAYOUT["cycle_start"] + i
        target = float(phases_target()[i])
        if mask_cycle.sum() > 10:
            a_c = alpha[mask_cycle, gidx]
            ph_c = phase[mask_cycle]
            peak_idx = np.argmax(a_c)
            obs = float(ph_c[peak_idx])
            err = circular_phase_error(obs, target)
        else:
            err = float("nan")
        rows.append(
            {
                "gene": synth.gene_names[gidx],
                "gene_idx": gidx,
                "type": "cycle_phase",
                "localization_score": float("nan"),
                "branch_specificity": float("nan"),
                "phase_peak_error": err,
                "mean_alpha": float(alpha[:, gidx].mean()),
            }
        )

    return rows

def phases_target() -> np.ndarray:
    return np.linspace(0, 2 * np.pi, GENE_LAYOUT["cycle_count"], endpoint=False)

def circular_phase_error(obs: float, target: float) -> float:
    diff = (obs - target + np.pi) % (2 * np.pi) - np.pi
    return abs(float(diff))

def draw_topology_overlay(ax: plt.Axes, system: UniversalGraphSystem) -> None:
    for edge in system.edges:
        s, e = edge.start, edge.end
        ax.plot([s[0], e[0]], [s[1], e[1]], "k--", lw=1.1, alpha=0.65, zorder=3)
        if isinstance(edge.node_start, CycleNode):
            ax.plot(s[0], s[1], "o", color="lime", ms=6, mec="k", mew=0.5, zorder=5)

    for node in system.nodes:
        if isinstance(node, CycleNode):
            ax.add_artist(
                plt.Circle(
                    node.pos[:2],
                    node.radius,
                    fill=False,
                    ls="--",
                    color="#0077bb",
                    lw=1.3,
                    alpha=0.9,
                    zorder=3,
                )
            )
        elif isinstance(node, GraphNode) and node.id.startswith("Fate"):
            ax.plot(node.pos[0], node.pos[1], "s", color="#cc3311", ms=7, zorder=5)
            ax.text(
                node.pos[0] + 0.08,
                node.pos[1],
                node.id,
                fontsize=7,
                color="#cc3311",
                va="center",
                zorder=5,
            )

def plot_topology_panel(
    ax: plt.Axes,
    system: UniversalGraphSystem,
    z: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    n = 80
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.asarray(np.stack([xx, yy], axis=-1), dtype=np.float32).reshape(-1, 2)
    u = system.potential_numpy(grid).reshape(n, n)
    ax.contourf(xx, yy, u, levels=30, cmap="Blues", alpha=0.18, zorder=0)
    ax.contour(xx, yy, u, levels=12, colors="0.75", linewidths=0.4, alpha=0.5, zorder=0)

    draw_topology_overlay(ax, system)

    z_np = z
    ax.scatter(
        z_np[:, 0],
        z_np[:, 1],
        s=3,
        c="#1a1a1a",
        alpha=0.35,
        linewidths=0,
        rasterized=True,
        zorder=4,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")

def plot_alpha_scatter(
    ax: plt.Axes,
    z: np.ndarray,
    alpha_g: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    s: float = 5.0,
    show_topology: bool = True,
    system: Optional[UniversalGraphSystem] = None,
) -> plt.cm.ScalarMappable:
    if show_topology and system is not None:
        draw_topology_overlay(ax, system)

    z_np = z
    a_np = alpha_g
    sc = ax.scatter(
        z_np[:, 0],
        z_np[:, 1],
        c=a_np,
        s=s,
        cmap="viridis",
        alpha=0.88,
        linewidths=0,
        vmin=vmin,
        vmax=vmax,
        zorder=6,
    )
    ax.set_title(title.replace(". ", "  ", 1))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    return sc

def plot_cycle_phase_heatmap(
    ax: plt.Axes, system: UniversalGraphSystem, z: np.ndarray, alpha: np.ndarray
) -> None:
    cycle_node = next(n for n in system.nodes if isinstance(n, CycleNode))
    phase = cycle_node.get_projection_phase(z)
    center = np.asarray(cycle_node.pos[:2], dtype=np.float32)
    dist_rim = np.abs(np.linalg.norm(z - center, axis=1) - cycle_node.radius)
    mask = dist_rim < 0.22

    if mask.sum() < 20:
        ax.text(0.5, 0.5, "Insufficient cycle-rim cells", ha="center", va="center", transform=ax.transAxes)
        return

    phase_c = phase[mask]
    g0, gc = GENE_LAYOUT["cycle_start"], GENE_LAYOUT["cycle_count"]
    alpha_c = alpha[mask][:, g0 : g0 + gc]

    # Rotate phase origin so 0/2pi seam avoids P0/P7 peak columns
    phase_shifted = (phase_c - PHASE_HEATMAP_OFFSET) % (2 * np.pi)
    order = np.argsort(phase_shifted)
    mat = alpha_c[order].T

    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("F  Cycle phase genes (ordered by phase)")
    ax.set_xlabel("Cells on cycle rim (phase-sorted)")
    ax.set_ylabel("Cycle genes")
    ax.set_yticks(range(gc))
    ax.set_yticklabels([f"P{i}" for i in range(gc)], fontsize=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\alpha$")
    cbar.ax.tick_params(labelsize=7)

def shared_alpha_color_limits(alpha: np.ndarray, gene_indices: List[int]) -> Tuple[float, float]:
    vals = alpha[:, gene_indices].reshape(-1)
    vmax = float(np.clip(np.quantile(vals, 0.98), 0.5, None))
    return 0.0, min(vmax, 5.0)

def save_gene_metadata(synth: GeneSynthesizer, path: Path) -> None:
    types = gene_types()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gene_idx", "gene_name", "gene_type"])
        writer.writeheader()
        for i, name in enumerate(synth.gene_names):
            writer.writerow(
                {
                    "gene_idx": i,
                    "gene_name": name if name else f"gene_{i}",
                    "gene_type": types[i] if i < len(types) else "unused",
                }
            )

def save_metrics_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def gene_index(key: str) -> int:
    if key in ("Branch1_sigmoid", "Branch2_sigmoid", "Junction_exit", "cycle_start"):
        return int(GENE_LAYOUT[key])
    raise KeyError(key)

def panel_gene_indices() -> List[int]:
    return [
        GENE_LAYOUT["Branch1_sigmoid"],
        GENE_LAYOUT["Branch2_sigmoid"],
        GENE_LAYOUT["Junction_exit"],
        GENE_LAYOUT["cycle_start"],
    ]

