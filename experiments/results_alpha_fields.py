"""
Figure 4: Transcription-rate fields encode spatially localized gene programs.

Demonstrates z -> alpha(z) on a cycle-bifurcation topology using GeneSynthesizer
(branch sigmoid, junction exit impulse, cycle phase, noise genes).
No ODE / u,s integration.

Usage (from repo root):
    python experiments/results_alpha_fields.py
"""


import csv
import pickle
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_module import (  # noqa: E402
    CycleNode,
    GeneSynthesizer,
    GraphEdge,
    GraphNode,
    UniversalGraphSystem,
)
from experiments.figure_style import FIGURE_2X3_SIZE, apply_publication_style, save_figure  # noqa: E402
from experiments.results_junction_continuity import init_cycle_rim_cells  # noqa: E402
from experiments.results_topology_primitives import simulate_langevin  # noqa: E402

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

OUT_DIR = REPO_ROOT / "results" / "alpha_fields"
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


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Junction exit alpha (cycle-rim dock Gaussians, max over exits)
# ---------------------------------------------------------------------------
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
    center = np.asarray(cycle.pos[:2], dtype=np.float32, )
    dist_rim = np.abs(np.linalg.norm(z - center, axis=1) - cycle.radius)
    rim_gate = np.exp(-(dist_rim**2) / (2 * JUNCTION_RIM_GATE_SIGMA**2))

    bumps = []
    for dock in docks:
        dock = dock
        d2 = ((z - dock) ** 2).sum(axis=1)
        bumps.append(amplitude * np.exp(-d2 / (2 * sigma**2)))
    exit_field = np.stack(bumps, axis=1).max(axis=1)
    return exit_field * rim_gate


# ---------------------------------------------------------------------------
# Gene modules (GeneSynthesizer: sigmoid + cycle + noise; junction exit injected)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Latent cells
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
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


def plot_figure4(
    system: UniversalGraphSystem,
    z: np.ndarray,
    alpha: np.ndarray,
) -> plt.Figure:
    apply_publication_style()
    xlim = (-2.0, 4.0)
    ylim = (-2.5, 2.5)
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_2X3_SIZE)

    plot_topology_panel(axes[0, 0], system, z, xlim, ylim)
    axes[0, 0].set_title("A  Latent topology & sampled cells")

    panel_genes = [0, 1, 2, GENE_LAYOUT["cycle_start"]]
    vmin, vmax = shared_alpha_color_limits(alpha, panel_genes)

    panel_specs = [
        (axes[0, 1], 0, "B  Branch 1 sigmoid $\\alpha$-field"),
        (axes[0, 2], 1, "C  Branch 2 sigmoid $\\alpha$-field"),
        (axes[1, 0], 2, "D  Junction exit impulse $\\alpha$-field"),
        (axes[1, 1], GENE_LAYOUT["cycle_start"], "E  Example cycle-phase $\\alpha$-field"),
    ]
    for ax, gidx, title in panel_specs:
        sc = plot_alpha_scatter(
            ax, z, alpha[:, gidx], title, vmin=vmin, vmax=vmax, system=system, show_topology=True
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$\alpha$")
        cbar.ax.tick_params(labelsize=7)

    plot_cycle_phase_heatmap(axes[1, 2], system, z, alpha)

    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    return fig


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Building cycle-bifurcation topology...")
    system = build_cycle_bifurcation_system()
    docks = cycle_exit_docks(system)
    for i, d in enumerate(docks):
        print(f"  exit dock {i}: ({d[0]:.3f}, {d[1]:.3f})")

    print("Simulating latent cells (Langevin)...")
    z_cells = sample_latent_cells(system)
    print(f"  n_cells = {z_cells.shape[0]}")

    print("Setting up GeneSynthesizer...")
    synth = setup_gene_synthesizer(system)

    print("Computing alpha(z)...")
    alpha = build_full_alpha_matrix(synth, system, z_cells)

    pickle.dump(z_cells, open(str(DATA_DIR / "latent_positions.pt"), "wb"))
    pickle.dump(alpha, open(str(DATA_DIR / "alpha_matrix.pt"), "wb"))
    save_gene_metadata(synth, DATA_DIR / "gene_metadata.csv")

    metrics = compute_alpha_metrics(system, z_cells, alpha, synth)
    save_metrics_csv(metrics, DATA_DIR / "alpha_field_metrics.csv")

    print("Plotting Figure 4...")
    fig = plot_figure4(system, z_cells, alpha)
    save_figure(fig, FIG_DIR / "figure4_alpha_fields.png")

    print(f"\nBranch1 specificity: {metrics[0]['branch_specificity']:.3f}")
    print(f"Branch2 specificity: {metrics[1]['branch_specificity']:.3f}")
    print(f"Junction exit localization: {metrics[2]['localization_score']:.3f}")
    print(f"Saved {FIG_DIR / 'figure4_alpha_fields.png'}")


def plot_only() -> None:
    z_path = DATA_DIR / "latent_positions.pt"
    alpha_path = DATA_DIR / "alpha_matrix.pt"
    if not z_path.exists() or not alpha_path.exists():
        raise FileNotFoundError("Missing cached alpha-field data. Run full pipeline first.")
    print("Loading cached results for Figure 4...")
    system = build_cycle_bifurcation_system()
    z_cells = pickle.load(open(str(z_path), "rb"))
    alpha = pickle.load(open(str(alpha_path), "rb"))
    fig = plot_figure4(system, z_cells, alpha)
    save_figure(fig, FIG_DIR / "figure4_alpha_fields.png")
    print(f"Saved {FIG_DIR / 'figure4_alpha_fields.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 4: alpha fields")
    parser.add_argument("--plot-only", action="store_true", help="Replot from cached pt files")
    args = parser.parse_args()
    if args.plot_only:
        plot_only()
    else:
        main()
