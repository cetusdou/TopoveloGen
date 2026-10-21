"""Figure 5 core: kinetic validation (numpy API)."""


import csv
import pickle
import io
import contextlib
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_module import GeneSynthesizer, GraphEdge, GraphNode, UniversalGraphSystem  # noqa: E402
from scripts.figures._common.style import FIGURE_2X3_SIZE, apply_publication_style, save_figure  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 1
DT = 0.01
N_STEPS = 800
ALPHA_SCALE = 1.5
BETA = 1.0
GAMMA_DEFAULT = 0.5
GAMMA_VALUES = [0.3, 0.5, 1.0]

OUT_DIR = REPO_ROOT / "data" / "results" / "kinetic_validation"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

GENE_SIGMOID = 0
GENE_IMPULSE = 1


# ---------------------------------------------------------------------------
# System & trajectory
# ---------------------------------------------------------------------------


SIGMOID_CENTER = 0.40
SIGMOID_SLOPE = 10.0
IMPULSE_CENTER = 0.38
IMPULSE_WIDTH = 0.055
SYNTH_CFG: Dict[str, Any] = {
    "n_genes_total": 2,
    "edge_strength": 5.0,
    "cycle_strength": 3.0,
    "background_noise": 0.0,
    "dropout_prob": 0.0,
}
GENE_NAMES = ["Sigmoid_lineage", "Impulse_transient"]
SCATTER_S_TRAJ = 5
SCATTER_S_VELOCITY = 4
PHASE_START_END_S = 20


def configure(*, out_dir=None, data_dir=None, fig_dir=None, pipeline: dict | None = None) -> None:
    global OUT_DIR, DATA_DIR, FIG_DIR
    global SEED, DT, N_STEPS, ALPHA_SCALE, BETA, GAMMA_DEFAULT, GAMMA_VALUES
    global GENE_SIGMOID, GENE_IMPULSE, SIGMOID_CENTER, SIGMOID_SLOPE
    global IMPULSE_CENTER, IMPULSE_WIDTH, SYNTH_CFG, GENE_NAMES
    global SCATTER_S_TRAJ, SCATTER_S_VELOCITY, PHASE_START_END_S

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
    DT = float(pipeline.get("dt", DT))
    N_STEPS = int(pipeline.get("n_steps", N_STEPS))
    ALPHA_SCALE = float(pipeline.get("alpha_scale", ALPHA_SCALE))
    BETA = float(pipeline.get("beta", BETA))
    GAMMA_DEFAULT = float(pipeline.get("gamma_default", GAMMA_DEFAULT))
    if "gamma_values" in pipeline:
        GAMMA_VALUES = [float(g) for g in pipeline["gamma_values"]]
    GENE_SIGMOID = int(pipeline.get("gene_sigmoid", GENE_SIGMOID))
    GENE_IMPULSE = int(pipeline.get("gene_impulse", GENE_IMPULSE))
    SIGMOID_CENTER = float(pipeline.get("sigmoid_center", SIGMOID_CENTER))
    SIGMOID_SLOPE = float(pipeline.get("sigmoid_slope", SIGMOID_SLOPE))
    IMPULSE_CENTER = float(pipeline.get("impulse_center", IMPULSE_CENTER))
    IMPULSE_WIDTH = float(pipeline.get("impulse_width", IMPULSE_WIDTH))
    if "synth" in pipeline:
        SYNTH_CFG = dict(pipeline["synth"])
    if "gene_names" in pipeline:
        GENE_NAMES = list(pipeline["gene_names"])
    SCATTER_S_TRAJ = int(pipeline.get("scatter_s_traj", SCATTER_S_TRAJ))
    SCATTER_S_VELOCITY = int(pipeline.get("scatter_s_velocity", SCATTER_S_VELOCITY))
    PHASE_START_END_S = int(pipeline.get("phase_start_end_s", PHASE_START_END_S))

def build_linear_system() -> UniversalGraphSystem:
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

def deterministic_edge_trajectory(
    system: UniversalGraphSystem,
    n_steps: int = N_STEPS,
) -> np.ndarray:
    """Move uniformly along the main edge: [Steps, 1, 2]."""
    edge = system.edges[0]
    p0 = edge.start.astype(np.float32)
    p1 = edge.end.astype(np.float32)
    t = np.linspace(0.0, 1.0, n_steps)
    z = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
    return z[:, None]

def make_synthesizer(
    system: UniversalGraphSystem,
    gamma: float = GAMMA_DEFAULT,
) -> GeneSynthesizer:
    synth = GeneSynthesizer(
        system=system,
        n_genes_total=2,
        edge_strength=5.0,
        cycle_strength=3.0,
        background_noise=0.0,
        splicing_rate=BETA,
        degradation_rate=gamma,
        dropout_prob=0.0,
        seed=SEED,
    )
    synth.add_trend_gene(
        edge_idx=0,
        n_genes=1,
        pattern="sigmoid",
        params={"center": np.asarray([SIGMOID_CENTER]), "slope": np.asarray([SIGMOID_SLOPE])},
        include_cycle_rim=True,
    )
    synth.add_trend_gene(
        edge_idx=0,
        n_genes=1,
        pattern="impulse",
        params={"center": np.asarray([IMPULSE_CENTER]), "width": np.asarray([IMPULSE_WIDTH])},
        include_cycle_rim=True,
    )
    synth.gene_names = list(GENE_NAMES)
    synth.beta[:] = BETA
    synth.gamma[:] = gamma
    return synth

def run_kinetics(
    synth: GeneSynthesizer,
    traj: np.ndarray,
    dt: float = DT,
) -> Dict[str, np.ndarray]:
    with contextlib.redirect_stdout(io.StringIO()):
        return synth.generate_kinetics(traj, dt=dt, scale=ALPHA_SCALE, simulate_counts=False)

def extract_traces(
    result: Dict[str, np.ndarray],
    steps: int,
    gene_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return alpha, u, s, v as 1D arrays [steps] for a single cell."""
    def _col(key: str) -> np.ndarray:
        x = result[key].reshape(steps, 1, -1)[:, 0, gene_idx]
        return x

    return _col("alpha"), _col("unspliced"), _col("spliced"), _col("velocity")

def peak_time(y: np.ndarray, dt: float) -> float:
    return float(np.argmax(y) * dt)

def half_rise_time(y: np.ndarray, dt: float) -> float:
    y0, y1 = y[0], y.max()
    target = y0 + 0.5 * (y1 - y0)
    idx = np.searchsorted(y, target)
    return float(min(idx, len(y) - 1) * dt)

def save_kinetic_trace_csv(path: Path, time: np.ndarray, traces: Dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["time"] + list(traces.keys())
        writer.writerow(header)
        for i, t in enumerate(time):
            row = [t] + [float(traces[k][i]) for k in traces]
            writer.writerow(row)

def save_velocity_consistency_csv(
    path: Path, v_stored: np.ndarray, s_num: np.ndarray, time: np.ndarray
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "v_stored", "ds_dt_numerical"])
        writer.writeheader()
        for i in range(len(v_stored)):
            writer.writerow(
                {
                    "time": float(time[i]),
                    "v_stored": float(v_stored[i]),
                    "ds_dt_numerical": float(s_num[i]),
                }
            )

def save_parameter_sensitivity_csv(path: Path, lags: Dict[float, float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gamma", "beta", "lag_t_s_minus_t_u"])
        writer.writeheader()
        for gamma, lag in lags.items():
            writer.writerow({"gamma": gamma, "beta": BETA, "lag_t_s_minus_t_u": lag})

def save_lag_metrics_csv(path: Path, metrics: Dict[str, float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

def compute_gamma_results(
    system: UniversalGraphSystem, traj: np.ndarray
) -> Dict[float, Dict[str, np.ndarray]]:
    out: Dict[float, Dict[str, np.ndarray]] = {}
    for gamma in GAMMA_VALUES:
        synth_g = make_synthesizer(system, gamma=gamma)
        out[gamma] = run_kinetics(synth_g, traj)
    return out

def time_axis(steps: int | None = None) -> np.ndarray:
    steps = N_STEPS if steps is None else steps
    return np.arange(steps) * DT

def plot_panel_a_trajectory_alpha(
    ax: plt.Axes,
    system: UniversalGraphSystem,
    traj: np.ndarray,
    result_default: Dict[str, np.ndarray],
    title: str,
) -> None:
    steps = traj.shape[0]
    z_path = traj[:, 0, :]
    a_imp, _, _, _ = extract_traces(result_default, steps, GENE_IMPULSE)
    ax.plot(z_path[:, 0], z_path[:, 1], color="0.4", lw=0.8, alpha=0.5, zorder=1)
    sc = ax.scatter(
        z_path[:, 0],
        z_path[:, 1],
        c=a_imp,
        s=SCATTER_S_TRAJ,
        cmap="viridis",
        linewidths=0,
        zorder=3,
    )
    edge = system.edges[0]
    s0, s1 = np.asarray(edge.start), np.asarray(edge.end)
    ax.plot([s0[0], s1[0]], [s0[1], s1[1]], color="0.35", ls="--", lw=1.0, alpha=0.6)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\alpha$")
    cbar.ax.tick_params(labelsize=7)

def plot_panel_b_sigmoid_traces(
    ax: plt.Axes,
    result_default: Dict[str, np.ndarray],
    steps: int,
    title: str,
) -> None:
    time = time_axis(steps)
    a_sig, u_sig, s_sig, _ = extract_traces(result_default, steps, GENE_SIGMOID)
    ax.plot(time, a_sig, label=r"$\alpha(t)$", color="C0", lw=1.5)
    ax.plot(time, u_sig, label=r"$u(t)$", color="C1", lw=1.5)
    ax.plot(time, s_sig, label=r"$s(t)$", color="C2", lw=1.5)
    t_half_u = half_rise_time(u_sig)
    t_half_s = half_rise_time(s_sig)
    ax.axvline(t_half_u, color="C1", ls=":", lw=0.8, alpha=0.7)
    ax.axvline(t_half_s, color="C2", ls=":", lw=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Abundance / rate")
    ax.legend()

def plot_panel_c_impulse_phase(
    ax: plt.Axes,
    result_default: Dict[str, np.ndarray],
    steps: int,
    title: str,
) -> None:
    _, u_imp, s_imp, _ = extract_traces(result_default, steps, GENE_IMPULSE)
    colors = cm.viridis(np.linspace(0, 1, steps))
    for i in range(steps - 1):
        ax.plot(s_imp[i : i + 2], u_imp[i : i + 2], color=colors[i], lw=1.0, alpha=0.85)
    ax.scatter(s_imp[0], u_imp[0], s=PHASE_START_END_S, c="C2", zorder=5, label="start")
    ax.scatter(s_imp[-1], u_imp[-1], s=PHASE_START_END_S, c="C1", zorder=5, label="end")
    ax.set_title(title)
    ax.set_xlabel(r"Spliced $s$")
    ax.set_ylabel(r"Unspliced $u$")
    ax.legend(loc="upper left")

def plot_panel_d_impulse_peaks(
    ax: plt.Axes,
    result_default: Dict[str, np.ndarray],
    steps: int,
    title: str,
) -> None:
    time = time_axis(steps)
    a_imp, u_imp, s_imp, _ = extract_traces(result_default, steps, GENE_IMPULSE)
    ax.plot(time, a_imp, label=r"$\alpha(t)$", color="C0", lw=1.5)
    ax.plot(time, u_imp, label=r"$u(t)$", color="C1", lw=1.5)
    ax.plot(time, s_imp, label=r"$s(t)$", color="C2", lw=1.5)
    t_a, t_u, t_s = peak_time(a_imp), peak_time(u_imp), peak_time(s_imp)
    for t_peak, col, lab in [(t_a, "C0", r"$t_\alpha$"), (t_u, "C1", r"$t_u$"), (t_s, "C2", r"$t_s$")]:
        ax.axvline(t_peak, color=col, ls="--", lw=0.9, alpha=0.8)
        ax.text(t_peak, ax.get_ylim()[1] * 0.95, lab, color=col, ha="center")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Abundance / rate")
    ax.legend()

def plot_panel_e_gamma_hysteresis(
    ax: plt.Axes,
    gamma_results: Dict[float, Dict[str, np.ndarray]],
    steps: int,
    title: str,
) -> None:
    for gamma, col in zip(GAMMA_VALUES, ["C0", "C1", "C3"]):
        _, u_g, s_g, _ = extract_traces(gamma_results[gamma], steps, GENE_IMPULSE)
        ax.plot(s_g, u_g, color=col, lw=1.2, alpha=0.85, label=rf"$\gamma={gamma}$")
    ax.set_title(title)
    ax.set_xlabel(r"Spliced $s$")
    ax.set_ylabel(r"Unspliced $u$")
    ax.legend(title=rf"$\beta={BETA}$")

def plot_panel_f_velocity_consistency(
    ax: plt.Axes,
    result_default: Dict[str, np.ndarray],
    steps: int,
    title: str,
) -> None:
    time = time_axis(steps)
    _, _, s_imp, v_imp = extract_traces(result_default, steps, GENE_IMPULSE)
    s_num = (s_imp[1:] - s_imp[:-1]) / DT
    v_stored = v_imp[:-1]
    ax.scatter(
        v_stored,
        s_num,
        s=SCATTER_S_VELOCITY,
        alpha=0.5,
        c=time[:-1],
        cmap="viridis",
        linewidths=0,
    )
    lim = max(float(v_stored.max()), float(s_num.max()), 1e-6)
    lo, hi = -0.05 * lim, lim * 1.05
    ax.plot([lo, hi], [lo, hi], color="C1", ls="--", lw=1.0, label="$y=x$")
    corr = float(np.corrcoef(v_stored, s_num)[0, 1])
    base = title.rstrip()
    if "($r$" not in base:
        base = f"{base} ($r$={corr:.4f})"
    ax.set_title(base)
    ax.set_xlabel(r"Stored $v=\beta u-\gamma s$")
    ax.set_ylabel(r"Numerical $\Delta s/\Delta t$")
    ax.legend()

