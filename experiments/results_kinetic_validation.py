"""
Figure 5: Kinetic integration produces splicing delays and ground-truth RNA velocity.

Validates alpha(t) -> u(t), s(t), v(t) on a linear latent trajectory using
GeneSynthesizer.generate_kinetics (clean ODE, no count noise).

Usage (from repo root):
    python experiments/results_kinetic_validation.py
"""

from __future__ import annotations

import csv
import io
import contextlib
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation_module import GeneSynthesizer, GraphEdge, GraphNode, UniversalGraphSystem  # noqa: E402
from experiments.figure_style import FIGURE_2X3_SIZE, apply_publication_style, save_figure  # noqa: E402

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

OUT_DIR = REPO_ROOT / "results" / "kinetic_validation"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"

GENE_SIGMOID = 0
GENE_IMPULSE = 1


# ---------------------------------------------------------------------------
# System & trajectory
# ---------------------------------------------------------------------------
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
) -> torch.Tensor:
    """Move uniformly along the main edge: [Steps, 1, 2]."""
    edge = system.edges[0]
    p0 = edge.start.float()
    p1 = edge.end.float()
    t = torch.linspace(0.0, 1.0, n_steps)
    z = p0.unsqueeze(0) + t[:, None] * (p1 - p0).unsqueeze(0)
    return z.unsqueeze(1)


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
        params={"center": torch.tensor([0.40]), "slope": torch.tensor([10.0])},
        include_cycle_rim=True,
    )
    synth.add_trend_gene(
        edge_idx=0,
        n_genes=1,
        pattern="impulse",
        params={"center": torch.tensor([0.38]), "width": torch.tensor([0.055])},
        include_cycle_rim=True,
    )
    synth.gene_names = ["Sigmoid_lineage", "Impulse_transient"]
    synth.beta.fill_(BETA)
    synth.gamma.fill_(gamma)
    return synth


def run_kinetics(
    synth: GeneSynthesizer,
    traj: torch.Tensor,
    dt: float = DT,
) -> Dict[str, torch.Tensor]:
    with contextlib.redirect_stdout(io.StringIO()):
        return synth.generate_kinetics(traj, dt=dt, scale=ALPHA_SCALE, simulate_counts=False)


def extract_traces(
    result: Dict[str, torch.Tensor],
    steps: int,
    gene_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return alpha, u, s, v as 1D arrays [steps] for a single cell."""
    def _col(key: str) -> np.ndarray:
        x = result[key].reshape(steps, 1, -1)[:, 0, gene_idx].detach().cpu().numpy()
        return x

    return _col("alpha"), _col("unspliced"), _col("spliced"), _col("velocity")


def peak_time(y: np.ndarray, dt: float) -> float:
    return float(np.argmax(y) * dt)


def half_rise_time(y: np.ndarray, dt: float) -> float:
    y0, y1 = y[0], y.max()
    target = y0 + 0.5 * (y1 - y0)
    idx = np.searchsorted(y, target)
    return float(min(idx, len(y) - 1) * dt)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_figure5(
    system: UniversalGraphSystem,
    traj: torch.Tensor,
    result_default: Dict[str, torch.Tensor],
    gamma_results: Dict[float, Dict[str, torch.Tensor]],
) -> plt.Figure:
    apply_publication_style()
    steps = traj.shape[0]
    time = np.arange(steps) * DT
    z_path = traj[:, 0, :].detach().cpu().numpy()

    a_sig, u_sig, s_sig, _ = extract_traces(result_default, steps, GENE_SIGMOID)
    a_imp, u_imp, s_imp, v_imp = extract_traces(result_default, steps, GENE_IMPULSE)

    fig, axes = plt.subplots(2, 3, figsize=FIGURE_2X3_SIZE)

    # A: trajectory + impulse alpha-field
    ax = axes[0, 0]
    ax.plot(z_path[:, 0], z_path[:, 1], color="0.4", lw=0.8, alpha=0.5, zorder=1)
    sc = ax.scatter(
        z_path[:, 0],
        z_path[:, 1],
        c=a_imp,
        s=5,
        cmap="viridis",
        linewidths=0,
        zorder=3,
    )
    edge = system.edges[0]
    s0, s1 = edge.start.cpu().numpy(), edge.end.cpu().numpy()
    ax.plot([s0[0], s1[0]], [s0[1], s1[1]], color="0.35", ls="--", lw=1.0, alpha=0.6)
    ax.set_title("A  Linear trajectory & impulse $\\alpha$-field")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$z_1$")
    ax.set_ylabel(r"$z_2$")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\alpha$")
    cbar.ax.tick_params(labelsize=7)

    # B: sigmoid alpha/u/s
    ax = axes[0, 1]
    ax.plot(time, a_sig, label=r"$\alpha(t)$", color="C0", lw=1.5)
    ax.plot(time, u_sig, label=r"$u(t)$", color="C1", lw=1.5)
    ax.plot(time, s_sig, label=r"$s(t)$", color="C2", lw=1.5)
    t_half_u = half_rise_time(u_sig, DT)
    t_half_s = half_rise_time(s_sig, DT)
    ax.axvline(t_half_u, color="C1", ls=":", lw=0.8, alpha=0.7)
    ax.axvline(t_half_s, color="C2", ls=":", lw=0.8, alpha=0.7)
    ax.set_title("B  Sigmoid: $\\alpha$, $u$, $s$ (half-rise lag)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Abundance / rate")
    ax.legend()

    # C: impulse u-s phase portrait
    ax = axes[0, 2]
    colors = cm.viridis(np.linspace(0, 1, steps))
    for i in range(steps - 1):
        ax.plot(s_imp[i : i + 2], u_imp[i : i + 2], color=colors[i], lw=1.0, alpha=0.85)
    ax.scatter(s_imp[0], u_imp[0], s=20, c="C2", zorder=5, label="start")
    ax.scatter(s_imp[-1], u_imp[-1], s=20, c="C1", zorder=5, label="end")
    ax.set_title("C  Impulse: $u$–$s$ phase portrait")
    ax.set_xlabel(r"Spliced $s$")
    ax.set_ylabel(r"Unspliced $u$")
    ax.legend(loc="upper left")

    # D: impulse alpha/u/s with peaks
    ax = axes[1, 0]
    ax.plot(time, a_imp, label=r"$\alpha(t)$", color="C0", lw=1.5)
    ax.plot(time, u_imp, label=r"$u(t)$", color="C1", lw=1.5)
    ax.plot(time, s_imp, label=r"$s(t)$", color="C2", lw=1.5)
    t_a, t_u, t_s = peak_time(a_imp, DT), peak_time(u_imp, DT), peak_time(s_imp, DT)
    for t_peak, col, lab in [(t_a, "C0", r"$t_\alpha$"), (t_u, "C1", r"$t_u$"), (t_s, "C2", r"$t_s$")]:
        ax.axvline(t_peak, color=col, ls="--", lw=0.9, alpha=0.8)
        ax.text(t_peak, ax.get_ylim()[1] * 0.95, lab, color=col, ha="center")
    ax.set_title(r"D  Impulse: $t_\alpha < t_u < t_s$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Abundance / rate")
    ax.legend()

    # E: gamma sensitivity (u-s loops)
    ax = axes[1, 1]
    for gamma, col in zip(GAMMA_VALUES, ["C0", "C1", "C3"]):
        _, u_g, s_g, _ = extract_traces(gamma_results[gamma], steps, GENE_IMPULSE)
        ax.plot(s_g, u_g, color=col, lw=1.2, alpha=0.85, label=rf"$\gamma={gamma}$")
    ax.set_title("E  $\\gamma$-dependent hysteresis loops")
    ax.set_xlabel(r"Spliced $s$")
    ax.set_ylabel(r"Unspliced $u$")
    ax.legend(title=rf"$\beta={BETA}$")

    # F: velocity consistency
    ax = axes[1, 2]
    s_num = (s_imp[1:] - s_imp[:-1]) / DT
    v_stored = v_imp[:-1]
    ax.scatter(v_stored, s_num, s=4, alpha=0.5, c=time[:-1], cmap="viridis", linewidths=0)
    lim = max(v_stored.max(), s_num.max(), 1e-6)
    lo, hi = -0.05 * lim, lim * 1.05
    ax.plot([lo, hi], [lo, hi], color="C1", ls="--", lw=1.0, label="$y=x$")
    corr = float(np.corrcoef(v_stored, s_num)[0, 1])
    ax.set_title(f"F  Velocity consistency ($r$={corr:.4f})")
    ax.set_xlabel(r"Stored $v=\beta u-\gamma s$")
    ax.set_ylabel(r"Numerical $\Delta s/\Delta t$")
    ax.legend()

    fig.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.8)
    return fig


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Building linear system...")
    system = build_linear_system()
    traj = deterministic_edge_trajectory(system, N_STEPS)
    time = np.arange(N_STEPS) * DT

    print("Running kinetics (default gamma)...")
    synth = make_synthesizer(system, gamma=GAMMA_DEFAULT)
    result_default = run_kinetics(synth, traj)

    a_sig, u_sig, s_sig, _ = extract_traces(result_default, N_STEPS, GENE_SIGMOID)
    a_imp, u_imp, s_imp, v_imp = extract_traces(result_default, N_STEPS, GENE_IMPULSE)

    gamma_results: Dict[float, Dict[str, torch.Tensor]] = {}
    lags: Dict[float, float] = {}
    for gamma in GAMMA_VALUES:
        synth_g = make_synthesizer(system, gamma=gamma)
        gamma_results[gamma] = run_kinetics(synth_g, traj)
        _, u_g, s_g, _ = extract_traces(gamma_results[gamma], N_STEPS, GENE_IMPULSE)
        lags[gamma] = peak_time(s_g, DT) - peak_time(u_g, DT)

    # Metrics
    lag_metrics = {
        "impulse_t_alpha": peak_time(a_imp, DT),
        "impulse_t_u": peak_time(u_imp, DT),
        "impulse_t_s": peak_time(s_imp, DT),
        "sigmoid_half_rise_u": half_rise_time(u_sig, DT),
        "sigmoid_half_rise_s": half_rise_time(s_sig, DT),
        "velocity_corr": float(np.corrcoef(v_imp[:-1], (s_imp[1:] - s_imp[:-1]) / DT)[0, 1]),
    }
    save_lag_metrics_csv(DATA_DIR / "kinetic_lag_metrics.csv", lag_metrics)

    save_kinetic_trace_csv(
        DATA_DIR / "kinetic_trace_sigmoid.csv",
        time,
        {"alpha": a_sig, "unspliced": u_sig, "spliced": s_sig},
    )
    save_kinetic_trace_csv(
        DATA_DIR / "kinetic_trace_impulse.csv",
        time,
        {"alpha": a_imp, "unspliced": u_imp, "spliced": s_imp, "velocity": v_imp},
    )
    save_parameter_sensitivity_csv(DATA_DIR / "parameter_sensitivity.csv", lags)
    save_velocity_consistency_csv(
        DATA_DIR / "velocity_consistency.csv",
        v_imp[:-1],
        (s_imp[1:] - s_imp[:-1]) / DT,
        time[:-1],
    )

    torch.save(
        {
            "traj": traj,
            "result_default": {k: v.cpu() for k, v in result_default.items()},
        },
        DATA_DIR / "kinetic_tensors.pt",
    )

    print("Plotting Figure 5...")
    fig = plot_figure5(system, traj, result_default, gamma_results)
    save_figure(fig, FIG_DIR / "figure5_kinetic_validation.png")

    print(f"\nImpulse peaks: t_alpha={lag_metrics['impulse_t_alpha']:.3f}, "
          f"t_u={lag_metrics['impulse_t_u']:.3f}, t_s={lag_metrics['impulse_t_s']:.3f}")
    print(f"Sigmoid half-rise: t_u={lag_metrics['sigmoid_half_rise_u']:.3f}, "
          f"t_s={lag_metrics['sigmoid_half_rise_s']:.3f}")
    print(f"Velocity correlation: {lag_metrics['velocity_corr']:.4f}")
    print(f"Gamma lags: {lags}")
    print(f"Saved {FIG_DIR / 'figure5_kinetic_validation.png'}")


def plot_only() -> None:
    tensor_path = DATA_DIR / "kinetic_tensors.pt"
    if not tensor_path.exists():
        raise FileNotFoundError("Missing cached kinetic tensors. Run full pipeline first.")
    print("Loading cached results for Figure 5...")
    data = torch.load(tensor_path, weights_only=False)
    traj = data["traj"]
    result_default = {k: v for k, v in data["result_default"].items()}
    system = build_linear_system()
    gamma_results: Dict[float, Dict[str, torch.Tensor]] = {}
    for gamma in GAMMA_VALUES:
        synth_g = make_synthesizer(system, gamma=gamma)
        gamma_results[gamma] = run_kinetics(synth_g, traj)
    fig = plot_figure5(system, traj, result_default, gamma_results)
    save_figure(fig, FIG_DIR / "figure5_kinetic_validation.png")
    print(f"Saved {FIG_DIR / 'figure5_kinetic_validation.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Figure 5: kinetic validation")
    parser.add_argument("--plot-only", action="store_true", help="Replot from cached tensors")
    args = parser.parse_args()
    if args.plot_only:
        plot_only()
    else:
        main()
