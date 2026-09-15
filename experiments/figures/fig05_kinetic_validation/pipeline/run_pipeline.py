"""Run kinetic ODE integration and write caches + CSV metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pickle

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.figures.fig05_kinetic_validation.lib import core  # noqa: E402
from experiments.figures.fig05_kinetic_validation.lib.data import load_cfg  # noqa: E402


def main(*, plot_only: bool = False) -> None:
    cfg = load_cfg()
    core.DATA_DIR.mkdir(parents=True, exist_ok=True)
    core.FIG_DIR.mkdir(parents=True, exist_ok=True)

    if plot_only:
        print("Pipeline plot-only: caches already expected under", core.DATA_DIR)
        return

    print("Building linear system...")
    system = core.build_linear_system()
    traj = core.deterministic_edge_trajectory(system)
    time = core.time_axis()

    print("Running kinetics (default gamma)...")
    synth = core.make_synthesizer(system, gamma=core.GAMMA_DEFAULT)
    result_default = core.run_kinetics(synth, traj)

    a_sig, u_sig, s_sig, _ = core.extract_traces(result_default, core.N_STEPS, core.GENE_SIGMOID)
    a_imp, u_imp, s_imp, v_imp = core.extract_traces(result_default, core.N_STEPS, core.GENE_IMPULSE)

    gamma_results = core.compute_gamma_results(system, traj)
    lags = {}
    for gamma, res in gamma_results.items():
        _, u_g, s_g, _ = core.extract_traces(res, core.N_STEPS, core.GENE_IMPULSE)
        lags[gamma] = core.peak_time(s_g) - core.peak_time(u_g)

    lag_metrics = {
        "impulse_t_alpha": core.peak_time(a_imp),
        "impulse_t_u": core.peak_time(u_imp),
        "impulse_t_s": core.peak_time(s_imp),
        "sigmoid_half_rise_u": core.half_rise_time(u_sig),
        "sigmoid_half_rise_s": core.half_rise_time(s_sig),
        "velocity_corr": float(np.corrcoef(v_imp[:-1], (s_imp[1:] - s_imp[:-1]) / core.DT)[0, 1]),
    }
    core.save_lag_metrics_csv(core.DATA_DIR / "kinetic_lag_metrics.csv", lag_metrics)
    core.save_kinetic_trace_csv(
        core.DATA_DIR / "kinetic_trace_sigmoid.csv",
        time,
        {"alpha": a_sig, "unspliced": u_sig, "spliced": s_sig},
    )
    core.save_kinetic_trace_csv(
        core.DATA_DIR / "kinetic_trace_impulse.csv",
        time,
        {"alpha": a_imp, "unspliced": u_imp, "spliced": s_imp, "velocity": v_imp},
    )
    core.save_parameter_sensitivity_csv(core.DATA_DIR / "parameter_sensitivity.csv", lags)
    core.save_velocity_consistency_csv(
        core.DATA_DIR / "velocity_consistency.csv",
        v_imp[:-1],
        (s_imp[1:] - s_imp[:-1]) / core.DT,
        time[:-1],
    )

    # Legacy cache format (gamma sweeps re-run at plot time for panel E)
    pickle.dump(
        {
            "traj": traj,
            "result_default": {k: v for k, v in result_default.items()},
        },
        open(str(core.DATA_DIR / "kinetic_tensors.pt"), "wb"),
    )

    print(
        f"Impulse peaks: t_alpha={lag_metrics['impulse_t_alpha']:.3f}, "
        f"t_u={lag_metrics['impulse_t_u']:.3f}, t_s={lag_metrics['impulse_t_s']:.3f}"
    )
    print(f"Velocity correlation: {lag_metrics['velocity_corr']:.4f}")
    print("Next: run panels or compose_main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fig5 pipeline: kinetic validation")
    parser.add_argument("--plot-only", action="store_true", help="Skip simulation (caches must exist)")
    args = parser.parse_args()
    main(plot_only=args.plot_only)
