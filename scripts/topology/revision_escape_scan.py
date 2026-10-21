"""Finite-time, sampled scaffold-excursion scan for R1 Major 3.

Run from the repository root with ``python -B scripts/topology/revision_escape_scan.py``.
The primary threshold and sensitivity grid below are fixed before this scan.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_generation_module import CycleNode, GraphSimulator, build_topology  # noqa: E402
from scripts.common.numerical_controls import scaffold_distance  # noqa: E402

OUT = ROOT / "data" / "results" / "revision_escape_scan"
CONFIG = {
    "design": "rim_initialized_first_exit_v2 (supersedes Stem-start pilot)",
    "topology": "cycle_bifurcation",
    "seeds": [11, 22, 33, 44, 55],
    "cells_per_seed": 24,
    "diffusion_D": [0.002, 0.006, 0.012, 0.024, 0.05],
    "stress_diffusion_D": [0.024, 0.05],
    "horizons": [4.0, 8.0, 16.0],
    "dt": 0.02,
    "observation_dt": 0.02,
    "initialization": "24 evenly spaced cycle-rim phases with seed-uniform phase rotation and seed-normal radial offsets",
    "initial_radial_sd": 0.02,
    "initial_max_scaffold_distance": 0.1,
    "thresholds": [0.25, 0.5, 1.0],
    "primary_threshold": 0.5,
    "distance_rule": "minimum Euclidean distance to finite edges, point nodes, and cycle rims",
    "crossing_rule": "first sampled distance strictly greater than threshold after t=0, conditional on initial distance <=0.1; sampled fractions include t=0",
    "noise_rule": "noise_level=sqrt(2*D), so Var[increment per coordinate]=2*D*dt",
    "boundary_mode": "none",
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_initial_on_rim(system, start, rng):
    cycle = next(node for node in system.nodes if isinstance(node, CycleNode))
    start_node = next(node for node in system.nodes if node.id == start)
    n = CONFIG["cells_per_seed"]
    phase = (np.arange(n) + rng.uniform(0, 1)) * (2 * np.pi / n)
    radial_offset = rng.normal(0, CONFIG["initial_radial_sd"], n)
    target = (np.asarray(cycle.pos) + (cycle.radius + radial_offset)[:, None]
              * np.column_stack((np.cos(phase), np.sin(phase))))
    # GraphSimulator obtains z0 via start_pos + normal * start_sigma * 0.1.
    # Inject the required normals only for that call; production drift is unchanged.
    initial_draw = (target - np.asarray(start_node.pos)) / (start_node.sigma * 0.1)
    return initial_draw, phase, radial_offset


def simulate(system, start, initial_draw, normals, D):
    draws = iter((initial_draw, *normals))
    calls = 0

    def randn(*shape):
        nonlocal calls
        value = next(draws)
        assert value.shape == shape
        calls += 1
        return value.copy()

    with patch.object(np.random, "randn", randn), contextlib.redirect_stdout(io.StringIO()):
        path = GraphSimulator.run(
            system, start, n_cells=CONFIG["cells_per_seed"], steps=len(normals),
            dt=CONFIG["dt"], noise_level=math.sqrt(2 * D),
            bounds=None, boundary_mode=CONFIG["boundary_mode"],
        )
    assert calls == len(normals) + 1
    assert path.shape == (len(normals) + 1, CONFIG["cells_per_seed"], 2)
    assert np.isfinite(path).all()
    return path


def measure(distances, seed, D, horizon, threshold):
    assert np.all(distances[0] <= CONFIG["initial_max_scaffold_distance"])
    above = distances > threshold
    sampled_count = int(above.size)
    excursion_count = int(np.count_nonzero(above))
    first_exit = np.any(above[1:], axis=0)
    ever_cell_count = int(np.count_nonzero(first_exit))
    return {
        "topology": CONFIG["topology"], "seed": seed, "D": D,
        "setting": "stress" if D in CONFIG["stress_diffusion_D"] else "baseline",
        "T": horizon,
        "noise_level": math.sqrt(2 * D), "dt": CONFIG["dt"],
        "threshold": threshold, "cells": distances.shape[1],
        "sampled_times": distances.shape[0], "sampled_count": sampled_count,
        "excursion_count": excursion_count,
        "sampled_fraction": excursion_count / sampled_count,
        "ever_cell_count": ever_cell_count,
        "first_exit_cell_count": ever_cell_count,
        "ever_cell_fraction": ever_cell_count / distances.shape[1],
        "any_trajectory_excursion": int(ever_cell_count > 0),
        "distance_q95": float(np.quantile(distances, 0.95)),
        "max_distance": float(np.max(distances)),
    }


def summarize(rows, distances_by_condition):
    summary = []
    for D in CONFIG["diffusion_D"]:
        for horizon in CONFIG["horizons"]:
            for threshold in CONFIG["thresholds"]:
                group = [r for r in rows if r["D"] == D and r["T"] == horizon
                         and r["threshold"] == threshold]
                assert len(group) == len(CONFIG["seeds"])
                sampled_count = sum(r["sampled_count"] for r in group)
                excursion_count = sum(r["excursion_count"] for r in group)
                any_count = sum(r["any_trajectory_excursion"] for r in group)
                total_cells = sum(r["cells"] for r in group)
                ever_cell_count = sum(r["ever_cell_count"] for r in group)
                pooled = np.concatenate(distances_by_condition[D, horizon])
                summary.append({
                    "topology": CONFIG["topology"], "D": D, "T": horizon,
                    "setting": "stress" if D in CONFIG["stress_diffusion_D"] else "baseline",
                    "threshold": threshold, "primary": int(threshold == CONFIG["primary_threshold"]),
                    "seeds": len(group), "sampled_count": sampled_count,
                    "excursion_count": excursion_count,
                    "sampled_fraction": excursion_count / sampled_count,
                    "seed_sampled_fraction_sd": float(np.std(
                        [r["sampled_fraction"] for r in group], ddof=1)),
                    "ever_cell_count": ever_cell_count, "total_cells": total_cells,
                    "ever_cell_fraction": ever_cell_count / total_cells,
                    "any_trajectory_count": any_count,
                    "any_trajectory_fraction": any_count / len(group),
                    "distance_q95": float(np.quantile(pooled, 0.95)),
                    "max_distance": float(np.max(pooled)),
                })
    return summary


def plot(summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(8.8, 6.8), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.85])
    axes = np.array([[fig.add_subplot(grid[row, column]) for column in range(3)]
                     for row in range(2)])
    distance_ax = fig.add_subplot(grid[2, :])
    colors = ["#216b75", "#c45b38", "#69529a", "#8b6b22", "#b43f58"]
    for column, threshold in enumerate(CONFIG["thresholds"]):
        for D, color in zip(CONFIG["diffusion_D"], colors):
            group = [r for r in summary if r["D"] == D and r["threshold"] == threshold]
            times = [r["T"] for r in group]
            for row, key in enumerate(("sampled_fraction", "ever_cell_fraction")):
                axes[row, column].plot(times, [100 * r[key] for r in group],
                                       "o-", color=color, lw=1.6, ms=4.5,
                                       label=f"D = {D:g}" + (" (stress)" if D in CONFIG["stress_diffusion_D"] else ""))
        title = f"Distance > {threshold:g}"
        if threshold == CONFIG["primary_threshold"]:
            title += " (primary)"
        axes[0, column].set_title(title, fontsize=9, fontweight="bold"
                                  if threshold == CONFIG["primary_threshold"] else "normal")
        axes[1, column].set_xlabel("Horizon T")
    axes[0, 0].set_ylabel("Sampled points (%)")
    axes[1, 0].set_ylabel("Cells ever beyond (%)")
    for ax in axes.flat:
        ax.set_xticks(CONFIG["horizons"])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
    for D, color in zip(CONFIG["diffusion_D"], colors):
        group = [r for r in summary if r["D"] == D
                 and r["threshold"] == CONFIG["primary_threshold"]]
        label = f"D = {D:g}" + (" (stress)" if D in CONFIG["stress_diffusion_D"] else "")
        distance_ax.plot([r["T"] for r in group], [r["max_distance"] for r in group],
                         "o-", color=color, lw=1.5, ms=4, label=label)
        distance_ax.plot([r["T"] for r in group], [r["distance_q95"] for r in group],
                         "--", color=color, lw=1, alpha=0.8)
    for threshold in CONFIG["thresholds"]:
        distance_ax.axhline(threshold, color="#777777", ls=":" if threshold != 0.5 else "-.", lw=0.9)
        distance_ax.text(16.2, threshold, f"{threshold:g}", fontsize=7, va="center", color="#555555")
    distance_ax.set(xlabel="Horizon T", ylabel="Distance: max (solid), q95 (dashed)")
    distance_ax.set_xticks(CONFIG["horizons"])
    distance_ax.set_xlim(3.5, 17.3)
    distance_ax.set_ylim(bottom=0)
    distance_ax.grid(axis="y", color="#dddddd", lw=0.6)
    distance_ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2)
    for row, key in enumerate(("sampled_fraction", "ever_cell_fraction")):
        for column, threshold in enumerate(CONFIG["thresholds"]):
            values = [r[key] for r in summary if r["threshold"] == threshold]
            if not any(values):
                axes[row, column].set_ylim(0, 1)
                label = "No sampled exceedances" if row == 0 else "No first exits"
                axes[row, column].text(0.5, 0.55, label, ha="center", va="center",
                                       transform=axes[row, column].transAxes, fontsize=8,
                                       color="#555555")
    fig.suptitle("Cycle-bifurcation: sampled scaffold excursions", fontsize=11)
    fig.savefig(OUT / "escape_scan.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    system, start = build_topology(CONFIG["topology"])
    steps = round(max(CONFIG["horizons"]) / CONFIG["dt"])
    assert all(round(t / CONFIG["dt"]) * CONFIG["dt"] == t for t in CONFIG["horizons"])
    rows, initial_rows = [], []
    distances_by_condition = {(D, horizon): [] for D in CONFIG["diffusion_D"]
                              for horizon in CONFIG["horizons"]}
    for seed in CONFIG["seeds"]:
        rng = np.random.default_rng(seed)
        initial_draw, phase, radial_offset = make_initial_on_rim(system, start, rng)
        normals = rng.standard_normal((steps, CONFIG["cells_per_seed"], 2))
        for D in CONFIG["diffusion_D"]:
            print(f"seed={seed}, D={D:g}", flush=True)
            path = simulate(system, start, initial_draw, normals, D)
            distances = scaffold_distance(system, path)
            assert distances.shape == path.shape[:2] and np.isfinite(distances).all()
            assert np.all(distances[0] <= CONFIG["initial_max_scaffold_distance"])
            if D == CONFIG["diffusion_D"][0]:
                initial_rows.extend({"seed": seed, "cell": cell, "phase_rad": float(phase[cell]),
                                     "radial_offset": float(radial_offset[cell]),
                                     "initial_scaffold_distance": float(distances[0, cell])}
                                    for cell in range(CONFIG["cells_per_seed"]))
            for horizon in CONFIG["horizons"]:
                prefix = distances[:round(horizon / CONFIG["dt"]) + 1]
                distances_by_condition[D, horizon].append(prefix.reshape(-1))
                for threshold in CONFIG["thresholds"]:
                    rows.append(measure(prefix, seed, D, horizon, threshold))
    summary = summarize(rows, distances_by_condition)
    assert len(rows) == (len(CONFIG["seeds"]) * len(CONFIG["diffusion_D"])
                         * len(CONFIG["horizons"]) * len(CONFIG["thresholds"]))
    write_csv(OUT / "per_seed.csv", rows)
    write_csv(OUT / "summary.csv", summary)
    write_csv(OUT / "initial_distances.csv", initial_rows)
    initial_distances = np.array([row["initial_scaffold_distance"] for row in initial_rows])
    assert len(initial_distances) == len(CONFIG["seeds"]) * CONFIG["cells_per_seed"]
    initial_distribution = {
        "n": len(initial_distances), "minimum": float(initial_distances.min()),
        "mean": float(initial_distances.mean()),
        "median": float(np.median(initial_distances)),
        "q95": float(np.quantile(initial_distances, 0.95)),
        "maximum": float(initial_distances.max()),
        "fraction_at_or_below_0.1": float(np.mean(
            initial_distances <= CONFIG["initial_max_scaffold_distance"])),
    }
    sources = [Path(__file__), ROOT / "experiments" / "revision_numerical_kinetics.py",
               *[ROOT / "data_generation_module" / name for name in
                 ("graph_core.py", "utils.py", "presets.py")]]
    metadata = {
        "config": CONFIG,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "initial_distance_distribution": initial_distribution,
        "interpretation": "Corrected rim-initialized first-exit design with separately labeled diffusion stress settings; supersedes the Stem-start pilot. Distance q95 is pooled over sampled cells and times within each D/T condition. Time points on a trajectory are dependent; no independent-time inference, continuous-time crossing, or infinite-time confinement claim.",
        "summary": summary,
    }
    (OUT / "results.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    plot(summary)
    print(f"Wrote {len(rows)} replicate rows and {len(summary)} summary rows to {OUT}")


if __name__ == "__main__":
    main()
