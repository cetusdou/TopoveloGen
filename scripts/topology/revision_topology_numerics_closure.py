"""Coupled-noise numerical and branch sensitivity for two topology examples.

Run from the nested repository with ``python scripts/topology/revision_topology_numerics_closure.py``.
The finest step is only a numerical comparator, not continuous-time truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_generation_module import build_topology
from scripts.topology import revision_figure8_junction as figure8
from scripts.kinetics import revision_numerical_kinetics as kinetics


OUT = ROOT / "data" / "results" / "revision_topology_numerics_closure"
SEEDS = (11, 22, 33, 44, 55)
DTS = (0.02, 0.01, 0.005, 0.0025)
FINE = DTS[-1]
NOISE = 0.15
GROUP_SIZE = 12
SETTINGS = {
    "seeds": SEEDS,
    "dts": DTS,
    "fine_dt": FINE,
    "noise_amplitude": NOISE,
    "diffusion_coefficient": NOISE**2 / 2,
    "cells_per_initial_group": GROUP_SIZE,
    "cycle_bifurcation": {
        "duration": 8.0,
        "groups": {
            "stem_upper": "12 independent N((-2,+0.12), 0.04^2 I) positions",
            "stem_lower": "12 independent N((-2,-0.12), 0.04^2 I) positions",
        },
        "terminal_rule": "upper/lower when final x > 1.2 and abs(final y) > 0.5; otherwise unresolved",
    },
    "figure_eight": {
        "duration": 4.0,
        "groups": {
            "incoming_left": "12 positions around (-0.18,-0.18), independent N(0,0.015^2 I) jitter",
            "incoming_right": "12 positions around (0.18,-0.18), independent N(0,0.015^2 I) jitter",
        },
        "crossing_rule": "first entry within radius 0.12 then first exit beyond radius 0.35; exit x sign",
        "terminal_rule": "sign of final x (left/right), with x=0 unresolved",
        "skeleton_points": 121,
    },
    "coupling": "one fine Brownian stream per seed/topology, summed into coarser increments",
    "initialization": "specified positions encoded as simulator normal draws using start position and start-node sigma*0.1",
    "comparator_caveat": "dt=0.0025 is a numerical comparator, not exact continuous-time truth",
    "figure_eight_caveat": "position-only state cannot preserve incoming branch at the self-crossing",
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cycle_initial(rng):
    upper = np.array([-2.0, 0.12]) + 0.04 * rng.standard_normal((GROUP_SIZE, 2))
    lower = np.array([-2.0, -0.12]) + 0.04 * rng.standard_normal((GROUP_SIZE, 2))
    return np.concatenate((upper, lower)), ("stem_upper", "stem_lower")


def cycle_labels(path):
    final = path[-1]
    labels = np.full(len(final), "unresolved", dtype="<U10")
    resolved = (final[:, 0] > 1.2) & (np.abs(final[:, 1]) > 0.5)
    labels[resolved & (final[:, 1] > 0)] = "upper"
    labels[resolved & (final[:, 1] < 0)] = "lower"
    return labels


def figure8_labels(path):
    crossing, entered = figure8.crossing_labels(path)
    final_x = path[-1, :, 0]
    terminal = np.where(final_x > 0, "right", np.where(final_x < 0, "left", "unresolved"))
    outgoing = np.where(crossing > 0, "right", np.where(crossing < 0, "left", "unresolved"))
    return outgoing, terminal, entered


def simulate(topology, seed):
    rng = np.random.default_rng(seed)
    if topology == "cycle_bifurcation":
        system, start = build_topology(topology)
        initial, groups = cycle_initial(rng)
    else:
        system, initial, rng = figure8.system_and_initial(seed)
        start = "Figure8"
        groups = ("incoming_left", "incoming_right")
    start_node = next(node for node in system.nodes if node.id == start)
    initial_draws = (initial - np.asarray(start_node.pos)) / (start_node.sigma * 0.1)
    duration = SETTINGS[topology if topology == "cycle_bifurcation" else "figure_eight"]["duration"]
    increments = rng.standard_normal((round(duration / FINE), *initial.shape)) * np.sqrt(FINE)
    paths = {}
    roundoff = {}
    for dt in DTS:
        paths[dt], roundoff[dt] = kinetics.run_coupled(system, start, initial_draws, increments, dt)
        if not np.isfinite(paths[dt]).all() or roundoff[dt] > 1e-5:
            raise ValueError(f"Nonfinite path or Brownian roundoff: {topology}, {seed}, {dt}")
        if not np.allclose(paths[dt][0], initial, atol=1e-6):
            raise ValueError(f"Initial positions not respected: {topology}, {seed}, {dt}")
    np.savez_compressed(OUT / f"{topology}_seed{seed}.npz", initial_positions=initial,
                        simulator_initial_draws=initial_draws,
                        fine_brownian_increments=increments,
                        **{f"path_dt{dt}": path for dt, path in paths.items()})
    return paths, roundoff, groups


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinetics.CONFIG["dts"] = list(DTS)
    kinetics.CONFIG["noise"] = NOISE
    figure8.CONFIG["dts"] = list(DTS)
    figure8.CONFIG["cells_per_incoming"] = GROUP_SIZE
    rows, cell_rows = [], []
    for topology in ("cycle_bifurcation", "figure_eight"):
        for seed in SEEDS:
            print(f"{topology} seed={seed}", flush=True)
            paths, roundoff, groups = simulate(topology, seed)
            finest = paths[FINE]
            fine_primary = cycle_labels(finest) if topology == "cycle_bifurcation" else figure8_labels(finest)[0]
            for dt in DTS:
                path = paths[dt]
                factor = round(dt / FINE)
                if path.shape[0] != finest[::factor].shape[0]:
                    raise ValueError("Unmatched observation grids")
                per_cell_rms = np.sqrt(np.mean(np.sum((path - finest[::factor])**2, axis=2), axis=0))
                if topology == "cycle_bifurcation":
                    primary = cycle_labels(path)
                    terminal = primary
                    entered = None
                else:
                    primary, terminal, entered = figure8_labels(path)
                for group_idx, group in enumerate(groups):
                    ids = range(group_idx * GROUP_SIZE, (group_idx + 1) * GROUP_SIZE)
                    subset = primary[list(ids)]
                    terminal_subset = terminal[list(ids)]
                    fine_subset = fine_primary[list(ids)]
                    joint = (subset != "unresolved") & (fine_subset != "unresolved")
                    record = dict(topology=topology, seed=seed, initial_group=group, dt=dt,
                                  n_initial=GROUP_SIZE, path_rms=float(np.sqrt(np.mean(per_cell_rms[list(ids)]**2))),
                                  brownian_roundoff_max=roundoff[dt],
                                  n_entered=int(np.sum(entered[list(ids)])) if entered is not None else "",
                                  n_primary_resolved=int(np.sum(subset != "unresolved")),
                                  n_primary_left_or_lower=int(np.sum(np.isin(subset, ("left", "lower")))),
                                  n_primary_right_or_upper=int(np.sum(np.isin(subset, ("right", "upper")))),
                                  n_primary_unresolved=int(np.sum(subset == "unresolved")),
                                  n_terminal_left_or_lower=int(np.sum(np.isin(terminal_subset, ("left", "lower")))),
                                  n_terminal_right_or_upper=int(np.sum(np.isin(terminal_subset, ("right", "upper")))),
                                  n_terminal_unresolved=int(np.sum(terminal_subset == "unresolved")),
                                  n_jointly_resolved=int(np.sum(joint)),
                                  n_primary_disagree_vs_finest=int(np.sum(subset[joint] != fine_subset[joint])))
                    rows.append(record)
                    for cell in ids:
                        cell_rows.append(dict(topology=topology, seed=seed, initial_group=group, cell=cell,
                                              dt=dt, path_rms=float(per_cell_rms[cell]),
                                              final_x=float(path[-1, cell, 0]), final_y=float(path[-1, cell, 1]),
                                              entered=bool(entered[cell]) if entered is not None else "",
                                              primary_label=primary[cell],
                                              terminal_label=terminal[cell], finest_primary_label=fine_primary[cell]))
    write_csv(OUT / "per_seed_group.csv", rows)
    write_csv(OUT / "per_cell.csv", cell_rows)
    summary = []
    for topology in ("cycle_bifurcation", "figure_eight"):
        for group in (SETTINGS[topology if topology == "cycle_bifurcation" else "figure_eight"]["groups"]):
            for dt in DTS:
                chosen = [r for r in rows if r["topology"] == topology and r["initial_group"] == group and r["dt"] == dt]
                n = sum(r["n_initial"] for r in chosen)
                record = dict(topology=topology, initial_group=group, dt=dt, n_seeds=len(chosen), n_initial=n,
                              path_rms_pooled=float(np.sqrt(np.mean([r["path_rms"]**2 for r in chosen]))),
                              n_entered=sum(r["n_entered"] for r in chosen) if topology == "figure_eight" else "")
                for key in ("n_primary_resolved", "n_primary_left_or_lower",
                            "n_primary_right_or_upper", "n_primary_unresolved", "n_terminal_left_or_lower",
                            "n_terminal_right_or_upper", "n_terminal_unresolved", "n_jointly_resolved",
                            "n_primary_disagree_vs_finest"):
                    record[key] = sum(r[key] for r in chosen)
                record["primary_right_or_upper_fraction_all"] = record["n_primary_right_or_upper"] / n
                record["terminal_right_or_upper_fraction_all"] = record["n_terminal_right_or_upper"] / n
                record["primary_disagreement_fraction_joint"] = (record["n_primary_disagree_vs_finest"] / record["n_jointly_resolved"]
                                                                 if record["n_jointly_resolved"] else "")
                summary.append(record)
    write_csv(OUT / "summary.csv", summary)
    sources = (Path(__file__), ROOT / "scripts/topology/revision_branch_convergence.py",
               ROOT / "scripts/topology/revision_figure8_junction.py",
               ROOT / "scripts/kinetics/revision_numerical_kinetics.py",
               ROOT / "data_generation_module/presets.py", ROOT / "data_generation_module/graph_core.py",
               ROOT / "data_generation_module/utils.py")
    config = dict(SETTINGS)
    config["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    canonical = json.dumps(SETTINGS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    config["settings_sha256"] = hashlib.sha256(canonical).hexdigest()
    (OUT / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    colors = {"stem_upper": "#286d79", "stem_lower": "#bc5d40",
              "incoming_left": "#286d79", "incoming_right": "#bc5d40"}
    for topology in ("cycle_bifurcation", "figure_eight"):
        for group in SETTINGS[topology if topology == "cycle_bifurcation" else "figure_eight"]["groups"]:
            selected = sorted((r for r in summary if r["topology"] == topology and r["initial_group"] == group),
                              key=lambda r: r["dt"])
            label = f"{topology.replace('_', ' ')}: {group.replace('_', ' ')}"
            steps = [r["dt"] for r in selected]
            axes[0].plot(steps, [r["path_rms_pooled"] for r in selected], "o", label=label, color=colors[group],
                         linestyle="-" if topology == "cycle_bifurcation" else "--")
            ax = axes[1] if topology == "cycle_bifurcation" else axes[2]
            ax.plot(steps, [r["n_primary_right_or_upper"] / r["n_initial"] for r in selected], "o-",
                    label=group.replace("_", " "), color=colors[group])
    axes[0].set(ylabel="Path RMS vs dt=0.0025", xlabel="Time step",
                title="Solid: cycle-bifurcation; dashed: figure-eight")
    axes[1].set(ylabel="Terminal upper / all initial", xlabel="Time step", ylim=(-0.05, 1.05))
    axes[2].set(ylabel="Crossing exit right / all initial", xlabel="Time step", ylim=(-0.05, 1.05))
    for index, ax in enumerate(axes):
        ax.set_xticks(DTS, [str(dt) for dt in DTS], rotation=35)
        ax.grid(alpha=0.2)
        if index:
            ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "topology_numerics_closure.pdf")
    plt.close(fig)
    assert len(rows) == 2 * len(SEEDS) * len(DTS) * 2
    assert len(cell_rows) == len(rows) * GROUP_SIZE
    assert all(r["n_primary_resolved"] + r["n_primary_unresolved"] == r["n_initial"] for r in rows)
    assert all(r["n_terminal_left_or_lower"] + r["n_terminal_right_or_upper"] + r["n_terminal_unresolved"] == r["n_initial"] for r in rows)
    assert all(r["path_rms"] == 0 for r in rows if r["dt"] == FINE)
    print("\n" + (OUT / "summary.csv").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
