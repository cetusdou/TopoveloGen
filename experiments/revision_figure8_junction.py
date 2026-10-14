"""Finite-time figure-eight crossing audit using coupled Brownian paths."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_generation_module import SkeletonNode, UniversalGraphSystem
from experiments.revision_numerical_kinetics import run_coupled

OUT = ROOT / "results" / "revision_figure8_junction"
CONFIG = {"seeds": [11, 22, 33, 44, 55], "cells_per_incoming": 12,
          "duration": 4.0, "dts": [0.02, 0.01, 0.005],
          "noise_amplitude": 0.15, "entry_radius": 0.12,
          "exit_radius": 0.35, "skeleton_points": 121}


def system_and_initial(seed):
    t = np.linspace(0, 2 * np.pi, CONFIG["skeleton_points"])
    skeleton = np.column_stack([2 * np.sin(t), np.sin(2 * t)]).astype(np.float32)
    node = SkeletonNode("Figure8", skeleton, potential=-3.0, sigma=0.4,
                        flow_strength=4.0, is_closed=True)
    system = UniversalGraphSystem(nodes=[node], edges=[])
    rng = np.random.default_rng(seed)
    n = CONFIG["cells_per_incoming"]
    # The two approaches have distinct incoming tangents at the same crossing.
    initial = np.repeat(np.array([[-0.18, -0.18], [0.18, -0.18]], dtype=np.float32), n, axis=0)
    initial += rng.normal(0, 0.015, initial.shape).astype(np.float32)
    return system, initial, rng


def crossing_labels(path):
    radius = np.linalg.norm(path, axis=2)
    labels = np.zeros(path.shape[1], dtype=int)
    entered = np.zeros(path.shape[1], dtype=bool)
    for i in range(path.shape[1]):
        near = np.flatnonzero(radius[:, i] < CONFIG["entry_radius"])
        if not len(near):
            continue
        entered[i] = True
        after = np.flatnonzero(radius[near[0] + 1:, i] > CONFIG["exit_radius"])
        if len(after):
            k = near[0] + 1 + after[0]
            labels[i] = 1 if path[k, i, 0] > 0 else -1
    return labels, entered


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in CONFIG["seeds"]:
        print(f"figure-eight seed={seed}", flush=True)
        system, initial, rng = system_and_initial(seed)
        fine_dt = min(CONFIG["dts"])
        increments = rng.normal(0, np.sqrt(fine_dt),
                                (round(CONFIG["duration"] / fine_dt), *initial.shape))
        paths = {}
        for dt in CONFIG["dts"]:
            paths[dt], _ = run_coupled(system, "Figure8", initial, increments, dt)
        fine = paths[fine_dt]
        fine_label, fine_enter = crossing_labels(fine)
        np.savez_compressed(OUT / f"seed{seed}.npz", initial=initial,
                            fine_brownian_increments=increments, **{f"path_{dt}": p for dt, p in paths.items()})
        for dt, path in paths.items():
            factor = round(dt / fine_dt)
            label, entered = crossing_labels(path)
            jointly_resolved = (label != 0) & (fine_label != 0)
            rows.append({"seed": seed, "dt": dt,
                         "path_rms_vs_finest": float(np.sqrt(np.mean(np.sum((path-fine[::factor])**2, axis=2)))),
                         "entered_fraction": float(entered.mean()),
                         "resolved_fraction": float(np.mean(label != 0)),
                         "jointly_resolved": int(jointly_resolved.sum()),
                         "outgoing_agreement": float(np.mean(label[jointly_resolved] == fine_label[jointly_resolved])) if jointly_resolved.any() else None,
                         "incoming_left_outgoing_right": float(np.mean(label[:len(label)//2] == 1)),
                         "incoming_right_outgoing_right": float(np.mean(label[len(label)//2:] == 1))})
    with (OUT / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for dt in CONFIG["dts"][:-1]:
        values = [r["path_rms_vs_finest"] for r in rows if r["dt"] == dt]
        axes[0].errorbar(dt, np.mean(values), yerr=np.std(values, ddof=1), fmt="o", color="#176b67")
    axes[0].set(xlabel="Time step", ylabel="Path RMS vs finest step")
    for offset, key, label, color in ((-.001, "incoming_left_outgoing_right", "Left incoming", "#176b67"),
                                      (.001, "incoming_right_outgoing_right", "Right incoming", "#b64558")):
        values = [np.mean([r[key] for r in rows if r["dt"] == dt]) for dt in CONFIG["dts"]]
        axes[1].plot(np.array(CONFIG["dts"]) + offset, values, "o", label=label, color=color)
    axes[1].set(xlabel="Time step", ylabel="Fraction exiting right", ylim=(-.05, 1.05))
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT / "figure8_junction.pdf")
    fig.savefig(OUT / "figure8_junction.png", dpi=180)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
