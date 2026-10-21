"""Coupled latent-path integration and graph-distance diagnostics."""

import contextlib
import io
from unittest.mock import patch

import numpy as np

from data_generation_module import CycleNode, GraphSimulator


CONFIG = dict(seeds=[11, 22, 33, 44, 55], cells=24, horizon=8.0,
              dts=[0.02, 0.01, 0.005], noise=0.15,
              topologies=["linear", "cycle_bifurcation"], threshold=0.5,
              rates=[[1.0, 0.5], [1.0, 1.0], [0.5, 2.0]],
              omegas=[0.1, 0.3, 1.0, 3.0, 10.0], harmonic_horizon=80.0,
              harmonic_fit_start=40.0, alpha_baseline=2.0, alpha_amplitude=1.0)


def run_coupled(system, start, initial, fine_dw, dt):
    fine_dt = min(CONFIG["dts"])
    factor = round(dt / fine_dt)
    dw = fine_dw.reshape(-1, factor, *initial.shape).sum(axis=1)
    draws = iter([initial] + list(dw / np.sqrt(dt)))
    calls = 0

    def randn(*shape):
        nonlocal calls
        value = next(draws)
        assert value.shape == shape
        calls += 1
        return value.copy()

    # The production simulator casts normals to float32. Coupling is exact before
    # this cast; the tiny resulting increment discrepancy is exported explicitly.
    with patch.object(np.random, "randn", randn), contextlib.redirect_stdout(io.StringIO()):
        trajectory = GraphSimulator.run(system, start, n_cells=len(initial),
                                       steps=len(dw), dt=dt, noise_level=CONFIG["noise"],
                                       bounds=None, boundary_mode="none")
    assert calls == len(dw) + 1
    effective = (dw / np.sqrt(dt)).astype(np.float32) * CONFIG["noise"] * np.sqrt(dt)
    fine_effective = (fine_dw / np.sqrt(fine_dt)).astype(np.float32)
    fine_effective = fine_effective * CONFIG["noise"] * np.sqrt(fine_dt)
    discrepancy = effective - fine_effective.reshape(-1, factor, *initial.shape).sum(axis=1)
    return trajectory, float(np.max(np.abs(discrepancy)))


def scaffold_distance(system, trajectory):
    z = trajectory.reshape(-1, trajectory.shape[-1]).astype(float)
    distances = [np.sqrt(edge.distance_sq(z)) for edge in system.edges]
    for node in system.nodes:
        radius = np.linalg.norm(z - np.asarray(node.pos), axis=1)
        distances.append(np.abs(radius - node.radius) if isinstance(node, CycleNode) else radius)
    return np.min(distances, axis=0).reshape(trajectory.shape[:2])
