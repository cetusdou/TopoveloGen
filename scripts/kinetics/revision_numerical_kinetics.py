"""Bounded numerical supplement; run directly with numpy/scipy/matplotlib installed.

No core patches persist. RNG injection is scoped to GraphSimulator.run, serially.
See generated README.md for estimands, time conventions, and limitations.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_generation_module import GeneSynthesizer, build_topology  # noqa: E402
from scripts.common.numerical_controls import CONFIG, run_coupled, scaffold_distance

OUT = ROOT / "data" / "results" / "revision_numerical_kinetics"
def save_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def synth_for(system):
    synth = GeneSynthesizer(system, n_genes_total=3, background_noise=0.0)
    for pattern, params in [("sigmoid", {"center": np.array([0.4]), "slope": np.array([10.0])}),
                            ("impulse", {"center": np.array([0.6]), "width": np.array([0.1])}),
                            ("sigmoid", {"center": np.array([0.7]), "slope": np.array([6.0])})]:
        synth.add_trend_gene(0, 1, pattern=pattern, params=params, include_cycle_rim=False)
    synth.beta[:] = [1.0, 1.0, 0.5]
    synth.gamma[:] = [0.5, 1.0, 2.0]
    return synth


def exact_zoh(alpha, initial, beta, gamma, dt):
    """Exact state at right endpoints for left-held alpha, via augmented expm.

    Handles beta == gamma without subtractive divided differences.
    """
    matrices = [expm(np.array([[-b, 0, 1], [b, -g, 0], [0, 0, 0.]]) * dt)
                for b, g in zip(beta, gamma)]
    state = initial.astype(float).copy()
    history = []
    for a in alpha:
        for j, matrix in enumerate(matrices):
            state[:, j, :] = state[:, j, :] @ matrix[:2, :2].T + a[:, j, None] * matrix[:2, 2]
        history.append(state.copy())
    return np.array(history)


def kinetics(synth, path, dt):
    # Supply N+1 samples for N updates; the final position only records the
    # endpoint and does not drive another update under left-held forcing.
    with contextlib.redirect_stdout(io.StringIO()):
        result = synth.generate_kinetics(np.concatenate([path, path[-1:]]), dt=dt, simulate_counts=False)
    shape = (len(path), path.shape[1], synth.n_genes)
    full_shape = (len(path)+1, path.shape[1], synth.n_genes)
    alpha = result["alpha"].reshape(full_shape)[:-1]
    u, s, velocity = [result[k].reshape(full_shape)[1:] for k in
                     ["unspliced", "spliced", "velocity"]]
    a0 = synth._compute_transcription_rate(path[0], current_s=None)
    u0, s0 = a0 / (synth.beta + 1e-8), a0 / (synth.gamma + 1e-8)
    pre_u = np.concatenate([u0[None], u[:-1]])
    pre_s = np.concatenate([s0[None], s[:-1]])
    raw_u = pre_u + dt * (alpha - synth.beta * pre_u)
    raw_s = pre_s + dt * (synth.beta * pre_u - synth.gamma * pre_s)
    np.testing.assert_allclose(u, np.maximum(raw_u, 0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(s, np.maximum(raw_s, 0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(velocity, synth.beta * u - synth.gamma * s, atol=1e-6)
    states = np.stack([u, s], axis=-1)
    initial = np.stack([u0, s0], axis=-1)
    diagnostics = dict(negative_u=int(np.count_nonzero(raw_u < 0)),
                       negative_s=int(np.count_nonzero(raw_s < 0)),
                       component_updates=int(raw_u.size + raw_s.size),
                       minimum_raw=float(min(raw_u.min(), raw_s.min())),
                       stored_velocity_poststate_rmse=rms(velocity - (synth.beta * u - synth.gamma * s)))
    return states, alpha, initial, diagnostics


def latent_experiment():
    rows, rna_rows, distance_rows, archives = [], [], [], {}
    for topology in CONFIG["topologies"]:
        system, start = build_topology(topology)
        for seed in CONFIG["seeds"]:
            print(f"latent/RNA: {topology}, seed={seed}", flush=True)
            rng = np.random.default_rng(seed)
            initial = rng.standard_normal((CONFIG["cells"], 2))
            fine_dt = min(CONFIG["dts"])
            dw = rng.standard_normal((round(CONFIG["horizon"] / fine_dt), *initial.shape)) * np.sqrt(fine_dt)
            paths, errors = {}, {}
            for dt in CONFIG["dts"]:
                paths[dt], errors[dt] = run_coupled(system, start, initial, dw, dt)
                assert np.isfinite(paths[dt]).all()
                archives[f"{topology}_{seed}_dt{dt}"] = paths[dt]
            fine = paths[fine_dt]
            synth = synth_for(system)
            # Fixed finest latent path isolates RNA error from latent-step changes.
            _, fine_alpha, init_state, _ = kinetics(synth, fine[:-1], fine_dt)
            exact_fine = exact_zoh(fine_alpha, init_state, synth.beta, synth.gamma, fine_dt)
            for dt, path in paths.items():
                factor = round(dt / fine_dt)
                delta = path - fine[::factor]
                common = path[::round(max(CONFIG["dts"]) / dt)]
                distances = scaffold_distance(system, common)
                all_distances = scaffold_distance(system, path)
                rows.append(dict(topology=topology, seed=seed, dt=dt,
                                 endpoint_rms_distance=float(np.sqrt(np.mean(np.sum(delta[-1] ** 2, axis=-1)))),
                                 path_rms_distance=float(np.sqrt(np.mean(np.sum(delta ** 2, axis=-1)))),
                                 brownian_roundoff_max=errors[dt]))
                for threshold in [0.25, CONFIG["threshold"], 1.0]:
                    distance_rows.append(dict(topology=topology, seed=seed, dt=dt, threshold=threshold,
                                              common_grid=0.02, sampled_points=int(distances.size),
                                              exceedances=int(np.sum(distances > threshold)),
                                              sampled_fraction=float(np.mean(distances > threshold)),
                                              ever_cell_fraction=float(np.mean(np.any(distances > threshold, axis=0))),
                                              native_ever_cell_fraction=float(np.mean(np.any(all_distances > threshold, axis=0))),
                                              distance_q95=float(np.quantile(distances, .95)),
                                              max_distance=float(distances.max())))
                for mode, driver in [("fixed_finest_latent", fine[::factor][:-1]), ("coupled_pipeline", path[:-1])]:
                    state, alpha, initial_state, diag = kinetics(synth, driver, dt)
                    exact = exact_zoh(alpha, initial_state, synth.beta, synth.gamma, dt)
                    reference = exact_fine[factor - 1::factor]
                    for j in range(3):
                        for k, species in enumerate(["u", "s"]):
                            rna_rows.append(dict(topology=topology, seed=seed, dt=dt, mode=mode, gene=j,
                                                 beta=float(synth.beta[j]), gamma=float(synth.gamma[j]), species=species,
                                                 euler_vs_own_zoh_rmse=rms(state[:, :, j, k] - exact[:, :, j, k]),
                                                 vs_finest_zoh_rmse=rms(state[:, :, j, k] - reference[:, :, j, k]),
                                                 **diag))
                archives[f"{topology}_{seed}_alpha_fine"] = fine_alpha
    np.savez_compressed(OUT / "latent_and_alpha.npz", **archives)
    return rows, distance_rows, rna_rows


class HarmonicSynth(GeneSynthesizer):
    """Only forcing overridden: production RNA integrator remains unchanged."""
    def __init__(self, system, beta, gamma, omega):
        super().__init__(system, n_genes_total=1, splicing_rate=beta,
                         degradation_rate=gamma, background_noise=0.0)
        self.omega = omega

    def _compute_transcription_rate(self, z, current_s=None, scale=1.5):
        return (CONFIG["alpha_baseline"] + CONFIG["alpha_amplitude"] * np.cos(self.omega * z[:, :1])).astype(np.float32)


def harmonic_exact(t, beta, gamma, omega, initial):
    transfer = np.array([1 / (beta + 1j * omega), beta / ((beta + 1j * omega) * (gamma + 1j * omega))])
    baseline = CONFIG["alpha_baseline"] / np.array([beta, gamma])
    particular = baseline + CONFIG["alpha_amplitude"] * np.real(np.exp(1j * omega * t[:, None]) * transfer)
    residual = initial - baseline - CONFIG["alpha_amplitude"] * transfer.real
    eb, eg = np.exp(-beta * t), np.exp(-gamma * t)
    cross = beta * t * eb if beta == gamma else beta * (eb - eg) / (gamma - beta)
    transient = np.stack([eb * residual[0], cross * residual[0] + eg * residual[1]], axis=-1)
    return particular + transient, transfer


def harmonic_experiment():
    rows, traces = [], []
    system, _ = build_topology("linear")
    for beta, gamma in CONFIG["rates"]:
        for omega in CONFIG["omegas"]:
            for dt in CONFIG["dts"]:
                left = np.arange(round(CONFIG["harmonic_horizon"] / dt)) * dt
                right = left + dt
                path = np.zeros((len(left), 1, 2), dtype=np.float32)
                path[:, 0, 0] = left
                synth = HarmonicSynth(system, beta, gamma, omega)
                state, _, initial, diag = kinetics(synth, path, dt)
                exact, transfer = harmonic_exact(right, beta, gamma, omega, initial[0, 0])
                fit = right >= CONFIG["harmonic_fit_start"]
                design = np.column_stack([np.ones(np.sum(fit)), np.cos(omega * right[fit]), np.sin(omega * right[fit])])
                for k, species in enumerate(["u", "s"]):
                    coefficients = np.linalg.lstsq(design, state[fit, 0, 0, k], rcond=None)[0]
                    measured = complex(coefficients[1], -coefficients[2]) / CONFIG["alpha_amplitude"]
                    phase_error = float(np.angle(measured / transfer[k]))
                    rows.append(dict(beta=beta, gamma=gamma, omega=omega, omega_over_beta=omega / beta,
                                     omega_over_gamma=omega / gamma, dt=dt, species=species,
                                     amplitude=float(abs(measured)), analytic_amplitude=float(abs(transfer[k])),
                                     phase_rad=float(np.angle(measured)), analytic_phase_rad=float(np.angle(transfer[k])),
                                     amplitude_relative_error=float(abs(measured) / abs(transfer[k]) - 1),
                                     phase_error_rad=phase_error, driven_solution_rmse=rms(state[:, 0, 0, k] - exact[:, k]),
                                     **diag))
                if beta == gamma and omega == 1.0 and dt == min(CONFIG["dts"]):
                    for i in range(0, len(right), 20):
                        traces.append(dict(time=right[i], alpha_left=2 + np.cos(left[i]),
                                           u=state[i, 0, 0, 0], s=state[i, 0, 0, 1],
                                           exact_u=exact[i, 0], exact_s=exact[i, 1]))
    return rows, traces


def summarize(rows, keys, metrics):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in grouped.items():
        record = dict(zip(keys, key))
        record["n"] = len(group)
        for metric in metrics:
            values = np.array([row[metric] for row in group])
            record[metric + "_mean"] = float(values.mean())
            record[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(record)
    return output


def plot_results(latent, distances, rna, harmonic):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for topology in CONFIG["topologies"]:
        group = [r for r in latent if r["topology"] == topology and r["dt"] > .005]
        axes[0, 0].errorbar([r["dt"] for r in group], [r["path_rms_distance_mean"] for r in group],
                            yerr=[r["path_rms_distance_sd"] for r in group], marker="o", label=topology)
        group = [r for r in distances if r["topology"] == topology and r["threshold"] == .5]
        axes[0, 1].plot([r["dt"] for r in group], [r["sampled_fraction_mean"] for r in group], "o-", label=topology)
    axes[0, 0].set(xlabel="dt", ylabel="Path RMS distance vs dt=0.005", title="Latent numerical refinement (mean +/- SD)")
    axes[0, 1].set(xlabel="dt", ylabel="Sampled exceedance fraction", title="Distance > 0.5, common 0.02 time grid")
    for species in ["u", "s"]:
        groups = [[r["euler_vs_own_zoh_rmse"] for r in rna if r["dt"] == dt and r["species"] == species and r["mode"] == "fixed_finest_latent"] for dt in CONFIG["dts"]]
        axes[1, 0].loglog(CONFIG["dts"], [np.mean(g) for g in groups], "o-", label=species)
        group = [r for r in harmonic if r["beta"] == r["gamma"] == 1 and r["dt"] == .005 and r["species"] == species]
        axes[1, 1].loglog([r["omega"] for r in group], [r["amplitude"] for r in group], "o", label=species + " Euler")
        axes[1, 1].loglog([r["omega"] for r in group], [r["analytic_amplitude"] for r in group], "-", label=species + " analytic")
    axes[1, 0].set(xlabel="dt", ylabel="Mean RNA RMSE", title="Euler vs exact own held input (subsystem)")
    axes[1, 1].set(xlabel="omega / beta (beta = gamma = 1)", ylabel="Transfer amplitude", title="Harmonic input (subsystem)")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT / f"summary.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    for ax, (beta, gamma) in zip(axes, CONFIG["rates"]):
        for species in ["u", "s"]:
            group = [r for r in harmonic if r["beta"] == beta and r["gamma"] == gamma and r["dt"] == .005 and r["species"] == species]
            ax.semilogx([r["omega_over_beta"] for r in group], [r["phase_rad"] for r in group], "o", label=species + " Euler")
            ax.semilogx([r["omega_over_beta"] for r in group], [r["analytic_phase_rad"] for r in group], "-", label=species + " analytic")
        ax.set(title=f"beta={beta}, gamma={gamma}", xlabel="omega / beta", ylabel="Phase (rad)")
        ax.legend(fontsize=8)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT / f"harmonic_phase.{suffix}", dpi=180)
    plt.close(fig)


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    save_json("config.json", CONFIG)
    sources = [Path(__file__), *[(ROOT / "data_generation_module" / name) for name in
                                ["graph_core.py", "utils.py", "synthesizer_gene.py", "presets.py"]]]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    provenance = dict(python=sys.executable, python_version=platform.python_version(),
                      numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                      platform=platform.platform(), command=sys.argv, source_sha256=hashes,
                      git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    latent, distances, rna = latent_experiment()
    print("Harmonic subsystem sweep", flush=True)
    harmonic, traces = harmonic_experiment()
    latent_summary = summarize(latent, ["topology", "dt"], ["path_rms_distance", "endpoint_rms_distance"])
    distance_summary = summarize(distances, ["topology", "dt", "threshold"], ["sampled_fraction", "ever_cell_fraction", "distance_q95", "max_distance"])
    rna_summary = summarize(rna, ["topology", "dt", "mode", "gene", "species"], ["euler_vs_own_zoh_rmse", "vs_finest_zoh_rmse"])
    for name, rows in [("latent", latent), ("distance", distances), ("rna", rna), ("harmonic", harmonic),
                       ("harmonic_trace", traces), ("latent_summary", latent_summary),
                       ("distance_summary", distance_summary), ("rna_summary", rna_summary)]:
        save_csv(name + ".csv", rows)
    save_json("summary.json", dict(latent=latent_summary, distance=distance_summary, rna=rna_summary,
                                  harmonic_max_abs_amplitude_relative_error=max(abs(r["amplitude_relative_error"]) for r in harmonic),
                                  harmonic_max_abs_phase_error_rad=max(abs(r["phase_error_rad"]) for r in harmonic)))
    plot_results(latent_summary, distance_summary, rna, harmonic)
    (OUT / "README.md").write_text(__doc__ + "\n\n" + METHODS, encoding="utf-8")
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in sources)
    provenance["runtime_seconds"] = time.time() - started
    provenance["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json("provenance.json", provenance)
    save_json("output_manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in OUT.iterdir() if p.is_file() and p.name != "output_manifest.json"})
    print(f"Completed in {provenance['runtime_seconds']:.1f}s: {OUT}", flush=True)


METHODS = """# Numerical and kinetic supplement

Run: python scripts/kinetics/revision_numerical_kinetics.py (from TopoveloGen).
Five independent seeds are the replication units, with 24 coupled cells per
seed. Two existing presets (linear and cycle_bifurcation) cover a simple flow
and rotating/branching geometry at modest cost; this is not a topology survey.
Horizon 8 is identical at dt 0.02, 0.01, 0.005. Initial normals are identical;
coarse Brownian increments sum finest increments. The production simulator is
called directly with a temporary RNG injection, restored on exit. Float32 noise
roundoff is measured. No reflecting/clipping boundary is used. Drift retains
the production finite-difference epsilon=1e-4 and component gradient clip=50.
Thus latent errors concern this numerical drift, not an exact analytic SDE.

Latent RMS is sqrt(mean squared Euclidean distance), against dt=0.005 at
matched times, including the initial point. Finest-zero rows are reference
identities, not measured zero error. Three resolutions do not establish an
asymptotic convergence order. CSV summary SD is between seeds, not cell SEM.
Distances are to the union of finite edges, point nodes and cycle rims.
The primary operational exceedance threshold is 0.5 latent units, comparable
to the narrowest node sigma in these presets; 0.25 and 1.0 are sensitivity
thresholds. Fractions use a common 0.02 observation grid; native-grid ever
fractions are also supplied. These are finite-horizon sampled excursions,
not escape probabilities, continuous-time crossing estimates or confinement
proofs. No independent-time binomial confidence intervals are appropriate.

RNA uses three fixed trend genes on edge 0 with (beta,gamma)=(1,.5),(1,1),(.5,2).
No count noise, GRN, global decay or stochastic transcription is enabled.
The corrected generate_kinetics method is called. It records initialization
at t_0 and advances from each left endpoint before recording the next state.
For N held-input intervals we provide N+1 samples and compare N right endpoints.
The endpoint velocity equals beta*u-gamma*s at the stored state; the exported
stored-velocity/poststate RMS is zero after correction. Clipping candidates are
reconstructed from the actual preceding states, asserted against outputs,
and counted before clipping. Diagnostic counts repeat on species/gene rows:
deduplicate by topology,seed,dt,mode before pooling. All selected dt*rate <= .04,
so nonnegative Euler updates are expected algebraically for nonnegative input;
zero negatives here say nothing about larger rates or dt.

Exact own-ZOH comparison uses matrix exponentials for each left-held alpha
interval and identical initial values, including production epsilon/float32.
The fixed_finest_latent mode resamples a single fine latent driver; its
vs_finest_zoh error combines Euler and alpha holding error, not latent error.
The coupled_pipeline mode also changes the coupled latent solution. Finest
ZOH is a discretized driving reference, not an exact continuous latent/RNA
solution. NPZ preserves all latent paths and finest alpha for audit.

Harmonic forcing overrides only the transcription function, not the RNA
integrator: alpha=2+cos(omega*t). This deliberately artificial subsystem check
does not validate spatial alpha fields, GRNs, counts or the full pipeline.
Deterministic harmonic cases need no seed repetitions. Rates and frequencies
span omega/beta and omega/gamma, including beta=gamma; all use T=80, fitting
40<=t<=80. Least squares fits intercept, cos and sin at post-update times.
H_u=1/(beta+i*omega); H_s=beta/((beta+i*omega)*(gamma+i*omega)).
Phase is arg(H), negative for lag. Full driven exact solution includes the
initial homogeneous transient, using beta*t*exp(-beta*t) when rates coincide.
Its RMSE includes Euler and left-sampled forcing errors. Harmonic input time
is float32 in the production API, a small additional roundoff source.
No analytical subsystem result is evidence of full-pipeline correctness.

Files: per-seed CSVs, seed mean/SD summaries, harmonic transfer/trace CSVs,
JSON config/summary/provenance, SHA256 output manifest, NPZ raw latent paths,
PNG/PDF summary and phase figures. Source hashes identify the dirty checkout;
git HEAD alone is not sufficient. No existing code or manuscript is edited.
"""


if __name__ == "__main__":
    main()
