# Revision Numerical/Kinetic Supplement

UPDATE: the production time-alignment correction has now been applied and the
experiment rerun successfully (257 seconds, project .venv-revision environment).
Stored-velocity/poststate mismatch is now zero. The historical mismatch values
below describe the pre-fix audit only. Current authoritative metrics are the CSVs
and README under results/revision_numerical_kinetics. The snapshot of the exact
run script preserves its recorded hash; subsequent edits only clarify METHODS.

## Reproduce

From `TopoveloGen`, run:

```powershell
& C:/Users/cetus/miniconda3/envs/scpy/python.exe -B experiments/revision_numerical_kinetics.py
```

The completed run took 197 seconds. PyYAML 6.0.3 was installed into scpy
with approval because the existing package imports it. No core, manuscript,
coordinate experiment, or test file was edited. No commit was made.
The script overwrites its own outputs on rerun. A separate review is pending.

## Design and Results

Five seeds, 24 cells per seed, existing linear and cycle_bifurcation presets,
physical horizon 8, noise coefficient 0.15, dt 0.02/0.01/0.005. Initial states
are shared and coarse Brownian increments sum fine increments. Maximum
effective increment discrepancy after production float32 casting was
1.49e-8. No boundary clipping or reflection was applied.

Path RMS Euclidean distance relative to dt=0.005, seed mean +/- sample SD:

| Topology | dt=0.02 | dt=0.01 |
| --- | --- | --- |
| Linear | 0.02149 +/- 0.01012 | 0.01313 +/- 0.00666 |
| Cycle/bifurcation | 0.57384 +/- 0.32126 | 0.21054 +/- 0.27432 |

Refinement reduces mean discrepancies, but branch/cycle sensitivity is large.
The finest trajectory is a numerical reference, not exact truth. This does
not establish an asymptotic order or prove that dt=0.01 is sufficiently small.

At the primary distance-to-scaffold threshold 0.5 latent units, the fraction
of sampled cell/time points exceeding the threshold on a common 0.02 grid was:

| Topology | dt=0.02 | dt=0.01 | dt=0.005 |
| --- | --- | --- | --- |
| Linear | 0% | 0% | 0% |
| Cycle/bifurcation | 0.0187% | 0.0249% | 0.0229% |

The cycle/bifurcation fraction of cells ever exceeding was 4.17%, 5%, 5%.
These are operational sampled finite-horizon excursions, not escape or
confinement results. Threshold sensitivity at 0.25 and 1.0 is exported.

For fixed finest latent drivers, mean Euler-versus-exact-held-input RMSE,
descriptively averaged over both topologies, five seeds and three genes:

| Species | dt=0.02 | dt=0.01 | dt=0.005 |
| --- | --- | --- | --- |
| Unspliced | 0.004593 | 0.002288 | 0.001142 |
| Spliced | 0.004232 | 0.002105 | 0.001050 |

No negative pre-clipping candidates: 0/4,032,000 component updates in the
coupled-pipeline cases. The separate fixed-driver cases also had none.
These updates are not independent samples. All selected dt*rate <= 0.04;
nonnegativity is expected for nonnegative forcing in this regime.

The real integrator updates at t=0, stores post-update u,s, and stores
pre-update ds. Left-endpoint input arrays avoid extending the horizon by dt;
the output states are interpreted at right endpoints. Stored velocity versus
the derivative evaluated at stored states had RMS differences 0.00251-0.01514
across latent/RNA cases. This mismatch is reported, not copied into a false
same-time derivative validation or fixed in core.

Harmonic forcing uses the production RNA integrator with only transcription
overridden to 2+cos(omega*t). Rates (beta,gamma)=(1,.5),(1,1),(.5,2), omega
0.1/0.3/1/3/10, horizon 80, fit window 40-80. These deterministic checks are
not replicated as artificial independent seeds. Across species/rates/frequencies:

| dt | Max relative amplitude error | Max absolute phase error |
| --- | --- | --- |
| 0.02 | 2.828% | 0.1993 rad |
| 0.01 | 1.311% | 0.0995 rad |
| 0.005 | 0.630% | 0.0497 rad |

## Verification and Scope

The completed script asserted production state reconstruction, velocity timing,
finite latent states, RNG call counts, and unchanged source hashes. All 17
output manifest entries were independently hash-verified. Both PNG figures
were visually inspected. No test files were created.

Separate in-memory checks compared all 15 harmonic rate/frequency analytic
solutions over t=0..8 against SciPy DOP853 at rtol=1e-11, atol=1e-12:
maximum absolute difference 2.013e-10. Matrix-exponential held-input updates
preserved equilibrium to 4.441e-15. A nonconstant random held-input check
(20 intervals, two cells, three rate pairs, dt=.02) agreed with intervalwise
adaptive integration at rtol=1e-12, atol=1e-13 to 3.433e-13.

The latent experiment retains production numerical drift, finite-difference
epsilon and gradient clipping. RNA checks omit count noise, GRNs, global
decay and stochastic transcription. Artificial harmonic forcing does not
validate spatial transcription, latent dynamics, counts, coordinate
projection, or the full pipeline. Full coupled RNA errors and isolated
fixed-driver errors are separately labeled in the CSVs.

All data, figures, configuration, source hashes, versions, raw latent paths
and detailed metric definitions are under `results/revision_numerical_kinetics`.
Read its generated README before pooling repeated diagnostic fields: RNA
counts repeat on gene/species rows and must be deduplicated by run identity.
