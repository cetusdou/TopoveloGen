Bounded numerical supplement; run directly with numpy/scipy/matplotlib installed.

No core patches persist. RNG injection is scoped to GraphSimulator.run, serially.
See generated README.md for estimands, time conventions, and limitations.


# Numerical and kinetic supplement

Run: python experiments/revision_numerical_kinetics.py (from TopoveloGen).
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
