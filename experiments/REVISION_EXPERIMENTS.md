# Supplementary revision experiments

For the current manuscript figure mapping and dependency order, see
[`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md). The factor, interaction, and
cell/gene runners now share `revision_factor_core.py` and reject an existing
cache when its recorded configuration or source hashes differ. Use their
`--out` option for a fresh run after code changes; archived results remain
untouched.

These experiments generate new synthetic data. They do not silently replace
the original Figure 5/6 benchmark or validate all three inference methods.

## Coordinate and projection audit

Run from the code repository root:

```powershell
.\.venv-revision\Scripts\python.exe experiments/revision_coordinate_projection.py
```

Default design: cycle, bifurcation and cycle-bifurcation; seeds 11, 22, 33,
44, 55; 48 trajectories per dataset; 160 genes; dt=0.01; duration=12;
diffusion=0.006; snapshots from time 2 through 12 every 0.2 units; 600
snapshots sampled without replacement. The discarded initial interval is
not a claim of equilibrium. Multiple snapshots may share a trajectory;
confidence intervals use the five independent dataset seeds, not cells.

The runner uses the existing topology, drift, transcription-field and count
sampler implementations. Its explicit Euler loop integrates RNA and latent
state on the same clock and records beta*u-gamma*s at the stored RNA state.
This avoids the legacy panel runner's saved-trajectory time-step mismatch
and the generator's pre-update velocity/post-update state pairing. It is a
supplementary synchronized simulation, not a claim that legacy results were
already generated this way. No GRN feedback is enabled.

Counts follow the existing negative-binomial/dropout model. Known simulated
capture factors and the unspliced capture ratio are removed before inference.
This is an **oracle measurement control**, not ordinary library normalization.
The same fixed uniform neighbor smoother (20 neighbors plus self) acts on
both expression and reference velocity. Neighbors use log1p corrected counts;
scVelo deterministic mode uses the supplied abundance-space Ms/Mu. No gene
selection, per-gene standardization or log-space velocity comparison is used.

For composition and log-composition coordinates, both inferred and true
vectors are pushed forward at the same smoothed reference state using the
exact chain rule. The gene universe is all 160 genes. The log transform is
log(1 + 10000*s/sum(s)), including the derivative of the denominator.
GT-only transformation is explicitly a mismatch diagnostic. Cosine is not
invariant under these non-isometric maps; changes are results, not failures
to be hidden. Positive fixed cellwise scaling is a separate algebraic control.

PCA uses the first two loadings of the smoothed corrected observations without
gene scaling. UMAP uses log1p observations (min_dist=0.3, 20 neighbors, seeded).
Its velocity operator is explicitly defined here, not asserted to equal
scVelo's native velocity_embedding: positive cosines with expression-space
neighbor displacements, normalized per row, weight the embedding displacement.
Rows without positive weights have zero projected velocity and are excluded
from directional scores. The graph/embedding/operator rule are shared by
prediction and GT; transition weights depend on the supplied vector.

Per-vector norms below 1e-6 times the median positive norm are excluded.
Additional relative floors 1e-8, 1e-4 and 1e-2 are reported. Masking sensitivity
uses projected thresholds 0.5/0.7/0.9 and gene thresholds 0/0.2/0.4, with
pairwise valid denominators. No grid interpolation or arrow rescaling enters
scores. Seed-level t intervals are descriptive with only five replicates.

Outputs in results/revision_coordinate_projection include raw counts,
continuous states, true/predicted velocity, capture factors, trajectory IDs,
times, PCA loadings, embedding, smoothing matrix, per-cell metrics, module
counts, threshold sensitivity, summary CSV, figure PDF/PNG and provenance.

## Scope of reviewer coverage

- R1 Major 7: supports a controlled matched-coordinate analysis; does not close
  default preprocessing or UniTVelo/DeepVelo output-coordinate audits.
- R1 Major 8/9: supplies a same-reference projection experiment and linear PCA
  control; does not establish equivalence with original Figure 6 projection.
- R1 Major 11/12: supplies three topologies, independent seeds and sensitivity;
  does not replace a topology x kinetics x noise factorial or simulator comparison.
- R1 Minor 12/13: stores module composition, seeds and run provenance for these
  new experiments; the original figures still need their own reproducibility map.

The numerical/kinetic runner is documented separately with its measured
scope. No test files are added. Scientific simulation and analysis outputs
must be distinguished from software unit tests.
