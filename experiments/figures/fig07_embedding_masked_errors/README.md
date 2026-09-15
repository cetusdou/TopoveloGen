# Figure 7 — Embedding masked errors

Low-dimensional velocity visualizations can mask high-dimensional kinetic errors.
Cycle-bifurcation (and sweep topologies) synthetic counts → scVelo → low-D vs high-D agreement.

## Layout

```
configs/     JSON parameters (shared, pipeline, each panel, compose)
pipeline/    simulate + scVelo → results/embedding_masked_errors/data/
panels/      one executable script per panel (A–F)
compose/     editable large figure importing panel draw()
lib/         analysis + plotting core
```

## NUMBA_CACHE_DIR

scanpy/scvelo may fail without a writable Numba cache. Prefer:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python experiments/figures/fig07_embedding_masked_errors/pipeline/run_pipeline.py
```

Or rely on the package default (`.numba_cache` under the repo root, set via `configs/shared.json`).

## Run (from repo root)

```bash
# 1) Full analysis (~tens of minutes; needs scvelo)
NUMBA_CACHE_DIR=/tmp/numba_cache \
  python experiments/figures/fig07_embedding_masked_errors/pipeline/run_pipeline.py

# 2) Single panels
python experiments/figures/fig07_embedding_masked_errors/panels/panel_a_gt_velocity.py
# ... panel_b … panel_f

# 3) Large figure
python experiments/figures/fig07_embedding_masked_errors/compose/compose_main.py
```

Caches already present:

```bash
python experiments/figures/fig07_embedding_masked_errors/pipeline/run_pipeline.py --plot-only
```

## Parameters

| File | Controls |
|------|----------|
| `configs/shared.json` | output dir, seed, `numba_cache_dir` |
| `configs/pipeline.json` | main/sweep sizes, tau_low/high, synthesizer, scvelo knobs |
| `configs/panel_*.json` | title, figsize, out_name |
| `configs/compose_main.json` | 2×3 layout |

## Outputs

Under `results/embedding_masked_errors/`:

- `data/velocity_accuracy_metrics.csv`, `mismatch_by_topology.csv`, `adata_velocity_benchmark.h5ad`
- `figures/panel_*.png|pdf`
- `figures/figure7_lowD_highD_mismatch.png|pdf`

## Legacy

Previous entrypoint: `experiments/results_embedding_masked_errors.py`.
