# Figure — Benchmark velocity methods

Panelized version of `experiments/results_benchmark_velo_test.py`.

This package renders:

- A single “main” composite figure (`compose_main.py`) that matches the legacy composite layout.
- Individual panels for each small plot the legacy script produces (UMAP, velocity streams/grids, cosine/MSE histograms, projection-masking index, and Gaussian projection distortion).

## Run

From repo root:

```bash
# 1) Ensure cached CSVs exist (runs legacy computation if missing)
python experiments/figures/fig_bench_velo_test/pipeline/run_pipeline.py

# 2) Single panels
python experiments/figures/fig_bench_velo_test/panels/panel_gene_mse_histogram.py

# 3) Compose the main composite figure
python experiments/figures/fig_bench_velo_test/compose/compose_main.py
```

## Outputs

Under `results/benchmark_velo_test/`:

- `data/*.csv` caches (from the legacy script)
- `figures/panel_*.png|pdf`
- `figures/figure_benchmark_velo_test.png|pdf`

