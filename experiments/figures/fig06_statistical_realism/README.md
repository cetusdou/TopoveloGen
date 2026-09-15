# Figure 6 — Statistical realism

TopoVeloGen count matrices from cycle-bifurcation latent cells; validates
observation-model statistics in a 4×3 figure (panels A–F topology/observation,
G–L NB+dropout vs Poisson).

Pancreas / Figure4Series stats live in `fig4series_pancreas_stats/` — this package
does **not** wire `--preset figure4`.

## Layout

```
configs/     JSON parameters (shared, pipeline, each panel, compose)
pipeline/    sample cells + GeneSynthesizer → results/statistical_realism/data/
panels/      one executable script per panel (A–L)
compose/     editable large figure importing panel draw()
lib/         simulation + metrics + plotting core
```

## Run (from repo root)

```bash
# 1) Generate count caches (~minutes)
python experiments/figures/fig06_statistical_realism/pipeline/run_pipeline.py

# 2) Single panels
python experiments/figures/fig06_statistical_realism/panels/panel_a_cells_library.py
python experiments/figures/fig06_statistical_realism/panels/panel_b_library_poisson.py
# ... panel_c … panel_l

# 3) Large figure
python experiments/figures/fig06_statistical_realism/compose/compose_main.py
```

Caches already present: skip step 1 and run panels / compose only.

```bash
python experiments/figures/fig06_statistical_realism/pipeline/run_pipeline.py --plot-only
```

## Parameters

| File | Controls |
|------|----------|
| `configs/shared.json` | output dir, seed, validation labels, colors |
| `configs/pipeline.json` | n_cells/genes/steps, MAIN_CONFIG / sweep_configs, observation knobs |
| `configs/panel_*.json` | title, figsize, out_name |
| `configs/compose_main.json` | 4×3 layout, figsize |

Edit JSON then re-run the corresponding script. Do not hard-code magic numbers in panels.

## Outputs

Under `results/statistical_realism/`:

- `data/count_matrices.pt`, `cell_statistics.csv`, `gene_statistics.csv`, `count_statistics.csv`
- `figures/panel_*.png|pdf`
- `figures/figure6_statistical_realism.png|pdf`

## Legacy

Previous entrypoint: `experiments/results_statistical_realism.py` (deprecated wrapper).
