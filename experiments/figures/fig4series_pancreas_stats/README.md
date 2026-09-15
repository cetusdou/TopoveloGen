# Figure4Series — Pancreas statistical validation

Two-layer pipeline matching `figure_4_fast.py` / `figure4_series_pipeline.py`:

1. **Stage 1** – continuous kinetics (PAGA → GraphSimulator → GeneSynthesizer), cached in `kinetics_cache.pt`
2. **Stage 2** – observation noise (`PostProcessConfig`: lib_size / NB / dropout / downsampling)

Main figure panels **A–I** (notebook validation). Supplementary **S1–S4** (formerly G–J in `plot_supplementary`).

## Layout

```
configs/     shared, pipeline (Stage1 + Stage2 fields), panels, compose
pipeline/    Stage1/2 runner → results/statistical_realism_figure4/
panels/      A–I + S1–S4 (each with draw() + main())
compose/     compose_main.py, compose_supp.py
lib/         thin-wrap of figure4_series_pipeline + metrics/draw from figure_4_fast
```

## Run (from repo root)

```bash
# 1a) Build Stage-1 cache only (~7 min first time)
python experiments/figures/fig4series_pancreas_stats/pipeline/run_pipeline.py --cache-only

# 1b) Or full Stage1+2 + CSV export
python experiments/figures/fig4series_pancreas_stats/pipeline/run_pipeline.py

# Force rebuild Stage-1
python experiments/figures/fig4series_pancreas_stats/pipeline/run_pipeline.py --force-kinetics

# 2) Single panels (re-apply Stage-2 from JSON on load)
python experiments/figures/fig4series_pancreas_stats/panels/panel_a_library_size.py
# ... panel_b … panel_i, panel_s1 … panel_s4

# 3) Large figures
python experiments/figures/fig4series_pancreas_stats/compose/compose_main.py
python experiments/figures/fig4series_pancreas_stats/compose/compose_supp.py
```

Retune noise without rebuilding kinetics: edit `configs/pipeline.json` → `stage2_postprocess`, then re-run panels/compose.

## Parameters

| File | Controls |
|------|----------|
| `configs/shared.json` | output dir, colors, labels |
| `configs/pipeline.json` | **stage1** hyper-params, **stage2_postprocess** (all PostProcessConfig fields), noise-stage demo |
| `configs/panel_*.json` | title, figsize, out_name |
| `configs/compose_*.json` | grid layout |

`lib/core.py` imports Stage1/2 from `experiments.figure4_series_pipeline` (no duplicated pipeline). Stage1 JSON values are pushed into that module’s globals before cache load/build.

## Outputs

Under `results/statistical_realism_figure4/`:

- `data/kinetics_cache.pt`, `cell_statistics.csv`, `gene_statistics.csv`, `count_statistics.csv`
- `figures/panel_*.png|pdf`
- `figures/figure4_statistical_validation.png|pdf`
- `figures/figure4_supplementary_metrics.png|pdf`

## Legacy

- `experiments/figure_4_fast.py`
- `experiments/figure4_series_pipeline.py`
