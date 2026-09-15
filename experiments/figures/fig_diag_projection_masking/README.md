# Figure — Projection masking diagnosis

Panelized version of `experiments/results_projection_masking_diagnosis.py`.

## Layout

```
configs/     JSON parameters (shared, pipeline, each panel, compose)
panels/      one executable script per small panel
compose/     editable large figures (main + supplementary)
lib/         cached-data loader + per-panel drawing helpers
pipeline/    refresh cached CSVs (reuses the legacy computation entrypoint)
```

## Run

From repo root:

```bash
# 1) Ensure cached CSVs exist (runs legacy computation if missing)
python experiments/figures/fig_diag_projection_masking/pipeline/run_pipeline.py

# 2) Single panels
python experiments/figures/fig_diag_projection_masking/panels/panel_a_mismatch_vs_gene_count.py
python experiments/figures/fig_diag_projection_masking/panels/panel_d_topology_masking_map.py

# 3) Compose large figures
python experiments/figures/fig_diag_projection_masking/compose/compose_main.py
python experiments/figures/fig_diag_projection_masking/compose/compose_supp.py
```

## Outputs

Under `results/projection_masking_diagnosis/`:

- `data/gene_sweep_summary.csv`, `data/cell_diagnostics.csv` (cached for panels)
- `figures/panel_*.png|pdf`
- `figures/figure_projection_masking_diagnosis.png|pdf`
- `figures/figure_projection_masking_diagnosis_supplementary.png|pdf`

## Legacy cache note

This package draws from the legacy CSV caches (`gene_sweep_summary.csv` and `cell_diagnostics.csv`). The pipeline wrapper will invoke the legacy script only when those files are missing.

