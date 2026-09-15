# Figure 4 — Alpha fields

Transcription-rate fields encode spatially localized gene programs on a
cycle–bifurcation topology (no ODE / u,s integration).

## Layout

```
configs/     JSON parameters (shared, pipeline, each panel, compose)
pipeline/    sample latent cells + alpha → results/alpha_fields/data/
panels/      one executable script per small panel (A–F)
compose/     editable large figure importing panel draw()
lib/         GeneSynthesizer setup + alpha plotting
```

## Run (from repo root)

```bash
# 1) Simulate / refresh caches (~minutes)
python experiments/figures/fig04_alpha_fields/pipeline/run_pipeline.py

# 2) Single panels
python experiments/figures/fig04_alpha_fields/panels/panel_a_topology.py
python experiments/figures/fig04_alpha_fields/panels/panel_b_branch1_sigmoid.py
python experiments/figures/fig04_alpha_fields/panels/panel_c_branch2_sigmoid.py
python experiments/figures/fig04_alpha_fields/panels/panel_d_junction_exit.py
python experiments/figures/fig04_alpha_fields/panels/panel_e_cycle_phase.py
python experiments/figures/fig04_alpha_fields/panels/panel_f_phase_heatmap.py

# 3) Large figure
python experiments/figures/fig04_alpha_fields/compose/compose_main.py
```

Caches already present: skip step 1 and run panels / compose only.

## Parameters

| File | Controls |
|------|----------|
| `configs/shared.json` | output dir, seed |
| `configs/pipeline.json` | simulation, gene layout, impulse / sigmoid params |
| `configs/panel_*.json` | title, figsize, gene_key |
| `configs/compose_main.json` | grid layout, figsize |

## Outputs

Under `results/alpha_fields/`:

- `data/latent_positions.pt`, `alpha_matrix.pt`, `gene_metadata.csv`, `alpha_field_metrics.csv`
- `figures/panel_*.png|pdf`
- `figures/figure4_alpha_fields.png|pdf`

## Legacy

Previous entrypoint: `experiments/results_alpha_fields.py`.
