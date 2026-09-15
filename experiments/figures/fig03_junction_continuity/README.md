# Figure 3 — Junction continuity

Soft energy-flow fusion preserves continuity at topological junctions
(cycle–linear, bifurcation, cycle–bifurcation).

## Layout

```
configs/     JSON parameters (shared, pipeline, each panel, compose)
pipeline/    run simulations → results/junction_continuity/data/
panels/      one executable script per small panel (A–F)
compose/     editable large figure importing panel draw()
lib/         junction systems + field/metrics plotting
```

## Run (from repo root)

```bash
# 1) Simulate / refresh caches (~minutes)
python experiments/figures/fig03_junction_continuity/pipeline/run_pipeline.py

# 2) Single panels
python experiments/figures/fig03_junction_continuity/panels/panel_a_cycle_linear_field.py
python experiments/figures/fig03_junction_continuity/panels/panel_b_cycle_linear_zoom.py
python experiments/figures/fig03_junction_continuity/panels/panel_c_cycle_linear_traj.py
python experiments/figures/fig03_junction_continuity/panels/panel_d_bifurcation_field.py
python experiments/figures/fig03_junction_continuity/panels/panel_e_cycle_bifurcation_field.py
python experiments/figures/fig03_junction_continuity/panels/panel_f_metrics.py

# 3) Large figure
python experiments/figures/fig03_junction_continuity/compose/compose_main.py
```

Caches already present: skip step 1 and run panels / compose only.

## Parameters

| File | Controls |
|------|----------|
| `configs/shared.json` | output dir, seed |
| `configs/pipeline.json` | n_cells, diffusion, junction bounds/steps, grids |
| `configs/panel_*.json` | title, figsize, topology, zoom |
| `configs/compose_main.json` | grid layout, figsize |

## Outputs

Under `results/junction_continuity/`:

- `data/trajectories.pt`, `junction_metrics.csv`
- `figures/panel_*.png|pdf`
- `figures/figure3_junction_continuity.png|pdf`

## Legacy

Previous entrypoint: `experiments/results_junction_continuity.py`.
