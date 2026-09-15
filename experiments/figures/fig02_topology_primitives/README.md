# Figure 2 — Topology primitives

Graph-based energy-flow primitives generate controllable latent manifolds
(linear, bifurcation, cycle, hybrids, skeleton).

## Layout

```
configs/     JSON parameters (shared, pipeline, each panel, compose)
pipeline/    run simulations → results/topology_primitives/data/
panels/      one executable script per small panel (A–F, S1)
compose/     editable large figures importing panel draw()
lib/         simulation + plotting core
```

## Run (from repo root)

```bash
# 1) Simulate / refresh caches (~minutes)
python experiments/figures/fig02_topology_primitives/pipeline/run_pipeline.py

# 2) Single panels
python experiments/figures/fig02_topology_primitives/panels/panel_a_linear.py
python experiments/figures/fig02_topology_primitives/panels/panel_b_bifurcation.py
# ... panel_c … panel_f, panel_s1_distance.py

# 3) Large figures
python experiments/figures/fig02_topology_primitives/compose/compose_main.py
python experiments/figures/fig02_topology_primitives/compose/compose_supp.py
```

Caches already present: skip step 1 and run panels / compose only.

## Parameters

| File | Controls |
|------|----------|
| `configs/shared.json` | output dir, seed, colors |
| `configs/pipeline.json` | n_cells, n_steps, diffusion, sim_overrides, bounds |
| `configs/panel_*.json` | title, figsize, out_name |
| `configs/compose_*.json` | grid layout, figsize |

Edit JSON then re-run the corresponding script. Do not hard-code magic numbers in panels.

## Outputs

Under `results/topology_primitives/`:

- `data/*_traj.npz`, `topology_metrics.csv`
- `figures/panel_*.png|pdf`
- `figures/figure2_topology_primitives.png|pdf`
- `figures/supplementary_s1_topology_quantification.png|pdf`

## Legacy

Previous entrypoint: `experiments/results_topology_primitives.py` (deprecated wrapper).
