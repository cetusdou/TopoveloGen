# Figure 5 — Kinetic validation

Kinetic integration produces splicing delays and ground-truth RNA velocity
on a linear latent trajectory (`GeneSynthesizer.generate_kinetics`).

## Layout

```
configs/     JSON parameters (shared, pipeline, each panel, compose)
pipeline/    ODE kinetics → results/kinetic_validation/data/
panels/      one executable script per small panel (A–F)
compose/     editable large figure importing panel draw()
lib/         synthesizer + kinetics + panel plots
```

## Run (from repo root)

```bash
# 1) Simulate / refresh caches
python experiments/figures/fig05_kinetic_validation/pipeline/run_pipeline.py

# 2) Single panels
python experiments/figures/fig05_kinetic_validation/panels/panel_a_trajectory_alpha.py
python experiments/figures/fig05_kinetic_validation/panels/panel_b_sigmoid_traces.py
python experiments/figures/fig05_kinetic_validation/panels/panel_c_impulse_phase.py
python experiments/figures/fig05_kinetic_validation/panels/panel_d_impulse_peaks.py
python experiments/figures/fig05_kinetic_validation/panels/panel_e_gamma_hysteresis.py
python experiments/figures/fig05_kinetic_validation/panels/panel_f_velocity_consistency.py

# 3) Large figure
python experiments/figures/fig05_kinetic_validation/compose/compose_main.py
```

Caches already present: skip step 1 and run panels / compose only.

**Note:** Panel E re-runs γ sweeps at plot time (legacy `kinetic_tensors.pt` does not store them).

## Parameters

| File | Controls |
|------|----------|
| `configs/shared.json` | output dir, seed |
| `configs/pipeline.json` | dt, n_steps, β/γ, gene pattern params |
| `configs/panel_*.json` | title, figsize |
| `configs/compose_main.json` | grid layout, figsize |

## Outputs

Under `results/kinetic_validation/`:

- `data/kinetic_tensors.pt`, `kinetic_*.csv`, `parameter_sensitivity.csv`, `velocity_consistency.csv`
- `figures/panel_*.png|pdf`
- `figures/figure5_kinetic_validation.png|pdf`

## Legacy

Previous entrypoint: `experiments/results_kinetic_validation.py`.
