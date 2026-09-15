# Coordinate consistency of ground-truth RNA velocity

Supplementary diagnostic: does directional cosine between scVelo and
continuous absolute RNA velocity `v_abs = ds/dt` depend on the coordinate
system used for the ground truth?

Primary GT is continuous spliced-abundance velocity. Because RNA velocity
methods internally normalize counts, we compare inferred velocities against
absolute, library-scaled, proportion-transformed, and approximate
log-transformed ground-truth velocities. Cell-wise library scaling preserves
velocity directions, whereas proportion normalization induces a nontrivial
velocity transformation. Qualitative benchmark conclusions are stable under
absolute and library-scaled coordinates; proportion/log results are reported
as preprocessing-coordinate diagnostics.

UniTVelo / DeepVelo are listed in a coordinate-explanation table only
(no full re-run).

## Layout

```
configs/     shared, pipeline, panel_*, compose
pipeline/    simulate → scVelo → metrics cache
panels/      A diagram · B cosine dist · C scatter · D summary
compose/     2×2 supplementary figure
lib/         GT transforms + scVelo wrap
```

## Run (from repo root)

```bash
# Needs scanpy + scvelo; set NUMBA_CACHE_DIR if needed
export NUMBA_CACHE_DIR=/tmp/numba_cache

python experiments/figures/fig_coord_velocity_consistency/pipeline/run_pipeline.py

python experiments/figures/fig_coord_velocity_consistency/panels/panel_a_diagram.py
python experiments/figures/fig_coord_velocity_consistency/panels/panel_b_cosine_dist.py
python experiments/figures/fig_coord_velocity_consistency/panels/panel_c_scatter.py
python experiments/figures/fig_coord_velocity_consistency/panels/panel_d_summary.py

python experiments/figures/fig_coord_velocity_consistency/compose/compose_main.py

# Replot from cache only
python experiments/figures/fig_coord_velocity_consistency/pipeline/run_pipeline.py --plot-only
python experiments/figures/fig_coord_velocity_consistency/compose/compose_main.py
```

## Expected outcomes

| Comparison | Expectation |
|------------|-------------|
| `cos_abs` vs `cos_lib` | Nearly identical (cell-wise positive scale) |
| `cos_prop` | Can differ substantially — do not mix `ds/dt` with proportion coords |
| `cos_log` | Preprocessing diagnostic; rank/trend may stay close to abs |

## Outputs

Under `results/coord_velocity_consistency/`:

- `data/coord_cosine_per_cell.csv`
- `data/coord_summary.csv`
- `data/method_coordinate_table.csv`
- `data/bundle.pkl`
- `figures/panel_*.{png,pdf}`
- `figures/figure_coord_velocity_consistency.{png,pdf}`

## Parameters

Edit `configs/pipeline.json` (`library_target`, `tau_high`, scVelo knobs, simulation size)
and `configs/panel_*.json` for plotting.
