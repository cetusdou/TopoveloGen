# Experiment figures (panelized)

These package names retain their historical numbering. The current paper's
figure-to-source map is in [`../../REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).

Each figure lives under `experiments/figures/fig*/` with the same contract:

1. **One executable script per small panel** (`panels/panel_*.py`)
2. **Compose scripts** import each panel’s `draw(ax, …)` to build editable large figures
3. **All parameters in JSON** under `configs/`
4. **Package README** with the exact run order

Figure packages import `data_generation_module` (numpy API; no torch).

## Packages

| Package | Content | Legacy script |
|---------|---------|---------------|
| [`fig02_topology_primitives`](fig02_topology_primitives/) | Fig 2 A–F + S1 | `results_topology_primitives.py` |
| [`fig03_junction_continuity`](fig03_junction_continuity/) | Fig 3 A–F | `results_junction_continuity.py` |
| [`fig04_alpha_fields`](fig04_alpha_fields/) | Fig 4 A–F | `results_alpha_fields.py` |
| [`fig05_kinetic_validation`](fig05_kinetic_validation/) | Fig 5 A–F | `results_kinetic_validation.py` |
| [`fig06_statistical_realism`](fig06_statistical_realism/) | Fig 6 A–L | `results_statistical_realism.py` |
| [`fig07_embedding_masked_errors`](fig07_embedding_masked_errors/) | Fig 7 A–F | `results_embedding_masked_errors.py` |
| [`fig4series_pancreas_stats`](fig4series_pancreas_stats/) | Pancreas count stats A–I + S1–S4 | `figure_4_fast.py` |
| [`fig_diag_projection_masking`](fig_diag_projection_masking/) | Projection masking diagnosis | `results_projection_masking_diagnosis.py` |
| [`fig_bench_velo_test`](fig_bench_velo_test/) | Velocity method benchmark | `results_benchmark_velo_test.py` |
| [`fig_coord_velocity_consistency`](fig_coord_velocity_consistency/) | GT velocity coordinate consistency (supp.) | — |

Shared helpers: [`_common/`](_common/) (`config.py`, `style.py`).

## Typical workflow

```bash
python experiments/figures/<pkg>/pipeline/run_pipeline.py   # or --plot-only
python experiments/figures/<pkg>/panels/panel_a_….py
python experiments/figures/<pkg>/compose/compose_main.py
```

Tune noise / simulation in `configs/pipeline.json`; tune drawing in `configs/panel_*.json`.
